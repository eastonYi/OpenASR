"""
Copyright 2020 Ye Bai by1993@qq.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import sys
import argparse
import logging
import torch
import yaml

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

import utils
import data
import sp_layers
import encoder_layers
import decoder_layers
from models import Conv_Transformer as Model


def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: feedforward.py <model_pkg> <wav_scp> <output_path>""")
    parser.add_argument("model_pkg", help="path to model package.")
    parser.add_argument("vocab_file", help="path to vocabulary file.")
    parser.add_argument("json_file", help="data directory")
    parser.add_argument("output", help="output")
    parser.add_argument("--feed-batchsize", type=int, default=20, help="batch_size")
    parser.add_argument("--nbest", type=int, default=13, help="nbest")
    parser.add_argument("--maxlen", type=int, default=80, help="max_length")
    parser.add_argument("--use_gpu", type=utils.str2bool, default=False, help="whether to use gpu.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    timer = utils.Timer()
    timer.tic()
    args = get_args()

    if args.output.strip() == "-":
        fd = sys.stdout.buffer
    else:
        fd = open(args.output, 'w', encoding="utf8")

    logging.info("Load package from {}.".format(args.model_pkg))
    pkg = torch.load(args.model_pkg, map_location=lambda storage, loc: storage)
    splayer = sp_layers.SPLayer(pkg["model"]["splayer_config"])
    encoder = encoder_layers.Transformer(pkg["model"]["encoder_config"])
    decoder = decoder_layers.TransformerDecoder(pkg["model"]["decoder_config"])

    model = Model(splayer, encoder, decoder)
    logging.info("\nModel info:\n{}".format(model))
    model.restore(pkg["model"])
    if args.use_gpu:
        model = model.cuda()
    model.eval()
    tokenizer = data.CharTokenizer(args.vocab_file)
    test_set = data.SpeechDataset(args.json_file)

    collate = data.WaveCollate(tokenizer, 60)
    validsampler = data.TimeBasedSampler(test_set, args.feed_batchsize, 1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set,
        collate_fn=collate, batch_sampler=validsampler, shuffle=False, num_workers=1)
    logging.info("Start feedforward...")

    tot_timer = utils.Timer()
    tot_utt = 0
    tot_timer.tic()
    for utts, padded_waveforms, wave_lengths, ids, labels, paddings in test_loader:
        if next(model.parameters()).is_cuda:
            padded_waveforms = padded_waveforms.cuda()
            wave_lengths = wave_lengths.cuda()

        with torch.no_grad():
            target_ids, scores = model.decode(padded_waveforms, wave_lengths, nbest_keep=args.nbest, maxlen=args.maxlen)
        all_ids_batch = target_ids.cpu().numpy()
        all_score_batch = scores.cpu().numpy()
        for i in range(all_ids_batch.shape[0]):
            utt = utts[i]
            msg = "Results for {}:\n".format(utt)
            for h in range(all_ids_batch.shape[1]):
                hyp = tokenizer.decode(all_ids_batch[i, h])
                score = all_score_batch[i, h]
                msg += "top{}: {} score: {:.10f}\n".format(h+1, hyp, score)
                if h == 0:
                    fd.write("{} {}\n".format(utt, hyp))
            logging.info("\n"+msg)
        tot_utt += len(utts)
        logging.info("Prossesed {} utterances in {:.3f} s".format(tot_utt, tot_timer.toc()))
    tot_time = tot_timer.toc()
    logging.info("Decoded {} utterances. The time cost is {:.2f} min."
        " Avg time cost is {:.2f} per utt.".format(tot_utt, tot_time/60., tot_time/tot_utt))

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

import utils
import data

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="""
     Usage: feedforward.py <model_pkg> <wav_scp> <output_path>""")
    parser.add_argument("model_type", help="path to model package.")
    parser.add_argument("model_pkg", help="path to model package.")
    parser.add_argument("vocab_file", help="path to vocabulary file.")
    parser.add_argument("json_file", help="data directory")
    parser.add_argument("output", help="output")
    parser.add_argument("--batch_frames", type=int, default=20, help="batch_size")
    parser.add_argument("--nbest", type=int, default=13, help="nbest")
    parser.add_argument("--maxlen", type=int, default=80, help="max_length")
    parser.add_argument("--config", type=str, default=None, help="config")
    parser.add_argument("--label_type", type=str, default='tokens', help="config")
    parser.add_argument("--offline", type=utils.str2bool, default=False, help=".")
    parser.add_argument("--add_blk", type=utils.str2bool, default=False, help=".")
    parser.add_argument("--use_gpu", type=utils.str2bool, default=False, help="whether to use gpu.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    timer = utils.Timer()
    timer.tic()
    args = get_args()

    tokenizer = data.CharTokenizer(args.vocab_file, add_blk=args.add_blk)

    if args.config:
        import yaml
        args.config = utils.AttrDict(yaml.load(open(args.config)))
        args.config.model.decoder['vocab_size'] = tokenizer.unit_num()

    if args.output.strip() == "-":
        fd = sys.stdout.buffer
    else:
        fd = open(args.output, 'w', encoding="utf8")

    logging.info("Load package from {}.".format(args.model_pkg))
    pkg = torch.load(args.model_pkg, map_location=lambda storage, loc: storage)

    if args.model_type.lower() == 'conv-transformer':
        from frameworks.Speech_Models import Conv_Transformer as Model

        if args.config:
            model = Model.create_model(args.config.model.signal,
                                       args.config.model.encoder,
                                       args.config.model.decoder)
        else:
            model = Model.create_model(pkg["model"]["splayer_config"],
                                       pkg["model"]["encoder_config"],
                                       pkg["model"]["decoder_config"])

    elif args.model_type.lower() == 'conv-ctc-transformer':
        from models import Conv_CTC_Transformer as Model

        model = Model.create_model(pkg["model"]["splayer_config"],
                                   pkg["model"]["encoder_config"],
                                   pkg["model"]["decoder_config"])

    elif args.model_type.lower() == 'cif':
        from models import CIF as Model

        model = Model.create_model(pkg["model"]["splayer_config"],
                                   pkg["model"]["encoder_config"],
                                   pkg["model"]["assigner_config"],
                                   pkg["model"]["decoder_config"])
    else:
        raise NotImplementedError('not found model_type!')

    # logging.info("\nModel info:\n{}".format(model))
    model.restore(pkg["model"])
    if args.use_gpu:
        model = model.cuda()
    model.eval()

    if args.offline:
        test_set = data.SpeechDataset(args.json_file, rate_in_out=None)
        collate = data.WaveCollate(tokenizer, 60)
        validsampler = data.TimeBasedSampler(test_set, args.batch_frames, 1, shuffle=False)
    else:
        test_set = data.ArkDataset(args.json_file, rate_in_out=None)
        collate = data.FeatureCollate(tokenizer, 60, args.add_blk, label_type=args.label_type)
        validsampler = data.FrameBasedSampler(test_set, args.batch_frames, 1, shuffle=False)

    test_loader = torch.utils.data.DataLoader(test_set,
        collate_fn=collate, batch_sampler=validsampler, shuffle=False, num_workers=1)
    logging.info("Start feedforward...")

    tot_timer = utils.Timer()
    tot_utt = 0
    tot_timer.tic()
    for utts, (padded_waveforms, wave_lengths, ids, labels, paddings) in test_loader:
        if next(model.parameters()).is_cuda:
            padded_waveforms = padded_waveforms.cuda()
            wave_lengths = wave_lengths.cuda()

        with torch.no_grad():
            encoded, len_encoded = model.get_encoded(padded_waveforms, wave_lengths)
            pred_ids, len_decodeds, scores = model.batch_beam_decode(
                encoded, len_encoded, model.decoder.step_forward,
                vocab_size=tokenizer.unit_num(),
                beam_size=args.nbest, max_decode_len=args.maxlen)
        pred_ids = pred_ids.cpu().numpy()
        len_decodeds = len_decodeds.cpu().tolist()
        scores = scores.cpu().numpy()
        for i, (n_pred_ids, n_len_decodeds, n_scores) in enumerate(zip(pred_ids, len_decodeds, scores)):
            utt = utts[i]
            msg = "Results for {}:\n".format(utt)
            for j, (pred_id, len_decoded, score) in enumerate(zip(n_pred_ids, n_len_decodeds, n_scores)):
                hyp = tokenizer.decode(pred_id[:len_decoded], split_token=False)
                msg += "top{}: {} score: {:.10f}\n".format(j+1, hyp, score)
                if j == 0:
                    fd.write("{} {}\n".format(utt, hyp))
            logging.info("\n"+msg)
        tot_utt += len(utts)
        logging.info("Prossesed {} utterances in {:.3f} s".format(tot_utt, tot_timer.toc()))
    tot_time = tot_timer.toc()
    logging.info("Decoded {} utterances. The time cost is {:.2f} min."
        " Avg time cost is {:.2f} per utt.".format(tot_utt, tot_time/60., tot_time/tot_utt))

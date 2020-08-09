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
    parser.add_argument("--model_type", default='Embed_Decoder', help="path to model package.")
    parser.add_argument("--model_pkg", help="path to model package.")
    parser.add_argument("--vocab_phone", default='data/callhome.IPA', help="path to vocabulary file.")
    parser.add_argument("--vocab_char", help="path to vocabulary file.")
    parser.add_argument("--json_file", help="data directory")
    parser.add_argument("--output", help="output")
    parser.add_argument("--batch_frames", type=int, default=20, help="batch_frames")
    parser.add_argument("--nbest", type=int, default=5, help="nbest")
    parser.add_argument("--maxlen", type=int, default=80, help="max_length")
    parser.add_argument("--config", type=str, default=None, help="config")
    parser.add_argument("--label_type", type=str, default='tokens', help="config")
    parser.add_argument("--add_blk", type=utils.str2bool, default=False, help=".")
    parser.add_argument("--add_eos", type=utils.str2bool, default=True, help=".")
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

    # logging.info("\nModel info:\n{}".format(model))

    tokenizer_phone = data.CharTokenizer(args.vocab_phone, add_blk=True)
    tokenizer_char = data.CharTokenizer(args.vocab_char, add_blk=args.add_blk)
    test_set = data.ArkDataset(args.json_file, rate_in_out=None)
    logging.info("Start feedforward...")

    if args.model_type.lower() == 'embed_decoder':
        from frameworks.Text_Models import Embed_Decoder as Model

        collate = data.Phone_Char_Collate(tokenizer_phone, tokenizer_char, args.add_eos)
        test_loader = torch.utils.data.DataLoader(test_set,
            collate_fn=collate, batch_size=args.batch_size, num_workers=1)

        model = Model.create_model(pkg["model"]["decoder"])

    elif args.model_type.lower() == 'cif_mix':
        from frameworks.Speech_Models import CIF_MIX as Model

        collate = data.Feat_Phone_Char_Collate(tokenizer_phone, tokenizer_char, args.add_eos)
        sampler = data.FrameBasedSampler(test_set, args.batch_frames, 1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set,
            collate_fn=collate, batch_sampler=sampler, shuffle=False, num_workers=2)

        model = Model.create_model(pkg["model"]["splayer_config"],
                                   pkg["model"]["encoder_config"],
                                   pkg["model"]["assigner_config"],
                                   pkg["model"]["decoder_config"])
    else:
        raise NotImplementedError('not found model_type!')

    model.restore(pkg["model"])
    if args.use_gpu:
        model = model.cuda()
    model.eval()
    device = torch.cuda.current_device() if args.use_gpu else 'cpu'

    tot_timer = utils.Timer()
    tot_utt = 0
    tot_timer.tic()
    for utts, batch in test_loader:
        feats, len_feat, phones, len_phone, target_in, target_out, paddings = (i.to(device) for i in batch)

        with torch.no_grad():
            encoded, len_encoded = model.get_encoded(feats, len_feat)
            pred_ids, len_decodeds, scores = model.batch_beam_decode(
                encoded, len_encoded,
                step_forward_fn=model.decoder.step_forward,
                vocab_size=tokenizer_char.unit_num(),
                beam_size=args.nbest,
                max_decode_len=args.maxlen)
        pred_ids = pred_ids.cpu().numpy()
        len_decodeds = len_decodeds.cpu().tolist()
        scores = scores.cpu().numpy()
        for i, (n_pred_ids, n_len_decodeds, n_scores) in enumerate(zip(pred_ids, len_decodeds, scores)):
            utt = utts[i]
            msg = "Results for {}:\n".format(utt)
            for j, (pred_id, len_decoded, score) in enumerate(zip(n_pred_ids, n_len_decodeds, n_scores)):
                hyp = tokenizer_char.decode(pred_id[:len_decoded], split_token=False)
                msg += "top{}: {} score: {:.10f}\n".format(j+1, hyp, score)
                if j == 0:
                    fd.write("{} {}\n".format(utt, hyp))
            logging.info("\n"+msg)
        tot_utt += len(utts)
        logging.info("Prossesed {} utterances in {:.3f} s".format(tot_utt, tot_timer.toc()))
    tot_time = tot_timer.toc()
    logging.info("Decoded {} utterances. The time cost is {:.2f} min."
        " Avg time cost is {:.2f} per utt.".format(tot_utt, tot_time/60., tot_time/tot_utt))

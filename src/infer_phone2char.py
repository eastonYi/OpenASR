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
import os
import sys
import argparse
import logging
import torch
from torch.utils.data import DataLoader
import editdistance as ed

import utils
from dataload import datasets, collates, samplers, data_utils

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
    parser.add_argument("--name", default='', help="name")
    parser.add_argument("--batch_frames", type=int, default=20, help="batch_frames")
    parser.add_argument("--batch_size", type=int, default=20, help="batch_size")
    parser.add_argument("--nbest", type=int, default=5, help="nbest")
    parser.add_argument("--maxlen", type=int, default=80, help="max_length")
    parser.add_argument("--config", type=str, default=None, help="config")
    parser.add_argument("--label_type", type=str, default='tokens', help="config")
    parser.add_argument("--add_blk", type=utils.str2bool, default=False, help=".")
    parser.add_argument("--add_eos", type=utils.str2bool, default=False, help=".")
    parser.add_argument("--use_gpu", type=utils.str2bool, default=False, help="whether to use gpu.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    timer = utils.Timer()
    timer.tic()
    args = get_args()

    f_hyp = open(os.path.join(os.path.dirname(args.model_pkg), 'hyp{}.txt'.format(args.name)), 'w', encoding="utf8")
    f_ref = open(os.path.join(os.path.dirname(args.model_pkg), 'ref{}.txt'.format(args.name)), 'w', encoding="utf8")

    logging.info("Load package from {}.".format(args.model_pkg))
    pkg = torch.load(args.model_pkg, map_location=lambda storage, loc: storage)

    # logging.info("\nModel info:\n{}".format(model))

    tokenizer_phone = data_utils.CharTokenizer(args.vocab_phone, add_blk=True)
    tokenizer_char = data_utils.CharTokenizer(args.vocab_char, add_blk=args.add_blk)
    test_set = datasets.PhoneCharDataset(args.json_file)

    # test_set = datasets.PhoneCharDataset(args.json_file, feat_range=(1,1000), label_range=(1,50))

    if args.model_type.lower() == 'embed_decoder':
        from frameworks.Text_Models import Embed_Decoder as Model

        collate = collates.Phone_Char_Collate(tokenizer_phone, tokenizer_char, args.add_eos)
        test_loader = DataLoader(test_set,
            collate_fn=collate, batch_size=args.batch_size, num_workers=1)

        model = Model.create_model(pkg["model"]["decoder"])

    elif args.model_type.lower() == 'embed_decoder_ctc':
        from frameworks.Text_Models import Embed_Decoder_CTC as Model

        collate = collates.Phone_Char_Collate(tokenizer_phone, tokenizer_char, args.add_eos)
        test_loader = DataLoader(test_set,
            collate_fn=collate, batch_size=args.batch_size, num_workers=1)

        model = Model.create_model(pkg["model"]["encoder_config"], pkg["model"]["decoder_config"])

        decode_fn = utils.ctc_decode_fn(list(tokenizer_char.unit2id.keys()),
                                        # beam_width=args.nbest,
                                        beam_width=1,
                                        blank_id=tokenizer_char.unit_num()-1)
    else:
        raise NotImplementedError('not found model_type: {}'.format(args.model_type))

    model.restore(pkg["model"])
    if args.use_gpu:
        model = model.cuda()
    model.eval()
    device = torch.cuda.current_device() if args.use_gpu else 'cpu'

    tot_timer = utils.Timer()
    tot_utt = 0
    num_tokens = 0
    num_dist = 0
    tot_timer.tic()
    for utts, batch in test_loader:
        feats, len_feat, _, targets, paddings = (i.to(device) for i in batch)

        with torch.no_grad():
            if 'ctc' in args.model_type.lower():
                logits, len_logits = model.get_logits(feats, len_feat)
                pred_ids, len_decodeds, scores = model.batch_beam_decode(
                    logits, len_logits, decode_fn)
            else:
                encoded, len_encoded = model.get_encoded(feats, len_feat)
                pred_ids, len_decodeds, scores = model.batch_beam_decode(
                    encoded, len_encoded,
                    step_forward_fn=model.decoder.step_forward,
                    vocab_size=tokenizer_char.unit_num(),
                    beam_size=args.nbest,
                    max_decode_len=args.maxlen)
        targets = targets.cpu().numpy()
        len_targets = (1-paddings).sum(-1).int().cpu().tolist()
        pred_ids = pred_ids.cpu().numpy()
        len_decodeds = len_decodeds.cpu().tolist()
        scores = scores.cpu().numpy()
        for i, (target, len_target, n_pred_ids, n_len_decodeds, n_scores) in enumerate(
            zip(targets, len_targets, pred_ids, len_decodeds, scores)):
            utt = utts[i]

            num_tokens += len_target
            ref = tokenizer_char.decode(target[:len_target], split_token=True)
            f_ref.write("{} {}\n".format(utt, ref))

            msg = "Results for {}:\n".format(utt)
            for j, (pred_id, len_decoded, score) in enumerate(zip(n_pred_ids, n_len_decodeds, n_scores)):
                hyp = tokenizer_char.decode(pred_id[:len_decoded], split_token=True)
                msg += "top{}: {} score: {:.10f}\n".format(j+1, hyp, score)
                if j == 0:
                    dist = ed.eval(pred_id[:len_decoded], target[:len_target])
                    num_dist += dist
                    f_hyp.write("{} {}\n".format(utt, hyp))

            logging.info("\n"+msg)
        tot_utt += len(utts)
        logging.info("Prossesed {} utterances in {:.3f} s".format(tot_utt, tot_timer.toc()))
    tot_time = tot_timer.toc()
    logging.info("Decoded {} utterances. The time cost is {:.2f} min. ref len: {} WER {:.2f}%"
        " Avg time cost is {:.2f} per utt.".format(tot_utt, tot_time/60., num_tokens, num_dist*100/num_tokens, tot_time/tot_utt))

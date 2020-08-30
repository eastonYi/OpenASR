#!/usr/bin/env python3
# encoding: utf-8
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat', type=str)
    parser.add_argument('--num_frames', type=str)
    parser.add_argument('--trans', type=str)
    parser.add_argument('--tokens', type=str)
    parser.add_argument('--phones', type=str, default=None)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    uttid2feat = {}
    with open(args.feat) as f:
        for line in f:
            uttid, feat = line.strip().split()
            uttid2feat[uttid] = feat

    uttid2num_frames = {}
    with open(args.num_frames) as f:
        for line in f:
            uttid, num_frames = line.strip().split()
            uttid2num_frames[uttid] = int(num_frames)

    uttid2trans = {}
    with open(args.trans) as f:
        for line in f:
            try:
                uttid, trans = line.strip().split(maxsplit=1)
                uttid2trans[uttid] = trans
            except:
                pass

    if args.tokens:
        uttid2tokens = {}
        with open(args.tokens) as f:
            for line in f:
                try:
                    uttid, tokens = line.strip().split(maxsplit=1)
                    uttid2tokens[uttid] = tokens
                except:
                    pass

    if args.phones:
        uttid2phones = {}
        with open(args.phones) as f:
            for line in f:
                try:
                    uttid, phones = line.strip().split(maxsplit=1)
                    uttid2phones[uttid] = phones
                except:
                    pass

    samples = []
    for i, uttid in enumerate(uttid2feat.keys()):
        try:
            sample = {}
            sample['uttid'] = uttid
            sample['feat'] = uttid2feat[uttid]
            sample['feat_length'] = uttid2num_frames[uttid]
            if args.tokens:
                sample['tokens'] = uttid2tokens[uttid]
                sample['token_length'] = len(sample['tokens'].split())
            else:
                sample['tokens'] = ' '.join(uttid2trans[uttid])
                sample['token_length'] = len(sample['tokens'])
            if args.phones:
                sample['phones'] = uttid2phones[uttid]
                sample['phone_length'] = len(sample['phones'].split())
            sample['trans'] = uttid2trans[uttid]

        except:
            print('skip', uttid)
            continue

        samples.append(sample)
    print('saved {}/{} samples'.format(len(samples), i+1))
    jsonstring = json.dumps(samples, indent=2, ensure_ascii=False)

    with open(args.output, 'w') as fw:
        fw.write(jsonstring)

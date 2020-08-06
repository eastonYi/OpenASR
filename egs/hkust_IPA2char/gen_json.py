#!/usr/bin/env python3
# encoding: utf-8
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trans', type=str)
    parser.add_argument('--phones', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    uttid2trans = {}
    with open(args.trans) as f:
        for line in f:
            try:
                uttid, trans = line.strip().split(maxsplit=1)
                uttid2trans[uttid] = trans
            except:
                pass

    uttid2phones = {}
    with open(args.phones) as f:
        for line in f:
            try:
                uttid, phones = line.strip().split(maxsplit=1)
                uttid2phones[uttid] = phones
            except:
                pass

    samples = []
    for i, uttid in enumerate(uttid2trans.keys()):
        try:
            sample = {}
            sample['uttid'] = uttid
            sample['tokens'] = uttid2trans[uttid]
            sample['phones'] = uttid2phones[uttid]
            sample['input_length'] = len(sample['phones'].split())
            sample['output_length'] = len(sample['tokens'].split())
        except:
            print('skip', uttid)
            continue

        samples.append(sample)
    print('saved {}/{} samples'.format(len(samples), i))
    jsonstring = json.dumps(samples, indent=2, ensure_ascii=False)

    with open(args.output, 'w') as fw:
        fw.write(jsonstring)

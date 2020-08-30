#!/usr/bin/env python3
# encoding: utf-8
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flist', type=str)
    parser.add_argument('--trans', type=str)
    parser.add_argument('--tokens', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()


    def fname2uttid(fname):
        list_ = []
        for i in fname.split('-'):
            list_.append('0' * (6-len(i)) +  i)
        uttid = '-'.join(list_)

        return uttid

    uttid2path = {}
    uttid2length = {}
    with open(args.flist) as f:
        for line in f:
            path, length = line.strip().split()
            fname = path.split('/')[-1].split('.')[0]
            uttid = fname2uttid(fname)
            uttid2path[uttid] = path
            uttid2length[uttid] = int(length)

    uttid2trans = {}
    with open(args.trans) as f:
        for line in f:
            try:
                uttid, trans = line.strip().split(maxsplit=1)
                uttid2trans[uttid] = trans
            except:
                pass

    uttid2tokens = {}
    with open(args.tokens) as f:
        for line in f:
            try:
                uttid, tokens = line.strip().split(maxsplit=1)
                uttid2tokens[uttid] = tokens
            except:
                pass

    samples = []
    for i, uttid in enumerate(uttid2path.keys()):
        try:
            sample = {}
            sample['uttid'] = uttid
            sample['feat'] = uttid2path[uttid]
            sample['feat_length'] = uttid2length[uttid]

            sample['tokens'] = uttid2tokens[uttid]
            sample['token_length'] = len(sample['tokens'].split())

            sample['phones'] = ' | '.join(' '.join(word) for word in uttid2trans[uttid].split())
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

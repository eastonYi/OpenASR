#!/usr/bin/env python3
# encoding: utf-8
import argparse
import re


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    with open(args.text) as f, open(args.output, 'w') as fw:
        for line in f:
            uttid, trans = line.strip().split(maxsplit=1)
            tokens = []
            for token in trans:
                if re.findall('[a-zA-Z]', token):
                    tokens.append(token)
                else:
                    tokens.extend(list(token))
            line = uttid + ' ' + ' '.join(tokens)
            fw.write(line + '\n')

#!/usr/bin/env python3
# encoding: utf-8
import argparse
import os
import soundfile as sf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-dir', type=str, dest='wav_dir')
    parser.add_argument('--ext', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    num = 0
    pwd = os.getcwd()
    with open(args.output, 'w') as fw:
        for root, dirs, fs in os.walk(args.wav_dir):
            for f in fs:
                if f.endswith(args.ext):
                    file_path = os.path.join(pwd, root, f)
                    sig, _ = sf.read(file_path)
                    fw.write('{}\t{}\n'.format(file_path, len(sig)))
                    num += 1

    print('saved {} samples'.format(num))

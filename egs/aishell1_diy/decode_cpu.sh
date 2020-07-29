#!/bin/bash
source path.sh
expdir=exp/base
ep=avg-last10
decode_dir=$expdir/decode_test_${ep}
mkdir -p $decode_dir

python -W ignore::UserWarning $SRC_ROOT/decode.py \
    --feed-batchsize 2 \
    --nbest 5 \
    $expdir/${ep}.pt \
    data/aishell1_train_chars.txt \
    data/test_small.json \
    $decode_dir/hyp.txt

#!/bin/bash
source path.sh

expdir=$1

# ep=avg-last1
ep=${2:?"avg-last10"}
decode_dir=$expdir/decode_test_${ep}
mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/decode.py \
    --feed-batchsize 2 \
    --nbest 5 \
    $expdir/${ep}.pt \
    ../aishell1/data/aishell1_train_chars.txt \
    data/test_ark_small.json \
    $decode_dir/hyp.txt

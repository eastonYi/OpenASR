#!/bin/bash
source path.sh

expdir=$1
model_type=$2
# ep=${2:?"last-ckpt"}

decode_dir=$expdir/decode_test_${last-ckpt}
mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/decode.py \
    --feed-batchsize 2 \
    --nbest 5 \
    $model_type \
    $expdir/last-ckpt.pt \
    ../aishell1/data/aishell1_train_chars.txt \
    data/test_ark_small.json \
    $decode_dir/hyp.txt

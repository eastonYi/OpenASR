#!/bin/bash
source path.sh

expdir=$1
model_type=$2

mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/decode.py \
    --batch_frames 1000 \
    --nbest 5 \
    $model_type \
    $expdir/last.pt \
    ../aishell1/data/aishell1_train_chars.txt \
    data/test_ark_small.json \
    $expdir/hyp_small.txt

#!/bin/bash
source path.sh
expdir=exp/base
ep=avg-last10
decode_dir=$expdir/decode_test_${ep}
mkdir -p $decode_dir

CUDA_VISIBLE_DEVICES="3" \
python -W ignore::UserWarning $MAIN_ROOT/src/decode.py \
    --feed-batchsize 80 \
    --nbest 5 \
    --use_gpu True \
    $expdir/${ep}.pt \
    data/aishell1_train_chars.txt \
    data/test \
    "file" \
    $decode_dir/hyp.txt



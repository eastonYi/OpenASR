#!/bin/bash
source path.sh

expdir=$1
model_type=$2

mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/infer.py \
    --batch_frames 5000 \
    --nbest 1 \
    --label_type tokens \
    --vocab_file data/vocab.char \
    --model_type $model_type \
    --model_pkg $expdir/last.pt \
    --add_blk True \
    --json_file data/hkust_dev-small.json \
    --output $expdir/hyp_small.txt

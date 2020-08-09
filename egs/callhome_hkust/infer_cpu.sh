#!/bin/bash
source path.sh

expdir=$1
model_type=$2

mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/infer_phone2char.py \
    --batch_frames 1000 \
    --nbest 5 \
    --label_type feat_phone_char \
    --vocab_char data/vocab_ja.char \
    --model_type $model_type \
    --model_pkg $expdir/last.pt \
    --add_blk False \
    --json_file data/dev/ja.json \
    --output $expdir/hyp_small.txt

#!/bin/bash
source path.sh

expdir=$1
model_type=$2
gpu=$3
ep=$4
decode_dir=$expdir/decode_test_${ep}
mkdir -p $decode_dir

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore::UserWarning $SRC_ROOT/infer_phone2char.py \
    --batch_frames 20000 \
    --nbest 5 \
    --label_type feat_phone_char \
    --vocab_char data/vocab_ja.char \
    --model_type $model_type \
    --model_pkg $expdir/last.pt \
    --add_blk False \
    --json_file /data3/easton/data/CALLHOME_Multilingual/jsons/test/ja_test.json \
    --output $decode_dir/hyp.txt \
    --use_gpu True

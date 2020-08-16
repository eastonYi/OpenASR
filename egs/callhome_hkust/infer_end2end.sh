#!/bin/bash
source path.sh

expdir=$1
model_type=$2
gpu=$3
decode_dir=$expdir/decode_test_${model_type}
mkdir -p $decode_dir

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore::UserWarning $SRC_ROOT/infer.py \
    --batch_frames 10000 \
    --nbest 5 \
    --label_type tokens \
    --vocab_path data/vocab_ja.char \
    --model_type $model_type \
    --model_pkg $expdir/avg10.pt \
    --add_blk False \
    --json_file /data3/easton/data/CALLHOME_Multilingual/jsons/test/ja_test.json \
    --output $decode_dir/hyp.txt \
    --use_gpu True

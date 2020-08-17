#!/bin/bash
source path.sh

expdir=$1
model_type=$2
gpu=$3
decode_dir=$expdir/decode_test_${model_type}
mkdir -p $decode_dir

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore::UserWarning $SRC_ROOT/infer.py \
    --batch_frames 20000 \
    --nbest 5 \
    --label_type phones \
    --vocab_path data/callhome.IPA \
    --model_type $model_type \
    --model_pkg $expdir/avg10.pt \
    --add_blk True \
    --split_token True \
    --json_file /data3/easton/data/CALLHOME_Multilingual/jsons/test/ja_test.json \
    --output $decode_dir/hyp.IPA \
    --use_gpu True

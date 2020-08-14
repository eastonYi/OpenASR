#!/bin/bash
source path.sh

expdir=$1
model_type=$2
gpu=$3
ep=$4
decode_dir=$expdir/decode_test_${ep}
mkdir -p $decode_dir

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore::UserWarning $SRC_ROOT/infer.py \
    --batch_frames 100000 \
    --nbest 5 \
    --label_type tokens \
    --vocab_file data/vocab.char \
    --model_type $model_type \
    --model_pkg $expdir/avg10.pt \
    --add_blk True \
    --json_file /data3/easton/data/HKUST/HKUST_80_new/dev/hkust_dev.json \
    --output $decode_dir/hyp.txt \
    --use_gpu True

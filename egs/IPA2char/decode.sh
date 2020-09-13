#!/bin/bash
source path.sh

expdir=$1
model_type=$2
gpu=$3
ep=avg5
decode_dir=$expdir/decode_test_${ep}
mkdir -p $decode_dir

CUDA_VISIBLE_DEVICES=$gpu \
python -W ignore::UserWarning $SRC_ROOT/infer_phone2char.py \
    --batch_size 50 \
    --nbest 5 \
    --use_gpu True \
    --model_type $model_type \
    --add_blk True \
    --model_pkg $expdir/${ep}.pt \
    --vocab_char /data3/easton/data/CALLHOME_Multilingual/ma/train/vocab.char \
    --vocab_phone /data3/easton/data/CALLHOME_Multilingual/jsons/callhome.IPA \
    --json_file /data3/easton/data/CALLHOME_Multilingual/ma/test/data.json


    # --json_file /data3/easton/data/CALLHOME_Multilingual/jsons/dev/hkust_dev.json

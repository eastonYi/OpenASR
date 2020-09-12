#!/bin/bash
source path.sh

expdir=$1
model_type=$2

mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/infer_phone2char.py \
    --model_type $model_type \
    --model_pkg $expdir/last.pt \
    --vocab_phone /Users/easton/Projects/OpenASR_BaiYe/egs/hkust_IPA2char/data/callhome.IPA \
    --vocab_char /Users/easton/Projects/OpenASR_BaiYe/egs/hkust_IPA2char/data/vocab.char \
    --json_file /Users/easton/Projects/OpenASR_BaiYe/egs/hkust_IPA2char/data/dev.json \
    --output $expdir/hyp_small.txt \
    --batch_size 100

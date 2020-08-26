#!/bin/bash
source path.sh

expdir=$1
model_type=$2

mkdir -p $decode_dir
python -W ignore::UserWarning $SRC_ROOT/infer.py \
    --batch_frames 1000 \
    --nbest 5 \
    --label_type tokens \
    --model_type $model_type \
    --model_pkg $expdir/wav2vec_small_960h.pt \
    --vocab_path data/aishell1_train_chars.txt \
    --json_file data/test_ark_small.json \
    --output $expdir/hyp_small.txt \

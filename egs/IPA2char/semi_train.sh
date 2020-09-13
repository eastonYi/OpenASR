#!/bin/bash
source path.sh

config=$1
gpu=$2

export CUDA_VISIBLE_DEVICES=$gpu
python $SRC_ROOT/semi_train_phone2char.py $config
# python $SRC_ROOT/train_phone2char.py --continue-training True $config

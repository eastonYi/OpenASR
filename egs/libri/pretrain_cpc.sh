#!/bin/bash
source path.sh

config=$1
gpu=$2

export CUDA_VISIBLE_DEVICES=$gpu
python $SRC_ROOT/train_CPC.py $config --type pretrain
# python $MAIN_ROOT/src/train.py --continue-training True $config

#!/bin/bash
source path.sh

config=$1
gpu=$2

export CUDA_VISIBLE_DEVICES=$gpu
python $MAIN_ROOT/src/train.py $config
# python $MAIN_ROOT/src/train.py --continue-training True $config

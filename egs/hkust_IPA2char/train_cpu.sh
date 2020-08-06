#!/bin/bash

source path.sh

config=$1

python $SRC_ROOT/train_phone2char.py $config
#python $MAIN_ROOT/src/train.py --continue-training True $config

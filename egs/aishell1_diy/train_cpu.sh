#!/bin/bash

source path.sh

config=$1

# python $MAIN_ROOT/src/train.py $config
python $MAIN_ROOT/src/train.py --continue-training True $config

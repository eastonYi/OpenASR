#!/bin/bash
source path.sh

expdir=$1

python $MAIN_ROOT/tools/avg_last_ckpts.py $expdir 10

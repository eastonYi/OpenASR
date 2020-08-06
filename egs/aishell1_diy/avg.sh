#!/bin/bash
source path.sh

expdir=$1

python $SRC_ROOT/tools/avg_last_ckpts.py $expdir 10

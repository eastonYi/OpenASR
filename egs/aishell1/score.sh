#!/bin/bash

#source path.sh

dir=$1


cat $dir/hyp.trn | python3 -c \
"
import sys
for line in sys.stdin:
  txt, utt = line.strip().split(' (')
  txt = txt.replace(' ', '')
  utt = utt[:-1]
  print('{} {}'.format(utt, txt))
" > hyp


exit 1;
$MAIN_ROOT/tools/sclite -r ${dir}/ref.trn trn -h ${dir}/hyp.trn trn -c NOASCII -i wsj -o all stdout > ${dir}/result.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -m 2 ${dir}/result.txt



ref=$1
hyp=$2
out=$3
/home/easton/files/sctk-2.4.10/bin/sclite -r ${ref} trn -h ${hyp} -i rm -c NOASCII -s -o all stdout > ${out}
vi ${out}

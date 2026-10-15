#!/bin/bash
# inside the Docker container, /work = data/external
set -e
cd /work
phonetisaurus-align --input=phonetisaurus_train.dict --ofile=phonetisaurus.corpus --seq1_del=false --seq2_del=false 2>&1 | tail -2
estimate-ngram -o 8 -t phonetisaurus.corpus -wl phonetisaurus.o8.arpa 2>&1 | tail -2
phonetisaurus-arpa2wfst --lm=phonetisaurus.o8.arpa --ofile=phonetisaurus.fst
phonetisaurus-g2pfst --model=phonetisaurus.fst --nbest=1 --beam=10000 --thresh=99.0 --wordlist=phonetisaurus_words.txt > out_phonetisaurus_raw.txt
echo finished

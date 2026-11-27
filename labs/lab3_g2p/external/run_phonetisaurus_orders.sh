#!/bin/bash
# The joint-graphone n-gram of Phonetisaurus at different orders, on the corpus aligned by run_phonetisaurus.sh
set -e
cd /work
for N in 2 3 4 5 6; do
  estimate-ngram -o $N -t phonetisaurus.corpus -wl phonetisaurus.o$N.arpa > /dev/null 2>&1
  phonetisaurus-arpa2wfst --lm=phonetisaurus.o$N.arpa --ofile=phonetisaurus.o$N.fst
  phonetisaurus-g2pfst --model=phonetisaurus.o$N.fst --nbest=1 --beam=10000 --thresh=99.0 --wordlist=phonetisaurus_words.txt > out_phonetisaurus_o$N.txt 2>/dev/null
  echo "order $N done"
done
echo finished

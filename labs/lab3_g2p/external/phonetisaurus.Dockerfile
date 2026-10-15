# Phonetisaurus (OpenFst-based joint n-gram G2P) built from source, lab 3 extra task 3.
#
#   python external/prepare_external_inputs.py            # training lexicon and words to decode -> data/external/
#   cp external/run_phonetisaurus.sh data/external/
#   docker build -t phonetisaurus-lab3 -f external/phonetisaurus.Dockerfile external
#   docker run --rm -v <lab dir>/data/external:/work phonetisaurus-lab3 bash /work/run_phonetisaurus.sh
#
# (Git Bash on Windows rewrites `/work/...`: prefix the command with MSYS_NO_PATHCONV=1.)
#
# Recipe from the Phonetisaurus README (OpenFst 1.7.2, mitlm for the n-gram model).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git g++ autoconf autoconf-archive automake make libtool gfortran wget ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt
RUN wget -q http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.2.tar.gz \
    && tar xzf openfst-1.7.2.tar.gz && cd openfst-1.7.2 \
    && ./configure --enable-static --enable-shared --enable-far --enable-ngram-fsts \
    && make -j"$(nproc)" && make install && ldconfig && cd .. && rm -rf openfst-1.7.2*
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/fst
RUN git clone --depth 1 https://github.com/AdolfVonKleist/Phonetisaurus.git \
    && cd Phonetisaurus && ./configure && make -j"$(nproc)" && make install && ldconfig
RUN git clone --depth 1 https://github.com/mitlm/mitlm.git \
    && cd mitlm && ./autogen.sh && make -j"$(nproc)" && make install && ldconfig
WORKDIR /work

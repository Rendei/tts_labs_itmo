"""Run the external transcribers of lab 3, extra task 3 — **in a separate environment**.

The tools need packages that the lab's own environment does not have (`ruaccent`, `phonemizer`,
`transformers` + `torch`) and one of them (`russian_g2p`) is a repository, not a package. Install them into a
separate virtual environment (never into the project's `.venv`) and run this file with its Python::

    python external_tools.py espeak          data/external/test_words.txt  data/external/out_espeak.csv
    python external_tools.py charsiu         data/external/test_words.txt  data/external/out_charsiu.csv
    python external_tools.py ruaccent_words  data/external/test_words.txt  data/external/out_ruaccent_words.csv
    python external_tools.py ruaccent_sent   data/external/test_sentences.csv data/external/out_ruaccent_sent.csv
    python external_tools.py russian_g2p     data/external/out_ruaccent_words.csv data/external/out_russian_g2p.csv <path to russian_g2p repo>

Every command writes `input|raw output`; mapping the outputs to the target phone inventory and scoring
them is done in the report (`ipa_map.py`), which needs none of these packages.
"""
import csv
import os
import re
import sys
import time


def read_words(path: str) -> list[str]:
    with open(path, encoding='utf-8') as fh:
        return [line.rstrip('\n') for line in fh if line.strip()]


def write_pairs(path: str, pairs) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for key, value in pairs:
            writer.writerow([key, value])


def run_espeak(words: list[str], runtime_dir: str = 'data/external/espeak_runtime') -> list[str]:
    """espeak-ng, Russian voice, one word at a time (context-free), IPA with stress marks.

    The espeak-ng library of `espeakng-loader` cannot read its data from a path with non-ASCII characters
    (a Windows user name in Cyrillic, 8.3 short names), so the library and the data are copied once into
    `runtime_dir`, which must have a plain ASCII path.
    """
    import os
    import shutil
    import espeakng_loader
    from phonemizer import phonemize
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    os.makedirs(runtime_dir, exist_ok=True)
    lib = os.path.join(runtime_dir, 'espeak-ng.dll')
    data = os.path.join(runtime_dir, 'espeak-ng-data')
    if not os.path.isfile(lib):
        shutil.copy(espeakng_loader.get_library_path(), lib)
    if not os.path.isdir(data):
        shutil.copytree(espeakng_loader.get_data_path(), data)
    EspeakWrapper.set_library(os.path.abspath(lib))
    EspeakWrapper.set_data_path(os.path.abspath(data))
    out = phonemize(words, language='ru', backend='espeak', strip=True, preserve_punctuation=False,
                    with_stress=True, njobs=1)
    return [o.strip() for o in out]


def run_charsiu(words: list[str], batch: int = 256) -> list[str]:
    """charsiu-g2p (ByT5-tiny multilingual), language tag `rus-ru`, greedy decoding."""
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    model = T5ForConditionalGeneration.from_pretrained('charsiu/g2p_multilingual_byT5_tiny_16_layers_100')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    model.eval()
    out: list[str] = []
    for start in range(0, len(words), batch):
        chunk = words[start:start + batch]
        inputs = tokenizer(['<rus-ru>: ' + w for w in chunk], padding=True, add_special_tokens=False,
                           return_tensors='pt')
        with torch.no_grad():
            preds = model.generate(**inputs, num_beams=1, max_length=64)
        out += tokenizer.batch_decode(preds.tolist(), skip_special_tokens=True)
        print(f'charsiu {start + len(chunk)}/{len(words)}', flush=True)
    return out


def load_ruaccent():
    from ruaccent import RUAccent
    accentizer = RUAccent()
    accentizer.load(omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False, workdir=os.path.abspath("data/external/ruaccent_models"))
    return accentizer


def run_ruaccent_words(words: list[str]) -> list[str]:
    """RUAccent on isolated words: dictionary stress (and `ё`), no context for homographs."""
    accentizer = load_ruaccent()
    out = []
    for k, word in enumerate(words):
        out.append(accentizer.process_all(word))
        if k % 1000 == 0:
            print(f'ruaccent words {k}/{len(words)}', flush=True)
    return out


def run_ruaccent_sentences(path: str) -> list[tuple[str, str]]:
    """RUAccent on whole sentences (tokens joined by spaces, no punctuation): homographs by context."""
    accentizer = load_ruaccent()
    with open(path, encoding='utf-8') as fh:
        rows = list(csv.reader(fh, delimiter='|', quoting=csv.QUOTE_NONE))[1:]
    out = []
    for k, (uid, text) in enumerate(rows):
        out.append((uid, accentizer.process_all(text)))
        if k % 200 == 0:
            print(f'ruaccent sentences {k}/{len(rows)}', flush=True)
    return out


def run_russian_g2p(pairs: list[tuple[str, str]], repo: str) -> list[tuple[str, str]]:
    """The rule-based transcriber of `russian_g2p` (its own phone set), stress taken from the accented word."""
    sys.path.insert(0, repo)
    from russian_g2p.Grapheme2Phoneme import Grapheme2Phoneme
    g2p = Grapheme2Phoneme(exception_for_nonaccented=False)
    out = []
    for word, accented in pairs:
        # RUAccent puts `+` before the stressed vowel, russian_g2p expects it after the vowel
        accented = re.sub(r'\+(.)', lambda m: m.group(1) + '+', accented)
        try:
            phones = g2p.word_to_phonemes(accented)
        except Exception as exc:                         # words the tool cannot handle are reported empty
            phones = []
            print('russian_g2p failed on', repr(accented), type(exc).__name__, flush=True)
        out.append((word, ' '.join(phones)))
    return out


def main(argv: list[str]) -> None:
    command, source, target = argv[1:4]
    t0 = time.time()
    if command in ('espeak', 'charsiu', 'ruaccent_words'):
        words = read_words(source)
        run = {'espeak': run_espeak, 'charsiu': run_charsiu, 'ruaccent_words': run_ruaccent_words}[command]
        write_pairs(target, zip(words, run(words)))
    elif command == 'ruaccent_sent':
        write_pairs(target, run_ruaccent_sentences(source))
    elif command == 'russian_g2p':
        with open(source, encoding='utf-8') as fh:
            pairs = [tuple(row[:2]) for row in csv.reader(fh, delimiter='|', quoting=csv.QUOTE_NONE)]
        write_pairs(target, run_russian_g2p(pairs, argv[4]))
    else:
        raise SystemExit(f'unknown command {command}')
    print(f'{command}: {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main(sys.argv)

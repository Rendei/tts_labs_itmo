"""Inputs of the external tools of lab 3, extra task 3 — run from `labs/lab3_g2p`::

    python external/prepare_external_inputs.py

Writes into `data/external/`:

* `test_words.txt`, `test_sentences.csv` - the `test` words and sentences for `external_tools.py`;
* `phonetisaurus_train.dict`, `phonetisaurus_words.txt`, `heldout_sample.json` - the training lexicon of
  Phonetisaurus and FRED-T5 (the same words as our OOV model, without its 5 % held-out words), the words to
  decode (the held-out sample plus the remainder of `test`) and the held-out sample with its reference.

Files are written with Unix line endings: Phonetisaurus runs in Linux and would read a `\\r` as a letter.
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())

import g2p as G                      # noqa: E402
import g2p_models as M               # noqa: E402
import train_g2p as T                # noqa: E402

OUT = 'data/external'


def write_lines(path: str, lines: list[str]) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('\n'.join(lines) + '\n')


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tokens = G.load_tokens(verbose=False)
    train, test = tokens[tokens.set == 'train'], tokens[tokens.set == 'test']
    lexicon, mfa = G.build_lexicon(train), G.load_mfa_dict()

    write_lines(f'{OUT}/test_words.txt', sorted(set(test.token)))
    with open(f'{OUT}/test_sentences.csv', 'w', encoding='utf-8', newline='\n') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE, lineterminator='\n')
        writer.writerow(['id', 'text'])
        for uid, sentence in test.groupby('id', sort=False):
            writer.writerow([uid, ' '.join(sentence.sort_values('token_id').token)])

    pairs = M.clean_pairs(T.training_pronunciations(lexicon, mfa), T.target_phones())
    reserved = set('}|_')                                        # characters Phonetisaurus uses itself
    write_lines(f'{OUT}/phonetisaurus_train.dict',
                [f"{w}\t{' '.join(p)}" for w, p in pairs if not T.is_holdout(w) and not reserved & set(w)])
    held = [(w, p) for w, p in pairs if T.is_holdout(w)]
    sample = random.Random(1).sample(held, 8000)                 # the same sample as in the report
    remainder = sorted(set(test.token[~test.token.isin(lexicon) & ~test.token.isin(mfa)]))
    write_lines(f'{OUT}/phonetisaurus_words.txt', [w for w, _ in sample] + [w for w in remainder if M.WORD_RE.fullmatch(w)])
    with open(f'{OUT}/heldout_sample.json', 'w', encoding='utf-8', newline='\n') as fh:
        json.dump({'heldout': [[w, ' '.join(p)] for w, p in sample]}, fh, ensure_ascii=False)
    write_lines(f'{OUT}/heldout_words.txt', [w for w, _ in sample])
    print(f'{len(pairs)} pairs, {len(sample)} held-out words, {len(remainder)} remainder types')


if __name__ == '__main__':
    main()

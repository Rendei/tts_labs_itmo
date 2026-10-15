"""Build the lab 3 training data from the MFA alignment — lab 3.

Reads the TextGrids downloaded in lab 2 and writes the files below (all in `data/`).

`tokens_phones.csv` — **the training table**, one row per aligned word, in sentence
order. Same shape as lab 2's `RUSLAN_pause_metadata.csv`, so the two labs read alike::

    id|token_id|token|phones|set

`tokens_phones_sil.csv` — the same table with a ``<SIL>`` row wherever the aligner put a
pause, for the acoustic model of lab 4 (`token_id` counts the pauses too).

`lexicon.csv` — the pronunciation dictionary, aggregated over the training fold::

    word|phones|count

`russian_mfa.dict` — the original MFA dictionary, copied as is.

The training table is the primary artifact: it keeps the sentence grouping, so a
word-level model, a sentence-level model and anything contextual can all be built from it.
`lexicon.csv` is a convenience summary for lookup and for word-level training, and it
necessarily throws the context away.

Utterances with a `spn` phone anywhere are dropped whole, in all three tables: `spn` marks
a word the aligner had no pronunciation for (or an artefact on the recording), and a
sentence with a hole in it is useless for a sequence model.

Run from the lab directory::

    python prepare_lexicon.py                                # the official markup, data/RUSLAN_align_v2
    python prepare_lexicon.py ../../data/mfa/align_v3 v3     # lab 2's re-alignment, into data/v3/
"""
import csv
import glob
import os
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict

import pandas as pd
from praatio import textgrid
import tqdm

ALIGN_DIR = '../../data/RUSLAN_align_v2/'
NORMALIZED_META = '../../data/metadata_RUSLAN_22200_normalized.csv'
MFA_DICT = '../../data/mfa/root/pretrained_models/dictionary/russian_mfa.dict'
DATA_DIR = 'data'

SIL = '<SIL>'
SPN = 'spn'
LENGTH_MARK = 'ː'
TEST_EVERY = 5  # lab 2's split rule: test iff the utterance number divides by 5


def phones_of_token(phones: list[tuple[float, float, str]],
                    start: float, end: float) -> list[str]:
    """Collect the phones belonging to one word interval.

    Membership is decided by the phone's midpoint rather than its boundaries: word and
    phone tiers are written independently and a shared boundary can differ in the last
    decimal.

    Args:
        phones: `(start, end, label)` of every phone interval of the utterance.
        start: Word interval start, seconds.
        end: Word interval end, seconds.

    Returns:
        Phone labels in time order, NFC-normalized with the length mark collapsed to match
        `configs/phones_mfa.txt`. An empty label (silence) becomes ``"<SIL>"``.
    """
    out = []
    for phone_start, phone_end, label in phones:
        if start <= (phone_start + phone_end) / 2 < end:
            label = unicodedata.normalize('NFC', label) if label else SIL
            out.append(label.replace(LENGTH_MARK, ''))
    return out


def read_kept_ids(path: str) -> set[str] | None:
    """Utterance ids that survived the lab 1 filter, or None if the file is absent.

    The alignment was built on **raw** text, so it contains utterances a normalizer would
    reject — including three all-caps abbreviations whose transcriptions `mfa g2p` invented
    (`кпсс` → `k r ə f ɨ s̪`). Restricting to the lab 1 output removes them.
    """
    if not os.path.isfile(path):
        print(f'{path} not found: keeping every aligned utterance')
        return None
    with open(path, encoding='utf-8') as fh:
        return {line.split('|', 1)[0] for line in fh if line.strip()}


def read_alignment(align_root: str, keep: set[str] | None) -> list[dict]:
    """Read every TextGrid in `align_root` into per-token records, pauses included.

    Args:
        align_root: Directory with `{id}.TextGrid` files.
        keep: Utterance ids to keep, or None for all.

    Returns:
        One dict per interval of the word tier with keys `id`, `token_id`, `token`, `phones`,
        `set`. A pause has `token == phones == "<SIL>"`.
    """
    rows = []
    for path in tqdm.tqdm(sorted(glob.glob(os.path.join(align_root, '*.TextGrid')))):
        utterance_id = os.path.basename(path)[:-len('.TextGrid')]
        if keep is not None and utterance_id not in keep:
            continue
        try:
            tg = textgrid.openTextgrid(path, True)
        except Exception as exc:
            print(f'Failed to read {utterance_id}: {exc}')
            continue
        word_tier, phone_tier = tg.tiers
        phones = [(p.start, p.end, p.label) for p in phone_tier.entries]
        fold = 'test' if int(utterance_id.split('_')[0]) % TEST_EVERY == 0 else 'train'

        for token_id, entry in enumerate(word_tier.entries):
            if entry.label == '':
                token = current_phones = SIL
            else:
                token = unicodedata.normalize('NFC', entry.label)
                current_phones = ' '.join(phones_of_token(phones, entry.start, entry.end))
            rows.append({
                'id': utterance_id,
                'token_id': token_id,
                'token': token,
                'phones': current_phones,
                'set': fold,
            })
    return rows


def drop_noisy_utterances(rows: list[dict]) -> tuple[list[dict], set[str]]:
    """Remove every utterance that has a `spn` phone or a word without any phone.

    Returns:
        The remaining rows and the set of dropped utterance ids.
    """
    bad = {r['id'] for r in rows if SPN in r['phones'].split() or not r['phones']}
    return [r for r in rows if r['id'] not in bad], bad


def without_pauses(rows: list[dict]) -> list[dict]:
    """Words only, numbered from zero inside each utterance."""
    out, counters = [], Counter()
    for row in rows:
        if row['token'] == SIL:
            continue
        out.append({**row, 'token_id': counters[row['id']]})
        counters[row['id']] += 1
    return out


def build_lexicon(rows: list[dict]) -> dict[str, Counter]:
    """Count pronunciations per token over the training fold only.

    Building this over every fold instead would put the test words in the dictionary and
    make the reported PER meaningless.
    """
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if row['set'] == 'train':
            lexicon[row['token']][row['phones']] += 1
    return lexicon


def write_table(rows: list[dict], path: str) -> None:
    pd.DataFrame(rows).to_csv(path, encoding='utf-8', quoting=csv.QUOTE_NONE, sep='|',
                              index=False, header=True)


def main(align_dir: str = ALIGN_DIR, subdir: str = '') -> None:
    """Read the alignment, write the training tables, the lexicon and the MFA dictionary."""
    out_dir = os.path.join(DATA_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    rows = read_alignment(align_dir, read_kept_ids(NORMALIZED_META))
    n_utterances = len({r['id'] for r in rows})
    rows, dropped = drop_noisy_utterances(rows)
    print(f'{n_utterances} utterances in {align_dir}, {len(dropped)} dropped for `spn` '
          f'({len(dropped) / n_utterances:.1%})')

    words = without_pauses(rows)
    test = {r['id'] for r in words if r['set'] == 'test'}
    utterances = {r['id'] for r in words}
    print(f'{len(words)} tokens in {len(utterances)} utterances '
          f'({len(test)} test, {len(test) / max(len(utterances), 1):.1%})')

    write_table(words, os.path.join(out_dir, 'tokens_phones.csv'))
    write_table(rows, os.path.join(out_dir, 'tokens_phones_sil.csv'))

    lexicon = build_lexicon(words)
    variants = sum(1 for counts in lexicon.values() if len(counts) > 1)
    print(f'lexicon from the train fold: {len(lexicon)} types, '
          f'{variants} with several pronunciations')

    with open(os.path.join(out_dir, 'lexicon.csv'), 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE)
        writer.writerow(['word', 'phones', 'count'])
        for word in sorted(lexicon):
            for phones, count in lexicon[word].most_common():
                writer.writerow([word, phones, count])

    if os.path.isfile(MFA_DICT):
        shutil.copyfile(MFA_DICT, os.path.join(out_dir, 'russian_mfa.dict'))
    else:
        print(f'{MFA_DICT} not found: download russian_mfa.dict by hand into {out_dir}')


if __name__ == '__main__':
    main(*sys.argv[1:3])

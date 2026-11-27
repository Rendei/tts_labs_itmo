"""Dictionaries with explicit stress — lab 3, extra tasks 1 and 2.

The phone inventory of the corpus has no stress mark. The stress is encoded in the quality of the vowel
(`a o e u i ɛ æ ɵ` stressed, `ɐ ə ɪ ʊ` reduced, `ɨ ʉ` ambiguous), which `g2p_models.derive_stress`
reads back. Two kinds of dictionaries are written from it:

* **stressed** (`lexicon_stressed.csv`, `russian_mfa_stressed.dict`): the IPA stress mark `ˈ` becomes a
  separate phone in front of the stressed vowel: `z̪ ˈ a m ə k`;
* **controllable** (`lexicon_controllable.csv`, `russian_mfa_controllable.dict`): the original entries
  plus, for every pronunciation whose stress is known, a spelling with `+` in front of the stressed vowel
  letter, `з+амок -> z̪ a m ə k`, so that the user can request a stress in the text.

A pronunciation whose stress cannot be read from the vowels (no full vowel, as in the unstressed clitics, or
several candidates) stays unmarked.
"""
import csv
import unicodedata
from collections import Counter

import g2p_models as M

STRESS_MARK = 'ˈ'


def mark_stress(phones: list[str]) -> tuple[list[str], bool]:
    """Insert `ˈ` before the stressed vowel; returns the phones and whether the stress was found."""
    plain = [p.replace('ː', '') for p in phones]
    index = M.derive_stress(plain)
    if index is None:
        return list(phones), False
    return phones[:index] + [STRESS_MARK] + phones[index:], True


def controllable_spelling(word: str, phones: list[str]) -> str | None:
    """`word` with `+` before the stressed vowel letter, or None if the stress is unknown."""
    index = M.stress_letter(word, [p.replace('ː', '') for p in phones])
    return None if index is None else word[:index] + '+' + word[index:]


def stress_statistics(items: list[tuple[str, list[str]]]) -> Counter:
    """How the stress of `(word, phones)` pairs reads off the vowels."""
    counts: Counter = Counter()
    for word, phones in items:
        plain = [p.replace('ː', '') for p in phones]
        vowels = [p for p in plain if p in M.VOWEL_PHONES]
        strong = [p for p in vowels if p in M.STRONG]
        if M.derive_stress(plain) is not None:
            counts['one stressed vowel found'] += 1
        elif not vowels:
            counts['no vowel'] += 1
        elif not strong and not [p for p in vowels if p in M.WEAK]:
            counts['only reduced vowels (no stress mark readable)'] += 1
        else:
            counts['several candidates'] += 1
    return counts


def write_lexicons(lexicon: dict[str, Counter], stressed_path: str, controllable_path: str) -> dict:
    """`lexicon_stressed.csv` and `lexicon_controllable.csv` from the training lexicon."""
    n = Counter()
    with open(stressed_path, 'w', encoding='utf-8', newline='') as fs, \
            open(controllable_path, 'w', encoding='utf-8', newline='') as fc:
        ws = csv.writer(fs, delimiter='|', quoting=csv.QUOTE_NONE)
        wc = csv.writer(fc, delimiter='|', quoting=csv.QUOTE_NONE)
        for out in (ws, wc):
            out.writerow(['word', 'phones', 'count'])
        for word in sorted(lexicon):
            for phones, count in lexicon[word].most_common():
                marked, ok = mark_stress(phones.split())
                ws.writerow([word, ' '.join(marked), count])
                wc.writerow([word, phones, count])
                n['entries'] += 1
                spelled = controllable_spelling(word, phones.split())
                if spelled is not None:
                    wc.writerow([spelled, phones, count])
                    n['with stress'] += 1
    return dict(n)


def write_mfa_dicts(path_in: str, stressed_path: str, controllable_path: str) -> dict:
    """`russian_mfa_stressed.dict` and `russian_mfa_controllable.dict` from the original dictionary.

    The original line format is kept (probability columns included); phones keep their length marks.
    """
    n = Counter()
    with open(path_in, encoding='utf-8') as fi, open(stressed_path, 'w', encoding='utf-8', newline='') as fs, \
            open(controllable_path, 'w', encoding='utf-8', newline='') as fc:
        for line in fi:
            line = line.rstrip('\n')
            parts = line.split('\t')
            word = unicodedata.normalize('NFC', parts[0])
            phones = unicodedata.normalize('NFC', parts[-1]).split()
            marked, ok = mark_stress(phones)
            fs.write('\t'.join(parts[:-1] + [' '.join(marked)]) + '\n')
            fc.write(line + '\n')
            n['entries'] += 1
            n['stressed'] += ok
            spelled = controllable_spelling(word, phones) if ok else None
            if spelled is not None:
                fc.write('\t'.join([spelled] + parts[1:]) + '\n')
                n['with +'] += 1
    return dict(n)

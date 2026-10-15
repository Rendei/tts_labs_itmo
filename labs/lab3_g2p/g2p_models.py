"""Statistical model for words that no dictionary knows — lab 3.

Russian spelling is close to phonemic: once the stressed vowel is known, the phones of a word
follow almost letter by letter from the letters around. So the model has two stages, both
learned from pronunciation dictionaries:

1. **Stress**: which vowel letter carries the stress (`StressModel`). This is the hard part, since
   Russian stress is free and not marked in the spelling.
2. **Letters to phones** given the stress (`LetterModel`): one small gradient-boosting classifier
   per letter predicts the phones of that letter (0, 1 or 2 of them) from the letters around it
   and from the position of the letter relative to the stressed vowel.

The dictionaries give pronunciations but no alignment between letters and phones, and no stress
mark: the alignment is estimated with hard EM (`Aligner`), the stress is read off the vowel
quality (`derive_stress`: `a o e u i ɛ æ ɵ` are stressed, `ɐ ə ɪ ʊ` are reduced, `ɨ ʉ` are
ambiguous).

Everything here is fitted on the MFA dictionary and the training fold of the corpus only.
"""
import gzip
import math
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

import g2p_rules as R

VOWEL_PHONES = frozenset('a e i o u æ ɐ ə ɛ ɨ ɪ ɵ ʉ ʊ'.split())
STRONG = frozenset('a e i o u æ ɛ ɵ'.split())      # full vowels: they mean stress
WEAK = frozenset('ɨ ʉ')                              # full-quality vowels that also occur unstressed
WORD_RE = re.compile('[а-яё]{2,}')
LETTER_CODE = {c: i + 1 for i, c in enumerate(R.LETTERS)}      # 0 is the word boundary
MAX_CHUNK = 2


def derive_stress(phones: list[str]) -> int | None:
    """Index of the stressed phone of a pronunciation, read off the vowel quality.

    A single full vowel (`a e i o u ɛ æ ɵ`) is the stressed one; a word-final `e` does not count
    when there is another, because MFA keeps the unstressed final `-е` unreduced. With no full
    vowel a single `ɨ`/`ʉ` is taken. Everything else (several candidates, or none: clitics) is
    ambiguous and returns None.
    """
    vowels = [i for i, p in enumerate(phones) if p in VOWEL_PHONES]
    strong = [i for i in vowels if phones[i] in STRONG]
    if len(strong) > 1 and phones[strong[-1]] == 'e' and strong[-1] == len(phones) - 1:
        strong = strong[:-1]
    if len(strong) == 1:
        return strong[0]
    if not strong:
        weak = [i for i in vowels if phones[i] in WEAK]
        if len(weak) == 1:
            return weak[0]
    return None


# --------------------------------------------------------------------------------------
# alignment of letters and phones
# --------------------------------------------------------------------------------------
class Aligner:
    """Monotonic alignment of letters to phones: every letter takes 0, 1 or 2 phones.

    The scores `log P(chunk | letter)` start from a rough prior (a vowel letter takes a chunk with
    exactly one vowel phone, a consonant letter one without vowels) and are re-estimated from the
    best alignments (hard EM), which is enough because the spelling is nearly phonemic.
    """

    def __init__(self) -> None:
        self.logp: dict[tuple[str, tuple[str, ...]], float] = {}

    @staticmethod
    def _prior(letter: str, chunk: tuple[str, ...]) -> float:
        n_vowels = sum(p in VOWEL_PHONES for p in chunk)
        if letter in R.VOWEL_LETTERS:
            if n_vowels == 1 and len(chunk) <= 2:
                return 0.0 if len(chunk) == 1 or chunk[0] == 'j' else -3.0
            return -4.0 if not chunk else -10.0
        if letter in 'ьъ':
            return 0.0 if not chunk else -8.0
        if n_vowels:
            return -12.0
        return {0: -3.0, 1: 0.0, 2: -3.0}[len(chunk)]

    def score(self, letter: str, chunk: tuple[str, ...]) -> float:
        value = self.logp.get((letter, chunk))
        return self._prior(letter, chunk) if value is None else value

    def align(self, word: str, phones: list[str]) -> list[tuple[int, int]] | None:
        """Best split of `phones` into one span `(start, end)` per letter, or None if impossible."""
        n, m = len(word), len(phones)
        if m > MAX_CHUNK * n:
            return None
        neg = -1e18
        best = [[neg] * (m + 1) for _ in range(n + 1)]
        back = [[0] * (m + 1) for _ in range(n + 1)]
        best[0][0] = 0.0
        chunks = [[tuple(phones[j - k:j]) if j >= k else None for k in range(MAX_CHUNK + 1)] for j in range(m + 1)]
        for i in range(1, n + 1):
            letter = word[i - 1]
            prev_row, row, brow = best[i - 1], best[i], back[i]
            for j in range(m + 1):
                top = neg
                arg = 0
                for k in range(MAX_CHUNK + 1):
                    if j - k < 0:
                        break
                    base = prev_row[j - k]
                    if base <= neg:
                        continue
                    value = base + self.score(letter, chunks[j][k])
                    if value > top:
                        top, arg = value, k
                row[j], brow[j] = top, arg
        if best[n][m] <= neg:
            return None
        spans, j = [], m
        for i in range(n, 0, -1):
            k = back[i][j]
            spans.append((j - k, j))
            j -= k
        return spans[::-1]

    def fit(self, pairs: list[tuple[str, list[str]]], rounds: int = 3, sample: int = 60000, seed: int = 0) -> None:
        """Hard EM on a sample of `(word, phones)` pairs."""
        rng = random.Random(seed)
        subset = rng.sample(pairs, min(sample, len(pairs)))
        for _ in range(rounds):
            counts: dict[str, Counter] = defaultdict(Counter)
            for word, phones in subset:
                spans = self.align(word, phones)
                if spans is None:
                    continue
                for letter, (a, b) in zip(word, spans):
                    counts[letter][tuple(phones[a:b])] += 1
            self.logp = {}
            for letter, counter in counts.items():
                total = sum(counter.values()) + 0.01 * len(counter)
                for chunk, c in counter.items():
                    self.logp[(letter, chunk)] = math.log((c + 0.01) / total)


def stress_letter(word: str, phones: list[str]) -> int | None:
    """Index of the stressed vowel *letter* of a dictionary word, or None when it is unknown.

    The stressed phone comes from `derive_stress`; it is mapped to a letter by counting vowels,
    which is valid when the word has as many vowel letters as vowel phones (99.7 % of the words).
    """
    phone = derive_stress(phones)
    if phone is None:
        return None
    vowel_phones = [i for i, p in enumerate(phones) if p in VOWEL_PHONES]
    vowel_letters = R.vowel_positions(word)
    if len(vowel_phones) != len(vowel_letters):
        return None
    return vowel_letters[vowel_phones.index(phone)]


class SiblingStress:
    """Stress evidence for a word from the *other forms of the same lexeme* that a dictionary knows.

    Russian stress is mostly stable across the forms of a word: either it stays on the same
    syllable of the stem (`мятеж-ный`, `мятеж-ном`) or on the same ending. A word that is missing
    from the dictionaries often has its relatives there (66 % of the remainder, report section 2.3).
    pymorphy3 lists the forms of the lexeme; for each known form whose stress lies in the part shared
    with the word (common prefix, or common suffix) a vote is cast for the same letter of the word.

    Args:
        pronunciation: `word -> phones` of a dictionary word, or None if it is not there.
    """

    def __init__(self, pronunciation: Callable[[str], list[str] | None]) -> None:
        self.pronunciation = pronunciation
        self._analyzer = None
        self._lexemes: dict[tuple, frozenset] = {}
        self._stress_cache: dict[str, int | None] = {}

    def _stress_of(self, spelling: str) -> int | None:
        if spelling not in self._stress_cache:
            phones = self.pronunciation(spelling)
            self._stress_cache[spelling] = stress_letter(spelling, phones) if phones else None
        return self._stress_cache[spelling]

    def forms(self, word: str) -> set[str]:
        if self._analyzer is None:
            import pymorphy3
            self._analyzer = pymorphy3.MorphAnalyzer()
        out: set[str] = set()
        for parse in self._analyzer.parse(word)[:2]:
            key = (parse.normal_form, str(parse.tag.POS))
            if key not in self._lexemes:
                self._lexemes[key] = frozenset(f.word for f in parse.lexeme)
            out |= self._lexemes[key]
        return out

    def votes(self, word: str) -> tuple[Counter, Counter, int]:
        """`(votes_by_prefix, votes_by_suffix, n_forms_with_known_stress)`; keys are letter indexes of `word`."""
        by_prefix, by_suffix, known = Counter(), Counter(), 0
        seen = {word, word.replace('ё', 'е')}
        for form in self.forms(word):
            for spelling in {form, form.replace('ё', 'е')}:
                if spelling in seen or not WORD_RE.fullmatch(spelling):
                    continue
                seen.add(spelling)
                st = self._stress_of(spelling)
                if st is None:
                    continue
                known += 1
                lcp = 0
                while lcp < min(len(word), len(spelling)) and word[lcp] == spelling[lcp]:
                    lcp += 1
                lcs = 0
                while lcs < min(len(word), len(spelling)) and word[-1 - lcs] == spelling[-1 - lcs]:
                    lcs += 1
                if st < lcp:
                    by_prefix[st] += 1
                from_end = len(spelling) - 1 - st
                if from_end < lcs:
                    by_suffix[len(word) - 1 - from_end] += 1
        return by_prefix, by_suffix, known

    def extra(self, word: str) -> dict[int, dict[str, str]]:
        """Features of each vowel letter (by its number among the vowels) from the votes."""
        by_prefix, by_suffix, known = self.votes(word)
        out: dict[int, dict[str, str]] = {}
        bins = lambda x: '0' if x == 0 else ('1' if x <= .25 else ('2' if x <= .5 else ('3' if x <= .75 else '4')))
        for k, i in enumerate(R.vowel_positions(word)):
            if known:
                p, s = by_prefix[i] / known, by_suffix[i] / known
            else:
                p = s = 0.0
            out[k] = {'sp': bins(p), 'ss': bins(s), 'sm': str(min(known, 4)),
                      'spm': f'{bins(p)}|{min(known, 3)}', 'ssm': f'{bins(s)}|{min(known, 3)}',
                      'sps': f'{bins(p)}|{bins(s)}'}
        return out


# --------------------------------------------------------------------------------------
# stage 1: stress
# --------------------------------------------------------------------------------------
def stress_candidate_features(word: str, vowels: list[int], k: int, extra: dict | None = None) -> list[str]:
    """Sparse string features of the k-th vowel letter of `word` being the stressed one."""
    i, nv, n = vowels[k], len(vowels), len(word)
    fe, fs, de = nv - 1 - k, k, n - 1 - i
    fe, fs, de_c = min(fe, 6), min(fs, 6), min(de, 8)
    pad = '^' + word + '$'
    j = i + 1                                   # position in the padded word
    feats = [
        f'L={word[i]}', f'fs={fs}', f'fe={fe}', f'nv={min(nv, 9)}|fs={fs}', f'nv={min(nv, 9)}|fe={fe}',
        f'L|fe={word[i]}|{fe}', f'L|fs={word[i]}|{fs}',
        f'Lm1={pad[j - 1]}', f'Lp1={pad[j + 1]}', f'Lm2={pad[j - 2] if j >= 2 else "^"}',
        f'Lp2={pad[j + 2] if j + 2 < len(pad) else "$"}', f'Lm1L={pad[j - 1]}{word[i]}', f'LLp1={word[i]}{pad[j + 1]}',
        f'Lm1LLp1={pad[j - 1]}{word[i]}{pad[j + 1]}',
    ]
    for s in range(1, 7):
        feats.append(f'suf{s}={word[-s:]}|de={de_c}')
        feats.append(f'suf{s}={word[-s:]}|fe={fe}')
    for p in range(1, 5):
        feats.append(f'pre{p}={word[:p]}|fs={min(k, 4)}')
    consonants_after = 0
    for c in word[i + 1:]:
        if c in R.VOWEL_LETTERS:
            break
        consonants_after += 1
    feats.append(f'ca={min(consonants_after, 3)}|L={word[i]}')
    if word[i] == 'ё':
        feats.append('yo')
    if extra:
        for name, value in extra.get(k, {}).items():
            feats.append(f'{name}={value}')
    return feats


class StressModel:
    """Logistic regression over hashed sparse features, one score per vowel letter."""

    def __init__(self, n_features: int = 2 ** 20) -> None:
        self.hasher = FeatureHasher(n_features=n_features, input_type='string', alternate_sign=False)
        self.clf = SGDClassifier(loss='log_loss', alpha=3e-7, max_iter=15, tol=None, random_state=0)

    def fit(self, words: list[str], stress_letters: list[int], extras: list[dict | None] | None = None) -> None:
        rows, labels = [], []
        for idx, (word, stress) in enumerate(zip(words, stress_letters)):
            vowels = R.vowel_positions(word)
            extra = extras[idx] if extras else None
            for k, i in enumerate(vowels):
                rows.append(stress_candidate_features(word, vowels, k, extra))
                labels.append(int(i == stress))
        x = self.hasher.transform(rows)
        self.clf.fit(x, np.asarray(labels))
        self.clf.coef_ = self.clf.coef_.astype(np.float32)          # halves the size of the saved model

    def scores(self, word: str, extra: dict | None = None) -> list[float]:
        vowels = R.vowel_positions(word)
        if not vowels:
            return []
        rows = [stress_candidate_features(word, vowels, k, extra) for k in range(len(vowels))]
        return list(self.clf.decision_function(self.hasher.transform(rows)))

    def predict(self, word: str, extra: dict | None = None) -> int | None:
        """Index in `word` of the vowel letter that is most likely stressed."""
        vowels = R.vowel_positions(word)
        if not vowels:
            return None
        if 'ё' in word:
            return word.index('ё')
        if len(vowels) == 1:
            return vowels[0]
        scores = self.scores(word, extra)
        return vowels[int(np.argmax(scores))]


# --------------------------------------------------------------------------------------
# stage 2: letters to phones
# --------------------------------------------------------------------------------------
N_CAT = 10                      # the first 10 columns are categorical for the boosting model
N_FEATURES = 15


def letter_features(word: str, stress: int | None) -> np.ndarray:
    """Feature matrix, one row per letter of `word`; `stress` is the index of the stressed vowel letter."""
    n = len(word)
    vowels = R.vowel_positions(word)
    stress_no = vowels.index(stress) if stress in vowels else None
    offsets = {}
    if stress_no is not None:
        offsets = {i: k - stress_no for k, i in enumerate(vowels)}
    codes = [0] * 3 + [LETTER_CODE[c] for c in word] + [0] * 3

    def off(i):
        return min(max(offsets[i], -5), 5) + 6 if i in offsets else 0

    rows = np.zeros((n, N_FEATURES), dtype=np.int16)
    next_vowel = [None] * (n + 1)
    last = None
    for i in range(n - 1, -1, -1):
        if i in offsets:
            last = i
        next_vowel[i] = last
    prev_vowel = None
    for i in range(n):
        rows[i, :7] = codes[i:i + 7]
        rows[i, 7] = off(i)
        nv = next_vowel[i + 1] if i + 1 < n else None
        rows[i, 8] = off(nv) if nv is not None else 0
        rows[i, 9] = off(prev_vowel) if prev_vowel is not None else 0
        rows[i, 10] = min(i, 5)
        rows[i, 11] = min(n - 1 - i, 5)
        rows[i, 12] = min(len(vowels), 8)
        rows[i, 13] = int(stress_no is not None)
        rows[i, 14] = min(sum(1 for c in word[i + 1:] if c in R.VOWEL_LETTERS), 6)
        if i in offsets:
            prev_vowel = i
    return rows


class LetterModel:
    """One boosting classifier per letter; classes are the phone chunks `"j a"`, `"ɐ"`, `""`, ..."""

    def __init__(self) -> None:
        self.models: dict[str, object] = {}        # letter -> classifier or constant chunk (str)

    @staticmethod
    def _fit_letter(letter: str, x: np.ndarray, y: list[str], max_iter: int, seed: int, min_class: int = 20):
        """Fit the classifier of one letter.

        Classes with fewer than `min_class` examples are dropped together with their rows: they are
        alignment errors rather than pronunciations, and a handful of such classes makes gradient
        boosting diverge.
        """
        y = np.asarray(y)
        classes, freq = np.unique(y, return_counts=True)
        keep = np.isin(y, classes[freq >= min_class])
        x, y = x[keep], y[keep]
        if len(set(y)) == 1:
            return letter, str(y[0])
        clf = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=0.15, l2_regularization=1.0, max_leaf_nodes=24, min_samples_leaf=20,
            categorical_features=list(range(N_CAT)), early_stopping=False, random_state=seed)
        clf.fit(x, y)
        return letter, clf

    def fit(self, data: dict[str, tuple[np.ndarray, list[str]]], max_iter: int = 60, n_jobs: int = 4) -> None:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_letter)(letter, x, y, max_iter, 0) for letter, (x, y) in data.items())
        self.models = dict(results)

    def predict_rows(self, letter: str, x: np.ndarray) -> list[list[str]]:
        model = self.models.get(letter)
        if model is None:
            return [[] for _ in range(len(x))]
        if isinstance(model, str):
            return [model.split()] * len(x)
        return [chunk.split() for chunk in model.predict(x)]

    def transcribe(self, word: str, stress: int | None) -> list[str]:
        return self.transcribe_many([word], [stress])[0]

    def transcribe_many(self, words: list[str], stresses: list[int | None]) -> list[list[str]]:
        """Transcribe many words at once: the rows of every letter go through its classifier in one batch."""
        rows_of: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for wi, (word, stress) in enumerate(zip(words, stresses)):
            x = letter_features(word, stress)
            for i, letter in enumerate(word):
                rows_of[letter][0].append(x[i])
                rows_of[letter][1].append((wi, i))
        out = [[[] for _ in word] for word in words]
        for letter, (rows, where) in rows_of.items():
            for (wi, i), chunk in zip(where, self.predict_rows(letter, np.asarray(rows, dtype=np.int16))):
                out[wi][i] = chunk
        return [[p for chunk in word for p in chunk] for word in out]


# --------------------------------------------------------------------------------------
# the whole model
# --------------------------------------------------------------------------------------
class OOVModel:
    """Pronunciation of a word that is in no dictionary: stress model + letter model."""

    def __init__(self, stress: StressModel, letters: LetterModel, use_siblings: bool = False,
                 stress_plain: StressModel | None = None) -> None:
        self.stress = stress                  # the stress model; reads the sibling evidence if `use_siblings`
        self.stress_plain = stress_plain      # the same without the morphological evidence (for the ablation)
        self.letters = letters
        self.use_siblings = use_siblings
        self.siblings: SiblingStress | None = None
        self.meta: dict = {}                  # numbers measured at training time, for the report

    def set_dictionaries(self, pronunciation: Callable[[str], list[str] | None]) -> None:
        """Give the model the dictionaries it needs at run time for the stress evidence of related forms."""
        self.siblings = SiblingStress(pronunciation)

    def predict_stress(self, word: str) -> int | None:
        if self.use_siblings and self.siblings is not None:
            return self.stress.predict(word, self.siblings.extra(word))
        return (self.stress_plain or self.stress).predict(word)

    def transcribe(self, word: str, stress: int | None = None) -> list[str]:
        """Phones of `word` (Cyrillic letters only). `stress` is the index of the stressed vowel
        letter; when it is None the stress model chooses it."""
        word = R.clean(word)
        if not word:
            return []
        if stress is None:
            stress = self.predict_stress(word)
        return self.letters.transcribe(word, stress)

    def transcribe_many(self, words: list[str], stresses: list[int | None] | None = None) -> list[list[str]]:
        """Batch version of :meth:`transcribe`; `stresses` defaults to the stress model's choice."""
        words = [R.clean(w) for w in words]
        stresses = list(stresses) if stresses is not None else [None] * len(words)
        stresses = [self.predict_stress(w) if (s is None and w) else s for w, s in zip(words, stresses)]
        keep = [i for i, w in enumerate(words) if w]
        done = self.letters.transcribe_many([words[i] for i in keep], [stresses[i] for i in keep])
        out: list[list[str]] = [[] for _ in words]
        for i, phones in zip(keep, done):
            out[i] = phones
        return out

    def __getstate__(self):
        state = dict(self.__dict__)
        state['siblings'] = None                 # holds the analyzer and dictionary lookups: rebuilt on load
        return state

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with gzip.open(path, 'wb', compresslevel=6) as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'OOVModel':
        with gzip.open(path, 'rb') as fh:
            return pickle.load(fh)


def clean_pairs(pronunciations: dict[str, list[tuple[float, str]]], target: set[str]) -> list[tuple[str, list[str]]]:
    """`(word, phones)` pairs of Cyrillic words whose best pronunciation uses only target phones."""
    pairs = []
    for word, variants in pronunciations.items():
        if not WORD_RE.fullmatch(word):
            continue
        phones = variants[0][1].split()
        if phones and all(p in target for p in phones):
            pairs.append((word, phones))
    return pairs


def prepare_training(pairs: list[tuple[str, list[str]]], aligner: Aligner):
    """Align every pair and read the stress off the phones.

    Returns:
        A list of `(word, spans, phones, stress_letter)` for the pairs that align and whose stress
        is unambiguous.
    """
    out = []
    for word, phones in pairs:
        stress_phone = derive_stress(phones)
        if stress_phone is None:
            continue
        spans = aligner.align(word, phones)
        if spans is None:
            continue
        stress_letter = next((i for i, (a, b) in enumerate(spans) if a <= stress_phone < b), None)
        if stress_letter is None or word[stress_letter] not in R.VOWEL_LETTERS:
            continue
        out.append((word, spans, phones, stress_letter))
    return out

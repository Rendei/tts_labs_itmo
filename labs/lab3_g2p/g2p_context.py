"""Choosing between several pronunciations of a word from its context — lab 3.

A word of the dictionary often has more than one pronunciation. Three kinds of reasons were seen
in the corpus (report, section 2.5):

* homographs and function words that are stressed or not depending on the sentence
  (`все`/`всё`, `что`, `на`, `не`, `то`);
* assimilation across the word boundary (`в` is `v` or `f`, `с` is `s̪`/`z̪`/`sʲ`, `из`, `к`):
  it is decided by the first phone of the *next* word;
* free variants of word endings.

`VariantChooser` learns this from the training sentences: for every frequent ambiguous word one small
gradient-boosting classifier predicts the variant from the neighbouring words, their first/last
phones and their morphology (pymorphy3). A pause token `<SIL>` is an ordinary neighbour, and the
neighbours *beyond* a pause are given as well, so a pause that appears or disappears does not blind
the model.
"""
from collections import Counter, defaultdict
from typing import Callable

import numpy as np
import pymorphy3
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

SIL = '<SIL>'


class Morph:
    """pymorphy3 tags of a word, cached: `(POS, case, gender, number)` as short strings."""

    def __init__(self) -> None:
        self._analyzer = None
        self._cache: dict[str, tuple[str, str, str, str]] = {}

    @property
    def analyzer(self):
        if self._analyzer is None:
            self._analyzer = pymorphy3.MorphAnalyzer()
        return self._analyzer

    def tags(self, word: str) -> tuple[str, str, str, str]:
        if word not in self._cache:
            tag = self.analyzer.parse(word)[0].tag
            self._cache[word] = (str(tag.POS or ''), str(tag.case or ''), str(tag.gender or ''), str(tag.number or ''))
        return self._cache[word]

    def __getstate__(self):                      # the analyzer is rebuilt on demand, never pickled
        return {'_analyzer': None, '_cache': {}}


class VariantChooser:
    """Per-word classifiers over contextual features; see the module docstring.

    Args:
        vocab_size: How many of the most frequent words get their own identity as a context feature.
        min_count: A word is modelled if it occurs at least this many times in the training sentences.
        min_variant: A pronunciation is a class only if it was seen at least this many times.
    """

    def __init__(self, vocab_size: int = 200, min_count: int = 30, min_variant: int = 3) -> None:
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.min_variant = min_variant
        self.morph = Morph()
        self.vocab: dict[str, int] = {}
        self.phone_ids: dict[str, int] = {}
        self.encoders: list[dict[str, int]] = [{} for _ in range(4)]
        self.models: dict[str, tuple] = {}
        self.fit_report: list[tuple] = []

    # ---- features -------------------------------------------------------------------------
    def _wid(self, word: str) -> int:
        if word == SIL:
            return self.vocab_size + 1
        if word is None:
            return self.vocab_size + 2
        return self.vocab.get(word, 0)

    def _tag_ids(self, word: str | None) -> tuple[int, int, int, int]:
        if word is None or word == SIL:
            return 0, 0, 0, 0
        return tuple(self.encoders[k].get(v, 0) for k, v in enumerate(self.morph.tags(word)))

    def features(self, tokens: list[str], i: int, phones_of: Callable[[str], list[str]]) -> list[int]:
        """Feature vector of the token at position `i` of a sentence (lowercase tokens, `<SIL>` allowed)."""
        n = len(tokens)

        def neighbour(start: int, step: int) -> int | None:
            j = start
            while 0 <= j < n:
                if tokens[j] != SIL:
                    return j
                j += step
            return None

        n1 = neighbour(i + 1, 1)
        n2 = neighbour(n1 + 1, 1) if n1 is not None else None
        p1 = neighbour(i - 1, -1)
        p2 = neighbour(p1 - 1, -1) if p1 is not None else None
        word = lambda j: tokens[j] if j is not None else None
        nxt_phones = phones_of(tokens[n1]) if n1 is not None else []
        prv_phones = phones_of(tokens[p1]) if p1 is not None else []
        next_letter = ord(tokens[n1][0]) - 0x430 if n1 is not None and tokens[n1] else -1
        cur, prv, nxt = self._tag_ids(tokens[i]), self._tag_ids(word(p1)), self._tag_ids(word(n1))
        phone_id = lambda p: self.phone_ids.get(p, 0) if p else 0
        return [
            self._wid(word(p1)), self._wid(word(p2)), self._wid(word(n1)), self._wid(word(n2)),
            phone_id(nxt_phones[0] if nxt_phones else ''), phone_id(prv_phones[-1] if prv_phones else ''),
            next_letter + 1, cur[0], prv[0], nxt[0], cur[1], nxt[1], prv[1], nxt[2], nxt[3], cur[2], cur[3],
            int(i > 0 and tokens[i - 1] == SIL), int(i + 1 < n and tokens[i + 1] == SIL),
            min(i, 15), min(n - 1 - i, 15),
        ]

    N_CATEGORICAL = 17

    # ---- fitting --------------------------------------------------------------------------
    def fit(self, sentences: list[list[str]], gold: list[list[str]], lexicon: dict[str, Counter],
            phones_of: Callable[[str], list[str]], verbose: bool = False) -> None:
        """Fit the classifiers.

        Args:
            sentences: Training sentences, lowercase tokens, with `<SIL>` where there are pauses.
            gold: The phone string of every token (`"<SIL>"` for a pause), parallel to `sentences`.
            lexicon: Training-fold lexicon; tells which words have several pronunciations.
            phones_of: Context-free pronunciation of a word, used for the neighbours' phones.
        """
        counts = Counter(t for s in sentences for t in s if t != SIL)
        self.vocab = {w: k + 1 for k, (w, _) in enumerate(counts.most_common(self.vocab_size))}
        phone_list = sorted({p for w in counts for p in phones_of(w)})
        self.phone_ids = {p: k + 1 for k, p in enumerate(phone_list)}
        for k in range(4):
            values = sorted({self.morph.tags(w)[k] for w in counts})
            self.encoders[k] = {v: j + 1 for j, v in enumerate(values) if v}

        targets = {w for w, c in lexicon.items()
                   if sum(1 for v in c.values() if v >= self.min_variant) >= 2 and counts[w] >= self.min_count}
        rows: dict[str, list[list[int]]] = defaultdict(list)
        labels: dict[str, list[str]] = defaultdict(list)
        for tokens, phones in zip(sentences, gold):
            for i, tok in enumerate(tokens):
                if tok in targets:
                    rows[tok].append(self.features(tokens, i, phones_of))
                    labels[tok].append(phones[i])
        self.models = {}
        report = []
        for word in sorted(targets):
            x = np.asarray(rows[word], dtype=np.int32)
            y = np.asarray(labels[word])
            keep = np.array([lexicon[word][p] >= self.min_variant for p in y])
            x, y = x[keep], y[keep]
            classes, freq = np.unique(y, return_counts=True)
            majority = classes[int(np.argmax(freq))]
            if len(classes) < 2 or freq.min() < 3:
                self.models[word] = ('const', majority)
                continue
            clf = HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_leaf_nodes=8, min_samples_leaf=5,
                                                 l2_regularization=1.0, categorical_features=list(range(self.N_CATEGORICAL)),
                                                 early_stopping=False, random_state=0)
            cv = StratifiedKFold(n_splits=min(3, int(freq.min())), shuffle=True, random_state=0)
            acc = float(np.mean(cross_val_score(clf, x, y, cv=cv)))
            base = float(freq.max() / freq.sum())
            report.append((word, len(y), base, acc))
            if acc > base + 0.005:
                clf.fit(x, y)
                self.models[word] = ('hgb', clf)
            else:
                self.models[word] = ('const', majority)
        self.fit_report = report
        if verbose:
            used = sum(1 for m in self.models.values() if m[0] == 'hgb')
            print(f'{len(self.models)} ambiguous words modelled, {used} with a classifier')

    # ---- prediction -----------------------------------------------------------------------
    def is_modelled(self, word: str) -> bool:
        return word in self.models

    def choose(self, tokens: list[str], i: int, candidates: list[str], phones_of: Callable[[str], list[str]]) -> str:
        """Pick one of `candidates` (phone strings) for `tokens[i]`; falls back to the first candidate."""
        model = self.models.get(tokens[i])
        if model is None or len(candidates) == 1:
            return candidates[0]
        if model[0] == 'const':
            return model[1] if model[1] in candidates else candidates[0]
        clf = model[1]
        proba = clf.predict_proba(np.asarray([self.features(tokens, i, phones_of)], dtype=np.int32))[0]
        ranked = sorted(zip(proba, clf.classes_), key=lambda x: -x[0])
        for _, cls in ranked:
            if cls in candidates:
                return cls
        return candidates[0]

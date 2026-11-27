"""Grapheme-to-phoneme front end — lab 3.

Run as a script to score the module on the prepared data::

    python g2p.py

Two phone error rates are reported on the split carried in the data's own `set` column: PER over
whole `test` utterances — the front end as lab 5 will run it — and PER over `test` tokens that are in
no dictionary, which is the out-of-vocabulary model alone.

How a word is pronounced, in this order:

1. the training lexicon (`lexicon.csv`: pronunciations seen in the corpus, with counts);
2. the original MFA dictionary (`russian_mfa.dict`);
3. the model for words that are in no dictionary (`g2p_models.OOVModel`: stress model + letters-to-phones
   classifiers, fitted on the dictionaries) or, in the baseline, the hand-written rules of `g2p_rules`.

If a word has several pronunciations the most frequent one is the default; `g2p_context.VariantChooser`
replaces it by a choice that looks at the neighbouring words.
"""
import csv
import os
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import g2p_rules as R
from g2p_models import stress_letter

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS_PHONES_DATA = 'data/tokens_phones.csv'
MFA_DICT = os.path.join(HERE, 'data', 'russian_mfa.dict')
OOV_MODEL = os.path.join(HERE, 'models', 'oov_model.pkl.gz')
CHOOSER_MODEL = os.path.join(HERE, 'models', 'variant_chooser.pkl.gz')
PHONES_FILE = os.path.join(HERE, '..', '..', 'common', 'phones_mfa.txt')

SIL = '<SIL>'
LENGTH_MARK = 'ː'
STRESS_MARKS = ('+', '\u0301')          # `+` before a vowel or a combining acute after it
HYPHENS = '‐‑‒–—−'


def parse_token(token: str) -> list[tuple[str, int | None]]:
    """Normalize one input token and read an explicit stress mark off it.

    A token is lowercased and NFC-normalized; hyphen variants become `-` and a hyphenated token is
    split into its parts. The stress can be given as `+` in front of the stressed vowel
    (`з+амок`) or as a combining acute accent (U+0301) after it.

    Returns:
        One `(word, stress)` per hyphen-separated part: `word` keeps Cyrillic letters and
        apostrophes (the dictionary key), `stress` is the index of the stressed letter in
        `word` or None.
    """
    text = unicodedata.normalize('NFC', str(token)).lower().replace('’', "'")
    for h in HYPHENS:
        text = text.replace(h, '-')
    parts = []
    for part in text.split('-'):
        letters: list[str] = []
        stress = None
        mark_next = False
        for ch in part:
            if ch == '+':
                mark_next = True
            elif ch == '\u0301':
                if letters and stress is None:
                    stress = len(letters) - 1
            elif ch in R.LETTERS or ch == "'":
                if mark_next and ch in R.VOWEL_LETTERS and stress is None:
                    stress = len(letters)
                mark_next = False
                letters.append(ch)
        word = ''.join(letters)
        if word:
            parts.append((word, stress))
    return parts


class G2P:
    """Turns a sentence into a phone sequence.

    Input is one sentence as the token sequence lab 2 produces: words without punctuation,
    with ``"<SIL>"`` wherever the pause predictor placed a pause::

        ["я", "вышел", "из", "дома", <SIL>, "когда", "стемнело"]

    Punctuation is gone by this point on purpose — its only role in the pipeline is to
    predict pauses, which lab 2 has already done. Tokens may be in any case; a stressed vowel can be
    marked with `+` (see :func:`parse_token`).

    With no arguments but `lexicon` the module is complete: the MFA dictionary, the fitted OOV model and the
    contextual chooser are loaded from `data/` and `models/` (whatever is there). Every part can be
    replaced or switched off, which is how the report measures their contribution.

    Args:
        lexicon: `word -> Counter({phones: count})` as written by `prepare_lexicon.py`.
            Pronunciations are space-separated NFC strings. Keys are lowercase and carry no
            stress mark, so look words up accordingly.
        mfa_dict: `word -> [(probability, phones)]` from :func:`load_mfa_dict`; None switches the MFA
            dictionary off; the default ``'auto'`` reads `data/russian_mfa.dict`.
        oov: How to pronounce a word found in no dictionary: None (no output), ``'rules0'`` ..
            ``'rules3'`` (levels of `g2p_rules`), an `OOVModel`, or ``'auto'``: the saved model if there is
            one, else ``'rules3'``.
        chooser: A fitted `VariantChooser` that picks between several pronunciations from the
            context; None takes the most frequent one; ``'auto'`` loads the saved one if there is one.
    """

    def __init__(self, lexicon: dict[str, Counter] | None = None,
                 mfa_dict: dict[str, list[tuple[float, str]]] | str | None = 'auto',
                 oov='auto', chooser='auto') -> None:
        if isinstance(mfa_dict, str) and mfa_dict == 'auto':
            mfa_dict = load_mfa_dict() if os.path.isfile(MFA_DICT) else None
        if isinstance(oov, str) and oov == 'auto':
            oov = load_saved(OOV_MODEL, 'rules3')
        if isinstance(chooser, str) and chooser == 'auto':
            chooser = load_saved(CHOOSER_MODEL, None)
        self.lexicon = lexicon or {}
        self.mfa_dict = mfa_dict or {}
        self.oov = oov
        self.chooser = chooser
        self._cache: dict[tuple[str, int | None], list[str]] = {}
        if hasattr(oov, 'set_dictionaries'):
            oov.set_dictionaries(self._dictionary_phones)

    def _dictionary_phones(self, word: str) -> list[str] | None:
        """Best dictionary pronunciation of a word, or None: the OOV model reads the stress of related forms from it."""
        found = self.candidates(word)
        return found[0].split() if found else None

    # ---- single words --------------------------------------------------------------------
    def candidates(self, word: str) -> list[str]:
        """Dictionary pronunciations of `word`, the preferred first; empty if the word is unknown."""
        if word in self.lexicon:
            return [p for p, _ in self.lexicon[word].most_common()]
        if word in self.mfa_dict:
            return [p for _, p in self.mfa_dict[word]]
        return []

    def in_dictionary(self, word: str) -> bool:
        return word in self.lexicon or word in self.mfa_dict

    def pronounce_oov(self, word: str, stress: int | None = None) -> list[str]:
        """Pronunciation of a word from the OOV model or from the rules."""
        key = (word, stress)
        if key not in self._cache:
            letters = R.clean(word)
            if stress is not None:
                stress = len(R.clean(word[:stress]))          # index among the Cyrillic letters (apostrophes dropped)
            if self.oov is None or not letters:
                phones: list[str] = []
            elif isinstance(self.oov, str):
                phones = R.transcribe_word(letters, stress, int(self.oov[-1]))
            else:
                phones = self.oov.transcribe(letters, stress)
            self._cache[key] = phones
        return self._cache[key]

    def warm(self, tokens) -> None:
        """Pronounce the out-of-dictionary words among `tokens` in one batch (a speed-up for evaluation).

        Words with an explicit stress that no dictionary pronunciation has are included: the OOV model builds them
        around the requested stress.
        """
        if self.oov is None or isinstance(self.oov, str):
            return
        todo = set()
        for token in set(tokens):
            parts = parse_token(token) if token != SIL else []
            if len(parts) != 1:
                continue
            word, stress = parts[0]
            if not R.clean(word) or (word, stress) in self._cache:
                continue
            if stress is None:
                if not self.in_dictionary(word):
                    todo.add((word, None))
            elif not any(stress_letter(word, c.split()) == stress for c in self.candidates(word)):
                todo.add((word, stress))
        todo = sorted(todo, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
        cleaned = [(w, None if s is None else len(R.clean(w[:s]))) for w, s in todo]
        phones = self.oov.transcribe_many([w for w, _ in cleaned], [s for _, s in cleaned])
        for key, ph in zip(todo, phones):
            self._cache[key] = ph

    def pronounce_word(self, word: str, stress: int | None = None) -> list[str]:
        """Context-free pronunciation of one dictionary key."""
        if stress is None:
            found = self.candidates(word)
            if found:
                return found[0].split()
        return self.pronounce_oov(word, stress)

    # ---- sentences -----------------------------------------------------------------------
    def transcribe(self, tokens: list[str] | np.ndarray) -> list[list[str]]:
        """Transcribe each token of one sentence.

        Args:
            tokens: Tokens of one sentence in order, possibly including ``"<SIL>"``.

        Returns:
            One phone list per input token, so that `len(result) == len(tokens)`. A token
            may map to an empty list, but the outer length must be preserved: the
            evaluation scores in-vocabulary, dictionary and residual tokens separately and
            needs to know which output belongs to which token. ``"<SIL>"`` maps to
            ``["<SIL>"]``.

        """
        tokens = list(tokens)
        parsed = [None if t == SIL else parse_token(t) for t in tokens]
        keys = [SIL if p is None else '-'.join(w for w, _ in p) for p in parsed]
        out: list[list[str]] = []
        for i, (token, parts) in enumerate(zip(tokens, parsed)):
            if parts is None:
                out.append([SIL])
            elif not parts:
                out.append([])
            elif len(parts) == 1 or self.in_dictionary(keys[i]):
                out.append(self._single(keys, parts, i))
            else:
                out.append([p for word, stress in parts
                            for p in (self.pronounce_word(word, None) if stress is None else self.pronounce_with_stress(word, stress))])
        return out

    def _single(self, keys: list[str], parts: list[tuple[str, int | None]], i: int) -> list[str]:
        word, stress = parts[0] if len(parts) == 1 else (keys[i], None)
        found = self.candidates(word)
        if stress is not None:
            found = [p for p in found if stress_letter(word, p.split()) == stress]
            if not found:                       # no dictionary pronunciation has that stress: build one around it
                return self.pronounce_oov(word, stress)
        elif not found:
            return self.pronounce_oov(word, None)
        if self.chooser is not None and len(found) > 1 and self.chooser.is_modelled(word):
            return self.chooser.choose(keys, i, found, self._best).split()
        return found[0].split()

    def pronounce_with_stress(self, word: str, stress: int) -> list[str]:
        """Pronunciation of one word with the stressed vowel letter given (`stress` = its index in `word`).

        A dictionary pronunciation whose stressed vowel is the requested one wins (most frequent first);
        if the dictionaries have none, the OOV model builds the word around the requested stress.
        """
        for phones in self.candidates(word):
            if stress_letter(word, phones.split()) == stress:
                return phones.split()
        return self.pronounce_oov(word, stress)

    def _best(self, word: str) -> list[str]:
        """Context-free pronunciation used for the neighbours' phones."""
        if word == SIL:
            return [SIL]
        return self.pronounce_word(word, None)


# ---------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------
def phone_error_rate(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Corpus-level PER: summed edit distance over summed reference length.

    Summing before dividing weights every phone equally. Averaging per-sentence rates
    would instead weight every sentence equally and let short sentences dominate.

    Args:
        reference: Reference phone sequences.
        hypothesis: Predicted phone sequences, same length and order.

    Returns:
        Phone error rate, or `nan` when the reference is empty.
    """
    total_distance = 0
    total_length = 0
    for ref, hyp in zip(reference, hypothesis):
        total_length += len(ref)
        if not hyp:
            total_distance += len(ref)
            continue
        # a common prefix and suffix never change the edit distance: trim them (a big speed-up on
        # whole sentences, where almost all phones are right)
        start = 0
        while start < len(ref) and start < len(hyp) and ref[start] == hyp[start]:
            start += 1
        end = 0
        while end < len(ref) - start and end < len(hyp) - start and ref[-1 - end] == hyp[-1 - end]:
            end += 1
        ref, hyp = ref[start:len(ref) - end], hyp[start:len(hyp) - end]
        if not hyp:
            total_distance += len(ref)
            continue
        if not ref:
            total_distance += len(hyp)
            continue
        previous = list(range(len(hyp) + 1))
        for i, ref_phone in enumerate(ref, start=1):
            current = [i] + [0] * len(hyp)
            for j, hyp_phone in enumerate(hyp, start=1):
                current[j] = min(previous[j] + 1,
                                 current[j - 1] + 1,
                                 previous[j - 1] + (ref_phone != hyp_phone))
            previous = current
        total_distance += previous[-1]
    return total_distance / total_length if total_length else float('nan')


def word_accuracy(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Fraction of words whose whole phone sequence is exactly right."""
    if not reference:
        return float('nan')
    return float(np.mean([ref == hyp for ref, hyp in zip(reference, hypothesis)]))


def load_tokens(path: str = TOKENS_PHONES_DATA, verbose: bool = True) -> pd.DataFrame:
    """Read the training table and drop tokens with no reference transcription.

    Rows whose phones are empty or contain `spn` carry no reference and are removed.

    The `set` column comes with the file. It is a rule on the utterance id — test iff the
    number divides by 5

    Args:
        path: Path to the table written by `prepare_lexicon.py`.
        verbose: Print how many tokens were dropped.

    Returns:
        The table with `token` and `phones` as NFC strings, in sentence order.
    """
    tokens = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE, keep_default_na=False)
    tokens['token'] = tokens.token.astype(str).map(lambda s: unicodedata.normalize('NFC', s))
    tokens['phones'] = tokens.phones.fillna('').astype(str)

    unpronounced = (tokens.phones == '') | tokens.phones.str.split().map(lambda p: 'spn' in p)
    if verbose:
        print(f'dropping {unpronounced.sum()} tokens without a transcription '
              f'({unpronounced.mean():.2%})')
    return tokens[~unpronounced].reset_index(drop=True)


def build_lexicon(tokens: pd.DataFrame) -> dict[str, Counter]:
    """Count pronunciations per token type over the given rows (pauses are not words)."""
    lexicon: dict[str, Counter] = defaultdict(Counter)
    for token, phones in tokens[['token', 'phones']].values:
        if token != SIL:
            lexicon[token][phones] += 1
    return lexicon


def load_mfa_dict(path: str = MFA_DICT) -> dict[str, list[tuple[float, str]]]:
    """Read the original MFA dictionary into `word -> [(probability, phones)]`.

    The file mixes two line formats: `word<TAB>phones` and `word<TAB>p<TAB>p_sil<TAB>..<TAB>phones`
    with pronunciation probabilities. Phones are NFC with the length mark collapsed, as in the
    training table; variants that become equal after that are merged. Variants come best first
    (by probability, ties in file order); pronunciations that are `spn` are dropped.

    Args:
        path: Path to `russian_mfa.dict`.
    """
    variants: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            word = unicodedata.normalize('NFC', parts[0])
            phones = unicodedata.normalize('NFC', parts[-1]).replace(LENGTH_MARK, '')
            if 'spn' in phones.split():
                continue
            prob = float(parts[1]) if len(parts) == 6 else 1.0
            known = variants[word]
            known[phones] = max(prob, known.get(phones, 0.0))
    return {word: sorted(((p, ph) for ph, p in known.items()), key=lambda x: -x[0])
            for word, known in variants.items()}


def evaluate(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter],
             mfa_dict: dict | None = None) -> dict:
    """Score `g2p` on one fold of the table, sentence by sentence.

    The rows of `tokens` may include ``"<SIL>"`` tokens (the table with pauses): they are given to
    the model and removed from the scoring on both sides.

    Args:
        g2p: The module under test.
        tokens: One fold of the table from :func:`load_tokens`, in sentence order.
        lexicon: The training lexicon: tells in-lexicon tokens from the others.
        mfa_dict: The MFA dictionary. Tokens that are in neither dictionary form the *remainder*
            (the "OOV" of the assignment); without it the remainder is "not in the lexicon".

    Returns:
        `{'text': PER over whole sentences, 'lexicon': .., 'mfa': .., 'rest': ..,` and for each
        group `*_n` (tokens) and `*_acc` (exact word accuracy)`}`.
    """
    groups = {'lexicon': ([], []), 'mfa': ([], []), 'rest': ([], [])}
    if hasattr(g2p, 'warm'):
        g2p.warm(tokens.token.unique())
    sentence_ref: list[list[str]] = []
    sentence_hyp: list[list[str]] = []
    for _, sentence in tokens.groupby('id', sort=False):
        sentence = sentence.sort_values('token_id')
        words = sentence.token.values
        predicted = g2p.transcribe(words)
        if len(predicted) != len(words):
            raise ValueError(f'transcribe returned {len(predicted)} outputs '
                             f'for {len(words)} tokens')
        reference = [p.split() for p in sentence.phones.values]

        # <SIL> is stripped before scoring on both sides. Leaving it in would grade lab 2's
        # pause placement inside lab 3's metric. It is still passed through at runtime --
        # see :meth:`G2P.transcribe`.
        sentence_ref.append([p for word in reference for p in word if p != SIL])
        sentence_hyp.append([p for word in predicted for p in word if p != SIL])
        for token, ref, hyp in zip(words, reference, predicted):
            if token == SIL:
                continue
            if token in lexicon:
                key = 'lexicon'
            elif mfa_dict is not None and token in mfa_dict:
                key = 'mfa'
            else:
                key = 'rest'
            groups[key][0].append(ref)
            groups[key][1].append(hyp)

    result = {'text': phone_error_rate(sentence_ref, sentence_hyp)}
    for key, (ref, hyp) in groups.items():
        result[key] = phone_error_rate(ref, hyp)
        result[key + '_n'] = len(ref)
        result[key + '_acc'] = word_accuracy(ref, hyp)
    result['tokens'] = sum(result[k + '_n'] for k in groups)
    return result


def calc_metrics(g2p: G2P, tokens: pd.DataFrame, lexicon: dict[str, Counter],
                 mfa_dict: dict | None = None) -> dict:
    """Print running-text PER and PER and word accuracy per group of tokens; returns :func:`evaluate`."""
    res = evaluate(g2p, tokens, lexicon, mfa_dict)
    n = res['tokens']
    print(f'running text  PER {res["text"]:.4f}')
    print(f'in lexicon    PER {res["lexicon"]:.4f}  ({res["lexicon_n"]} words, {res["lexicon_n"] / n:.1%})')
    if mfa_dict is not None:
        print(f'MFA dict only PER {res["mfa"]:.4f}  ({res["mfa_n"]} words, {res["mfa_n"] / n:.1%})')
    print(f'remainder     PER {res["rest"]:.4f}  word accuracy {res["rest_acc"]:.4f}  '
          f'({res["rest_n"]} words, {res["rest_n"] / n:.1%})')
    return res


def load_saved(path: str, default):
    """Unpickle a gzip-compressed model saved by `train_g2p.py`, or return `default` if there is none."""
    import gzip
    import pickle
    if not os.path.isfile(path):
        return default
    with gzip.open(path, 'rb') as fh:
        return pickle.load(fh)


def load_g2p(lexicon: dict[str, Counter], mfa_dict: dict | None = None, context: bool = True) -> G2P:
    """The full module: lexicon, MFA dictionary, fitted OOV model and (if `context`) the variant chooser."""
    return G2P(lexicon, load_mfa_dict() if mfa_dict is None else mfa_dict, oov='auto',
               chooser='auto' if context else None)


def test_g2p() -> None:
    """Build the lexicon from `train` and score the module on both folds."""
    tokens = load_tokens()
    train = tokens[tokens.set == 'train']
    test = tokens[tokens.set == 'test']
    print(f'train {len(train)} tokens / test {len(test)} tokens')

    lexicon = build_lexicon(train)
    print(f'lexicon: {len(lexicon)} types, '
          f'{sum(1 for c in lexicon.values() if len(c) > 1)} with several pronunciations')

    mfa_dict = load_mfa_dict()
    g2p = load_g2p(lexicon, mfa_dict)

    print('\nCalculate metrics, training fold:')
    calc_metrics(g2p, train, lexicon, mfa_dict)

    print('\nCalculate metrics, testing fold:')
    calc_metrics(g2p, test, lexicon, mfa_dict)


if __name__ == '__main__':
    test_g2p()

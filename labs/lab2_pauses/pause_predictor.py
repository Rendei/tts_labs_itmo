"""Pause predictor — lab 2.

Run as a script to score the predictor on the prepared data::

    python pause_predictor.py

Precision, recall and F1 are computed for `is_pause_after`, and MAE for `pause_duration`
on true positives only. The last word of every utterance is excluded.

The predictor is a pair of gradient boosting models (pause / no pause, and pause length)
over hand-made features: punctuation after the word, position in the sentence, the word
itself and its neighbours, and their morphology from pymorphy3. A trained model is stored
in `models/pause_model.pkl`; if the file is missing it is trained from the prepared data.
"""
import csv
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tqdm
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error

PAUSE_PREDICTOR_DATA = 'data/RUSLAN_pause_metadata.csv'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'pause_model.pkl'

MIN_PAUSE = 0.06      # pauses shorter than this are treated as "no pause" when training
MIN_DURATION = 0.03   # shortest pause the predictor ever returns (the floor of the MFA markup)

_TOKEN_RE = re.compile(r"^([^\w]*)(\w[\w\u0301\-'’]*)?(.*)$", re.S)   # \u0301: combining stress mark, part of a word
_END_MARKS = '.!?…'
_BOUNDARY_MARKS = ',;:—–.!?…'
_DASHES = '—–-‐‑'
_QUOTES = '«»"\'“”„’'
_PARENS = '()[]<>'

_POS = ['NOUN', 'ADJF', 'ADJS', 'COMP', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND', 'NUMR', 'ADVB', 'NPRO',
        'PRED', 'PREP', 'CONJ', 'PRCL', 'INTJ']
_CASES = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2', 'acc2', 'loc2']


def split_token(token: str) -> tuple[str, str, str]:
    """Split a token into leading punctuation, the word itself and trailing punctuation."""
    lead, core, trail = _TOKEN_RE.match(str(token)).groups()
    return lead, (core or ''), trail


_WORD_RE = re.compile(r"[\w\u0301]+(?:[-'‐‑’][\w\u0301]+)*")   # a word; hyphens and apostrophes inside it do not split it


def tokenize_text(text: str) -> list[str]:
    """Split text into the tokens `PausePredictor` expects: words with their surrounding punctuation.

    A token is a word plus all punctuation up to the next word (whitespace around it is dropped, punctuation
    standing alone — a dash, a quote — is glued to the previous word); leading punctuation goes to the first
    word. Hyphenated and apostrophe words (``что-то``, ``д'Артаньян``) stay whole. The same layout as
    the `label_raw` column of the training table::

        tokenize_text('Хватает, — согласился он.')  ->  ['Хватает, —', 'согласился', 'он.']

    Non-breaking hyphens and ’ are replaced with ``-`` and ``'`` first, as in `prepare_training_data.py`.
    """
    text = str(text).replace('‐', '-').replace('‑', '-').replace('’', "'")
    words = list(_WORD_RE.finditer(text))
    tokens = []
    for k, m in enumerate(words):
        stop = words[k + 1].start() if k + 1 < len(words) else len(text)
        token = text[:m.end()].strip() if k == 0 else m.group(0)
        tokens.append(token + text[m.end():stop].strip())
    return tokens


def split_on_whitespace(text: str) -> list[str]:
    """The naive tokenizer: split by whitespace (kept to compare with :func:`tokenize_text`)."""
    return str(text).split()


class TextFeaturizer:
    """Turns a sentence (list of `label_raw` tokens) into one feature row per token."""

    GROUPS = {
        'punct': ['trail_class', 'end', 'comma', 'semicolon', 'dash', 'quote', 'paren', 'ellipsis', 'lead_quote',
                  'lead_dash', 'trail_len', 'prev_trail_class', 'next_lead_quote', 'next_lead_dash', 'next_trail_class'],
        'position': ['idx', 'n_words', 'rel_pos', 'n_remaining', 'idx_in_sent', 'sent_len', 'to_sent_end',
                     'since_boundary', 'until_boundary', 'n_commas_before', 'is_last', 'next_is_last',
                     'cur_len', 'next_len', 'prev_len', 'cur_cap', 'next_cap', 'cur_hyphen'],
        'words': ['cur_w', 'next_w', 'prev_w', 'next2_w'],
        'morph': ['cur_pos', 'next_pos', 'prev_pos', 'next2_pos', 'cur_case', 'next_case',
                  'cur_grnd', 'cur_prtf', 'cur_infn', 'cur_verb', 'next_conj', 'next_prep', 'next_prcl',
                  'next_grnd', 'next_prtf', 'next_infn', 'next_verb', 'cur_prep', 'cur_conj'],
    }
    CATEGORICAL = {'trail_class', 'prev_trail_class', 'next_trail_class', 'cur_w', 'next_w', 'prev_w', 'next2_w',
                   'cur_pos', 'next_pos', 'prev_pos', 'next2_pos', 'cur_case', 'next_case'}

    def __init__(self, top_words: int = 200, top_trails: int = 40, groups: tuple[str, ...] | None = None):
        """Args: vocabulary sizes (categorical features must have < 256 values) and feature groups to use."""
        self.top_words = top_words
        self.top_trails = top_trails
        self.groups = tuple(groups) if groups else tuple(self.GROUPS)
        self.word_ids: dict[str, int] = {}
        self.trail_ids: dict[str, int] = {}
        self._morph_cache: dict[str, tuple] = {}
        self._analyzer = None

    @property
    def feature_names(self) -> list[str]:
        """Names of the produced columns, in order."""
        return [name for g in self.groups for name in self.GROUPS[g]]

    @property
    def categorical_mask(self) -> list[bool]:
        """Which columns are categorical — for HistGradientBoosting."""
        return [name in self.CATEGORICAL for name in self.feature_names]

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_analyzer'] = None     # pymorphy3 analyzer is rebuilt lazily after loading
        state['_morph_cache'] = {}
        return state

    def fit(self, sentences: list[list[str]]) -> 'TextFeaturizer':
        """Learn the most frequent words and trailing-punctuation strings from training sentences."""
        words, trails = Counter(), Counter()
        for sent in sentences:
            for tok in sent:
                _, core, trail = split_token(tok)
                words[core.lower()] += 1
                trails[trail.strip()] += 1
        self.word_ids = {w: i + 1 for i, (w, _) in enumerate(words.most_common(self.top_words)) if w}
        self.trail_ids = {t: i + 1 for i, (t, _) in enumerate(trails.most_common(self.top_trails))}
        return self

    def _word_id(self, core: str) -> int:
        return self.word_ids.get(core.lower(), 0)

    def _morph(self, core: str) -> tuple:
        """(pos id, case id, is_grnd, is_prtf, is_infn, is_verb, is_conj, is_prep, is_prcl) of the most likely parse."""
        key = core.lower()
        if key not in self._morph_cache:
            if self._analyzer is None:
                import pymorphy3
                self._analyzer = pymorphy3.MorphAnalyzer()
            if not key or not re.search(r'[а-яё]', key):
                res = (0, 0, 0, 0, 0, 0, 0, 0, 0)
            else:
                tag = self._analyzer.parse(key)[0].tag
                pos = tag.POS or ''
                case = tag.case or ''
                res = (_POS.index(pos) + 1 if pos in _POS else 0, _CASES.index(case) + 1 if case in _CASES else 0,
                       int(pos == 'GRND'), int(pos in ('PRTF', 'PRTS')), int(pos == 'INFN'),
                       int(pos in ('VERB', 'INFN')), int(pos == 'CONJ'), int(pos == 'PREP'), int(pos == 'PRCL'))
            self._morph_cache[key] = res
        return self._morph_cache[key]

    def transform_sentence(self, tokens) -> np.ndarray:
        """Feature matrix (len(tokens) x n_features) of one sentence."""
        n = len(tokens)
        parts = [split_token(t) for t in tokens]
        trails = [p[2] for p in parts]
        cores = [p[1] for p in parts]
        morph = [self._morph(c) for c in cores]
        empty_morph = (0,) * 9
        is_end = [any(c in t for c in _END_MARKS) for t in trails]
        is_boundary = [any(c in t for c in _BOUNDARY_MARKS) for t in trails]

        # sentence boundaries: a token with an end mark closes a sentence
        sent_start, sent_end = [0] * n, [0] * n
        start = 0
        for i in range(n):
            sent_start[i] = start
            if is_end[i]:
                for j in range(start, i + 1):
                    sent_end[j] = i
                start = i + 1
        for j in range(start, n):
            sent_end[j] = n - 1

        rows = []
        last_boundary, commas = -1, 0
        next_boundary = [0] * n
        nb = n
        for i in range(n - 1, -1, -1):
            if is_boundary[i]:
                nb = i
            next_boundary[i] = nb

        def get(lst, i, default):
            return lst[i] if 0 <= i < n else default

        for i in range(n):
            lead, core, trail = parts[i]
            tr = trail.strip()
            nxt = parts[i + 1] if i + 1 < n else ('', '', '')
            m_cur, m_next = morph[i], get(morph, i + 1, empty_morph)
            m_prev, m_next2 = get(morph, i - 1, empty_morph), get(morph, i + 2, empty_morph)
            row = {
                'trail_class': self.trail_ids.get(tr, 0),
                'end': int(is_end[i]),
                'comma': int(',' in trail),
                'semicolon': int(';' in trail or ':' in trail),
                'dash': int(any(c in trail for c in '—–')),
                'quote': int(any(c in trail for c in _QUOTES)),
                'paren': int(any(c in trail for c in _PARENS)),
                'ellipsis': int('…' in trail or '..' in trail),
                'lead_quote': int(any(c in lead for c in _QUOTES)),
                'lead_dash': int(any(c in lead for c in _DASHES)),
                'trail_len': len(tr),
                'prev_trail_class': self.trail_ids.get(trails[i - 1].strip(), 0) if i > 0 else -1,
                'next_lead_quote': int(any(c in nxt[0] for c in _QUOTES)),
                'next_lead_dash': int(any(c in nxt[0] for c in _DASHES)),
                'next_trail_class': self.trail_ids.get(nxt[2].strip(), 0) if i + 1 < n else -1,
                'idx': i,
                'n_words': n,
                'rel_pos': i / max(n - 1, 1),
                'n_remaining': n - 1 - i,
                'idx_in_sent': i - sent_start[i],
                'sent_len': sent_end[i] - sent_start[i] + 1,
                'to_sent_end': sent_end[i] - i,
                'since_boundary': i - last_boundary,
                'until_boundary': next_boundary[i] - i,
                'n_commas_before': commas,
                'is_last': int(i == n - 1),
                'next_is_last': int(i == n - 2),
                'cur_len': len(core),
                'next_len': len(nxt[1]),
                'prev_len': len(parts[i - 1][1]) if i > 0 else 0,
                'cur_cap': int(core[:1].isupper()),
                'next_cap': int(nxt[1][:1].isupper()),
                'cur_hyphen': int('-' in core),
                'cur_w': self._word_id(core),
                'next_w': self._word_id(nxt[1]) if i + 1 < n else -1,
                'prev_w': self._word_id(parts[i - 1][1]) if i > 0 else -1,
                'next2_w': self._word_id(parts[i + 2][1]) if i + 2 < n else -1,
                'cur_pos': m_cur[0], 'next_pos': m_next[0] if i + 1 < n else -1,
                'prev_pos': m_prev[0] if i > 0 else -1, 'next2_pos': m_next2[0] if i + 2 < n else -1,
                'cur_case': m_cur[1], 'next_case': m_next[1] if i + 1 < n else -1,
                'cur_grnd': m_cur[2], 'cur_prtf': m_cur[3], 'cur_infn': m_cur[4], 'cur_verb': m_cur[5],
                'next_conj': m_next[6], 'next_prep': m_next[7], 'next_prcl': m_next[8],
                'next_grnd': m_next[2], 'next_prtf': m_next[3], 'next_infn': m_next[4], 'next_verb': m_next[5],
                'cur_prep': m_cur[7], 'cur_conj': m_cur[6],
            }
            rows.append([row[name] for name in self.feature_names])
            if is_boundary[i]:
                last_boundary = i
            if ',' in trail:
                commas += 1
        return np.array(rows, dtype=np.float32).reshape(n, len(self.feature_names))

    def transform(self, sentences: list[list[str]], progress: bool = False) -> np.ndarray:
        """Feature matrix of many sentences, rows in sentence order."""
        it = tqdm.tqdm(sentences) if progress else sentences
        mats = [self.transform_sentence(s) for s in it if len(s)]
        return np.concatenate(mats) if mats else np.zeros((0, len(self.feature_names)), np.float32)


def sentences_from_frame(df: pd.DataFrame) -> tuple[list[str], list[list[str]], np.ndarray]:
    """Group the word table by utterance.

    Returns:
        Utterance ids, their `label_raw` token lists, and for every row of `df` the index of its utterance.
        Rows of an utterance must be contiguous and in order — as written by `prepare_training_data.py`.
    """
    ids = df.id.to_numpy(dtype=object)
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    tokens = df.label_raw.astype(str).to_numpy(dtype=object)
    bounds = np.r_[starts, len(df)]
    sentence_ids = [ids[s] for s in starts]
    sentences = [list(tokens[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]
    row_sentence = np.repeat(np.arange(len(starts)), np.diff(bounds))
    return sentence_ids, sentences, row_sentence


class PausePredictor():
    """Predicts where pauses fall in a sentence and how long they are.

    Input is one sentence as a sequence of `label_raw` tokens — words with their
    trailing punctuation, in order::

        ["Я", "вышел", "из", "дома,", "когда", "стемнело."]
    """

    def __init__(self, model_path: str | Path | None = MODEL_PATH, train_if_missing: bool = True):
        """Load the trained model from `model_path`; train and save one if it does not exist yet."""
        self.featurizer: TextFeaturizer | None = None
        self.clf = None
        self.reg = None
        self.threshold = 0.5
        if model_path is not None and Path(model_path).exists():
            self.load(model_path)
        elif train_if_missing and model_path is not None:
            print(f'{model_path} not found: training a model from {PAUSE_PREDICTOR_DATA}')
            self.fit_from_csv()
            self.save(model_path)

    # ------------------------------------------------------------------ training
    def fit(self, sentences: list[list[str]], is_pause: np.ndarray, durations: np.ndarray,
            duration_mask: np.ndarray | None = None, groups: tuple[str, ...] | None = None,
            threshold: float = 0.5, params: dict | None = None) -> 'PausePredictor':
        """Train both models.

        Args:
            sentences: Token lists of the training sentences.
            is_pause: 0/1 target for every token (rows of all sentences concatenated).
            durations: Pause length in seconds for every token.
            duration_mask: Tokens the duration model is trained on; by default the tokens with `is_pause == 1`.
            groups: Feature groups of :class:`TextFeaturizer` to use (all by default).
            threshold: Probability above which a pause is placed.
            params: Overrides for the boosting hyperparameters.
        """
        self.featurizer = TextFeaturizer(groups=groups).fit(sentences)
        x = self.featurizer.transform(sentences)
        mask = self.featurizer.categorical_mask
        p = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0,
                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=0)
        p.update(params or {})
        self.clf = HistGradientBoostingClassifier(categorical_features=mask, **p).fit(x, is_pause)
        sel = (is_pause == 1) if duration_mask is None else duration_mask
        self.reg = HistGradientBoostingRegressor(loss='absolute_error', categorical_features=mask, **p).fit(
            x[sel], durations[sel])
        self.threshold = threshold
        return self

    def fit_from_csv(self, path: str = PAUSE_PREDICTOR_DATA) -> 'PausePredictor':
        """Train on the `train` part of the prepared table.

        Pauses shorter than MIN_PAUSE are treated as "no pause"; the last word of every sentence is kept
        with label 0 so the model sees sentence endings but is never scored on them.
        """
        df = pd.read_csv(path, sep='|', quoting=csv.QUOTE_NONE)
        df = df[df.set == 'train']
        _, sentences, _ = sentences_from_frame(df)
        y = ((df.pause_duration.values >= MIN_PAUSE) & (df.is_last_word.values == 0)).astype(int)
        return self.fit(sentences, y, df.pause_duration.values, threshold=0.4)

    def save(self, path: str | Path = MODEL_PATH) -> None:
        """Pickle the trained models."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'featurizer': self.featurizer, 'clf': self.clf, 'reg': self.reg,
                         'threshold': self.threshold}, f)

    def load(self, path: str | Path = MODEL_PATH) -> None:
        """Load models saved by :meth:`save`."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.featurizer, self.clf, self.reg, self.threshold = (
            state['featurizer'], state['clf'], state['reg'], state['threshold'])

    # ------------------------------------------------------------------ inference
    def predict_proba_many(self, sentences: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Pause probability and predicted pause length for every token of every sentence (concatenated)."""
        x = self.featurizer.transform(sentences)
        if len(x) == 0:
            return np.zeros(0), np.zeros(0)
        return self.clf.predict_proba(x)[:, 1], self.reg.predict(x)

    def predict_many(self, sentences: list[list[str]], threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Batch version of :meth:`predict` — same rows as `predict_proba_many`."""
        proba, dur = self.predict_proba_many(sentences)
        thr = self.threshold if threshold is None else threshold
        is_pause = (proba >= thr).astype(int)
        pause_duration = np.where(is_pause == 1, np.maximum(dur, MIN_DURATION), 0.0)
        return is_pause, pause_duration

    def predict(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decide for each token whether a pause follows it.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            `is_pause` (int, 1 if a pause follows the token) and `pause_duration`
            (float, seconds, 0.0 where there is no pause), both of length `len(tokens)`.

        Note:
            Wherever `is_pause` is 1 the duration must be positive:
            :meth:`predict_durations` relies on it.
        """
        if len(tokens) == 0:
            return np.zeros(0, int), np.zeros(0, float)
        return self.predict_many([[str(t) for t in tokens]])

    def predict_text(self, text: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Tokenize a raw (normalized) sentence with :func:`tokenize_text` and predict its pauses.

        Returns:
            The tokens, and `is_pause` and `pause_duration` as in :meth:`predict`.
        """
        tokens = tokenize_text(text)
        is_pause, durations = self.predict(tokens)
        return tokens, is_pause, durations

    def predict_durations(self, tokens: list[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Insert predicted pauses into the token sequence.

        This is the form the acoustic model consumes in labs 4 and 5.

        Args:
            tokens: Tokens of one sentence.

        Returns:
            Tokens with ``"<SIL>"`` after every predicted pause, and one duration per
            output token: seconds for ``"<SIL>"``, ``-1.0`` for words (left to the
            acoustic model).
        """
        def expand_is_pause(token: str, is_pause: int) -> list[str]:
            if bool(is_pause):
                return [token, '<SIL>']
            return [token]

        def expand_durations(pause_duration: float) -> list[float]:
            if pause_duration>0.:
                return [-1., pause_duration]
            return [-1.]

        is_pause, durations = self.predict(tokens)

        tokens_w_pauses = np.concatenate([expand_is_pause(a, b) for a, b in zip(tokens, is_pause)])
        durations_w_pauses = np.concatenate([expand_durations(dur) for dur in durations]).astype(np.float32)

        return tokens_w_pauses, durations_w_pauses

def calc_metrics(df: pd.DataFrame) -> None:
    """Print precision, recall and F1 for pause placement, and MAE for pause duration.

    MAE counts only rows where both the reference and the prediction have a pause.
    """
    rec =recall_score(df.is_pause_after, df.is_pause_hat)
    prc = precision_score(df.is_pause_after, df.is_pause_hat)
    f1 = f1_score(df.is_pause_after, df.is_pause_hat)

    mae = mean_absolute_error(df[(df.is_pause_after==1) & (df.is_pause_hat==1)].pause_duration, df[(df.is_pause_after==1) & (df.is_pause_hat==1)].pause_duration_hat)
    print(f'PRC: {prc}, REC: {rec}, F1: {f1}; MAE: {mae};')

def test_pause_predictor() -> None:
    """Run the predictor on every sentence and print train and test metrics.

    Expects the layout written by `prepare_training_data.py`: rows grouped by utterance
    in order, each utterance ending with its `is_last_word` row.
    """
    pause_df =pd.read_csv(PAUSE_PREDICTOR_DATA, sep='|', quoting=csv.QUOTE_NONE)

    pp = PausePredictor()

    lens = {i:l for i, l in pause_df.groupby('id').count().reset_index(drop=False)[['id', 'label']].values}

    is_pause_after_hat = []
    pause_duration_hat = []

    idx = 0
    for i, is_last in tqdm.tqdm(pause_df[['id', 'is_last_word']].values):
        if not is_last:
            continue
        sentence = pause_df.iloc[idx:idx+lens[i]]
        idx += lens[i]

        is_pause_hat, pause_dur_hat = pp.predict(sentence.label_raw.values)
        is_pause_after_hat += list(is_pause_hat)
        pause_duration_hat += list(pause_dur_hat)

    pause_df['is_pause_hat'] = is_pause_after_hat
    pause_df['pause_duration_hat'] = pause_duration_hat

    print('Calculate metrics, traning fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set=='train') & (pause_df.is_last_word==0)])

    print('\nCalculate metrics, testing fold; Exclude last tokens in every sentence!')
    calc_metrics(pause_df[(pause_df.set=='test') & (pause_df.is_last_word==0)])

if __name__=='__main__':
    test_pause_predictor()

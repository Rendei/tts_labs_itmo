"""What does not help the letter model of the OOV stage — lab 3, report section 5.3.

Phonetisaurus beats our OOV model on the remainder of `test` (PER 0.033 against 0.0585). Three natural
explanations are tested here by retraining the per-letter classifiers (`g2p_models.LetterModel`) with

* the same window of +-3 letters and the same small sample of training rows (the model of the report),
* more rows and trees (200 000 rows per letter, 60 trees),
* a window of +-6 letters,

and scoring them on the remainder with the stress of the saved stress model and with the stress read from the
reference. Run from `labs/lab3_g2p` (about 10 minutes)::

    python experiments/letter_window.py
"""
import sys, re, time, random, zlib
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
import g2p as G, g2p_models as M, g2p_rules as R, train_g2p as T

def feats(word, stress, W):
    n = len(word); vowels = R.vowel_positions(word)
    stress_no = vowels.index(stress) if stress in vowels else None
    offsets = {i: k - stress_no for k, i in enumerate(vowels)} if stress_no is not None else {}
    codes = [0] * W + [M.LETTER_CODE[c] for c in word] + [0] * W
    off = lambda i: min(max(offsets[i], -5), 5) + 6 if i in offsets else 0
    nv = [None] * (n + 1); last = None
    for i in range(n - 1, -1, -1):
        if i in offsets: last = i
        nv[i] = last
    rows = np.zeros((n, 2 * W + 1 + 8), dtype=np.int16); prev = None
    for i in range(n):
        rows[i, :2 * W + 1] = codes[i:i + 2 * W + 1]
        b = 2 * W + 1
        rows[i, b] = off(i)
        j = nv[i + 1] if i + 1 < n else None
        rows[i, b + 1] = off(j) if j is not None else 0
        rows[i, b + 2] = off(prev) if prev is not None else 0
        rows[i, b + 3] = min(i, 5); rows[i, b + 4] = min(n - 1 - i, 5); rows[i, b + 5] = min(len(vowels), 8)
        rows[i, b + 6] = int(stress_no is not None); rows[i, b + 7] = min(sum(1 for c in word[i + 1:] if c in R.VOWEL_LETTERS), 6)
        if i in offsets: prev = i
    return rows

def fit_letter(letter, X, y, n_cat, iters):
    y = np.asarray(y); cl, fr = np.unique(y, return_counts=True)
    keep = np.isin(y, cl[fr >= 20]); X, y = X[keep], y[keep]
    if len(set(y)) == 1: return letter, str(y[0])
    clf = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.15, l2_regularization=1.0, max_leaf_nodes=24, min_samples_leaf=20,
                                          categorical_features=list(range(n_cat)), early_stopping=False, random_state=0)
    return letter, clf.fit(X, y)

def predict_many(models, W, words, stresses):
    by = defaultdict(lambda: ([], [])); 
    for wi, (w, s) in enumerate(zip(words, stresses)):
        X = feats(w, s, W)
        for i, c in enumerate(w): by[c][0].append(X[i]); by[c][1].append((wi, i))
    out = [[[] for _ in w] for w in words]
    for c, (rows, where) in by.items():
        m = models.get(c)
        if m is None: preds = [[] for _ in rows]
        elif isinstance(m, str): preds = [m.split()] * len(rows)
        else: preds = [p.split() for p in m.predict(np.asarray(rows, dtype=np.int16))]
        for (wi, i), p in zip(where, preds): out[wi][i] = p
    return [[p for ch in o for p in ch] for o in out]

if __name__ == '__main__':
    tokens = G.load_tokens(verbose=False)
    lex = G.build_lexicon(tokens[tokens.set == 'train']); mfa = G.load_mfa_dict()
    test = tokens[tokens.set == 'test']
    rest = test[~test.token.isin(lex) & ~test.token.isin(mfa)]
    rest = rest[rest.token.str.fullmatch('[а-яё]{2,}')]
    words_all = T.training_pronunciations(lex, mfa)
    pairs = M.clean_pairs(words_all, T.target_phones())
    al = M.Aligner(); al.fit(pairs, rounds=3, sample=40000)
    prep = M.prepare_training(pairs, al)
    train = [p for p in prep if not T.is_holdout(p[0])]
    print('train words', len(train))
    # stress predictions of the existing model (same for every variant)
    oov = M.OOVModel.load('models/oov_model.pkl.gz'); oov.use_siblings = True
    g = G.G2P(lex, mfa, oov=oov, chooser=None)
    ws = sorted(set(rest.token))
    pred_st = [oov.predict_stress(w) for w in ws]
    ref_by = {}
    for w, p in zip(rest.token, rest.phones): ref_by[w] = p.split()
    true_st = {w: M.stress_letter(w, ref_by[w]) for w in ws}
    def score(models, W, tag):
        hyp = dict(zip(ws, predict_many(models, W, ws, pred_st)))
        per = G.phone_error_rate([ref_by[w] for w in rest.token], [hyp[w] for w in rest.token])
        kn = [w for w in ws if true_st[w] is not None]
        hyp2 = dict(zip(kn, predict_many(models, W, kn, [true_st[w] for w in kn])))
        per2 = G.phone_error_rate([ref_by[w] for w in kn], [hyp2[w] for w in kn])
        print(f'{tag}: remainder PER (predicted stress) {per:.4f} | PER with reference stress on {len(kn)} words {per2:.4f}', flush=True)
    rng = random.Random(0)
    sub = train
    for W, cap, iters in ((3, 60000, 30), (3, 200000, 60), (6, 200000, 60)):
        t0 = time.time(); buckets = defaultdict(lambda: ([], []))
        for word, spans, phones, st in sub:
            X = feats(word, st, W)
            for i, (a, b) in enumerate(spans):
                buckets[word[i]][0].append(X[i]); buckets[word[i]][1].append(' '.join(phones[a:b]))
        jobs = []
        for letter, (rows, labels) in buckets.items():
            X = np.asarray(rows, dtype=np.int16)
            if len(labels) > cap:
                idx = np.random.RandomState(0).choice(len(labels), cap, replace=False); X = X[idx]; labels = [labels[j] for j in idx]
            jobs.append((letter, X, labels))
        n_cat = 2 * W + 1 + 3
        res = Parallel(n_jobs=10)(delayed(fit_letter)(l, X, y, n_cat, iters) for l, X, y in jobs)
        models = dict(res)
        score(models, W, f'window ±{W}, rows/letter {cap}, iters {iters} ({time.time() - t0:.0f}s)')

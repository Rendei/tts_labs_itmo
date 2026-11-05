"""Joint choice of the stressed vowel by the likelihood of the transcription — lab 3, report section 5.3.

The stress model proposes the three most probable stressed vowels; for each one the letter classifiers score the
transcription they would produce (sum of the log of the winning class probabilities); the stress is chosen by
`log P(stress) + lambda * log-likelihood`. It does not change the PER of the remainder, so the bottleneck is not the
way the two stages are combined. Run from `labs/lab3_g2p`::

    python experiments/joint_decoding.py
"""
import sys, re, copy, time
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from collections import defaultdict
import g2p as G, g2p_models as M, g2p_rules as R
tokens = G.load_tokens(verbose=False)
lex = G.build_lexicon(tokens[tokens.set == 'train']); mfa = G.load_mfa_dict()
test = tokens[tokens.set == 'test']
rest = test[~test.token.isin(lex) & ~test.token.isin(mfa)]
rest = rest[rest.token.str.fullmatch('[а-яё]{2,}')]
oov = M.OOVModel.load('models/oov_model.pkl.gz'); oov.use_siblings = True
g = G.G2P(lex, mfa, oov=oov, chooser=None)          # sets dictionaries for the sibling feature
words = sorted(set(rest.token))
extras = {w: oov.siblings.extra(w) for w in words}
# candidate stresses by the stress model
cand = {}
for w in words:
    vs = R.vowel_positions(w)
    if len(vs) <= 1 or 'ё' in w:
        cand[w] = [(vs[0] if vs else None, 0.0)] if 'ё' not in w else [(w.index('ё'), 0.0)]
        continue
    sc = np.array(oov.stress.scores(w, extras[w]))
    p = 1 / (1 + np.exp(-sc)); p = p / p.sum()
    order = np.argsort(-sc)[:3]
    cand[w] = [(vs[k], float(np.log(p[k] + 1e-9))) for k in order]
# letter-model log-likelihood of the transcription given each candidate stress
pairs = [(w, s, lp) for w in words for s, lp in cand[w]]
rows_by_letter = defaultdict(lambda: ([], []))
for pi, (w, s, lp) in enumerate(pairs):
    X = M.letter_features(w, s)
    for i, c in enumerate(w):
        rows_by_letter[c][0].append(X[i]); rows_by_letter[c][1].append(pi)
ll = np.zeros(len(pairs))
for c, (rows, idx) in rows_by_letter.items():
    m = oov.letters.models.get(c)
    if m is None or isinstance(m, str): continue
    pr = m.predict_proba(np.asarray(rows, dtype=np.int16)).max(axis=1)
    np.add.at(ll, idx, np.log(pr + 1e-9))
ref_of = dict(zip(rest.token, rest.phones))
def evaluate(lam):
    best = {}
    for pi, (w, s, lp) in enumerate(pairs):
        sc = lp + lam * ll[pi]
        if w not in best or sc > best[w][0]: best[w] = (sc, s)
    ws = list(best); hyp = oov.letters.transcribe_many(ws, [best[w][1] for w in ws])
    hmap = dict(zip(ws, hyp))
    ref = [r.split() for r in rest.phones]; h = [hmap[t] for t in rest.token]
    return G.phone_error_rate(ref, h), np.mean([a == b for a, b in zip(ref, h)])
for lam in (0, 0.25, 0.5, 1, 2, 4):
    per, acc = evaluate(lam)
    print(f'lambda {lam}: remainder PER {per:.4f} exact {acc:.3f}')

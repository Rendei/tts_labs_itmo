"""Fit the statistical parts of the G2P module — lab 3.

Two models are fitted and saved to `models/`::

    python train_g2p.py oov        # OOV model: stress + letters to phones   -> models/oov_model.pkl
    python train_g2p.py chooser    # contextual choice between pronunciations -> models/variant_chooser.pkl

Both use the training fold of the corpus only. The OOV model additionally learns from the MFA
dictionary, which is independent of the corpus split; 5 % of its words are held out (by a hash of the
word) and never seen in training, to measure the model on words it did not learn.
"""
import json
import os
import pickle
import random
import sys
import time
import zlib

import numpy as np

import g2p as G
import g2p_models as M
import g2p_rules as R

MODELS_DIR = 'models'
HOLDOUT_EVERY = 20


def is_holdout(word: str) -> bool:
    return zlib.crc32(word.encode('utf-8')) % HOLDOUT_EVERY == 0


def target_phones() -> set[str]:
    with open(G.PHONES_FILE, encoding='utf-8') as fh:
        return set(fh.read().split())


def training_pronunciations(lexicon: dict, mfa_dict: dict) -> dict[str, list[tuple[float, str]]]:
    """Words and their best pronunciation: the MFA dictionary plus the training lexicon.

    Words that only the corpus lexicon has (its pronunciations came from `mfa g2p`) are added; for the
    words in both the MFA variants win, since they do not depend on one aligner's choice.
    """
    words = dict(mfa_dict)
    for word, counter in lexicon.items():
        if word not in words:
            words[word] = [(1.0, counter.most_common(1)[0][0])]
    return words


def train_oov(lexicon: dict, mfa_dict: dict, n_jobs: int = 10, letter_words: int = 250000,
              rows_per_letter: int = 60000, log=print) -> M.OOVModel:
    """Fit the stress models (without and with morphological evidence) and the letter model."""
    t0 = time.time()
    target = target_phones()
    words = training_pronunciations(lexicon, mfa_dict)
    pairs = M.clean_pairs(words, target)
    aligner = M.Aligner()
    aligner.fit(pairs, rounds=3, sample=40000)
    prepared = M.prepare_training(pairs, aligner)
    train = [p for p in prepared if not is_holdout(p[0])]
    held = [p for p in prepared if is_holdout(p[0])]
    log(f'{len(pairs)} words, {len(prepared)} aligned with a known stress; train {len(train)}, held out {len(held)} '
        f'({time.time() - t0:.0f} s)')

    # stage 1a: stress from the spelling only
    stress_plain = M.StressModel()
    stress_plain.fit([p[0] for p in train], [p[3] for p in train])
    log(f'stress model fitted ({time.time() - t0:.0f} s)')

    # stage 1b: + the stress of the other forms of the lexeme (the held-out words are invisible to it)
    visible = {p[0]: p[2] for p in train}
    siblings = M.SiblingStress(visible.get)
    extras = [siblings.extra(p[0]) for p in train]
    stress_morph = M.StressModel()
    stress_morph.fit([p[0] for p in train], [p[3] for p in train], extras)
    log(f'stress model with morphology fitted ({time.time() - t0:.0f} s)')

    # stage 2: letters to phones, given the stress read off the dictionary
    rng = random.Random(0)
    buckets: dict[str, tuple[list, list]] = {}
    for word, spans, phones, st in rng.sample(train, min(letter_words, len(train))):
        x = M.letter_features(word, st)
        for i, (a, b) in enumerate(spans):
            rows, labels = buckets.setdefault(word[i], ([], []))
            rows.append(x[i])
            labels.append(' '.join(phones[a:b]))
    data = {}
    for letter, (rows, labels) in buckets.items():
        x = np.asarray(rows, dtype=np.int16)
        if len(labels) > rows_per_letter:
            idx = np.random.RandomState(0).choice(len(labels), rows_per_letter, replace=False)
            x, labels = x[idx], [labels[j] for j in idx]
        data[letter] = (x, labels)
    letters = M.LetterModel()
    letters.fit(data, max_iter=30, n_jobs=n_jobs)
    log(f'letter model fitted ({time.time() - t0:.0f} s)')

    model = M.OOVModel(stress_morph, letters, use_siblings=True, stress_plain=stress_plain)
    held_visible = dict(visible)
    for p in held:
        held_visible.pop(p[0], None)
    model.meta = evaluate_heldout(model, held, visible, log)
    model.meta['n_pairs'], model.meta['n_train'], model.meta['n_heldout'] = len(pairs), len(train), len(held)
    return model


def evaluate_heldout(model: M.OOVModel, held: list, visible: dict, log=print, n: int = 8000) -> dict:
    """Accuracy of the stress models and PER of the full model on held-out dictionary words."""
    sample = random.Random(1).sample(held, min(n, len(held)))
    words = [p[0] for p in sample]
    ref = [p[2] for p in sample]
    truth = [p[3] for p in sample]
    meta = {}
    meta['stress_guess_acc'] = float(np.mean([R.guess_stress(w) == s for w, s in zip(words, truth)]))
    plain = model.stress_plain or model.stress
    meta['stress_plain_acc'] = float(np.mean([plain.predict(w) == s for w, s in zip(words, truth)]))
    sib = M.SiblingStress(visible.get)
    meta['stress_morph_acc'] = float(np.mean([model.stress.predict(w, sib.extra(w)) == s for w, s in zip(words, truth)]))
    hyp_true = model.letters.transcribe_many(words, truth)
    meta['per_letters_true_stress'] = G.phone_error_rate(ref, hyp_true)
    model.set_dictionaries(visible.get)
    model.use_siblings = False
    meta['per_oov_plain'] = G.phone_error_rate(ref, model.transcribe_many(words))
    model.use_siblings = True
    meta['per_oov_morph'] = G.phone_error_rate(ref, model.transcribe_many(words))
    for level in range(4):
        meta[f'per_rules{level}_true_stress'] = G.phone_error_rate(
            ref, [R.transcribe_word(w, s, level) for w, s in zip(words, truth)])
    meta['per_rules3_guessed_stress'] = G.phone_error_rate(ref, [R.transcribe_word(w, None, 3) for w in words])
    log(json.dumps(meta, ensure_ascii=False, indent=1))
    return meta


def train_chooser(train_sil, lexicon: dict, g2p: G.G2P, log=print):
    """Fit the contextual choice between pronunciations on the training sentences with pauses.

    Every sentence is used twice: with its pauses and without them, so that the model sees both and
    does not depend on pauses being there.
    """
    from g2p_context import VariantChooser, SIL
    sentences, gold = [], []
    for _, sentence in train_sil.groupby('id', sort=False):
        sentence = sentence.sort_values('token_id')
        tokens = list(sentence.token)
        phones = list(sentence.phones)
        sentences.append(tokens)
        gold.append(phones)
        plain = [(t, p) for t, p in zip(tokens, phones) if t != SIL]
        if len(plain) != len(tokens):
            sentences.append([t for t, _ in plain])
            gold.append([p for _, p in plain])
    chooser = VariantChooser()
    t0 = time.time()
    chooser.fit(sentences, gold, lexicon, g2p._best, verbose=True)
    log(f'chooser fitted in {time.time() - t0:.0f} s')
    return chooser


def save_chooser(chooser, path: str) -> None:
    import gzip
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with gzip.open(path, 'wb', compresslevel=6) as fh:
        pickle.dump(chooser, fh, protocol=pickle.HIGHEST_PROTOCOL)


def main(argv: list[str]) -> None:
    what = argv[1] if len(argv) > 1 else 'oov'
    os.makedirs(MODELS_DIR, exist_ok=True)
    tokens = G.load_tokens()
    train = tokens[tokens.set == 'train']
    lexicon = G.build_lexicon(train)
    mfa_dict = G.load_mfa_dict()
    if what == 'oov':
        model = train_oov(lexicon, mfa_dict)
        model.save(os.path.join(MODELS_DIR, 'oov_model.pkl.gz'))
    elif what == 'chooser':
        sil = G.load_tokens('data/tokens_phones_sil.csv')
        train_sil = sil[sil.set == 'train']
        g2p = G.G2P(lexicon, mfa_dict, oov='rules3', chooser=None)
        chooser = train_chooser(train_sil, lexicon, g2p)
        save_chooser(chooser, os.path.join(MODELS_DIR, 'variant_chooser.pkl.gz'))


if __name__ == '__main__':
    main(sys.argv)

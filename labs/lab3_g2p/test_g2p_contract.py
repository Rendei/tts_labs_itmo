"""Contract tests of the G2P module (run from this directory: `pytest test_g2p_contract.py`).

The first tests need nothing but the code; the last ones need the files of `prepare_lexicon.py` and
`train_g2p.py` and are skipped when they are absent.
"""
import os
from collections import Counter

import pytest

import g2p as G

SIL = G.SIL
TOY_LEXICON = {'дом': Counter({'d̪ o m': 3}), 'в': Counter({'v': 5, 'f': 4})}
TARGET = set(open(G.PHONES_FILE, encoding='utf-8').read().split())


def test_parse_token_normalizes_and_reads_stress():
    assert G.parse_token('ЗАМОК') == [('замок', None)]
    assert G.parse_token('з+амок') == [('замок', 1)]
    assert G.parse_token('зам+ок') == [('замок', 3)]
    assert G.parse_token('кто‐То') == [('кто', None), ('то', None)]
    assert G.parse_token('…') == []


@pytest.mark.parametrize('oov', [None, 'rules0', 'rules3'])
def test_length_and_pauses_are_preserved(oov):
    g = G.G2P(TOY_LEXICON, mfa_dict=None, oov=oov, chooser=None)
    tokens = [SIL, 'дом', SIL, SIL, 'Неизвестное', '!!!', 'кто-то', SIL]
    out = g.transcribe(tokens)
    assert len(out) == len(tokens)
    for token, phones in zip(tokens, out):
        assert (phones == [SIL]) == (token == SIL)
    assert g.transcribe([]) == []
    assert g.transcribe([SIL]) == [[SIL]]


def test_rules_only_use_target_phones():
    g = G.G2P({}, mfa_dict=None, oov='rules3', chooser=None)
    for word in ['яблоко', 'ёлка', 'съезд', 'щука', 'июль', 'объём', 'обезьяна', 'вообще']:
        phones = g.transcribe([word])[0]
        assert phones and set(phones) <= TARGET, (word, phones)


def test_explicit_stress_selects_the_variant():
    lexicon = {'замок': Counter({'z̪ a m ə k': 2, 'z̪ ɐ m o k': 1})}
    g = G.G2P(lexicon, mfa_dict=None, oov=None, chooser=None)
    assert g.transcribe(['замок'])[0] == ['z̪', 'a', 'm', 'ə', 'k']
    assert g.transcribe(['зам+ок'])[0] == ['z̪', 'ɐ', 'm', 'o', 'k']


needs_data = pytest.mark.skipif(not (os.path.exists(G.TOKENS_PHONES_DATA) and os.path.exists(G.MFA_DICT)
                                      and os.path.exists(G.OOV_MODEL)),
                                reason='run prepare_lexicon.py and train_g2p.py first')


@needs_data
def test_full_module_on_a_test_sample():
    tokens = G.load_tokens(verbose=False)
    train, test = tokens[tokens.set == 'train'], tokens[tokens.set == 'test']
    lexicon = G.build_lexicon(train)
    g = G.load_g2p(lexicon, G.load_mfa_dict())
    sample = test[test.id.isin(test.id.unique()[:200])]
    used = Counter()
    for _, sentence in sample.groupby('id', sort=False):
        words = list(sentence.sort_values('token_id').token)
        out = g.transcribe(words)
        assert len(out) == len(words)
        for phones in out:
            used.update(phones)
    assert set(used) <= TARGET
    result = G.evaluate(g, sample, lexicon, G.load_mfa_dict())
    assert result['text'] < 0.025


@needs_data
def test_default_module_is_complete():
    """`G2P(lexicon)` alone loads the MFA dictionary, the OOV model and the chooser from the lab folders."""
    tokens = G.load_tokens(verbose=False)
    lexicon = G.build_lexicon(tokens[tokens.set == 'train'])
    g = G.G2P(lexicon)
    assert g.mfa_dict and not isinstance(g.oov, str)
    out = g.transcribe(['мятежном', SIL, 'Фамилиями'])
    assert out[0] and out[1] == [SIL] and out[2]

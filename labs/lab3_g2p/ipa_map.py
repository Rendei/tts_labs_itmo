"""Bring the output of external transcribers to the target phone inventory — lab 3, extra task 3.

Every tool has its own alphabet (IPA with different conventions, or its own ASCII codes). To score them
against the corpus with the same PER, their outputs are mapped to `common/phones_mfa.txt` here. The mapping
is a best effort and it is part of the comparison: where a tool's alphabet is coarser than the target one
(`russian_g2p` has one unstressed `A` where the target has `ɐ` and `ə`), the choice made below decides how
many errors are counted, so the report gives a score that does not depend on vowel quality as well.

No external package is needed: the tools are run by `external_tools.py` in another environment.
"""
import re

import g2p_rules as R

VOWELS = frozenset('a e i o u æ ɐ ə ɛ ɨ ɪ ɵ ʉ ʊ'.split())
IPA_VOWELS = {'a': 'a', 'ɑ': 'a', 'æ': 'æ', 'ɐ': 'ɐ', 'ʌ': 'ɐ', 'ə': 'ə', 'e': 'e', 'ɛ': 'ɛ', 'ɜ': 'ə', 'i': 'i',
              'ɪ': 'ɪ', 'ɨ': 'ɨ', 'ɯ': 'ɨ', 'o': 'o', 'ɔ': 'o', 'ɵ': 'ɵ', 'ø': 'ɵ', 'u': 'u', 'ʊ': 'ʊ', 'ʉ': 'ʉ',
              'y': 'ɨ'}
# consonant -> (hard, soft); a soft variant exists only for the paired consonants
IPA_CONSONANTS = {
    'p': ('p', 'pʲ'), 'b': ('b', 'bʲ'), 't': ('t̪', 'tʲ'), 'd': ('d̪', 'dʲ'), 'k': ('k', 'c'), 'g': ('ɡ', 'ɟ'),
    'ɡ': ('ɡ', 'ɟ'), 'f': ('f', 'fʲ'), 'v': ('v', 'vʲ'), 's': ('s̪', 'sʲ'), 'z': ('z̪', 'zʲ'), 'x': ('x', 'ç'),
    'm': ('m', 'mʲ'), 'n': ('n̪', 'ɲ'), 'l': ('ɫ', 'ʎ'), 'ɫ': ('ɫ', 'ʎ'), 'ɭ': ('ɫ', 'ʎ'), 'r': ('r', 'rʲ'),
    'ɾ': ('r', 'rʲ'), 'ʃ': ('ʂ', 'ɕ'), 'ʂ': ('ʂ', 'ɕ'), 'ʒ': ('ʐ', 'ʑ'), 'ʐ': ('ʐ', 'ʑ'), 'ɕ': ('ɕ', 'ɕ'),
    'ʑ': ('ʑ', 'ʑ'), 'j': ('j', 'j'), 'ɲ': ('ɲ', 'ɲ'), 'ʎ': ('ʎ', 'ʎ'), 'ɣ': ('ɣ', 'ɣ'), 'ç': ('ç', 'ç'),
    'ɟ': ('ɟ', 'ɟ'), 'c': ('c', 'c'), 'h': ('x', 'ç'), 'w': ('v', 'vʲ'), 'q': ('k', 'c'),
}
IGNORED = set('ˈˌː͜͡ ̪̩̃̚˞ʰʷ˕˗ˑ.ː̪͜͡')


def map_ipa(ipa: str) -> list[str]:
    """Map an IPA string (espeak-ng, charsiu-g2p) to target phones.

    Stress marks, length marks, tie bars and dental diacritics are dropped; affricates `ts tʃ tɕ dz dʒ` become the
    target affricates; `ʲ` turns a consonant into its palatalized partner.
    """
    s = ''.join(ch for ch in ipa if ch not in IGNORED)
    out: list[str] = []
    i = 0
    while i < len(s):
        # affricates first
        two = s[i:i + 2]
        if two in ('ts', 'tʃ', 'tɕ', 'dz', 'dʒ', 'tʂ', 'dʑ', 'dʐ'):
            soft = i + 2 < len(s) and s[i + 2] == 'ʲ'
            out.append({'ts': 't̪s̪', 'tʃ': 'tɕ', 'tɕ': 'tɕ', 'dz': 'd̪z̪', 'dʒ': 'dʐ', 'tʂ': 'tʂ', 'dʑ': 'dʐ', 'dʐ': 'dʐ'}[two])
            i += 2 + soft
            continue
        ch = s[i]
        soft = i + 1 < len(s) and s[i + 1] == 'ʲ'
        if ch in IPA_VOWELS:
            out.append(IPA_VOWELS[ch])
        elif ch in IPA_CONSONANTS:
            hard, soft_form = IPA_CONSONANTS[ch]
            out.append(soft_form if soft else hard)
        elif ch == 'ʲ':
            pass
        i += 1 + (1 if soft and ch in IPA_CONSONANTS else 0)
    return out


RG2P_CONSONANTS = {
    'B': 'b', 'B0': 'bʲ', 'V': 'v', 'V0': 'vʲ', 'G': 'ɡ', 'G0': 'ɟ', 'GH': 'ɣ', 'GH0': 'ɣ', 'D': 'd̪', 'D0': 'dʲ',
    'DZ': 'd̪z̪', 'DZ0': 'd̪z̪', 'DZH': 'dʐ', 'DZH0': 'dʐ', 'ZH': 'ʐ', 'ZH0': 'ʑ', 'Z': 'z̪', 'Z0': 'zʲ', 'J0': 'j',
    'K': 'k', 'K0': 'c', 'L': 'ɫ', 'L0': 'ʎ', 'M': 'm', 'M0': 'mʲ', 'N': 'n̪', 'N0': 'ɲ', 'P': 'p', 'P0': 'pʲ',
    'R': 'r', 'R0': 'rʲ', 'S': 's̪', 'S0': 'sʲ', 'T': 't̪', 'T0': 'tʲ', 'F': 'f', 'F0': 'fʲ', 'KH': 'x', 'KH0': 'ç',
    'TS': 't̪s̪', 'TS0': 't̪s̪', 'TSH': 'tʂ', 'TSH0': 'tɕ', 'SH': 'ʂ', 'SH0': 'ɕ',
}
RG2P_STRESSED = {'A': 'a', 'O': 'o', 'U': 'u', 'E': 'e', 'I': 'i', 'Y': 'ɨ'}
RG2P_REDUCED = {'U': 'ʊ', 'E': 'ɪ', 'I': 'ɪ', 'Y': 'ɨ'}
SOFT_BEFORE = frozenset('bʲ pʲ dʲ tʲ vʲ fʲ zʲ sʲ mʲ rʲ c ɟ ç ɲ ʎ ɕ tɕ j ʑ'.split())


def map_russian_g2p(codes: str) -> list[str]:
    """Map the phone codes of `russian_g2p` (`Z A0 M A K`; `0` after a vowel = stressed, after a consonant = soft).

    The tool has one unstressed `A`/`O`; it is mapped to `ɐ` at the start of the word and right before the stressed
    vowel and to `ə` elsewhere, as the target alphabet does. A stressed `E0` after a hard consonant is `ɛ`.
    """
    codes = [c.rstrip('l') for c in codes.split() if c != 'sil']          # a trailing `l` marks a long phone
    is_vowel = lambda c: c[:1] in 'AOUEIY'
    vowels = [i for i, c in enumerate(codes) if is_vowel(c)]
    stressed = next((n for n, i in enumerate(vowels) if codes[i].endswith('0')), None)
    out: list[str] = []
    for i, code in enumerate(codes):
        if code in RG2P_CONSONANTS:
            out.append(RG2P_CONSONANTS[code])
        elif is_vowel(code):
            letter, n = code[0], vowels.index(i)
            if code.endswith('0'):
                phone = RG2P_STRESSED[letter]
                if letter == 'E' and out and out[-1] not in SOFT_BEFORE:
                    phone = 'ɛ'
                out.append(phone)
            elif letter in 'AO':
                out.append('ɐ' if n == 0 or (stressed is not None and stressed - n == 1) else 'ə')
            else:
                out.append(RG2P_REDUCED[letter])
    return out


def consonants_only(phones: list[str]) -> list[str]:
    """Phones with every vowel replaced by one class symbol: scores that do not depend on vowel quality."""
    return ['V' if p in VOWELS else p for p in phones]


def plus_to_stress_index(accented: str) -> tuple[str, int | None]:
    """Split a RUAccent word (`з+амок`, `вс+ё`) into the plain word and the index of the stressed letter."""
    stress = None
    letters = []
    mark = False
    for ch in accented:
        if ch == '+':
            mark = True
            continue
        if mark and ch in R.VOWEL_LETTERS and stress is None:
            stress = len(letters)
        mark = False
        letters.append(ch)
    return ''.join(letters), stress

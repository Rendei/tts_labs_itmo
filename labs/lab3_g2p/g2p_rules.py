"""Hand-written grapheme-to-phoneme rules for Russian - lab 3.

The rules map a lowercase Cyrillic word to phones of the target inventory
(`common/phones_mfa.txt`). They are built up in levels so that every group of rules can be
switched off and its contribution measured (see the report):

0. a naive letter-by-letter table (the baseline of the assignment);
1. + palatalization of consonants and iotation of `е ё ю я`;
2. + voicing assimilation, final devoicing, doubled letters, `-ого`, `-тся`, `сч` and a few
   consonant clusters;
3. + vowel reduction. It needs the stressed vowel, which the rules cannot know: it is passed in
   (`stress`) or guessed (`guess_stress`).

Interaction between words (`в тот` -> `f t̪ ...`) is not handled here: that is the job of the
contextual model in `g2p.py`.
"""
LETTERS = 'абвгдежзийклмнопрстуфхцчшщъыьэюяё'
VOWEL_LETTERS = frozenset('аеёиоуыэюя')

# consonant letter -> (hard phone, soft phone)
CONSONANTS = {
    'б': ('b', 'bʲ'), 'в': ('v', 'vʲ'), 'г': ('ɡ', 'ɟ'), 'д': ('d̪', 'dʲ'), 'з': ('z̪', 'zʲ'),
    'к': ('k', 'c'), 'л': ('ɫ', 'ʎ'), 'м': ('m', 'mʲ'), 'н': ('n̪', 'ɲ'), 'п': ('p', 'pʲ'),
    'р': ('r', 'rʲ'), 'с': ('s̪', 'sʲ'), 'т': ('t̪', 'tʲ'), 'ф': ('f', 'fʲ'), 'х': ('x', 'ç'),
    'ж': ('ʐ', 'ʐ'), 'ш': ('ʂ', 'ʂ'), 'ц': ('t̪s̪', 't̪s̪'), 'ч': ('tɕ', 'tɕ'), 'щ': ('ɕ', 'ɕ'),
    'й': ('j', 'j'),
}
ALWAYS_HARD = frozenset('жшц')
ALWAYS_SOFT = frozenset('чщй')
SOFTENING = frozenset('еёиюя')          # vowel letters that palatalize the consonant before them
IOTATED = frozenset('еёюя')

# the naive table: every vowel in its full form, every consonant hard
NAIVE_VOWELS = {'а': ['a'], 'о': ['o'], 'у': ['u'], 'э': ['ɛ'], 'ы': ['ɨ'], 'и': ['i'],
                'е': ['j', 'e'], 'ё': ['j', 'ɵ'], 'ю': ['j', 'u'], 'я': ['j', 'a']}

VOICELESS_OF = {'b': 'p', 'bʲ': 'pʲ', 'v': 'f', 'vʲ': 'fʲ', 'ɡ': 'k', 'ɟ': 'c', 'd̪': 't̪',
                'dʲ': 'tʲ', 'z̪': 's̪', 'zʲ': 'sʲ', 'ʐ': 'ʂ'}
VOICED_OF = {v: k for k, v in VOICELESS_OF.items()}
VOICELESS = frozenset(VOICELESS_OF.values()) | {'x', 'ç', 't̪s̪', 'tɕ', 'ɕ'}
VOICING_TRIGGERS = frozenset(VOICELESS_OF) - {'v', 'vʲ'}      # `в` does not voice a consonant before it


def vowel_positions(word: str) -> list[int]:
    """Indexes of the vowel letters of `word`."""
    return [i for i, c in enumerate(word) if c in VOWEL_LETTERS]


def clean(word: str) -> str:
    """Lowercase, keep only Cyrillic letters."""
    return ''.join(c for c in word.lower() if c in LETTERS)


def guess_stress(word: str) -> int | None:
    """Index of the stressed vowel letter of `word`, guessed without any data.

    `ё` is always stressed, a single vowel is stressed; otherwise the penultimate vowel, the most
    frequent position in Russian words of three and more syllables.
    """
    vowels = vowel_positions(word)
    if not vowels:
        return None
    if 'ё' in word:
        return word.index('ё')
    return vowels[0] if len(vowels) <= 2 else vowels[-2]


def transcribe_word(word: str, stress: int | None = None, level: int = 3) -> list[str]:
    """Rule-based transcription of one word.

    Args:
        word: Cyrillic word; other characters are ignored.
        stress: Index of the stressed vowel letter in the *cleaned* word, or None to guess it
            (level 3 only).
        level: How many groups of rules to apply, 0..3 (see the module docstring).

    Returns:
        Phones of the target inventory.
    """
    w = clean(word)
    n = len(w)
    if n == 0:
        return []
    if level == 0:
        out: list[str] = []
        for c in w:
            if c in NAIVE_VOWELS:
                out += NAIVE_VOWELS[c]
            elif c in CONSONANTS:
                out.append(CONSONANTS[c][0])
        return out

    vowels = vowel_positions(w)
    if level >= 3 and stress is None:
        stress = guess_stress(w)
    stress_no = vowels.index(stress) if stress in vowels else None

    units: list[list[str]] = [[] for _ in range(n)]
    for i, c in enumerate(w):
        prev = w[i - 1] if i else ''
        nxt = w[i + 1] if i + 1 < n else ''
        if c in CONSONANTS:
            hard, soft = CONSONANTS[c]
            soft_here = c in ALWAYS_SOFT or (c not in ALWAYS_HARD and (nxt in SOFTENING or nxt == 'ь'))
            units[i] = [soft if soft_here else hard]
        elif c in VOWEL_LETTERS:
            iotated = c in IOTATED and (i == 0 or prev in VOWEL_LETTERS or prev in 'ьъ')
            after_hard = i > 0 and prev in CONSONANTS and prev not in ALWAYS_SOFT and (
                prev in ALWAYS_HARD or c not in SOFTENING)
            vowel_no = vowels.index(i)
            offset = vowel_no - stress_no if (level >= 3 and stress_no is not None) else 0
            phone = _vowel_phone(c, offset, level >= 3, vowel_no == len(vowels) - 1,
                                 vowel_no == 0, after_hard, iotated, prev)
            units[i] = (['j'] if iotated and c in IOTATED else []) + [phone]

    if level >= 2:
        _spelling_rules(w, units)
        _voicing(w, units)
    return [p for u in units for p in u]


def _vowel_phone(c: str, offset: int, reduce: bool, is_last: bool, is_first: bool,
                 after_hard: bool, iotated: bool, prev: str) -> str:
    """Phone of vowel letter `c`; `offset` = vowel number minus the number of the stressed vowel.

    Without `reduce` every vowel gets its stressed (full) form.
    """
    if not reduce or offset == 0:
        if c == 'е':
            return 'ɛ' if after_hard else 'e'
        if c == 'ё':
            return 'o' if after_hard else 'ɵ'
        if c == 'ю':
            return 'u' if iotated or after_hard else 'ʉ'
        return {'а': 'a', 'о': 'o', 'у': 'u', 'э': 'ɛ', 'ы': 'ɨ', 'и': 'i', 'я': 'a'}[c]
    if c in 'ао':
        return 'ɐ' if (offset == -1 or is_first) else 'ə'
    if c == 'э':
        return 'ɪ'
    if c in 'уюы':
        return {'у': 'ʊ', 'ю': 'ʊ', 'ы': 'ɨ'}[c]
    if c == 'е':
        if is_last and not iotated:
            return 'e'
        return 'ɨ' if prev in ALWAYS_HARD else ('ə' if is_last else 'ɪ')
    if c == 'я':
        return 'ə' if is_last else 'ɪ'
    if c == 'ё':
        return 'ɪ'
    return 'i' if is_last else 'ɪ'                                   # и


def _spelling_rules(w: str, units: list[list[str]]) -> None:
    """Orthographic rules that do not depend on the stress; modifies `units` in place."""
    n = len(w)
    for i in range(n - 1):
        if w[i] == w[i + 1] and w[i] in CONSONANTS and w[i] != 'й':      # doubled letters -> one phone
            units[i] = []
    if n > 4 and w.endswith(('ого', 'его')):                           # -ого/-его: г -> в
        units[n - 2] = ['v']
    if w.endswith('ться') and n > 4:
        units[n - 4], units[n - 2] = [], ['t̪s̪']
    elif w.endswith('тся') and n > 3:
        units[n - 3], units[n - 2] = [], ['t̪s̪']
    for i in range(n - 1):
        pair = w[i:i + 2]
        if pair in ('сч', 'зч', 'жч'):
            units[i], units[i + 1] = [], ['ɕ']
        elif pair in ('дц', 'тц'):
            units[i], units[i + 1] = [], ['t̪s̪']
        elif pair in ('сш', 'зш'):
            units[i], units[i + 1] = [], ['ʂ']
        elif pair in ('сж', 'зж'):
            units[i], units[i + 1] = [], ['ʐ']
    for i in range(n - 2):
        if w[i:i + 3] in ('стн', 'здн', 'стл'):                          # silent т, д
            units[i + 1] = []
    if w in ('что', 'чтобы'):
        units[0] = ['ʂ']


def _voicing(w: str, units: list[list[str]]) -> None:
    """Regressive voicing assimilation inside the word and devoicing at its end."""
    sounding = [i for i, u in enumerate(units) if u]
    for pos in range(len(sounding) - 1, -1, -1):
        i = sounding[pos]
        if w[i] not in CONSONANTS or len(units[i]) != 1:
            continue
        ph = units[i][0]
        if pos == len(sounding) - 1:                                       # final devoicing
            if ph in VOICELESS_OF:
                units[i] = [VOICELESS_OF[ph]]
            continue
        j = sounding[pos + 1]
        if w[j] not in CONSONANTS:
            continue
        nxt = units[j][0]
        if nxt in VOICELESS and ph in VOICELESS_OF:
            units[i] = [VOICELESS_OF[ph]]
        elif nxt in VOICING_TRIGGERS and ph in VOICED_OF:
            units[i] = [VOICED_OF[ph]]

"""Re-align RUSLAN with Montreal Forced Aligner on the lab 1 normalized text — lab 2, extra task 1.

The baseline markup (`data/RUSLAN_align_v2/`) was built on the raw text. Here the aligner gets the normalized
text instead, and the dictionary is extended with pronunciations for the words it does not know, so that fewer
words are aligned as noise (`spn`). The steps, each a function and a sub-command::

    python realign_mfa.py corpus     # corpus dir: hard links to wav + one .txt per utterance
    python realign_mfa.py models     # download the Russian MFA models
    python realign_mfa.py oov        # find words missing from the dictionary
    python realign_mfa.py g2p        # predict their pronunciation, write the extended dictionary
    python realign_mfa.py align      # run `mfa align` (add `--subset 300` to try on a few utterances first)
    python realign_mfa.py table      # build the word table `labs/lab2_pauses/data/RUSLAN_pause_metadata_v3.csv`

MFA runs in Docker (image `mmcauliffe/montreal-forced-aligner`); everything lives in `data/mfa/`:
`corpus/`, `models/`, `align_v3/` (TextGrid output) and `oov/`.
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from praatio import textgrid

from pause_predictor import split_token, tokenize_text

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / 'data' / 'metadata_RUSLAN_22200_normalized.csv'
WAV_DIR = ROOT / 'data' / 'RUSLAN' / 'RUSLAN'
MFA_DIR = ROOT / 'data' / 'mfa'
CORPUS_DIR = MFA_DIR / 'corpus'
MODELS_DIR = MFA_DIR / 'models'
ALIGN_DIR = MFA_DIR / 'align_v3'
OOV_DIR = MFA_DIR / 'oov'
MODELS_DICT = MFA_DIR / 'root' / 'pretrained_models' / 'dictionary' / 'russian_mfa.dict'
EXT_DICT = MODELS_DIR / 'russian_mfa_ruslan.dict'
TABLE_V3 = Path(__file__).resolve().parent / 'data' / 'RUSLAN_pause_metadata_v3.csv'
TABLE_COLUMNS = ['id', 'label', 'label_raw', 'duration', 'is_last_word', 'is_pause_after', 'pause_duration', 'set', 'spn', 'start']

IMAGE = 'mmcauliffe/montreal-forced-aligner:v3.4.1'
STRESS = '́'
WORD_RE = r"\w+(?:'\w+)*"


def mfa_text(text: str) -> str:
    """Text as the aligner gets it: without the stress mark (MFA does not know it), with plain hyphens and apostrophes."""
    text = unicodedata.normalize('NFD', unicodedata.normalize('NFC', text))
    text = unicodedata.normalize('NFC', ''.join(c for c in text if c != STRESS))
    return text.replace('’', "'").replace('‐', '-').replace('‑', '-')


def prepare_corpus() -> int:
    """Create `data/mfa/corpus/`: for every utterance a hard link to its wav and a `.txt` with the text."""
    meta = pd.read_csv(META, sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for uid, text in meta[['id', 'nrm']].values:
        wav, link = WAV_DIR / f'{uid}.wav', CORPUS_DIR / f'{uid}.wav'
        if not link.exists():
            os.link(wav, link)        # hard link: no copy of the 9 GB, both names are the same file
        (CORPUS_DIR / f'{uid}.txt').write_text(mfa_text(str(text)), encoding='utf-8')
        n += 1
    return n


def docker(*args: str) -> subprocess.CompletedProcess:
    """Run a command inside the MFA container.

    `data/mfa` is mounted at `/mfa` (input, dictionary, output). MFA's own working directory (its database and
    pretrained models) is a Docker volume, not a folder on the host: on a mounted Windows drive MFA spends minutes
    on every step writing its database.
    """
    cmd = ['docker', 'run', '--rm', '-u', '0', '-v', f'{MFA_DIR}:/mfa', '-v', 'mfa_root:/mfa_root', '-e', 'MFA_ROOT_DIR=/mfa_root',
           IMAGE, *args]
    print('>', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, check=True)


def download_models() -> None:
    """Download the Russian dictionary, acoustic and G2P models (v3.1.0) into the `mfa_root` volume."""
    for kind in ('dictionary', 'acoustic', 'g2p'):
        docker('mfa', 'model', 'download', kind, 'russian_mfa')


def corpus_words(text: str) -> list[str]:
    """Words of a text the way MFA sees them: lowercase letter sequences; a hyphen separates two words."""
    return re.findall(WORD_RE, mfa_text(text).lower())


def find_oov() -> pd.DataFrame:
    """Words of the corpus that are not in the MFA dictionary.

    Writes `oov/oov_words.txt` (one word per line, for the G2P model) and `oov/oov_counts.csv`; returns the counts.
    """
    known = set(pd.read_csv(MODELS_DICT, sep='\t', header=None, usecols=[0], quoting=csv.QUOTE_NONE)[0].astype(str))
    meta = pd.read_csv(META, sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    counts, oov_counts, utts_with_oov = Counter(), Counter(), 0
    for text in meta.nrm.astype(str):
        words = corpus_words(text)
        counts.update(words)
        missing = [w for w in words if w not in known]
        oov_counts.update(missing)
        utts_with_oov += bool(missing)
    oov = pd.DataFrame(oov_counts.items(), columns=['word', 'count']).sort_values('count', ascending=False)
    OOV_DIR.mkdir(parents=True, exist_ok=True)
    oov.word.to_csv(OOV_DIR / 'oov_words.txt', index=False, header=False)
    oov.to_csv(OOV_DIR / 'oov_counts.csv', index=False)
    total = sum(counts.values())
    print(f'words: {total}, distinct: {len(counts)}; OOV distinct: {len(oov)}, OOV tokens: {oov["count"].sum()} '
          f'({oov["count"].sum() / total:.2%}); utterances with OOV: {utts_with_oov} of {len(meta)}')
    return oov


def predict_oov() -> None:
    """Pronunciations for the OOV words from the G2P model, and the extended dictionary `russian_mfa_ruslan.dict`."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    docker('mfa', 'g2p', '/mfa/oov/oov_words.txt', 'russian_mfa', '/mfa/oov/oov_g2p.txt', '--clean')
    base = MODELS_DICT.read_text(encoding='utf-8').strip().split('\n')
    extra = (OOV_DIR / 'oov_g2p.txt').read_text(encoding='utf-8').strip().split('\n')
    EXT_DICT.write_text('\n'.join(sorted(base + extra)) + '\n', encoding='utf-8')
    print(f'dictionary: {len(base)} + {len(extra)} = {len(base) + len(extra)} entries -> {EXT_DICT}')


def align(subset: int | None = None, extra: tuple[str, ...] = (), name: str | None = None) -> None:
    """Run `mfa align` on the corpus (or on its first `subset` utterances) with the extended dictionary.

    `extra` are additional `mfa align` options (for experiments), `name` the output folder inside `data/mfa`.
    """
    corpus = '/mfa/corpus'
    if subset:
        sub = MFA_DIR / 'corpus_subset'
        shutil.rmtree(sub, ignore_errors=True)
        sub.mkdir(parents=True)
        for txt in sorted(CORPUS_DIR.glob('*.txt'))[:subset]:
            shutil.copy(txt, sub / txt.name)
            os.link(CORPUS_DIR / (txt.stem + '.wav'), sub / (txt.stem + '.wav'))
        corpus = '/mfa/corpus_subset'
    out = f'/mfa/{name or ("align_subset" if subset else "align_v3")}'
    docker('mfa', 'align', corpus, '/mfa/models/russian_mfa_ruslan.dict', 'russian_mfa', out,
           '--clean', '--single_speaker', '-j', '8', '--beam', '100', '--retry_beam', '400', *extra)


def _plain(word: str) -> str:
    """Lowercase word without stress marks — how MFA writes it in the `words` tier."""
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', word.lower()) if c != STRESS))


def utterance_rows(uid: str, text: str, textgrid_path: Path) -> list[dict]:
    """Word table rows of one utterance: predictor tokens matched with the MFA words.

    The tokens come from :func:`pause_predictor.tokenize_text`. A hyphenated token (``что-то``) is one
    token for the predictor but two words for MFA; its MFA parts are merged. The pause after a token is the
    silence interval between its last word and the first word of the next token.

    Raises:
        ValueError: A word from the TextGrid does not match the text.
    """
    tokens = tokenize_text(text)
    tiers = textgrid.openTextgrid(str(textgrid_path), True).tiers
    entries, phones = tiers[0].entries, tiers[1].entries
    spn = [(p.start, p.end) for p in phones if p.label == 'spn']

    rows, pos = [], 0
    for token in tokens:
        core = _plain(split_token(token)[1])
        parts = re.findall(WORD_RE, core)
        start = end = None
        for part in parts:
            silence = 0.0
            while pos < len(entries) and entries[pos].label == '':
                silence += entries[pos].end - entries[pos].start
                pos += 1
            if start is None and pos < len(entries) and entries[pos].label == core:
                part = core                                 # the dictionary knows the whole hyphenated word (э-э)
            if pos >= len(entries) or entries[pos].label != part:
                got = entries[pos].label if pos < len(entries) else '<end>'
                raise ValueError(f'{uid}: expected "{part}", TextGrid has "{got}"')
            if start is None:
                start = entries[pos].start
                if rows:                                   # silence between the previous token and this one
                    rows[-1]['pause_duration'] = silence
            end = entries[pos].end
            pos += 1
            if part == core:
                break
        rows.append({'label': '-'.join(parts), 'label_raw': token, 'duration': end - start, 'start': start, 'end': end,
                     'is_last_word': 0, 'pause_duration': 0.0,
                     'spn': int(any(a < end and b > start for a, b in spn))})
    if pos < len(entries) and not rows:
        raise ValueError(f'{uid}: no words in the text')
    trailing = sum(e.end - e.start for e in entries[pos:] if e.label == '')
    if rows:
        rows[-1]['is_last_word'] = 1
        rows[-1]['pause_duration'] = trailing
    for r in rows:
        r['id'] = uid
        r['is_pause_after'] = int(r['pause_duration'] > 0)
        r['set'] = 'test' if int(uid.split('_')[0]) % 5 == 0 else 'train'
    return rows


def build_table(align_dir: Path = ALIGN_DIR, out_path: Path = TABLE_V3) -> pd.DataFrame:
    """Build the word table (the lab 2 format plus `spn` and `start`) from the new TextGrids.

    Utterances whose TextGrid is missing or does not match the text are skipped; the reasons are saved next to the table.
    """
    meta = pd.read_csv(META, sep='|', names=['id', 'raw', 'nrm'], quoting=csv.QUOTE_NONE)
    rows, failures = [], []
    for uid, text in meta[['id', 'nrm']].values:
        path = Path(align_dir) / f'{uid}.TextGrid'
        if not path.exists():
            failures.append((uid, 'no TextGrid'))
            continue
        try:
            rows += utterance_rows(uid, str(text), path)
        except ValueError as err:
            failures.append((uid, str(err)))
    table = pd.DataFrame(rows)[TABLE_COLUMNS]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, sep='|', index=False, quoting=csv.QUOTE_NONE)
    pd.DataFrame(failures, columns=['id', 'reason']).to_csv(out_path.with_name(out_path.stem + '_failures.csv'), index=False)
    print(f'{table.id.nunique()} utterances, {len(table)} words; skipped: {len(failures)}')
    return table


if __name__ == '__main__':
    commands = {'corpus': prepare_corpus, 'models': download_models, 'oov': find_oov, 'g2p': predict_oov, 'align': align, 'table': build_table}
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd not in commands:
        sys.exit(f'usage: python realign_mfa.py [{"|".join(commands)}]')
    if cmd == 'align' and '--subset' in sys.argv:
        rest = sys.argv[sys.argv.index('--subset') + 2:]
        align(int(sys.argv[sys.argv.index('--subset') + 1]), extra=tuple(rest[2:]) if rest[:1] == ['--name'] else tuple(rest),
              name=rest[1] if rest[:1] == ['--name'] else None)
    else:
        print(commands[cmd]())

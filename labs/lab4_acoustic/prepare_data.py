"""Prepare the phone metadata for lab 4: copy lab 3 output into `data/` and split it.

    python prepare_data.py

Inputs (lab 3, MFA re-alignment "v3", see the lab 3 report):

- `labs/lab3_g2p/data/metadata_phones_gt.csv` - phones from the forced alignment (reference);
- `labs/lab3_g2p/data/metadata_phones.csv`    - phones predicted by our G2P module.

Outputs in `data/` (all `id|text|phones`):

| file | content |
|---|---|
| `metadata_phones_gt.csv`, `metadata_phones.csv` | full sets, the lab 3 artifacts the lab 4 README expects |
| `*_train.csv` | utterances whose number is **not** divisible by 5 - the training data |
| `*_test.csv`  | utterances whose number is divisible by 5 - never seen in training |
| `eval_sentences_g2p.csv` | the 120 shipped evaluation sentences with **our G2P** phones |

The shipped `labs/lab4_acoustic/data/eval_sentences.csv` carries reference phones (official
v2 alignment). The G2P copy lets us synthesize the same sentences the way the full pipeline
would: from text through our own transcription, with no reference phones available.

Every phone is checked against `common/phones_mfa.txt` (NFC), as the trainer's pre-flight does.
"""
import argparse
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lab3-data", type=Path, default=ROOT / "labs/lab3_g2p/data")
    p.add_argument("--out", type=Path, default=ROOT / "data")
    p.add_argument("--phones", type=Path, default=ROOT / "common/phones_mfa.txt")
    p.add_argument("--eval-set", type=Path,
                   default=ROOT / "labs/lab4_acoustic/data/eval_sentences.csv")
    return p.parse_args()


def read_rows(path: Path) -> list[list[str]]:
    """Read `id|text|phones` rows, NFC-normalizing the phone column."""
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            if len(cols) < 3:
                continue
            cols[2] = unicodedata.normalize("NFC", cols[2])
            rows.append(cols[:3])
    return rows


def write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write("|".join(row) + "\n")


def is_test(utt_id: str) -> bool:
    """`000020_RUSLAN` -> 20 -> test, because 20 % 5 == 0."""
    return int(utt_id.split("_")[0]) % 5 == 0


def main() -> None:
    args = parse_args()
    inventory = {unicodedata.normalize("NFC", line.strip())
                 for line in args.phones.read_text(encoding="utf-8").splitlines() if line.strip()}

    for name in ("metadata_phones_gt", "metadata_phones"):
        rows = read_rows(args.lab3_data / f"{name}.csv")
        unknown = {p for r in rows for p in r[2].split()} - inventory
        if unknown:
            raise SystemExit(f"{name}: phones outside the inventory: {sorted(unknown)}")
        train = [r for r in rows if not is_test(r[0])]
        test = [r for r in rows if is_test(r[0])]
        write_rows(args.out / f"{name}.csv", rows)
        write_rows(args.out / f"{name}_train.csv", train)
        write_rows(args.out / f"{name}_test.csv", test)
        print(f"{name:20s} all {len(rows):6d}  train {len(train):6d}  test {len(test):5d}")

    g2p = {r[0]: r for r in read_rows(args.lab3_data / "metadata_phones.csv")}
    eval_rows = read_rows(args.eval_set)
    missing = [r[0] for r in eval_rows if r[0] not in g2p]
    if missing:
        raise SystemExit(f"eval sentences missing from the G2P metadata: {missing}")
    write_rows(args.out / "eval_sentences_g2p.csv", [g2p[r[0]] for r in eval_rows])
    same = sum(g2p[r[0]][2] == r[2] for r in eval_rows)
    print(f"eval_sentences_g2p   {len(eval_rows)} sentences, {same} with phones identical "
          f"to the shipped reference")


if __name__ == "__main__":
    main()

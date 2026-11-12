"""Intelligibility of synthesized speech: Whisper transcribes the wavs, CER/WER against the text.

    python asr_eval.py runs/gt/eval:gt runs/pitch_energy/eval_40k:pitch_energy@40k [--variants model]

Each argument is `<evaluate.py output dir>:<label>`. For every listed variant subdirectory
(`natural`, `copysynth`, `model`) all wavs are transcribed and appended to
`runs/asr_results.csv` (`run|variant|id|ref|hyp|cer|wer`); rows already there are skipped,
so the script can be re-run as new evaluations appear.

Natural and copy-synthesis are the reference points: the first is what ASR does on real
speech (its own error floor), the second shows how much intelligibility Griffin-Lim alone
takes away. Text is normalized before scoring: lower case, `ё`->`е`, punctuation removed.
"""
import argparse
import csv
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))  # ASCII path, inside data/

import numpy as np
import soundfile as sf

MODEL = "openai/whisper-large-v3-turbo"
OUT = Path(__file__).resolve().parent / "runs" / "asr_results.csv"
FIELDS = ["run", "variant", "id", "ref", "hyp", "cer", "wer"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="<eval dir>:<label>")
    p.add_argument("--variants", default="natural,copysynth,model")
    p.add_argument("--eval-set", type=Path, default=Path("data/eval_sentences.csv"))
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def main() -> None:
    import torch
    from transformers import pipeline

    args = parse_args()
    texts = {}
    with args.eval_set.open(encoding="utf-8") as fh:
        for line in fh:
            cols = line.rstrip("\n").split("|")
            texts[cols[0]] = cols[1]

    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as fh:
            done = {(r["run"], r["variant"], r["id"]) for r in csv.DictReader(fh, delimiter="|")}

    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline("automatic-speech-recognition", model=MODEL, device=device,
                   torch_dtype=torch.float16 if device == 0 else torch.float32)
    print(f"{MODEL} on {'cuda' if device == 0 else 'cpu'}")

    new = not OUT.exists()
    with OUT.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        if new:
            writer.writeheader()
        for spec in args.dirs:
            eval_dir, label = spec.rsplit(":", 1)
            for variant in args.variants.split(","):
                ids = [i for i in texts if (Path(eval_dir) / variant / f"{i}.wav").exists()
                       and (label, variant, i) not in done]
                if not ids:
                    continue
                audio = []
                for i in ids:
                    y, sr = sf.read(str(Path(eval_dir) / variant / f"{i}.wav"), dtype="float32")
                    audio.append({"raw": y, "sampling_rate": sr})
                hyps = asr(audio, batch_size=args.batch_size,
                           generate_kwargs={"language": "russian", "task": "transcribe"})
                cers, wers = [], []
                for i, h in zip(ids, hyps):
                    ref, hyp = normalize(texts[i]), normalize(h["text"])
                    cer = edit_distance(ref, hyp) / max(1, len(ref))
                    wer = edit_distance(ref.split(), hyp.split()) / max(1, len(ref.split()))
                    cers.append(cer); wers.append(wer)
                    writer.writerow({"run": label, "variant": variant, "id": i, "ref": ref,
                                     "hyp": hyp.replace("|", " "), "cer": round(cer, 4),
                                     "wer": round(wer, 4)})
                fh.flush()
                print(f"{label:24s} {variant:10s} n={len(ids):3d}  CER {np.mean(cers):.3f}  "
                      f"WER {np.mean(wers):.3f}")


if __name__ == "__main__":
    main()

"""Helpers for the lab 4 report: training curves, F0 (intonation) and duration analysis.

Everything here reads files that the training (`runs/<name>/`) and `evaluate.py`
(`runs/<name>/<eval dir>/`) already wrote; nothing needs the GPU.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SR = 22050
HOP = 256
FMIN, FMAX = 50.0, 300.0     # male speaker, same range as the --pitch-energy ablation


# ----------------------------------------------------------------------------------------
# training curves
# ----------------------------------------------------------------------------------------
def read_scalars(run_dir: Path) -> pd.DataFrame:
    """All TensorBoard scalars of a run as a long table `tag | step | value | wall_time`."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    frames = []
    for events in sorted(Path(run_dir).rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            ev = acc.Scalars(tag)
            frames.append(pd.DataFrame({"tag": tag, "step": [e.step for e in ev],
                                        "value": [e.value for e in ev],
                                        "wall_time": [e.wall_time for e in ev]}))
    if not frames:
        return pd.DataFrame(columns=["tag", "step", "value", "wall_time"])
    return pd.concat(frames, ignore_index=True)


def smooth(values: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """Exponential moving average with bias correction, as TensorBoard's smoothing slider."""
    out, acc = np.empty(len(values)), 0.0
    for i, v in enumerate(values):
        acc = alpha * acc + (1 - alpha) * v
        out[i] = acc / (1 - alpha ** (i + 1))
    return out


# ----------------------------------------------------------------------------------------
# F0
# ----------------------------------------------------------------------------------------
def f0_contour(wav_path: Path) -> np.ndarray:
    """pYIN F0 in Hz on the mel frame grid (hop 256 at 22.05 kHz); 0 where unvoiced."""
    import librosa
    import soundfile as sf

    y, sr = sf.read(str(wav_path))
    assert sr == SR, f"{wav_path}: {sr} Hz"
    f0, voiced, _ = librosa.pyin(y.astype(np.float64), fmin=FMIN, fmax=FMAX, sr=SR,
                                 frame_length=1024, hop_length=HOP)
    f0 = np.where(voiced, f0, 0.0)
    return np.nan_to_num(f0)


def _f0_job(args):
    key, path = args
    return key, f0_contour(path)


def f0_contours(paths: dict, workers: int = 6) -> dict:
    """`{key: wav path}` -> `{key: f0 array}`, in parallel processes."""
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_f0_job, paths.items(), chunksize=4))


def semitones(f0: np.ndarray, ref: float = 100.0) -> np.ndarray:
    return 12.0 * np.log2(f0 / ref)


def f0_summary(f0: np.ndarray) -> dict:
    """Voiced share, mean F0 and spread (std in semitones) of one utterance."""
    v = f0[f0 > 0]
    if len(v) < 5:
        return {"voiced": len(v) / max(1, len(f0)), "f0_mean": np.nan, "f0_std_st": np.nan,
                "f0_range_st": np.nan}
    st = semitones(v)
    return {"voiced": len(v) / len(f0), "f0_mean": float(np.mean(v)),
            "f0_std_st": float(np.std(st)),
            "f0_range_st": float(np.percentile(st, 95) - np.percentile(st, 5))}


def f0_agreement(f0_ref: np.ndarray, f0_hyp: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> dict:
    """Compare two F0 contours along a DTW path `(ia, ib)` computed on the mels.

    Returns the correlation of the semitone contours and their RMSE over frames voiced in
    both, plus the voicing decision agreement (V/UV accuracy) over all path frames.
    """
    ia = np.clip(ia, 0, len(f0_ref) - 1)
    ib = np.clip(ib, 0, len(f0_hyp) - 1)
    a, b = f0_ref[ia], f0_hyp[ib]
    both = (a > 0) & (b > 0)
    out = {"vuv_acc": float(np.mean((a > 0) == (b > 0)))}
    if both.sum() < 10:
        out.update(f0_corr=np.nan, f0_rmse_st=np.nan)
        return out
    sa, sb = semitones(a[both]), semitones(b[both])
    out["f0_corr"] = float(np.corrcoef(sa, sb)[0, 1])
    out["f0_rmse_st"] = float(np.sqrt(np.mean((sa - sb) ** 2)))
    return out


def harmonic_ripple(mel, bands=30, top=0.5):
    """Mean |second difference along frequency| in the lowest `bands` mel bands (log-mel).

    Harmonics of the voice show up there as a ripple across neighbouring bands; an over-smoothed
    mel has a smooth envelope and a ripple close to 0. Only the loudest `top` share of frames
    (by low-band energy) is used, so pauses and quiet consonants don't dilute the measure.
    """
    low = mel[:bands]
    e = low.mean(0)
    keep = e >= np.quantile(e, 1 - top)
    return float(np.abs(np.diff(low[:, keep], n=2, axis=0)).mean())

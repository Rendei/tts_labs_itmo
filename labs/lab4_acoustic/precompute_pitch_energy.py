"""Pre-compute the F0 and energy caches for the `--pitch-energy` ablation (lab 4, extra task 2).

    python precompute_pitch_energy.py --cache-dir ../../data/lab4_cache --workers 6
    python train_fastspeech2.py --run-name pitch_energy --pitch-energy --cache-dir ../../data/lab4_cache

coqui computes F0 with librosa's pYIN when the trainer builds the dataset; for ~17 000
utterances that takes about an hour of CPU before the first training step, during which the
GPU idles. Doing it here, separately, lets it run while another model trains. The caches are
exactly what coqui would write itself: the same `F0Dataset` / `EnergyDataset` classes, the
same AudioProcessor settings (including the 50-300 Hz F0 range the training script sets for
this male speaker) and the same per-file names, plus `pitch_stats.npy` / `energy_stats.npy`
(mean and std over voiced frames) used for normalization.
"""
import argparse
from pathlib import Path

# first: sets the one-thread-per-process BLAS limit before numpy is imported
from train_fastspeech2 import PITCH_FMAX, PITCH_FMIN, build_audio_config, make_formatter

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import EnergyDataset, F0Dataset
from TTS.utils.audio import AudioProcessor


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=Path("../../data/metadata_phones_gt_train.csv"))
    p.add_argument("--wavs", type=Path, default=Path("../../data/RUSLAN/RUSLAN"))
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio = build_audio_config()
    audio.pitch_fmin, audio.pitch_fmax = PITCH_FMIN, PITCH_FMAX
    ap = AudioProcessor.init_from_config(audio)

    dataset_config = BaseDatasetConfig(formatter="", dataset_name="ruslan",
                                       meta_file_train=str(args.metadata), path=str(args.wavs),
                                       language="ru")
    samples, _ = load_tts_samples(dataset_config, eval_split=False,
                                  formatter=make_formatter(args.wavs))
    print(f"{len(samples)} utterances -> {args.cache_dir}")

    for name, cls in (("energy", EnergyDataset), ("f0", F0Dataset)):
        path = args.cache_dir / name
        if path.exists():
            print(f"{path} exists, skipping (delete it to recompute)")
            continue
        print(f"computing {name} ...")
        cls(samples, ap, cache_path=str(path), precompute_num_workers=args.workers)
    print("done")


if __name__ == "__main__":
    main()

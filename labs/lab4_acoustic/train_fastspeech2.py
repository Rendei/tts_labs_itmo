"""Train FastSpeech2 on RUSLAN phone metadata — lab 4.

Baseline (required)::

    python train_fastspeech2.py --run-name baseline

Ablations (optional), each a flag on the same script so the baseline stays the control::

    python train_fastspeech2.py --run-name pitch_energy --pitch-energy
    python train_fastspeech2.py --run-name mfa_dur --durations ../../data/mfa_durations/manifest.csv
    python train_fastspeech2.py --run-name lr3e4 --lr 3e-4 --batch-size 16

Input is the phone metadata from lab 3 — `id|text|phones`, phones space-separated, every
phone drawn from `common/phones_mfa.txt`. The third column is what the model sees; the
second is carried along only so the file stays readable.

Several audio settings differ from coqui's defaults and are **not** free choices. They are
dictated by the pretrained HiFi-GAN that lab 5 will use, and were verified by comparing
coqui's mel against SpeechBrain's frame by frame:



"""
import os

# One BLAS/OpenMP thread per process. Every data-loader worker computes mels with numpy, and
# by default each one starts as many BLAS threads as there are cores: 8 workers x 20 threads
# on a 20-thread CPU. On Windows that oversubscription made batch loading take 5-20 s and the
# GPU sat idle (measured: 1-5 s/step -> 0.1-0.2 s/step after this). Must precede numpy import;
# spawned workers inherit the environment.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import copy
import sys
import unicodedata
from pathlib import Path

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

MULTICHAR_CLASS = "TTS.tts.utils.text.characters.MulticharPhonemes"
SAMPLE_RATE = 22050
PITCH_FMIN, PITCH_FMAX = 50.0, 300.0  # male speaker, see --pitch-energy


def fix_attention_mask_order() -> None:
    """Give every utterance its own precomputed alignment in coqui's collate_fn.

    `TTSDataset.collate_fn` sorts the batch by text length and then indexes the attention
    masks with the same permutation a second time (`batch["attn"][idx] for idx in
    ids_sorted_decreasing`, on a list that is already sorted). With --durations and shuffled
    batches each row then trains on another utterance's durations, trimmed to fit, and the
    decoder never leaves the mean-spectrum plateau. Returning the identity permutation from
    `_sort_batch` makes the second indexing a no-op; nothing else in collate_fn uses it.
    Applied at import time so that spawned DataLoader workers, which re-import this script
    but do not run main(), get it too. Without --durations the masks are None and this
    changes nothing.
    """
    import torch
    from TTS.tts.datasets.dataset import TTSDataset

    original = TTSDataset._sort_batch

    def _sort_batch(batch, text_lengths):
        batch, text_lengths, ids = original(batch, text_lengths)
        return batch, text_lengths, torch.arange(len(ids))

    TTSDataset._sort_batch = staticmethod(_sort_batch)


fix_attention_mask_order()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, default=Path("../../data/metadata_phones.csv"),
                   help="phone metadata from lab 3: id|text|phones")
    p.add_argument("--wavs", type=Path, default=Path("../../data/RUSLAN/RUSLAN"),
                   help="directory of {id}.wav at 22050 Hz")
    p.add_argument("--phones", type=Path, default=Path("../../common/phones_mfa.txt"),
                   help="the phone inventory; the model vocabulary is built from it")
    p.add_argument("--output", type=Path, default=Path("runs"),
                   help="parent directory for run directories")
    p.add_argument("--run-name", default="baseline",
                   help="subdirectory under --output; use a different one per experiment")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="coqui's Fastspeech2 default, with a NoamLR schedule")
    p.add_argument("--max-audio-seconds", type=float, default=10.0,
                   help="utterances longer than this are dropped; attention is O(n^2)")

    p.add_argument("--pitch-energy", action="store_true",
                   help="ablation: enable the pitch and energy predictors. Sets the f0 "
                        "search range for a male speaker, which the coqui defaults "
                        "(1-640 Hz) get wrong badly enough to matter")
    p.add_argument("--durations", type=Path, default=None,
                   help="ablation: a wav|npy manifest of precomputed alignments from "
                        "prepare_mfa_durations.py. Turns the internal aligner off")

    # --- additions for lab 4 experiments; defaults keep the shipped behaviour ---
    p.add_argument("--lr-schedule", choices=("epoch", "step"), default="epoch",
                   help="when NoamLR advances. 'epoch' is what the shipped config does "
                        "(the trainer's scheduler_after_epoch=True): warm-up then lasts "
                        "4000 *epochs* and the effective lr is lr*epoch/4000. 'step' is the "
                        "Transformer/FastSpeech2 schedule: peak lr at step 4000, then 1/sqrt")
    p.add_argument("--warmup-steps", type=int, default=4000, help="NoamLR warm-up")
    p.add_argument("--precision", choices=("fp32", "tf32", "bf16", "fp16"), default="fp32",
                   help="tf32/bf16/fp16 use the tensor cores of Ampere+ GPUs; the audio "
                        "pipeline is untouched")
    p.add_argument("--init-from", type=Path, default=None,
                   help="initialize the weights from this checkpoint (strict=False, so new "
                        "modules such as pitch/energy predictors start fresh and a fresh "
                        "optimizer is used), as suggested by the README for the ablations")
    p.add_argument("--num-workers", type=int, default=8, help="data loader workers")
    p.add_argument("--save-step", type=int, default=10000, help="checkpoint every N steps")
    p.add_argument("--keep-checkpoints", type=int, default=5,
                   help="how many periodic checkpoints to keep (comparisons at equal steps "
                        "need the older ones)")
    p.add_argument("--shuffle", action="store_true",
                   help="shuffle the training batches. coqui's default (shuffle=False, "
                        "batch_group_size=0) feeds every epoch in the same order, sorted from "
                        "the shortest utterance to the longest")
    p.add_argument("--gpu-mem-fraction", type=float, default=None,
                   help="cap PyTorch's share of GPU memory (e.g. 0.8 on an 8 GB card). Without a "
                        "cap the caching allocator fragments past the card's size and the "
                        "Windows driver silently spills to system RAM, several times slower")
    p.add_argument("--persistent-workers", action="store_true",
                   help="keep data-loader workers alive between epochs. On Windows workers are "
                        "spawned, not forked: re-creating 6 of them costs ~45 s every epoch")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="shared F0/energy cache from precompute_pitch_energy.py; by default "
                        "coqui computes it inside the run directory before the first step")
    p.add_argument("--continue-path", type=Path, default=None,
                   help="resume a finished or interrupted run from its newest checkpoint in this "
                        "folder: weights, optimizer, scheduler and step counter. The run keeps its "
                        "own config.json; only --epochs is taken from the command line, so the "
                        "other flags must repeat the original ones (they still build the data)")
    return p.parse_args()


def load_phone_inventory(path: Path) -> list[str]:
    """Read one phone per line, NFC-normalized.

    NFC matters: `s̪` is two codepoints and `t̪s̪` is four. An NFD spelling here against
    NFC in the metadata compares unequal, every lookup misses, and the only symptom is a
    model that trains on unknown tokens.
    """
    phones = [unicodedata.normalize("NFC", line.strip())
              for line in path.read_text(encoding="utf-8").splitlines()]
    phones = [p for p in phones if p]
    if not phones:
        msg = f"No phones read from {path}"
        raise ValueError(msg)
    return phones


def make_formatter(wav_dir: Path):
    """Build a coqui sample formatter over `id|text|phones`.

    Column 3 is the model input. Rows whose wav is missing are skipped and counted rather
    than raising, because a partially downloaded corpus is the common case.
    """

    def formatter(root_path, meta_file, **kwargs):  # noqa: ARG001 - coqui's signature
        items = []
        skipped = 0
        with Path(meta_file).open(encoding="utf-8") as fh:
            for line in fh:
                cols = line.rstrip("\n").split("|")
                if len(cols) < 3:
                    skipped += 1
                    continue
                utt_id, _text, phones = cols[0], cols[1], cols[2]
                wav_path = wav_dir / f"{utt_id}.wav"
                if not wav_path.is_file():
                    skipped += 1
                    continue
                items.append({
                    "text": phones,
                    "audio_file": str(wav_path),
                    "speaker_name": "ruslan",
                    "root_path": str(wav_dir),
                })
        if skipped:
            print(f"  formatter skipped {skipped} rows")
        return items

    return formatter


def build_audio_config() -> BaseAudioConfig:
    """The audio config the lab 5 vocoder dictates. See the module docstring."""
    return BaseAudioConfig(
        sample_rate=SAMPLE_RATE, # Required for vocoder
        fft_size=1024,           # Required for vocoder
        win_length=1024,         # Required for vocoder
        hop_length=256,          # Required for vocoder
        num_mels=80,             # Required for vocoder
        mel_fmin=0.0,            # Required for vocoder
        mel_fmax=8000.0,         # Required for vocoder
        spec_gain=1.0,           # Required for vocoder 
        log_func="np.log",       # Required for vocoder
        signal_norm=False,       # Required for vocoder
        preemphasis=0.0,
        ref_level_db=20,
        do_trim_silence=False,   # Required for external durations
        trim_db=60.0,
    )



def enable_persistent_workers() -> None:
    """Make coqui's data loaders keep their worker processes between epochs.

    coqui builds `torch.utils.data.DataLoader` inside `BaseTTS.get_data_loader` without
    `persistent_workers`, so every epoch tears the workers down and starts new ones. The
    trainer creates each loader once per run, so swapping the class it calls is enough.
    """
    import TTS.tts.models.base_tts as base_tts

    original = base_tts.DataLoader

    def loader(*args, **kwargs):
        if kwargs.get("num_workers", 0) > 0:
            kwargs.setdefault("persistent_workers", True)
        return original(*args, **kwargs)

    base_tts.DataLoader = loader
    print("data loader workers: persistent")


def preflight(tokenizer, samples, n: int = 200) -> None:
    """Abort before the first epoch if any phone is missing from the vocabulary.

    Without this the run starts, spends the data-loading pass mapping unknown phones to a
    single id, and trains to convergence on corrupted input. The failure has no symptom
    until synthesis sounds wrong hours later.
    """
    print("\n--- tokenizer pre-flight ---")
    print(f"vocabulary size: {tokenizer.characters.num_chars}")
    probe = samples[0]["text"]
    ids = tokenizer.text_to_ids(probe)
    print(f"sample phones  : {probe[:70]}...")
    print(f"sample ids     : {ids[:12]}... ({len(ids)} ids for {len(probe.split())} phones)")
    if len(ids) != len(probe.split()):
        print("  MISMATCH: id count != phone count. Tokens are being dropped.", file=sys.stderr)

    for item in samples[:n]:
        tokenizer.text_to_ids(item["text"])
    if tokenizer.not_found_characters:
        print(f"\n  ABORT: {len(tokenizer.not_found_characters)} tokens are not in the "
              f"vocabulary:", file=sys.stderr)
        for token in tokenizer.not_found_characters[:20]:
            codepoints = " ".join(f"U+{ord(c):04X}" for c in token)
            print(f"    {token!r}  {codepoints}", file=sys.stderr)
        print("\n  Most likely your phone metadata and phones_mfa.txt disagree, or one "
              "side is NFD and the other NFC.", file=sys.stderr)
        sys.exit(1)
    print(f"checked {min(n, len(samples))} utterances, no unknown tokens")
    print("--- pre-flight ok ---\n")


def main() -> None:
    """Build the config from the flags, check the data, and hand over to the trainer."""
    args = parse_args()
    output_path = args.output / args.run_name
    output_path.mkdir(parents=True, exist_ok=True)

    phones = load_phone_inventory(args.phones)
    print(f"phone inventory: {len(phones)} types from {args.phones}")

    dataset_config = BaseDatasetConfig(
        formatter="",  # unused: a callable goes to load_tts_samples below
        dataset_name="ruslan",
        meta_file_train=str(args.metadata),
        path=str(args.wavs),
        language="ru",
    )
    if args.durations is not None:
        # use_aligner=False makes the model read durations from here instead of learning
        # an alignment. The manifest is wav_path|npy_path, one line per utterance.
        dataset_config.meta_file_attn_mask = str(args.durations)
        print(f"durations: precomputed, from {args.durations}")
    else:
        print("durations: learned by the internal aligner")

    model_args = ForwardTTSArgs(
        use_pitch=args.pitch_energy,
        use_energy=args.pitch_energy,
        use_aligner=args.durations is None,
    )

    config = Fastspeech2Config(
        run_name=args.run_name,
        audio=build_audio_config(),
        batch_size=args.batch_size,
        eval_batch_size=max(2, args.batch_size // 2),
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=min(4, args.num_workers),
        model_args=model_args,
        lr=args.lr,
        compute_input_seq_cache=True,
        compute_f0=args.pitch_energy,
        f0_cache_path=str((args.cache_dir or output_path) / ("f0" if args.cache_dir else "f0_cache")),
        compute_energy=args.pitch_energy,
        energy_cache_path=str((args.cache_dir or output_path) / ("energy" if args.cache_dir else "energy_cache")),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=args.epochs,
        use_phonemes=False,
        text_cleaner="multicharacter_phoneme_cleaners",
        characters=CharactersConfig(
            characters_class=MULTICHAR_CLASS,
            characters=phones,
            punctuations="",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            is_unique=False,
            is_sorted=True,
        ),
        precompute_num_workers=4,
        print_step=50,
        print_eval=False,
        mixed_precision=args.precision in ("bf16", "fp16"),  # a 1080Ti is Pascal; fp16 buys nothing there
        precision=args.precision if args.precision in ("bf16", "fp16") else "fp16",
        allow_tf32=args.precision != "fp32",
        scheduler_after_epoch=args.lr_schedule == "epoch",
        lr_scheduler_params={"warmup_steps": args.warmup_steps},
        save_step=args.save_step,
        save_n_checkpoints=args.keep_checkpoints,
        shuffle=args.shuffle,
        max_seq_len=200,
        max_audio_len=int(args.max_audio_seconds * SAMPLE_RATE),
        output_path=str(output_path),
        datasets=[dataset_config],
    )

    if args.pitch_energy:
        # The coqui defaults are pitch_fmin=1.0 and pitch_fmax=640.0. For a male speaker
        # at roughly 70-200 Hz that window is wrong in both directions: it admits octave
        # errors at the top and DC at the bottom, and it makes librosa.pyin crawl through
        # the f0 cache because the search range is nine times wider than it needs to be.
        config.audio.pitch_fmin = PITCH_FMIN
        config.audio.pitch_fmax = PITCH_FMAX
        print(f"f0 range: {config.audio.pitch_fmin}-{config.audio.pitch_fmax} Hz "
              f"(coqui's 1-640 default is wrong for this speaker)")

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=make_formatter(args.wavs),
    )
    print(f"train: {len(train_samples)}   eval: {len(eval_samples)}")

    preflight(tokenizer, train_samples)

    # Test sentences go through this same tokenizer, so they must be phone strings rather
    # than Russian text. Taking them from the eval split guarantees every token is known.
    config.test_sentences = [item["text"] for item in eval_samples[:3]]

    print(f"lr {args.lr:g}, NoamLR warm-up {args.warmup_steps}, advanced per {args.lr_schedule}; "
          f"precision {args.precision}")

    import torch
    if args.gpu_mem_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction)
        print(f"GPU memory capped at {args.gpu_mem_fraction:.0%} of "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print(f"shuffle batches: {args.shuffle}")
    if args.persistent_workers:
        enable_persistent_workers()

    model = ForwardTTS(config, ap, tokenizer, speaker_manager=None)
    if args.init_from is not None:
        state = torch.load(str(args.init_from), map_location="cpu")["model"]
        own = model.state_dict()
        # drop tensors whose shape changed; strict=False alone does not forgive that
        state = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        result = model.load_state_dict(state, strict=False)
        print(f"init from {args.init_from}: loaded {len(state)} tensors, "
              f"{len(result.missing_keys)} left at random init")
        for key in result.missing_keys[:20]:
            print(f"    fresh: {key}")
    trainer_args = TrainerArgs()
    characters = copy.deepcopy(config.characters)   # the continue path overwrites config in place
    if args.continue_path is not None:
        trainer_args.continue_path = str(args.continue_path)
    trainer = Trainer(
        trainer_args,
        config,
        str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    if args.continue_path is not None:
        # the trainer has replaced the config with the run's config.json, epochs included;
        # extending a finished run needs the new total, and config.json should record it.
        # Coqpit cannot read the phone list back from JSON (a list where it expects a str)
        # and sets it to None; the tokenizer is already built, but evaluate.py reads the
        # list from config.json, so put it back.
        trainer.config.epochs = args.epochs
        trainer.config.characters = characters
        trainer.config.save_json(str(Path(trainer.output_path) / "config.json"))
        print(f"continuing {args.continue_path} from step {trainer.total_steps_done - 1}, "
              f"epoch {trainer.epochs_done}, up to epoch {args.epochs}")
    trainer.fit()


if __name__ == "__main__":
    main()

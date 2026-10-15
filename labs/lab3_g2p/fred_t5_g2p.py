"""FRED-T5 as a G2P model — lab 3, extra task 3 (run in the separate CUDA environment).

FRED-T5 (ai-forever) is a Russian T5 language model; it has no transcription head, so it is adapted: the
large model is fine-tuned with LoRA to turn a word into its phones (the target alphabet written as
space-separated one-symbol codes of the target phones). It sees the same training words as our own OOV model and Phonetisaurus, and is
decoded on the same held-out words and on the remainder of `test`::

    python fred_t5_g2p.py data/external/phonetisaurus_train.dict data/external/phonetisaurus_words.txt \
        data/external/out_fred_t5.csv [max_minutes] [n_train] [model dir or hub id]
"""
import csv
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL = 'ai-forever/FRED-T5-large'
PREFIX = '<LM>'
PHONES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'common', 'phones_mfa.txt')
SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def phone_codes() -> tuple[dict[str, str], dict[str, str]]:
    """One ASCII symbol per target phone: the byte-level BPE of FRED-T5 spends 3-4 tokens on a single IPA
    character, which makes the targets four times longer and the model slower to train."""
    with open(PHONES_FILE, encoding='utf-8') as fh:
        phones = [p for p in fh.read().split() if p != '<SIL>']
    to_symbol = {p: SYMBOLS[i] for i, p in enumerate(phones)}
    return to_symbol, {v: k for k, v in to_symbol.items()}


def read_dict(path: str) -> list[tuple[str, str]]:
    with open(path, encoding='utf-8') as fh:
        return [tuple(line.rstrip('\n').split('\t')) for line in fh if line.strip()]


def main(argv: list[str]) -> None:
    train_path, words_path, out_path = argv[1:4]
    max_minutes = float(argv[4]) if len(argv) > 4 else 30.0
    n_train = int(argv[5]) if len(argv) > 5 else 200000
    model_id = argv[6] if len(argv) > 6 else MODEL
    torch.manual_seed(0)
    rng = random.Random(0)
    to_symbol, from_symbol = phone_codes()
    pairs = [(w, ' '.join(to_symbol[p] for p in ph.split())) for w, ph in read_dict(train_path)]
    pairs = rng.sample(pairs, min(n_train, len(pairs)))
    print(f'{len(pairs)} training pairs', flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, eos_token='</s>')
    model = T5ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type='SEQ_2_SEQ_LM',
                                             target_modules=['q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo']))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-4, weight_decay=0.0)

    batch_size = 32
    max_lr, warmup = 4e-4, 50
    model.train()
    t0 = time.time()
    steps = len(pairs) // batch_size
    for step in range(steps):
        elapsed = (time.time() - t0) / 60
        if elapsed > max_minutes:
            print('time budget reached, stopping', flush=True)
            break
        # linear warm-up, then decay to zero over the time budget (the speed of the GPU is not known in advance)
        lr = max_lr * min(1.0, (step + 1) / warmup) * max(0.02, 1.0 - elapsed / max_minutes)
        for group in optimizer.param_groups:
            group['lr'] = lr
        batch = pairs[step * batch_size:(step + 1) * batch_size]
        enc = tokenizer([PREFIX + w for w, _ in batch], return_tensors='pt', padding=True).to('cuda')
        lab = tokenizer([p + tokenizer.eos_token for _, p in batch], return_tensors='pt', padding=True).input_ids.to('cuda')
        lab[lab == tokenizer.pad_token_id] = -100
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0:
            print(f'step {step}/{steps} loss {loss.item():.4f} lr {lr:.2e} {time.time() - t0:.0f}s', flush=True)

    model.eval()
    with open(words_path, encoding='utf-8') as fh:
        words = [w.strip() for w in fh if w.strip()]
    out = []
    for start in range(0, len(words), 128):
        chunk = words[start:start + 128]
        enc = tokenizer([PREFIX + w for w in chunk], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            ids = model.generate(**enc, max_new_tokens=40, num_beams=1, eos_token_id=tokenizer.eos_token_id)
        out += [' '.join(from_symbol.get(c, '') for c in s.split()) for s in tokenizer.batch_decode(ids, skip_special_tokens=True)]
        if start % 1280 == 0:
            print(f'decoded {start + len(chunk)}/{len(words)}', flush=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh, delimiter='|', quoting=csv.QUOTE_NONE, escapechar='\\')
        for w, p in zip(words, out):
            writer.writerow([w, p])
    model.save_pretrained(os.path.join(os.path.dirname(out_path), 'fred_t5_lora'))
    print(f'done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main(sys.argv)

#!/usr/bin/env python3
"""
stream_train.py
===============
Keep pretraining the baby's language model on TinyStories (a pool of ~2.1M stories; the default
100M tokens reads about 450k of them) without storing them: stories are streamed from Hugging Face, tokenized on the fly, learned from once,
and discarded. Only a tiny held-out validation set is kept on disk (data/val_stream.pt, ~1 MB).

Why this helps the baby learn instead of memorize: it never sees the same story twice.

Healthy by design:
  - Continues from the current brain (checkpoints/baby_model_best.pt) instead of starting over.
  - The new weights replace the old ones only when they score better on stories neither has seen.
  - Pause any time with Ctrl+C; run the same command again to resume where it stopped.
  - If the internet drops, it saves, waits, and carries on.

Usage:
  python train.py --stream                    # 100M tokens (~13 h on a laptop CPU)
  python train.py --stream --tokens 300M      # a longer, fuller pass
"""

import os
import sys
import math
import time
from typing import Iterator, List, Optional

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch
from tokenizers import Tokenizer

from train import (BabyGPT, BabyGPTConfig, PretokenizedTensorDataset, configure_optimizers, evaluate,
                   generate_sample, get_lr_scheduler, save_checkpoint)

DATASET = "roneneldan/TinyStories"
START, END = "<|im_start|>", "<|im_end|>"


def parse_tokens(text: str) -> int:
    """'100M' -> 100_000_000, '1.5B' -> 1_500_000_000."""
    text = str(text).strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1], 1)
    return int(float(text.rstrip("KMB")) * mult)


def clean(text: str) -> str:
    from prepare_baby_data import clean_story
    return clean_story(text)


def story_stream(skip: int, seed: int) -> Iterator[str]:
    """Shuffled TinyStories train split, resumable by skipping the stories already learned."""
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    if skip:
        ds = ds.skip(skip)
    for record in ds:
        text = clean(record.get("text", ""))
        if len(text) >= 30:
            yield text


def validation_tokens(tokenizer: Tokenizer, path: str, n_stories: int = 400) -> torch.Tensor:
    """Held-out stories from the official validation split, cached once (~1 MB)."""
    if os.path.exists(path):
        return torch.load(path, weights_only=True)
    from datasets import load_dataset
    print(f"[*] Fetching {n_stories} held-out validation stories (one time)...")
    ds = load_dataset(DATASET, split="validation", streaming=True)
    texts = []
    for record in ds:
        text = clean(record.get("text", ""))
        if len(text) >= 30:
            texts.append(f"{START}\n{text}\n{END}")
        if len(texts) >= n_stories:
            break
    ids = [i for enc in tokenizer.encode_batch(texts) for i in enc.ids]
    t = torch.tensor(ids, dtype=torch.long).to(torch.uint16)
    torch.save(t, path)
    return t


class Packer:
    """Turns a stream of stories into (batch, seq_len+1) token blocks, counting stories used."""

    def __init__(self, stories: Iterator[str], tokenizer: Tokenizer, batch: int, seq: int):
        self.stories, self.tok, self.batch, self.seq = stories, tokenizer, batch, seq
        self.buffer: List[int] = []
        self.used = 0  # stories fully consumed into emitted batches (approximately)

    def next_batch(self) -> torch.Tensor:
        need = self.batch * (self.seq + 1)
        while len(self.buffer) < need:
            chunk = []
            for text in self.stories:
                chunk.append(f"{START}\n{text}\n{END}")
                if len(chunk) == 64:
                    break
            if not chunk:
                raise StopIteration
            for enc in self.tok.encode_batch(chunk):
                self.buffer.extend(enc.ids)
            self.used += len(chunk)
        block, self.buffer = self.buffer[:need], self.buffer[need:]
        return torch.tensor(block, dtype=torch.long).view(self.batch, self.seq + 1)


def no_dropout(model: BabyGPT):
    # Dropout protects against memorizing repeated data. A stream never repeats, so it only costs time.
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0


def train_stream(init: str = "checkpoints/baby_model_best.pt", out: str = "checkpoints/baby_model_best.pt",
                 tokenizer_path: str = "tokenizer/tokenizer.json", tokens: str = "100M", batch_size: int = 16,
                 seq_len: int = 512, lr: float = 2e-4, warmup: int = 500, eval_interval: int = 500,
                 threads: int = 0, seed: int = 42, resume_path: str = "checkpoints/stream_resume.pt",
                 val_path: str = "data/val_stream.pt"):
    torch.manual_seed(seed)
    torch.set_num_threads(threads or max(1, (os.cpu_count() or 2) // 2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    total_tokens = parse_tokens(tokens)
    max_steps = math.ceil(total_tokens / (batch_size * seq_len))

    resume = torch.load(resume_path, map_location=device, weights_only=False) if os.path.exists(resume_path) else None
    if resume:
        model = BabyGPT(BabyGPTConfig(**resume["config"])).to(device)
        model.load_state_dict(resume["model"])
        max_steps = resume["max_steps"]
        print(f"[*] Resuming: step {resume['step']:,}/{max_steps:,}, {resume['stories']:,} stories learned so far")
    else:
        model = BabyGPT.from_checkpoint(init, device=str(device))
        print(f"[*] Continuing from {init} ({model.count_parameters()['total'] / 1e6:.1f}M params)")
    no_dropout(model)
    model.train()

    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup, max_steps=max_steps)
    step, stories_done = 0, 0
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        step, stories_done, best_val = resume["step"], resume["stories"], resume["best_val"]

    val = validation_tokens(tokenizer, val_path)
    val_loader = torch.utils.data.DataLoader(PretokenizedTensorDataset(val, seq_len=seq_len), batch_size=batch_size)
    amp = torch.amp.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    if not resume:
        # the bar to beat: the current brain on held-out stories
        best_val, _ = evaluate(model, val_loader, device, amp, max_eval_batches=50)
        model.train()
    print(f"[*] Plan: {max_steps:,} steps = {total_tokens / 1e6:,.0f}M tokens | held-out loss to beat: {best_val:.4f}")
    print("[*] Ctrl+C pauses safely; run the same command again to resume.\n")

    def save_resume():
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "config": model.cfg.__dict__, "step": step, "stories": stories_done + packer.used,
                    "best_val": best_val, "max_steps": max_steps}, resume_path)

    packer = Packer(story_stream(stories_done, seed), tokenizer, batch_size, seq_len)
    t0, t_log, failures = time.time(), time.time(), 0
    try:
        while step < max_steps:
            try:
                block = packer.next_batch().to(device)
                failures = 0
            except StopIteration:
                print("[✓] Reached the end of TinyStories.")
                break
            except Exception as e:  # network hiccup: save, wait, reconnect where we were
                failures += 1
                if failures > 10:
                    raise
                save_resume()
                print(f"\n[!] Stream error ({type(e).__name__}); retrying in 30s ({failures}/10)...")
                time.sleep(30)
                stories_done += packer.used
                packer = Packer(story_stream(stories_done, seed), tokenizer, batch_size, seq_len)
                continue

            with amp:
                _, loss = model(block[:, :-1].contiguous(), block[:, 1:].contiguous())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % 10 == 0:
                rate = 10 * batch_size * seq_len / (time.time() - t_log)
                eta = (max_steps - step) * batch_size * seq_len / max(rate, 1) / 3600
                print(f"step {step:,}/{max_steps:,} | loss {loss.item():.3f} | {rate:,.0f} tok/s | "
                      f"stories {stories_done + packer.used:,} | eta {eta:.1f}h", end="\r", flush=True)
                t_log = time.time()

            if step % eval_interval == 0 or step == max_steps:
                val_loss, ppl = evaluate(model, val_loader, device, amp, max_eval_batches=50)
                model.train()
                note = ""
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, out, step=step, val_loss=val_loss, val_ppl=ppl)
                    note = f"  [★ better than before: saved to {out}]"
                print(f"\n── step {step:,}: held-out loss {val_loss:.4f} (ppl {ppl:.2f}){note}")
                print("   sample: " + generate_sample(model, tokenizer, f"{START}\nOne day, a little bunny", device,
                                                     max_new_tokens=40).replace("\n", " ")[len(START) + 1:][:160])
                model.train()
                save_resume()
    except KeyboardInterrupt:
        save_resume()
        print(f"\n[⏸] Paused at step {step:,}. Run the same command again to continue.")
        return False

    if os.path.exists(resume_path):
        os.remove(resume_path)
    print(f"\n[✓] Done in {(time.time() - t0) / 3600:.1f} h. Learned from {stories_done + packer.used:,} stories "
          f"without storing them. Best held-out loss {best_val:.4f}.")
    return True

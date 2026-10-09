#!/usr/bin/env python3
"""
sft.py
======
Stage 2: turn the pretrained story model (checkpoints/baby_model_best.pt) into a baby that
converses and reasons over what it has been taught.

  - Infinite stream of freshly generated dialogues (sft_data.py): no example is ever seen
    twice, so there is nothing to memorize. Only the skill of reading memory and context generalizes.
  - Loss only on the baby's replies (caregiver/system tokens are masked with -100).
  - Story replay every few steps keeps the language ability from pretraining.
  - Dynamic padding: batches are only as long as their longest dialogue (~140 tokens avg),
    which makes CPU fine-tuning practical.
  - Evaluation on a HELD-OUT vocabulary (unseen names, invented words built from different
    syllables). Seen-vs-held-out accuracy gap = how much is memorized vs understood.

Output: checkpoints/baby_chat.pt (weights only), used automatically by chat.py / app.py.

Usage:
  python sft.py                         # ~defaults tuned for a laptop CPU
  python sft.py --steps 3000 --batch-size 24
  python sft.py --eval-only --init checkpoints/baby_chat.pt
"""

import os
import sys
import math
import time
import argparse
from typing import List

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from tokenizers import Tokenizer

from train import BabyGPT, PretokenizedTensorDataset, configure_optimizers, get_lr_scheduler, save_checkpoint
from sft_data import DialogueGenerator, encode_dialogue, format_chatml

PAD_ID = 1
STOP_ID = 3  # <|im_end|>


def collate(samples, pad_id: int = PAD_ID):
    T = max(len(x) for x, _ in samples)
    X = torch.full((len(samples), T), pad_id, dtype=torch.long)
    Y = torch.full((len(samples), T), -100, dtype=torch.long)
    for i, (x, y) in enumerate(samples):
        X[i, :len(x)] = torch.tensor(x)
        Y[i, :len(y)] = torch.tensor(y)
    return X, Y


def reply_passes(category: str, keys: List[str], reply: str) -> bool:
    r = reply.lower()
    if category == "unknown":
        return "know" in r and ("don't" in r or "not sure" in r)
    if keys and keys[0] in ("yes", "no"):
        return r.startswith(keys[0]) and all(k.lower() in r for k in keys[1:])
    return all(k.lower() in r for k in keys)


@torch.no_grad()
def evaluate_reasoning(model, tokenizer, split: str, n: int, seed: int, device, show: int = 0):
    """Generate the last baby reply of n dialogues (greedy) and grade it by category."""
    model.eval()
    gen = DialogueGenerator(split, seed=seed)
    per_cat = {}
    shown = 0
    done = 0
    while done < n:
        msgs, checks = gen.sample()
        cat, keys, gold = checks[-1]
        if cat in ("chat", "care", "teach") or not keys and cat != "unknown":
            continue  # grade only turns with a checkable right answer
        prompt = format_chatml(msgs[:-1], add_generation_prompt=True)
        ids = tokenizer.encode(prompt).ids
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=40, temperature=0, stop_token_id=STOP_ID)
        reply = tokenizer.decode(out[0, len(ids):].tolist(), skip_special_tokens=True).strip()
        ok = reply_passes(cat, keys, reply)
        hit, tot = per_cat.get(cat, (0, 0))
        per_cat[cat] = (hit + ok, tot + 1)
        done += 1
        if shown < show:
            shown += 1
            print(f"    [{cat}{' ✓' if ok else ' ✗'}] Q: {msgs[-2]['content']!r}\n        baby: {reply!r}\n        gold: {gold!r}")
    total = sum(h for h, _ in per_cat.values()) / max(1, sum(t for _, t in per_cat.values()))
    model.train()
    return total, {k: h / t for k, (h, t) in per_cat.items()}


def fmt_cats(cats):
    return "  ".join(f"{k}={v:.0%}" for k, v in sorted(cats.items()))


def main():
    ap = argparse.ArgumentParser(description="Teach the baby to converse and reason over its memory")
    ap.add_argument("--init", default="checkpoints/baby_model_best.pt", help="Pretrained checkpoint to start from")
    ap.add_argument("--out", default="checkpoints/baby_chat.pt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--stories", default="data/train.pt", help="Story tokens for replay ('' to disable)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--replay-every", type=int, default=5, help="Every Nth step trains on stories instead")
    ap.add_argument("--replay-len", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-n", type=int, default=80)
    ap.add_argument("--final-eval-n", type=int, default=300)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--keep-better", action="store_true",
                    help="Train into a candidate file; replace --out only if the candidate scores higher on new words")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU threads (physical cores is usually fastest)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    model = BabyGPT.from_checkpoint(args.init, device=str(device))
    print(f"[*] Loaded {args.init} ({model.count_parameters()['total'] / 1e6:.1f}M params) on {device}")

    if args.eval_only:
        for split in ("train", "eval"):
            acc, cats = evaluate_reasoning(model, tokenizer, split, args.final_eval_n, 1234, device, show=6)
            print(f"[{split:5s} vocab] accuracy {acc:.1%}  |  {fmt_cats(cats)}")
        return

    # Dropout fights memorization of *repeated* data. This stream never repeats, so dropout
    # only costs time (its random masks were ~30% of each CPU step). Turn it off.
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if hasattr(mod, "dropout") and isinstance(mod.dropout, float):
            mod.dropout = 0.0

    model.train()
    optimizer = configure_optimizers(model, weight_decay=0.05, learning_rate=args.lr)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=args.warmup, max_steps=args.steps)
    amp = torch.amp.autocast("cuda", dtype=torch.float16) if use_cuda else torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    stories = None
    if args.stories and os.path.exists(args.stories) and args.replay_every > 0:
        stories = PretokenizedTensorDataset(args.stories, seq_len=args.replay_len, random_offsets=True)
        print(f"[*] Story replay: every {args.replay_every} steps, {stories.data.numel():,} tokens available")

    gen = DialogueGenerator("train", seed=args.seed)
    acc0, cats0 = evaluate_reasoning(model, tokenizer, "eval", 40, 99, device)
    print(f"[*] Before SFT, held-out reasoning accuracy: {acc0:.0%}  ({fmt_cats(cats0)})")

    live_out = args.out
    if args.keep_better and os.path.exists(live_out):
        args.out = live_out + ".candidate"  # the current brain stays untouched until the new one proves better
    best = -1.0
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    for step in range(1, args.steps + 1):
        if stories is not None and step % args.replay_every == 0:
            batch = [stories[0] for _ in range(max(4, args.batch_size // 2))]
            X = torch.stack([b[0] for b in batch])
            Y = torch.stack([b[1] for b in batch])
            is_replay = True
        else:
            X, Y = collate([encode_dialogue(gen.sample()[0], tokenizer, args.max_len) for _ in range(args.batch_size)])
            is_replay = False
        X, Y = X.to(device), Y.to(device)

        with amp:
            _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if not is_replay:
            run_loss += loss.item()
            run_n += 1
        if step % 25 == 0:
            el = time.time() - t0
            eta = el / step * (args.steps - step)
            print(f"step {step:5d}/{args.steps} | chat loss {run_loss / max(1, run_n):.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.1e} | {el / step:.2f}s/step | eta {eta / 60:.1f}m", flush=True)
            run_loss, run_n = 0.0, 0

        if step % args.eval_every == 0 or step == args.steps:
            acc, cats = evaluate_reasoning(model, tokenizer, "eval", args.eval_n, 1234, device)
            mark = ""
            if acc > best:
                best = acc
                save_checkpoint(model, args.out, step=step, heldout_accuracy=acc)
                mark = "  [★ saved]"
            print(f"  ── held-out reasoning {acc:.1%}{mark}  |  {fmt_cats(cats)}")

    print("\n" + "=" * 72)
    print("  FINAL EVALUATION (best checkpoint)")
    print("=" * 72)
    model = BabyGPT.from_checkpoint(args.out, device=str(device))
    results = {}
    for split in ("train", "eval"):
        acc, cats = evaluate_reasoning(model, tokenizer, split, args.final_eval_n, 4321, device, show=4 if split == "eval" else 0)
        results[split] = acc
        print(f"  [{'seen words' if split == 'train' else 'NEW words '}] accuracy {acc:.1%}  |  {fmt_cats(cats)}")
    gap = results["train"] - results["eval"]
    print(f"\n  Generalization gap: {gap:+.1%}  (small gap = reasoning over memory, not memorizing)")
    if args.out != live_out:
        current = BabyGPT.from_checkpoint(live_out, device=str(device))
        old_acc, _ = evaluate_reasoning(current, tokenizer, "eval", args.final_eval_n, 4321, device)
        print(f"\n  Current brain on the same exam: {old_acc:.1%}  |  new brain: {results['eval']:.1%}")
        if results["eval"] > old_acc:
            os.replace(args.out, live_out)
            print(f"  [✓] The new brain is better: it is now {live_out}")
        else:
            os.remove(args.out)
            print(f"  [=] Not better, so the current brain stays. Nothing was changed.")
        args.out = live_out
    print(f"  Brain: {args.out}  ({os.path.getsize(args.out) / 1e6:.0f} MB)  in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

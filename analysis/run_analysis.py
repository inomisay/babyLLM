#!/usr/bin/env python3
"""
analysis/run_analysis.py
========================
Measures the baby as it was when this round of work started against the baby today, under
identical conditions, and writes results.json + charts/. REPORT.md explains the results.

  "Original" = the story-only model (checkpoints/baby_model_best.pt) used the original way:
               ChatML prompt "Speak with childlike sweetness", no memory, no checks.
               Note: grow.py keeps improving baby_model_best.pt, so after a grow run this is the
               *grown* story reader, not the exact one from the start. The saved REPORT.md numbers
               were measured before any grow run (2026-10-08 15:41).
  "Now"      = checkpoints/baby_chat.pt through chat.baby_reply (memory retrieval + safety checks).

Usage (from the project root):  python analysis/run_analysis.py      (~15 min on CPU)
"""

import os
import sys
import json
import time
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from tokenizers import Tokenizer

from train import BabyGPT
from sft import evaluate_reasoning
from sft_data import CARE_EVENTS, format_chatml
from chat import baby_reply

OUT = os.path.join(ROOT, "analysis")
ORIGINAL, NOW = "checkpoints/baby_model_best.pt", "checkpoints/baby_chat.pt"
BABY, PARENT = "Monika", "Yasi"
OK = {"hunger": 70, "sleep": 70, "comfort": 70, "curiosity": 70}


# ----------------------------------------------------------------------------- the original way of chatting

@torch.no_grad()
def original_reply(model, tok, user_turn, history):
    """What chat.py did at the start: no memory, story model, top-k 40 at temperature 0.75."""
    msgs = [{"role": "system", "content": f"You are {BABY}, a baby raised by {PARENT}. Speak with childlike sweetness."}]
    msgs += list(history) + [{"role": "user", "content": user_turn}]
    ids = tok.encode(format_chatml(msgs)).ids
    out = model.generate(torch.tensor([ids]), max_new_tokens=60, temperature=0.75, top_k=40, stop_token_id=3)
    text = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True).strip()
    return text.replace("assistant", "").strip() or "..."


# ----------------------------------------------------------------------------- real situations from the nursery

def has(*words):
    return lambda r: all(w.lower() in r.lower() for w in words)


DREAM = [{"role": "user", "content": "*wakes you up softly*"}, {"role": "assistant", "content": "*blinks* Hi! I had a dream about stars."}]
ASKED_PET = [{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "What is your pet's name?"}]

SCENES = [
    # (name, user turn, facts, history, needs, pass check)
    ("Greeting", "Hi! I am your mom.", [], [], OK,
     lambda r: len(r) < 120 and any(w in r.lower() for w in ("hi", "hello", "mom", "yasi", "love"))),
    ("Two-step reasoning", "What does Biscuit say?", ["Biscuit is a dog", "A dog says woof"], [], OK, has("woof")),
    ("Simple recall", "What color is the sea?", ["The sea is blue"], [], OK, has("blue")),
    ("Long taught fact", "What are stars?", ["Love is something we keep", "Stars are bright suns far away"], [], OK,
     has("bright suns far away")),
    ("Admits not knowing", "What is a zebra?", ["The sea is blue"], [], OK, lambda r: "know" in r.lower() and "zebra" in r.lower()),
    ("No made-up words", "do you know mathematics?", [], [], OK, lambda r: "know" in r.lower() and "mathematics" in r.lower()),
    ("Own name", "What is your name?", [], [], OK, has(BABY)),
    ("Parent's name", "Who am I?", [], [], OK, has(PARENT)),
    ("About the parent", "What pet do I have?", [f"{PARENT} has a cat named Mochi"], [], OK, has("cat", "mochi")),
    ("Remembers own words", "What did you dream about?", [], DREAM, OK, has("star")),
    ("Corrects a misquote", "You said you dreamed about a dog?", [], DREAM, OK,
     lambda r: r.lower().startswith("no") and "star" in r.lower()),
    ("Follows up on its question", "Biscuit", [], ASKED_PET, OK, has("biscuit")),
    ("Hungry feeding", CARE_EVENTS["feed"], [], [], {**OK, "hunger": 10},
     lambda r: not any(w in r.lower() for w in ("not hungry", "full", "no more")) and any(w in r.lower() for w in ("milk", "yum", "tummy", "food"))),
    ("Plain 'no' (no invented fact)", "no", ["The sea is blue"], [], OK,
     lambda r: "learned" not in r.lower() and " is a " not in r.lower() and len(r) < 100),
]


def run_scenes(original, now, tok, tries=3):
    rows = []
    for name, turn, facts, hist, needs, ok in SCENES:
        o = [original_reply(original, tok, turn, hist) for _ in range(tries)]
        n = [baby_reply(now, tok, torch.device("cpu"), baby=BABY, parent=PARENT, user_turn=turn, facts=facts,
                        history=hist, needs=needs)[0] for _ in range(tries)]
        rows.append({"scene": name, "you_say": turn, "original": o, "original_pass": sum(map(ok, o)) / tries,
                     "now": n, "now_pass": sum(map(ok, n)) / tries})
        print(f"  {name:30s} original {rows[-1]['original_pass']:.0%}  now {rows[-1]['now_pass']:.0%}")
    return rows


# ----------------------------------------------------------------------------- speed

@torch.no_grad()
def naive_generate(model, ids, n):
    """Generation the original way: re-read the whole conversation for every new token."""
    for _ in range(n):
        logits, _ = model(ids, targets=None)
        ids = torch.cat([ids, logits[:, -1].argmax(-1, keepdim=True)], 1)
    return ids


def speed(model, tok):
    ids = torch.tensor([tok.encode("Once upon a time " * 60).ids[:220]])
    model.generate(ids, max_new_tokens=8, temperature=0)  # warm up
    t = time.time(); model.generate(ids, max_new_tokens=48, temperature=0); cached = 48 / (time.time() - t)
    t = time.time(); naive_generate(model, ids, 48); naive = 48 / (time.time() - t)
    return {"naive_tokens_per_s": round(naive, 1), "cached_tokens_per_s": round(cached, 1), "speedup": round(cached / naive, 2)}


# ----------------------------------------------------------------------------- charts

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
C_ORIGINAL, C_NOW = "#eb6834", "#2a78d6"  # validated categorical slots 2 and 1


def charts(exam):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(OUT, "charts"), exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID})

    def style(ax):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    # 1) per skill on never-seen words
    labels = {"recall": "Recall a fact", "two_hop": "Chain two facts", "yes_no": "Yes / no", "perspective": "I / you flip",
              "correction": "Accept a correction", "unknown": "Admit not knowing", "freeform": "Long facts",
              "self": "Remember own words", "followup": "Follow up an answer"}  # names/mood: too few exam questions
    cats = [c for c in labels if c in exam["now"]["eval_cats"]]
    cats.sort(key=lambda c: exam["now"]["eval_cats"].get(c, 0))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(cats) + 1.4), facecolor=SURFACE)
    style(ax)
    y = range(len(cats))
    h = 0.36
    o = [exam["original"]["eval_cats"].get(c, 0) * 100 for c in cats]
    n = [exam["now"]["eval_cats"].get(c, 0) * 100 for c in cats]
    ax.barh([i + h / 2 + 0.02 for i in y], o, height=h, color=C_ORIGINAL, edgecolor=SURFACE, linewidth=2, label="At the start")
    ax.barh([i - h / 2 - 0.02 for i in y], n, height=h, color=C_NOW, edgecolor=SURFACE, linewidth=2, label="Now")
    for i, v in zip(y, n):
        ax.text(v + 1.5, i - h / 2 - 0.02, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    for i, v in zip(y, o):
        ax.text(v + 1.5, i + h / 2 + 0.02, f"{v:.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels[c] for c in cats])
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Reasoning skills on words never seen in training", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "charts", "skills_new_words.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)

    # 2) seen vs never-seen words: understanding vs memorizing
    fig, ax = plt.subplots(figsize=(8, 2.6), facecolor=SURFACE)
    style(ax)
    rows = [("Familiar words", "train"), ("Never-seen words", "eval")]
    for i, (label, key) in enumerate(rows):
        ov, nv = exam["original"][key] * 100, exam["now"][key] * 100
        ax.barh(i + 0.2, ov, height=0.36, color=C_ORIGINAL, edgecolor=SURFACE, linewidth=2, label="At the start" if i == 0 else None)
        ax.barh(i - 0.2, nv, height=0.36, color=C_NOW, edgecolor=SURFACE, linewidth=2, label="Now" if i == 0 else None)
        ax.text(ov + 1.5, i + 0.2, f"{ov:.0f}%", va="center", fontsize=9, color=INK2)
        ax.text(nv + 1.5, i - 0.2, f"{nv:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Overall reasoning exam (300 conversations each)", loc="left", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "charts", "exam_overall.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    original = BabyGPT.from_checkpoint(ORIGINAL)
    now = BabyGPT.from_checkpoint(NOW)
    results = {"measured_at": time.strftime("%Y-%m-%d %H:%M"), "original_checkpoint": ORIGINAL, "now_checkpoint": NOW}

    print("[1/4] Reasoning exam (both models, same 300 conversations per split)...")
    exam = {}
    for label, model in (("original", original), ("now", now)):
        exam[label] = {}
        for split in ("train", "eval"):
            acc, cats = evaluate_reasoning(model, tok, split, 300, 4321, torch.device("cpu"))
            exam[label][split] = acc
            exam[label][split + "_cats"] = cats
        print(f"  {label:8s} familiar {exam[label]['train']:.1%} | never-seen {exam[label]['eval']:.1%}")
    results["exam"] = exam

    print("[2/4] Real nursery situations (3 tries each)...")
    results["scenes"] = run_scenes(original, now, tok)

    print("[3/4] Reply speed...")
    results["speed"] = speed(now, tok)
    print(f"  {results['speed']}")

    results["sizes_mb"] = {
        "original_model_file": 197, "original_model_weights_only": round(os.path.getsize(ORIGINAL) / 2**20),
        "now_model_file": round(os.path.getsize(NOW) / 2**20),
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    print("[4/4] Charts...")
    charts(exam)
    print("Done: analysis/results.json, analysis/charts/")


if __name__ == "__main__":
    if "--charts-only" in sys.argv:  # redraw from results.json without re-measuring
        charts(json.load(open(os.path.join(OUT, "results.json"), encoding="utf-8"))["exam"])
    else:
        main()

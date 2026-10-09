#!/usr/bin/env python3
"""
deploy/export_web.py
====================
Turns the baby's brain (checkpoints/baby_chat.pt) into web/baby.onnx so it can think inside each
visitor's browser: free hosting, any number of visitors, and chats never leave their device.

  - One ONNX graph with a KV cache: the prompt is read once, then one token per step.
  - The causal mask is written out explicitly, so the same graph handles prompt and reply.
  - Weights are quantized to int8 (66 MB -> ~17 MB) after checking that replies stay the same.

Needs onnx + onnxruntime (kept in a separate environment from your main Python):
  python -m venv --system-site-packages .onnx-env && .onnx-env/Scripts/pip install onnx onnxruntime
  .onnx-env/Scripts/python deploy/export_web.py
"""

import os
import sys
import json
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train import BabyGPT

WEB = os.path.join(ROOT, "web")


class ExportBaby(nn.Module):
    """Same weights as BabyGPT; attention spelled out so it exports to one cache-aware graph."""

    def __init__(self, m: BabyGPT):
        super().__init__()
        self.m = m
        self.H = m.cfg.n_head
        self.D = m.cfg.n_embd // m.cfg.n_head

    def forward(self, input_ids, *past):
        m, H, D = self.m, self.H, self.D
        B, T = input_ids.shape
        P = past[0].shape[2]
        pos = torch.arange(T, device=input_ids.device) + P
        x = m.transformer.wte(input_ids) + m.transformer.wpe(pos)
        q_idx = (torch.arange(T) + P).unsqueeze(1)
        k_idx = torch.arange(P + T).unsqueeze(0)
        mask = (k_idx > q_idx).to(torch.float32) * -1e9  # a token may only look at itself and the past
        presents = []
        for i, block in enumerate(m.transformer.h):
            a = block.attn
            h = block.ln_1(x)
            q, k, v = a.c_attn(h).chunk(3, dim=-1)
            q = q.view(B, T, H, D).transpose(1, 2)
            k = torch.cat([past[2 * i], k.view(B, T, H, D).transpose(1, 2)], dim=2)
            v = torch.cat([past[2 * i + 1], v.view(B, T, H, D).transpose(1, 2)], dim=2)
            presents += [k, v]
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(D) + mask, dim=-1)
            y = (att @ v).transpose(1, 2).reshape(B, T, H * D)
            x = x + a.c_proj(y)
            x = x + block.mlp(block.ln_2(x))
        logits = m.lm_head(m.transformer.ln_f(x)[:, -1, :])
        return (logits, *presents)


def empty_past(cfg, P=0):
    D = cfg.n_embd // cfg.n_head
    return [np.zeros((1, cfg.n_head, P, D), dtype=np.float32) for _ in range(2 * cfg.n_layer)]


def ort_greedy(sess, cfg, ids, n=30, stop=3):
    names = [i.name for i in sess.get_inputs()]
    past = empty_past(cfg)
    feed = np.array([ids], dtype=np.int64)
    out = []
    for _ in range(n):
        res = sess.run(None, {names[0]: feed, **{nm: p for nm, p in zip(names[1:], past)}})
        tok = int(np.argmax(res[0][0]))
        out.append(tok)
        if tok == stop:
            break
        past = res[1:]
        feed = np.array([[tok]], dtype=np.int64)
    return out


def main():
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from tokenizers import Tokenizer
    from sft_data import DialogueGenerator, format_chatml

    os.makedirs(WEB, exist_ok=True)
    model = BabyGPT.from_checkpoint("checkpoints/baby_chat.pt")
    model.eval()
    cfg = model.cfg
    wrapper = ExportBaby(model).eval()
    L = cfg.n_layer

    # sanity: the rewritten attention matches the training model exactly
    ids = torch.randint(4, cfg.vocab_size, (1, 12))
    ref, _ = model(ids)
    got = wrapper(ids, *[torch.from_numpy(p) for p in empty_past(cfg)])[0]
    print(f"[*] rewritten attention vs original: max diff {(ref[0, -1] - got[0]).abs().max():.2e}")

    fp32 = os.path.join(WEB, "baby.fp32.onnx")
    past_names = [f"past_{kind}_{i}" for i in range(L) for kind in ("k", "v")]
    present_names = [f"present_{kind}_{i}" for i in range(L) for kind in ("k", "v")]
    dyn = {"input_ids": {1: "T"}, **{n: {2: "P"} for n in past_names}, **{n: {2: "PT"} for n in present_names}}
    example = (torch.randint(4, cfg.vocab_size, (1, 5)), *[torch.zeros(1, cfg.n_head, 3, cfg.n_embd // cfg.n_head)] * (2 * L))
    torch.onnx.export(wrapper, example, fp32, input_names=["input_ids", *past_names],
                      output_names=["logits", *present_names], dynamic_axes=dyn, opset_version=17, dynamo=False)
    print(f"[✓] Exported fp32 graph ({os.path.getsize(fp32) / 2**20:.0f} MB)")

    int8 = os.path.join(WEB, "baby.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f"[✓] Quantized to int8 ({os.path.getsize(int8) / 2**20:.0f} MB)")

    # check: greedy replies from ONNX (fp32 and int8) vs PyTorch on real exam prompts
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    s32 = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
    gen = DialogueGenerator("eval", seed=7)
    same32 = same8 = total = 0
    for _ in range(40):
        msgs, _ = gen.sample()
        ids = tok.encode(format_chatml(msgs[:-1])).ids
        with torch.no_grad():
            want = model.generate(torch.tensor([ids]), max_new_tokens=30, temperature=0, stop_token_id=3)[0, len(ids):].tolist()
        a, b = ort_greedy(s32, cfg, ids), ort_greedy(s8, cfg, ids)
        same32 += a == want
        same8 += tok.decode(b, skip_special_tokens=True).strip() == tok.decode(want, skip_special_tokens=True).strip()
        total += 1
    print(f"[*] identical replies: fp32 ONNX {same32}/{total} | int8 ONNX {same8}/{total}")
    os.remove(fp32)

    # what the browser needs besides the graph
    meta = {"n_layer": L, "n_head": cfg.n_head, "head_dim": cfg.n_embd // cfg.n_head, "n_positions": cfg.n_positions,
            "vocab_size": cfg.vocab_size, "stop_token_id": 3, "past_names": past_names}
    with open(os.path.join(WEB, "baby.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[✓] Wrote web/baby.onnx + web/baby.meta.json")


if __name__ == "__main__":
    main()

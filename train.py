#!/usr/bin/env python3
"""
train.py
========
Standalone, self-contained PyTorch pretraining script for a Baby LLM (~17M parameters)
trained from scratch on the educational TinyStories dataset.

Architecture:
  - GPT-style Decoder-Only Transformer (PyTorch native)
  - Vocab size: 8,192 (matching tokenizer/tokenizer.json)
  - Context window (n_positions): 512
  - Hidden dimension (n_embd): 384
  - Layers (n_layer): 6
  - Attention heads (n_head): 6 (d_head = 64)
  - PyTorch 2.0+ scaled dot-product attention (F.scaled_dot_product_attention)
  - Parameter count: ~17.1M parameters

Features:
  - Ingests pre-tokenized token streams `data/train.pt` and `data/val.pt` and samples
    random-offset windows, so no fixed chunk is ever repeated verbatim
  - Tied input/output embeddings and early stopping on validation loss to curb memorization
  - KV-cached generation (prompt encoded once, then one position per new token)
  - AdamW optimizer with decoupled weight decay (0.01)
  - Linear warmup + Cosine decay learning rate scheduler
  - Automatic mixed precision (AMP) when CUDA is available
  - Gradient clipping (max_norm = 1.0)
  - Periodic validation tracking loss and perplexity
  - Best-model checkpointing to `checkpoints/baby_model_best.pt` (weights only)
  - Next step: `python sft.py` teaches the pretrained model to converse and reason over memory
  - Sample autoregressive text generation at each eval interval with prompt:
    "<|im_start|>\nOnce upon a time, a little dog named Minik"

Usage:
  python train.py
  python train.py --batch-size 32 --lr 5e-4 --max-steps 10000 --eval-interval 500
"""

import os
import sys
import math
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from contextlib import nullcontext

# Force UTF-8 on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


# ==============================================================================
# 1. Model Configuration & Architecture Definition
# ==============================================================================

@dataclass
class BabyGPTConfig:
    vocab_size: int = 8192      # Matches tokenizer/tokenizer.json
    n_positions: int = 512      # Context length
    n_embd: int = 384           # Hidden dimension (384 / 6 = 64 head dim)
    n_layer: int = 6            # Number of Transformer layers
    n_head: int = 6             # Attention heads
    dropout: float = 0.1        # Dropout probability
    bias: bool = True           # Use bias in linear projections and layernorms
    tie_weights: bool = True    # Tied LM head: ~13.8M params (vs 17.1M untied) and better generalization


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch 2.0+ scaled_dot_product_attention."""
    def __init__(self, cfg: BabyGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # Batched Q, K, V projection
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, past_kv=None):
        B, T, C = x.shape  # Batch, Sequence Length, Embedding Dim

        # Compute query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Transpose to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV cache: append this step's keys/values to the ones from earlier steps
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)

        # FlashAttention / memory-efficient causal dot-product attention.
        # With a cache and a single new token, that token may attend to every cached key.
        dropout_p = self.dropout if self.training else 0.0
        is_causal = past_kv is None and T > 1
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal)

        # Re-assemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""
    def __init__(self, cfg: BabyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Decoder Transformer block with pre-LayerNorm architecture."""
    def __init__(self, cfg: BabyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, past_kv=None):
        a, present = self.attn(self.ln_1(x), past_kv)
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, present


class BabyGPT(nn.Module):
    """
    GPT-style Decoder-Only Language Model.
    Target parameter count: ~17M parameters with n_embd=384, n_layer=6, n_head=6, vocab=8192.
    """
    def __init__(self, cfg: BabyGPTConfig):
        super().__init__()
        self.cfg = cfg

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "wpe": nn.Embedding(cfg.n_positions, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "h": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": nn.LayerNorm(cfg.n_embd, elementwise_affine=cfg.bias),
        })

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.transformer.wte.weight

        # Initialize weights with standard Gaussian N(0, 0.02)
        self.apply(self._init_weights)

        # Scale residual projections at initialization (GPT-2 style)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = self.transformer.wte.weight.numel() + self.transformer.wpe.weight.numel()
        return {"total": total, "trainable": trainable, "embeddings": emb, "backbone": total - emb}

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "BabyGPT":
        """Load a trained BabyGPT model from a checkpoint dictionary."""
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        cfg_dict = ckpt["config"]
        cfg = BabyGPTConfig(**cfg_dict)
        model = cls(cfg)
        model.load_state_dict({k: v.float() for k, v in ckpt["model_state_dict"].items()})
        model.to(device)
        model.eval()
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs=None,
        use_cache: bool = False,
    ):
        """
        targets may contain -100 to mask positions out of the loss (used by SFT so the
        baby is only graded on its own replies, never on the caregiver's words).
        With use_cache=True, returns (logits, loss, present_kvs) for incremental decoding.
        """
        device = input_ids.device
        B, T = input_ids.shape
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        assert past_len + T <= self.cfg.n_positions, f"Sequence length {past_len + T} exceeds maximum {self.cfg.n_positions}"

        # Position indices: [past_len, ..., past_len + T - 1]
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(input_ids)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)        # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        presents = []
        for i, block in enumerate(self.transformer.h):
            x, present = block(x, past_kvs[i] if past_kvs is not None else None)
            presents.append(present)
        x = self.transformer.ln_f(x)

        loss = None
        if targets is not None:
            keep = targets != -100
            if keep.all():
                logits = self.lm_head(x)  # (B, T, vocab_size)
                loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1))
            else:
                # Masked training (SFT): only project the graded positions onto the vocabulary
                logits = self.lm_head(x[keep])
                loss = F.cross_entropy(logits, targets[keep])
        else:
            # Inference only needs the last position; skips a (T x vocab) matmul
            logits = self.lm_head(x[:, -1:, :])

        if use_cache:
            return logits, loss, presents
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        stop_token_id: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive generation with a KV cache: the prompt is encoded once, then each new
        token costs a single-position forward pass instead of re-reading the whole context.
        Supports greedy (temperature=0), top-k, top-p and a repetition penalty on generated tokens.
        """
        self.eval()
        # Leave room for the reply inside the context window
        max_prompt = self.cfg.n_positions - max_new_tokens
        if input_ids.size(1) > max_prompt:
            input_ids = input_ids[:, -max_prompt:]

        logits, _, past = self(input_ids, use_cache=True)
        generated = []
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :].float()

            if repetition_penalty != 1.0 and generated:
                seen = torch.tensor(sorted(set(generated)), device=logits.device)
                picked = logits[0, seen]
                logits[0, seen] = torch.where(picked > 0, picked / repetition_penalty, picked * repetition_penalty)

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                # Top-k filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    logits[0, sorted_idx[remove]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            tok = next_token.item()
            generated.append(tok)
            if stop_token_id is not None and tok == stop_token_id:
                break
            logits, _, past = self(next_token, past_kvs=past, use_cache=True)

        return input_ids


# ==============================================================================
# 2. Data Loading
# ==============================================================================

def load_token_stream(path: str) -> torch.Tensor:
    """Load pre-tokenized data as one flat token stream (accepts the old (N, 512) layout too)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tensor not found at '{path}'.")
    data = torch.load(path, weights_only=True)
    return data.reshape(-1)


class PretokenizedTensorDataset(Dataset):
    """
    Next-token-prediction windows over a flat token stream.

    random_offsets=True (training): every item is a window starting at a random position,
    so the model never sees the same fixed chunk twice. It has to learn how language works
    rather than memorize 512-token blocks. random_offsets=False (validation): fixed,
    non-overlapping windows so the loss is comparable between evaluations.
    """
    def __init__(self, tensor_or_path, seq_len: int = 512, random_offsets: bool = False):
        self.data = load_token_stream(tensor_or_path) if isinstance(tensor_or_path, str) else tensor_or_path.reshape(-1)
        self.seq_len = seq_len
        self.random_offsets = random_offsets
        self.num_windows = (self.data.numel() - 1) // seq_len

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.random_offsets:
            start = int(torch.randint(0, self.data.numel() - self.seq_len - 1, (1,)))
        else:
            start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].long()
        # Next-token prediction: input is x[:-1], target is x[1:]
        return chunk[:-1], chunk[1:]


def save_checkpoint(model: "BabyGPT", path: str, **extra):
    """Weights-only checkpoint (no optimizer state): ~3x smaller and all chat.py needs."""
    cfg = model.cfg
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": cfg.vocab_size,
            "n_positions": cfg.n_positions,
            "n_embd": cfg.n_embd,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "dropout": cfg.dropout,
            "bias": cfg.bias,
            "tie_weights": cfg.tie_weights,
        },
        **extra,
    }, path)


# ==============================================================================
# 3. Optimizer & Learning Rate Schedule
# ==============================================================================

def configure_optimizers(model: BabyGPT, weight_decay: float = 0.01, learning_rate: float = 5e-4):
    """
    Decouple parameters: Apply weight decay only to 2D matrices (embeddings and linear weights),
    exempt 1D parameters (biases and LayerNorm scales/biases).
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer


def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    """Linear warmup followed by cosine decay to min_lr_ratio * max_lr."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 4. Evaluation & Generation Loop
# ==============================================================================

@torch.no_grad()
def evaluate(
    model: BabyGPT,
    val_loader: DataLoader,
    device: torch.device,
    amp_ctx,
    max_eval_batches: int = 50,
) -> Tuple[float, float]:
    """Calculate validation loss and perplexity on val_loader."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_evaluated = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_eval_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx:
            _, loss = model(x, y)

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
        batches_evaluated += 1

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 20.0))  # Prevent numerical overflow
    return avg_loss, perplexity


def generate_sample(
    model: BabyGPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Generate completion and decode back to text."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    # Stop token: <|im_end|> if available
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token_id=im_end_id,
        )

    output_list = output_ids[0].tolist()
    # Decode only the generated portion or full completion
    full_text = tokenizer.decode(output_list, skip_special_tokens=False)
    return full_text.strip()


# ==============================================================================
# 5. Main Training Pipeline
# ==============================================================================

def train(
    data_dir: str = "data",
    tokenizer_path: str = "tokenizer/tokenizer.json",
    checkpoint_dir: str = "checkpoints",
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    max_steps: int = 5000,
    warmup_steps: int = 200,
    eval_interval: int = 500,
    eval_steps: int = 50,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    n_embd: int = 384,
    n_layer: int = 6,
    n_head: int = 6,
    n_positions: int = 512,
    tie_weights: bool = True,
    patience: int = 3,
    device_override: Optional[str] = None,
    seed: int = 42,
):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device selection
    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = device.type == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 72)
    print("  BABY LLM PRETRAINING PIPELINE — FROM SCRATCH")
    print("=" * 72)
    print(f"[*] Target Device:            {str(device).upper()} (CUDA Available: {torch.cuda.is_available()})")
    print(f"[*] Mixed Precision (AMP):    {'Enabled (fp16)' if use_cuda else 'Disabled (CPU FP32)'}")

    # 1. Load Tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run prepare_baby_data.py first!")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"[*] Loaded Tokenizer:         {vocab_size:,} vocabulary size from '{tokenizer_path}'")

    # 2. Load Datasets
    train_path = os.path.join(data_dir, "train.pt")
    val_path = os.path.join(data_dir, "val.pt")
    print(f"[*] Loading training tensor:   '{train_path}'...")
    train_dataset = PretokenizedTensorDataset(train_path, seq_len=n_positions, random_offsets=True)
    print(f"[*] Loading validation tensor: '{val_path}'...")
    val_dataset = PretokenizedTensorDataset(val_path, seq_len=n_positions)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
        drop_last=False,
    )

    train_tokens = train_dataset.data.numel()
    print(f"[✓] Train Dataset:            {train_tokens:,} tokens (random-offset windows)")
    print(f"[✓] Validation Dataset:       {val_dataset.data.numel():,} tokens")
    epochs = max_steps * batch_size * n_positions * gradient_accumulation_steps / max(1, train_tokens)
    print(f"[*] Planned passes over data: {epochs:.1f}")
    if epochs > 2:
        print("[!] More than ~2 passes over the same text pushes a small model toward memorizing it.")
        print("    Prefer more data (prepare_baby_data.py --num-stories 300000) over more steps.")

    # 3. Instantiate Model
    cfg = BabyGPTConfig(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        dropout=0.1,
        tie_weights=tie_weights,
    )
    model = BabyGPT(cfg).to(device)
    param_counts = model.count_parameters()
    print("\n" + "-" * 72)
    print(f"  Model Architecture: BabyGPT ({n_layer}L, {n_head}H, {n_embd}D, ctx={n_positions})")
    print(f"  • Total Parameters:         {param_counts['total']:,} ({param_counts['total'] / 1e6:.2f}M)")
    print(f"  • Trainable Parameters:     {param_counts['trainable']:,}")
    print(f"  • Embedding Parameters:     {param_counts['embeddings']:,}")
    print(f"  • Backbone Parameters:      {param_counts['backbone']:,}")
    print("-" * 72)

    # 4. Setup Optimizer & Scheduler
    optimizer = configure_optimizers(model, weight_decay=weight_decay, learning_rate=learning_rate)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, max_steps=max_steps)

    sample_prompt = "<|im_start|>\nOnce upon a time, a little dog named Minik"
    best_val_loss = float("inf")
    bad_evals = 0
    global_step = 0
    t_start = time.time()
    t_step_start = time.time()

    print(f"\n[*] Starting Pretraining Loop:")
    print(f"    - Max Steps:              {max_steps:,}")
    print(f"    - Batch Size:             {batch_size} (effective tokens/step = {batch_size * n_positions * gradient_accumulation_steps:,})")
    print(f"    - Learning Rate:          {learning_rate} (cosine decay to {learning_rate * 0.1})")
    print(f"    - Warmup Steps:           {warmup_steps}")
    print(f"    - Eval Interval:          every {eval_interval} steps\n")

    # Initial Zero-Step Evaluation & Generation
    print("[*] Running initial baseline evaluation before training (Step 0)...")
    init_val_loss, init_val_ppl = evaluate(model, val_loader, device, amp_ctx, max_eval_batches=eval_steps)
    init_gen = generate_sample(model, tokenizer, sample_prompt, device, max_new_tokens=40)
    print(f"[*] Step 0 Baseline -> Val Loss: {init_val_loss:.4f} | Val PPL: {init_val_ppl:.2f}")
    print(f"[*] Step 0 Untrained Sample Completion:\n    \"{init_gen}\"\n")

    data_iter = iter(train_loader)

    while global_step < max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        for micro_step in range(gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                logits, loss = model(x, y)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1
        current_lr = scheduler.get_last_lr()[0]

        # Log training step progress
        if global_step % 10 == 0 or global_step == 1:
            step_time = (time.time() - t_step_start) / (10 if global_step > 1 else 1)
            tokens_per_sec = (batch_size * n_positions * gradient_accumulation_steps) / max(1e-5, step_time)
            print(
                f"Step {global_step:5d}/{max_steps} | "
                f"Train Loss: {loss_accum:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Grad: {grad_norm:.2f} | "
                f"{tokens_per_sec:,.0f} tok/s",
                end="\r" if (global_step % eval_interval != 0) else "\n",
                flush=True
            )
            t_step_start = time.time()

        # Evaluation, Checkpointing & Sample Generation
        if global_step % eval_interval == 0 or global_step == max_steps:
            val_loss, val_ppl = evaluate(model, val_loader, device, amp_ctx, max_eval_batches=eval_steps)
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                bad_evals = 0
                best_checkpoint_path = os.path.join(checkpoint_dir, "baby_model_best.pt")
                save_checkpoint(model, best_checkpoint_path, step=global_step, val_loss=val_loss, val_ppl=val_ppl)
                status_marker = " [★ NEW BEST]"
            else:
                bad_evals += 1
                status_marker = f" (no improvement {bad_evals}/{patience})"

            print("\n" + "=" * 72)
            print(f"  EVALUATION AT STEP {global_step:,} / {max_steps:,}{status_marker}")
            print(f"  • Validation Loss:       {val_loss:.4f}")
            print(f"  • Validation Perplexity: {val_ppl:.2f}")
            print(f"  • Best Validation Loss:  {best_val_loss:.4f}")
            print("-" * 72)

            # Generate sample completion
            sample_out = generate_sample(model, tokenizer, sample_prompt, device, max_new_tokens=60)
            print(f"  Sample Generation (Prompt: \"{sample_prompt}\"):")
            print(f"  -------------------------------------------------------------")
            print(f"  {sample_out}")
            if patience > 0 and bad_evals >= patience:
                # Validation loss rising while training loss falls = memorizing. Stop there.
                print(f"[!] Early stop: validation loss has not improved for {patience} evaluations.")
                break
            print("=" * 72 + "\n")
            t_step_start = time.time()

    elapsed = time.time() - t_start
    print(f"\n[✓] Training complete in {elapsed / 60:.2f} minutes!")
    print(f"[✓] Best validation loss: {best_val_loss:.4f}")
    print(f"[✓] Checkpoint saved at: {os.path.join(checkpoint_dir, 'baby_model_best.pt')}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain a Baby LLM from scratch on pre-tokenized TinyStories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing train.pt and val.pt")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per forward step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay on 2D weights")
    parser.add_argument("--max-steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation and checkpointing interval")
    parser.add_argument("--eval-steps", type=int, default=50, help="Batches to evaluate on during validation")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--n-embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n-layer", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--n-head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n-positions", type=int, default=512, help="Context length")
    parser.add_argument("--no-tie-weights", action="store_true", help="Untie input embedding and LM head (more params, more memorization)")
    parser.add_argument("--patience", type=int, default=3, help="Early-stop after this many evals without val improvement (0 = off)")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cpu', 'cuda')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stream", action="store_true",
                        help="Keep learning from TinyStories streamed from Hugging Face, storing nothing (see stream_train.py)")
    parser.add_argument("--tokens", type=str, default="100M", help="With --stream: how much to read, e.g. 100M, 300M")
    parser.add_argument("--init", type=str, default="checkpoints/baby_model_best.pt", help="With --stream: brain to continue from")

    args = parser.parse_args()

    if args.stream:
        from stream_train import train_stream
        train_stream(init=args.init, out=os.path.join(args.checkpoint_dir, "baby_model_best.pt"),
                     tokenizer_path=args.tokenizer_path, tokens=args.tokens, batch_size=args.batch_size,
                     seq_len=args.n_positions, seed=args.seed)
        return

    train(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.n_positions,
        tie_weights=not args.no_tie_weights,
        patience=args.patience,
        device_override=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

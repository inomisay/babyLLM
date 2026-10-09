#!/usr/bin/env python3
"""
prepare_baby_data.py
====================
Standalone, self-contained data ingestion and tokenizer pipeline for a Baby LLM
using the educational TinyStories dataset (roneneldan/TinyStories).

Pipeline Stages:
  1. Dataset Streaming & Extraction:
     Streams clean stories from Hugging Face `roneneldan/TinyStories` into `data/raw_stories.txt`.
  2. Train Compact Tokenizer from Scratch:
     Trains a Byte-level BPE tokenizer (vocab size 8,192) with ChatML special tokens
     (<unk>, <pad>, <|im_start|>, <|im_end|>) and saves artifacts to `tokenizer/`.
  3. Dataset Tokenization & Chunking & Train/Val Split:
     Tokenizes the corpus, chunks into fixed-length windows of 512 tokens, splits 90/10,
     and saves binary PyTorch tensors to `data/train.pt` and `data/val.pt`.
  4. Diagnostics & Sanity Check:
     Calculates compression metrics and verifies sample English reconstruction.

Usage:
  python prepare_baby_data.py --num-stories 300000 --vocab-size 8192 --seq-len 512
"""

import os
import sys
import json
import time
import unicodedata
import re
import argparse
from typing import List, Generator

# Suppress Hugging Face Windows symlink warning if running on non-admin/dev-mode Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Force UTF-8 on Windows consoles to cleanly display subwords and progress
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from tokenizers import ByteLevelBPETokenizer, Tokenizer

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm is unavailable
    def tqdm(iterable, total=None, desc=""):
        step = max(1, (total or 100) // 20)
        for i, item in enumerate(iterable):
            if i % step == 0 or (total and i == total - 1):
                pct = f"({(i + 1) / total * 100:.1f}%)" if total else ""
                print(f"[*] {desc}: {i + 1}/{total} {pct}", end="\r", flush=True)
            yield item
        print()


SPECIAL_TOKENS = ["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]
START_TOKEN = "<|im_start|>"
END_TOKEN = "<|im_end|>"


def clean_story(text: str) -> str:
    """
    Standardize story text:
      - NFKC unicode normalization
      - Strip unprintable and malformed control characters
      - Collapse whitespace and normalize paragraph newlines
    """
    if not text:
        return ""
    
    # 1. Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)

    # 2. Strip control characters (preserve standard newlines \n)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # 3. Normalize carriage returns and horizontal whitespace
    text = text.replace("\r", "")
    text = text.replace("\t", " ")

    # 4. Collapse repeated internal spaces per line
    lines = [re.sub(r" +", " ", line).strip() for line in text.split("\n")]

    # 5. Join lines and normalize excessive blank lines (cap at 2 consecutive newlines)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return cleaned


def stage1_extract_stories(
    output_path: str,
    num_stories: int = 50000,
    min_chars: int = 30,
    skip_if_exists: bool = False,
) -> int:
    """
    Stream stories from roneneldan/TinyStories on Hugging Face, clean whitespace,
    and write to `data/raw_stories.txt` delimited by special markers and standard newlines.
    """
    print("\n" + "=" * 70)
    print("  STAGE 1: Streaming & Extracting TinyStories Dataset")
    print("=" * 70)

    if skip_if_exists and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"[*] Found existing raw dataset at: {output_path}")
        print(f"[*] File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB. Skipping download.")
        # Count stories in existing file
        count = 0
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() == END_TOKEN:
                    count += 1
        return count

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Initializing Hugging Face streaming for 'roneneldan/TinyStories' (split='train')...")
    from datasets import load_dataset
    dataset_stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    saved_count = 0
    total_chars = 0
    t0 = time.time()

    print(f"[*] Extracting and cleaning {num_stories:,} stories into '{output_path}'...")
    pbar = tqdm(total=num_stories, desc="Streaming stories")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for record in dataset_stream:
            raw_text = record.get("text", "")
            cleaned = clean_story(raw_text)

            if len(cleaned) < min_chars:
                continue

            # Write story bounded by ChatML delimiters with clean newlines
            out_f.write(f"{START_TOKEN}\n{cleaned}\n{END_TOKEN}\n\n")

            saved_count += 1
            total_chars += len(cleaned)
            pbar.update(1)

            if saved_count >= num_stories:
                break

    pbar.close()
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"[✓] Extracted {saved_count:,} stories in {elapsed:.2f}s ({saved_count / max(0.1, elapsed):.1f} stories/s)")
    print(f"[✓] Output file: {output_path} ({file_size_mb:.2f} MB, {total_chars:,} text chars)")
    return saved_count


def stage2_train_tokenizer(
    raw_stories_path: str,
    tokenizer_dir: str,
    vocab_size: int = 8192,
) -> Tokenizer:
    """
    Train a compact Byte-level BPE Tokenizer from scratch directly on `data/raw_stories.txt`.
    Saves tokenizer.json, vocab.json, merges.txt, and tokenizer_config.json into `tokenizer/`.
    """
    print("\n" + "=" * 70)
    print("  STAGE 2: Training Compact Byte-Level BPE Tokenizer from Scratch")
    print("=" * 70)

    os.makedirs(tokenizer_dir, exist_ok=True)
    print(f"[*] Source corpus: {raw_stories_path}")
    print(f"[*] Target vocabulary size: {vocab_size:,} tokens (optimized for baby LLMs)")
    print(f"[*] Special tokens: {SPECIAL_TOKENS}")

    # Initialize ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()

    t0 = time.time()
    print("[*] Training Byte-Level BPE tokenizer model on corpus...")
    tokenizer.train(
        files=[raw_stories_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    train_time = time.time() - t0

    # Save complete tokenizer artifacts
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)
    tokenizer.save_model(tokenizer_dir)

    # Save structured config metadata for easy inspection and loader compatibility
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")
    token_id_map = {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS}
    config_data = {
        "tokenizer_class": "ByteLevelBPETokenizer",
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": SPECIAL_TOKENS,
        "special_token_ids": token_id_map,
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "im_start_token": "<|im_start|>",
        "im_end_token": "<|im_end|>",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"[✓] Tokenizer trained in {train_time:.2f}s!")
    print(f"[✓] Actual vocabulary size: {tokenizer.get_vocab_size():,} tokens")
    print(f"[✓] Special token IDs: {token_id_map}")
    print(f"[✓] Saved artifacts in '{tokenizer_dir}':")
    for fname in ["tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"]:
        fpath = os.path.join(tokenizer_dir, fname)
        size_kb = os.path.getsize(fpath) / 1024 if os.path.exists(fpath) else 0
        print(f"    - {fname:22s} ({size_kb:.1f} KB)")

    # Load unified Tokenizer to guarantee exact special-token preservation in Stage 3
    loaded_tok = Tokenizer.from_file(tokenizer_json_path)
    return loaded_tok


def read_stories_in_batches(file_path: str, batch_size: int = 2000) -> Generator[List[str], None, None]:
    """
    Generator that parses stories from `data/raw_stories.txt` and yields them in batches.
    """
    current_story = []
    batch = []
    in_story = False

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped == START_TOKEN:
                in_story = True
                current_story = [START_TOKEN]
            elif stripped == END_TOKEN:
                if in_story:
                    current_story.append(END_TOKEN)
                    batch.append("\n".join(current_story))
                    current_story = []
                    in_story = False
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            elif in_story:
                current_story.append(line.rstrip("\r\n"))

    # Flush remaining story or batch
    if current_story:
        batch.append("\n".join(current_story))
    if batch:
        yield batch


def stage3_tokenize_and_split(
    raw_stories_path: str,
    tokenizer_dir: str,
    output_dir: str,
    seq_len: int = 512,
    train_ratio: float = 0.9,
    batch_size: int = 2000,
) -> dict:
    """
    Tokenize raw stories, chunk into fixed-length windows of `seq_len` tokens,
    split into 90% train / 10% validation, and save as `train.pt` and `val.pt`.
    """
    print("\n" + "=" * 70)
    print("  STAGE 3: Dataset Tokenization, Chunking & Train/Val Split")
    print("=" * 70)

    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_json_path}. Run Stage 2 first.")

    print(f"[*] Loading trained tokenizer from '{tokenizer_json_path}'...")
    tokenizer = Tokenizer.from_file(tokenizer_json_path)

    os.makedirs(output_dir, exist_ok=True)
    train_out_path = os.path.join(output_dir, "train.pt")
    val_out_path = os.path.join(output_dir, "val.pt")

    print(f"[*] Batch encoding stories from '{raw_stories_path}' (batch_size={batch_size})...")
    t0 = time.time()
    all_token_ids: List[int] = []
    total_stories = 0

    for story_batch in read_stories_in_batches(raw_stories_path, batch_size=batch_size):
        encodings = tokenizer.encode_batch(story_batch)
        for enc in encodings:
            all_token_ids.extend(enc.ids)
        total_stories += len(story_batch)
        print(f"    - Processed {total_stories:,} stories | Tokens so far: {len(all_token_ids):,}", end="\r", flush=True)

    print()
    tokenization_time = time.time() - t0
    total_raw_tokens = len(all_token_ids)
    print(f"[✓] Tokenized {total_stories:,} stories into {total_raw_tokens:,} tokens in {tokenization_time:.2f}s "
          f"({total_raw_tokens / max(0.1, tokenization_time):,.0f} tokens/s)")

    # Fixed-length chunking
    num_chunks = total_raw_tokens // seq_len
    total_usable_tokens = num_chunks * seq_len
    remainder = total_raw_tokens % seq_len

    print(f"[*] Chunking sequence into fixed-length windows of {seq_len} tokens...")
    print(f"    - Total full chunks: {num_chunks:,}")
    print(f"    - Tokens retained: {total_usable_tokens:,} ({total_usable_tokens / total_raw_tokens * 100:.2f}%)")
    print(f"    - Trailing remainder discarded: {remainder} tokens")

    # Reshape into (N, seq_len) long tensor
    tensor_data = torch.tensor(all_token_ids[:total_usable_tokens], dtype=torch.long)
    chunks_tensor = tensor_data.view(num_chunks, seq_len)

    # Train / Val Split (90% / 10%)
    train_size = int(num_chunks * train_ratio)
    val_size = num_chunks - train_size

    train_tensor = chunks_tensor[:train_size].clone()
    val_tensor = chunks_tensor[train_size:].clone()

    print(f"[*] Partitioning dataset: {train_ratio * 100:.0f}% Train / {(1 - train_ratio) * 100:.0f}% Val:")
    print(f"    - Train set: {train_size:,} chunks | Shape: {list(train_tensor.shape)} ({train_tensor.numel() * 2 / (1024**2):.1f} MB as uint16)")
    print(f"    - Val set:   {val_size:,} chunks | Shape: {list(val_tensor.shape)} ({val_tensor.numel() * 2 / (1024**2):.1f} MB as uint16)")

    # Save as flat uint16 token streams: 4x smaller than int64, and train.py samples
    # random-offset windows from the stream instead of replaying these fixed chunks.
    store_dtype = torch.uint16 if tokenizer.get_vocab_size() <= 65535 else torch.int32
    print(f"[*] Saving flat {str(store_dtype).replace('torch.', '')} token streams to disk...")
    torch.save(train_tensor.reshape(-1).to(store_dtype), train_out_path)
    torch.save(val_tensor.reshape(-1).to(store_dtype), val_out_path)

    print(f"[✓] Saved Train tensor -> {train_out_path} ({os.path.getsize(train_out_path) / (1024**2):.2f} MB)")
    print(f"[✓] Saved Val tensor   -> {val_out_path} ({os.path.getsize(val_out_path) / (1024**2):.2f} MB)")

    return {
        "total_stories": total_stories,
        "total_raw_tokens": total_raw_tokens,
        "num_chunks": num_chunks,
        "train_chunks": train_size,
        "val_chunks": val_size,
        "seq_len": seq_len,
        "train_path": train_out_path,
        "val_path": val_out_path,
        "first_train_chunk": train_tensor[0].tolist(),
    }


def stage4_diagnostics(
    raw_stories_path: str,
    tokenizer_dir: str,
    stats: dict,
):
    """
    Report compression statistics and verify English reconstruction fidelity.
    """
    print("\n" + "=" * 70)
    print("  STAGE 4: Pipeline Diagnostics & Reconstruction Sanity Check")
    print("=" * 70)

    # File and character metrics
    raw_file_bytes = os.path.getsize(raw_stories_path)
    with open(raw_stories_path, "r", encoding="utf-8") as f:
        raw_text_sample = f.read(500000)
    total_tokens = stats["total_raw_tokens"]

    chars_per_token = len(raw_text_sample) / max(1, len(Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json")).encode(raw_text_sample).ids))
    bytes_per_token = raw_file_bytes / max(1, total_tokens)
    compression_pct = (1.0 - (total_tokens / max(1, raw_file_bytes))) * 100.0

    print("[1] Quantitative Metrics:")
    print(f"    - Total Stories Extracted:      {stats['total_stories']:,}")
    print(f"    - Raw Corpus Size:              {raw_file_bytes / (1024 * 1024):.2f} MB")
    print(f"    - Total Tokens Ingested:        {total_tokens:,}")
    print(f"    - Total 512-Token Windows:      {stats['num_chunks']:,} ({stats['train_chunks']:,} train / {stats['val_chunks']:,} val)")
    print(f"    - Tokenizer Compression Ratio:  {bytes_per_token:.2f} bytes/token ({chars_per_token:.2f} chars/token)")
    print(f"    - Space Reduction:              {compression_pct:.1f}% vs raw UTF-8 byte stream")

    print("\n[2] Tokenizer Special Token Verification:")
    tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
    for tok in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(tok)
        print(f"    - Special Token {tok:15s} -> ID {tid}")

    print("\n[3] Reconstruction & Round-Trip Decode Check (First 512-Token Train Chunk):")
    first_chunk_ids = stats["first_train_chunk"]
    decoded_with_special = tokenizer.decode(first_chunk_ids[:120], skip_special_tokens=False)
    decoded_clean = tokenizer.decode(first_chunk_ids[:120], skip_special_tokens=True)

    print("\n--- [Token IDs (first 30 tokens)] ---")
    print(first_chunk_ids[:30])

    print("\n--- [Decoded with Special Tokens (ChatML representation)] ---")
    print(decoded_with_special[:350].strip() + ("..." if len(decoded_with_special) > 350 else ""))

    print("\n--- [Decoded Clean English (Model output view)] ---")
    print(decoded_clean[:350].strip() + ("..." if len(decoded_clean) > 350 else ""))

    # Verification checks
    assert len(first_chunk_ids) == stats["seq_len"], f"Chunk length mismatch: {len(first_chunk_ids)} != {stats['seq_len']}"
    assert START_TOKEN in decoded_with_special, f"Missing start token marker in decoded sample"
    print("\n" + "=" * 70)
    print("  PIPELINE READY: Pre-tokenized tensors are ready for model pretraining!")
    print(f"  Training tensor:   {stats['train_path']}")
    print(f"  Validation tensor: {stats['val_path']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Data Ingestion & Tokenizer Pipeline for Baby LLM (TinyStories)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-stories", type=int, default=300000,
                        help="Stories to stream from TinyStories. More data is the strongest defense against memorization "
                             "(~300k stories = ~60M tokens, about one pass for 5000 steps at batch 24)")
    parser.add_argument("--vocab-size", type=int, default=8192, help="BPE vocabulary size")
    parser.add_argument("--seq-len", type=int, default=512, help="Fixed token window length for chunking")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split fraction (remainder is validation)")
    parser.add_argument("--data-dir", type=str, default="data", help="Output directory for text and tensor files")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Output directory for tokenizer artifacts")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for parallel tokenization")
    parser.add_argument("--min-chars", type=int, default=30, help="Minimum character length for accepted stories")
    parser.add_argument("--skip-download", action="store_true", help="Skip streaming if data/raw_stories.txt already exists")

    args = parser.parse_args()

    raw_stories_path = os.path.join(args.data_dir, "raw_stories.txt")

    # 1. Dataset Extraction
    stage1_extract_stories(
        output_path=raw_stories_path,
        num_stories=args.num_stories,
        min_chars=args.min_chars,
        skip_if_exists=args.skip_download,
    )

    # 2. Tokenizer Training
    stage2_train_tokenizer(
        raw_stories_path=raw_stories_path,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
    )

    # 3. Tokenization & Chunking
    stats = stage3_tokenize_and_split(
        raw_stories_path=raw_stories_path,
        tokenizer_dir=args.tokenizer_dir,
        output_dir=args.data_dir,
        seq_len=args.seq_len,
        train_ratio=args.train_ratio,
        batch_size=args.batch_size,
    )

    # 4. Diagnostics & Sanity Check
    stage4_diagnostics(
        raw_stories_path=raw_stories_path,
        tokenizer_dir=args.tokenizer_dir,
        stats=stats,
    )


if __name__ == "__main__":
    main()

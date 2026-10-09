import math
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import BabyTransformerLM, BabyLMConfig
from dataset import (
    MultiTurnDialogueDataset,
    SimpleBPETokenizer,
    generate_entity_tracking_dialogues,
    generate_belief_correction_dialogues,
    get_curated_sample_stories,
    format_chatml_turn,
    format_chatml_conversation
)

def configure_optimizers(model: BabyTransformerLM, lr: float = 4e-4, weight_decay: float = 0.05):
    """
    Decoupled weight decay: apply weight decay only to 2D matrices (embeddings and linear weights),
    exempt 1D parameters (biases and RMSNorm scales) to preserve representation stability.
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_multiturn_sft(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 4e-4,
    max_seq_len: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 68)
    print("  BabyLM Multi-Turn SFT & Belief Updating Training Pipeline")
    print("  Entity Tracking, Recency Overriding & Loss Masking")
    print("=" * 68)
    print(f"[*] Running on device: {device.upper()}")

    # 1. Generate multi-turn dialogues: Entity tracking + Belief corrections
    print("[*] Generating synthetic multi-turn dialogues...")
    tracking_dialogues = generate_entity_tracking_dialogues(num_samples=250)
    correction_dialogues = generate_belief_correction_dialogues(num_samples=250)
    all_dialogues = tracking_dialogues + correction_dialogues
    
    import random
    random.seed(42)
    random.shuffle(all_dialogues)

    curated_stories = get_curated_sample_stories()

    # Collect raw text corpora to build a unified vocabulary
    vocab_corpus = []
    for d in all_dialogues:
        vocab_corpus.append(format_chatml_conversation(d))
    vocab_corpus.extend(curated_stories)

    # 2. Build Tokenizer with ChatML support
    tokenizer = SimpleBPETokenizer(vocab_size=2048)
    tokenizer.build_vocab_from_texts(vocab_corpus)
    print(f"[*] Built Tokenizer vocabulary size: {len(tokenizer.token_to_id)} tokens")
    print(f"[*] ChatML Special Tokens -> <|im_start|>: {tokenizer.im_start_id}, <|im_end|>: {tokenizer.im_end_id}")

    # 3. Create Multi-Turn Dataset with Loss Masking (-100 on user/prompt tokens)
    dataset = MultiTurnDialogueDataset(all_dialogues, tokenizer, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"[*] Dataset samples: {len(dataset)} dialogues ({len(correction_dialogues)} belief updates) | Batches/epoch: {len(dataloader)}")

    # 4. Initialize Architecture
    cfg = BabyLMConfig(
        vocab_size=len(tokenizer.token_to_id),
        d_model=192,
        n_layers=4,
        n_heads=6,
        d_ff=512,
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.im_end_id,  # Use <|im_end|> as default conversational EOS
    )
    model = BabyTransformerLM(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Model parameters: {total_params:,} (~{total_params/1e6:.2f}M)")

    # 5. Optimizer & Cosine Schedule
    total_steps = epochs * len(dataloader)
    warmup_steps = max(10, int(0.08 * total_steps))
    optimizer = configure_optimizers(model, lr=lr, weight_decay=0.05)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 6. Training Loop with Loss Masking
    """
    HOW LOSS MASKING ENFORCES BELIEF CORRECTION & RECENCY:
    In dataset.py, all user prompt tokens have target = -100.
    PyTorch F.cross_entropy ignores targets == -100.
    
    When a dialogue contains conflicting facts:
      Turn 1: User says 'I have a dog'
      Turn 3: User corrects 'Actually, my pet is a cat, not a dog'
      Turn 5: User asks 'What kind of pet do I have?'
      Turn 6: Assistant outputs 'You have a cat!'
    
    Only the assistant's tokens in Turn 6 receive loss supervision.
    Because the target is 'cat' and NOT 'dog':
      - Gradients penalize attention weights that route to the outdated key 'dog'.
      - Gradients reward attention heads that condition on the revision cue ('Actually', 'not a dog')
        and route the Query representation directly to the revised key 'cat'.
      - The model learns a generalizable rule: when a correction cue is present, the later attribute
        supersedes the earlier one.
    """
    model.train()
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for step, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(inputs, targets=targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{epochs:02d} | SFT Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    elapsed = time.time() - start_time
    print(f"[*] Multi-turn SFT training completed in {elapsed:.2f}s!")

    # 7. Multi-Turn Anaphora & Belief Correction Live Evaluation
    print("\n" + "=" * 68)
    print("  MULTI-TURN BELIEF UPDATING & CORRECTION LIVE EVALUATION")
    print("=" * 68)

    test_conversations = [
        # Scenario 1: Reference Resolution & Pet Anaphora
        {
            "title": "Pronoun & Entity Anaphora Resolution",
            "history": [
                {"role": "system", "content": "You are Monika, a caring baby companion."},
                {"role": "user", "content": "You know I have a dog."},
                {"role": "assistant", "content": "A dog! Animals are so wonderful. What is their name?"},
                {"role": "user", "content": "Her name is Minik."},
            ]
        },
        # Scenario 2: Explicit Belief Correction (Pet: dog -> cat)
        {
            "title": "Belief Correction (Pet: dog -> cat)",
            "history": [
                {"role": "system", "content": "You are Monika, a caring baby companion who updates memory when corrected."},
                {"role": "user", "content": "I have a little dog at home."},
                {"role": "assistant", "content": "A dog! Animals are so wonderful. How is your dog doing?"},
                {"role": "user", "content": "Actually, my pet is a cat, not a dog. I made a typo earlier!"},
                {"role": "assistant", "content": "Oh, got it! A cat! Thank you for correcting me, I will remember that you have a cat."},
                {"role": "user", "content": "What kind of pet do I have?"},
            ]
        },
        # Scenario 3: Explicit Belief Correction (Color: green -> blue)
        {
            "title": "Belief Correction (Color: green -> blue)",
            "history": [
                {"role": "system", "content": "You are Monika, a caring baby companion who updates memory when corrected."},
                {"role": "user", "content": "My favorite color is green."},
                {"role": "assistant", "content": "Green is a peaceful color! Why do you like it?"},
                {"role": "user", "content": "Wait, no, my favorite color is definitely blue! I made a typo."},
                {"role": "assistant", "content": "Got it! I will remember that your favorite color is blue."},
                {"role": "user", "content": "What is my favorite color?"},
            ]
        },
        # Scenario 4: Negative Constraint Verification ('Do I have a dog?')
        {
            "title": "Negative Constraint Check (Do I have a dog?)",
            "history": [
                {"role": "system", "content": "You are Monika, a caring baby companion who updates memory when corrected."},
                {"role": "user", "content": "I adopted a puppy yesterday."},
                {"role": "assistant", "content": "How sweet! A new puppy must be so playful."},
                {"role": "user", "content": "Wait, no, it is a parrot! I gave you the wrong animal before."},
                {"role": "assistant", "content": "Understood! I updated my notes: you have a parrot, not a puppy."},
                {"role": "user", "content": "Do I have a puppy?"},
            ]
        },
    ]

    model.eval()
    for idx, test_case in enumerate(test_conversations, 1):
        print(f"\n--- Evaluation Case {idx}: {test_case['title']} ---")
        prompt_text = format_chatml_conversation(test_case["history"], add_generation_prompt=True)
        print(f"[Prompt Context]:\n{prompt_text.strip()}\n")

        tokens = tokenizer.encode(prompt_text)
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        gen_tokens = model.generate(
            input_tensor,
            max_new_tokens=40,
            temperature=0.6,
            top_k=25,
            stop_token_ids=[tokenizer.im_end_id]
        )

        response_ids = gen_tokens[0][len(tokens):].tolist()
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        print(f"[Assistant Generated Response]: {response_text.strip()}")

    # 8. Save Weights and Tokenizer
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/babylm_sft_weights.pt"
    tokenizer_path = "checkpoints/tokenizer.json"

    torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_path)
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)

    print(f"\n[*] Checkpoint successfully saved to: {checkpoint_path}")
    print(f"[*] Tokenizer serialized to: {tokenizer_path}")


if __name__ == "__main__":
    train_multiturn_sft()

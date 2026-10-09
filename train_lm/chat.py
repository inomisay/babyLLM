import os
import sys
import re
import json
import torch
from typing import List, Dict, Optional, Tuple

# Add root directory to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tokenizers import Tokenizer
try:
    from train import BabyGPT, BabyGPTConfig
except ImportError:
    BabyGPT = None

from model import BabyTransformerLM, BabyLMConfig
from dataset import SimpleBPETokenizer, format_chatml_conversation


class CaregiverMemoryStore:
    """
    Persistent Key-Value Knowledge Store for Caregiver Belief State.
    
    DUAL-LAYER BELIEF UPDATING ARCHITECTURE:
    ---------------------------------------
    Layer 1 (Neural In-Context Attention):
      Within the prompt window, the transformer model's causal attention layers dynamically
      route queries away from superseded keys and towards the most recent revision tokens.
      
    Layer 2 (Persistent External Grounding Hook):
      Extracts explicit entity assertions and corrections from user messages into a persistent
      dictionary, saves to 'checkpoints/caregiver_memory.json', and injects active facts
      directly into the system prompt prefix.
      
      Why this is critical:
      - Cross-Session Continuity: Belief updates persist across terminal sessions.
      - Truncation Immunity: When dialogue turns are pruned to stay within the context window,
        verified facts remain permanently anchored in the system instruction turn.
      - Deterministic Grounding: The model always knows the ground-truth attributes even when
        multiple conflicting assertions occurred earlier.
    """
    def __init__(self, memory_file: str = "checkpoints/caregiver_memory.json"):
        self.memory_file = memory_file
        self.facts: Dict[str, str] = {}
        self.load()

    def load(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    self.facts = json.load(f)
            except Exception:
                self.facts = {}
        else:
            self.facts = {}

    def save(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.facts, f, indent=2)
        except Exception as e:
            print(f"[!] Warning: Failed to save persistent memory: {e}")

    def clear(self):
        self.facts = {}
        if os.path.exists(self.memory_file):
            try:
                os.remove(self.memory_file)
            except Exception:
                pass

    def extract_and_update(self, text: str) -> Dict[str, Tuple[Optional[str], str]]:
        """
        Extract entity updates and corrections from user input.
        Returns a dict of modified attributes: {key: (old_value, new_value)}.
        """
        updates: Dict[str, Tuple[Optional[str], str]] = {}
        text_lower = text.lower()

        # Check for correction indicators
        is_correction = any(marker in text_lower for marker in [
            "actually", "wait, no", "wait no", "typo", "mistake", "correction",
            "not a", "not my", "meant", "changed my mind", "instead of"
        ])

        # 1. Pet Species Extraction (dog, cat, rabbit, puppy, parrot, hamster, turtle, kitten, bunny, bird)
        pet_species_pattern = r"^(dog|cat|rabbit|puppy|parrot|hamster|turtle|kitten|bunny|bird)$"
        if is_correction:
            # Pattern A: "Actually, my pet is a cat, not a dog" -> 'cat' overrides 'dog'
            m_corr = re.search(r"(?:actually|no|meant|is|have)\s+(?:my\s+pet\s+is\s+a\s+|a\s+|an\s+)?([a-z]+)[,\s]+not\s+(?:a\s+|an\s+)?([a-z]+)", text_lower)
            if m_corr and re.match(pet_species_pattern, m_corr.group(1).strip()):
                new_cand = m_corr.group(1).strip()
                old_val = self.facts.get("pet_type")
                self.facts["pet_type"] = new_cand
                updates["pet_type"] = (old_val, new_cand)
            else:
                # Pattern B: "Wait, no, it is a cat! I made a typo"
                m_sub = re.search(r"(?:pet is a|is a|it is a|have a|adopted a|meant a)\s+([a-z]+)", text_lower)
                if m_sub and re.match(pet_species_pattern, m_sub.group(1).strip()):
                    new_val = m_sub.group(1).strip()
                    old_val = self.facts.get("pet_type")
                    if old_val != new_val:
                        self.facts["pet_type"] = new_val
                        updates["pet_type"] = (old_val, new_val)
        else:
            m = re.search(r"(?:have a|have an|my pet is a|adopted a|own a)\s+([a-z]+)", text_lower)
            if m and re.match(pet_species_pattern, m.group(1).strip()):
                new_val = m.group(1).strip()
                old_val = self.facts.get("pet_type")
                if old_val != new_val:
                    self.facts["pet_type"] = new_val
                    updates["pet_type"] = (old_val, new_val)

        # 2. Pet Name Extraction
        # e.g. "her name is Minik", "his name is Milo", "call her Bella", "named Rio", "pet's name is Luna"
        m_name = re.search(r"(?:(?:her|his|its|my pet's)\s+name\s+is|named|call (?:her|him|it))\s+([A-Z][a-zA-Z]+)", text)
        if m_name:
            new_name = m_name.group(1).strip()
            old_name = self.facts.get("pet_name")
            if old_name != new_name:
                self.facts["pet_name"] = new_name
                updates["pet_name"] = (old_name, new_name)

        # 3. Favorite Color Extraction
        m_col = re.search(r"(?:favorite color is|love the color|prefer)\s+([a-z]+)", text_lower)
        color_choices = {"blue", "green", "yellow", "red", "purple", "pink", "golden", "orange", "turquoise", "lavender", "white"}
        if m_col and m_col.group(1) in color_choices:
            new_col = m_col.group(1)
            old_col = self.facts.get("favorite_color")
            if old_col != new_col:
                self.facts["favorite_color"] = new_col
                updates["favorite_color"] = (old_col, new_col)

        # 4. Favorite Food/Snack Extraction
        m_food = re.search(r"(?:favorite (?:snack|food|treat) is|prefer)\s+([a-z]+)", text_lower)
        if m_food:
            food_cand = m_food.group(1)
            old_food = self.facts.get("favorite_food")
            if old_food != food_cand:
                self.facts["favorite_food"] = food_cand
                updates["favorite_food"] = (old_food, food_cand)

        # 5. Sibling Name Extraction
        m_sib = re.search(r"(?:my (?:sister|brother)(?:'s name is|\s+is|\s+))\s*([A-Z][a-zA-Z]+)", text)
        if m_sib:
            new_sib = m_sib.group(1).strip()
            old_sib = self.facts.get("sibling_name")
            if old_sib != new_sib:
                self.facts["sibling_name"] = new_sib
                updates["sibling_name"] = (old_sib, new_sib)

        if updates:
            self.save()

        return updates

    def get_system_prompt_prefix(self, base_prompt: str) -> str:
        """Injects verified facts directly into the system prompt."""
        if not self.facts:
            return base_prompt
        fact_items = [f"{k.replace('_', ' ')}: {v}" for k, v in sorted(self.facts.items())]
        facts_summary = "; ".join(fact_items)
        return (
            f"{base_prompt} "
            f"Known verified facts about your caregiver (always prioritize these): [{facts_summary}]."
        )


def encode_text(tokenizer, text: str) -> List[int]:
    """Helper to encode text whether tokenizer is Tokenizer or SimpleBPETokenizer."""
    res = tokenizer.encode(text)
    return res.ids if hasattr(res, "ids") else res


def decode_tokens(tokenizer, tokens: List[int]) -> str:
    """Helper to decode tokens whether tokenizer is Tokenizer or SimpleBPETokenizer."""
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(tokens)
    return ""


def truncate_history_to_budget(
    history: List[Dict[str, str]],
    tokenizer,
    max_budget: int = 440
) -> List[Dict[str, str]]:
    """
    Context Window Management with Graceful Old-Turn Truncation:
    Preserves the primary system instruction (history[0]) while dropping the oldest
    (user, assistant) interaction pairs until the tokenized prompt safely fits within 512 tokens.
    """
    if len(history) <= 1:
        return history

    system_turn = history[0] if history[0]["role"] == "system" else None
    dialogue_turns = history[1:] if system_turn else history[:]

    while dialogue_turns:
        candidate = ([system_turn] if system_turn else []) + dialogue_turns
        formatted = format_chatml_conversation(candidate, add_generation_prompt=True)
        tokens = encode_text(tokenizer, formatted)
        if len(tokens) <= max_budget:
            return candidate
        # Drop oldest pair (user and its corresponding assistant response)
        dialogue_turns = dialogue_turns[2:] if len(dialogue_turns) >= 2 else []

    return [system_turn] if system_turn else []


def load_model_and_tokenizer(checkpoint_dir: str = "checkpoints", device: str = "cpu"):
    # 1. Check for baby_model_best.pt from train.py
    baby_ckpt_candidates = [
        os.path.join(checkpoint_dir, "baby_model_best.pt"),
        os.path.join(ROOT_DIR, "checkpoints", "baby_model_best.pt"),
    ]
    tok_candidates = [
        os.path.join("tokenizer", "tokenizer.json"),
        os.path.join(ROOT_DIR, "tokenizer", "tokenizer.json"),
    ]

    baby_ckpt = next((p for p in baby_ckpt_candidates if os.path.exists(p)), None)
    tok_path = next((p for p in tok_candidates if os.path.exists(p)), None)

    if baby_ckpt and tok_path and BabyGPT is not None:
        print(f"[*] Loading BabyGPT from '{baby_ckpt}'...")
        tokenizer = Tokenizer.from_file(tok_path)
        model = BabyGPT.from_checkpoint(baby_ckpt, device=device)
        model.eval()
        cfg = model.cfg
        # Add stop token ID 3 (<|im_end|>)
        setattr(tokenizer, "im_end_id", 3)
        return model, tokenizer, cfg

    # 2. Fallback to older babylm_sft_weights.pt if present
    checkpoint_path = os.path.join(checkpoint_dir, "babylm_sft_weights.pt")
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")

    if not os.path.exists(checkpoint_path) or not os.path.exists(tokenizer_path):
        print("[!] No trained checkpoint found.")
        print(f"[!] Please run 'python train.py' first to train the model.")
        sys.exit(1)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    tokenizer = SimpleBPETokenizer.from_dict(tok_data)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: BabyLMConfig = ckpt["config"]
    model = BabyTransformerLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, tokenizer, cfg


def interactive_chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("  BabyLM Multi-Turn Interactive Inference Session")
    print("  Belief Updating, Recency Overriding & Persistent Fact Store")
    print("=" * 68)
    print(f"[*] Loading model on device: {device.upper()}...")

    model, tokenizer, cfg = load_model_and_tokenizer(device=device)
    print(f"[*] Model loaded! Context length: {cfg.max_seq_len} tokens | Vocab: {cfg.vocab_size}")
    print("[*] Special commands:")
    print("    - 'exit' / 'quit' : End session")
    print("    - 'clear'         : Reset current dialogue history")
    print("    - 'memory'        : View active persistent facts")
    print("    - 'clear memory'  : Reset external persistent knowledge store\n")

    # Initialize external persistent memory store
    memory_store = CaregiverMemoryStore()
    if memory_store.facts:
        print(f"[*] Loaded existing caregiver memory ({len(memory_store.facts)} facts): {memory_store.facts}")

    base_system_prompt = (
        "You are Monika, a caring and attentive baby companion. "
        "You pay close attention to your caregiver, remember facts they share, "
        "and immediately update your memory when they correct a fact."
    )

    # Persistent history buffer with dynamic fact injection
    history: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": memory_store.get_system_prompt_prefix(base_system_prompt)
        }
    ]

    max_gen_tokens = 50
    # Reserve room for generated tokens within max_seq_len
    max_prompt_budget = max(40, cfg.max_seq_len - max_gen_tokens - 8)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat session. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history = [{"role": "system", "content": memory_store.get_system_prompt_prefix(base_system_prompt)}]
            print("[*] Dialogue history cleared. Persistent facts retained.")
            continue
        if user_input.lower() == "clear memory":
            memory_store.clear()
            history[0]["content"] = memory_store.get_system_prompt_prefix(base_system_prompt)
            print("[*] Persistent memory store cleared!")
            continue
        if user_input.lower() == "memory":
            print(f"[*] Current Verified Facts: {json.dumps(memory_store.facts, indent=2)}")
            continue

        # Extract entity assertions / corrections and update persistent store
        updates = memory_store.extract_and_update(user_input)
        if updates:
            for k, (old_val, new_val) in updates.items():
                if old_val:
                    print(f"[*] [Belief Updated] {k}: '{old_val}' -> '{new_val}' (Overwrote superseded fact)")
                else:
                    print(f"[*] [Fact Learned] {k} = '{new_val}'")

            # Dynamically refresh system prompt with updated ground-truth facts
            history[0]["content"] = memory_store.get_system_prompt_prefix(base_system_prompt)

        # Append new user turn to persistent conversation history
        history.append({"role": "user", "content": user_input})

        # Graceful context window truncation (preserves history[0] system prompt)
        active_history = truncate_history_to_budget(history, tokenizer, max_budget=max_prompt_budget)

        # Format ChatML with assistant generation prompt (<|im_start|>assistant\n)
        prompt_str = format_chatml_conversation(active_history, add_generation_prompt=True)
        prompt_tokens = encode_text(tokenizer, prompt_str)
        input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            if hasattr(model, "generate") and "stop_token_id" in model.generate.__code__.co_varnames:
                output_tokens = model.generate(
                    input_tensor,
                    max_new_tokens=max_gen_tokens,
                    temperature=0.75,
                    top_k=40,
                    stop_token_id=3,
                )
            else:
                output_tokens = model.generate(
                    input_tensor,
                    max_new_tokens=max_gen_tokens,
                    temperature=0.6,
                    top_k=25,
                    top_p=0.9,
                    stop_token_ids=[getattr(tokenizer, "im_end_id", 3)],
                )

        # Extract only the newly generated assistant tokens
        generated_ids = output_tokens[0][len(prompt_tokens):].tolist()
        assistant_reply = decode_tokens(tokenizer, generated_ids).strip()

        # Clean trailing delimiter artifacts if any
        assistant_reply = assistant_reply.replace("<|im_end|>", "").strip()

        print(f"Monika: {assistant_reply}")

        # Store assistant response in persistent memory for the next turn
        history.append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    interactive_chat()

import re
import os
import json
import random
import unicodedata
from typing import List, Dict, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset

class TextCleaner:
    """Rigorous cleaning pipeline to filter out corrupted text, web junk, and syntax fragments."""
    
    @staticmethod
    def clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"http[s]?://\S+", " ", text)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def is_high_quality(text: str, min_words: int = 6, max_rep_ratio: float = 0.4) -> bool:
        words = text.split()
        if len(words) < min_words:
            return False

        alpha_chars = sum(1 for c in text if c.isalpha())
        if len(text) > 0 and (alpha_chars / len(text)) < 0.6:
            return False

        word_counts = {}
        for w in words:
            w_lower = w.lower()
            word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
        most_common = max(word_counts.values()) if word_counts else 0
        if (most_common / len(words)) > max_rep_ratio:
            return False

        return True


class SimpleBPETokenizer:
    """
    Byte-level Tokenizer with ChatML special token support (<|im_start|>, <|im_end|>).
    Supports special token preservation, vocabulary training, and serialization.
    """
    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"
        
        self.special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
            self.im_start_token,
            self.im_end_token,
        ]
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.special_tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self.special_tokens)}
        
        # Seed initial single-byte vocabulary (characters 0-255)
        for b in range(256):
            ch = chr(b)
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def im_start_id(self) -> int:
        return self.token_to_id[self.im_start_token]

    @property
    def im_end_id(self) -> int:
        return self.token_to_id[self.im_end_token]

    def build_vocab_from_texts(self, texts: List[str], target_vocab_size: Optional[int] = None):
        """Train frequent word/subword pieces from dataset up to target vocabulary budget."""
        target_size = target_vocab_size or self.vocab_size
        word_counts = {}
        for text in texts:
            # Strip special tokens so they are not fragmented during counting
            clean_text = text
            for st in self.special_tokens:
                clean_text = clean_text.replace(st, " ")
            words = re.findall(r"\w+|[^\w\s]", clean_text, re.UNICODE)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1

        sorted_words = sorted(word_counts.keys(), key=lambda w: word_counts[w], reverse=True)
        for w in sorted_words:
            if len(self.token_to_id) >= target_size:
                break
            if w not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[w] = idx
                self.id_to_token[idx] = w

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Tokenize string into integer token IDs, preserving ChatML delimiters atomically.
        """
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)

        # Regex split that treats special tokens as atomic units
        special_pattern = r"(<\|im_start\|>|<\|im_end\|>|<pad>|<bos>|<eos>|<unk>)"
        parts = re.split(special_pattern, text)

        for part in parts:
            if not part:
                continue
            if part in self.token_to_id:
                tokens.append(self.token_to_id[part])
            else:
                words = re.findall(r"\w+|[^\w\s]", part, re.UNICODE)
                for w in words:
                    if w in self.token_to_id:
                        tokens.append(self.token_to_id[w])
                    else:
                        for ch in w:
                            tokens.append(self.token_to_id.get(ch, self.unk_id))

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        words = []
        for tid in token_ids:
            if skip_special_tokens and tid in self.id_to_token:
                val = self.id_to_token[tid]
                if val in self.special_tokens:
                    continue
            token_str = self.id_to_token.get(tid, "")
            words.append(token_str)

        raw = " ".join(words)
        raw = re.sub(r"\s+([.,!?;:])", r"\1", raw)
        raw = re.sub(r"([(\[{])\s+", r"\1", raw)
        raw = re.sub(r"\s+<\|im_end\|>", "<|im_end|>", raw)
        raw = re.sub(r"<\|im_start\|>\s+", "<|im_start|>", raw)
        return raw.strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "id_to_token": {str(k): v for k, v in self.id_to_token.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleBPETokenizer":
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        return tok


def format_chatml_turn(role: str, content: str) -> str:
    """Format an individual message in standard ChatML format."""
    return f"<|im_start|>{role}\n{content.strip()}<|im_end|>\n"


def format_chatml_conversation(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """
    Format a multi-turn conversation into a unified ChatML sequence.
    Example:
      <|im_start|>system\nYou are Monika...<|im_end|>\n
      <|im_start|>user\nYou know I have a dog.<|im_end|>\n
      <|im_start|>assistant\nA dog! What is their name?<|im_end|>\n
    """
    formatted = "".join(format_chatml_turn(m["role"], m["content"]) for m in messages)
    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"
    return formatted


class MultiTurnDialogueDataset(Dataset):
    """
    Multi-Turn Supervised Fine-Tuning (SFT) Dataset with LOSS MASKING.
    
    KEY LEARNING PRINCIPLE:
    - User tokens, system instructions, and role delimiters (<|im_start|>, <|im_end|>) are labeled as -100.
    - Loss is calculated ONLY on the assistant's response tokens.
    - When the assistant generates 'she' or 'Minik', the loss gradient encourages the attention heads
      to attend back to earlier turns (e.g. 'I have a dog', 'her name is Minik').
    - This forces the network to learn a generalizable key-value anaphora lookup rather than wasting
      parameter budget memorizing user prompts.
    """
    def __init__(
        self,
        dialogues: List[List[Dict[str, str]]],
        tokenizer: SimpleBPETokenizer,
        max_seq_len: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[int], List[int]]] = []

        for dialogue in dialogues:
            input_ids: List[int] = []
            labels: List[int] = []

            for turn in dialogue:
                role = turn["role"]
                content = turn["content"].strip()

                if role in ("system", "user"):
                    # Prompt turns: entire turn is masked from loss (label = -100)
                    turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    turn_tokens = self.tokenizer.encode(turn_str)
                    input_ids.extend(turn_tokens)
                    labels.extend([-100] * len(turn_tokens))
                elif role == "assistant":
                    # Assistant turn: role header is masked; response + im_end are supervised
                    header_str = "<|im_start|>assistant\n"
                    body_str = f"{content}<|im_end|>\n"

                    header_tokens = self.tokenizer.encode(header_str)
                    body_tokens = self.tokenizer.encode(body_str)

                    input_ids.extend(header_tokens + body_tokens)
                    # Mask header, train on body
                    labels.extend([-100] * len(header_tokens) + body_tokens)

            if len(input_ids) >= 4:
                self.samples.append((input_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = self.samples[idx]

        # Truncate if exceeds max_seq_len + 1 (needed for causal shift: input = x[:-1], target = y[1:])
        if len(input_ids) > self.max_seq_len + 1:
            input_ids = input_ids[: self.max_seq_len + 1]
            labels = labels[: self.max_seq_len + 1]

        # Causal shift target alignment:
        # At position t, inputs[t] = token_t, targets[t] = token_{t+1}
        inputs = input_ids[:-1]
        targets = labels[1:]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(inputs)
        if pad_len > 0:
            inputs = inputs + [self.tokenizer.pad_id] * pad_len
            targets = targets + [-100] * pad_len  # Pad tokens are masked from loss

        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


class TinyStoriesDataset(Dataset):
    """Pretraining dataset for causal language modeling."""
    def __init__(self, texts: List[str], tokenizer: SimpleBPETokenizer, max_seq_len: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        cleaner = TextCleaner()
        for raw in texts:
            cleaned = cleaner.clean(raw)
            if cleaner.is_high_quality(cleaned):
                tokens = self.tokenizer.encode(cleaned, add_bos=True, add_eos=True)
                if len(tokens) >= 4:
                    self.samples.append(tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.samples[idx]
        if len(tokens) > self.max_seq_len + 1:
            tokens = tokens[: self.max_seq_len + 1]

        input_ids = tokens[:-1]
        targets = tokens[1:]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            targets = targets + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
        )


def generate_entity_tracking_dialogues(num_samples: int = 400) -> List[List[Dict[str, str]]]:
    """
    Synthesize multi-turn dialogues with explicit entity tracking and pronoun reference resolution.
    
    Covers:
    1. Pet entities with gender pronouns and names (e.g., 'dog' -> 'Minik' -> 'her/she')
    2. Family/friend entities with activities (e.g., 'sister' -> 'Sarah' -> 'she')
    3. Favorite items/objects with neuter pronouns (e.g., 'blanket' -> 'it')
    4. Follow-up memory questions requiring retrieval across previous turns.
    """
    pet_entities = [
        {"kind": "dog", "name": "Minik", "subj": "she", "obj": "her", "poss": "her", "cap_subj": "She", "activities": ["loves chasing tennis balls", "sleeps next to my crib", "barks softly at birds", "cuddles on the sofa"]},
        {"kind": "cat", "name": "Milo", "subj": "he", "obj": "him", "poss": "his", "cap_subj": "He", "activities": ["purrs when petted", "chases colorful yarn", "takes sunny naps on the rug", "watches butterflies from the window"]},
        {"kind": "puppy", "name": "Bella", "subj": "she", "obj": "her", "poss": "her", "cap_subj": "She", "activities": ["chews little rubber toys", "runs in happy circles", "gives warm puppy kisses", "sleeps in a wicker basket"]},
        {"kind": "parrot", "name": "Rio", "subj": "he", "obj": "him", "poss": "his", "cap_subj": "He", "activities": ["whistles sweet tunes", "mimics cheerful greetings", "eats crisp apple slices", "perches on my shoulder"]},
        {"kind": "rabbit", "name": "Bunbun", "subj": "she", "obj": "her", "poss": "her", "cap_subj": "She", "activities": ["hops around the playroom", "munches crunchy carrots", "wiggles her soft nose", "enjoys gentle ear scratches"]},
        {"kind": "hamster", "name": "Peanut", "subj": "he", "obj": "him", "poss": "his", "cap_subj": "He", "activities": ["runs on his exercise wheel", "stuffs seeds in his cheeks", "burrows in cedar shavings", "naps during the day"]},
    ]

    family_entities = [
        {"relation": "sister", "name": "Sarah", "subj": "she", "obj": "her", "cap_subj": "She", "hobby": "bakes chocolate chip cookies"},
        {"relation": "brother", "name": "Leo", "subj": "he", "obj": "him", "cap_subj": "He", "hobby": "builds tall wooden towers"},
        {"relation": "grandmother", "name": "Elena", "subj": "she", "obj": "her", "cap_subj": "She", "hobby": "knits cozy wool blankets"},
        {"relation": "friend", "name": "Maya", "subj": "she", "obj": "her", "cap_subj": "She", "hobby": "paints colorful garden pictures"},
    ]

    object_entities = [
        {"item": "blanket", "color": "soft blue", "location": "in the crib", "feature": "keeps you warm and snug"},
        {"item": "mobile", "color": "golden celestial", "location": "above the crib", "feature": "spins stars and crescent moons gently"},
        {"item": "teddy bear", "color": "fuzzy brown", "location": "on the rocking chair", "feature": "has velvety ears and a red bow"},
        {"item": "storybook", "color": "silver starry", "location": "on the nightstand", "feature": "tells tales of kind little animals"},
    ]

    intro_phrasings = [
        "You know I have a {kind}.",
        "I adopted a sweet {kind} recently.",
        "We have a lovely {kind} at home.",
        "My family has a little {kind}.",
        "I wanted to tell you about my {kind}."
    ]

    name_phrasings = [
        "{poss_cap} name is {name}.",
        "We call {obj} {name}.",
        "{cap_subj} is named {name}.",
        "Her name is {name}." if "{poss_cap}" == "Her" else "His name is {name}."
    ]

    dialogues: List[List[Dict[str, str]]] = []
    random.seed(42)

    for i in range(num_samples):
        d_type = i % 3
        system_msg = {"role": "system", "content": "You are Monika, a caring and perceptive baby assistant who tracks conversational entities and learns about your caregiver's life."}

        if d_type == 0:
            # Multi-turn Pet Tracking with Pronouns & Entity Linking
            pet = random.choice(pet_entities)
            intro = random.choice(intro_phrasings).format(kind=pet["kind"])
            name_p = random.choice(name_phrasings).format(
                poss_cap=pet["poss"].capitalize(),
                name=pet["name"],
                obj=pet["obj"],
                cap_subj=pet["cap_subj"]
            )
            act = random.choice(pet["activities"])

            d = [
                system_msg,
                {"role": "user", "content": intro},
                {"role": "assistant", "content": f"A {pet['kind']}! Animals are so wonderful. What is their name?"},
                {"role": "user", "content": name_p},
                {"role": "assistant", "content": f"{pet['name']}! That is such a darling name for a {pet['kind']}. Does {pet['subj']} like to play with you?"},
                {"role": "user", "content": f"{pet['cap_subj']} {act}."},
                {"role": "assistant", "content": f"Aww, {pet['name']} sounds so delightful! Give {pet['obj']} a gentle pet for me."},
            ]

            # In half the dialogues, add a 4th turn testing direct entity memory
            if random.random() < 0.6:
                q = random.choice([
                    f"Who is {pet['name']}?",
                    f"Do you remember my {pet['kind']}'s name?",
                    f"Tell me about {pet['name']}."
                ])
                if "Who is" in q or "Tell me" in q:
                    ans = f"{pet['name']} is your {pet['kind']}! You told me that {pet['subj']} {act}."
                else:
                    ans = f"{pet['name']}! You told me that {pet['name']} is your {pet['kind']}."
                d.extend([
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans}
                ])
            dialogues.append(d)

        elif d_type == 1:
            # Multi-turn Family/Friend Entity Tracking
            fam = random.choice(family_entities)
            d = [
                system_msg,
                {"role": "user", "content": f"My {fam['relation']} {fam['name']} is visiting today."},
                {"role": "assistant", "content": f"How wonderful! What does {fam['name']} enjoy doing?"},
                {"role": "user", "content": f"{fam['cap_subj']} {fam['hobby']}."},
                {"role": "assistant", "content": f"That sounds lovely! Please tell {fam['obj']} hello from me."},
                {"role": "user", "content": f"What does {fam['name']} like to do?"},
                {"role": "assistant", "content": f"{fam['cap_subj']} {fam['hobby']}! I remember what you told me."},
            ]
            dialogues.append(d)

        else:
            # Multi-turn Object & Anaphora ('it') Resolution
            obj = random.choice(object_entities)
            d = [
                system_msg,
                {"role": "user", "content": f"I bought a {obj['color']} {obj['item']} today."},
                {"role": "assistant", "content": f"A {obj['item']}! Where did you place it?"},
                {"role": "user", "content": f"I placed it {obj['location']}."},
                {"role": "assistant", "content": f"That is a cozy spot for it. How does it look?"},
                {"role": "user", "content": f"It {obj['feature']}."},
                {"role": "assistant", "content": f"The {obj['color']} {obj['item']} makes the nursery feel so warm and safe."},
            ]
            dialogues.append(d)

    return dialogues


def generate_belief_correction_dialogues(num_samples: int = 360) -> List[List[Dict[str, str]]]:
    """
    Synthesize multi-turn dialogues featuring explicit belief updating and recency overriding.
    
    ATTENTION & LOSS MASKING LOGIC FOR BELIEF UPDATING:
    When a user introduces a fact ('I have a dog') and later revises it
    ('Actually, my pet is a cat, not a dog'), the context window contains conflicting keys:
      - Key 1: 'dog' at early position t_1
      - Key 2: 'cat' at later position t_3
      - Correction marker: 'Actually', 'not a dog', 'made a typo'
    
    Because loss is masked (-100) on all user prompt tokens and evaluated ONLY on
    assistant tokens, the model is penalized if it outputs the outdated attribute ('dog')
    and rewarded when it outputs the revised attribute ('cat').
    
    Through RoPE (Rotary Position Embeddings) and multi-head causal attention:
      1. Recency Bias: Tokens at larger position indices have lower relative distance to the query.
      2. Semantic Gating: Attention heads learn that correction triggers ('Actually', 'not')
         suppress obsolete entity keys and route query vectors to the revised token.
    """
    pet_corrections = [
        {"old": "dog", "new": "cat", "old_name": "Buddy", "new_name": "Milo", "trait": "purrs softly"},
        {"old": "cat", "new": "rabbit", "old_name": "Milo", "new_name": "Bunbun", "trait": "hops around gently"},
        {"old": "puppy", "new": "parrot", "old_name": "Bella", "new_name": "Rio", "trait": "whistles cheerful songs"},
        {"old": "hamster", "new": "turtle", "old_name": "Peanut", "new_name": "Shelly", "trait": "walks very slowly"},
        {"old": "rabbit", "new": "puppy", "old_name": "Bunny", "new_name": "Bella", "trait": "wags her tail happily"},
        {"old": "bird", "new": "kitten", "old_name": "Pip", "new_name": "Luna", "trait": "plays with yarn"},
        {"old": "dog", "new": "bunny", "old_name": "Max", "new_name": "Clover", "trait": "twitches her soft nose"},
        {"old": "parrot", "new": "dog", "old_name": "Kiwi", "new_name": "Minik", "trait": "fetches tennis balls"},
    ]

    color_corrections = [
        {"entity": "favorite color", "old": "green", "new": "blue"},
        {"entity": "favorite color", "old": "blue", "new": "yellow"},
        {"entity": "favorite color", "old": "red", "new": "purple"},
        {"entity": "favorite color", "old": "pink", "new": "golden"},
        {"entity": "favorite color", "old": "orange", "new": "turquoise"},
        {"entity": "room blanket color", "old": "white", "new": "lavender"},
    ]

    food_corrections = [
        {"entity": "favorite snack", "old": "apples", "new": "strawberries"},
        {"entity": "favorite snack", "old": "cookies", "new": "pancakes"},
        {"entity": "favorite treat", "old": "carrots", "new": "blueberries"},
        {"entity": "breakfast", "old": "toast", "new": "oatmeal"},
        {"entity": "favorite fruit", "old": "bananas", "new": "peaches"},
    ]

    name_corrections = [
        {"relation": "sister", "old": "Sarah", "new": "Emma"},
        {"relation": "brother", "old": "Leo", "new": "Noah"},
        {"relation": "best friend", "old": "Maya", "new": "Chloe"},
        {"relation": "grandmother", "old": "Elena", "new": "Clara"},
    ]

    # Diverse correction phrasings to learn generalizable recency overriding
    pet_correction_phrases = [
        "Actually, my pet is a {new}, not a {old}. I made a typo earlier!",
        "Wait, no, it is a {new}! I gave you the wrong animal before.",
        "Sorry, I meant a {new}, not a {old}.",
        "Correction: I actually have a sweet {new}.",
        "I mistyped earlier! My pet is definitely a {new}, not a {old}.",
        "Oh wait, actually it's a {new}, not a {old}. My mistake!",
        "Wait, I made an error before: my pet is a {new}.",
    ]

    assistant_ack_phrases = [
        "Oh, got it! A {new}! Thank you for correcting me, I will remember that you have a {new}.",
        "Understood! I updated my notes: you have a {new}, not a {old}.",
        "Ah, thank you for clarifying! I will keep in mind that your pet is a {new}.",
        "Noted! A {new}! I have updated my memory.",
        "I understand completely! Your pet is a {new}. I'll remember the correction.",
    ]

    dialogues: List[List[Dict[str, str]]] = []
    random.seed(1337)

    for i in range(num_samples):
        system_msg = {
            "role": "system",
            "content": "You are Monika, a caring baby companion. You pay close attention to facts your caregiver shares and immediately update your memory when they correct a fact."
        }

        mode = i % 5

        if mode == 0:
            # Pet species belief update + direct query
            pair = random.choice(pet_corrections)
            corr_phrase = random.choice(pet_correction_phrases).format(new=pair["new"], old=pair["old"])
            ack_phrase = random.choice(assistant_ack_phrases).format(new=pair["new"], old=pair["old"])
            
            queries = [
                "What kind of pet do I have?",
                "Do you remember what pet I have?",
                "Tell me about my pet.",
                "What animal do I have at home?",
            ]
            q = random.choice(queries)

            answers = [
                f"You have a {pair['new']}! You corrected that earlier, so I remember.",
                f"It is a {pair['new']}! You let me know you have a {pair['new']}, not a {pair['old']}.",
                f"You have a {pair['new']}! I updated my memory when you told me.",
                f"A {pair['new']}! I remember your correction.",
            ]
            ans = random.choice(answers)

            d = [
                system_msg,
                {"role": "user", "content": f"I have a little {pair['old']} at home."},
                {"role": "assistant", "content": f"A {pair['old']}! Animals are so wonderful. How is your {pair['old']} doing?"},
                {"role": "user", "content": corr_phrase},
                {"role": "assistant", "content": ack_phrase},
                {"role": "user", "content": q},
                {"role": "assistant", "content": ans},
            ]
            dialogues.append(d)

        elif mode == 1:
            # Pet species belief update + Negative validation ('Do I have a dog?')
            pair = random.choice(pet_corrections)
            corr_phrase = random.choice(pet_correction_phrases).format(new=pair["new"], old=pair["old"])
            ack_phrase = random.choice(assistant_ack_phrases).format(new=pair["new"], old=pair["old"])

            neg_questions = [
                f"Do I have a {pair['old']}?",
                f"Is my pet a {pair['old']}?",
                f"Wait, did I tell you I have a {pair['old']}?",
            ]
            neg_q = random.choice(neg_questions)

            neg_answers = [
                f"No, you have a {pair['new']}! You corrected that earlier, telling me it's not a {pair['old']}.",
                f"No, you have a {pair['new']}! You let me know you made a typo and it is actually a {pair['new']}.",
                f"No, your pet is a {pair['new']}. I remember your correction.",
            ]
            neg_ans = random.choice(neg_answers)

            d = [
                system_msg,
                {"role": "user", "content": f"I adopted a {pair['old']} yesterday."},
                {"role": "assistant", "content": f"How sweet! A new {pair['old']} must be so playful."},
                {"role": "user", "content": corr_phrase},
                {"role": "assistant", "content": ack_phrase},
                {"role": "user", "content": neg_q},
                {"role": "assistant", "content": neg_ans},
            ]
            dialogues.append(d)

        elif mode == 2:
            # Color preference belief update
            col = random.choice(color_corrections)
            corr_phrases = [
                f"Actually, my {col['entity']} is {col['new']}, not {col['old']}.",
                f"Wait, no, my {col['entity']} is definitely {col['new']}! I made a typo.",
                f"Sorry, I meant {col['new']}, not {col['old']}.",
                f"Correction: my {col['entity']} is {col['new']}.",
            ]
            ack_phrases = [
                f"Oh, wonderful! {col['new'].capitalize()} is a gorgeous color. I have updated my memory.",
                f"Got it! I will remember that your {col['entity']} is {col['new']}.",
                f"Thank you for clarifying! I've noted that it is {col['new']}, not {col['old']}.",
            ]
            query_phrases = [
                f"What is my {col['entity']}?",
                f"Do you remember my {col['entity']}?",
                f"Which color do I like best?",
            ]
            ans_phrases = [
                f"Your {col['entity']} is {col['new']}! You corrected that earlier.",
                f"{col['new'].capitalize()}! You told me it is {col['new']}, not {col['old']}.",
                f"It is {col['new']}! I remember your update.",
            ]

            d = [
                system_msg,
                {"role": "user", "content": f"My {col['entity']} is {col['old']}."},
                {"role": "assistant", "content": f"{col['old'].capitalize()} is a peaceful color! Why do you like it?"},
                {"role": "user", "content": random.choice(corr_phrases)},
                {"role": "assistant", "content": random.choice(ack_phrases)},
                {"role": "user", "content": random.choice(query_phrases)},
                {"role": "assistant", "content": random.choice(ans_phrases)},
            ]
            dialogues.append(d)

        elif mode == 3:
            # Food / snack preference belief update
            fd = random.choice(food_corrections)
            corr_phrases = [
                f"Actually, my {fd['entity']} is {fd['new']}, not {fd['old']}.",
                f"Wait, no, it is {fd['new']}! I mistyped earlier.",
                f"Correction: I much prefer {fd['new']} over {fd['old']}.",
                f"Sorry, I meant {fd['new']}, not {fd['old']}.",
            ]
            ack_phrases = [
                f"Yum, {fd['new']}! That sounds delicious. I will remember that.",
                f"Noted! I have updated my memory: you love {fd['new']}, not {fd['old']}.",
                f"Understood! {fd['new'].capitalize()} is so yummy. I will remember your preference.",
            ]

            d = [
                system_msg,
                {"role": "user", "content": f"My {fd['entity']} is {fd['old']}."},
                {"role": "assistant", "content": f"Mmm, {fd['old']}! A tasty treat. Do you eat that often?"},
                {"role": "user", "content": random.choice(corr_phrases)},
                {"role": "assistant", "content": random.choice(ack_phrases)},
                {"role": "user", "content": f"What is my {fd['entity']}?"},
                {"role": "assistant", "content": f"Your {fd['entity']} is {fd['new']}! I remember your correction."},
            ]
            dialogues.append(d)

        else:
            # Person / Relation name belief update
            nm = random.choice(name_corrections)
            corr_phrases = [
                f"Actually, my {nm['relation']}'s name is {nm['new']}, not {nm['old']}. I made a typo!",
                f"Wait, no, her name is {nm['new']}!" if "sister" in nm["relation"] or "friend" in nm["relation"] else f"Wait, no, his name is {nm['new']}!",
                f"Correction: my {nm['relation']} is named {nm['new']}.",
                f"Sorry, I meant {nm['new']}, not {nm['old']}.",
            ]
            ack_phrases = [
                f"Oh, thank you for telling me! {nm['new']} is a lovely name. I will remember that.",
                f"Understood! I have updated my memory: your {nm['relation']} is {nm['new']}.",
                f"Got it! Hello to {nm['new']}! I will remember the correct name.",
            ]

            d = [
                system_msg,
                {"role": "user", "content": f"My {nm['relation']} {nm['old']} is coming over."},
                {"role": "assistant", "content": f"How nice! What do you and {nm['old']} like to do together?"},
                {"role": "user", "content": random.choice(corr_phrases)},
                {"role": "assistant", "content": random.choice(ack_phrases)},
                {"role": "user", "content": f"What is my {nm['relation']}'s name?"},
                {"role": "assistant", "content": f"Your {nm['relation']}'s name is {nm['new']}! You corrected that earlier, and I remember."},
            ]
            dialogues.append(d)

    return dialogues


def get_curated_sample_stories() -> List[str]:
    """Curated high-syntactic quality nursery stories for baseline vocabulary initialization."""
    return [
        "Once upon a time, there was a little bird named Pip. Pip loved to fly over the green meadow. Every morning, Pip sang a cheerful song to the rising sun.",
        "The night sky was dark and full of wonder. Millions of distant stars sparkled like diamonds. The moon looked down warmly at the quiet world below.",
        "A small kitten walked softly across the wooden floor. She found a ball of red yarn and played with it all afternoon. When she grew tired, she curled up to sleep.",
        "Water is very important for all living things. Rivers and rain give drinks to tall trees and tiny flowers. Without fresh water, gardens could not grow green.",
        "Leo and his mother went to the quiet park. They sat on a wooden bench beneath an oak tree. Leo read his favorite storybook aloud and learned new words.",
        "The sun rose high above the golden hills. Birds chirped in the warm air, and bees danced around purple blossoms. It was a lovely, peaceful morning.",
        "A little girl named Lily found a hidden treasure box in the garden. Inside the box was a golden key and a sweet note from her grandfather.",
        "Every evening, the mother held her baby close and rocked the crib gently. The nursery was calm and warm, and the baby fell into sweet dreams.",
    ]

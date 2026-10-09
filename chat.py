#!/usr/bin/env python3
"""
chat.py
=======
Inference for the Baby LLM: terminal chat, and the HTTP backend for the nursery (index.html).

How the baby "learns" at runtime:
  A 14M-parameter model cannot reliably absorb new facts into its weights one sentence at a
  time (it would overwrite old knowledge and mostly memorize wording). Instead, what you teach
  goes into an explicit Memory, and the model was trained (sft.py) to *reason over* that memory:
  look facts up, chain two of them, flip I/you, admit when it doesn't know, prefer corrections.
  Each request retrieves the facts most relevant to what you just said and places them in
  the system prompt in exactly the format used during training (sft_data.py).

Usage:
  python chat.py                    # terminal chat (/teach, /feed, /cuddle, /nap, /memory)
  python chat.py --server           # serve the nursery at http://localhost:8765
"""

import os
import re
import sys
import json
import time
import argparse
import random
import threading
from typing import List, Dict, Optional, Tuple

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from tokenizers import Tokenizer
from train import BabyGPT
from sft_data import CARE_EVENTS, build_system_prompt, format_chatml, mood_from_needs

STOP_TOKEN_ID = 3  # <|im_end|>
CHAT_CHECKPOINT = "checkpoints/baby_chat.pt"
BASE_CHECKPOINT = "checkpoints/baby_model_best.pt"
STOPWORDS = set("the a an and or to of in is are you i it we my your be do does did was for on at that this with what who "
                "where how why can me am".split())


# ==============================================================================
# 1. Loading
# ==============================================================================

def resolve_checkpoint(path: Optional[str]) -> str:
    if path:
        return path
    if os.path.exists(CHAT_CHECKPOINT):
        return CHAT_CHECKPOINT
    print(f"[!] {CHAT_CHECKPOINT} not found. Using the story-only model; it will ramble.")
    print("    Run `python sft.py` to teach it to converse and reason over memory.")
    return BASE_CHECKPOINT


def load_chat_pipeline(checkpoint_path: Optional[str] = None, tokenizer_path: str = "tokenizer/tokenizer.json",
                       device: Optional[str] = None) -> Tuple[BabyGPT, Tokenizer, torch.device]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = resolve_checkpoint(checkpoint_path)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    model = BabyGPT.from_checkpoint(checkpoint_path, device=str(dev))
    model.eval()
    print(f"[✓] Loaded {checkpoint_path} ({model.count_parameters()['total'] / 1e6:.1f}M params) on {dev}")
    return model, tokenizer, dev


# ==============================================================================
# 2. Memory retrieval + prompt building
# ==============================================================================

MATCH_STOPWORDS = STOPWORDS | {"have", "has", "had", "know", "tell", "about", "please", "yes", "no", "they", "them",
                               "there", "very", "really", "remember", "teach", "did"}


def _stem(w: str) -> str:
    """Crude singular form so "stars" finds "star" and "berries" finds "berry"."""
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _words(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z']+", text.lower()) if w not in MATCH_STOPWORDS and len(w) > 1}


def select_facts(facts: List[str], query: str, k: int = 8, focus: str = "") -> List[str]:
    """
    Retrieval is done here, not by the model. When the caregiver's message names something in
    memory, the model sees ONLY those facts plus facts linked to them (so "What does Rex say?" also
    gets "A dog says woof" via "Rex is a dog"). Fewer distractors = it copies the right fact.
    With no match, the most recent facts are shown so it can chat or say it doesn't know.
    """
    order = {f: i for i, f in enumerate(facts)}
    q = _words(focus or query)
    scored = sorted(((len(_words(f) & q), i, f) for i, f in enumerate(facts)), key=lambda t: (-t[0], -t[1]))
    hits = [f for s, _, f in scored if s > 0][:4]
    if hits:
        linked = set().union(*(_words(f) for f in hits)) - q
        for f in reversed(facts):  # one hop out from the hits
            if len(hits) >= 6:
                break
            if f not in hits and _words(f) & linked:
                hits.append(f)
        return sorted(hits, key=lambda f: order[f])  # later facts (corrections) stay later
    if focus and query != focus:
        return select_facts(facts, query, k)  # maybe the topic is in the last exchange
    return facts[-k:]


def build_prompt(tokenizer: Tokenizer, baby: str, parent: str, needs: Optional[Dict], facts: List[str],
                 history: List[Dict[str, str]], user_turn: str, budget: int) -> List[int]:
    query = user_turn + " " + " ".join(m["content"] for m in history[-2:])
    k = 8
    turns = list(history)
    while True:
        system = build_system_prompt(baby, parent, mood_from_needs(needs), select_facts(facts, query, k, focus=user_turn))
        msgs = [{"role": "system", "content": system}] + turns + [{"role": "user", "content": user_turn}]
        ids = tokenizer.encode(format_chatml(msgs, add_generation_prompt=True)).ids
        if len(ids) <= budget:
            return ids
        if len(turns) >= 2:
            turns = turns[2:]  # forget the oldest exchange first
        elif k > 2:
            k -= 2
        else:
            return ids[-budget:]


def clean_reply(text: str) -> str:
    text = text.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    text = text.split("\n")[0].strip()  # replies are one line in training
    return text or "*babbles happily*"


_known_words_cache: Dict[int, set] = {}


def known_words(tokenizer: Tokenizer) -> set:
    """Every whole word the tokenizer knows (e.g. 'bear', 'mathematics' pieces aside)."""
    key = id(tokenizer)
    if key not in _known_words_cache:
        words = set()
        for tok in tokenizer.get_vocab():
            w = tok[1:] if tok.startswith("Ġ") else tok
            if w.isalpha() and len(w) >= 2:
                words.add(w.lower())
        _known_words_cache[key] = words
    return _known_words_cache[key]


_template_words: Optional[set] = None


def template_words() -> set:
    """Every word the baby's own trained replies use ("gulp", "yawns", "zzz"...): never 'made up'."""
    global _template_words
    if _template_words is None:
        import random as _r
        from sft_data import CARE_EVENTS as _events, care_turn, chat_turn, teach_reply
        rng = _r.Random(0)
        words = set()
        for _ in range(400):
            mood = [m for m in ("hungry", "sleepy", "fussy", "bored") if rng.random() < 0.3]
            for ev in _events:
                words.update(w.lower() for w in re.findall(r"[A-Za-z]+", care_turn(ev, mood, "P", "B", rng)[1]))
            words.update(w.lower() for w in re.findall(r"[A-Za-z]+", chat_turn(mood, "P", "B", rng)[1]))
            words.update(w.lower() for w in re.findall(r"[A-Za-z]+", teach_reply("Zq", "P", rng)))
        for phrase in ("Hmm, I don't know about it yet. Can you teach me? Ooh I don't know yet. Teach me, please! "
                       "Will you tell me? I don't know what a is yet", "Oh! Got it! Okay, not a. Yes! No, so, because, "
                       "and, I remember. Hehe. Your my you have named is are can says eats lives in the favorite",
                       "I dreamed about it. I saw a by the window. I want to play with my. I really like. Guess what? "
                       "What a sweet name! I want to meet. Hi! Yummy! I want to try it too! Is it yummy? Was it fun? "
                       "Can I come next time? Yay! Let's play! Claps. Okay... later then. Pouts. Aww. I love you too! "
                       "Happy wiggle. Hugs teddy. I love it!"):
            words.update(w.lower() for w in re.findall(r"[A-Za-z]+", phrase))
        _template_words = words
    return _template_words


def ungrounded_words(reply: str, context: str, tokenizer: Tokenizer) -> List[str]:
    """
    Words the baby could only have made up: not said by anyone in the conversation, not in its
    memory, and not even a real word it knows. "Buqueageheimit is a bear!" -> ['Buqueageheimit'].
    """
    seen = {w.lower() for w in re.findall(r"[A-Za-z]+", context)} | template_words()
    vocab = known_words(tokenizer)
    return [w for w in re.findall(r"[A-Za-z]+", reply)
            if len(w) >= 4 and w.lower() not in seen and w.lower() not in vocab]


def repair_names(reply: str, context: str, bad: List[str]) -> Optional[str]:
    """
    Fix near-miss copies of words from memory/conversation: the model often clips or blurs long
    names it has to copy piece by piece ("Yasam" -> "Yasamin", "brorp" -> "glorp").
    Returns None if some invented word has no close match (a real hallucination).
    """
    import difflib
    ctx = {w.lower(): w for w in re.findall(r"[A-Za-z]+", context) if len(w) >= 3}
    for w in set(bad):
        match = difflib.get_close_matches(w.lower(), list(ctx), n=1, cutoff=0.7)
        if not match:
            return None
        fixed = ctx[match[0]]
        reply = re.sub(rf"\b{re.escape(w)}\b", fixed[0].upper() + fixed[1:] if w[0].isupper() else fixed, reply)
    return reply


def off_script_words(reply: str, context: str) -> List[str]:
    """
    The baby was only trained to answer with words from what it was told plus its own reply
    phrases. Any other content word means it invented something ("no" -> "Drum is a dog!"),
    even when that word is a real English word.
    """
    allowed = {_stem(w.lower()) for w in re.findall(r"[A-Za-z]+", context)}
    allowed |= {_stem(w) for w in template_words()} | FRAME_WORDS
    return [w for w in re.findall(r"[A-Za-z]+", reply)
            if len(w) >= 3 and w.lower() not in MATCH_STOPWORDS and _stem(w.lower()) not in allowed]


# The nursery answers teaching messages itself, so a model reply that claims to have learned
# something is always invented ("no" -> "Drum is a dog! Wow, I learned something new!")
TEACH_ACK = re.compile(r"learned something new|i will remember|i'll remember|keep it in my memory|"
                       r"in my memory now|thank you for teaching|thanks for teaching", re.I)

# Care reactions that contradict the baby's mood
CARE_CONTRADICTIONS = {
    ("*feeds you*", "hungry"): re.compile(r"not hungry|i'm full|no more milk|tiny sip", re.I),
    ("*tucks you in for a nap*", "sleepy"): re.compile(r"not sleepy|five more minutes", re.I),
}
CARE_ONLY_WHEN = {
    "*feeds you*": ("hungry", re.compile(r"tummy was so empty|yay, food", re.I)),
}


def contradicts_mood(reply: str, user_turn: str, mood: List[str]) -> bool:
    for (event, need), pattern in CARE_CONTRADICTIONS.items():
        if user_turn == event and need in mood and pattern.search(reply):
            return True
    rule = CARE_ONLY_WHEN.get(user_turn)
    return bool(rule and rule[0] not in mood and rule[1].search(reply))


def is_question(user_turn: str) -> bool:
    t = user_turn.strip().lower()
    return t.endswith("?") or bool(re.match(r"^(what|who|where|when|why|how|can|is|are|do|does|did|tell me)\b", t))


QUOTE = re.compile(r"^\s*(?:you said|did you say|you told me|didn't you say)(?: that)?(?: you)?\s+(.+)", re.I)
VERBISH = {"dream", "dreamed", "dreamt", "dreaming", "had", "saw", "see", "seen", "want", "wanted", "play", "playing",
           "like", "liked", "really", "about", "with", "window", "said", "say", "told"}
SHORT_WORDS = {"i", "a", "an", "oh", "ok", "hi", "no", "so", "to", "me", "my", "we", "be", "is", "it", "in", "on", "at",
               "up", "us", "do", "go", "he", "of", "or", "by", "am", "as", "if", "aw", "ah", "mm", "s", "t", "m", "ll", "re", "ve", "d"}
GREETING = re.compile(r"^\s*(hi|hello|hey|hiya|good morning|morning|yo)\b", re.I)

# an article followed by a pronoun or function word is broken grammar ("Yay is a we")
BROKEN_GRAMMAR = re.compile(r"\b(a|an|the)\s+(we|you|i|he|she|they|it|very|the|a|an|and|is|are|to|so|too|my|your)\b", re.I)


def own_words(history: List[Dict[str, str]]) -> List[str]:
    return [m["content"] for m in history[-6:] if m.get("role") == "assistant"]


def quote_check(reply: str, user_turn: str, history: List[Dict[str, str]]) -> str:
    """
    "You said you dreamed about X?": compare the quote with what the baby really said earlier.
    Agree only if it matches; otherwise correct the caregiver with the baby's actual words.
    """
    m = QUOTE.match(user_turn.strip().rstrip("?.!"))
    said = own_words(history)
    if not m or not said:
        return reply
    quoted = _words(m.group(1)) - {_stem(w) for w in VERBISH}
    if not quoted:
        return reply
    spoken = _words(" ".join(said))
    if quoted <= spoken:
        return reply if re.match(r"^\s*yes\b", reply, re.I) else "Yes! " + re.sub(r"^\s*(no|yes)\b[!.,]*\s*", "", reply, flags=re.I)
    # find the baby's own line that is being misquoted (the one sharing most words), quote it back
    original = max(said, key=lambda s: len(_words(s) & (_words(user_turn) | quoted)))
    original = re.sub(r"^(\*[^*]*\*\s*|hi!\s*|guess what\?\s*|ooh!\s*)+", "", original, flags=re.I).strip()
    if not original:
        return "No, I didn't say that!"
    if not re.match(r"^I\b", original):
        original = original[0].lower() + original[1:]
    return "No! I said: " + original


def answer_to_my_question(user_turn: str, history: List[Dict[str, str]], parent: str) -> Optional[str]:
    """A plain yes/no reply to the baby's own last question gets a fitting reaction."""
    said = own_words(history)
    if not said or not said[-1].strip().endswith("?"):
        return None
    t = user_turn.strip().lower()
    if re.match(r"^(yes|yeah|yep|sure|okay|ok|of course|let's|alright)\b", t):
        return random.choice(["Yay! *claps*", f"Yay! Thank you, {parent}!", "Hehe! Yay!"])
    if re.match(r"^(no|nope|not now|later|maybe later)\b", t):
        return random.choice(["Aww. Okay.", "Okay... later then. *pouts*"])
    return None


def identity_answer(user_turn: str, baby: str, parent: str) -> Optional[str]:
    """Name questions have one right answer, already known: never leave them to chance."""
    t = user_turn.lower()
    if re.search(r"\b(what'?s your name|what is your name|who are you|your name)\b", t):
        return random.choice([f"My name is {baby}! Hehe.", f"I'm {baby}!", f"{baby}! That's me!"])
    if re.search(r"\b(who am i|what'?s my name|what is my name|do you know my name)\b", t):
        return random.choice([f"You're {parent}!", f"You are {parent}, my parent!", f"{parent}! I know you!"])
    return None


def topic_of(user_turn: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z']+", user_turn) if w.lower() not in MATCH_STOPWORDS and len(w) > 2]
    return words[-1] if words else "that"


# Words the baby uses around facts ("Ooh! ... I remember.") that aren't claims about the world
FRAME_WORDS = {"ooh", "yes", "yay", "hehe", "remember", "so", "because", "and", "wow", "okay", "yeah", "too", "now", "know"}
PERSONAL = re.compile(r"\b(i|i'm|my|me|mine|you|your|you're|we|our)\b")


def memory_hits(facts: List[str], user_turn: str) -> List[str]:
    """Facts that share a content word with the caregiver's message (the retrieval's direct hits)."""
    q = _words(user_turn)
    return [f for f in facts if _words(f) & q] if q else []


def world_question_topic(user_turn: str) -> Optional[str]:
    """'What is a cloud?' / 'Tell me about stars' / 'Do you know mathematics?' -> the topic; else None."""
    t = user_turn.strip().lower()
    stripped = re.sub(r"^(do you know( about| what)?|tell me about|what do you know about)\s+", "", t)
    asks = stripped != t or re.match(r"^(what|who|where)\s+(is|are|does|do|was|were)\b", t)
    if not asks or PERSONAL.search(stripped):
        return None
    return topic_of(user_turn)


def fact_check(reply: str, hits: List[str], user_turn: str) -> str:
    """
    The model answers from memory, but small models drop or swap words in long facts
    ("Stars are bright suns far away" -> "Stars are bright suns far!"). If the answer doesn't
    faithfully contain a retrieved fact, or adds content found in none of them, say the fact exactly.
    """
    if not hits:
        return reply
    if re.search(r"don'?t know|not sure", reply, re.I):
        reply = ""  # memory has it, so "I don't know" is wrong: fall through to the exact fact
    rw = _words(reply)
    allowed = set().union(*(_words(f) for f in hits)) | _words(user_turn) | FRAME_WORDS
    best = max(hits, key=lambda f: len(_words(f) & rw))
    fw = _words(best)
    coverage = len(fw & rw) / max(1, len(fw))
    if coverage >= 0.99 and not (rw - allowed):  # every word of the fact, nothing invented
        return reply
    fact = best.rstrip(".!") + "!"
    if re.match(r"^\s*(can|is|are|do|does|did)\b", user_turn.lower()) and _words(user_turn) <= fw | MATCH_STOPWORDS:
        return "Yes! " + fact
    return random.choice(["Ooh! ", "I remember! ", ""]) + fact


@torch.no_grad()
def baby_reply(model: BabyGPT, tokenizer: Tokenizer, device, *, baby: str, parent: str, user_turn: str,
               facts: List[str] = (), history: List[Dict[str, str]] = (), needs: Optional[Dict] = None,
               max_new_tokens: int = 48, temperature: float = 0.6, tries: int = 3) -> Tuple[str, int, int]:
    facts = list(facts)
    known = identity_answer(user_turn, baby, parent)
    if known:
        return known, 0, 0
    hits = memory_hits(facts, user_turn)
    topic = world_question_topic(user_turn)
    if topic and not hits:
        # Nothing in memory mentions it: the honest answer is known without asking the model
        return random.choice([f"Hmm, I don't know about {topic} yet. Can you teach me, {parent}?",
                              f"Ooh, {topic}? I don't know yet. Teach me, please!",
                              f"I don't know {topic}! Will you tell me?"]), 0, 0

    budget = model.cfg.n_positions - max_new_tokens
    ids = build_prompt(tokenizer, baby, parent, needs, facts, list(history), user_turn, budget)
    context = " ".join([baby, parent, user_turn, *facts, *(m["content"] for m in history)])
    mood = mood_from_needs(needs)
    care_event = user_turn in CARE_EVENTS.values()
    # A plain remark ("okay", "you are so cute") shouldn't be answered by reciting memory
    # (the last exchange stays allowed, so the baby can follow up on its own question)
    recent = " ".join(m["content"] for m in history[-2:])
    check_context = context if (is_question(user_turn) or care_event) else " ".join([baby, parent, user_turn, recent])
    # Check answers to impersonal questions; personal ones flip I/you, so word overlap can't judge them
    impersonal_q = bool(topic) or (user_turn.strip().endswith("?") and not PERSONAL.search(user_turn.lower()))
    reply, gen = "", []
    for attempt in range(tries):
        temp = temperature if attempt == 0 else max(0.3, temperature - 0.2 * attempt)
        out = model.generate(torch.tensor([ids], device=device), max_new_tokens=max_new_tokens, temperature=temp,
                             top_k=30, top_p=0.9, repetition_penalty=1.15, stop_token_id=STOP_TOKEN_ID)
        gen = out[0, len(ids):].tolist()
        reply = clean_reply(tokenizer.decode(gen, skip_special_tokens=True))
        bad = ungrounded_words(reply, context, tokenizer) + off_script_words(reply, check_context)
        if bad:
            reply = repair_names(reply, context, bad) or reply
            bad = ungrounded_words(reply, context, tokenizer) + off_script_words(reply, check_context)
        fragments = [w for w in re.findall(r"[A-Za-z]+", reply) if len(w) <= 2 and w.lower() not in SHORT_WORDS]
        broken_action = (reply.count("*") % 2 == 1 or re.search(r"[A-Za-z]\*[A-Za-z]", reply)
                         or BROKEN_GRAMMAR.search(reply) or fragments)
        if TEACH_ACK.search(reply) or contradicts_mood(reply, user_turn, mood) or broken_action:
            bad = bad or ["<invented>"]
        if not bad:
            reply = fact_check(reply, hits if impersonal_q else [], user_turn)
            return quote_check(reply, user_turn, list(history)), len(ids), len(gen)
    # Every sample invented something. If memory has the answer, say it exactly; otherwise be honest
    if hits and impersonal_q:
        return fact_check("", hits, user_turn), len(ids), len(gen)
    if QUOTE.match(user_turn.strip()):
        return quote_check("", user_turn, list(history)) or "Hmm?", len(ids), len(gen)
    followup = answer_to_my_question(user_turn, list(history), parent)
    if followup:
        return followup, len(ids), len(gen)
    if GREETING.match(user_turn):
        return random.choice([f"Hi {parent}! *waves tiny hand*", f"Hello, {parent}! I missed you!", "Hiii! Hehe."]), len(ids), len(gen)
    if care_event:
        from sft_data import care_turn
        event = next(k for k, v in CARE_EVENTS.items() if v == user_turn)
        return care_turn(event, mood, parent, baby, random.Random())[1], len(ids), len(gen)
    if not (user_turn.strip().endswith("?") or topic):
        # a statement or "no"/"okay": react warmly instead of claiming not to know something
        return random.choice([f"Okay, {parent}! Hehe.", "Mm-hmm! *listens*", f"Hehe! Tell me more, {parent}!",
                              "Ooh! *wiggles*"]), len(ids), len(gen)
    return f"Hmm, I don't know about {topic_of(user_turn)} yet. Can you teach me, {parent}?", len(ids), len(gen)


# ==============================================================================
# 3. Terminal chat
# ==============================================================================

def interactive_terminal_chat(checkpoint_path=None, tokenizer_path="tokenizer/tokenizer.json", device=None,
                              baby="Pip", parent="Mom"):
    model, tokenizer, dev = load_chat_pipeline(checkpoint_path, tokenizer_path, device)
    facts: List[str] = []
    history: List[Dict[str, str]] = []
    needs = {"hunger": 70, "sleep": 70, "comfort": 70, "curiosity": 70}
    print(f"\nTalking with {baby}. You are {parent}.")
    print("Commands: /teach <fact>  /memory  /forget  /feed /cuddle /rock /play /nap /tickle  /quit\n")
    while True:
        try:
            msg = input(f"{parent}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not msg:
            continue
        if msg in ("/quit", "/exit"):
            break
        if msg == "/memory":
            print("\n".join(f"  - {f}" for f in facts) or "  (empty)")
            continue
        if msg == "/forget":
            facts, history = [], []
            print("  (memory cleared)")
            continue
        if msg.startswith("/teach "):
            facts.append(msg[7:].strip().rstrip("."))
            print(f"  (remembered: {facts[-1]})")
            continue
        if msg[1:] in CARE_EVENTS:
            msg = CARE_EVENTS[msg[1:]]
        t0 = time.time()
        reply, n_prompt, n_gen = baby_reply(model, tokenizer, dev, baby=baby, parent=parent, user_turn=msg,
                                            facts=facts, history=history[-8:], needs=needs)
        print(f"{baby}: {reply}   [{n_prompt}+{n_gen} tok, {time.time() - t0:.2f}s]")
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]


# ==============================================================================
# 4. HTTP backend for the nursery
# ==============================================================================

# Words a baby should never be taught or say. Kept short on purpose: it's a guard rail, not a
# full moderation system. Matching is on whole words, case-insensitive.
BLOCKED = re.compile(
    r"\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|dicks?|pussy|whore\w*|slut\w*|bastard\w*|"
    r"nigg\w*|fag\w*|retard\w*|rape\w*|porn\w*|sex\w*|nazi\w*|kill yourself|kys)\b", re.I)
KIND_REPLY = "Hmm, that's not a nice word. Let's use kind words! 💛"

# Only the nursery itself is served; the brain, code, data and saves never leave the server.
PUBLIC_PREFIXES = ("/css/", "/js/", "/sounds/")
PUBLIC_FILES = {"/", "/index.html"}


def clean_needs(raw) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    out = {}
    for key in ("hunger", "sleep", "comfort", "curiosity", "bond"):
        try:
            out[key] = max(0.0, min(100.0, float(raw.get(key, 70))))
        except (TypeError, ValueError):
            out[key] = 70.0
    return out


def run_http_server(checkpoint_path=None, tokenizer_path="tokenizer/tokenizer.json", port=8765, host="127.0.0.1",
                    device=None, open_browser=False, public=False, rate_per_minute=30):
    """
    public=True is for hosting where anyone can visit: every visitor's baby lives in their own browser
    (nothing is stored on the server), there is no shared save file, and each visitor gets at most
    rate_per_minute replies a minute so nobody can hog the brain.
    """
    import http.server
    import urllib.parse
    from collections import defaultdict, deque

    recent_calls = defaultdict(deque)  # visitor -> timestamps of recent replies
    rate_lock = threading.Lock()

    def allowed(visitor: str) -> bool:
        now = time.time()
        with rate_lock:
            calls = recent_calls[visitor]
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= rate_per_minute:
                return False
            calls.append(now)
            if len(recent_calls) > 10_000:  # forget idle visitors
                for v in [v for v, c in recent_calls.items() if not c or now - c[-1] > 60]:
                    del recent_calls[v]
            return True

    model, tokenizer, dev = load_chat_pipeline(checkpoint_path, tokenizer_path, device)
    lock = threading.Lock()  # one generation at a time; static files are served concurrently
    root = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(root, "saves", "baby.json")
    save_lock = threading.Lock()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

        def log_message(self, fmt, *args):
            # Keep the console quiet for routine static files; still show API calls and errors.
            # args differ by caller: (request_line, code, size) for requests, (code, message) for errors.
            first = str(args[0]) if args else ""
            code = str(args[1]) if len(args) > 1 else ""
            if "/api/" in first or not code.startswith(("2", "3")):
                super().log_message(fmt, *args)

        def _json(self, code: int, payload: Dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def visitor(self) -> str:
            # behind a host's proxy the real visitor is the first X-Forwarded-For address
            forwarded = self.headers.get("X-Forwarded-For", "")
            return (forwarded.split(",")[0].strip() if public and forwarded else self.client_address[0])

        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if not path.startswith("/api/") and not (path in PUBLIC_FILES or path.startswith(PUBLIC_PREFIXES)) \
                    or ".." in path:
                return self.send_error(404, "Not found")
            if path == "/api/state" and public:
                return self.send_error(404, "Not found")  # public: babies live in visitors' browsers
            if urllib.parse.urlparse(self.path).path == "/api/state":
                # the baby lives in a file next to the project, so every port/tab/browser sees the same baby
                if os.path.exists(save_path):
                    with open(save_path, encoding="utf-8") as f:
                        return self._json(200, {"state": json.load(f)})
                return self._json(200, {"state": None})
            if urllib.parse.urlparse(self.path).path == "/api/status":
                return self._json(200, {"status": "online", "params": model.count_parameters()["total"],
                                        "context_limit": model.cfg.n_positions, "device": str(dev),
                                        "mode": "public" if public else "local"})
            super().do_GET()

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path == "/api/state" and public:
                return self.send_error(404, "Not found")
            if urllib.parse.urlparse(self.path).path == "/api/state":
                length = int(self.headers.get("Content-Length", 0))
                if length > 5_000_000:
                    return self._json(413, {"error": "state too large"})
                try:
                    incoming = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return self._json(400, {"error": "invalid JSON"})
                if not isinstance(incoming, dict) or not incoming.get("babyName"):
                    return self._json(400, {"error": "not a baby"})
                with save_lock:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    tmp = save_path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(incoming, f, ensure_ascii=False)
                    if os.path.exists(save_path):
                        os.replace(save_path, save_path + ".bak")  # previous save kept as a backup
                    os.replace(tmp, save_path)  # atomic: a crash never leaves a half-written baby
                return self._json(200, {"ok": True})
            if urllib.parse.urlparse(self.path).path != "/api/chat":
                return self.send_error(404, "Endpoint not found")
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length > 200_000:
                return self._json(413, {"error": "request too large"})
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except (ValueError, UnicodeDecodeError):
                return self._json(400, {"error": "invalid JSON"})
            if not isinstance(data, dict):
                return self._json(400, {"error": "invalid request"})
            if not allowed(self.visitor()):
                return self._json(429, {"error": "The baby needs a little break. Try again in a minute!"})

            event = data.get("event")
            if event and event not in CARE_EVENTS:
                return self._json(400, {"error": "unknown care event"})
            user_turn = CARE_EVENTS.get(event) if event else str(data.get("message", "")).strip()[:280]
            if not user_turn:
                return self._json(400, {"error": "message or event required"})
            history = [{"role": h.get("role"), "content": str(h.get("content", ""))[:280]}
                       for h in (data.get("history") or [])[-8:]
                       if isinstance(h, dict) and h.get("role") in ("user", "assistant") and not BLOCKED.search(str(h.get("content", "")))]
            facts = [str(f)[:160] for f in (data.get("facts") or [])[-60:] if not BLOCKED.search(str(f))]
            if BLOCKED.search(user_turn):
                return self._json(200, {"reply": KIND_REPLY, "prompt_tokens": 0, "gen_tokens": 0, "ms": 0})
            try:
                temperature = max(0.1, min(1.0, float(data.get("temperature", 0.6))))
            except (TypeError, ValueError):
                temperature = 0.6

            t0 = time.time()
            with lock:
                reply, n_prompt, n_gen = baby_reply(
                    model, tokenizer, dev,
                    baby=str(data.get("babyName") or "Pip")[:24], parent=str(data.get("parentName") or "Mom")[:24],
                    user_turn=user_turn, facts=facts, history=history, needs=clean_needs(data.get("needs")),
                    temperature=temperature,
                )
            if BLOCKED.search(reply):
                reply = KIND_REPLY
            self._json(200, {"reply": reply, "prompt_tokens": n_prompt, "gen_tokens": n_gen,
                             "ms": int((time.time() - t0) * 1000)})

    class Server(http.server.ThreadingHTTPServer):
        # On Windows, address reuse lets two servers share one port and the OTHER one keeps
        # answering. Without it, a busy port fails loudly and we move on to a free one.
        allow_reuse_address = False

    httpd = None
    for candidate in range(port, port + 20):
        try:
            httpd = Server((host, candidate), Handler)
            break
        except OSError:
            print(f"[!] Port {candidate} is already used by another program; trying {candidate + 1}...")
    if httpd is None:
        raise SystemExit(f"No free port between {port} and {port + 19}. Try: python app.py --port 9000")
    port = httpd.server_address[1]
    url = f"http://localhost:{port}"
    print(f"[✓] Nursery open at {url}   (Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


def main():
    ap = argparse.ArgumentParser(description="Baby LLM chat & nursery server")
    ap.add_argument("--checkpoint", default=None, help=f"Default: {CHAT_CHECKPOINT} if present, else {BASE_CHECKPOINT}")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    ap.add_argument("--server", action="store_true", help="Serve the nursery web app")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default=None)
    ap.add_argument("--baby", default="Pip")
    ap.add_argument("--parent", default="Mom")
    args = ap.parse_args()
    if args.server:
        run_http_server(args.checkpoint, args.tokenizer, args.port, args.host, args.device)
    else:
        interactive_terminal_chat(args.checkpoint, args.tokenizer, args.device, args.baby, args.parent)


if __name__ == "__main__":
    main()

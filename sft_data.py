#!/usr/bin/env python3
"""
sft_data.py
===========
Procedural conversation curriculum that teaches the baby to *think with its memory*
instead of memorizing answers.

Why procedural:
  A fixed dataset of Q&A pairs can be memorized by a 14M-parameter model. Here every
  dialogue is generated fresh from templates, and many entities are invented words
  ("Mipo is a zorbel. A zorbel says floop."). The only way to answer "What does Mipo say?"
  is to read the memory, find Mipo's kind, find what that kind says, and chain them.
  That skill transfers to anything the caregiver teaches in the nursery.

Skills covered:
  recall      "Rex is a dog."                      -> "What is Rex?"            -> "Rex is a dog!"
  two_hop     "Rex is a dog." + "A dog says woof." -> "What does Rex say?"      -> "Rex is a dog, so Rex says woof!"
  yes_no      "Is Rex a cat?"                                                   -> "No, Rex is a dog."
  unknown     question about something not in memory                            -> "I don't know ... teach me?"
  perspective "Your favorite color is blue." / "I like tea." (you <-> I / my <-> your)
  teach       caregiver states a fact mid-chat; baby repeats it back and uses it later
  correction  caregiver corrects themselves; the newest statement wins
  care        *feeds you*, *cuddles you*, *tucks you in* ... reactions depend on mood
  chat        greetings, feelings, love, identity

Evaluation uses a held-out vocabulary (different invented-word syllables and unseen real
names), so its score measures generalization, not recall of training examples.

The prompt format defined here (build_system_prompt / format_chatml) is the single source
of truth shared with chat.py, so training and inference always match.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

CARE_EVENTS = {
    "feed": "*feeds you*",
    "cuddle": "*cuddles you*",
    "rock": "*rocks you gently*",
    "play": "*plays peekaboo with you*",
    "nap": "*tucks you in for a nap*",
    "tickle": "*tickles you*",
    "wake": "*wakes you up softly*",
}


# ==============================================================================
# 1. Prompt format (shared with chat.py)
# ==============================================================================

def mood_from_needs(needs: Optional[Dict[str, float]]) -> List[str]:
    """Turn 0-100 need meters into words the model reads in its system prompt."""
    if not needs:
        return []
    words = []
    if needs.get("hunger", 100) < 35:
        words.append("hungry")
    if needs.get("sleep", 100) < 35:
        words.append("sleepy")
    if needs.get("comfort", 100) < 35:
        words.append("fussy")
    if needs.get("curiosity", 100) < 30:
        words.append("bored")
    return words


def build_system_prompt(baby: str, parent: str, mood: List[str], facts: List[str]) -> str:
    lines = [
        f"You are {baby}, a baby. {parent} is your parent.",
        f"Mood: {', '.join(mood) if mood else 'happy'}.",
    ]
    if facts:
        lines.append("Memory:")
        lines.extend(f"- {f.rstrip('.!')}." for f in facts)
    else:
        lines.append("Memory: nothing yet.")
    return "\n".join(lines)


def format_chatml(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    out = []
    for m in messages:
        out.append(f"{IM_START}{m['role']}\n{m['content'].strip()}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


# ==============================================================================
# 2. Vocabulary pools (train / held-out eval)
# ==============================================================================

TRAIN_NAMES = [
    "Rex", "Bella", "Max", "Lily", "Tom", "Mia", "Sam", "Lucy", "Ben", "Coco", "Milo", "Daisy",
    "Leo", "Ruby", "Oscar", "Zoe", "Finn", "Rosie", "Jack", "Luna", "Toby", "Ella", "Pip", "Bo",
    "Nala", "Teddy", "Poppy", "Ollie", "Ziggy", "Biscuit", "Pepper", "Mochi", "Peanut", "Sunny",
]
EVAL_NAMES = ["Gus", "Hazel", "Juniper", "Waffles", "Clementine", "Bramble", "Otis", "Wren"]

TRAIN_PARENTS = ["Mom", "Dad", "Mama", "Papa", "Anna", "David", "Maria", "Ken", "Sara", "Omar", "Priya", "Leo"]
EVAL_PARENTS = ["Grandma", "Nina", "Hugo", "Ama"]

TRAIN_BABIES = ["Pip", "Bean", "Mo", "Kiki", "Nugget", "Dot", "Button", "Pumpkin", "Tiny", "Momo"]
EVAL_BABIES = ["Sprout", "Pebble", "Noodle"]

# (kind, sound) pairs. Facts sometimes use a *different* sound than real life, so the
# baby learns to trust what it was told rather than what it memorized from stories.
ANIMALS = [
    ("dog", "woof"), ("cat", "meow"), ("cow", "moo"), ("duck", "quack"), ("sheep", "baa"),
    ("pig", "oink"), ("frog", "ribbit"), ("lion", "roar"), ("owl", "hoot"), ("horse", "neigh"),
    ("bee", "buzz"), ("mouse", "squeak"), ("snake", "hiss"), ("chick", "peep"), ("goat", "maa"),
]
EVAL_ANIMALS = [("donkey", "hee-haw"), ("crow", "caw"), ("wolf", "howl")]
SOUNDS_EXTRA = ["boop", "beep", "honk", "chirp", "tweet", "growl", "purr", "toot", "squeee", "bonk"]

COLORS = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "white", "black", "brown", "gold", "silver"]
EVAL_COLORS = ["teal", "violet", "gray"]
OBJECTS = ["ball", "hat", "cup", "blanket", "car", "boat", "kite", "sock", "door", "chair", "bike",
           "box", "bed", "shoe", "spoon", "book", "drum", "truck", "bear", "balloon"]
EVAL_OBJECTS = ["umbrella", "teapot", "lamp", "rocket"]
FOODS = ["apples", "bananas", "carrots", "cookies", "rice", "soup", "milk", "bread", "berries",
         "cheese", "fish", "pancakes", "grapes", "honey", "pasta", "peas"]
EVAL_FOODS = ["mangoes", "noodles", "yogurt", "plums"]
PLACES = ["garden", "barn", "forest", "pond", "house", "park", "sea", "tree", "cave", "field", "box", "nest"]
EVAL_PLACES = ["meadow", "castle", "village"]
ACTIONS = ["fly", "swim", "jump", "dance", "sing", "run", "climb", "dig", "hop", "roll"]
EVAL_ACTIONS = ["skate", "whistle"]
ADJECTIVES = ["fluffy", "tiny", "big", "soft", "shiny", "loud", "sleepy", "fast", "slow", "silly", "kind", "brave"]
EVAL_ADJECTIVES = ["sparkly", "wobbly"]
FAV_THINGS = ["color", "toy", "song", "food", "animal", "book"]

# Invented words: one broad generator for both splits. Eval strings are freshly sampled, so they
# are novel; what eval holds out is the *real* words (EVAL_* pools never appear in training).
SYLLABLES = (list("bdfgklmnprstvzcjhwy") + ["sh", "ch", "th", "qu", "br", "fl", "gr", "sp", "x"],
             list("aeiou") + ["oo", "ee", "ai", "ou"],
             ["", "", "", "n", "r", "l", "p", "k", "sh", "m", "x", "t"])

EVAL_WORDS = {w.lower() for pool in (EVAL_NAMES, EVAL_PARENTS, EVAL_BABIES, EVAL_COLORS, EVAL_OBJECTS, EVAL_FOODS,
                                     EVAL_PLACES, EVAL_ACTIONS, EVAL_ADJECTIVES) for w in pool}
EVAL_WORDS |= {a for a, _ in EVAL_ANIMALS}


def _tokenizer_words(path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer", "vocab.json")):
    """
    Thousands of whole words from the tokenizer's vocabulary. Using them as names/kinds/places
    forces the baby to learn copying as a *general* skill: on a narrow name list, a small model
    learns "what names look like" and hallucinates familiar ones for anything new.
    """
    try:
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)
    except OSError:
        return [], []
    ranked = sorted(vocab.items(), key=lambda kv: kv[1])
    lower = [k[1:] for k, _ in ranked if k.startswith("Ġ") and k[1:].isalpha() and k[1:].islower() and len(k) >= 5]
    all_lower = [k[1:] for k, _ in ranked if k.startswith("Ġ") and k[1:].isalpha() and k[1:].islower()]
    common = set(all_lower[:400])  # function words and the most frequent story words
    lower = [w for w in lower[300:] if w not in EVAL_WORDS]
    caps = [k[1:] for k, _ in ranked if k.startswith("Ġ") and k[1:].isalpha() and len(k) >= 4
            and k[1].isupper() and k[2:].islower() and k[1:].lower() not in common and k[1:].lower() not in EVAL_WORDS]
    return lower, caps


VOCAB_WORDS, VOCAB_NAMES = _tokenizer_words()


class Vocab:
    def __init__(self, split: str, rng: random.Random):
        self.rng = rng
        ev = split == "eval"
        self.eval = ev
        self.names = EVAL_NAMES if ev else TRAIN_NAMES + VOCAB_NAMES
        self.parents = EVAL_PARENTS if ev else TRAIN_PARENTS
        self.babies = EVAL_BABIES if ev else TRAIN_BABIES
        self.animals = EVAL_ANIMALS + ANIMALS[:4] if ev else ANIMALS
        self.colors = EVAL_COLORS if ev else COLORS
        self.objects = EVAL_OBJECTS if ev else OBJECTS
        self.foods = EVAL_FOODS if ev else FOODS
        self.places = EVAL_PLACES if ev else PLACES
        self.actions = EVAL_ACTIONS if ev else ACTIONS
        self.adjectives = EVAL_ADJECTIVES if ev else ADJECTIVES
        self.p_pseudo = 0.5 if ev else 0.35
        # training also draws from the whole tokenizer vocabulary
        self.p_vocab = 0.0 if ev else 0.3

    def pseudo(self, n_syll: Optional[int] = None) -> str:
        c, v, coda = SYLLABLES
        n = n_syll or self.rng.choice([1, 2, 2, 3])
        return "".join(self.rng.choice(c) + self.rng.choice(v) + (self.rng.choice(coda) if i == n - 1 else "")
                       for i in range(n))

    def _wild(self) -> Optional[str]:
        """A word from outside the curated pools: invented, or any vocab word (train only)."""
        r = self.rng.random()
        if r < self.p_pseudo:
            return self.pseudo()
        if r < self.p_pseudo + self.p_vocab and VOCAB_WORDS:
            return self.rng.choice(VOCAB_WORDS)
        return None

    def person(self, pool: List[str]) -> str:
        """Parent/baby names: mostly the pool, sometimes novel, so the system prompt must be read."""
        if self.rng.random() < 0.35:
            w = self._wild()
            if w:
                return w.capitalize()
        return self.rng.choice(pool)

    def name(self) -> str:
        w = self._wild()
        if w:
            return w.capitalize()
        return self.rng.choice(self.names)

    def kind_and_sound(self) -> Tuple[str, str]:
        w = self._wild()
        if w:
            return w.lower(), self.pseudo(1)
        kind, sound = self.rng.choice(self.animals)
        if self.rng.random() < 0.2:  # caregiver's world beats storybook priors
            sound = self.rng.choice(SOUNDS_EXTRA)
        return kind, sound

    def pick(self, attr: str) -> str:
        if attr != "actions" and self.rng.random() < 0.6:
            w = self._wild()
            if w:
                return w.lower()
        return self.rng.choice(getattr(self, attr))


# ==============================================================================
# 3. Fact families: each yields (memory sentences, [(question, answer, eval_key)])
# ==============================================================================

def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


class FactFactory:
    """Generates facts plus the questions they license. Answers vary in wording on purpose."""

    def __init__(self, vocab: Vocab, rng: random.Random, parent: str):
        self.v = vocab
        self.rng = rng
        self.parent = parent

    def exclaim(self, s: str) -> str:
        return s + self.rng.choice(["!", "!", ".", "! Hehe.", "! I remember."])

    def category(self):
        n = self.v.name()
        kind, _ = self.v.kind_and_sound()
        other, _ = self.v.kind_and_sound()
        while other == kind:
            other, _ = self.v.kind_and_sound()
        fact = f"{n} is {article(kind)} {kind}"
        qa = [
            (self.rng.choice([f"What is {n}?", f"Who is {n}?", f"Tell me about {n}.", f"Do you know {n}?"]),
             self.exclaim(fact), "recall", [kind]),
            (f"Is {n} {article(kind)} {kind}?", f"Yes! {fact}.", "yes_no", ["yes", kind]),
            (f"Is {n} {article(other)} {other}?", f"No, {fact}.", "yes_no", ["no", kind]),
        ]
        return [fact], qa, {"name": n, "kind": kind}

    def sound(self, kind: Optional[str] = None):
        k2, s = self.v.kind_and_sound()
        kind = kind or k2
        fact = f"{article(kind).capitalize()} {kind} says {s}"
        qa = [(f"What does {article(kind)} {kind} say?", self.exclaim(fact), "recall", [s])]
        return [fact], qa, {"kind": kind, "sound": s}

    def two_hop_sound(self):
        f1, _, c = self.category()
        f2, _, s = self.sound(c["kind"])
        n, kind, snd = c["name"], c["kind"], s["sound"]
        ans = self.rng.choice([
            f"{n} is {article(kind)} {kind}, so {n} says {snd}!",
            f"{n} says {snd}! Because {n} is {article(kind)} {kind}.",
        ])
        qa = [(self.rng.choice([f"What does {n} say?", f"What sound does {n} make?"]), ans, "two_hop", [snd])]
        return f1 + f2, qa, {}

    def ability(self):
        kind, _ = self.v.kind_and_sound()
        act = self.v.pick("actions")
        fact = f"{article(kind).capitalize()} {kind} can {act}"
        qa = [(f"Can {article(kind)} {kind} {act}?", f"Yes! {fact}.", "recall", ["yes", act])]
        return [fact], qa, {"kind": kind, "act": act}

    def two_hop_ability(self):
        f1, _, c = self.category()
        n, kind = c["name"], c["kind"]
        act = self.v.pick("actions")
        f2 = [f"{article(kind).capitalize()} {kind} can {act}"]
        ans = f"Yes! {n} is {article(kind)} {kind}, and {article(kind)} {kind} can {act}."
        return f1 + f2, [(f"Can {n} {act}?", ans, "two_hop", ["yes", act])], {}

    def color(self):
        obj = self.v.pick("objects")
        col = self.v.pick("colors")
        owner = self.rng.choice(["The", f"{self.v.name()}'s"])
        subj = f"{owner} {obj}" if owner != "The" else f"the {obj}"
        fact = f"{subj[0].upper() + subj[1:]} is {col}"
        return [fact], [(f"What color is {subj}?", self.exclaim(fact), "recall", [col])], {}

    def eats(self):
        n = self.v.name()
        food = self.v.pick("foods")
        fact = f"{n} eats {food}"
        qa = [(self.rng.choice([f"What does {n} eat?", f"What does {n} like to eat?"]), self.exclaim(fact), "recall", [food])]
        return [fact], qa, {}

    def lives(self):
        n = self.v.name()
        place = self.v.pick("places")
        fact = f"{n} lives in the {place}"
        return [fact], [(f"Where does {n} live?", self.exclaim(fact), "recall", [place])], {}

    def describe(self):
        kind, _ = self.v.kind_and_sound()
        adj = self.v.pick("adjectives")
        fact = f"{article(kind).capitalize()} {kind} is {adj}"
        qa = [(self.rng.choice([f"What is {article(kind)} {kind} like?", f"Tell me about {article(kind)} {kind}."]),
               self.exclaim(fact), "recall", [adj])]
        return [fact], qa, {}

    def parent_likes(self):
        food = self.v.pick("foods")
        p = self.parent
        fact = f"{p} likes {food}"
        qa = [(self.rng.choice(["What do I like?", "Do you remember what I like?", "What do I like to eat?"]),
               self.exclaim(f"You like {food}"), "perspective", [food])]
        return [fact], qa, {}

    def parent_pet(self):
        n = self.v.name()
        kind, _ = self.v.kind_and_sound()
        p = self.parent
        fact = f"{p} has {article(kind)} {kind} named {n}"
        qa = [
            ("What pet do I have?", self.exclaim(f"You have {article(kind)} {kind} named {n}"), "perspective", [kind]),
            ("What is my pet's name?", self.exclaim(f"Your {kind} is named {n}"), "perspective", [n]),
        ]
        return [fact], qa, {}

    def baby_favorite(self):
        thing = self.rng.choice(FAV_THINGS)
        val = self._fav_value(thing)
        fact = f"My favorite {thing} is {val}"
        qa = [(f"What is your favorite {thing}?", self.exclaim(fact), "perspective", [val])]
        return [fact], qa, {}

    def parent_favorite(self):
        thing = self.rng.choice(FAV_THINGS)
        val = self._fav_value(thing)
        fact = f"{self.parent}'s favorite {thing} is {val}"
        qa = [(f"What is my favorite {thing}?", self.exclaim(f"Your favorite {thing} is {val}"), "perspective", [val])]
        return [fact], qa, {}

    def freeform(self):
        """
        Facts the way people actually teach: "Stars are bright suns far away", "Birds have feathers
        and can fly high". Plural subjects, long predicates; half the predicates are random word
        strings, so the only way to answer is to copy the whole span from memory.
        """
        R, v = self.rng, self.v
        plural = R.random() < 0.6
        noun = v.kind_and_sound()[0] if R.random() < 0.5 else v.pick("objects")
        subj = (noun + ("es" if noun.endswith(("s", "sh", "ch", "x")) else "s")) if plural else R.choice(["the ", ""]) + noun
        be, have, can = ("are", "have", "can") if plural else ("is", "has", "can")

        def words(n):
            pool = VOCAB_WORDS if (VOCAB_WORDS and not v.eval) else None
            return " ".join(R.choice(pool) if pool and R.random() < 0.7 else v.pseudo() for _ in range(n))

        kind = R.choice(["be", "be", "have", "can", "verb"])
        if R.random() < 0.5:
            tail = words(R.randint(2, 5))
        else:
            tail = {
                "be": f"{v.pick('adjectives')} and {v.pick('adjectives')}",
                "have": f"{v.pick('adjectives')} {v.pick('objects')}s",
                "can": f"{v.pick('actions')} very {v.pick('adjectives')}",
                "verb": f"live in the {v.pick('places')}",
            }[kind]
        verb = {"be": be, "have": have, "can": can, "verb": ""}[kind]
        pred = f"{verb} {tail}".strip()
        cap = subj[0].upper() + subj[1:]
        fact = f"{cap} {pred}"
        key = [tail.split()[-1]]
        bare = subj[4:] if subj.startswith("the ") else subj
        q = R.choice([f"What {be} {subj}?", f"Tell me about {subj}.", f"What do you know about {subj}?",
                      f"What {'do' if plural else 'does'} {subj} {'have' if kind == 'have' else 'do'}?" if kind in ("have", "verb") else f"What {be} {subj}?"])
        if R.random() < 0.3:
            q = q.replace(subj, bare) if not subj.startswith("the ") else q
        ans = R.choice([f"{fact}!", f"{fact}! I remember.", f"Ooh! {fact}."])
        qa = [(q, ans, "freeform", key)]
        if kind == "can":
            act = tail.split()[0]
            qa.append((f"Can {subj} {act}?", f"Yes! {fact}.", "freeform", ["yes", act]))
        return [fact], qa, {}

    def _fav_value(self, thing: str) -> str:
        if thing == "color":
            return self.v.pick("colors")
        if thing == "food":
            return self.v.pick("foods")
        if thing == "animal":
            return self.v.kind_and_sound()[0]
        return f"the {self.v.pseudo()} {thing}" if self.rng.random() < 0.5 else f"the {self.v.pick('adjectives')} {thing}"

    FAMILIES = [
        ("category", 3), ("sound", 1), ("two_hop_sound", 3), ("ability", 1), ("two_hop_ability", 2),
        ("color", 2), ("eats", 2), ("lives", 2), ("describe", 1), ("parent_likes", 1),
        ("parent_pet", 1), ("baby_favorite", 1), ("parent_favorite", 1), ("freeform", 5),
    ]

    def random_family(self):
        names, weights = zip(*self.FAMILIES)
        return getattr(self, self.rng.choices(names, weights=weights)[0])()


# ==============================================================================
# 4. Non-fact turns: unknowns, teaching, corrections, care, chit-chat
# ==============================================================================

def unknown_turn(v: Vocab, rng: random.Random, parent: str, known_text: str = ""):
    r = rng.random()
    thing = v.name() if r < 0.4 else v.pseudo() if r < 0.65 else v.kind_and_sound()[0]
    while thing.lower() in known_text.lower():  # must truly be unknown
        thing = v.pseudo()
    if thing.islower() and rng.random() < 0.25:
        q = rng.choice([f"What are {thing}s?", f"Tell me about {thing}s."])
        a = rng.choice([f"Hmm, I don't know about {thing}s yet. Can you teach me, {parent}?", f"{thing.capitalize()}s? I don't know yet. Teach me, please!"])
        return q, a, "unknown", ["know"]
    if thing.islower() and rng.random() < 0.6:
        q = rng.choice([f"What is {article(thing)} {thing}?", f"What does {article(thing)} {thing} say?",
                        f"Do you know what {article(thing)} {thing} is?"])
        a = rng.choice([f"Hmm, I don't know what {article(thing)} {thing} is yet. Can you teach me, {parent}?",
                        f"{article(thing).capitalize()} {thing}? I don't know yet. Teach me, please!"])
        return q, a, "unknown", ["know"]
    q = rng.choice([f"What is {thing}?", f"Who is {thing}?", f"What does {thing} eat?",
                    f"Where does {thing} live?", f"What does {article(thing)} {thing} say?", f"Tell me about {thing}."])
    a = rng.choice([
        f"Hmm, I don't know about {thing} yet. Can you teach me, {parent}?",
        f"I don't know {thing}! Will you tell me?",
        f"Ooh, {thing}? I don't know yet. Teach me, please!",
    ])
    return q, a, "unknown", ["know"]


def teach_reply(fact: str, parent: str, rng: random.Random) -> str:
    return rng.choice([
        f"Ooh! {fact}! I will remember that, {parent}.",
        f"{fact}! Wow, I learned something new!",
        f"Okay! {fact}. I will keep it in my memory.",
        f"{fact}? Hehe, thank you for teaching me!",
    ])


def teach_phrase(fact: str, rng: random.Random) -> str:
    return rng.choice([f"{fact}.", f"Did you know? {fact}.", f"Remember this: {fact}.", f"Listen, {fact[0].lower() + fact[1:]}."])


def perspective_teach(v: Vocab, rng: random.Random, parent: str):
    """Caregiver speaks in first/second person; the baby must flip I/you correctly."""
    choice = rng.random()
    if choice < 0.35:
        food = v.pick("foods")
        said = rng.choice([f"I like {food}.", f"I love {food}."])
        verb = "like" if "like" in said else "love"
        return said, rng.choice([f"You {verb} {food}! I will remember, {parent}.", f"Ooh, you {verb} {food}!"]), \
            (f"What do I {verb}?", f"You {verb} {food}!", "perspective", [food])
    if choice < 0.7:
        thing = rng.choice(FAV_THINGS[:3])
        val = v.pick("colors") if thing == "color" else v.pseudo()
        said = f"Your favorite {thing} is {val}."
        return said, rng.choice([f"My favorite {thing} is {val}! Yay!", f"Okay! My favorite {thing} is {val}."]), \
            (f"What is your favorite {thing}?", f"My favorite {thing} is {val}!", "perspective", [val])
    kind, _ = v.kind_and_sound()
    n = v.name()
    said = rng.choice([f"I have {article(kind)} {kind} named {n}.", f"You know, I have {article(kind)} {kind}. Her name is {n}."])
    return said, rng.choice([f"You have {article(kind)} {kind} named {n}! I want to meet {n}!",
                             f"{article(kind).capitalize()} {kind} named {n}! So cute!"]), \
        ("What pet do I have?", f"You have {article(kind)} {kind} named {n}!", "perspective", [kind])


def correction_dialogue(v: Vocab, rng: random.Random, parent: str):
    """I have a cat ... actually a dog ... -> newest wins."""
    n = v.name()
    k1, _ = v.kind_and_sound()
    k2, _ = v.kind_and_sound()
    while k2 == k1:
        k2, _ = v.kind_and_sound()
    turns = [
        (f"I have {article(k1)} {k1} named {n}.", f"You have {article(k1)} {k1} named {n}! So sweet!"),
        (rng.choice([f"Oops, no. {n} is {article(k2)} {k2}, not {article(k1)} {k1}.",
                     f"Actually, {n} is {article(k2)} {k2}. I made a mistake.",
                     f"Wait, I was wrong. {n} is {article(k2)} {k2}!"]),
         rng.choice([f"Oh! {n} is {article(k2)} {k2}. Got it, {parent}!", f"Okay! {n} is {article(k2)} {k2}, not {article(k1)} {k1}."])),
    ]
    q = rng.choice([f"What is {n}?", f"Is {n} {article(k1)} {k1}?"])
    a = f"{n} is {article(k2)} {k2}!" if q.startswith("What") else f"No! {n} is {article(k2)} {k2}."
    return turns, (q, a, "correction", [k2])


def care_turn(event: str, mood: List[str], parent: str, baby: str, rng: random.Random) -> Tuple[str, str]:
    hungry, sleepy, fussy = "hungry" in mood, "sleepy" in mood, "fussy" in mood
    R = rng.choice
    if event == "feed":
        a = R([f"Mmm, milk! Yummy yummy. Thank you, {parent}!", "*gulp gulp* Mmm! My tummy was so empty!",
               "Yay, food! *happy wiggle*"]) if hungry else \
            R(["Hehe, I'm full! No more milk, please.", "*turns head* I'm not hungry, silly!", "Just a tiny sip. My tummy is happy!"])
    elif event == "cuddle":
        a = R([f"*snuggles* I feel so safe with you, {parent}.", "Warm and squishy! I love cuddles!",
               f"Hehe! I love you, {parent}!", "*nuzzles* Don't let go, okay?"])
        if fussy:
            a = R([f"*sniff* ...better now. Thank you, {parent}.", "Mmm, I needed a hug. *sniffle*"])
    elif event == "rock":
        a = R(["Swing... swing... so cozy.", "*yawns* That feels nice, rocky rocky.", "Wheee... softly... I like this."])
    elif event == "play":
        a = R(["Peekaboo! Hehe! Again, again!", "*giggles* Where did you go? There you are!", "Haha! You found me!"])
        if sleepy:
            a = R(["*yawn* Peek... a... boo. I'm so sleepy.", "Hehe... but my eyes are heavy."])
    elif event == "nap":
        a = R([f"*yawns* Night night, {parent}...", "Mmm... cozy blanket... zzz.", "Sing me a song? ...zzz."]) if sleepy else \
            R(["But I'm not sleepy! ...okay, maybe a little. *yawn*", "Just five more minutes of play? ...zzz."])
    elif event == "tickle":
        a = R(["Hahaha! Stop! No, don't stop!", "*squeals* That tickles!", "Hehehe! My tummy!"])
    else:  # wake
        a = R([f"*stretches* Good morning, {parent}!", "*blinks* Hi! I had a dream about stars.", "Mmm... I'm awake! Hehe."])
    return CARE_EVENTS[event], a


def chat_turn(mood: List[str], parent: str, baby: str, rng: random.Random) -> Tuple[str, str, str, List[str]]:
    R = rng.choice
    kind = R(["greet", "love", "name", "parent", "how", "sad", "happy", "night", "morning", "thanks", "story"])
    if kind == "greet":
        return R(["Hi!", "Hello there!", f"Hey {baby}!", "Hi baby!"]), R([f"Hi {parent}! *waves tiny hand*", f"Hello, {parent}! I missed you!", "Hiii! Hehe."]), "chat", []
    if kind == "love":
        return R(["I love you.", "I love you so much!", "You're so cute."]), R([f"I love you too, {parent}!", "*blushes* Hehe, I love you more!", "You make my heart warm!"]), "chat", []
    if kind == "name":
        return R(["What is your name?", "Who are you?", "What's your name, little one?"]), R([f"I'm {baby}!", f"My name is {baby}! Hehe.", f"{baby}! That's me!"]), "identity", [baby]
    if kind == "parent":
        return R(["Who am I?", "What is my name?", "Do you know who I am?"]), R([f"You're {parent}!", f"You are {parent}, my parent!", f"{parent}! I know you!"]), "identity", [parent]
    if kind == "how":
        q = R(["How are you?", "How do you feel?", "Are you okay?"])
        if "hungry" in mood:
            return q, R(["My tummy is rumbly... I'm hungry.", "Hungry! Milk, please?"]), "mood", ["hungry"]
        if "sleepy" in mood:
            return q, R(["*yawns* I'm so sleepy...", "Sleepy... my eyes are heavy."]), "mood", ["sleepy"]
        if "fussy" in mood:
            return q, R(["*sniffle* I want a cuddle.", "I feel fussy. Hold me?"]), "mood", ["cuddle"]
        if "bored" in mood:
            return q, R(["I'm bored! Can we play?", "Play with me, please!"]), "mood", ["play"]
        return q, R(["I'm happy! Hehe!", f"I feel great, {parent}!", "Happy happy! *kicks feet*"]), "mood", ["happy"]
    if kind == "sad":
        return R(["I'm sad today.", "I had a bad day.", "I feel tired."]), R([f"Oh no... *pats you* I'm here, {parent}.", "Can I give you a big hug?", "Don't be sad. I love you!"]), "chat", []
    if kind == "happy":
        return R(["I'm so happy!", "Today was a good day!"]), R(["Yay! *claps* I'm happy too!", f"Hehe! Happy {parent} makes me happy!"]), "chat", []
    if kind == "night":
        return R(["Good night!", "Sweet dreams.", "Time for bed."]), R([f"Night night, {parent}! *yawn*", "Sweet dreams to you too!"]), "chat", []
    if kind == "morning":
        return R(["Good morning!", "Rise and shine!"]), R([f"Good morning, {parent}! *stretches*", "Morning! Is it milk time?"]), "chat", []
    if kind == "thanks":
        return R(["Thank you!", "Thanks, baby."]), R(["You're welcome! Hehe.", f"Anything for you, {parent}!"]), "chat", []
    return R(["Tell me a story.", "Can you tell me a story?"]), R([
        "Once upon a time, a tiny baby looked at the moon and said hi! The end. Hehe.",
        "Once there was a little duck. It said quack and went to sleep. The end!"]), "chat", []


# ==============================================================================
# 4b. Consistency with itself: remember what *I* said, and follow up on answers
# ==============================================================================

# Things the baby says on its own: (statement, follow-up questions, answer templates)
SELF_TOPICS = [
    ("I had a dream about {x}!", ["What did you dream about?", "Tell me about your dream.", "What was your dream?"],
     ["I dreamed about {x}!", "{X}! I dreamed about {x}!"], "dreamed about {x}"),
    ("I saw {a} {x} by the window!", ["What did you see?", "What was by the window?"],
     ["I saw {a} {x}!", "{A} {x}! By the window!"], "saw {a} {x}"),
    ("I want to play with my {x}!", ["What do you want to play with?", "What did you want?"],
     ["My {x}! Can we play?", "I want my {x}!"], "want to play with your {x}"),
    ("I really like {x}!", ["What do you like?", "What did you say you like?"],
     ["I like {x}!", "{X}! I really like {x}!"], "like {x}"),
]
SELF_LEADS = ["*blinks* Hi! ", "*stretches* ", "Guess what? ", "Ooh! ", ""]


def self_reference_turns(v: Vocab, rng: random.Random, parent: str):
    """The baby says something spontaneously, then is asked about it (or misquoted)."""
    say, questions, answers, quote = rng.choice(SELF_TOPICS)
    x = v.pick("objects") if rng.random() < 0.5 else v.kind_and_sound()[0]
    if rng.random() < 0.3:
        x = v.pseudo()
    fill = {"x": x, "X": x.capitalize(), "a": article(x), "A": article(x).capitalize()}
    opener = rng.choice(["*wakes you up softly*", "Hi!", "Good morning!", "*cuddles you*", "What's up, little one?"])
    turns = [(opener, rng.choice(SELF_LEADS) + say.format(**fill), "chat", [])]
    r = rng.random()
    if r < 0.6:
        turns.append((rng.choice(questions), rng.choice(answers).format(**fill), "self", [x]))
    elif r < 0.8:
        # caregiver repeats it back correctly
        turns.append(("You said you " + quote.format(**fill) + "?", f"Yes! {rng.choice(answers).format(**fill)}", "self", ["yes", x]))
    else:
        # caregiver misquotes: the baby corrects from what it actually said
        y = v.pick("objects")
        while y == x:
            y = v.pseudo()
        wrong = dict(fill, x=y, a=article(y))
        turns.append(("You said you " + quote.format(**wrong) + "?",
                      f"No! {rng.choice(answers).format(**fill)}", "self", ["no", x]))
    return turns


# The baby asks; the caregiver answers; the baby responds to THAT answer
def followup_turns(v: Vocab, rng: random.Random, parent: str):
    R = rng.choice
    kind = R(["pet", "food", "day", "play", "love", "toy"])
    if kind == "pet":
        n = v.name()
        ask = R([f"Do you have a pet, {parent}? What is their name?", "What is your pet's name?"])
        ans = R([n, n.lower(), f"Her name is {n}.", f"His name is {n}.", f"It's {n}."])
        reply = R([f"{n}! What a sweet name!", f"{n}! I want to meet {n}!", f"Hi, {n}! Hehe."])
        return [(R(["Hi!", "*cuddles you*"]), ask, "chat", []), (ans, reply, "followup", [n])]
    if kind == "food":
        f = v.pick("foods")
        ask = R([f"What is your favorite food, {parent}?", "What do you like to eat?"])
        ans = R([f, f"I love {f}.", f"{f.capitalize()}!", f"Probably {f}."])
        reply = R([f"{f.capitalize()}! Yummy! I want to try {f} too!", f"Ooh, {f}! Is it yummy?"])
        return [(R(["*feeds you*", "Hi!"]), ask, "chat", []), (ans, reply, "followup", [f])]
    if kind == "day":
        p = v.pick("places")
        ask = R(["What did you do today?", f"Where did you go today, {parent}?"])
        ans = R([f"I went to the {p}.", f"The {p}.", f"I was at the {p} all day."])
        reply = R([f"The {p}! Was it fun?", f"Ooh, the {p}! Can I come next time?"])
        return [(R(["I'm home!", "Hi!"]), ask, "chat", []), (ans, reply, "followup", [p])]
    if kind == "play":
        ask = R(["Can we play?", "Will you play with me?"])
        if rng.random() < 0.6:
            return [(R(["Hi!", "*tickles you*"]), ask, "chat", []),
                    (R(["Yes!", "Sure!", "Okay, let's play.", "of course"]), R(["Yay! Let's play!", "Yay! *claps*"]), "followup", ["yay"])]
        return [(R(["Hi!", "*tickles you*"]), ask, "chat", []),
                (R(["Not now.", "Later, sweetie.", "No, it's bedtime."]), R(["Okay... later then. *pouts*", "Aww. Okay."]), "followup", ["okay"])]
    if kind == "love":
        return [(R(["*cuddles you*", "Hi!"]), R(["Do you love me?", f"{parent}, do you love me?"]), "chat", []),
                (R(["Yes, so much!", "Of course I do.", "More than anything."]), R([f"Hehe! I love you too, {parent}!", "Yay! *happy wiggle*"]), "followup", ["love"] if "love" in "" else [])]
    n = v.name()
    return [(R(["*plays peekaboo with you*", "Hi!"]), R(["What should I name my teddy?", "My teddy needs a name! What should it be?"]), "chat", []),
            (R([n, f"Call it {n}.", f"How about {n}?"]), R([f"{n}! Hi, {n}! *hugs teddy*", f"{n}! I love it!"]), "followup", [n])]


# ==============================================================================
# 5. Dialogue assembly
# ==============================================================================

class DialogueGenerator:
    def __init__(self, split: str = "train", seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.split = split
        self.vocab = Vocab(split, self.rng)

    def sample(self) -> Tuple[List[Dict[str, str]], List[Tuple[str, List[str], str]]]:
        """Returns (messages, checks). checks = [(category, keys, expected_reply)] for each assistant turn."""
        rng, v = self.rng, self.vocab
        baby, parent = v.person(v.babies), v.person(v.parents)
        mood = [m for m in ("hungry", "sleepy", "fussy", "bored") if rng.random() < 0.15]
        care_first = None
        if rng.random() < 0.3:
            # the same care action must get a different reaction depending on mood
            care_first = rng.choice(["feed", "feed", "nap", "nap", "cuddle", "play", "rock"])
            need = {"feed": "hungry", "nap": "sleepy", "cuddle": "fussy", "play": "sleepy", "rock": "sleepy"}[care_first]
            mood = [m for m in mood if m != need] + ([need] if rng.random() < 0.5 else [])
        ff = FactFactory(v, rng, parent)

        # Memory = a few fact families (each may bring several linked facts) + their questions
        memory, askable = [], []
        for _ in range(rng.randint(0, 4)):
            facts, qa, _ = ff.random_family()
            memory.extend(facts)
            askable.extend(qa)
        rng.shuffle(memory)

        turns: List[Tuple[str, str, str, List[str]]] = []
        n_turns = rng.randint(1, 4)
        pending: List[Tuple[str, str, str, List[str]]] = []  # questions about things taught mid-chat

        if care_first:
            u, a = care_turn(care_first, mood, parent, baby, rng)
            turns.append((u, a, "care", []))
        elif rng.random() < 0.3:
            # conversations where the baby must stay consistent with itself
            turns.extend(self_reference_turns(v, rng, parent) if rng.random() < 0.5 else followup_turns(v, rng, parent))
        if rng.random() < 0.12:
            corr, q = correction_dialogue(v, rng, parent)
            turns.extend((u, a, "chat", []) for u, a in corr)
            pending.append(q)

        while len(turns) < n_turns:
            r = rng.random()
            if pending and r < 0.45:
                turns.append(pending.pop(0))
            elif askable and r < 0.65:
                turns.append(askable.pop(rng.randrange(len(askable))))
            elif r < 0.72:
                said_so_far = " ".join(memory) + " " + " ".join(t[0] + " " + t[1] for t in turns)
                turns.append(unknown_turn(v, rng, parent, said_so_far))
            elif r < 0.81:
                facts, qa, _ = ff.random_family()
                fact = facts[-1]
                turns.append((teach_phrase(fact, rng), teach_reply(fact, parent, rng), "teach", []))
                # the question for that newest fact can come later in the same chat
                for q in qa:
                    if any(k.lower() in fact.lower() for k in q[3] if k not in ("yes", "no")):
                        pending.append(q)
                        break
                # two-hop families taught in pieces: earlier pieces go to memory
                memory.extend(facts[:-1])
            elif r < 0.86:
                said, reply, q = perspective_teach(v, rng, parent)
                turns.append((said, reply, "teach", []))
                pending.append(q)
            elif r < 0.94:
                ev = rng.choice(list(CARE_EVENTS))
                u, a = care_turn(ev, mood, parent, baby, rng)
                turns.append((u, a, "care", []))
            else:
                turns.append(chat_turn(mood, parent, baby, rng))
        # flush one pending question so taught facts actually get used
        if pending:
            turns.append(pending.pop(0))

        msgs = [{"role": "system", "content": build_system_prompt(baby, parent, mood, memory)}]
        checks = []
        for u, a, cat, keys in turns:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
            checks.append((cat, keys, a))
        return msgs, checks


# ==============================================================================
# 6. Tokenization with assistant-only loss mask
# ==============================================================================

def encode_dialogue(messages: List[Dict[str, str]], tokenizer, max_len: int):
    """
    Returns (input_ids, targets). targets is -100 everywhere except on the baby's own
    reply tokens (and its <|im_end|>), so gradients only teach *how to respond*.
    Character spans + tokenizer offsets keep the mask exact at every token boundary.
    """
    text = ""
    spans = []
    for m in messages:
        head = f"{IM_START}{m['role']}\n"
        body = f"{m['content'].strip()}{IM_END}"
        text += head
        if m["role"] == "assistant":
            spans.append((len(text), len(text) + len(body)))
        text += body + "\n"

    enc = tokenizer.encode(text)
    ids = enc.ids
    trainable = [any(s <= start < e for s, e in spans) for start, _ in enc.offsets]

    ids = ids[:max_len + 1]
    trainable = trainable[:max_len + 1]
    x = ids[:-1]
    y = [ids[i + 1] if trainable[i + 1] else -100 for i in range(len(ids) - 1)]
    return x, y


if __name__ == "__main__":
    gen = DialogueGenerator("train", seed=1)
    for _ in range(3):
        msgs, _ = gen.sample()
        print(format_chatml(msgs, add_generation_prompt=False))
        print("-" * 60)
    gen = DialogueGenerator("eval", seed=2)
    msgs, _ = gen.sample()
    print("[eval sample]\n" + format_chatml(msgs, add_generation_prompt=False))

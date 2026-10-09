const KEY = "baby-llm-pip";

const STOP = new Set([
  "the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "you", "i", "it",
  "we", "they", "me", "my", "your", "be", "do", "did", "was", "for", "on", "at",
  "that", "this", "with", "from", "but", "not", "have", "has", "just", "about",
]);

const QWORDS = new Set(["what", "who", "why", "where", "how", "when", "do", "does", "did", "can", "could", "would", "should", "is", "are", "am"]);

const GREET = /^(hi|hello|hey|yo|ok|okay|yes|no|thanks|thank you)[.!\s]*$/i;

function tokens(text) {
  return (String(text).toLowerCase().match(/[a-z']+/g) || []).filter((w) => w.length > 1);
}

function contentWords(text) {
  return tokens(text).filter((w) => !STOP.has(w) && !QWORDS.has(w));
}

function pick(list) {
  return list[Math.floor(Math.random() * list.length)];
}

function clamp(n) {
  return Math.max(0, Math.min(100, n));
}

function blankState() {
  return {
    babyName: "",
    parentName: "",
    bornAt: Date.now(),
    ageHours: 0,
    lastTick: Date.now(),
    lastHeard: "",
    topic: "",
    awaiting: null,
    xp: 0,
    voice: true,
    lexicon: {},
    knowledge: [],
    caregiverInfo: {
      pet: "",
      petName: "",
      favorites: []
    },
    convoContext: {
      lastQuestionAsked: null,
      turnCount: 0
    },
    firsts: {},
    chat: [],
    asleep: false,
    outfit: { onesie: "sky", pattern: "star", hat: "none", acc: "none" },
    needs: {
      hunger: 72,
      sleep: 78,
      comfort: 64,
      curiosity: 88,
      bond: 12,
    },
  };
}

function sanitizeKnowledge(st) {
  if (!st || !Array.isArray(st.knowledge)) return;
  st.caregiverInfo = st.caregiverInfo || { pet: "", petName: "", favorites: [] };
  st.convoContext = st.convoContext || { lastQuestionAsked: null, turnCount: 0 };

  const cleanList = [];
  const seen = new Set();

  for (const item of st.knowledge) {
    if (!item || !item.text) continue;
    const txt = String(item.text).trim();
    const lower = txt.toLowerCase();

    // Check if pet was stored as an accidental fact previously
    if (/\b(dog|cat|puppy|kitten|pet)\b/i.test(lower) && /\b(have|got|my)\b/i.test(lower)) {
      const petM = lower.match(/\b(dog|cat|puppy|kitten|pet|bunny|rabbit|bird|hamster|fish)\b/i);
      if (petM && !st.caregiverInfo.pet) st.caregiverInfo.pet = petM[1];
    }
    const nameM = txt.match(/\bname\s+is\s+([A-Za-z]+)/i);
    if (nameM) {
      if (lower.includes("my name is") && !st.parentName) {
        st.parentName = nameM[1];
      } else if (!lower.includes("my name is") && !st.caregiverInfo.petName) {
        st.caregiverInfo.petName = nameM[1];
      }
    }

    // Filter out conversational chat fragments and fillers mistakenly saved as facts
    if (/\b(you said|i said|said you|said that)\b/i.test(lower)) continue;
    if (/^(because|since|if|so|actually|well|you know|her name is|his name is|my name is|i have|i am|correct|yes|no|okay|sure)\b/i.test(lower)) {
      continue;
    }
    if (/^(you know i have a dog|her name is minik)\b/i.test(lower)) {
      continue;
    }
    if (tokens(txt).length < 3 && !/^[A-Za-z]+\s+is\s+[A-Za-z]+$/i.test(txt)) {
      continue;
    }

    const key = lower.replace(/[^a-z0-9]/g, "");
    if (!seen.has(key)) {
      seen.add(key);
      cleanList.push(item);
    }
  }
  st.knowledge = cleanList;
}

function load() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = { ...blankState(), ...JSON.parse(raw) };
    sanitizeKnowledge(parsed);
    save(parsed);
    return parsed;
  } catch {
    return null;
  }
}

// The browser keeps a separate copy per address (localhost:8765, :8766, the plain file...), so on
// its own the baby would "restart" whenever the port changed. With app.py running, the real home
// is saves/baby.json on the server; the browser copy is just a cache and an offline fallback.
let serverSync = false; // only true once the server's copy has been read, so a fresh tab can't overwrite it
let pushTimer = null;

function save(state) {
  if (!state) return;
  state.savedAt = Date.now();
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch (err) {
    /* storage full or blocked: the server copy still works */
  }
  if (serverSync && state.babyName) {
    clearTimeout(pushTimer);
    pushTimer = setTimeout(function () {
      pushToServer(state);
    }, 1500);
  }
}

function pushToServer(st) {
  if (!serverSync || !st || !st.babyName) return;
  fetch("/api/state", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(st) }).catch(
    function () {}
  );
}

// Adopt whichever copy is newer: the server's (shared by every port/tab) or this browser's
function syncFromServer() {
  if (location.protocol === "file:" || !window.fetch) return;
  fetch("/api/state")
    .then(function (r) {
      return r.ok ? r.json() : null;
    })
    .then(function (d) {
      if (!d) return;
      const remote = d.state;
      const local = state && state.babyName ? state : null;
      if (remote && remote.babyName && (!local || (remote.savedAt || 0) >= (local.savedAt || 0))) {
        state = Object.assign(blankState(), remote);
        sanitizeKnowledge(state);
        try {
          localStorage.setItem(KEY, JSON.stringify(state));
        } catch (err) {
          /* fine */
        }
        $("welcome").hidden = true;
        applyOutfit();
        Baby.rest();
        render();
        showBubble(state.asleep ? "zzz…" : "Hi " + (state.parentName || "") + "! 💛");
        serverSync = true;
      } else {
        serverSync = true;
        if (local) pushToServer(local); // this browser had the newer baby: give the server a copy
      }
    })
    .catch(function () {});
}

window.addEventListener("pagehide", function () {
  // last save when the tab closes
  if (serverSync && state && state.babyName && navigator.sendBeacon) {
    navigator.sendBeacon("/api/state", new Blob([JSON.stringify(state)], { type: "application/json" }));
  }
});

function mindScore(state) {
  return clamp((state.knowledge || []).length * 9 + (state.xp || 0) / 4);
}

function stageFrom(state) {
  const facts = (state.knowledge || []).length;
  const score = mindScore(state);
  if (facts < 2 && score < 18) return "spark";
  if (facts < 6) return "learner";
  if (facts < 14) return "thinker";
  return "young mind";
}

function renameInMemory(state, oldName, newName) {
  if (!oldName || !newName || oldName === newName) return;
  const re = new RegExp("\\b" + oldName.replace(/[^A-Za-z]/g, "") + "\\b", "g");
  (state.knowledge || []).forEach(function (k) {
    k.text = k.text.replace(re, newName);
    k.tags = contentWords(k.text);
  });
}

// Set when the rules know the exact right reply (naming), so the model isn't asked
let ruleCertain = false;

function hear(state, utterance) {
  const words = tokens(utterance);
  const now = Date.now();
  for (const word of words) {
    const entry = state.lexicon[word] || { count: 0, firstHeard: now };
    entry.count += 1;
    entry.lastHeard = now;
    state.lexicon[word] = entry;
  }
  const nameCue = utterance.match(/\b(?:your name is|you're called|you are called|i'll call you)\s+([A-Z][A-Za-z]+)/);
  if (nameCue) {
    renameInMemory(state, state.babyName, nameCue[1]);
    state.babyName = nameCue[1];
  }
  const parentCue = utterance.match(/\b(?:my name is|call me)\s+([A-Za-z]+)/i);
  if (parentCue) {
    const name = parentCue[1].charAt(0).toUpperCase() + parentCue[1].slice(1);
    renameInMemory(state, state.parentName, name);
    state.parentName = name;
  }
  const content = contentWords(utterance);
  if (content.length) state.topic = content[content.length - 1];
  state.lastHeard = utterance;
  return words;
}

function isQuestion(text) {
  const t = text.trim();
  return /\?/.test(t) || /^(what|who|why|where|how|when|do|does|did|can|could|is|are|am|tell me)\b/i.test(t);
}

const FILLERS = new Set([
  "correct", "yes", "yeah", "yep", "no", "nope", "okay", "ok", "sure", "fine",
  "good", "bad", "right", "wrong", "true", "false", "thanks", "thank you",
  "maybe", "idk", "please", "nothing", "anything", "everything", "cool",
  "nice", "great", "awesome", "hello", "hi", "hey"
]);

function shiftPerspective(text) {
  // Convert caregiver's 2nd-person teachings into baby's 1st-person knowledge:
  // "your name is Monika" -> "my name is Monika"
  // "you are loved" -> "I am loved"
  // "you can speak" -> "I can speak"
  let s = " " + text.trim() + " ";
  s = s.replace(/\b(?:your|ur)\s+name\s+is\b/gi, "my name is");
  s = s.replace(/\byour\b/gi, "my");
  s = s.replace(/\byou are\b/gi, "I am");
  s = s.replace(/\byou're\b/gi, "I am");
  s = s.replace(/\byou have\b/gi, "I have");
  s = s.replace(/\byou can\b/gi, "I can");
  s = s.replace(/\byou will\b/gi, "I will");
  s = s.replace(/\byou\b/gi, "I");
  return s.trim();
}

function formatAsSubclause(text) {
  let s = String(text || "").trim().replace(/[.?!]+$/, "");
  if (!s) return "";
  if (!/^I\b/.test(s) && !/^My\b/.test(s)) {
    s = s.charAt(0).toLowerCase() + s.slice(1);
  }
  return s;
}

function cleanFact(text) {
  return text
    .trim()
    .replace(/^[.!\s,]+/, "")
    .replace(/[.?!]+$/, "")
    .replace(/^(teach\s*:|teach\s+me\s+that|remember\s+that|learn\s+that|fact\s*:|here\s+is\s+a\s+fact\s*:)/i, "")
    .replace(/^(please\s+)?(remember|learn|know|keep|store)(\s+that)?\s+/i, "")
    .replace(/^(did you know\s+(that)?\s*)/i, "")
    .trim();
}

function extractFact(text) {
  const isExplicitTeaching = /^(teach\s*:|teach\s+me\s+that|remember\s+that|did\s+you\s+know\s+(that)?|here\s+is\s+a\s+fact\s*:|fact\s*:)/i.test(text);

  const t = cleanFact(text);
  if (!t || GREET.test(t) || isQuestion(t)) return null;
  if (/^(i love you|love you)\b/i.test(t)) return null;

  const tLower = t.toLowerCase();
  if (FILLERS.has(tLower)) return null;

  // Disqualify personal/conversational 1st & 2nd person statements unless explicitly prefixed with teach:
  if (!isExplicitTeaching) {
    if (/^(\s*you\s+know\s+)?(i|we|my|our|you|your|he|she|his|her|they|their)\b/i.test(t)) return null;
    if (/^(because|since|if|although|unless|so that|actually|just|maybe|well|honestly|so|also|no|yes)\b/i.test(t)) return null;
    if (/\b(name\s+is|named|called)\b/i.test(t)) return null;
  }

  const shifted = shiftPerspective(t);

  // Pattern 1: Copula ("X is/are/means/was/were/can/has/have Y")
  let m = shifted.match(/^(.{2,48}?)\s+(is|are|means|was|were|has|have|can)\s+(.+)$/i);
  if (m && !/^(there|here|what|who|how|why|i|you|he|she|they|it|we)$/i.test(m[1].trim())) {
    const subj = m[1].trim();
    const verb = m[2].toLowerCase();
    const rest = m[3].trim();
    const formatted = subj.charAt(0).toUpperCase() + subj.slice(1) + " " + verb + " " + rest;
    return {
      text: formatted,
      subject: subj.toLowerCase(),
      kind: "isa",
    };
  }

  // Pattern 2: Action or trait statements with sufficient tokens (ONLY for explicit teaching)
  if (isExplicitTeaching) {
    const words = tokens(shifted);
    if (words.length >= 3) {
      const formatted = shifted.charAt(0).toUpperCase() + shifted.slice(1);
      const subj = contentWords(shifted)[0] || words[0] || "idea";
      return {
        text: formatted,
        subject: subj.toLowerCase(),
        kind: "statement",
      };
    }
  }

  return null;
}

function remember(state, raw, extraSubject) {
  let parsed = extractFact(raw);
  const shortAnswer = tokens(raw).length <= 6 && !/\b(i|you|me|my|your|said|no|not|don't|yes|okay)\b/i.test(raw);
  if (!parsed && extraSubject && shortAnswer && !isQuestion(raw) && !GREET.test(raw.trim())) {
    parsed = {
      text: extraSubject + " is " + cleanFact(raw),
      subject: extraSubject.toLowerCase(),
      kind: "taught",
    };
  }
  if (!parsed) return null;

  const tags = contentWords(parsed.text);
  const key = parsed.text.toLowerCase();
  const same = state.knowledge.find(function (k) {
    return k.text.toLowerCase() === key;
  });
  if (same) {
    same.text = parsed.text;
    same.tags = tags;
    same.updated = Date.now();
    return { item: same, updated: true };
  }

  const item = {
    text: parsed.text,
    subject: parsed.subject,
    kind: parsed.kind,
    tags: tags,
    at: Date.now(),
  };
  state.knowledge.push(item);
  state.xp += 12;
  state.needs.curiosity = clamp(state.needs.curiosity + 10);
  if (!state.firsts.fact) state.firsts.fact = { at: Date.now(), text: item.text };
  return { item: item, updated: false };
}

// Facts about the caregiver/baby (pets, likes, favorites) are keyed, so a correction
// replaces the old fact instead of leaving two contradicting ones in memory.
function upsertFact(state, key, text) {
  const existing = state.knowledge.find(function (k) {
    return k.key === key;
  });
  if (existing) {
    existing.text = text;
    existing.tags = contentWords(text);
    existing.updated = Date.now();
    return existing;
  }
  const item = { text: text, subject: key, kind: "about", key: key, tags: contentWords(text), at: Date.now() };
  state.knowledge.push(item);
  state.xp += 12;
  if (!state.firsts.fact) state.firsts.fact = { at: Date.now(), text: text };
  return item;
}

function recall(state, query) {
  const q = contentWords(query);
  if (!q.length || !state.knowledge.length) return [];
  return state.knowledge
    .map(function (k) {
      let score = 0;
      (k.tags || []).forEach(function (t) {
        q.forEach(function (w) {
          if (t === w) score += 3;
          else if (t.indexOf(w) !== -1 || w.indexOf(t) !== -1) score += 1;
        });
      });
      if (k.subject && q.indexOf(k.subject) !== -1) score += 4;
      return { k: k, score: score };
    })
    .filter(function (x) {
      return x.score > 0;
    })
    .sort(function (a, b) {
      return b.score - a.score;
    })
    .map(function (x) {
      return x.k;
    });
}

function relatedFact(state, item) {
  if (!item) return null;
  const tags = item.tags || [];
  return (
    state.knowledge.find(function (k) {
      if (k.text === item.text || k.key) return false;
      const shared = (k.tags || []).filter(function (t) {
        return tags.indexOf(t) !== -1;
      });
      return shared.length >= 2 || (item.subject && k.subject === item.subject);
    }) || null
  );
}

function maybeFirsts(state, kind, text) {
  if (state.firsts[kind]) return null;
  state.firsts[kind] = { at: Date.now(), text: text };
  return kind + ": " + text;
}

function respondToCaregiver(state, heard) {
  const parent = state.parentName || "you";
  const me = state.babyName || "I";
  const raw = heard.trim();
  const lower = raw.toLowerCase();
  state.caregiverInfo = state.caregiverInfo || {};
  state.convoContext = state.convoContext || {};

  // 0. Naming: "call me Yasi" / "your name is Pip" (hear() already stored the name)
  const callMe = raw.match(/\b(?:call me|my name is)\s+([A-Za-z]+)/i);
  if (callMe) {
    ruleCertain = true;
    const name = state.parentName;
    return pick([
      "Yay! Hi, " + name + "! I'll call you " + name + " now! 💛",
      name + "! I love your name! Hehe.",
      "Okay, " + name + "! " + name + ", " + name + "... I'll remember!",
    ]);
  }
  const yourName = raw.match(/\b(?:your name is|you're called|you are called|i'll call you)\s+([A-Z][A-Za-z]+)/);
  if (yourName) {
    ruleCertain = true;
    return pick(["My name is " + state.babyName + "? I love it! Hehe!", state.babyName + "! That's me! Yay!"]);
  }

  // 1. Follow-up to baby's previous question
  const lastQ = state.convoContext.lastQuestionAsked;
  if (lastQ === "pet_name") {
    // E.g. "her name is Minik", "his name is Charlie", "Minik", "it's Minik", "she is called Minik"
    const nameMatch = raw.match(/(?:her|his|their|its|my pet's|my dog's|my cat's)?\s*(?:name\s+is|is\s+called|named|it's|its)?\s*([a-zA-Z]+)/i);
    let petName = nameMatch ? nameMatch[1] : raw.replace(/[^a-zA-Z]/g, "").trim();
    if (petName && !/^(yes|no|she|he|they|it|the|my|her|his|a|an)$/i.test(petName)) {
      petName = petName.charAt(0).toUpperCase() + petName.slice(1).toLowerCase();
      state.caregiverInfo.petName = petName;
      state.convoContext.lastQuestionAsked = "pet_play";
      const petKind = state.caregiverInfo.pet || "pet";
      upsertFact(state, "pet", parent + " has a " + petKind + " named " + petName);
      return pick([
        petName + "! Oh, what a sweet name for a " + petKind + "! Does " + petName + " come visit my crib?",
        petName + "! That is such a darling name, " + parent + ". Does " + petName + " like to play with you?",
        petName + "! I love that name! When I get bigger, can I pet " + petName + "?"
      ]);
    }
  }

  if (lastQ === "pet_play") {
    state.convoContext.lastQuestionAsked = null;
    if (/\b(yes|yeah|yep|sure|a lot|always|she does|he does|they do|of course)\b/i.test(lower)) {
      return pick([
        "Aww, that makes me smile! I bet " + (state.caregiverInfo.petName || "your pet") + " is so happy having you as a parent too.",
        "Hehe! When " + (state.caregiverInfo.petName || "your pet") + " plays, do they make silly sounds? I giggle when I hear playful sounds."
      ]);
    }
    if (/\b(no|nope|not really|sleeps|sleepy|lazy)\b/i.test(lower)) {
      return (state.caregiverInfo.petName || "Your pet") + " likes quiet naps, just like me! We can nap together in the nursery.";
    }
  }

  if (lastQ === "general_chat") {
    state.convoContext.lastQuestionAsked = null;
    return pick([
      "Ooh, really?! Tell me more, " + parent + "!",
      "I love hearing your stories! What happened next?",
      "That is so interesting! I'm listening with both of my ears wide open."
    ]);
  }

  // 2. Questions / chat about the pet
  if (/\b(do you (like|love) (dogs|cats|pets|animals)|do you want to meet (my dog|my cat|minik)|meet minik)\b/i.test(lower)) {
    if (state.caregiverInfo.petName) {
      return "Yes! I really want to meet " + state.caregiverInfo.petName + "! Can you bring " + state.caregiverInfo.petName + " to say hello to me?";
    }
    return "Yes! I love animals. When I get bigger, I want to wave my hands and play with them!";
  }

  if (/\b(who is minik|what is my (dog|cat|pet)'s name|what is (her|his|their) name|do you remember my (dog|cat|pet))\b/i.test(lower)) {
    if (state.caregiverInfo.petName) {
      return state.caregiverInfo.petName + "! " + state.caregiverInfo.petName + " is your " + (state.caregiverInfo.pet || "dog") + "! See, I remember what you tell me, " + parent + "!";
    }
    return "You told me you have a pet! Tell me their name again so I can remember it forever.";
  }

  // 3. Pet's name declared directly
  const directName = raw.match(/\b(?:her|his|their|my\s+pet's|my\s+dog's|my\s+cat's)\s+name\s+is\s+([A-Za-z]+)\b/i) ||
                     raw.match(/\b(?:she|he)\s+is\s+called\s+([A-Za-z]+)\b/i);
  if (directName && !/^(a|an|the|good|bad|cute|what)$/i.test(directName[1])) {
    const pName = directName[1].charAt(0).toUpperCase() + directName[1].slice(1).toLowerCase();
    state.caregiverInfo.petName = pName;
    state.convoContext.lastQuestionAsked = "pet_play";
    const petKind = state.caregiverInfo.pet || "pet";
    upsertFact(state, "pet", parent + " has a " + petKind + " named " + pName);
    return pick([
      pName + "! Oh, what a sweet name for a " + petKind + "! Does " + pName + " come visit my crib?",
      pName + "! That is such a darling name, " + parent + ". Does " + pName + " like to play with you?",
      pName + "! I love that name. Does " + pName + " cuddle with you when you're sleepy?"
    ]);
  }

  // 4. Pet Declarations
  // E.g. "you know i have a dog", "i have a puppy", "we have a cat", "i got a kitten"
  if (!isQuestion(raw)) {
    const petMatch = lower.match(/\b(?:you\s+know\s+)?(?:i|we)\s+(?:have|got|own)\s+(?:a|an)\s+(dog|cat|puppy|kitten|pet|bunny|rabbit|bird|hamster|fish|parrot|turtle)\b/i) ||
                     lower.match(/\b(?:my|our)\s+(dog|cat|puppy|kitten|pet|bunny|rabbit|bird|hamster|fish)\b/i);
    if (petMatch) {
      const petKind = petMatch[1].toLowerCase();
      state.caregiverInfo.pet = petKind;
      
      // Check if name was already given in this sentence: e.g. "i have a dog named Minik" or "her name is Minik"
      const inlineName = raw.match(/\b(?:named|her\s+name\s+is|his\s+name\s+is|their\s+name\s+is|called)\s+([A-Za-z]+)\b/i);
      if (inlineName && !/^(a|an|the|my|your)$/i.test(inlineName[1])) {
        const pName = inlineName[1].charAt(0).toUpperCase() + inlineName[1].slice(1).toLowerCase();
        state.caregiverInfo.petName = pName;
        state.convoContext.lastQuestionAsked = "pet_play";
        upsertFact(state, "pet", parent + " has a " + petKind + " named " + pName);
        return pick([
          "A " + petKind + " named " + pName + "! That sounds so wonderful, " + parent + "! Does " + pName + " like to cuddle?",
          pName + " the " + petKind + "! What a sweet name. Does " + pName + " know about me yet?"
        ]);
      }

      state.convoContext.lastQuestionAsked = "pet_name";
      upsertFact(state, "pet", parent + " has a " + petKind);
      return pick([
        "You have a " + petKind + ", " + parent + "?! Really? What is their name?",
        "A " + petKind + "! I love animals so much. What is your " + petKind + "'s name?",
        "Aww, a " + petKind + "! Tell me, what do you call them, " + parent + "?"
      ]);
    }
  }

  // 5. Affection, compliments, and reassurance
  if (/\b(you are|you're)\s+(so\s+)?(cute|adorable|sweet|precious|smart|clever|pretty|beautiful|good|funny)\b/i.test(lower)) {
    const adjMatch = lower.match(/\b(cute|adorable|sweet|precious|smart|clever|pretty|beautiful|good|funny)\b/);
    const adj = adjMatch ? adjMatch[1] : "sweet";
    return pick([
      "Hehe, thank you " + parent + "! You make me feel so warm inside.",
      "Giggle! If I am " + adj + ", it's because you are such a wonderful " + parent + ".",
      "You are the sweetest, " + parent + "! I love hearing you talk to me."
    ]);
  }

  if (/\b(i love you|love you so much|i adore you|i care about you)\b/i.test(lower)) {
    return pick([
      "I love you too, " + parent + "! More than all the stars above my crib.",
      "I love you so much! You are my safe place.",
      "Hearing you say that makes my heart flutter happily, " + parent + "!"
    ]);
  }

  if (/\bdo you love me\b/i.test(lower)) {
    return "Of course I love you, " + parent + "! You are my parent and my favorite person.";
  }

  if (/\bare you happy\b/i.test(lower)) {
    return "I am so happy whenever you are here talking with me, " + parent + "!";
  }

  if (/\b(you are my baby|my sweet baby|my little one)\b/i.test(lower)) {
    return "I am your baby, " + parent + "! And you are my whole world.";
  }

  // 6. Caregiver's feelings, routine & day
  if (/\b(i'm|i am|feeling)\s+(so\s+)?(tired|sleepy|exhausted|worn out|sleep deprived)\b/i.test(lower) || /\b(long day|hard day|busy day)\b/i.test(lower)) {
    return pick([
      "Aw, " + parent + "... you work so hard. Rest your eyes next to my crib. We can have peaceful quiet time together.",
      "Rest a little bit, " + parent + ". I will stay right here watching the mobile peacefully."
    ]);
  }

  if (/\b(i'm|i am)\s+(so\s+)?(happy|glad|excited|joyful)\b/i.test(lower)) {
    return "Yay! When you are happy, I feel like smiling and kicking my little feet!";
  }

  if (/\b(i'm|i am)\s+(so\s+)?(sad|down|upset|crying|stressed)\b/i.test(lower)) {
    return "Oh no... I wish I could reach out and give you a warm cuddle, " + parent + ". I am always right here with you.";
  }

  if (/\b(i am|i'm)\s+(eating|drinking|cooking|having coffee|having tea|having lunch|having dinner|having breakfast)\b/i.test(lower)) {
    return "Ooh, yum! Enjoy your food, " + parent + "! Check on me later in case my tummy rumbles for milk too.";
  }

  if (/\b(good night|sweet dreams|sleep well|going to bed|going to sleep)\b/i.test(lower)) {
    return "Good night, " + parent + "! Dream of soft stars. I will sleep softly in my crib.";
  }

  if (/\bgood morning\b/i.test(lower)) {
    return "Good morning, " + parent + "! I saw the light and was waiting to hear your voice.";
  }

  // 7. Favorites ("your favorite color is blue" / "my favorite food is soup") and likes
  const fav = raw.match(/^(?:and\s+|well\s+)?(your|my)\s+favou?rite\s+([a-z]+)\s+is\s+(.+?)[.!]*$/i);
  if (fav) {
    const thing = fav[2].toLowerCase();
    const val = fav[3].trim();
    if (fav[1].toLowerCase() === "your") {
      upsertFact(state, "fav-me-" + thing, "My favorite " + thing + " is " + val);
      return "My favorite " + thing + " is " + val + "! Yay!";
    }
    upsertFact(state, "fav-you-" + thing, (state.parentName || "You") + "'s favorite " + thing + " is " + val);
    return "Your favorite " + thing + " is " + val + "! I will remember, " + parent + ".";
  }

  const prefMatch = lower.match(/\b(?:i\s+like|i\s+love|my\s+favorite)\s+([a-z0-9\s]{2,30})/i);
  if (prefMatch && !/^(learning|teaching|you|reading|facts|monika|pip)\b/i.test(prefMatch[1].trim())) {
    const item = prefMatch[1].trim();
    upsertFact(state, "like-" + item.toLowerCase(), (state.parentName || "You") + " likes " + item);
    return pick([
      "You like " + item + ", " + parent + "? I want to learn everything about what makes you smile!",
      "Ooh, " + item + "! I will remember that you love that, " + parent + "."
    ]);
  }

  // 8. Playful & interactive questions
  if (/\bwhat are you doing\b/i.test(lower)) {
    return pick([
      "Right now? Looking up at the celestial mobile, wiggling my toes, and listening to you!",
      "Just resting softly in my crib, happy that you are talking with me, " + parent + "."
    ]);
  }

  if (/\bwhat are you thinking\b/i.test(lower)) {
    if (state.caregiverInfo.petName) {
      return "I was just thinking about " + state.caregiverInfo.petName + ", and waiting to hear what you say next!";
    }
    if (state.knowledge.length) {
      return "I was thinking about you, and about how " + formatAsSubclause(pick(state.knowledge).text) + ".";
    }
    return "I was just daydreaming about the stars and listening to your voice.";
  }

  if (/\b(are you asleep|are you awake|are you there)\b/i.test(lower)) {
    return "I'm right here! Wide awake and listening to you, " + parent + ".";
  }

  if (/\b(sing|sing for me|sing a song|lullaby)\b/i.test(lower)) {
    return "Laa-laa-loo... 🎵 Twinkle, twinkle, little star... that's my lullaby for you, " + parent + "!";
  }

  if (/\btell me a secret\b/i.test(lower)) {
    return "Shh... promise not to tell? Sometimes when you rock me, I pretend to stay awake just to hear you talk a little longer.";
  }

  if (/\bguess what\b/i.test(lower)) {
    state.convoContext.lastQuestionAsked = "general_chat";
    return "What?! Tell me, tell me!";
  }

  if (/\b(thank you|thanks)\b/i.test(lower)) {
    return "You are always welcome, " + parent + "! I love talking with you.";
  }

  if (/\b(haha|hehe|lol|funny)\b/i.test(lower)) {
    return "Hehe! Hearing you laugh is my favorite sound in the nursery.";
  }

  // 9. Identity & Parent questions
  if (/\bwho are you\b/i.test(lower)) return "I am " + me + ", and I'm your baby!";
  if (/\bwho am i\b/i.test(lower)) return "You are " + parent + "! You are my parent and my teacher.";
  if (/\bwhat is my name\b/i.test(lower)) return "You are " + parent + "! I could never forget that.";
  if (/\bwhat is your name\b/i.test(lower)) return "My name is " + me + "! You gave it to me.";
  if (/\b(how are you|how do you feel)\b/i.test(lower)) {
    if (state.needs.hunger < 40) return "A little hungry, but very happy you came to talk with me!";
    if (state.needs.sleep < 40) return "A bit sleepy, but I love listening to your voice.";
    return "I feel warm, safe, and happy talking with you, " + parent + "!";
  }

  // 10. Greetings
  if (GREET.test(raw) || /^(hi|hello|hey|yo)\b/i.test(raw)) {
    return pick([
      "Hi " + parent + "! I was waiting to hear your voice.",
      "Hello " + parent + "! What are we doing together today?",
      "Hey " + parent + "! I am so happy you're here."
    ]);
  }

  // 11. Fillers & short acknowledgments
  if (/^(okay|ok|yeah|yep|sure|nice|cool|really|oh|aww|wow|oh really|no way)[.!]?$/i.test(lower)) {
    return pick([
      "Mm-hmm! What else is on your mind, " + parent + "?",
      "Hehe, yeah! Tell me more, " + parent + ".",
      "I'm listening! Tell me whatever you like."
    ]);
  }

  // 12. Conversational first-person sharing (e.g. "I went outside today", "it's raining outside")
  if (/^(\s*you\s+know\s+)?(i|we|my|our)\b/i.test(lower)) {
    return pick([
      "Really, " + parent + "? Tell me more about that!",
      "I love hearing about your day, " + parent + ". What else happened?",
      "That's so nice to know. I'm always happy when you share things with me."
    ]);
  }

  return null; // Let question or fact teaching handle it
}

// Pick an idle line that wasn't said recently, so the baby doesn't loop
function freshIdleLine(state, lines) {
  const recent = state.recentIdle || [];
  const options = lines.filter(function (l) {
    return recent.indexOf(l) === -1;
  });
  const line = pick(options.length ? options : lines);
  state.recentIdle = recent.concat(line).slice(-6);
  return line;
}

const BAD_WORDS = /\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|dicks?|pussy|whore\w*|slut\w*|bastard\w*|nigg\w*|fag\w*|retard\w*|rape\w*|porn\w*|sex\w*|nazi\w*|kill yourself|kys)\b/i;

function speak(state, intent, heard) {
  if (heard && BAD_WORDS.test(heard)) {
    ruleCertain = true;
    return pick(["Hmm, that's not a nice word. Let's use kind words! 💛", "*covers ears* Kind words, please!"]);
  }
  const parent = state.parentName || "you";
  const me = state.babyName || "I";
  const n = state.needs;
  const known = (state.knowledge || []).length;
  state.caregiverInfo = state.caregiverInfo || {};
  state.convoContext = state.convoContext || {};

  if (intent === "feed") return n.hunger > 85 ? "Thank you. I am full! Teach me something or chat while I rest." : "That was yummy! Tell me about your day while I eat.";
  if (intent === "rock" || intent === "cuddle") {
    if (state.caregiverInfo.petName) {
      return pick([
        "I feel safe with you, " + parent + ". Does " + state.caregiverInfo.petName + " cuddle with you like this too?",
        "I love being held by you, " + parent + ". You are the best teacher and parent."
      ]);
    }
    return pick(["I love you, " + parent + ". *snuggles*", "So warm and safe... *happy sigh*", "Hehe! More cuddles, please!"]);
  }
  if (intent === "play") {
    if (state.caregiverInfo.petName) {
      return "Let's play! I want to wiggle my hands like " + state.caregiverInfo.petName + " wiggles their tail!";
    }
    return pick(["Peekaboo! Hehe! Again!", "*giggles* Where did you go? There you are!", "Haha! You found me!"]);
  }
  if (intent === "tickle") return pick(["Hahaha! Stop! No, don't stop!", "*squeals* That tickles!", "Hehehe! My tummy!"]);
  if (intent === "wake") return pick(["*stretches* Good morning, " + parent + "!", "*blinks* Hi! I had a dream about stars."]);
  if (intent === "nap") return "Okay. I will dream about what we talked about. Good night, " + parent + ".";

  if (heard) {
    // Priority 1: Check conversational dialogue with caregiver
    // (Recognizes chatting, sharing pets, feelings, daily life, banter)
    const convoReply = respondToCaregiver(state, heard);
    if (convoReply) {
      return convoReply;
    }

    // Priority 2: Answering a prompt baby was awaiting (from a previous question)
    if (state.awaiting && !isQuestion(heard)) {
      const topic = state.awaiting.topic;
      const saved = remember(state, heard, topic);
      state.awaiting = null;
      if (saved) {
        return pick(["Ohh, now I know about " + topic + "! Thank you, " + parent + "!", "Ooh, " + topic + "! Got it! ✨"]);
      }
    }

    // Priority 3: Caregiver asking a question to test knowledge or ask baby
    if (isQuestion(heard)) {
      const hits = recall(state, heard);
      if (hits.length) {
        return hits[0].text.replace(/[.?!]+$/, "") + "!";
      }
      const topic = contentWords(heard)[0] || state.topic || "that";
      state.awaiting = { topic: topic, asked: heard };
      return "I do not know about " + topic + " yet. Teach me in a sentence, and I will remember it!";
    }

    // Priority 4: Attempting to learn a general fact or explicit teaching
    const saved = remember(state, heard);
    if (saved) {
      if (saved.updated) return pick(["Oh! Okay, I'll remember it that way now!", "Got it, " + parent + "!"]);
      const about = saved.item.kind === "isa" && tokens(saved.item.subject).length <= 3 ? saved.item.subject : "";
      return pick(
        (about
          ? ["Ooh, " + about + "! Got it! ✨", "Wow, " + about + "! I'll remember!", "Hehe, " + about + "! Thank you, " + parent + "!"]
          : []
        ).concat(["Ooh! I'll remember that! ✨", "Wow! Thank you, " + parent + "!", "Okay! Got it!", "Hehe, I learned something new!"])
      );
    }

    // Default conversational acknowledgment if not a fact and not caught above:
    return pick([
      "I'm listening, " + parent + "! Tell me more.",
      "I love hearing your voice, " + parent + ". Tell me whatever you like!",
      "Hehe, tell me more about your day, " + parent + "."
    ]);
  }

  // Idle thoughts
  if (n.hunger < 28) return parent + ", I am hungry. After food, let's talk more!";
  if (n.sleep < 28) return "I am sleepy. Rest with me for a bit, " + parent + ".";
  if (n.comfort < 28) return parent + ", hold me and talk to me softly.";
  if (state.awaiting) {
    return "I am still wondering! What can you teach me about " + state.awaiting.topic + "?";
  }
  if (state.caregiverInfo && state.caregiverInfo.petName && Math.random() < 0.4) {
    return pick([
      "I was just wondering what " + state.caregiverInfo.petName + " is doing right now! Does " + state.caregiverInfo.petName + " take naps too?",
      "When " + state.caregiverInfo.petName + " comes by, can I see them? I want to say hello!"
    ]);
  }
  return freshIdleLine(state, [
    "*kicks feet* Hehe!",
    "Ba-ba-ba! Goo!",
    "Ooh, the mobile is spinning!",
    "*looks at you* Hi, " + parent + "!",
    "Where's my teddy?",
    "Can we play, " + parent + "?",
    "*sucks thumb*",
    "I wonder what the moon is doing.",
    "Ga-ga! *claps*",
    "What did you do today, " + parent + "?",
    "*yawns a tiny yawn*",
    known ? "Can you teach me something new?" : "Will you teach me something?",
  ]);
}

// ==========================================================================
// Voice & little sounds
// ==========================================================================

let voiceTimer = null;
let chosenVoice = null;

// Prefer a real child voice (Edge ships "Microsoft Ana", a child), then light female voices.
// Male voices are never chosen: even at max pitch they don't sound like a baby.
function voiceScore(v) {
  const n = v.name.toLowerCase();
  if (!/^en/i.test(v.lang)) return -1;
  if (/\bana\b/.test(n)) return 100;
  if (/david|mark|guy|ryan|eric|christopher|roger|steffan|brian|andrew|daniel|fred|thomas|male/.test(n)) return 0;
  let score = 10;
  if (/jenny|aria|ava|emma|michelle|sara|zira|samantha|karen|moira|tessa|libby|sonia|google us english|female/.test(n)) score += 40;
  if (/natural|online|neural/.test(n)) score += 15;
  if (/en-us/i.test(v.lang)) score += 5;
  return score;
}

function pickVoice() {
  const voices = (window.speechSynthesis && window.speechSynthesis.getVoices()) || [];
  let best = null;
  voices.forEach(function (v) {
    if (voiceScore(v) > 0 && (!best || voiceScore(v) > voiceScore(best))) best = v;
  });
  return best;
}

// Toddler pronunciation, for the spoken voice only (the chat text stays readable).
// Strong while the baby is tiny, fading as it grows.
const BABY_WORDS = {
  love: "wuv", loves: "wuvs", loved: "wuvd", little: "widdle", really: "weawy", very: "vewy", sleepy: "sweepy",
  sleep: "sweep", please: "pwease", pretty: "pwetty", rabbit: "wabbit", friend: "fwiend", friends: "fwiends",
  sorry: "sowwy", story: "stowy", thank: "tank", thanks: "tanks", hello: "hewwo", remember: "member",
  stomach: "tummy", water: "wawa", bottle: "baba", blanket: "bwankie", dog: "doggy", cat: "kitty",
  yes: "yesh", the: "da", this: "dis", that: "dat", them: "dem", there: "dere", they: "dey", with: "wif",
};
const LIGHT_WORDS = ["love", "loves", "little", "sleepy", "please", "hello", "sorry", "thank", "thanks", "blanket"];
const FUNCTION_WORDS = ["the", "this", "that", "them", "there", "they", "with", "yes"];

function babyTalkLevel() {
  const stage = stageFrom(state);
  return stage === "spark" ? 3 : stage === "learner" ? 2 : 1;
}

function babyTalk(text, level) {
  let out = text.replace(/[A-Za-z']+/g, function (word) {
    const w = word.toLowerCase();
    if (!BABY_WORDS[w]) return word;
    if (level === 1 && LIGHT_WORDS.indexOf(w) === -1) return word;
    if (level === 2 && FUNCTION_WORDS.indexOf(w) !== -1) return word;
    return BABY_WORDS[w];
  });
  if (level >= 2) out = out.replace(/\b([bcdfgkpt])r(?=[aeiou])/gi, "$1w"); // brave -> bwave
  if (level >= 3) {
    out = out.replace(/\br(?=[aeiou])/gi, "w"); // red -> wed
    out = out.replace(/\b([bcfgkps])l(?=[aeiou])/gi, "$1w"); // play -> pway
  }
  return out;
}

function forVoice(line) {
  let text = String(line || "")
    .replace(/\*[^*]*\*/g, " ") // *giggles* is acted out by the synth, not read aloud
    .replace(/\b(he+he+|hi+hi+|ha+ha+|hi{3,}|ye+ay|yay)\b[!.]*/gi, " ") // giggles are vocalized, not spelled
    .replace(/[-—…~]/g, " ")
    .replace(/[!]+/g, ".")
    .replace(/[^\p{L}\p{N}\s.,?']/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
  const real = text.replace(/[?.!,]/g, "").split(" ").filter(function (w) {
    return w.length >= 2;
  });
  if (real.length < 1) return "";
  if (!/[.!?]$/.test(text)) text += ".";
  return babyTalk(text, babyTalkLevel());
}

// Which baby sound fits the line
function vocalFor(line) {
  const l = String(line).toLowerCase();
  if (/wa+h|\*(cr(y|ies)|sob|wail)|😭/.test(l)) return "cry";
  if (/\*(sniff|sniffle|whimper)/.test(l)) return "whimper";
  if (/\*(laugh|squeal)|ha+ha|hahaha/.test(l)) return "laugh";
  if (/\*giggl|he+he|hi+hi|yay/.test(l)) return "giggle";
  if (/\*yawn|zzz|sleepy/.test(l)) return "yawn";
  if (/\*(snuggl|nuzzl|hum|happy wiggle)|mmm/.test(l)) return "hum";
  if (/\*(sniff|sniffle)|oh no|sad/.test(l)) return "whine";
  if (/\?\s*$|ooh|wow/.test(l)) return "ooh";
  return pick(["coo", "gaga", "baba", "ah", "dada"]);
}

function voiceSay(line) {
  if (!state.voice || state.asleep || !audioUnlocked) return; // speech is also blocked before the first click
  if (voiceTimer) clearTimeout(voiceTimer);
  const synth = window.speechSynthesis;
  if (synth && (synth.speaking || synth.pending)) synth.cancel();
  // a coo / giggle first, then words
  const lead = vocal(vocalFor(line), 1.6);
  const text = forVoice(line);
  if (!text || !synth) return;
  voiceTimer = setTimeout(function () {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US";
    chosenVoice = chosenVoice || pickVoice();
    if (chosenVoice) u.voice = chosenVoice;
    const isChild = chosenVoice && /\bana\b/i.test(chosenVoice.name);
    u.pitch = isChild ? 1.35 : 2; // 2 is the browser maximum
    u.rate = isChild ? 1 : 1.12;
    if (synth.paused) synth.resume(); // Chrome can leave speech paused after the tab was hidden
    synth.speak(u);
  }, lead * 1000 + 60);
}

let audioCtx = null;
let audioUnlocked = false;

function getAudio() {
  audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

// Audio created before any click starts muted for the whole visit, so wait for the first gesture
["pointerdown", "keydown", "touchstart"].forEach(function (type) {
  document.addEventListener(
    type,
    function () {
      audioUnlocked = true;
      try {
        getAudio();
      } catch (err) {
        /* no Web Audio */
      }
    },
    { passive: true, capture: true }
  );
});
const SFX = {
  giggle: [[880, 0.07], [1046, 0.07], [1318, 0.07], [1046, 0.07], [1396, 0.1]],
  boop: [[520, 0.09], [392, 0.12]],
  suck: [[240, 0.05]],
  burp: [[150, 0.24]],
  kiss: [[1568, 0.05], [2093, 0.07]],
  lullaby: [[523, 0.4], [659, 0.4], [784, 0.4], [659, 0.4], [587, 0.4], [523, 0.7]],
  yawn: [[440, 0.3], [330, 0.45]],
  peek: [[784, 0.06], [1175, 0.14]],
  pat: [[1200, 0.04]],
};

function sfx(kind) {
  if (BABBLES[kind]) return void vocal(kind); // giggle, laugh, cry, yawn...: real baby sounds
  if (!state || !state.voice || !SFX[kind] || !audioUnlocked) return;
  try {
    getAudio();
    let t = audioCtx.currentTime;
    SFX[kind].forEach(function (note) {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.type = kind === "burp" ? "sawtooth" : "sine";
      o.frequency.setValueAtTime(note[0], t);
      if (kind === "burp") o.frequency.exponentialRampToValueAtTime(70, t + note[1]);
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(kind === "lullaby" ? 0.05 : kind === "burp" ? 0.05 : 0.08, t + 0.015);
      g.gain.exponentialRampToValueAtTime(0.0001, t + note[1]);
      o.connect(g).connect(audioCtx.destination);
      o.start(t);
      o.stop(t + note[1] + 0.02);
      t += note[1] * 0.92;
    });
  } catch (err) {
    /* audio is optional */
  }
}


// ==========================================================================
// Baby vocal synth: a buzzing "vocal cord" tone shaped by two resonances (formants)
// into vowels, pitched like a baby (~420 Hz). Makes coos, ga-ga babble, giggles, yawns.
// ==========================================================================

// Formant pairs [F1, F2] scaled up for a tiny vocal tract
const VOWELS = { a: [1100, 1900], o: [720, 1250], u: [480, 1150], e: [760, 2600], i: [480, 3100], m: [320, 1250] };
const VOWEL_LOUDNESS = { a: 1.7, e: 1.5, i: 1.2, o: 1, u: 1, m: 1 }; // narrow resonances pass less energy on open vowels

// [vowel, seconds, pitch multiplier, consonant onset, pitch glide]
const BABBLES = {
  coo: [["o", 0.2, 1.0, "", 0.08], ["u", 0.3, 1.15, "", 0.12]],
  ah: [["a", 0.32, 1.05, "", -0.1]],
  gaga: [["a", 0.13, 1.0, "g"], ["a", 0.15, 1.08, "g"], ["u", 0.26, 1.2, "g", 0.1]],
  baba: [["a", 0.14, 1.0, "b"], ["a", 0.22, 0.94, "b", -0.08]],
  dada: [["a", 0.13, 1.02, "d"], ["a", 0.22, 0.95, "d", -0.08]],
  giggle: [["i", 0.08, 1.45, "h"], ["i", 0.08, 1.55, "h"], ["e", 0.08, 1.5, "h"], ["i", 0.08, 1.6, "h"], ["i", 0.12, 1.7, "h", 0.1]],
  yawn: [["a", 0.75, 1.15, "", -0.45], ["m", 0.25, 0.75, "", -0.1]],
  hum: [["m", 0.5, 1.0, "", 0.06]],
  whine: [["e", 0.45, 1.15, "", -0.22], ["e", 0.3, 1.05, "", -0.2]],
  ooh: [["u", 0.38, 0.95, "", 0.4]],
  // belly laugh: breathy "ha-ha-ha-ha" sliding down, a gasp, then a happy squeal
  laugh: [["a", 0.11, 1.5, "h", -0.05], ["a", 0.1, 1.44, "h", -0.05], ["a", 0.1, 1.38, "h", -0.05], ["a", 0.1, 1.32, "h", -0.05],
          ["a", 0.13, 1.26, "h", -0.1], ["_", 0.2, 1], ["i", 0.18, 1.7, "h", 0.18]],
  // cry: "wa-aaah" with a wobble, gasp for air, again, then a sobbing tail
  cry: [["a", 0.16, 1.2, "w", 0.3], ["a", 0.95, 1.55, "", -0.22], ["_", 0.28, 1], ["a", 0.14, 1.25, "w", 0.3],
        ["a", 0.8, 1.5, "", -0.3], ["_", 0.25, 1], ["e", 0.4, 1.4, "", -0.28]],
  whimper: [["e", 0.22, 1.3, "", -0.12], ["_", 0.15, 1], ["e", 0.25, 1.25, "", -0.15], ["e", 0.4, 1.2, "", -0.25]],
  sniffle: [["_", 0.08, 1], ["_", 0.08, 1], ["_", 0.14, 1]],
};

// [rate Hz, depth as a fraction of pitch]: crying wobbles hard, laughing barely
const VIBRATO = { cry: [6.5, 0.06], whimper: [6, 0.045], laugh: [5, 0.015] };

let noiseBuffer = null;

// ==========================================================================
// Real recorded baby sounds (sounds/, from Pixabay — see sounds/CREDITS.md).
// Plain <audio> playback works both from a server and when index.html is opened as a file.
// Anything without a recording, or a clip that fails to load, falls back to the synth.
// ==========================================================================

const REAL_SOUNDS = {
  cry: [["sounds/cry-1.mp3", 4.2], ["sounds/cry-2.mp3", 4.0], ["sounds/cry-3.mp3", 2.6]],
  laugh: [["sounds/laugh-1.mp3", 2.9], ["sounds/laugh-2.mp3", 3.0], ["sounds/laugh-3.mp3", 3.0], ["sounds/laugh-4.mp3", 3.0]],
  giggle: [["sounds/giggle-1.mp3", 1.4], ["sounds/giggle-2.mp3", 2.7], ["sounds/giggle-3.mp3", 2.5], ["sounds/giggle-4.mp3", 2.5]],
};
const REAL_VOLUME = { cry: 0.55, laugh: 0.75, giggle: 0.75 };
const brokenClips = {};
const lastClip = {};
let playingClip = null;

function playReal(kind, maxSeconds) {
  const clips = (REAL_SOUNDS[kind] || []).filter(function (c) {
    return !brokenClips[c[0]];
  });
  if (!clips.length) return null;
  let clip = pick(clips);
  if (clips.length > 1 && clip[0] === lastClip[kind]) clip = clips[(clips.indexOf(clip) + 1) % clips.length];
  lastClip[kind] = clip[0];

  stopReal(0);
  const audio = new Audio(clip[0]);
  audio.volume = REAL_VOLUME[kind] || 0.7;
  playingClip = audio;
  audio.play().catch(function () {
    brokenClips[clip[0]] = true; // missing or blocked: use the synth next time
    babble(kind);
  });
  if (maxSeconds && maxSeconds < clip[1]) {
    setTimeout(function () {
      if (playingClip === audio) stopReal(250);
    }, maxSeconds * 1000);
    return maxSeconds;
  }
  return clip[1];
}

// Fade out whatever recording is playing (e.g. a cry once the baby is comforted)
function stopReal(fadeMs) {
  const audio = playingClip;
  if (!audio) return;
  playingClip = null;
  if (!fadeMs) {
    audio.pause();
    return;
  }
  const steps = 10;
  const drop = audio.volume / steps;
  let i = 0;
  const iv = setInterval(function () {
    audio.volume = Math.max(0, audio.volume - drop);
    if (++i >= steps) {
      clearInterval(iv);
      audio.pause();
    }
  }, fadeMs / steps);
}

// One entry point for every baby noise: a real recording when there is one, else the synth
function vocal(kind, maxSeconds) {
  if (!state || !state.voice || !audioUnlocked) return 0;
  const real = playReal(kind, maxSeconds);
  return real == null ? babble(kind) : real;
}

function babble(kind) {
  if (!state || !state.voice || !BABBLES[kind] || !audioUnlocked) return 0;
  try {
    getAudio();
    const ctx = audioCtx;
    if (!noiseBuffer) {
      noiseBuffer = ctx.createBuffer(1, ctx.sampleRate * 0.2, ctx.sampleRate);
      const d = noiseBuffer.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    }
    const out = ctx.createGain();
    out.gain.value = 4.2; // formant filters are narrow, so the raw buzz needs lifting
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -9;
    limiter.ratio.value = 12;
    const tone = ctx.createBiquadFilter(); // soften the buzz
    tone.type = "lowpass";
    tone.frequency.value = 4200;
    const trim = ctx.createGain();
    trim.gain.value = 0.8; // headroom: the limiter's attack lets the first transient through
    tone.connect(out).connect(limiter).connect(trim).connect(ctx.destination);

    const f0 = 410 + Math.random() * 50;
    const start = ctx.currentTime + 0.02;
    let t = start;
    const vib = VIBRATO[kind] || [5.5, 0.025];
    BABBLES[kind].forEach(function (syl) {
      if (syl[0] === "_") {
        // a breath: the gasp between sobs and laughs (or a sniffle)
        const n = ctx.createBufferSource();
        n.buffer = noiseBuffer;
        const bp = ctx.createBiquadFilter();
        bp.type = "bandpass";
        bp.frequency.value = kind === "sniffle" ? 3200 : 1800;
        bp.Q.value = 1.2;
        const ng = ctx.createGain();
        ng.gain.setValueAtTime(0.0001, t);
        ng.gain.exponentialRampToValueAtTime(0.22, t + syl[1] * 0.7);
        ng.gain.exponentialRampToValueAtTime(0.0001, t + syl[1]);
        n.connect(bp).connect(ng).connect(tone);
        n.start(t);
        n.stop(t + syl[1] + 0.02);
        t += syl[1] + 0.03;
        return;
      }
      const v = VOWELS[syl[0]];
      const dur = syl[1];
      const pitch = f0 * syl[2];
      const cons = syl[3] || "";
      const glide = syl[4] || 0;

      const src = ctx.createOscillator();
      src.type = "sawtooth";
      src.frequency.setValueAtTime(pitch, t);
      src.frequency.linearRampToValueAtTime(pitch * (1 + glide), t + dur);
      const lfo = ctx.createOscillator(); // baby wobble
      const lfoGain = ctx.createGain();
      lfo.frequency.value = vib[0];
      lfoGain.gain.value = pitch * vib[1];
      lfo.connect(lfoGain).connect(src.frequency);

      const env = ctx.createGain();
      const peak = (syl[0] === "m" ? 0.5 : 0.32) * VOWEL_LOUDNESS[syl[0]];
      const onset = cons && cons !== "h" && cons !== "w" ? 0.025 : 0; // closed lips before b/d/g
      env.gain.setValueAtTime(0.0001, t);
      env.gain.setValueAtTime(0.0001, t + onset);
      env.gain.exponentialRampToValueAtTime(peak, t + onset + 0.03);
      env.gain.setValueAtTime(peak, t + dur * 0.6);
      env.gain.exponentialRampToValueAtTime(0.0001, t + dur);

      // Two formant resonances in parallel; consonants glide F2 into the vowel
      // ("w" is a slow glide from rounded lips: the "wa" of "waaah")
      const f2Start = { b: 0.65, d: 1.2, g: 1.3, w: 0.45 }[cons] || 1;
      const f1Start = cons === "w" ? 0.6 : 1;
      const glideTime = cons === "w" ? 0.14 : 0.05;
      [[v[0], 6, 1, f1Start], [v[1], 9, 0.55, f2Start]].forEach(function (f) {
        const bp = ctx.createBiquadFilter();
        bp.type = "bandpass";
        bp.Q.value = f[1];
        bp.frequency.setValueAtTime(f[0] * f[3], t);
        bp.frequency.exponentialRampToValueAtTime(f[0], t + onset + glideTime);
        const g = ctx.createGain();
        g.gain.value = f[2] * (syl[0] === "m" && f[2] < 1 ? 0.25 : 1);
        src.connect(bp).connect(g).connect(env);
      });
      env.connect(tone);

      if (cons === "h") {
        // breathy giggle: a puff of air at the start of each "hee"
        const n = ctx.createBufferSource();
        n.buffer = noiseBuffer;
        const hp = ctx.createBiquadFilter();
        hp.type = "highpass";
        hp.frequency.value = 2500;
        const ng = ctx.createGain();
        ng.gain.setValueAtTime(0.12, t);
        ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
        n.connect(hp).connect(ng).connect(tone);
        n.start(t);
        n.stop(t + 0.06);
      }

      src.start(t);
      lfo.start(t);
      src.stop(t + dur + 0.02);
      lfo.stop(t + dur + 0.02);
      t += dur + (cons === "h" ? 0.035 : 0.05);
    });
    return t - start;
  } catch (err) {
    return 0; // audio is optional
  }
}

// ==========================================================================
// DOM helpers, particles, speech bubble
// ==========================================================================

const $ = function (id) {
  return document.getElementById(id);
};
let state = load();

function wait(ms) {
  return new Promise(function (resolve) {
    setTimeout(resolve, ms);
  });
}

function moodOf(n) {
  if (n.sleep < 25) return "sleepy";
  if (n.hunger < 28) return "hungry";
  if (n.comfort < 25) return "sad";
  if (n.bond > 60) return "loved";
  return "curious";
}

function babyPoint(fx, fy) {
  const r = $("room").getBoundingClientRect();
  const b = $("baby").getBoundingClientRect();
  return [b.left - r.left + b.width * fx, b.top - r.top + b.height * fy];
}

function fxAt(x, y, glyph, opts) {
  opts = opts || {};
  const s = document.createElement("span");
  s.className = "fx " + (opts.cls || "");
  s.textContent = glyph;
  s.style.left = x + "px";
  s.style.top = y + "px";
  s.style.setProperty("--dx", (Math.random() - 0.5) * (opts.spread || 80) + "px");
  s.style.setProperty("--rot", (Math.random() - 0.5) * 50 + "deg");
  s.style.setProperty("--rise", (opts.rise || 110) + "px");
  s.style.setProperty("--size", (opts.size || 20) + "px");
  s.style.setProperty("--dur", (opts.dur || 1.6) + "s");
  s.addEventListener("animationend", function () {
    s.remove();
  });
  $("fx").appendChild(s);
}

function burst(glyphs, count, opts) {
  opts = opts || {};
  for (let i = 0; i < count; i++) {
    setTimeout(function () {
      const p = babyPoint(opts.fx == null ? 0.5 : opts.fx, opts.fy == null ? 0.3 : opts.fy);
      const g = Array.isArray(glyphs) ? pick(glyphs) : glyphs;
      fxAt(p[0] + (Math.random() - 0.5) * (opts.jitter || 70), p[1] + (Math.random() - 0.5) * 30, g, opts);
    }, i * (opts.stagger || 90));
  }
}

function showBubble(text, cls) {
  const b = $("bubble");
  b.classList.remove("thinking", "pop");
  if (cls) b.classList.add(cls);
  b.textContent = text;
  void b.offsetWidth;
  b.classList.add("pop");
}

// ==========================================================================
// Baby animation controller: action (what they're doing) + mood -> face
// ==========================================================================

const FACES = {
  sleeping: ["closed", "sleep"],
  stirring: ["closed", "pout"],
  feeding: ["happy", "o"],
  burp: ["open", "o"],
  cuddling: ["happy", "laugh"],
  giggling: ["happy", "laugh"],
  "peek-hide": ["open", "o"],
  "peek-boo": ["happy", "laugh"],
  refusing: ["closed", "pout"],
  yawning: ["closed", "o"],
  listening: ["open", "smile"],
  dressing: ["happy", "laugh"],
  crying: ["cry", "cry"],
};

const MOOD_FACES = {
  sleepy: ["open", "sleep"],
  hungry: ["open", "pout"],
  sad: ["open", "pout"],
  loved: ["open", "smile"],
  curious: ["open", "smile"],
};

const Baby = {
  action: "idle",
  timer: null,
  talkTimer: null,

  busy: function () {
    return this.action !== "idle" && this.action !== "listening";
  },

  rest: function () {
    this.set(state && state.asleep ? "sleeping" : "idle");
  },

  set: function (action, ms, then) {
    clearTimeout(this.timer);
    this.action = action;
    this.apply();
    const self = this;
    if (ms) {
      this.timer = setTimeout(function () {
        if (then) then();
        else self.rest();
      }, ms);
    }
  },

  apply: function () {
    const el = $("baby");
    if (!el || !state) return;
    const mood = moodOf(state.needs);
    let face = FACES[this.action] || MOOD_FACES[mood];
    if (this.action === "rocking") face = [mood === "sleepy" ? "closed" : "happy", "smile"];
    el.dataset.action = this.action;
    el.dataset.mood = mood;
    el.dataset.eyes = face[0];
    if (!this.talkTimer) el.dataset.mouth = face[1];
  },

  // Lip flap while the bubble appears
  talk: function (text) {
    clearInterval(this.talkTimer);
    this.talkTimer = null;
    if (this.busy()) return this.apply();
    const el = $("baby");
    const self = this;
    let flaps = Math.min(22, String(text).split(/\s+/).length * 2);
    let open = false;
    this.talkTimer = setInterval(function () {
      open = !open;
      el.dataset.mouth = open ? "o" : "smile";
      if (--flaps <= 0 || self.busy()) {
        clearInterval(self.talkTimer);
        self.talkTimer = null;
        self.apply();
      }
    }, 140);
  },
};

function lookAt(x, y) {
  document.querySelectorAll(".b-eyeball").forEach(function (g) {
    g.style.transform = "translate(" + x.toFixed(1) + "px," + y.toFixed(1) + "px)";
  });
}

// ==========================================================================
// Rendering
// ==========================================================================

function pushChat(who, text, learned, extra) {
  state.chat.push(Object.assign({ who: who, text: text, learned: !!learned, at: Date.now() }, extra || {}));
  state.chat = state.chat.slice(-60);
}

function escapeHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatChatTime(ts) {
  if (!ts) return "";
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  } catch (err) {
    return "";
  }
}

// Mini portrait of the baby for the chat (matches the big SVG baby)
function babyFace(look) {
  look = look || "open";
  const eyes = {
    open:
      '<ellipse cx="15" cy="22" rx="2.2" ry="2.7" fill="#3b2a22"/><ellipse cx="25" cy="22" rx="2.2" ry="2.7" fill="#3b2a22"/>' +
      '<circle cx="15.8" cy="20.9" r=".9" fill="#fff"/><circle cx="25.8" cy="20.9" r=".9" fill="#fff"/>',
    happy: '<path d="M12.6 23.2q2.4-3.2 4.8 0M22.6 23.2q2.4-3.2 4.8 0" fill="none" stroke="#3b2a22" stroke-width="1.5" stroke-linecap="round"/>',
    closed: '<path d="M12.6 22q2.4 2.2 4.8 0M22.6 22q2.4 2.2 4.8 0" fill="none" stroke="#3b2a22" stroke-width="1.4" stroke-linecap="round"/>',
  }[look];
  const mouth = look === "closed" ? "M18.8 27.6q1.2.8 2.4 0" : "M17.4 27.2q2.6 2.6 5.2 0";
  const cap =
    look === "closed"
      ? '<path d="M6.5 16Q11 3 25 5q10 2 11.5 11Q27 11.5 18 11.5 11 11.5 6.5 16z" fill="#a5b4fc"/>' +
        '<path d="M6.5 16.4Q19 10 33 15" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/><circle cx="36.6" cy="16.6" r="2.6" fill="#fff"/>'
      : '<path d="M19 8.5c-1.6-3.8 3.6-5.6 4.6-2.2.6 2-1.8 3-2.8 1.4" fill="none" stroke="#c98b5e" stroke-width="1.8" stroke-linecap="round"/>';
  return (
    '<svg viewBox="0 0 40 40" aria-hidden="true">' +
    '<circle cx="6.5" cy="23" r="4" fill="#ffd6bd"/><circle cx="33.5" cy="23" r="4" fill="#ffd6bd"/>' +
    '<circle cx="20" cy="22" r="14.5" fill="#ffe2cf"/><circle cx="15" cy="16.5" r="6" fill="#fff4ec" opacity=".75"/>' +
    '<ellipse cx="11.6" cy="26.2" rx="3.2" ry="2" fill="#ff9fb2" opacity=".75"/><ellipse cx="28.4" cy="26.2" rx="3.2" ry="2" fill="#ff9fb2" opacity=".75"/>' +
    eyes +
    '<path d="' + mouth + '" fill="none" stroke="#a8473f" stroke-width="1.5" stroke-linecap="round"/>' +
    cap +
    "</svg>"
  );
}

function renderNeeds() {
  const n = state.needs;
  ["hunger", "sleep", "comfort", "curiosity", "bond"].forEach(function (key) {
    const el = $("need-" + key);
    if (el) el.value = n[key];
    const valEl = $("val-" + key);
    if (valEl) valEl.textContent = Math.round(n[key]) + "%";
  });
  const mind = mindScore(state);
  if ($("need-mind")) $("need-mind").value = mind;
  if ($("val-mind")) $("val-mind").textContent = Math.round(mind) + "%";
  Baby.apply();
}

function render() {
  const stage = stageFrom(state);
  const babyName = state.babyName || "???";
  const parentName = state.parentName || "you";
  const know = state.knowledge || [];
  const mood = moodOf(state.needs);

  $("baby-name").textContent = babyName;
  $("age-label").textContent = stage + " · day " + Math.floor(state.ageHours / 24) + " · " + know.length + " facts";
  $("stage-chip").textContent = stage;
  $("room").classList.toggle("is-asleep", !!state.asleep);
  renderNeeds();

  if ($("know-count-badge")) $("know-count-badge").textContent = know.length + (know.length === 1 ? " fact" : " facts");
  $("know-list").innerHTML = know.length
    ? know
        .slice()
        .reverse()
        .map(function (k) {
          return '<li class="know-card"><span class="know-icon">💡</span><span class="know-text">' + escapeHtml(k.text) + "</span></li>";
        })
        .join("")
    : '<li class="know-empty"><span>🌱</span> No facts yet — teach something true!</li>';

  if ($("convo-title")) $("convo-title").textContent = "Talking with " + babyName;
  if ($("convo-avatar")) $("convo-avatar").innerHTML = babyFace(state.asleep ? "closed" : mood === "loved" ? "happy" : "open");
  const shownMood = state.asleep ? "sleeping" : mood;
  const moodBadge = $("convo-mood-badge");
  if (moodBadge) {
    moodBadge.textContent = shownMood;
    moodBadge.className = "mood-badge mood-" + (state.asleep ? "sleepy" : mood);
  }
  const statusSub = $("convo-status-sub");
  if (statusSub) {
    statusSub.innerHTML =
      escapeHtml(babyName + " is " + shownMood + " · " + know.length + (know.length === 1 ? " fact" : " facts")) +
      (brainOnline
        ? ' <span class="brain-chip" title="Replies come from the BabyGPT model reasoning over the Memory Bank">🧠 neural brain</span>'
        : ' <span class="brain-chip offline" title="Run python app.py to give the baby its neural brain">rules only</span>');
  }

  const chatList = state.chat || [];
  if ($("journal-empty")) $("journal-empty").hidden = chatList.length > 0;
  $("journal").innerHTML = chatList
    .map(function (e) {
      const timeStr = formatChatTime(e.at);
      const time = timeStr ? '<time class="msg-time">' + timeStr + "</time>" : "";
      if (e.who === "nursery") {
        return (
          '<li class="chat-msg system"><div class="system-pill"><span class="system-icon">' + (e.icon || "📖") + "</span> " +
          '<span class="system-text">' + escapeHtml(e.text) + "</span>" + (timeStr ? '<time class="system-time">' + timeStr + "</time>" : "") +
          "</div></li>"
        );
      }
      const isParent = e.role ? e.role === "user" : e.who === parentName || /^you$/i.test(e.who || "");
      const cls = isParent ? "from-parent" : "from-baby";
      const avatar = isParent
        ? '<span class="avatar-mini parent-avatar">' + escapeHtml((e.who || "?").charAt(0).toUpperCase()) + "</span>"
        : '<span class="avatar-mini baby-avatar">' + babyFace(/zzz|\*mumbles|\*snore/i.test(e.text) ? "closed" : /hehe|love|yay|\*giggl/i.test(e.text) ? "happy" : "open") + "</span>";
      const learnedTag = e.learned ? '<div class="learned-badge"><span class="badge-sparkle">✨</span> Learned fact</div>' : "";
      return (
        '<li class="chat-msg ' + cls + '"><div class="msg-bubble"><div class="msg-meta"><span class="msg-sender">' + avatar + " " +
        escapeHtml(e.who) + "</span>" + time + '</div><div class="msg-text">' + escapeHtml(e.text) + "</div>" + learnedTag + "</div></li>"
      );
    })
    .join("");

  if ($("voice-toggle")) $("voice-toggle").checked = !!state.voice;
  updateNapButton();

  const viewport = $("journal-viewport");
  if (viewport) {
    const nearBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < 160;
    if (nearBottom || !viewport.dataset.initScrolled) {
      viewport.dataset.initScrolled = "true";
      setTimeout(function () {
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: "smooth" });
      }, 30);
    }
  }
}

function updateNapButton() {
  const btn = $("nap-btn");
  if (!btn) return;
  btn.classList.toggle("is-active", !!state.asleep);
  btn.querySelector(".care-icon").textContent = state.asleep ? "☀️" : "💤";
  btn.querySelector(".care-label").textContent = state.asleep ? "wake" : "nap";
}

// ==========================================================================
// Neural brain (python app.py) with offline rule fallback
// ==========================================================================

const CARE_EVENTS = {
  feed: "*feeds you*",
  cuddle: "*cuddles you*",
  rock: "*rocks you gently*",
  play: "*plays peekaboo with you*",
  nap: "*tucks you in for a nap*",
  tickle: "*tickles you*",
  wake: "*wakes you up softly*",
};

const CARE_LOG = {
  feed: ["🍼", "fed"],
  cuddle: ["🧸", "cuddled"],
  rock: ["🌙", "rocked"],
  play: ["🙈", "played peekaboo with"],
  nap: ["💤", "tucked in"],
  tickle: ["😆", "tickled"],
  wake: ["☀️", "woke up"],
};

let brainOnline = false;

function checkBrain() {
  if (location.protocol === "file:" || !window.fetch) return;
  fetch("/api/status")
    .then(function (r) {
      return r.ok ? r.json() : null;
    })
    .then(function (d) {
      brainOnline = !!(d && d.status === "online");
      if (state && state.babyName) render();
    })
    .catch(function () {
      brainOnline = false;
    });
}

// Past turns in the same shape the model was trained on
function modelHistory() {
  const parentName = state.parentName;
  return (state.chat || [])
    .filter(function (e) {
      return e.who !== "nursery" || e.event;
    })
    .slice(-8)
    .map(function (e) {
      if (e.event) return { role: "user", content: CARE_EVENTS[e.event] };
      const isParent = e.role ? e.role === "user" : e.who === parentName;
      return { role: isParent ? "user" : "assistant", content: e.text };
    });
}

function factsForModel() {
  return (state.knowledge || []).map(function (k) {
    return k.text;
  });
}

function askBrain(payload) {
  if (!brainOnline) return Promise.resolve(null);
  const ctrl = window.AbortController ? new AbortController() : null;
  const timer = setTimeout(function () {
    if (ctrl) ctrl.abort();
  }, 15000);
  return fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: ctrl ? ctrl.signal : undefined,
  })
    .then(function (res) {
      return res.ok ? res.json() : null;
    })
    .then(function (d) {
      return d && d.reply ? d.reply.trim() : null;
    })
    .catch(function () {
      return null;
    })
    .finally(function () {
      clearTimeout(timer);
    });
}

// intent: "talk" | "idle" | a care event. animDone resolves when the care animation finishes.
function say(intent, heard, animDone) {
  animDone = animDone || Promise.resolve();
  const history = modelHistory();
  const memoryBefore = JSON.stringify(state.knowledge || []);
  const before = (state.knowledge || []).length;
  const ruleLine = speak(state, intent, heard); // also files new facts into memory
  const learned = state.knowledge.length > before;
  // When this message taught something, the rules already know the exact fact. A small model
  // re-typing it can garble words ("the sea is blue" -> "Quarch is a brown"), so echo it exactly.
  const filedFact = !!heard && JSON.stringify(state.knowledge) !== memoryBefore;
  const certain = ruleCertain;
  ruleCertain = false;

  if (heard) {
    pushChat(state.parentName || "you", heard, false, { role: "user" });
  } else if (CARE_LOG[intent]) {
    const log = CARE_LOG[intent];
    const text = intent === "wake" ? state.babyName + " woke up" : state.parentName + " " + log[1] + " " + state.babyName;
    pushChat("nursery", text, false, { event: intent, icon: log[0] });
  }
  save(state);
  render();

  if (intent === "idle" || !brainOnline || filedFact || certain) {
    animDone.then(function () {
      deliver(ruleLine, intent, learned);
    });
    return;
  }

  const reply = askBrain({
    message: heard || "",
    event: heard ? null : intent,
    babyName: state.babyName,
    parentName: state.parentName,
    needs: state.needs,
    facts: factsForModel(),
    history: history,
  });

  let thinking = null;
  if (heard) {
    Baby.set("listening");
    thinking = setTimeout(function () {
      showBubble("…", "thinking");
    }, 120);
  }
  Promise.all([reply, animDone]).then(function (r) {
    clearTimeout(thinking);
    if (Baby.action === "listening") Baby.rest();
    deliver(r[0] || ruleLine, intent, learned);
  });
}

function deliver(line, intent, learned) {
  if (state.awaiting && !/\bteach me|tell me|will you tell\b/i.test(line)) state.awaiting = null;
  showBubble(line);
  Baby.talk(line);
  if (intent === "play") maybeFirsts(state, "laugh", line);
  if (/\blove\b/i.test(line)) maybeFirsts(state, "love", line);
  pushChat(state.babyName || "baby", line, learned, { role: "assistant" });
  save(state);
  render();
  voiceSay(line);
  if (learned) {
    burst(["💡", "✨"], 3, { fy: 0.15 });
    const first = $("know-list").firstElementChild;
    if (first) first.classList.add("fresh");
  }
}

// ==========================================================================
// Care: interactive sequences
// ==========================================================================

let careBusy = false;

function nudge(key, amount) {
  state.needs[key] = clamp(state.needs[key] + amount);
}

function care(kind) {
  if (!state || !state.babyName || careBusy) return;
  const n = state.needs;
  const wasCrying = Baby.action === "crying";
  if (wasCrying && kind !== "nap" && kind !== "play") {
    stopReal(600);
    Baby.rest();
    sfx("sniffle");
  }

  if (state.asleep && kind !== "nap") {
    if (kind === "rock" || kind === "cuddle") {
      // soothing a sleeping baby is allowed — they sleep deeper
      nudge("comfort", 15);
      nudge("sleep", 6);
      $("crib").classList.add("rocking");
      setTimeout(function () {
        $("crib").classList.remove("rocking");
      }, 3300);
      burst(["💤", "♪"], 3, { fx: 0.62, fy: 0.15, cls: "fx-z", stagger: 300 });
      showBubble("mmm… zzz");
    } else {
      Baby.set("stirring", 900);
      showBubble(state.babyName + " is sleeping… shh 🤫");
    }
    save(state);
    renderNeeds();
    return;
  }

  if (kind === "feed") {
    if (n.hunger >= 92) {
      Baby.set("refusing", 1200);
      sfx("boop");
      say("feed", null, wait(1000));
      return;
    }
    careBusy = true;
    Baby.set("feeding");
    const iv = setInterval(function () {
      nudge("hunger", 1.3);
      if (Math.random() < 0.35) {
        const p = babyPoint(0.62, 0.5);
        fxAt(p[0], p[1], "🤍", { size: 9, cls: "fx-drop", spread: 24, dur: 0.9 });
      }
      if (Math.random() < 0.45) sfx("suck");
      renderNeeds();
    }, 120);
    const done = wait(3400)
      .then(function () {
        clearInterval(iv);
        Baby.set("burp");
        sfx("burp");
        const p = babyPoint(0.5, 0.45);
        fxAt(p[0] + 34, p[1], "*burp*", { cls: "fx-word", rise: 60, spread: 30, dur: 1.4 });
        nudge("bond", 4);
        return wait(700);
      })
      .then(function () {
        Baby.set("giggling", 1200);
        sfx("giggle");
        careBusy = false;
      });
    say("feed", null, done);
    return;
  }

  if (kind === "cuddle") {
    nudge("comfort", 34);
    nudge("bond", 12);
    Baby.set("cuddling", 2400);
    burst(["💗", "💕", "💖"], 9, { stagger: 120 });
    sfx("kiss");
    say("cuddle", null, wait(1300));
    return;
  }

  if (kind === "rock") {
    nudge("comfort", 22);
    nudge("sleep", 12);
    careBusy = true;
    Baby.set("rocking", 3300);
    $("crib").classList.add("rocking");
    burst(["♪", "♫"], 4, { fx: 0.5, fy: 0.1, stagger: 500, size: 18 });
    wait(3300).then(function () {
      $("crib").classList.remove("rocking");
      careBusy = false;
      if (state.needs.sleep < 45) enterSleep(); // rocked to sleep
    });
    say("rock", null, wait(1800));
    return;
  }

  if (kind === "play") {
    if (n.sleep < 15) {
      Baby.set("yawning", 1400);
      sfx("yawn");
      say("play", null, wait(1200));
      return;
    }
    nudge("curiosity", 36);
    nudge("sleep", -8);
    nudge("bond", 6);
    careBusy = true;
    Baby.set("peek-hide", 1200, function () {
      Baby.set("peek-boo", 1500);
      sfx("peek");
      setTimeout(function () {
        sfx("laugh");
      }, 150);
      const p = babyPoint(0.5, 0.25);
      fxAt(p[0], p[1] - 20, "boo!", { cls: "fx-word", rise: 70, spread: 10 });
      burst(["✨", "⭐", "🌟"], 6, { stagger: 70 });
      careBusy = false;
    });
    say("play", null, wait(1600));
    return;
  }

  if (kind === "nap") {
    if (state.asleep) wakeUp();
    else {
      enterSleep();
      say("nap", null, wait(700));
    }
  }
}

function enterSleep() {
  state.asleep = true;
  Baby.set("sleeping");
  sfx("lullaby");
  burst(["♪", "♫"], 4, { fy: 0.1, stagger: 280 });
  save(state);
  render();
}

function wakeUp() {
  state.asleep = false;
  save(state);
  render();
  if (state.needs.sleep < 40) {
    cry("woken"); // woken up too early: grumpy!
    return;
  }
  Baby.set("yawning", 1400);
  sfx("yawn");
  say("wake", null, wait(1300));
}

// ---------------------------------------------------------------- crying

const CRY_LINES = {
  hungry: ["Waaah! Hungwy! 😭", "*sob* Milk... waaah!", "Waaaah! My tummy! 😭"],
  lonely: ["Waaah! Hold me! 😭", "*sob* Waaah... where are you?", "Waaaah! Cuddle! 😭"],
  woken: ["Waaah! Too early! 😭", "*grumpy cry* Waaah!", "Waaah! I was sleeping! 😭"],
  tired: ["Waaah... so sleepy! 😭", "*cranky cry* Waaah!"],
};

function distress() {
  const n = state.needs;
  if (state.asleep) return null;
  if (n.hunger < 15) return "hungry";
  if (n.comfort < 15) return "lonely";
  if (n.sleep < 10) return "tired";
  return null;
}

function cry(reason) {
  Baby.set("crying", 3000);
  sfx("cry");
  showBubble(pick(CRY_LINES[reason] || CRY_LINES.lonely));
  burst("💧", 4, { fx: 0.36, fy: 0.42, cls: "fx-drop", size: 12, spread: 20, stagger: 250, dur: 1 });
  burst("💧", 4, { fx: 0.64, fy: 0.42, cls: "fx-drop", size: 12, spread: 20, stagger: 250, dur: 1 });
  if (state.cryingFor !== reason) {
    // one note per crying spell, not one per sob
    state.cryingFor = reason;
    const why = { hungry: "hungry", lonely: "wants a cuddle", woken: "woken too early", tired: "overtired" }[reason];
    pushChat("nursery", state.babyName + " is crying (" + (why || "fussy") + ")", false, { icon: "😭" });
    save(state);
    render();
  }
}

// Sleeping refills the sleep meter in real time and wakes them when rested
setInterval(function () {
  if (!state || !state.babyName || !state.asleep) return;
  nudge("sleep", 1.6);
  nudge("comfort", 0.3);
  if (Math.random() < 0.5) {
    const p = babyPoint(0.68, 0.18);
    fxAt(p[0], p[1], pick(["z", "Z", "z"]), { cls: "fx-z", size: 14 + Math.random() * 12, rise: 90, spread: 50, dur: 2.4 });
  }
  if (state.needs.sleep >= 100) wakeUp();
  else renderNeeds();
}, 1000);

// ==========================================================================
// Touch: tap = tickle, hold = cuddle, stroke = pat; eyes follow the pointer
// ==========================================================================

let holdTimer = null;
let holdIv = null;
let holding = false;
let holdStart = 0;
let tickles = [];
let strokeDist = 0;
let lastPt = null;

function tickle() {
  if (state.asleep) {
    Baby.set("stirring", 900);
    showBubble(pick(["mmh… 💤", "*snuffle*", "zzz… hehe… zzz"]));
    return;
  }
  if (careBusy) return;
  nudge("comfort", 2);
  nudge("bond", 1);
  nudge("curiosity", 2);
  Baby.set("giggling", 1100);
  sfx(tickles.length >= 2 ? "laugh" : "giggle");
  burst(["✨", "💛"], 2, { stagger: 80 });
  const now = Date.now();
  tickles = tickles.filter(function (t) {
    return now - t < 6000;
  });
  tickles.push(now);
  if (tickles.length >= 8) {
    tickles = [];
    nudge("comfort", -6);
    Baby.set("crying", 1300);
    sfx("whimper");
    showBubble(pick(["Too much! *whimper*", "*sniff* No more tickles..."]));
  } else if (tickles.length === 4) {
    Baby.set("giggling", 1600);
    sfx("laugh");
    say("tickle", null, wait(900));
  } else {
    showBubble(pick(["hehe!", "hihihi!", "eeee! 😆", "*giggles*", "again!"]));
  }
  renderNeeds();
}

function startHoldCuddle() {
  holding = true;
  holdStart = Date.now();
  if (!state.asleep) {
    Baby.set("cuddling");
    sfx("kiss");
  }
  holdIv = setInterval(function () {
    nudge("comfort", 1.2);
    nudge("bond", 0.35);
    burst(state.asleep ? "💤" : pick(["💗", "💕"]), 1, { jitter: 90 });
    renderNeeds();
  }, 230);
}

function endHoldCuddle() {
  clearInterval(holdIv);
  holding = false;
  if (state.asleep) return;
  if (Date.now() - holdStart > 1200) say("cuddle", null, wait(200));
  else Baby.rest();
  save(state);
}

function petHead() {
  nudge("comfort", 1.5);
  nudge("bond", 0.5);
  sfx("pat");
  burst(state.asleep ? "💤" : "💗", 1, { fy: 0.08, jitter: 40, size: 16 });
  if (!state.asleep && !Baby.busy() && Math.random() < 0.3) showBubble(pick(["mmm~", "hehe", "*happy wiggle*", "more pats!"]));
  renderNeeds();
}

function bindBabyTouch() {
  const baby = $("baby");
  baby.addEventListener("pointerdown", function (e) {
    if (!state || !state.babyName) return;
    if (baby.setPointerCapture) baby.setPointerCapture(e.pointerId);
    holding = false;
    holdTimer = setTimeout(startHoldCuddle, 420);
  });
  baby.addEventListener("pointerup", function () {
    clearTimeout(holdTimer);
    if (holding) endHoldCuddle();
    else if (state && state.babyName) tickle();
  });
  baby.addEventListener("pointercancel", function () {
    clearTimeout(holdTimer);
    if (holding) endHoldCuddle();
  });
  baby.addEventListener("pointermove", function (e) {
    if (e.buttons || !state || !state.babyName) return;
    const b = baby.getBoundingClientRect();
    const onHead = e.clientY < b.top + b.height * 0.55;
    if (lastPt && onHead) {
      strokeDist += Math.hypot(e.clientX - lastPt[0], e.clientY - lastPt[1]);
      if (strokeDist > 280) {
        strokeDist = 0;
        petHead();
      }
    }
    lastPt = [e.clientX, e.clientY];
  });
  baby.addEventListener("pointerleave", function () {
    lastPt = null;
    strokeDist = 0;
  });
  baby.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      tickle();
    }
  });

  document.addEventListener("pointermove", function (e) {
    if (!state || state.asleep) return;
    const b = baby.getBoundingClientRect();
    const dx = e.clientX - (b.left + b.width / 2);
    const dy = e.clientY - (b.top + b.height * 0.4);
    const d = Math.hypot(dx, dy) || 1;
    const k = Math.min(1, d / 260);
    lookAt((dx / d) * 4.5 * k, (dy / d) * 3.5 * k);
  });
}

// ==========================================================================
// Time passing & idle life
// ==========================================================================

function tick() {
  if (!state || !state.babyName) return;
  const now = Date.now();
  const grown = ((now - state.lastTick) / 3600000) * 14;
  state.ageHours += grown;
  state.lastTick = now;
  const n = state.needs;
  n.hunger = clamp(n.hunger - grown * (state.asleep ? 3 : 6));
  if (!state.asleep) n.sleep = clamp(n.sleep - grown * 4);
  n.comfort = clamp(n.comfort - grown * 3);
  n.curiosity = clamp(n.curiosity - grown * 2);
  save(state);
  renderNeeds();
}

function idleLife() {
  if (!state || !state.babyName || careBusy || holding) return;
  const upset = distress();
  if (!upset) state.cryingFor = null; // calmed down
  if (upset && Baby.action !== "crying" && Math.random() < 0.8) {
    cry(upset);
    return;
  }
  if (Baby.busy()) return;
  if (state.asleep) {
    if (Math.random() < 0.08) {
      const k = state.knowledge.length && Math.random() < 0.3 ? pick(state.knowledge) : null;
      const word = k ? contentWords(k.text)[0] : "";
      const dream = word ? "mmm… " + word + "… zzz" : pick(["zzz… *smiles in sleep*", "mmm… milk… zzz", "*tiny snore*"]);
      showBubble(dream);
      pushChat(state.babyName, dream, false, { role: "assistant" });
      save(state);
      render();
    }
    return;
  }
  const mood = moodOf(state.needs);
  const r = Math.random();
  if (mood === "sleepy" && r < 0.5) {
    Baby.set("yawning", 1400);
    sfx("yawn");
  } else if (mood === "hungry" && r < 0.3) {
    const p = babyPoint(0.5, 0.82);
    fxAt(p[0], p[1], "*rumble*", { cls: "fx-word", rise: 40, spread: 30 });
  } else if (r < 0.35) {
    // looks around the room
    lookAt((Math.random() - 0.5) * 9, (Math.random() - 0.5) * 6);
  } else if (r < 0.45) {
    burst("✨", 1, { fy: 0.1 });
  } else if (r < 0.5 && !window.speechSynthesis.speaking) {
    say("idle");
  }
}

// ==========================================================================
// Closet: dress the baby (onesie color, pattern, hat, extras)
// ==========================================================================

const ICON_PACIFIER =
  '<svg viewBox="0 0 30 30"><ellipse cx="15" cy="12" rx="11" ry="7" fill="#a5b4fc"/><circle cx="15" cy="12" r="3" fill="#818cf8"/>' +
  '<circle cx="15" cy="21" r="5.5" fill="none" stroke="#818cf8" stroke-width="2.6"/></svg>';
const ICON_BOWTIE =
  '<svg viewBox="0 0 30 30"><path d="M15 15 L3 8 Q1 15 3 22 Z M15 15 L27 8 Q29 15 27 22 Z" fill="#8b5cf6"/><circle cx="15" cy="15" r="3.6" fill="#7c3aed"/></svg>';

const WARDROBE = {
  onesie: [
    { id: "sky", label: "sky", colors: ["#c7e6fa", "#9fcbeb"] },
    { id: "rose", label: "rose", colors: ["#fbd5e0", "#f4a5bd"] },
    { id: "mint", label: "mint", colors: ["#c9f2df", "#93dbb8"] },
    { id: "lemon", label: "lemon", colors: ["#fff1b8", "#fbd56a"] },
    { id: "lilac", label: "lilac", colors: ["#e4dafc", "#bba8f2"] },
    { id: "peach", label: "peach", colors: ["#ffe0c7", "#fdb98a"] },
    { id: "cloud", label: "cloud", colors: ["#ffffff", "#e2e8f0"] },
  ],
  pattern: [
    { id: "none", label: "plain", icon: "⬜" },
    { id: "star", label: "star", icon: "⭐" },
    { id: "hearts", label: "hearts", icon: "💗" },
    { id: "stripes", label: "stripes", icon: "〰️" },
    { id: "dots", label: "dots", icon: "⚪" },
    { id: "rainbow", label: "rainbow", icon: "🌈" },
  ],
  hat: [
    { id: "none", label: "no hat", icon: "🙂" },
    { id: "beanie", label: "beanie", icon: "🧶" },
    { id: "bunny", label: "bunny ears", icon: "🐰" },
    { id: "bear", label: "bear ears", icon: "🐻" },
    { id: "bow", label: "big bow", icon: "🎀" },
    { id: "crown", label: "crown", icon: "👑" },
    { id: "sunhat", label: "sun hat", icon: "👒" },
  ],
  acc: [
    { id: "none", label: "nothing", icon: "🙂" },
    { id: "pacifier", label: "binky", svg: ICON_PACIFIER },
    { id: "glasses", label: "glasses", icon: "👓" },
    { id: "bowtie", label: "bow tie", svg: ICON_BOWTIE },
    { id: "scarf", label: "scarf", icon: "🧣" },
  ],
};

const DEFAULT_OUTFIT = { onesie: "sky", pattern: "star", hat: "none", acc: "none" };

const DRESS_LINES = {
  beanie: ["So warm and cozy! *wiggles*", "My head is toasty!"],
  bunny: ["Hop hop! I'm a bunny!", "Look, I have long ears! Hehe!"],
  bear: ["Rawr! I'm a little bear!", "Grr! A tiny cuddly bear!"],
  bow: ["Ooh, a pretty bow! Am I cute?", "Bow bow! Hehe!"],
  crown: ["I'm royal baby! Hehe!", "A crown! Bow to me! *giggles*"],
  sunhat: ["Sunny day hat! Can we go outside?", "I'm ready for the beach!"],
  pacifier: ["*suck suck* Mmm, binky...", "*happy binky noises*"],
  glasses: ["Now I can see everything! Ooh!", "I look so smart! Hehe."],
  bowtie: ["I'm so fancy!", "Party time! *wiggles*"],
  scarf: ["Snuggly scarf! So warm!", "Ooh, it's soft!"],
  none: ["Ahh, free!", "Hehe, that tickles!"],
};

let closetTab = "onesie";
let closetChanged = false;

function outfit() {
  state.outfit = Object.assign({}, DEFAULT_OUTFIT, state.outfit || {});
  return state.outfit;
}

function wardrobeItem(slot, id) {
  return WARDROBE[slot].find(function (i) {
    return i.id === id;
  }) || WARDROBE[slot][0];
}

function applyOutfit() {
  const o = outfit();
  const baby = $("baby");
  baby.dataset.hat = o.hat;
  baby.dataset.acc = o.acc;
  baby.dataset.pattern = o.pattern;
  const colors = wardrobeItem("onesie", o.onesie).colors;
  document.querySelectorAll("#g-onesie stop").forEach(function (stop, i) {
    stop.setAttribute("stop-color", colors[i] || colors[0]);
  });
}

function renderShelf() {
  const o = outfit();
  $("closet-title").textContent = (state.babyName || "Baby") + "'s closet";
  document.querySelectorAll(".closet-tab").forEach(function (t) {
    const on = t.dataset.tab === closetTab;
    t.classList.toggle("is-active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
  let html = WARDROBE[closetTab]
    .map(function (item) {
      const visual = item.colors
        ? '<span class="swatch" style="background:linear-gradient(160deg,' + item.colors[0] + "," + item.colors[1] + ')"></span>'
        : '<span class="item-icon">' + (item.svg || item.icon) + "</span>";
      return (
        '<button type="button" class="closet-item' + (o[closetTab] === item.id ? " is-on" : "") +
        '" data-slot="' + closetTab + '" data-id="' + item.id + '" aria-pressed="' + (o[closetTab] === item.id) + '">' +
        visual + "<span>" + item.label + "</span></button>"
      );
    })
    .join("");
  if (closetTab === "hat" && state.asleep) html += '<p class="closet-hint">💤 Hats show when ' + escapeHtml(state.babyName) + " wakes up (the nightcap wins at bedtime).</p>";
  $("closet-shelf").innerHTML = html;
}

function dress(slot, id, quiet) {
  const o = outfit();
  if (o[slot] === id) return;
  o[slot] = id;
  closetChanged = true;
  applyOutfit();
  save(state);
  renderShelf();
  if (quiet) return;
  if (state.asleep) {
    showBubble("zzz… *dressed in their sleep*");
    return;
  }
  Baby.set("dressing", 800);
  burst(["✨", "💫", "🌟"], 5, { stagger: 60, fy: 0.25 });
  nudge("bond", 2);
  nudge("curiosity", 3);
  const line = pick(
    DRESS_LINES[id] ||
      (slot === "onesie"
        ? ["Ooh, " + wardrobeItem("onesie", id).label + "! My favorite color now!", "New onesie! Do I look cute?"]
        : ["Yay! New clothes!", "Ooh! Do I look cute?"])
  );
  showBubble(line);
  Baby.talk(line);
  voiceSay(line);
  renderNeeds();
}

function surpriseOutfit() {
  ["onesie", "pattern", "hat", "acc"].forEach(function (slot) {
    dress(slot, pick(WARDROBE[slot]).id, true);
  });
  if (!state.asleep) {
    Baby.set("dressing", 800);
    burst(["✨", "🎀", "🌟", "💫"], 8, { stagger: 50, fy: 0.25 });
    const line = pick(["Ta-da! A whole new me!", "Surprise! How do I look?", "Ooh! Fancy baby! Hehe!"]);
    showBubble(line);
    voiceSay(line);
  }
}

function toggleCloset(open) {
  const closet = $("closet");
  const willOpen = open == null ? closet.hidden : open;
  closet.hidden = !willOpen;
  $("closet-btn").classList.toggle("is-active", willOpen);
  $("closet-btn").setAttribute("aria-expanded", willOpen ? "true" : "false");
  if (willOpen) {
    closetChanged = false;
    renderShelf();
    if (!state.asleep && !Baby.busy()) showBubble(pick(["Ooh, dress-up time!", "What should I wear?", "Closet! Closet! *claps*"]));
  } else if (closetChanged) {
    pushChat("nursery", state.parentName + " dressed " + state.babyName + " up", false, { icon: "👗" });
    save(state);
    render();
  }
}

function bindCloset() {
  $("closet-btn").addEventListener("click", function () {
    if (state && state.babyName) toggleCloset();
  });
  $("closet-close").addEventListener("click", function () {
    toggleCloset(false);
  });
  $("closet-random").addEventListener("click", surpriseOutfit);
  document.querySelectorAll(".closet-tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      closetTab = tab.dataset.tab;
      renderShelf();
    });
  });
  $("closet-shelf").addEventListener("click", function (e) {
    const btn = e.target.closest(".closet-item");
    if (btn) dress(btn.dataset.slot, btn.dataset.id);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !$("closet").hidden) toggleCloset(false);
  });
}

// ==========================================================================
// Setup
// ==========================================================================

function seedSelf(st) {
  remember(st, st.babyName + " is a baby who learns from " + st.parentName);
  remember(st, st.parentName + " is my parent and teacher");
  remember(st, "Love is something we keep");
}

function beginRaising() {
  const babyName = $("name-input").value.trim();
  const parentName = $("parent-input").value.trim();
  if (!babyName || !parentName) return;
  state = blankState();
  state.babyName = babyName;
  state.parentName = parentName;
  hear(state, parentName + " " + babyName + " love you");
  seedSelf(state);
  pushChat("nursery", babyName + " opened their eyes. " + parentName + " was the first teacher.", false, { icon: "🌅" });
  save(state);
  $("welcome").hidden = true;
  applyOutfit();
  render();
  care("cuddle");
}

document.querySelectorAll("[data-care]").forEach(function (btn) {
  btn.addEventListener("click", function () {
    care(btn.getAttribute("data-care"));
  });
});

$("talk-form").addEventListener("submit", function (e) {
  e.preventDefault();
  const input = $("speech");
  const text = input.value.trim();
  if (!text || !state || !state.babyName) return;
  input.value = "";
  if (state.asleep) {
    if (/\bwake\b/i.test(text)) {
      pushChat(state.parentName, text, false, { role: "user" });
      wakeUp();
      return;
    }
    pushChat(state.parentName, text, false, { role: "user" });
    const mumble = pick(["*mumbles* mmm… " + (contentWords(text)[0] || "") + "… zzz", "zzz… *smiles in sleep*", "*snore* …hehe… zzz"]);
    pushChat(state.babyName, mumble, false, { role: "assistant" });
    Baby.set("stirring", 900);
    showBubble(mumble);
    save(state);
    render();
    return;
  }
  hear(state, text);
  nudge("curiosity", 12);
  nudge("bond", 5);
  say("talk", text);
});

$("start-btn").addEventListener("click", beginRaising);
$("welcome-form").addEventListener("submit", function (e) {
  e.preventDefault();
  beginRaising();
});

$("voice-toggle").addEventListener("change", function () {
  if (!state) return;
  state.voice = $("voice-toggle").checked;
  save(state);
  if (state.voice) voiceSay("Hehe! Hi " + (state.parentName || "") + "! I love you!");
});

document.querySelectorAll(".starter-chip").forEach(function (btn) {
  btn.addEventListener("click", function () {
    const phrase = btn.getAttribute("data-phrase") || btn.textContent.trim();
    if (phrase) {
      $("speech").value = phrase;
      $("talk-form").dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
  });
});

document.querySelectorAll(".topic-chip").forEach(function (btn) {
  btn.addEventListener("click", function () {
    const topic = btn.getAttribute("data-topic");
    if (topic) {
      $("speech").value = topic;
      $("speech").focus();
      btn.classList.add("tapped");
      setTimeout(function () {
        btn.classList.remove("tapped");
      }, 300);
    }
  });
});

// ---------------------------------------------------------------- backup: save / load the baby as a file

function exportBaby() {
  if (!state || !state.babyName) return;
  save(state);
  const blob = new Blob([JSON.stringify(state, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = state.babyName.replace(/[^A-Za-z0-9_-]/g, "") + ".baby.json";
  document.body.appendChild(a);
  a.click();
  setTimeout(function () {
    URL.revokeObjectURL(a.href);
    a.remove();
  }, 500);
  showBubble("I'm saved! 💾 Keep me safe!");
}

function importBaby(file) {
  if (!file || file.size > 5000000) return;
  const reader = new FileReader();
  reader.onload = function () {
    let loaded;
    try {
      loaded = JSON.parse(reader.result);
    } catch (err) {
      window.alert("That file isn't a saved baby.");
      return;
    }
    if (!loaded || typeof loaded !== "object" || !loaded.babyName || !Array.isArray(loaded.knowledge || [])) {
      window.alert("That file isn't a saved baby.");
      return;
    }
    if (state && state.babyName && state.babyName !== loaded.babyName &&
        !window.confirm("Replace " + state.babyName + " with " + loaded.babyName + "? Save " + state.babyName + " first if you want to keep them.")) {
      return;
    }
    state = Object.assign(blankState(), loaded);
    state.lastTick = Date.now();
    sanitizeKnowledge(state);
    save(state);
    $("welcome").hidden = true;
    applyOutfit();
    Baby.rest();
    render();
    showBubble("Hi " + (state.parentName || "") + "! I'm home! 💛");
  };
  reader.readAsText(file);
}

$("export-btn").addEventListener("click", exportBaby);
$("import-btn").addEventListener("click", function () {
  $("import-file").click();
});
$("welcome-import").addEventListener("click", function () {
  $("import-file").click();
});
$("import-file").addEventListener("change", function () {
  importBaby(this.files[0]);
  this.value = "";
});

if ($("clear-chat-btn")) {
  $("clear-chat-btn").addEventListener("click", function () {
    if (!state) return;
    if (window.confirm("Clear conversation history for " + (state.babyName || "baby") + "?")) {
      state.chat = [];
      save(state);
      render();
    }
  });
}

const scrollBottomBtn = $("scroll-bottom-btn");
const journalViewport = $("journal-viewport");
if (scrollBottomBtn && journalViewport) {
  scrollBottomBtn.addEventListener("click", function () {
    journalViewport.scrollTo({ top: journalViewport.scrollHeight, behavior: "smooth" });
    scrollBottomBtn.hidden = true;
  });
  journalViewport.addEventListener("scroll", function () {
    const dist = journalViewport.scrollHeight - journalViewport.scrollTop - journalViewport.clientHeight;
    scrollBottomBtn.hidden = !(dist > 140 && (state.chat || []).length > 2);
  });
}

if (window.speechSynthesis) {
  chosenVoice = pickVoice();
  window.speechSynthesis.onvoiceschanged = function () {
    chosenVoice = pickVoice();
  };
}

bindBabyTouch();
bindCloset();
checkBrain();
syncFromServer();

if (!state || !state.babyName) {
  state = state || blankState();
  $("welcome").hidden = false;
  Baby.apply();
} else {
  if (!state.knowledge) state.knowledge = [];
  if (!state.chat) state.chat = [];
  if (!state.knowledge.length) seedSelf(state);
  $("welcome").hidden = true;
  applyOutfit();
  Baby.rest();
  render();
  showBubble(state.asleep ? "zzz…" : "Hi " + (state.parentName || "") + "! 💛");
}

setInterval(tick, 8000);
setInterval(idleLife, 7000);
tick();

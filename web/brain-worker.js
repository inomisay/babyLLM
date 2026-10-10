// The baby's brain, running inside the visitor's browser (a Web Worker, so the nursery never freezes).
// A faithful port of chat.py: memory retrieval, the prompt format the model was trained on,
// KV-cached sampling, and every honesty check. Nothing the visitor says leaves their device.
/* global ort, BabyTokenizer */

const ORT_VERSION = "1.30.0";
importScripts(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.wasm.min.js`, "tokenizer.js");
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
ort.env.wasm.numThreads = 1; // static hosts can't enable the isolation that threads need

let session = null;
let tok = null;
let meta = null;
let knownWords = null;
let templateWords = null;
let templateStems = null;

// ----------------------------------------------------------------------------- constants (as in chat.py)

const CARE_EVENTS = {
  feed: "*feeds you*",
  cuddle: "*cuddles you*",
  rock: "*rocks you gently*",
  play: "*plays peekaboo with you*",
  nap: "*tucks you in for a nap*",
  tickle: "*tickles you*",
  wake: "*wakes you up softly*",
};
const STOPWORDS = new Set(
  "the a an and or to of in is are you i it we my your be do does did was for on at that this with what who where how why can me am".split(" ")
);
const MATCH_STOPWORDS = new Set([...STOPWORDS, "have", "has", "had", "know", "tell", "about", "please", "yes", "no", "they", "them",
  "there", "very", "really", "remember", "teach", "did"]);
const FRAME_WORDS = new Set(["ooh", "yes", "yay", "hehe", "remember", "so", "because", "and", "wow", "okay", "yeah", "too", "now", "know"]);
const SHORT_WORDS = new Set(["i", "a", "an", "oh", "ok", "hi", "no", "so", "to", "me", "my", "we", "be", "is", "it", "in", "on", "at",
  "up", "us", "do", "go", "he", "of", "or", "by", "am", "as", "if", "aw", "ah", "mm", "s", "t", "m", "ll", "re", "ve", "d"]);
const VERBISH = ["dream", "dreamed", "dreamt", "dreaming", "had", "saw", "see", "seen", "want", "wanted", "play", "playing",
  "like", "liked", "really", "about", "with", "window", "said", "say", "told"];
const PERSONAL = /\b(i|i'm|my|me|mine|you|your|you're|we|our)\b/;
const QUOTE = /^\s*(?:you said|did you say|you told me|didn't you say)(?: that)?(?: you)?\s+(.+)/i;
const GREETING = /^\s*(hi|hello|hey|hiya|good morning|morning|yo)\b/i;
const BROKEN_GRAMMAR = /\b(a|an|the)\s+(we|you|i|he|she|they|it|very|the|a|an|and|is|are|to|so|too|my|your)\b/i;
const REPEATED_AND = /\b(\w{3,})\s+and\s+\1\b/i; // "friendly and friendly"
const TEACH_ACK = /learned something new|i will remember|i'll remember|keep it in my memory|in my memory now|thank you for teaching|thanks for teaching/i;
const BLOCKED = /\b(fuck\w*|shit\w*|bitch\w*|cunt\w*|dicks?|pussy|whore\w*|slut\w*|bastard\w*|nigg\w*|fag\w*|retard\w*|rape\w*|porn\w*|sex\w*|nazi\w*|kill yourself|kys)\b/i;
const KIND_REPLY = "Hmm, that's not a nice word. Let's use kind words! 💛";

const pick = (list) => list[Math.floor(Math.random() * list.length)];
const letters = (text) => String(text).match(/[A-Za-z]+/g) || [];

// ----------------------------------------------------------------------------- words & retrieval

function stem(w) {
  if (w.length > 4 && w.endsWith("ies")) return w.slice(0, -3) + "y";
  if (w.length > 4 && /(ches|shes|xes|sses)$/.test(w)) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

function words(text) {
  const out = new Set();
  for (const w of String(text).toLowerCase().match(/[a-z']+/g) || []) {
    if (!MATCH_STOPWORDS.has(w) && w.length > 1) out.add(stem(w));
  }
  return out;
}

const overlap = (a, b) => [...a].filter((x) => b.has(x)).length;
const union = (sets) => sets.reduce((acc, s) => (s.forEach((x) => acc.add(x)), acc), new Set());

function selectFacts(facts, query, k, focus) {
  const q = words(focus || query);
  const scored = facts.map((f, i) => [overlap(words(f), q), i, f]).sort((a, b) => b[0] - a[0] || b[1] - a[1]);
  const hits = scored.filter((t) => t[0] > 0).map((t) => t[2]).slice(0, 4);
  if (hits.length) {
    const linked = union(hits.map(words));
    q.forEach((w) => linked.delete(w));
    for (let i = facts.length - 1; i >= 0 && hits.length < 6; i--) {
      if (!hits.includes(facts[i]) && overlap(words(facts[i]), linked)) hits.push(facts[i]);
    }
    return hits.sort((a, b) => facts.indexOf(a) - facts.indexOf(b));
  }
  if (focus && query !== focus) return selectFacts(facts, query, k, "");
  return facts.slice(-k);
}

function moodFromNeeds(needs) {
  if (!needs) return [];
  const m = [];
  if ((needs.hunger ?? 100) < 35) m.push("hungry");
  if ((needs.sleep ?? 100) < 35) m.push("sleepy");
  if ((needs.comfort ?? 100) < 35) m.push("fussy");
  if ((needs.curiosity ?? 100) < 30) m.push("bored");
  return m;
}

function systemPrompt(baby, parent, mood, facts) {
  const lines = [`You are ${baby}, a baby. ${parent} is your parent.`, `Mood: ${mood.length ? mood.join(", ") : "happy"}.`];
  if (facts.length) {
    lines.push("Memory:");
    facts.forEach((f) => lines.push(`- ${f.replace(/[.!]+$/, "")}.`));
  } else {
    lines.push("Memory: nothing yet.");
  }
  return lines.join("\n");
}

function chatml(messages) {
  return messages.map((m) => `<|im_start|>${m.role}\n${m.content.trim()}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
}

function buildPrompt(baby, parent, needs, facts, history, userTurn, budget) {
  const query = userTurn + " " + history.slice(-2).map((m) => m.content).join(" ");
  let k = 8;
  let turns = history.slice();
  for (;;) {
    const sys = systemPrompt(baby, parent, moodFromNeeds(needs), selectFacts(facts, query, k, userTurn));
    const ids = tok.encode(chatml([{ role: "system", content: sys }, ...turns, { role: "user", content: userTurn }]));
    if (ids.length <= budget) return ids;
    if (turns.length >= 2) turns = turns.slice(2);
    else if (k > 2) k -= 2;
    else return ids.slice(-budget);
  }
}

// ----------------------------------------------------------------------------- honesty checks

function ungroundedWords(reply, context) {
  const seen = new Set(letters(context).map((w) => w.toLowerCase()));
  return letters(reply).filter((w) => w.length >= 4 && !seen.has(w.toLowerCase()) && !templateWords.has(w.toLowerCase()) &&
    !knownWords.has(w.toLowerCase()));
}

function offScriptWords(reply, context) {
  const allowed = new Set(letters(context).map((w) => stem(w.toLowerCase())));
  return letters(reply).filter((w) => w.length >= 3 && !MATCH_STOPWORDS.has(w.toLowerCase()) && !allowed.has(stem(w.toLowerCase())) &&
    !templateStems.has(stem(w.toLowerCase())) && !FRAME_WORDS.has(stem(w.toLowerCase())));
}

// difflib.SequenceMatcher(None, a, b).ratio(), as used by get_close_matches
function similarity(a, b) {
  const matches = (alo, ahi, blo, bhi) => {
    let best = 0, bi = 0, bj = 0;
    for (let i = alo; i < ahi; i++) {
      for (let j = blo; j < bhi; j++) {
        let n = 0;
        while (i + n < ahi && j + n < bhi && a[i + n] === b[j + n]) n++;
        if (n > best) { best = n; bi = i; bj = j; }
      }
    }
    if (!best) return 0;
    return best + matches(alo, bi, blo, bj) + matches(bi + best, ahi, bj + best, bhi);
  };
  return (2 * matches(0, a.length, 0, b.length)) / (a.length + b.length || 1);
}

function repairNames(reply, context, bad) {
  const ctx = new Map(letters(context).filter((w) => w.length >= 3).map((w) => [w.toLowerCase(), w]));
  for (const w of new Set(bad)) {
    let best = null, bestScore = 0.7;
    for (const [low] of ctx) {
      const s = similarity(w.toLowerCase(), low);
      if (s >= bestScore) { best = low; bestScore = s; }
    }
    if (!best) return null;
    const fixed = ctx.get(best);
    reply = reply.replace(new RegExp(`\\b${w}\\b`, "g"), /[A-Z]/.test(w[0]) ? fixed[0].toUpperCase() + fixed.slice(1) : fixed);
  }
  return reply;
}

function contradictsMood(reply, userTurn, mood) {
  if (userTurn === CARE_EVENTS.feed && mood.includes("hungry") && /not hungry|i'm full|no more milk|tiny sip/i.test(reply)) return true;
  if (userTurn === CARE_EVENTS.nap && mood.includes("sleepy") && /not sleepy|five more minutes/i.test(reply)) return true;
  return userTurn === CARE_EVENTS.feed && !mood.includes("hungry") && /tummy was so empty|yay, food/i.test(reply);
}

const isQuestion = (t) => /\?\s*$/.test(t.trim()) || /^(what|who|where|when|why|how|can|is|are|do|does|did|tell me)\b/.test(t.trim().toLowerCase());
const ownWords = (history) => history.slice(-6).filter((m) => m.role === "assistant").map((m) => m.content);

function quoteCheck(reply, userTurn, history) {
  const m = userTurn.trim().replace(/[?.!]+$/, "").match(QUOTE);
  const said = ownWords(history);
  if (!m || !said.length) return reply;
  const verbStems = new Set(VERBISH.map(stem));
  const quoted = new Set([...words(m[1])].filter((w) => !verbStems.has(w)));
  if (!quoted.size) return reply;
  const spoken = words(said.join(" "));
  if ([...quoted].every((w) => spoken.has(w))) {
    return /^\s*yes\b/i.test(reply) ? reply : "Yes! " + reply.replace(/^\s*(no|yes)\b[!.,]*\s*/i, "");
  }
  const target = union([words(userTurn), quoted]);
  let original = said.reduce((best, s) => (overlap(words(s), target) > overlap(words(best), target) ? s : best), said[0]);
  original = original.replace(/^(\*[^*]*\*\s*|hi!\s*|guess what\?\s*|ooh!\s*)+/i, "").trim();
  if (!original) return "No, I didn't say that!";
  if (!/^I\b/.test(original)) original = original[0].toLowerCase() + original.slice(1);
  return "No! I said: " + original;
}

function answerToMyQuestion(userTurn, history, parent) {
  const said = ownWords(history);
  if (!said.length || !said[said.length - 1].trim().endsWith("?")) return null;
  const t = userTurn.trim().toLowerCase();
  if (/^(yes|yeah|yep|sure|okay|ok|of course|let's|alright)\b/.test(t)) return pick(["Yay! *claps*", `Yay! Thank you, ${parent}!`, "Hehe! Yay!"]);
  if (/^(no|nope|not now|later|maybe later)\b/.test(t)) return pick(["Aww. Okay.", "Okay... later then. *pouts*"]);
  return null;
}

function identityAnswer(userTurn, baby, parent) {
  const t = userTurn.toLowerCase();
  if (/\b(what'?s your name|what is your name|who are you|your name)\b/.test(t)) return pick([`My name is ${baby}! Hehe.`, `I'm ${baby}!`, `${baby}! That's me!`]);
  if (/\b(who am i|what'?s my name|what is my name|do you know my name)\b/.test(t)) return pick([`You're ${parent}!`, `You are ${parent}, my parent!`, `${parent}! I know you!`]);
  return null;
}

function topicOf(userTurn) {
  const ws = (userTurn.match(/[A-Za-z']+/g) || []).filter((w) => !MATCH_STOPWORDS.has(w.toLowerCase()) && w.length > 2);
  return ws.length ? ws[ws.length - 1] : "that";
}

function memoryHits(facts, userTurn) {
  const q = words(userTurn);
  return q.size ? facts.filter((f) => overlap(words(f), q)) : [];
}

function worldQuestionTopic(userTurn) {
  const t = userTurn.trim().toLowerCase();
  const stripped = t.replace(/^(do you know( about| what)?|tell me about|what do you know about)\s+/, "");
  const asks = stripped !== t || /^(what|who|where)\s+(is|are|does|do|was|were)\b/.test(t);
  if (!asks || PERSONAL.test(stripped)) return null;
  return topicOf(userTurn);
}

function factCheck(reply, hits, userTurn) {
  if (!hits.length) return reply;
  if (/don'?t know|not sure/i.test(reply)) reply = "";
  const rw = words(reply);
  const allowed = union([...hits.map(words), words(userTurn), FRAME_WORDS]);
  const best = hits.reduce((b, f) => (overlap(words(f), rw) > overlap(words(b), rw) ? f : b), hits[0]);
  const fw = words(best);
  const coverage = overlap(fw, rw) / Math.max(1, fw.size);
  if (coverage >= 0.99 && [...rw].every((w) => allowed.has(w))) return reply;
  const fact = best.replace(/[.!]+$/, "") + "!";
  if (/^\s*(can|is|are|do|does|did)\b/.test(userTurn.toLowerCase()) && [...words(userTurn)].every((w) => fw.has(w) || MATCH_STOPWORDS.has(w))) {
    return "Yes! " + fact;
  }
  return pick(["Ooh! ", "I remember! ", ""]) + fact;
}

function careLine(event, mood, parent) {
  const hungry = mood.includes("hungry"), sleepy = mood.includes("sleepy"), fussy = mood.includes("fussy");
  if (event === "feed") return hungry ? pick([`Mmm, milk! Yummy yummy. Thank you, ${parent}!`, "*gulp gulp* Mmm! My tummy was so empty!", "Yay, food! *happy wiggle*"])
    : pick(["Hehe, I'm full! No more milk, please.", "*turns head* I'm not hungry, silly!", "Just a tiny sip. My tummy is happy!"]);
  if (event === "cuddle") return fussy ? pick([`*sniff* ...better now. Thank you, ${parent}.`, "Mmm, I needed a hug. *sniffle*"])
    : pick([`*snuggles* I feel so safe with you, ${parent}.`, "Warm and squishy! I love cuddles!", `Hehe! I love you, ${parent}!`, "*nuzzles* Don't let go, okay?"]);
  if (event === "rock") return pick(["Swing... swing... so cozy.", "*yawns* That feels nice, rocky rocky.", "Wheee... softly... I like this."]);
  if (event === "play") return sleepy ? pick(["*yawn* Peek... a... boo. I'm so sleepy.", "Hehe... but my eyes are heavy."])
    : pick(["Peekaboo! Hehe! Again, again!", "*giggles* Where did you go? There you are!", "Haha! You found me!"]);
  if (event === "nap") return sleepy ? pick([`*yawns* Night night, ${parent}...`, "Mmm... cozy blanket... zzz.", "Sing me a song? ...zzz."])
    : pick(["But I'm not sleepy! ...okay, maybe a little. *yawn*", "Just five more minutes of play? ...zzz."]);
  if (event === "tickle") return pick(["Hahaha! Stop! No, don't stop!", "*squeals* That tickles!", "Hehehe! My tummy!"]);
  return pick([`*stretches* Good morning, ${parent}!`, "*blinks* Hi! I had a dream about stars.", "Mmm... I'm awake! Hehe."]);
}

// ----------------------------------------------------------------------------- the model

function emptyPast() {
  const feeds = {};
  for (const name of meta.past_names) {
    feeds[name] = new ort.Tensor("float32", new Float32Array(0), [1, meta.n_head, 0, meta.head_dim]);
  }
  return feeds;
}

function sampleToken(logits, generated, temperature) {
  const V = logits.length;
  const x = Float32Array.from(logits);
  for (const t of new Set(generated)) x[t] = x[t] > 0 ? x[t] / 1.15 : x[t] * 1.15; // repetition penalty
  if (temperature <= 0) return x.indexOf(Math.max(...x));
  for (let i = 0; i < V; i++) x[i] /= temperature;
  const order = Array.from(x.keys()).sort((a, b) => x[b] - x[a]).slice(0, 30); // top-k 30
  const maxLogit = x[order[0]];
  let probs = order.map((i) => Math.exp(x[i] - maxLogit));
  let total = probs.reduce((s, p) => s + p, 0);
  probs = probs.map((p) => p / total);
  let cum = 0, cut = order.length; // top-p 0.9 (always keep the best token)
  for (let i = 0; i < order.length; i++) {
    cum += probs[i];
    if (cum > 0.9) { cut = i + 1; break; }
  }
  const kept = probs.slice(0, cut);
  total = kept.reduce((s, p) => s + p, 0);
  let r = Math.random() * total;
  for (let i = 0; i < kept.length; i++) {
    r -= kept[i];
    if (r <= 0) return order[i];
  }
  return order[0];
}

async function generate(ids, maxNew, temperature) {
  let feeds = { input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]), ...emptyPast() };
  const generated = [];
  for (let step = 0; step < maxNew; step++) {
    const out = await session.run(feeds);
    const next = sampleToken(out.logits.data, generated, temperature);
    generated.push(next);
    if (next === meta.stop_token_id) break;
    feeds = { input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(next)]), [1, 1]) };
    meta.past_names.forEach((name, i) => (feeds[name] = out[name.replace("past", "present")]));
  }
  return generated;
}

function cleanReply(text) {
  text = text.replace(/<\|im_start\|>|<\|im_end\|>/g, "").trim().split("\n")[0].trim();
  return text || "*babbles happily*";
}

// ----------------------------------------------------------------------------- baby_reply (chat.py)

async function babyReply(baby, parent, userTurn, facts, history, needs, temperature = 0.6) {
  const known = identityAnswer(userTurn, baby, parent);
  if (known) return known;
  const hits = memoryHits(facts, userTurn);
  const topic = worldQuestionTopic(userTurn);
  if (topic && !hits.length) {
    return pick([`Hmm, I don't know about ${topic} yet. Can you teach me, ${parent}?`, `Ooh, ${topic}? I don't know yet. Teach me, please!`,
      `I don't know ${topic}! Will you tell me?`]);
  }
  const maxNew = 48;
  const ids = buildPrompt(baby, parent, needs, facts, history, userTurn, meta.n_positions - maxNew);
  const context = [baby, parent, userTurn, ...facts, ...history.map((m) => m.content)].join(" ");
  const mood = moodFromNeeds(needs);
  const careEvent = Object.values(CARE_EVENTS).includes(userTurn);
  const recent = history.slice(-2).map((m) => m.content).join(" ");
  const checkContext = isQuestion(userTurn) || careEvent ? context : [baby, parent, userTurn, recent].join(" ");
  const impersonalQ = !!topic || (/\?\s*$/.test(userTurn.trim()) && !PERSONAL.test(userTurn.toLowerCase()));

  for (let attempt = 0; attempt < 3; attempt++) {
    const temp = attempt === 0 ? temperature : Math.max(0.3, temperature - 0.2 * attempt);
    let reply = cleanReply(tok.decode(await generate(ids, maxNew, temp)));
    let bad = [...ungroundedWords(reply, context), ...offScriptWords(reply, checkContext)];
    if (bad.length) {
      reply = repairNames(reply, context, bad) || reply;
      bad = [...ungroundedWords(reply, context), ...offScriptWords(reply, checkContext)];
    }
    const fragments = letters(reply).filter((w) => w.length <= 2 && !SHORT_WORDS.has(w.toLowerCase()));
    const broken = (reply.split("*").length - 1) % 2 === 1 || /[A-Za-z]\*[A-Za-z]/.test(reply) || BROKEN_GRAMMAR.test(reply) || REPEATED_AND.test(reply) || fragments.length;
    if (TEACH_ACK.test(reply) || contradictsMood(reply, userTurn, mood) || broken) bad = bad.length ? bad : ["<invented>"];
    if (!bad.length) return quoteCheck(factCheck(reply, impersonalQ ? hits : [], userTurn), userTurn, history);
  }
  if (hits.length && impersonalQ) return factCheck("", hits, userTurn);
  if (QUOTE.test(userTurn.trim())) return quoteCheck("", userTurn, history) || "Hmm?";
  const followup = answerToMyQuestion(userTurn, history, parent);
  if (followup) return followup;
  if (GREETING.test(userTurn)) return pick([`Hi ${parent}! *waves tiny hand*`, `Hello, ${parent}! I missed you!`, "Hiii! Hehe."]);
  if (careEvent) return careLine(Object.keys(CARE_EVENTS).find((k) => CARE_EVENTS[k] === userTurn), mood, parent);
  if (!(/\?\s*$/.test(userTurn.trim()) || topic)) {
    return pick([`Okay, ${parent}! Hehe.`, "Mm-hmm! *listens*", `Hehe! Tell me more, ${parent}!`, "Ooh! *wiggles*"]);
  }
  return `Hmm, I don't know about ${topicOf(userTurn)} yet. Can you teach me, ${parent}?`;
}

// Same request shape and validation as the server's /api/chat
async function handle(data) {
  const event = data.event;
  if (event && !CARE_EVENTS[event]) return { error: "unknown care event" };
  const userTurn = event ? CARE_EVENTS[event] : String(data.message || "").trim().slice(0, 280);
  if (!userTurn) return { error: "message or event required" };
  if (BLOCKED.test(userTurn)) return { reply: KIND_REPLY };
  const history = (Array.isArray(data.history) ? data.history : []).slice(-8)
    .filter((h) => h && (h.role === "user" || h.role === "assistant") && !BLOCKED.test(String(h.content || "")))
    .map((h) => ({ role: h.role, content: String(h.content || "").slice(0, 280) }));
  const facts = (Array.isArray(data.facts) ? data.facts : []).slice(-60).map((f) => String(f).slice(0, 160)).filter((f) => !BLOCKED.test(f));
  let needs = null;
  if (data.needs && typeof data.needs === "object") {
    needs = {};
    for (const k of ["hunger", "sleep", "comfort", "curiosity", "bond"]) {
      const v = Number(data.needs[k]);
      needs[k] = Number.isFinite(v) ? Math.max(0, Math.min(100, v)) : 70;
    }
  }
  const baby = String(data.babyName || "Pip").slice(0, 24);
  const parent = String(data.parentName || "Mom").slice(0, 24);
  const reply = await babyReply(baby, parent, userTurn, facts, history, needs);
  return { reply: BLOCKED.test(reply) ? KIND_REPLY : reply };
}

// ----------------------------------------------------------------------------- loading

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, got);
    got += value.length;
    onProgress(got / total);
  }
  return buf.subarray(0, got);
}

async function init() {
  const [tokJson, metaJson, runtime] = await Promise.all([
    fetch("../tokenizer/tokenizer.json").then((r) => r.json()),
    fetch("baby.meta.json").then((r) => r.json()),
    fetch("runtime.json").then((r) => r.json()),
  ]);
  tok = new BabyTokenizer(tokJson);
  meta = metaJson;
  templateWords = new Set(runtime.template_words);
  templateStems = new Set(runtime.template_words.map(stem));
  knownWords = new Set();
  for (const t of Object.keys(tokJson.model.vocab)) {
    const w = t.startsWith("Ġ") ? t.slice(1) : t;
    if (w.length >= 2 && /^\p{L}+$/u.test(w)) knownWords.add(w.toLowerCase());
  }
  const bytes = await fetchWithProgress("baby.onnx", (p) => postMessage({ type: "progress", value: p }));
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  postMessage({ type: "ready" });
}

self.onmessage = async (e) => {
  const msg = e.data || {};
  if (msg.type === "init") {
    try {
      await init();
    } catch (err) {
      postMessage({ type: "failed", error: String(err && err.message || err) });
    }
    return;
  }
  if (msg.type === "chat") {
    try {
      postMessage({ type: "reply", id: msg.id, ...(await handle(msg.payload || {})) });
    } catch (err) {
      postMessage({ type: "reply", id: msg.id, error: String(err && err.message || err) });
    }
  }
};


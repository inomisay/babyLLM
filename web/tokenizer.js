// Byte-level BPE tokenizer (GPT-2 style), reading the same tokenizer.json the Python side trains.
// Must produce exactly the token ids Python's `tokenizers` does, or the brain reads garbage.
/* exported BabyTokenizer */

class BabyTokenizer {
  constructor(json) {
    const model = json.model;
    this.vocab = model.vocab; // token string -> id
    this.idToToken = [];
    for (const [tok, id] of Object.entries(this.vocab)) this.idToToken[id] = tok;
    this.ranks = new Map();
    model.merges.forEach((m, i) => this.ranks.set(Array.isArray(m) ? m[0] + " " + m[1] : m, i));
    this.special = new Map(json.added_tokens.map((t) => [t.content, t.id]));
    this.specialIds = new Set(json.added_tokens.map((t) => t.id));
    const escaped = [...this.special.keys()].map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.specialSplit = new RegExp("(" + escaped.join("|") + ")");
    // GPT-2 pre-tokenizer pattern (HF ByteLevel with use_regex)
    this.pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.byteToChar = BabyTokenizer.bytesToUnicode();
    this.charToByte = new Map([...this.byteToChar.entries()].map(([b, c]) => [c, b]));
    this.cache = new Map();
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder("utf-8");
  }

  // The reversible byte <-> printable-character map GPT-2 uses
  static bytesToUnicode() {
    const bs = [];
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    const map = new Map();
    bs.forEach((b, i) => map.set(b, String.fromCharCode(cs[i])));
    return map;
  }

  bpe(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    let parts = Array.from(word);
    while (parts.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      const a = parts[best];
      const b = parts[best + 1];
      const merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
          merged.push(a + b);
          i++;
        } else {
          merged.push(parts[i]);
        }
      }
      parts = merged;
    }
    if (this.cache.size < 50000) this.cache.set(word, parts);
    return parts;
  }

  encode(text) {
    const ids = [];
    for (const segment of text.split(this.specialSplit)) {
      if (!segment) continue;
      if (this.special.has(segment)) {
        ids.push(this.special.get(segment));
        continue;
      }
      for (const piece of segment.match(this.pattern) || []) {
        let mapped = "";
        for (const byte of this.encoder.encode(piece)) mapped += this.byteToChar.get(byte);
        for (const tok of this.bpe(mapped)) {
          const id = this.vocab[tok];
          if (id !== undefined) ids.push(id);
        }
      }
    }
    return ids;
  }

  decode(ids, skipSpecial = true) {
    const bytes = [];
    let out = "";
    const flush = () => {
      if (bytes.length) out += this.decoder.decode(new Uint8Array(bytes));
      bytes.length = 0;
    };
    for (const id of ids) {
      if (this.specialIds.has(id)) {
        if (!skipSpecial) {
          flush();
          out += this.idToToken[id];
        }
        continue;
      }
      for (const ch of this.idToToken[id] || "") {
        const b = this.charToByte.get(ch);
        if (b !== undefined) bytes.push(b);
      }
    }
    flush();
    return out;
  }
}

if (typeof module !== "undefined") module.exports = { BabyTokenizer };

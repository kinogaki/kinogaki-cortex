#!/usr/bin/env python3
"""Experiment C — does a grown CONCEPT hierarchy earn its keep? (English, associative substrate.)

The north-star claim is that the model grows higher-level concepts that make it *better* and *inspectable*.
The decisive, cheap test: does conditioning next-char prediction on a learned WORD lexicon (the first concept
level) lower bits-per-char vs a flat char model? Pure association (counts + a frequency-weighted lexicon trie),
no backprop — the substrate Exp B pointed us to.

  flat        : order-K backoff char n-gram (the strong associative baseline from Exp B).
  hierarchical: mix the char model with a LEXICON PRIOR — at each step, the learned words consistent with the
                current word-prefix vote on the next char (and on 'is the word done? → space'). P = λ·char + (1-λ)·lex.

C.1 uses TRUE word boundaries (an upper bound: can a word layer help AT ALL?).
C.2 uses DISCOVERED words (branching-entropy segmentation from Exp A) — does it still help with no supervision?
The lexicon is written to a .prism document (the inspectable concept store) via kinogaki.
"""
import os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
VOCAB = " abcdefghijklmnopqrstuvwxyz"      # 27: space + a-z
CH2I = {c: i for i, c in enumerate(VOCAB)}; V = len(VOCAB); SP = 0
K = 8                                       # char backoff order
ALPHA = 0.02

def load(name="english.txt"):
    raw = open(os.path.join(DATA, name), encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S); m2 = re.search(r"\*\*\* END OF", raw, re.S)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    body = re.sub(r"[^a-z]+", " ", body.lower()).strip()
    return re.sub(r" +", " ", body)

# ---------- order-K backoff char model (associative, online-friendly) ----------
class CharNGram:
    def __init__(self): self.c = [dict() for _ in range(K + 1)]
    def fit(self, s):
        for t in range(len(s)):
            for k in range(K + 1):
                ctx = s[t - k:t] if k <= t else None
                if ctx is None: continue
                d = self.c[k].get(ctx)
                if d is None: d = np.zeros(V); self.c[k][ctx] = d
                d[CH2I[s[t]]] += 1
    def dist(self, ctx):
        for k in range(min(K, len(ctx)), -1, -1):
            d = self.c[k].get(ctx[len(ctx) - k:] if k else "")
            if d is not None and d.sum() > 0:
                return (d + ALPHA) / (d.sum() + ALPHA * V)
        return np.full(V, 1.0 / V)

# ---------- lexicon trie (frequency-weighted) → next-char prior given a word prefix ----------
class Lexicon:
    def __init__(self): self.kids = {}; self.end = 0.0; self.tot = 0.0
    def add(self, word, w=1.0):
        node = self
        for ch in word:
            node.tot += w
            node = node.kids.setdefault(ch, Lexicon())
        node.tot += w; node.end += w
    def walk(self, prefix):
        node = self
        for ch in prefix:
            node = node.kids.get(ch)
            if node is None: return None
        return node
    def prior(self, prefix):
        """Distribution over next char (incl space=word-end) consistent with `prefix`, by word frequency."""
        node = self.walk(prefix)
        p = np.zeros(V)
        if node is None or node.tot <= 0:
            return None
        p[SP] = node.end                                  # 'word ends here' → space
        for ch, kid in node.kids.items():
            p[CH2I[ch]] = kid.tot                          # continue with this letter
        s = p.sum()
        return p / s if s > 0 else None

def build_lexicon(words):
    lex = Lexicon()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    for w, f in freq.items():
        lex.add(w, w=f)
    return lex, freq

def bpc_flat(model, s):
    tot = 0.0
    for t in range(1, len(s)):
        tot += -math.log2(model.dist(s[max(0, t - K):t])[CH2I[s[t]]] + 1e-12)
    return tot / (len(s) - 1)

def bpc_hier(model, lex, s, lam):
    tot = 0.0; prefix = ""
    for t in range(1, len(s)):
        ctx = s[max(0, t - K):t]
        pc = model.dist(ctx)
        pl = lex.prior(prefix)
        P = pc if pl is None else lam * pc + (1 - lam) * pl
        tot += -math.log2(P[CH2I[s[t]]] + 1e-12)
        prefix = "" if s[t] == " " else prefix + s[t]
    return tot / (len(s) - 1)

def build_word_bigram(words):
    big = {}
    for a, b in zip(words, words[1:]):
        d = big.setdefault(a, {}); d[b] = d.get(b, 0) + 1
    return big

def firstchar_prior(bigram, freq, prev_word):
    """Level-2 top-down: P(first char of next word) from word→word context (backoff to unigram lexicon)."""
    nxt = bigram.get(prev_word)
    p = np.zeros(V)
    if nxt:
        for w, c in nxt.items():
            if w: p[CH2I[w[0]]] += c
    else:
        for w, c in freq.items():
            if w: p[CH2I[w[0]]] += c
    s = p.sum()
    return p / s if s > 0 else None

def bpc_ctx(model, bigram, freq, s, lam, topN=400):
    """Proper Level-2: a context-aware lexical prior. At each word, hold a candidate next-word distribution Q
    (from word->word bigram, backed off to unigram), filter it by the prefix as chars arrive, and marginalize
    over Q for the next-char prior. Conditions the WHOLE word on context, not just its first char."""
    uni = sorted(freq.items(), key=lambda x: -x[1])[:topN]
    uni_w = np.array([f for _, f in uni], dtype=float); uni_w /= uni_w.sum()
    def start_Q(prev):
        nxt = bigram.get(prev)
        Q = {}
        if nxt:
            tot = sum(nxt.values())
            for w, c in nxt.items(): Q[w] = 0.7 * c / tot
        for (w, _), pw in zip(uni, uni_w):                # backoff mass to unigram top-N
            Q[w] = Q.get(w, 0.0) + 0.3 * pw
        items = sorted(Q.items(), key=lambda x: -x[1])[:topN]
        return [w for w, _ in items], np.array([v for _, v in items])
    tot = 0.0; prefix = ""; prev_word = ""
    words, wt = start_Q("")
    for t in range(1, len(s)):
        pc = model.dist(s[max(0, t - K):t])
        p = np.zeros(V); pos = len(prefix)
        for w, q in zip(words, wt):
            p[CH2I[w[pos]] if pos < len(w) else SP] += q
        sm = p.sum(); pl = p / sm if sm > 0 else None
        P = pc if pl is None else lam * pc + (1 - lam) * pl
        tot += -math.log2(P[CH2I[s[t]]] + 1e-12)
        if s[t] == " ":
            prev_word = prefix; prefix = ""; words, wt = start_Q(prev_word)
        else:
            prefix += s[t]
            keep = [(w, q) for w, q in zip(words, wt) if len(w) > len(prefix) - 1 and w[:len(prefix)] == prefix]
            if keep: words, wt = [w for w, _ in keep], np.array([q for _, q in keep])
            else: words, wt = [], np.array([])
    return tot / (len(s) - 1)

def bpc_cache(model, bigram, freq, s, lam, topN=400, decay=0.997, cache_w=0.4):
    """Level-2 done right: add a decaying word CACHE (recently-seen concepts are likelier again) — the
    long-range/topical signal the char window can't reach, and exactly our north-star recency/activation rule.
    Candidate next-word dist Q = bigram(0.4) + unigram(0.2) + cache(0.4); marginalize over the prefix-filtered Q."""
    uni = sorted(freq.items(), key=lambda x: -x[1])[:topN]
    uni_w = np.array([f for _, f in uni], float); uni_w /= uni_w.sum()
    cache = {}
    def start_Q(prev):
        nxt = bigram.get(prev); Q = {}
        if nxt:
            tot = sum(nxt.values())
            for w, c in nxt.items(): Q[w] = (1 - cache_w) * 0.7 * c / tot
        for (w, _), pw in zip(uni, uni_w): Q[w] = Q.get(w, 0.0) + (1 - cache_w) * 0.3 * pw
        cs = sum(cache.values())
        if cs > 0:
            for w, v in cache.items(): Q[w] = Q.get(w, 0.0) + cache_w * v / cs
        items = sorted(Q.items(), key=lambda x: -x[1])[:topN]
        return [w for w, _ in items], np.array([v for _, v in items])
    tot = 0.0; prefix = ""; prev_word = ""; words, wt = start_Q("")
    for t in range(1, len(s)):
        pc = model.dist(s[max(0, t - K):t])
        p = np.zeros(V); pos = len(prefix)
        for w, q in zip(words, wt):
            p[CH2I[w[pos]] if pos < len(w) else SP] += q
        sm = p.sum(); pl = p / sm if sm > 0 else None
        P = pc if pl is None else lam * pc + (1 - lam) * pl
        tot += -math.log2(P[CH2I[s[t]]] + 1e-12)
        if s[t] == " ":
            for w in list(cache): cache[w] *= decay
            if prefix: cache[prefix] = cache.get(prefix, 0.0) + 1.0
            prev_word = prefix; prefix = ""; words, wt = start_Q(prev_word)
        else:
            prefix += s[t]
            keep = [(w, q) for w, q in zip(words, wt) if len(w) > len(prefix) - 1 and w[:len(prefix)] == prefix]
            words, wt = ([w for w, _ in keep], np.array([q for _, q in keep])) if keep else ([], np.array([]))
    return tot / (len(s) - 1)

def bpc_hier_cache(model, lex, s, lam, gamma=0.4, decay=0.997):
    """CLEAN additive compounding test: keep L1's FULL within-word trie prior, and ADD a decaying word cache
    that boosts the first char of recently-seen words at word starts. Only ever adds signal on top of L1."""
    tot = 0.0; prefix = ""; cache = {}
    def cache_fc():
        p = np.zeros(V);
        for w, v in cache.items():
            if w: p[CH2I[w[0]]] += v
        s_ = p.sum(); return p / s_ if s_ > 0 else None
    for t in range(1, len(s)):
        pc = model.dist(s[max(0, t - K):t])
        pl = lex.prior(prefix)
        if prefix == "" and pl is not None:
            fc = cache_fc()
            if fc is not None: pl = (1 - gamma) * pl + gamma * fc
        P = pc if pl is None else lam * pc + (1 - lam) * pl
        tot += -math.log2(P[CH2I[s[t]]] + 1e-12)
        if s[t] == " ":
            for w in list(cache): cache[w] *= decay
            if prefix: cache[prefix] = cache.get(prefix, 0.0) + 1.0
            prefix = ""
        else:
            prefix += s[t]
    return tot / (len(s) - 1)

def bpc_hier2(model, lex, bigram, freq, s, lam):
    """Level-1 (within-word lexicon) + Level-2 (word-context first-char prior). Tests if the hierarchy compounds."""
    tot = 0.0; prefix = ""; prev_word = ""
    for t in range(1, len(s)):
        ctx = s[max(0, t - K):t]
        pc = model.dist(ctx)
        if prefix == "":                                  # predicting the first char of a new word
            pl = firstchar_prior(bigram, freq, prev_word)
        else:                                              # mid-word: the within-word lexicon prior
            pl = lex.prior(prefix)
        P = pc if pl is None else lam * pc + (1 - lam) * pl
        tot += -math.log2(P[CH2I[s[t]]] + 1e-12)
        if s[t] == " ":
            prev_word = prefix; prefix = ""
        else:
            prefix += s[t]
    return tot / (len(s) - 1)

# ---------- branching-entropy word discovery (from Exp A) for the unsupervised C.2 ----------
def discover_words(model_fwd, model_bwd, s):
    n = len(s)
    Hf = np.array([float(-(p := model_fwd.dist(s[max(0, t - K):t]))[1:] @ np.log2(p[1:] + 1e-12)) for t in range(n)])
    rs = s[::-1]
    Hb = np.array([float(-(p := model_bwd.dist(rs[max(0, t - K):t]))[1:] @ np.log2(p[1:] + 1e-12)) for t in range(n)])[::-1]
    rise = lambda H: np.concatenate([[0.0], np.maximum(0.0, H[1:] - H[:-1])])
    score = rise(Hf) + rise(Hb[::-1])[::-1]
    # cut where score is in the top quantile matching the true space rate (so segment lengths are realistic)
    rate = s.count(" ") / n
    thr = np.quantile(score, 1 - rate)
    words, cur = [], ""
    for t in range(n):
        ch = s[t]
        if ch == " ":
            if cur: words.append(cur); cur = ""
            continue
        if t > 0 and score[t] >= thr and cur:
            words.append(cur); cur = ""
        cur += ch
    if cur: words.append(cur)
    return words

def write_prism(freq, tag):
    try:
        import kinogaki as kg
    except Exception as e:
        print(f"  (.prism skipped: {e})"); return
    doc = kg.Document(); doc.append("/lexicon", "field")
    for i, (w, f) in enumerate(sorted(freq.items(), key=lambda x: -x[1])[:200]):
        doc.append(f"/lexicon/c{i}", "concept").set_meta("text", w).set("freq", float(f))
    out = os.path.join(HERE, f"lexicon_{tag}.prisma"); doc.save(out)
    print(f"  inspectable concept store ({len(freq)} words, top 200 written) → {out}")

if __name__ == "__main__":
    s = load("english.txt")
    cut = int(len(s) * 0.85)
    train, test = s[:cut], s[cut:cut + 120_000]
    print(f"english: {len(s):,} chars | train {len(train):,} | test {len(test):,} | vocab {V}")

    cm = CharNGram(); cm.fit(train)
    base = bpc_flat(cm, test)
    print(f"\n  flat char {K}-gram (baseline) bpc = {base:.4f}")

    # --- C.1: upper bound, TRUE words ---
    true_words = train.split(" ")
    lex_true, freq_true = build_lexicon(true_words)
    print(f"\n  C.1  TRUE-word lexicon: {len(freq_true):,} unique words")
    best = (base, 1.0)
    for lam in (0.9, 0.7, 0.5, 0.3, 0.1):
        b = bpc_hier(cm, lex_true, test, lam)
        better = (base - b) / base * 100
        print(f"    λ={lam:.1f}  hier bpc = {b:.4f}   ({better:+.1f}% vs flat)")
        if b < best[0]: best = (b, lam)
    print(f"  >> best TRUE-word: bpc {best[0]:.4f} at λ={best[1]} ({(base-best[0])/base*100:+.1f}% vs flat {base:.4f})")
    write_prism(freq_true, "true")

    # --- C.2: DISCOVERED words (unsupervised, branching entropy) ---
    cm_b = CharNGram(); cm_b.fit(train[::-1])
    disc = discover_words(cm, cm_b, train)
    lex_disc, freq_disc = build_lexicon(disc)
    print(f"\n  C.2  DISCOVERED-word lexicon (branching-entropy, unsupervised): {len(freq_disc):,} unique 'words'")
    print(f"       sample discovered: {' '.join(disc[1000:1018])}")
    best2 = (base, 1.0)
    for lam in (0.9, 0.7, 0.5, 0.3):
        b = bpc_hier(cm, lex_disc, test, lam)
        print(f"    λ={lam:.1f}  hier bpc = {b:.4f}   ({(base-b)/base*100:+.1f}% vs flat)")
        if b < best2[0]: best2 = (b, lam)
    print(f"  >> best DISCOVERED: bpc {best2[0]:.4f} at λ={best2[1]} ({(base-best2[0])/base*100:+.1f}% vs flat)")
    write_prism(freq_disc, "discovered")

    # --- C.3: does the hierarchy COMPOUND? add level-2 word-context (TRUE words) ---
    big_true = build_word_bigram(true_words)
    print(f"\n  C.3  + Level-2 word-context (word→word bigram constrains first chars):")
    best3 = (best[0], None)
    for lam in (0.5, 0.3, 0.2):
        b = bpc_hier2(cm, lex_true, big_true, freq_true, test, lam)
        print(f"    λ={lam:.1f}  L1+L2 bpc = {b:.4f}   ({(base-b)/base*100:+.1f}% vs flat;  L1-only best was {best[0]:.4f})")
        if b < best3[0]: best3 = (b, lam)
    print(f"\n  C.4  context-aware lexical prior (marginalize next-word over word-context):")
    best4 = (base, None)
    for lam in (0.3, 0.2):
        b = bpc_ctx(cm, big_true, freq_true, test, lam)
        print(f"    λ={lam:.1f}  ctx bpc = {b:.4f}   ({(base-b)/base*100:+.1f}% vs flat)")
        if b < best4[0]: best4 = (b, lam)

    print(f"\n  C.5  + decaying word CACHE (recency/activation — the long-range concept signal):")
    best5 = (base, None)
    for lam in (0.3, 0.2, 0.1):
        b = bpc_cache(cm, big_true, freq_true, test, lam)
        print(f"    λ={lam:.1f}  cache bpc = {b:.4f}   ({(base-b)/base*100:+.1f}% vs flat;  L1 best {best[0]:.4f})")
        if b < best5[0]: best5 = (b, lam)

    print(f"\n  C.6  CLEAN additive: L1 full trie + word cache (recency) on top:")
    best6 = (best[0], None)
    for lam in (0.3, 0.2):
        for g in (0.3, 0.5):
            b = bpc_hier_cache(cm, lex_true, test, lam, gamma=g)
            mark = "  <-- beats L1" if b < best[0] - 1e-4 else ""
            print(f"    λ={lam:.1f} γ={g:.1f}  bpc = {b:.4f}   ({(base-b)/base*100:+.1f}% vs flat){mark}")
            if b < best6[0]: best6 = (b, (lam, g))

    print(f"\n  === LADDER (bits-per-char, lower=better) ===")
    print(f"    flat char {K}-gram               {base:.4f}")
    print(f"    + word concepts (L1)            {best[0]:.4f}   ({(base-best[0])/base*100:+.1f}%)")
    print(f"    + context-aware lexical (L2')   {best4[0]:.4f}   ({(base-best4[0])/base*100:+.1f}%)")
    print(f"    + word cache / recency (L2c)    {best5[0]:.4f}   ({(base-best5[0])/base*100:+.1f}%)")
    print(f"    + L1 + cache (clean additive)   {best6[0]:.4f}   ({(base-best6[0])/base*100:+.1f}%)")

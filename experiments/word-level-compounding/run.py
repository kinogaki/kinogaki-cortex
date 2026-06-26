#!/usr/bin/env python3
"""Experiment E — does the hierarchy compound at the RIGHT granularity? (word-level prediction, English.)

Exp D showed char-bits-per-char saturates with local+lexical context, so higher levels can't compound there.
The right place to test "do phrases/themes help" is predicting the next WORD, where context matters. Same
multi-level voting idea: experts = word-unigram, word-bigram (phrase), word-trigram (longer phrase), and a
decaying word CACHE (topic/recency). Combined by learned product-of-experts. Metric: bits-per-WORD.

If each higher level lowers bits-per-word, the hierarchy compounds at the level it operates on — i.e. each
concept level helps predict the level below it, online and inspectably.
"""
import os, re, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")

def load(name="english.txt"):
    raw = open(os.path.join(DATA, name), encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S); m2 = re.search(r"\*\*\* END OF", raw, re.S)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    return re.sub(r"[^a-z]+", " ", body.lower()).split()

if __name__ == "__main__":
    words = load(); cut = int(len(words)*0.85); train, test = words[:cut], words[cut:]
    # vocab: top VW words + <unk>
    from collections import Counter
    cnt = Counter(train); VW = 6000
    vocab = [w for w, _ in cnt.most_common(VW)]; w2i = {w: i for i, w in enumerate(vocab)}; UNK = VW
    M = VW + 1
    def ids(ws): return np.array([w2i.get(w, UNK) for w in ws])
    tr, te = ids(train), ids(test)
    print(f"english words: train {len(tr):,} / test {len(te):,}; vocab {VW}+unk; OOV rate {np.mean(te==UNK):.3f}")

    ALPHA = 0.1
    uni = np.zeros(M)
    big = {}; trg = {}
    for w in tr: uni[w] += 1
    for a, b in zip(tr, tr[1:]):
        d = big.get(a)
        if d is None: d = {}; big[a] = d
        d[b] = d.get(b, 0)+1
    for a, b, c in zip(tr, tr[1:], tr[2:]):
        d = trg.get((a, b))
        if d is None: d = {}; trg[(a, b)] = d
        d[c] = d.get(c, 0)+1
    uni_p = (uni + ALPHA) / (uni.sum() + ALPHA*M)

    def dist_from(d):
        p = np.full(M, ALPHA)
        if d:
            for k, v in d.items(): p[k] += v
        return p / p.sum()

    # experts: unigram, bigram, trigram, cache(topic)
    EXP = ["unigram", "bigram", "trigram", "cache"]; NE = len(EXP)
    a = np.full(NE, 0.5); lr = 0.02
    def softmax(z): z = z - z.max(); e = np.exp(z); return e/e.sum()

    cache = np.zeros(M); decay = 0.99
    solo = np.zeros(NE); prod = 0.0; uni_only = 0.0
    p1 = p2 = UNK
    for w in te:
        ds = np.empty((NE, M))
        ds[0] = uni_p
        ds[1] = dist_from(big.get(p2))
        ds[2] = dist_from(trg.get((p1, p2)))
        cs = cache.sum()
        ds[3] = (cache + ALPHA) / (cs + ALPHA*M) if cs > 0 else uni_p
        ds = np.clip(ds, 1e-12, None)
        z = np.log(ds)
        Pp = softmax(a @ z)
        prod += -math.log2(Pp[w] + 1e-12)
        uni_only += -math.log2(uni_p[w])
        solo += -np.log2(ds[:, w])
        a = np.clip(a + lr*(z[:, w] - z @ Pp), 0.0, 5.0)
        # advance
        cache *= decay; cache[w] += 1.0
        p1, p2 = p2, w
    N = len(te)
    print("\n  === solo views (bits-per-word) ===")
    for j, n in enumerate(EXP): print(f"    {n:<9} {solo[j]/N:.3f}")
    print("\n  === LADDER (bits-per-word, lower=better) — does the hierarchy compound? ===")
    print(f"    unigram (baseline)              {uni_only/N:.3f}")
    print(f"    + bigram (phrase, L2)           {solo[1]/N:.3f}   ({(uni_only/N-solo[1]/N)/(uni_only/N)*100:+.1f}%)")
    print(f"    + trigram (longer phrase, L3)   {solo[2]/N:.3f}   ({(uni_only/N-solo[2]/N)/(uni_only/N)*100:+.1f}%)")
    print(f"    PRODUCT-of-experts (all levels) {prod/N:.3f}   ({(uni_only/N-prod/N)/(uni_only/N)*100:+.1f}% vs unigram)")
    print(f"\n  learned product weights: " + "  ".join(f"{EXP[j]}={a[j]:.2f}" for j in range(NE)))
    print(f"  word-level perplexity: unigram {2**(uni_only/N):.0f}  →  full hierarchy {2**(prod/N):.0f}")

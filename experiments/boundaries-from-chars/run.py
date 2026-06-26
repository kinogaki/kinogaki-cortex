#!/usr/bin/env python3
"""Experiment A — the boundary kill-test for kinogaki-cortex.

The bet: prediction-error segmentation of RAW characters recovers word boundaries, and *transient Bayesian
surprise* (a shift in the predictive belief) beats plain surprisal / entropy — as Kumar & Zacks (2023) found
for event boundaries. We test it on space-stripped English: feed an online char predictor the letters with no
spaces, score each position, and see whether the high-scoring positions land on the removed word boundaries.

Predictor: an online variable-order char model (orders 0..K) with a *belief b over which order is generating
the stream*. Predicting: P(next)=Σ_k b_k p_k(next). Observing c: posterior b'_k ∝ b_k p_k(c). The latent
"regime" belief shifts hard at structural boundaries (a finished word makes the long context uninformative).
  - surprisal(t)       = -log2 P(c)                  [baseline the paper says FAILS]
  - entropy(t)         = H(P)                          [baseline the paper says FAILS]
  - bayes_surprise(t)  = KL(b' || b)                   [belief shift — the paper's winner]
  - transient(t)       = bayes_surprise - EMA(bayes_surprise)   [stands out vs low-error background]
No neural net, no GPU, fully online — exactly the regime kinogaki-cortex commits to.
"""
import os, re, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
K = 6                       # max context order
ALPHA = 0.05               # add-alpha smoothing
LEAK = 0.05                # belief leak toward uniform (keeps it tracking, avoids lock-in)
EMA = 0.02                 # slow baseline rate for the "transient" signal
WARMUP = 8000              # skip while statistics build
A = 26                     # alphabet: a-z

def load_corpus():
    raw = open(os.path.join(HERE, "data", "raw.txt"), encoding="utf-8", errors="ignore").read()
    # strip Project Gutenberg header/footer
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S)
    m2 = re.search(r"\*\*\* END OF", raw, re.S)
    body = raw[m1.end():m2.start()] if (m1 and m2) else raw
    words = re.sub(r"[^a-z]+", " ", body.lower()).split()
    # build the space-stripped stream + the true word-boundary indices (start-of-word positions)
    stream, bnd = [], set()
    for i, w in enumerate(words):
        if i > 0:
            bnd.add(len(stream))
        stream.extend(ord(ch) - 97 for ch in w)
    return np.array(stream, dtype=np.int8), bnd

def run(stream, bnd):
    n = len(stream)
    counts = [dict() for _ in range(K + 1)]            # counts[k][ctx] -> np.array(A)
    b = np.full(K + 1, 1.0 / (K + 1))                  # belief over orders
    surprisal = np.zeros(n); entropy = np.zeros(n); bsurp = np.zeros(n)
    s = "".join(chr(97 + int(x)) for x in stream)      # context lookups as strings

    for t in range(n):
        c = int(stream[t])
        # per-order smoothed predictive distributions for the current context
        pk = np.empty((K + 1, A))
        ctxs = []
        for k in range(K + 1):
            ctx = s[t - k:t] if k <= t else None
            ctxs.append(ctx)
            arr = counts[k].get(ctx) if ctx is not None else None
            if arr is None:
                pk[k] = np.full(A, 1.0 / A)
            else:
                pk[k] = (arr + ALPHA) / (arr.sum() + ALPHA * A)
        P = b @ pk                                     # mixture prediction over next char
        P /= P.sum()
        surprisal[t] = -math.log2(max(P[c], 1e-12))
        entropy[t] = float(-(P * np.log2(P + 1e-12)).sum())
        # Bayesian update of the belief over orders + surprise = KL(posterior || prior)
        like = pk[:, c]
        bp = b * like
        bp /= bp.sum()
        bsurp[t] = float(np.sum(bp * np.log((bp + 1e-12) / (b + 1e-12))))
        b = (1 - LEAK) * bp + LEAK / (K + 1)           # leak toward uniform for next step
        # online count update for every order
        for k in range(K + 1):
            ctx = ctxs[k]
            if ctx is None:
                continue
            arr = counts[k].get(ctx)
            if arr is None:
                arr = np.zeros(A); counts[k][ctx] = arr
            arr[c] += 1
    # transient = raw minus a slow EMA baseline
    base = np.zeros(n); acc = 0.0
    for t in range(n):
        acc = (1 - EMA) * acc + EMA * bsurp[t]
        base[t] = acc
    transient = bsurp - base
    return dict(surprisal=surprisal, entropy=entropy, bayes_surprise=bsurp, transient=transient)

def evaluate(signals, names, bnd, n):
    # evaluate on positions past warmup; rank by signal, take top-K (= #true boundaries in region), F1 @ ±1 tol
    region = list(range(WARMUP, n))
    true = sorted(b for b in bnd if b >= WARMUP)
    trueset = set(true)
    K_true = len(true)
    density = K_true / len(region)
    reg_arr = np.array(region)
    def f1_at(score):
        idx = reg_arr[np.argsort(-score[reg_arr])[:K_true]]
        predset = set(int(x) for x in idx)
        hit = sum(1 for p in predset if (p in trueset or p - 1 in trueset or p + 1 in trueset))
        prec = hit / max(len(predset), 1)
        rhit = sum(1 for b in true if (b in predset or b - 1 in predset or b + 1 in predset))
        rec = rhit / max(K_true, 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"\n  region positions={len(region):,}  true word-boundaries={K_true:,}  density={density:.3f}")
    print(f"  {'signal':<22} {'precision':>10} {'recall':>9} {'F1':>8}")
    rng = np.random.default_rng(0)
    rows = [("random", *f1_at(rng.random(n)))]
    for name in names:
        rows.append((name, *f1_at(signals[name])))
    best = max(rows, key=lambda r: r[3])[0]
    for name, p, r, f in rows:
        print(f"  {name:<22} {p:>10.3f} {r:>9.3f} {f:>8.3f}{'   <<<' if name == best else ''}")
    return rows

def write_prism(stream, signals, which, bnd, n):
    try:
        import kinogaki as kg
    except Exception as e:
        print(f"\n  (.prism output skipped: {e})"); return
    # segment the stream at the BEST signal's peaks (top-K) and write the discovered words as a .prism doc
    region = np.arange(WARMUP, n)
    K_true = len([b for b in bnd if b >= WARMUP])
    peaks = sorted(int(x) for x in region[np.argsort(-signals[which][region])[:K_true]])
    s = "".join(chr(97 + int(x)) for x in stream)
    doc = kg.Document(); doc.append("/cortex", "field")
    seg, prev, shown = 0, WARMUP, []
    for p in peaks:
        word = s[prev:p]
        if word:
            if seg < 60:
                doc.append(f"/cortex/seg{seg}", "concept").set_meta("text", word).set("activation", 1.0)
            if len(shown) < 16:
                shown.append(word)
            seg += 1
        prev = p
    out = os.path.join(HERE, "data", "segments.prisma")
    doc.save(out)
    print(f"\n  dogfood (.prism via kinogaki {kg.__version__}): {seg} discovered segments from '{which}' → {out}")
    print("  first discovered 'words':", " ".join(shown))

def add_branching(sig, stream):
    """Add the literature's actual word-segmentation cue: forward+backward branching entropy and its rise.
    A word boundary is where BOTH the next-char (forward) and prev-char (backward) become uncertain."""
    n = len(stream)
    Hf = sig["entropy"]                                  # forward entropy (predict next | left context)
    back = run(stream[::-1], set())                      # same model on the reversed stream
    Hb = back["entropy"][::-1]                            # align: backward entropy (predict prev | right context)
    sig["bwd_entropy"] = Hb
    sig["branch_sum"] = Hf + Hb                           # both uncertain at a boundary
    rise = lambda H: np.concatenate([[0.0], np.maximum(0.0, H[1:] - H[:-1])])
    sig["branch_rise"] = rise(Hf) + rise(Hb[::-1])[::-1]  # rise in forward + rise in backward (mirrored)
    return sig

if __name__ == "__main__":
    stream, bnd = load_corpus()
    print(f"corpus: {len(stream):,} letters, {len(bnd):,} word boundaries (space-stripped)")
    sig = run(stream, bnd)
    sig = add_branching(sig, stream)
    names = ["surprisal", "entropy", "bwd_entropy", "branch_sum", "branch_rise", "bayes_surprise", "transient"]
    rows = evaluate(sig, names, bnd, len(stream))
    write_prism(stream, sig, "branch_rise", bnd, len(stream))

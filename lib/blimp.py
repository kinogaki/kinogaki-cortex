"""blimp.py — a read-side grammaticality Probe over the existing char vote, plus the
impossible-language (Kallini 2024) ablation.

THE FIELD'S WAY, ON COUNTS. BLiMP (Warstadt et al. 2020) scores a model by minimal pairs:
a grammatical sentence s+ and a minimally-different ungrammatical s-, and the model is
"right" on that item iff it assigns s+ a HIGHER probability (lower surprisal) than s-.
Macro accuracy = fraction of pairs the model orders correctly. No training on the eval —
the Probe only READS the model's `.dist`.

We score a sentence by summed per-position surprisal under whatever next-char distribution
a model exposes (anything with `.K` and `.dist(suffix)->np(27)`):

    surprisal(s) = Σ_t  -log2 P(s[t] | s[max(0,t-K):t])

A minimal pair is scored CORRECT when surprisal(s+) < surprisal(s-). Because s+ and s-
differ in only a few characters, the shared prefix/suffix surprisal mostly cancels — the
decision rides on the locally-ungrammatical span, exactly as the field intends.

THE ABLATION. Kallini et al. (2024), "Mission: Impossible Language Models": a learner with
the right inductive bias should acquire a NATURAL language more easily than an IMPOSSIBLE
one (a deterministically-scrambled counterfactual). We train the SAME char band on natural
English and on a position-scrambled English of EQUAL bytes, and compare how grammatical the
two learners look (minimal-pair accuracy) AND how surprised each is by held-out text of its
own kind (bpc). THE CONFOUND (the spec's required fix): a scramble may simply be HARDER
(higher entropy), so a natural-minus-scramble gap could be entropy-driven, not
naturalness-driven. We therefore report the scramble's own bpc next to the gap, and run a
LOCAL scramble (shuffle within a small window) that preserves most short-range statistics —
if naturalness still wins under matched local entropy, the bias is about structure, not raw
predictability.

Online single streaming pass (the models are fit once, read-only here); bounded memory;
no gradients / k-means / SVD. The Probe itself never learns.
"""
import math
import numpy as np

A = "abcdefghijklmnopqrstuvwxyz "; V = len(A); CH = {c: i for i, c in enumerate(A)}


# ─────────────────────────── the minimal-pair Probe ───────────────────────────

def surprisal(model, s):
    """Total bits to encode `s` under the model's next-char vote (the field's sentence score).
    Lower = the model finds the string more grammatical/expected."""
    K = getattr(model, "K", 64)
    tot = 0.0
    for t in range(1, len(s)):
        ctx = s[max(0, t - K):t]
        p = model.dist(ctx)[CH[s[t]]]
        tot += -math.log2(p + 1e-12)
    return tot


def length_norm_surprisal(model, s):
    """Per-character surprisal — guards against length artefacts when s+ and s- differ in length."""
    return surprisal(model, s) / max(1, len(s) - 1)


def score_pair(model, good, bad, length_norm=False):
    """One BLiMP item. Returns (correct, sg, sb): correct iff the model prefers `good`
    (assigns it the lower surprisal)."""
    f = length_norm_surprisal if length_norm else surprisal
    sg, sb = f(model, good), f(model, bad)
    return (sg < sb), sg, sb


def evaluate(model, pairs, length_norm=False):
    """Macro BLiMP-style accuracy over a list of (phenomenon, good, bad) triples, plus a
    per-phenomenon breakdown and the mean surprisal margin (bad - good, in bits — positive
    means the model leans the right way)."""
    by = {}
    margins = []
    for phen, good, bad in pairs:
        ok, sg, sb = score_pair(model, good, bad, length_norm)
        b = by.setdefault(phen, [0, 0, 0.0])
        b[0] += int(ok); b[1] += 1; b[2] += (sb - sg)
        margins.append(sb - sg)
    per_phen = {k: (v[0] / v[1], v[1], v[2] / v[1]) for k, v in by.items()}
    n = sum(v[1] for v in by.values()); c = sum(v[0] for v in by.values())
    return {
        "macro": c / n if n else 0.0,
        "n": n,
        "per_phen": per_phen,                       # phen -> (acc, count, mean_margin_bits)
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
    }


# ─────────────────────────── the impossible-language scrambles ───────────────────────────

def scramble_global(text, seed=0):
    """*HOP-style* impossible language: a deterministic GLOBAL permutation of token (here word)
    positions within each sentence-ish window. Destroys word order entirely while preserving the
    exact word multiset (the unigram distribution is untouched — only structure is removed)."""
    rng = np.random.default_rng(seed)
    words = text.split(" ")
    out = []
    i = 0
    while i < len(words):
        win = words[i:i + 20]
        idx = rng.permutation(len(win))
        out.extend(win[j] for j in idx)
        i += 20
    return " ".join(out)


def scramble_local(text, window=4, seed=0):
    """A LOCAL scramble — shuffle words only within a tiny `window`. Preserves most short-range
    co-occurrence (so its bpc stays close to natural), but still breaks long-range word order.
    The entropy-matched control: if natural still beats THIS, the bias is structural, not entropic."""
    rng = np.random.default_rng(seed)
    words = text.split(" ")
    out = []
    i = 0
    while i < len(words):
        win = words[i:i + window]
        idx = rng.permutation(len(win))
        out.extend(win[j] for j in idx)
        i += window
    return " ".join(out)


def scramble_reverse(text):
    """Deterministic word-reversal within each sentence window — the simplest *REVERSE* impossible
    language from Kallini (a non-count-able position rule). No RNG; fully reproducible."""
    words = text.split(" ")
    out = []
    i = 0
    while i < len(words):
        win = words[i:i + 20]
        out.extend(reversed(win))
        i += 20
    return " ".join(out)

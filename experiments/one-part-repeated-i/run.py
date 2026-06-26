#!/usr/bin/env python3
"""Exp I — the UNIFORM-COMPONENT cortex: one Column, wired bigger → better ("cortex small vs big").

The thesis (the user's steer): we don't add new mechanisms; we replicate and stack the SAME part. A Column is
an online associative predictor over a token stream. A Level is N Columns voting (product-of-experts). A Cortex
stacks Levels: char-Columns predict chars; word-Columns predict the current word from the previous words and
hand a top-down char prior back down. Growing the cortex along two axes —

   WIDTH  : more char Columns voting           (1 col → 3 cols)
   DEPTH  : stack a word Level, then widen it   (+word band → +phrase band = longer-timescale word columns)

— should monotonically improve the standard scorecard (lib/metrics): bits-per-char, overfit gap, real-word %,
phrase-coherence %. Same component throughout; only the wiring changes.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import metrics as M
from metrics import CH
from cortex import Cortex

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "..", "data")

def text(n0, n1):
    raw = open(os.path.join(DATA, "text8"), "rb").read()[n0:n1].decode("ascii", "ignore")
    return "".join(c for c in raw if c in CH)

if __name__ == "__main__":
    np.random.seed(0)
    train = text(0, 2_000_000); test = text(98_000_000, 100_000_000)
    vocab = set(w for w in train.split(" ") if w)
    toks = [w for w in train.split(" ") if w]
    bigram_set = set(zip(toks, toks[1:]))
    tr_s, te_s = train[:200_000], test[:200_000]; seed = test[400_000:600_000]
    print(f"train {len(train):,} chars / test {len(test):,} / vocab {len(vocab):,} words\n")

    # Same Column everywhere — only the wiring (column count, stacked levels) grows.
    configs = [
        ("1 char col            ", dict(char_orders=[6])),                              # smallest
        ("3 char cols (vote)     ", dict(char_orders=[2, 4, 6])),                       # WIDER
        ("+ word level           ", dict(char_orders=[2, 4, 6], word_orders=[1, 2, 3])),# DEEPER
        ("+ phrase band (wider)  ", dict(char_orders=[2, 4, 6], word_orders=[1, 2, 3, 4, 5])),  # longer timescale
    ]
    rows = []
    for name, kw in configs:
        cx = Cortex(**kw).fit(train)
        r = M.report(name.strip(), cx, tr_s, te_s, vocab, seed)
        gen = M.generate(cx, seed[:48], 3000, 0.5)            # calibrated pool → sharpen at SAMPLING (temp 0.5)
        r["validg"] = M.text_validity(gen, vocab)
        r["phrase"] = M.phrase_coherence(gen, bigram_set)
        r["sample"] = " ".join(gen.split()[1:26])
        rows.append((name, r))

    print("\n=== cortex small → big: same Column, more of it (recipe: geometric-mean pool + temp-0.5 sampling) ===")
    print(f"    {'config':<24}{'test bpc':>10}{'overfit':>9}{'real-word%':>12}{'phrase-coh%':>13}")
    for name, r in rows:
        print(f"    {name:<24}{r['test_bpc']:>10.3f}{r['test_bpc']-r['train_bpc']:>+9.3f}"
              f"{r['validg']*100:>12.1f}{r['phrase']*100:>13.1f}")
    print(f"    {'(real held-out text)':<24}{'—':>10}{'—':>9}"
          f"{M.text_validity(test[400_000:700_000], vocab)*100:>12.1f}"
          f"{M.phrase_coherence(test[400_000:700_000], bigram_set)*100:>13.1f}")
    print("\n  generated samples (same seed), small → big:")
    for name, r in rows:
        print(f"    [{name.strip():<20}] {r['sample']}")

#!/usr/bin/env python3
"""Exp M — discover phrases and topics by surprise (no delimiters given).

Phrases: branching entropy over the WORD stream (Exp A's char-level signal, one level up) → multi-word units.
Topics: predictive surprise between adjacent content-word windows → topic boundaries, scored by F1 against
enwik9 article (<page>) boundaries — real ground truth. Baseline: random boundaries at the same rate.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids_pages, split_words, ids_to_str
from boundaries import phrase_cuts, topic_cuts, f1_boundaries

if __name__ == "__main__":
    np.random.seed(0)
    ids, page_char = load_ids_pages("enwik9", nbytes=60_000_000)
    spans = split_words(ids)
    words = [ids_to_str(ids[s:e]) for s, e in spans]
    starts = np.array([s for s, _ in spans])
    w2id = {}; wids = np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    vocab = len(w2id)
    page_word = np.unique(np.searchsorted(starts, page_char))   # article boundaries → word index
    page_word = page_word[(page_word > 0) & (page_word < len(words))]
    print(f"{len(words):,} words, vocab {vocab:,}, {len(page_word):,} article boundaries "
          f"(avg {len(words)//max(len(page_word),1)} words/article)\n")

    # ── phrases: discovered multi-word units (qualitative) ──
    spans_ph = phrase_cuts(wids, vocab, target_rate=0.45)
    from collections import Counter
    phr = Counter(" ".join(words[s:e]) for s, e in spans_ph if e - s >= 2)
    print("=== discovered PHRASES (top recurring multi-word units, branching-entropy cuts) ===")
    for p, c in phr.most_common(25): print(f"    {c:>5}  {p}")

    # ── topics: surprise between content-word windows vs article boundaries ──
    stop = set(np.argsort(np.bincount(wids))[::-1][:100].tolist())   # 100 most frequent ids = stopwords
    keep = np.array([w not in stop for w in wids])
    content = wids[keep]; content_to_word = np.nonzero(keep)[0]      # content idx → word idx
    peaks, gaps, surprise = topic_cuts(content, window=120, smooth=5)
    pred_word = content_to_word[np.clip(peaks, 0, len(content_to_word) - 1)]
    TOL = 30
    p, r, f = f1_boundaries(pred_word, page_word, TOL)
    rnd = np.sort(np.random.choice(len(words), size=len(pred_word), replace=False))
    rp, rr, rf = f1_boundaries(rnd, page_word, TOL)
    print(f"\n=== TOPIC boundaries vs article truth (tol ±{TOL} words) ===")
    print(f"    detected {len(pred_word):,} boundaries")
    print(f"    surprise (TextTiling):  precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}")
    print(f"    random baseline:        precision {rp:.3f}  recall {rr:.3f}  F1 {rf:.3f}")
    print(f"    → lift over random: {f/max(rf,1e-9):.1f}×")

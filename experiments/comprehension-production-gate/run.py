#!/usr/bin/env python3
"""Exp AY — a two-threshold comprehension/production gate: the C>P lag organ (M17).

The cognitive target. A child UNDERSTANDS a word weeks-to-months before it will SAY it — comprehension
runs ahead of production through development. The bet: this is ONE leaky binding-count read at TWO
operating points. COMPREHENSION is a cheap recognition read — a word is understood once ANY incoming
binding (a context->word count) clears a LOW threshold. PRODUCTION is an expensive generation read used
by act() — a word may be emitted in a context only if it is that context's ARGMAX and the context's NARS
truth f·c (AB's hit/miss split) clears a HIGH bar. The lag between the two onsets falls out of the
threshold gap, and should WIDEN for words with many competitors (high AO fan: many forms fighting to be
the argmax of a shared context) — "understands but won't say it yet," worse for confusable words.

The test (the axis this idea can win on):
  Q1  Does a C-before-P lag EXIST (median lag > 0 in stream-time bins)? Single-threshold acquisition
      (produce == comprehend) is the baseline; it predicts exactly zero lag.
  Q2  Does the lag WIDEN with competitor density (fan)? Spearman(lag, fan) > 0 across two seeds is the
      mechanism's signature. THIS is the kill axis — absent or non-widening across seeds => negative.
  Q3  Robustness — does the lag appear across a sweep of (low, high) operating points, not just one
      lucky pair? (FRAGILE: run the variations before judging.)

Corpus. The spec names CHILDES-CDS; not in data/. SUBSTITUTED: a text8 slice as the input stream
(generic, not child-directed — noted in RESULTS). Watch-set = a CDI-like band of mid-frequency content
WORD ids (skip the ultra-frequent function words, skip the hapax tail). Tokens are WORDS (a word = the
production unit). Single online streaming pass; no gradients / k-means / SVD; bounded memory. Seeds 0,1.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus, cpgate

NBYTES = 6_000_000        # text8 slice (~6 MB chars -> ~1.1M words); small/fast first pass
N_BINS = 40               # stream-time bins (onset resolution)
WATCH_N = 300             # CDI-like watch-set size
LOW = 2                   # comprehension threshold: one repeated binding = "recognized"
HIGH = 0.6                # production bar on f·c
SEEDS = (0, 1)


def word_stream(ids):
    """char-id stream -> word-id sequence. A word = the chars between spaces; map each distinct word
    string to a dense id. Returns (word_ids int32, id->string list)."""
    spans = corpus.split_words(ids)
    s2i = {}; vocab = []; out = np.empty(len(spans), np.int32)
    for k, (a, b) in enumerate(spans):
        w = corpus.ids_to_str(ids[a:b])
        j = s2i.get(w)
        if j is None:
            j = len(vocab); s2i[w] = j; vocab.append(w)
        out[k] = j
    return out, vocab


def run_one(words, vocab, watch, low, high, single, n_bins):
    g = cpgate.CPGate(len(vocab), watch, low=low, high=high, n_bins=n_bins,
                      single_threshold=single)
    g.run(None, words)
    return g


def main():
    t0 = time.time()
    print(f"loading text8 slice ({NBYTES:,} bytes)…")
    ids = corpus.load_ids("text8", nbytes=NBYTES)
    words, vocab = word_stream(ids)
    print(f"  {len(ids):,} chars -> {len(words):,} words, {len(vocab):,} distinct  "
          f"({time.time()-t0:.1f}s)")

    # CDI-like watch-set: a fixed band of mid-frequency content words (same set across conditions).
    watch = cpgate.build_watch_set(words, vocab, n_top=WATCH_N, lo_rank=20, hi_rank=900)
    print(f"  watch-set: {len(watch)} mid-frequency words "
          f"(e.g. {', '.join(vocab[w] for w in watch[:8])}…)\n")

    # ── Q1+Q2: two-threshold gate vs single-threshold baseline, across two seeds (seed only changes
    # the stream slice offset, to test robustness; the gate itself is deterministic) ──
    print("=== Q1/Q2: C>P lag and its widening with competitor density ===")
    print(f"  {'cond':>18} | {'seed':>4} | {'#both':>5} | {'med lag':>7} | "
          f"{'mean lag':>8} | {'>0 frac':>7} | {'rho(lag,fan)':>12}")
    summary = {}
    for seed in SEEDS:
        off = seed * 200_000
        w = words[off:] if off < len(words) - 50_000 else words
        for name, single in [("two-threshold", False), ("single-thresh", True)]:
            g = run_one(w, vocab, watch, LOW, HIGH, single, N_BINS)
            rows = g.lags()
            if not rows:
                print(f"  {name:>18} | {seed:>4} | {0:>5} | (no words reached both onsets)")
                continue
            lags = np.array([r[3] for r in rows]); fans = np.array([r[4] for r in rows])
            rho, nn = cpgate.spearman(lags, fans)
            medl = float(np.median(lags)); meanl = float(lags.mean())
            posf = float((lags > 0).mean())
            summary[(name, seed)] = (len(rows), medl, meanl, posf, rho)
            print(f"  {name:>18} | {seed:>4} | {len(rows):>5} | {medl:>7.1f} | "
                  f"{meanl:>8.2f} | {posf:>7.2%} | {rho:>12.3f}")
    print()

    # ── lag vs fan, bucketed (the widening, shown directly) — two-threshold, seed 0 ──
    print("=== lag vs competitor-density buckets (two-threshold, seed 0) ===")
    g0 = run_one(words, vocab, watch, LOW, HIGH, False, N_BINS)
    rows = g0.lags()
    if rows:
        lags = np.array([r[3] for r in rows]); fans = np.array([r[4] for r in rows])
        qs = np.quantile(fans, [0, 0.25, 0.5, 0.75, 1.0])
        print(f"  {'fan bucket':>16} | {'n':>4} | {'mean lag':>8} | {'med lag':>7}")
        edges = np.unique(qs)
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            sel = (fans >= lo) & (fans <= hi if i == len(edges) - 2 else fans < hi)
            if sel.sum():
                print(f"  [{lo:>5.0f},{hi:>5.0f}] | {int(sel.sum()):>4} | "
                      f"{lags[sel].mean():>8.2f} | {np.median(lags[sel]):>7.1f}")
        rho, nn = cpgate.spearman(lags, fans)
        print(f"  overall Spearman(lag, fan) = {rho:.3f}  (n={nn})")
    print()

    # ── Q3: robustness sweep over operating points (FRAGILE — many variations before judging) ──
    print("=== Q3: operating-point sweep (does the lag survive across thresholds?) ===")
    print(f"  {'low':>4} {'high':>5} | {'#both':>5} | {'med lag':>7} | {'>0 frac':>7} | {'rho':>6}")
    n_pos = 0; n_total = 0
    for low in (1, 2, 3):
        for high in (0.4, 0.5, 0.6, 0.7, 0.8):
            g = run_one(words, vocab, watch, low, high, False, N_BINS)
            rows = g.lags()
            if not rows:
                print(f"  {low:>4} {high:>5} | {0:>5} | --"); continue
            lags = np.array([r[3] for r in rows]); fans = np.array([r[4] for r in rows])
            rho, _ = cpgate.spearman(lags, fans)
            medl = float(np.median(lags)); posf = float((lags > 0).mean())
            n_total += 1
            if medl > 0 and rho > 0:
                n_pos += 1
            print(f"  {low:>4} {high:>5} | {len(rows):>5} | {medl:>7.1f} | "
                  f"{posf:>7.2%} | {rho:>6.3f}")
    print(f"\n  operating points with (med-lag>0 AND rho>0): {n_pos}/{n_total}")
    print(f"\ndone ({time.time()-t0:.1f}s)")
    return summary


if __name__ == "__main__":
    main()

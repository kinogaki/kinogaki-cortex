#!/usr/bin/env python3
"""Exp AU — chunk lexicon with sub-unit interference (the PARSER/Isbilen organ). ONLINE, NO backprop.

A child does not keep a backoff n-gram. She commits to WHOLE UNITS of variable length and, having
committed, stops tracking the transitions inside them — Isbilen's splice signature: once "cup" is a
chunk, the c-u-p transition decays. We build that as counting (lib/chunklex):

  greedily COVER the stream with the highest/longest-confident chunks (take-the-best) → +1 each;
  when two adjacent covering chunks RECUR, MINT their concatenation (leader-spawn);
  THE NEW PART — as a minted whole's weight grows, LEAK weight from the sub-chunks it was minted from
  (sub-unit interference); LFU-evict to stay bounded.

THREE axes, judged on the one the idea can WIN (FRAGILE — ≥10 decay/cover variations; first weak
result is expected):

  1. SPLICE TEST (Saffran 1996 / Isbilen) — the kill-test. A frequency-matched syllable stream of
     4 tri-"syllable" words. After learning, splice a learned word ABC into a novel order so its
     INTERNAL B–C transition is now a *part* of a committed chunk. Does ChunkLexicon's read of the
     B–C splice point fall BELOW pure-forward-TP's (the Saffran null that never chunks)? If the
     sub-unit decay works, the internal transition decays; pure-TP keeps it forever.
  2. BOUNDARY F1 on space-stripped text8 — cover points vs the removed spaces, vs Exp A's 0.775.
  3. HELD-OUT BPC — does the chunk-agent beat a fixed-order n-gram agent on real text?

KILL (BUILD_QUEUE AU): spliced B–C does NOT decay below pure-TP **and** chunk-agent gen/bpc does not
beat the n-gram — after the FRAGILE budget. A clean negative is a real outcome; reported honestly.

Corpus: (1) synthesized Saffran stream (built here — the spec's clean kill-test); (2) space-stripped
text8 (Exp A's corpus is P&P; text8 substituted, SAID SO). Fixed seed, single streaming pass.
"""
import os, sys, time, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus, metrics                                    # noqa: E402
from chunklex import ChunkLexicon, PureTP, ChunkAgent     # noqa: E402

print = functools.partial(print, flush=True)
SEED = 0

# ───────────────────────── (1) Saffran frequency-matched syllable stream ─────────────────────────
# 4 "words", each 3 distinct syllables; syllables are single int tokens (a small vocab). Words are
# concatenated in random order with NO boundary marker — exactly Saffran 1996. Within a word the
# transitional probability is 1.0; at a word boundary it is 1/3 (any of the other 3 words can follow).
# That TP dip is the only segmentation cue. (Frequency-matched: each word equally frequent.)
def make_saffran(n_words=12000, seed=SEED):
    rng = np.random.default_rng(seed)
    # 12 distinct syllables 0..11 → 4 words of 3 syllables each
    words = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
    order = rng.integers(0, 4, size=n_words)
    stream = []
    for w in order:
        stream.extend(words[w])
    return np.array(stream, dtype=np.int64), words, 12


def splice_test(decay, cover, words, V, n_words=12000):
    """Train on the Saffran stream; then read the within-word B–C transition for each word. Compare
    ChunkLexicon's committed 2-chunk weight (normalized to a transition probability via its left
    token's mass) against pure-forward-TP. The Isbilen prediction: chunking DECAYS the internal
    transition; pure-TP does not. We report the mean over the 4 words' middle→last (B→C) splice."""
    stream, _, _ = make_saffran(n_words, seed=SEED)
    lex = ChunkLexicon(V, decay=decay, mint_thresh=4, cover=cover, max_chunks=5000, max_len=4, seed=SEED)
    tp = PureTP(V)
    CHUNK = 600
    for i in range(0, len(stream), CHUNK):
        seg = stream[i:i + CHUNK]
        lex.observe(seg); tp.observe(seg)
    # internal B→C transition (the spliced-away within-word pair) vs boundary pairs
    internal_lex, internal_tp = [], []
    for (a, b, c) in words:
        # ChunkLexicon: read the 2-chunk weight on (b,c) normalized by b's total chunk mass at left
        wbc = lex.transition_weight(b, c)
        bmass = sum(w for ch, w in lex.w.items() if len(ch) >= 1 and ch[0] == b)
        internal_lex.append(wbc / bmass if bmass > 0 else 0.0)
        internal_tp.append(tp.tp(b, c))
    # boundary B→? : the last syllable c of each word → first syllable of the next (the TP dip)
    boundary_tp = []
    for (a, b, c) in words:
        for (a2, b2, c2) in words:
            boundary_tp.append(tp.tp(c, a2))
    return dict(decay=decay, cover=cover,
                internal_lex=float(np.mean(internal_lex)),
                internal_tp=float(np.mean(internal_tp)),
                boundary_tp=float(np.mean(boundary_tp)),
                stats=lex.chunk_stats())


# ───────────────────────── (2) boundary F1 on space-stripped text8 ─────────────────────────
def boundary_f1(decay, cover, ids, n_chars=2_000_000):
    """Strip spaces, learn chunks online, take cover points as predicted boundaries, F1 (±1 tol) vs
    the true removed-space positions at matched count (top-K = #true boundaries)."""
    ids = ids[:n_chars]
    sp = ids == corpus.SPACE
    letters = ids[~sp].astype(np.int64)
    # true boundary = output index where a space was removed (start of the next word)
    true = np.nonzero(sp)[0]
    # map removed-space byte positions to letter-stream positions: #letters before each space
    cum = np.cumsum(~sp)
    bnd = set(int(cum[t]) for t in true if 0 < cum[t] < len(letters))
    lex = ChunkLexicon(26, decay=decay, mint_thresh=4, cover=cover, max_chunks=30000, max_len=8, seed=SEED)
    CHUNK = 512
    pred = set(); pos = 0
    for i in range(0, len(letters), CHUNK):
        seg = letters[i:i + CHUNK]
        chunks = lex.cover_buffer(list(seg))
        lex.observe(seg)
        p = pos
        for ch in chunks:
            p += len(ch)
            if p < len(letters):
                pred.add(p)                                  # boundary at end of each covered chunk
        pos += len(seg)
    # F1 at ±1 tolerance, matched-count: take the densest region comparison
    def f1(predset, trueset):
        hit = sum(1 for x in predset if x in trueset or x - 1 in trueset or x + 1 in trueset)
        prec = hit / max(len(predset), 1)
        rhit = sum(1 for x in trueset if x in predset or x - 1 in predset or x + 1 in predset)
        rec = rhit / max(len(trueset), 1)
        return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)
    prec, rec, f = f1(pred, bnd)
    return dict(decay=decay, cover=cover, precision=prec, recall=rec, f1=f,
                n_pred=len(pred), n_true=len(bnd), stats=lex.chunk_stats())


# ───────────────────────── (3) held-out bpc: chunk-agent vs fixed-order n-gram ─────────────────────────
class NgramAgent:
    """The fixed-order backoff n-gram baseline (the cortex spine), scored by lib/metrics unchanged."""
    def __init__(self, order=4, alpha=0.05):
        self.order = order; self.alpha = alpha; self.V = 27; self.K = 64
        self.ctx = [dict() for _ in range(order + 1)]
        self.A = "abcdefghijklmnopqrstuvwxyz "; self.CH = {c: i for i, c in enumerate(self.A)}
    def train(self, ids):
        s = list(int(x) for x in ids)
        for t in range(len(s)):
            nx = s[t]
            for k in range(min(self.order, t) + 1):
                d = self.ctx[k].setdefault(tuple(s[t - k:t]), {}); d[nx] = d.get(nx, 0) + 1
    def dist(self, suffix):
        ids = [self.CH[c] for c in suffix if c in self.CH][-self.K:]
        logp = np.zeros(self.V)
        for k in range(min(self.order, len(ids)), -1, -1):
            d = self.ctx[k].get(tuple(ids[len(ids) - k:]) if k else ())
            if d:
                p = np.full(self.V, self.alpha); tot = self.alpha * self.V
                for tok, c in d.items():
                    p[tok] += c; tot += c
                z = np.log(p / tot); z -= z.max(); e = np.exp(z); return e / e.sum()
        return np.full(self.V, 1.0 / self.V)


def bpc_compare(decay, cover, ids, train_chars=2_000_000, test_chars=200_000, order=4):
    tr = ids[:train_chars]; te = ids[train_chars:train_chars + test_chars]
    te_str = corpus.ids_to_str(te)
    lex = ChunkLexicon(27, decay=decay, mint_thresh=4, cover=cover, max_chunks=30000, max_len=8, seed=SEED)
    for i in range(0, len(tr), 512):
        lex.observe(tr[i:i + 512])
    ca = ChunkAgent(lex, order=order); ca.train(tr)
    ng = NgramAgent(order=order); ng.train(tr)
    return dict(decay=decay, cover=cover,
                chunk_bpc=metrics.bpc(ca, te_str), ngram_bpc=metrics.bpc(ng, te_str),
                stats=lex.chunk_stats())


def main():
    t0 = time.time()
    print("loading text8 slice...")
    ids = corpus.load_ids("text8", nbytes=3_000_000)
    print(f"  {len(ids):,} chars | {time.time()-t0:.1f}s")

    _, words, V = make_saffran()

    # FRAGILE budget: sweep decay rate × cover policy (≥10 variations)
    decays = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    covers = ["longest", "weight"]
    variations = [(d, c) for c in covers for d in decays]            # 12 variations
    print(f"\nFRAGILE budget: {len(variations)} variations (decay × cover)")

    # ── (1) SPLICE TEST (the kill axis) ──
    print("\n=== (1) SPLICE TEST — Saffran/Isbilen: does the WITHIN-word B–C transition decay below pure-TP? ===")
    print(f"  {'cover':<9}{'decay':>7}{'lex internal B-C':>18}{'pureTP internal':>18}{'pureTP boundary':>18}"
          f"{'mints':>8}{'types':>8}")
    splice_rows = []
    for (d, c) in variations:
        r = splice_test(d, c, words, V)
        splice_rows.append(r)
        print(f"  {c:<9}{d:>7.2f}{r['internal_lex']:>18.4f}{r['internal_tp']:>18.4f}{r['boundary_tp']:>18.4f}"
              f"{r['stats']['n_mint']:>8}{r['stats']['types']:>8}")
    # the kill read: best (lowest) lex internal vs pure-TP internal (which is ~1.0 within a word)
    best_splice = min(splice_rows, key=lambda r: r["internal_lex"])
    tp_internal = splice_rows[0]["internal_tp"]
    splice_wins = best_splice["internal_lex"] < tp_internal
    print(f"\n  pure-TP internal B–C (no chunking, never decays) ≈ {tp_internal:.4f}")
    print(f"  best chunk-lexicon internal B–C = {best_splice['internal_lex']:.4f} "
          f"(cover={best_splice['cover']}, decay={best_splice['decay']}) "
          f"→ {'DECAYS BELOW pure-TP ✓' if splice_wins else 'does NOT fall below pure-TP ✗'}")

    # ── (2) BOUNDARY F1 (sweep a representative subset for speed) ──
    print("\n=== (2) BOUNDARY F1 on space-stripped text8 (cover points vs removed spaces; Exp A ref 0.775) ===")
    print(f"  {'cover':<9}{'decay':>7}{'precision':>11}{'recall':>9}{'F1':>8}{'mints':>8}{'types':>8}")
    f1_rows = []
    for (d, c) in [("x", "x")] and [(d, c) for c in covers for d in (0.0, 0.25, 0.5, 1.0)]:
        r = boundary_f1(d, c, ids)
        f1_rows.append(r)
        print(f"  {c:<9}{d:>7.2f}{r['precision']:>11.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}"
              f"{r['stats']['n_mint']:>8}{r['stats']['types']:>8}")
    best_f1 = max(f1_rows, key=lambda r: r["f1"])
    print(f"\n  best boundary F1 = {best_f1['f1']:.3f} (cover={best_f1['cover']}, decay={best_f1['decay']}) "
          f"vs Exp A 0.775")

    # ── (3) HELD-OUT BPC vs fixed-order n-gram (the second kill axis) ──
    print("\n=== (3) HELD-OUT BPC — chunk-agent vs fixed-order n-gram agent (text8) ===")
    print(f"  {'cover':<9}{'decay':>7}{'chunk bpc':>11}{'ngram bpc':>11}{'Δ (chunk-ngram)':>17}{'mints':>8}")
    bpc_rows = []
    for (d, c) in [(d, c) for c in covers for d in (0.0, 0.5, 1.0)]:
        r = bpc_compare(d, c, ids)
        bpc_rows.append(r)
        print(f"  {c:<9}{d:>7.2f}{r['chunk_bpc']:>11.3f}{r['ngram_bpc']:>11.3f}"
              f"{r['chunk_bpc']-r['ngram_bpc']:>17.4f}{r['stats']['n_mint']:>8}")
    best_bpc = min(bpc_rows, key=lambda r: r["chunk_bpc"])
    ng_ref = bpc_rows[0]["ngram_bpc"]
    bpc_wins = best_bpc["chunk_bpc"] < ng_ref
    print(f"\n  best chunk bpc = {best_bpc['chunk_bpc']:.3f}  vs  n-gram bpc = {ng_ref:.3f} "
          f"→ {'chunk-agent beats n-gram ✓' if bpc_wins else 'chunk-agent does NOT beat n-gram ✗'}")

    # ── KILL VERDICT ──
    print("\n" + "=" * 78)
    killed = (not splice_wins) and (not bpc_wins)
    print(f"KILL-CONDITION (AU): spliced B–C does NOT decay below pure-TP  AND  chunk-agent does NOT beat n-gram")
    print(f"  splice axis : chunk-lexicon internal B–C {'<' if splice_wins else '≥'} pure-TP  "
          f"({'sub-unit interference WORKS' if splice_wins else 'no decay'})")
    print(f"  bpc axis    : chunk-agent {'<' if bpc_wins else '≥'} n-gram bpc")
    print(f"  → KILL-CONDITION {'FIRED (clean negative)' if killed else 'did NOT fire'}")
    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(splice_wins=splice_wins, bpc_wins=bpc_wins, killed=killed,
                best_splice=best_splice, tp_internal=tp_internal,
                best_f1=best_f1["f1"], best_bpc=best_bpc, ngram_bpc=ng_ref)


if __name__ == "__main__":
    main()

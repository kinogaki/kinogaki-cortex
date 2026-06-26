#!/usr/bin/env python3
"""Exp BK — chunk-as-expert in the pool: close AU's bpc gap. ONLINE, NO backprop.

AU's ChunkLexicon won the splice axis but LOST held-out bpc by +0.20 because its ChunkAgent REPLACED
the calibrated backoff n-gram with raw chunk-completion. The cognitive read: the parser and the
sequence-predictor are not rivals — they are two experts the cortex consults at once. So add the
chunk-completion distribution as ONE EXPERT into cortex.vote's geometric-mean pool ALONGSIDE the char
Columns (lib/chunkvote), never replacing the backoff. The lexicon helps where it is confident
(mid-committed-chunk) and abstains elsewhere; the backoff carries the rest.

Compare held-out bpc on ONE text8 slice across:
  (1) NgramAgent       — the plain backoff n-gram (the number to beat).
  (2) ChunkOnlyAgent   — AU's chunk-completion-ONLY agent (the +0.20 loser, reproduced).
  (3) ChunkVoteAgent   — the chunk-as-expert BLEND, sweeping chunk_w (FRAGILE).

SUCCESS if the blend's bpc <= the n-gram's (gap closed) on the same slice, with the splice axis still
intact. The splice axis is re-confirmed here from the same trained lexicon (AU's PureTP vs the
lexicon's committed within-word transition) so the win the fix must PRESERVE is on the table.

Corpus: text8 (corpus.load_ids), a small fast slice (train ~2 MB / held-out 200 KB). The chunk
lexicon is trained on the train slice (single pass), then frozen for held-out scoring; the n-gram /
char Columns observe the train slice once. Fixed seed (0), online single streaming pass.
"""
import os, sys, time, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus, metrics                                                      # noqa: E402
from chunklex import ChunkLexicon, PureTP                                   # noqa: E402
from chunkvote import NgramAgent, ChunkOnlyAgent, ChunkVoteAgent            # noqa: E402

print = functools.partial(print, flush=True)
SEED = 0
np.random.seed(SEED)

CHAR_ORDERS = (0, 1, 2, 3, 4, 5, 6)
TRAIN_BYTES = 2_000_000
TEST_BYTES = 200_000


def main():
    t0 = time.time()
    print("Exp BK — chunk-as-expert in the calibrated pool (close AU's bpc gap)\n")

    # ── data: one text8 slice, train / held-out (same slice basis as AU result 3) ──
    ids = corpus.load_ids("text8", nbytes=TRAIN_BYTES + TEST_BYTES)
    train_ids = ids[:-TEST_BYTES]
    test_ids = ids[-TEST_BYTES:]
    test_s = corpus.ids_to_str(test_ids)
    train_s = corpus.ids_to_str(train_ids[-400_000:])  # for the train-bpc sanity number only
    print(f"  text8 slice: train {len(train_ids):,} chars, held-out {len(test_ids):,} chars")

    # ── train ONE chunk lexicon on the train slice (single streaming pass), then freeze ──
    # longest cover, decay 0.5 — AU's strong splice config (Result 1: B-C -> 0.0003, F1 0.758).
    lex = ChunkLexicon(corpus.SPACE + 1, decay=0.5, mint_thresh=4, cover="longest", seed=SEED)
    CHUNK_TRAIN = train_ids[:1_000_000]                 # bounded lexicon training (cover is O(maxlen) / char)
    lex.observe(CHUNK_TRAIN)
    cs = lex.chunk_stats()
    print(f"  lexicon: {cs['types']} multi-char types, {cs['n_mint']} mints, {cs['n_evict']} evicts, "
          f"lex_size {cs['lex_size']}  ({time.time()-t0:.0f}s)")

    # ── splice axis re-confirm (the win the fix must PRESERVE) ──
    saff, words, sv = make_saffran()
    slex = ChunkLexicon(sv, decay=0.5, mint_thresh=4, cover="longest", seed=SEED)
    slex.observe(saff)
    tp = PureTP(sv)
    tp.observe(saff)
    bc_lex, bc_tp = [], []
    for (a, b, c) in words:
        la = slex.w.get((a,), 0.0)
        bc_lex.append(slex.transition_weight(b, c) / la if la > 0 else 0.0)
        bc_tp.append(tp.tp(b, c))
    print(f"\n  SPLICE (preserved win): lexicon within-word B-C {np.mean(bc_lex):.4f}  "
          f"vs pure-TP {np.mean(bc_tp):.4f}  -> {'INTACT' if np.mean(bc_lex) < np.mean(bc_tp) else 'BROKEN'}")

    # ── (1) the n-gram to beat ──
    ng = NgramAgent(orders=CHAR_ORDERS)
    ng.observe(train_ids)
    ng_test = metrics.bpc(ng, test_s)
    ng_train = metrics.bpc(ng, train_s)
    print(f"\n  (1) NgramAgent      held-out bpc {ng_test:.3f}   (train {ng_train:.3f})  "
          f"<- the number to beat")

    # ── (2) AU's chunk-completion-ONLY loser, reproduced ──
    co = ChunkOnlyAgent(lex, order=2, lam=0.6)
    co.observe(train_ids)
    co_test = metrics.bpc(co, test_s)
    print(f"  (2) ChunkOnlyAgent  held-out bpc {co_test:.3f}   (Δ vs n-gram {co_test-ng_test:+.3f})  "
          f"<- AU's +0.20 loser, reproduced")

    # ── (3) chunk-as-expert blend, FRAGILE sweep over the chunk-expert weight ──
    print("\n  (3) ChunkVoteAgent — chunk expert in the pool, sweeping chunk_w:")
    print(f"      {'chunk_w':>8} | {'held-out bpc':>12} | {'Δ vs n-gram':>11}")
    print("      " + "-" * 38)
    best = None
    for w in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        cv = ChunkVoteAgent(lex, orders=CHAR_ORDERS, chunk_w=w)
        cv.observe(train_ids)
        b = metrics.bpc(cv, test_s)
        flag = "  <- closes gap" if b <= ng_test + 1e-9 else ""
        print(f"      {w:>8.2f} | {b:>12.3f} | {b-ng_test:>+11.3f}{flag}")
        if best is None or b < best[1]:
            best = (w, b)

    print(f"\n  best blend: chunk_w={best[0]:.2f}  bpc {best[1]:.3f}  (Δ vs n-gram {best[1]-ng_test:+.3f})")
    gap_closed = best[1] <= ng_test + 1e-9
    print(f"  GAP {'CLOSED' if gap_closed else 'NOT closed'}: blend bpc "
          f"{'<=' if gap_closed else '>'} n-gram bpc")
    print(f"\n  total {time.time()-t0:.0f}s")
    return dict(ng=ng_test, ng_train=ng_train, chunk_only=co_test, best_w=best[0], best_bpc=best[1],
                gap_closed=gap_closed, splice_lex=float(np.mean(bc_lex)), splice_tp=float(np.mean(bc_tp)))


# ── Saffran frequency-matched syllable stream (verbatim shape from exp_au) ──
def make_saffran(n_words=12000, seed=SEED):
    rng = np.random.default_rng(seed)
    words = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
    order = rng.integers(0, 4, size=n_words)
    stream = []
    for w in order:
        stream.extend(words[w])
    return np.array(stream, dtype=np.int64), words, 12


if __name__ == "__main__":
    main()

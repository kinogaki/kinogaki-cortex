#!/usr/bin/env python3
"""Exp AP — permutation-bound n-grams + FlyHash addressing. ONLINE, count-based, NO backprop.

The phrase level suffers from SPARSITY: a phrase seen once (or never in this exact form) has a near-zero literal
count, even when many SIMILAR phrases were common. VSA fix (Kanerva HD-computing; Joshi/Kanerva HD text;
Dasgupta FlyHash): encode each n-gram as an ORDER-PRESERVING hypervector address by binding shifted atom vectors
(addr = rho^2(A) (x) rho(B) (x) C; rho = cyclic shift carries order, bind = elementwise product), then FlyHash
that address (sparse expansive random projection + top-k WTA) so SIMILAR phrases get OVERLAPPING sparse addresses
and we COUNT next-token at those shared buckets. Similar phrases pool their counts (generalization); rho keeps
"abc" != "cab" (not a bag).

KEY TEST.
  (a) GENERALIZE — on phrases held out from EXACT-form training (their literal n-gram count is 0), does
      perm-bind+FlyHash predict the true next token better than the literal phrase-count baseline? (the sparsity
      win). Compare also to a bag-of-context control (pools, but order-blind).
  (b) ORDER — scramble the context word order at test. A bag is invariant (no degradation). The VSA model and
      the literal n-gram should DEGRADE (they encode order). If FlyHash crosstalk has destroyed order, the VSA
      model won't degrade — that would be an honest negative.

Honest if hashing crosstalk hurts more than pooling helps. Corpus: text8. Fixed seed, single streaming pass.
"""
import os, sys, time, functools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from permngram import (atom_vectors, FlyHash, LiteralNgram, BagContext, PermFlyNgram,
                       make_windows, _pack, perplexity)

print = functools.partial(print, flush=True)

# ── config ──
TRAIN_BYTES = 14_000_000
N           = 10_000      # top-N words get a dense id + an atom vector; rest OOV (-1)
NGRAM       = 3           # phrase length (context = 2 preceding words -> predict the 3rd... actually n words ctx)
CTX         = 3           # context words bound into the address (a 3-word phrase -> next word)
D           = 256         # hypervector dimension (atoms + bound address)
M           = 4000        # FlyHash buckets (expansive: M >> D)
K           = 16          # top-k winners per address (the sparse code size)
S           = 12          # input dims sampled per bucket (sparse projection fan-in)
ALPHA       = 0.1         # smoothing
HOLDOUT_FRAC = 0.20       # fraction of DISTINCT context-phrase STRINGS held out of training
N_PROBE     = 60_000      # eval probes
SEED        = 0


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN_BYTES)
    spans = split_words(ids)
    words = [ids_to_str(ids[s:e]) for s, e in spans]
    w2id, wids = {}, np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    counts_g = np.bincount(wids, minlength=len(w2id))
    top = np.argsort(counts_g)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    seq = remap[wids]                                                        # dense top-id stream (-1=OOV)
    topword = [id2word[t] for t in top]
    print(f"{len(words):,} words, {len(w2id):,} types | top-N={N} | load+map {time.time()-t0:.1f}s")

    # ── windows: CTX context words -> next word, no OOV ──
    ctx, nxt = make_windows(seq, CTX)
    print(f"{len(ctx):,} clean {CTX}-word windows (-> next word) | {time.time()-t0:.1f}s")

    # ── HOLD OUT distinct context-phrase STRINGS (the sparsity setup) ──
    # A held-out phrase string is REMOVED from every model's training, so the literal n-gram has count 0 for it
    # (it must floor to unigram). The VSA model can still hit buckets that SIMILAR, non-held-out phrases filled.
    rng = np.random.default_rng(SEED)
    ckey = _pack(ctx, N)
    uniq = np.unique(ckey)
    held = rng.random(len(uniq)) < HOLDOUT_FRAC
    held_set = set(uniq[held].tolist())
    is_held = np.array([k in held_set for k in ckey])
    tr = ~is_held                                                            # training = non-held windows
    print(f"holdout: {len(uniq):,} distinct context phrases, held {held.sum():,} ({held.mean()*100:.0f}%) "
          f"-> {is_held.sum():,} held windows never trained in-form | {time.time()-t0:.1f}s")

    ctx_tr, nxt_tr = ctx[tr], nxt[tr]

    # ── fixed random VSA structure ──
    atoms = atom_vectors(N, D, seed=SEED)
    fly = FlyHash(D, M, K, s=S, seed=SEED + 1)

    # ── fit all three on the SAME training windows (one pass each) ──
    t1 = time.time()
    lit = LiteralNgram(N, CTX, alpha=ALPHA).fit(ctx_tr, nxt_tr)
    bag = BagContext(N, CTX, alpha=ALPHA).fit(ctx_tr, nxt_tr)
    pfn = PermFlyNgram(atoms, fly, N, CTX, alpha=ALPHA).fit(ctx_tr, nxt_tr)
    print(f"fit literal/bag/permfly in {time.time()-t1:.1f}s | "
          f"literal phrase keys={len(lit.table):,} | fly buckets used={int((pfn.bucket.sum(1)>0).sum()):,}/{M} "
          f"| {time.time()-t0:.1f}s")

    # ── eval probe sets ──
    rng2 = np.random.default_rng(SEED + 7)
    held_idx = np.nonzero(is_held)[0]
    seen_idx = np.nonzero(tr)[0]
    hp = held_idx[rng2.permutation(len(held_idx))[:N_PROBE]]
    sp = seen_idx[rng2.permutation(len(seen_idx))[:N_PROBE]]

    def eval_set(idx, scramble=False):
        c = ctx[idx].copy()
        if scramble:
            # permute the CTX columns per row (destroys order; keeps the multiset)
            for r in range(len(c)):
                c[r] = c[r, rng2.permutation(CTX)]
        y = nxt[idx]
        # literal + bag are per-row (dict lookups); permfly is batched
        p_lit = np.array([lit.prob(c[r])[y[r]] for r in range(len(c))])
        p_bag = np.array([bag.prob(c[r])[y[r]] for r in range(len(c))])
        P_pfn = pfn.prob_batch(c)
        p_pfn = P_pfn[np.arange(len(c)), y]
        return p_lit, p_bag, p_pfn

    print("\n=== (a) GENERALIZATION — held-out phrases (literal count = 0; must floor) ===")
    p_lit, p_bag, p_pfn = eval_set(hp)
    print(f"  literal n-gram   ppl {perplexity(p_lit):10.1f}")
    print(f"  bag-of-context   ppl {perplexity(p_bag):10.1f}   (order-blind control)")
    print(f"  perm+FlyHash     ppl {perplexity(p_pfn):10.1f}   <- VSA model")
    print(f"  permfly beats literal on {np.mean(p_pfn > p_lit)*100:5.1f}% of held probes")
    print(f"  permfly beats bag     on {np.mean(p_pfn > p_bag)*100:5.1f}% of held probes")
    held_lit, held_bag, held_pfn = perplexity(p_lit), perplexity(p_bag), perplexity(p_pfn)

    print("\n=== sanity — SEEN phrases (in training; literal should be strong here) ===")
    s_lit, s_bag, s_pfn = eval_set(sp)
    print(f"  literal n-gram   ppl {perplexity(s_lit):10.1f}")
    print(f"  bag-of-context   ppl {perplexity(s_bag):10.1f}")
    print(f"  perm+FlyHash     ppl {perplexity(s_pfn):10.1f}")
    seen_lit, seen_bag, seen_pfn = perplexity(s_lit), perplexity(s_bag), perplexity(s_pfn)

    print("\n=== (b) ORDER-SENSITIVITY — scramble context word order at test (SEEN phrases) ===")
    sc_lit, sc_bag, sc_pfn = eval_set(sp, scramble=True)
    o_lit, o_bag, o_pfn = perplexity(sc_lit), perplexity(sc_bag), perplexity(sc_pfn)
    print(f"  literal n-gram   ppl {seen_lit:10.1f} -> {o_lit:10.1f}   (x{o_lit/seen_lit:.2f}  degrade={o_lit>seen_lit})")
    print(f"  bag-of-context   ppl {seen_bag:10.1f} -> {o_bag:10.1f}   (x{o_bag/seen_bag:.2f}  should be ~x1.0)")
    print(f"  perm+FlyHash     ppl {seen_pfn:10.1f} -> {o_pfn:10.1f}   (x{o_pfn/seen_pfn:.2f}  must degrade if order kept)")

    # show some held-out phrases the VSA model recovered the true next-word for and the literal floored
    print("\n=== examples — held-out phrases where perm+FlyHash beat the floored literal ===")
    win = np.nonzero((p_pfn > p_lit) & (p_pfn > 1e-3))[0]
    shown = 0
    for j in win[rng2.permutation(len(win))]:
        r = hp[j]
        phrase = " ".join(topword[w] for w in ctx[r])
        tgt = topword[nxt[r]]
        if len(phrase) < 30:
            print(f"  '{phrase} ___' -> '{tgt}'  permfly p={p_pfn[j]:.4f}  literal p={p_lit[j]:.5f}")
            shown += 1
        if shown >= 10:
            break

    print(f"\nwhole run {time.time()-t0:.1f}s on CPU, single pass. fixed seed.")

    # stash for RESULTS
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.npz"),
             held_lit=held_lit, held_bag=held_bag, held_pfn=held_pfn,
             seen_lit=seen_lit, seen_bag=seen_bag, seen_pfn=seen_pfn,
             o_lit=o_lit, o_bag=o_bag, o_pfn=o_pfn,
             beat_lit=float(np.mean(p_pfn > p_lit)), beat_bag=float(np.mean(p_pfn > p_bag)))


if __name__ == "__main__":
    main()

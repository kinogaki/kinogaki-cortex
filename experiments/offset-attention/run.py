#!/usr/bin/env python3
"""Exp S — Offset-keyed count-attention: the count-based, no-backprop form of attention.

WORD level on text8. Build per-OFFSET count tables T_d (next word | word d steps back), weight each
offset by its information gain IG(d), and pool the per-offset experts (geometric mean) to predict the
next word. Three questions:

  (1) Does offset-attention beat fixed n-grams (bigram d=1, fixed trigram)?
  (2) KILLS BAG-OF-WORDS: a 2x2 of {offset-attn, bag} x {ordered ctx, scrambled ctx}. Offset-attn must
      DEGRADE when we shuffle the D context words; the bag must NOT (it's order-blind); and offset-attn
      must BEAT the bag on ordered context. That gap = "this is not a bag of words".
  (3) The learned IG(d) per offset — do nearer offsets carry more, and how fast does it decay?

No neural net, no GPU, no gradients — just counts and an entropy-derived weight per relative position.
"""
import os, sys, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus
from offsetattn import OffsetAttn, build_word_stream, _weighted_pool, ALPHA

D = 8                    # number of relative offsets / context width
GAMMA = 8.0              # IG sharpening: pooling weight ∝ IG(d)**GAMMA (1=flat; ->inf = pure bigram)
VOCAB = 40000            # top words kept; rest -> UNK
NBYTES = 40_000_000      # text8 prefix to read (≈7M words)
EVAL_WORDS = 120_000     # held-out eval slice (last words)
SEED = 0


def ids_array_from_dict(dd):
    """Sparse {wid:count} -> (wids, counts) numpy arrays."""
    ks = np.fromiter(dd.keys(), dtype=np.int64, count=len(dd))
    cs = np.fromiter(dd.values(), dtype=np.float64, count=len(dd))
    return ks, cs


# ---- n-gram baselines (their own count tables, evaluated the same way) ---------------------------

def bigram_tables(tr):
    big = {}
    for a, b in zip(tr[:-1], tr[1:]):
        d = big.get(int(a))
        if d is None:
            d = {}; big[int(a)] = d
        d[int(b)] = d.get(int(b), 0) + 1
    return big


def trigram_tables(tr):
    trg = {}
    for a, b, c in zip(tr[:-2], tr[1:-1], tr[2:]):
        key = (int(a), int(b))
        d = trg.get(key)
        if d is None:
            d = {}; trg[key] = d
        d[int(c)] = d.get(int(c), 0) + 1
    return trg


# ---- evaluation -------------------------------------------------------------------------------

def eval_predictor(predict_fn, stream, eval_start, M, label, scramble=False, rng=None):
    """Walk eval positions, call predict_fn(ctx_window) -> {wid:prob} or None, score top-1 + perplexity.
    Abstain (None) => uniform over M (fair, never skipped). ctx_window is the D words before position t,
    oldest..newest; if scramble, its order is shuffled per-position (set of words preserved)."""
    n_eval = 0
    correct = 0
    logloss = 0.0
    base = 1.0 / M
    t0 = time.time()
    for t in range(eval_start, len(stream)):
        ctx = stream[t - D:t].copy()
        if scramble:
            rng.shuffle(ctx)
        true = int(stream[t])
        dist = predict_fn(ctx)
        n_eval += 1
        if dist:
            p_true = dist.get(true, 0.0)
            # mass not on listed candidates is treated as smoothed uniform residual
            if p_true <= 0.0:
                p_true = base * 1e-3
            # top-1
            top = max(dist.items(), key=lambda kv: kv[1])[0]
            if top == true:
                correct += 1
        else:
            p_true = base
        logloss += -math.log(max(p_true, 1e-12))
    acc = correct / n_eval
    ppl = math.exp(logloss / n_eval)
    dt = time.time() - t0
    print(f"  {label:<34} acc={acc*100:6.2f}%  ppl={ppl:10.1f}   ({n_eval} pts, {dt:.1f}s)")
    return acc, ppl


def main():
    print(f"Exp S — offset-keyed count-attention   (D={D}, vocab={VOCAB}, {NBYTES//1_000_000}MB text8)")
    print("loading text8 ...")
    ids = corpus.load_ids("text8", NBYTES)
    spans = corpus.split_words(ids)
    stream, vocab_list, UNK = build_word_stream(ids, spans, VOCAB)
    M = UNK + 1
    print(f"words: {len(stream):,}   vocab {VOCAB}+UNK   OOV(UNK) rate {np.mean(stream==UNK):.3f}")

    eval_start = len(stream) - EVAL_WORDS
    train = stream[:eval_start]
    print(f"train {len(train):,} words   /   eval {EVAL_WORDS:,} words")

    # --- fit offset-attention (and the bag, which reuses its tables) ---
    print("fitting offset tables ...")
    t0 = time.time()
    oa = OffsetAttn(D, gamma=GAMMA).fit(train)
    oa.build_bag()
    print(f"  fitted in {time.time()-t0:.1f}s")

    print(f"\nlearned IG(d) per offset (bits) and pooling weight ∝ IG**{GAMMA:g}:")
    tot_ig = oa.ig[1:].sum()
    wn = oa.w[1:] / oa.w[1:].sum()
    for d in range(1, D + 1):
        bar = "#" * int(round(oa.ig[d] / oa.ig[1:].max() * 40))
        print(f"  d={d}: IG={oa.ig[d]:.4f} bits  (IG share {oa.ig[d]/tot_ig*100:4.1f}%, "
              f"pool wt {wn[d-1]*100:5.1f}%)  {bar}")

    # --- fixed n-gram baselines ---
    print("\nfitting n-gram baselines ...")
    big = bigram_tables(train)
    trg = trigram_tables(train)

    def predict_bigram(ctx):
        return _weighted_pool([(1.0, big[int(ctx[-1])])]) if int(ctx[-1]) in big else None

    def predict_trigram(ctx):
        key = (int(ctx[-2]), int(ctx[-1]))
        if key in trg:
            return _weighted_pool([(1.0, trg[key])])
        if int(ctx[-1]) in big:                       # back off to bigram
            return _weighted_pool([(1.0, big[int(ctx[-1])])])
        return None

    # === Q1: offset-attention vs fixed n-grams (ORDERED context) ===
    print("\n=== Q1: offset-attention vs fixed n-grams (ordered context) ===")
    r = {}
    r["bigram"] = eval_predictor(predict_bigram, stream, eval_start, M, "bigram (d=1 only)")
    r["trigram"] = eval_predictor(predict_trigram, stream, eval_start, M, "trigram (fixed, +backoff)")
    r["offset_ord"] = eval_predictor(oa.predict, stream, eval_start, M, "OFFSET-ATTENTION (ordered)")

    # === Q2: KILLS BAG-OF-WORDS — 2x2 {offset, bag} x {ordered, scrambled} ===
    print("\n=== Q2: kills bag-of-words (2x2: model x context order) ===")
    rng1 = np.random.default_rng(SEED)
    rng2 = np.random.default_rng(SEED)
    r["offset_scr"] = eval_predictor(oa.predict, stream, eval_start, M, "OFFSET-ATTENTION (scrambled)",
                                     scramble=True, rng=rng1)
    r["bag_ord"] = eval_predictor(oa.predict_bag, stream, eval_start, M, "BAG (ordered)")
    r["bag_scr"] = eval_predictor(oa.predict_bag, stream, eval_start, M, "BAG (scrambled)",
                                  scramble=True, rng=rng2)

    print("\n2x2 top-1 accuracy:")
    print(f"                 ordered    scrambled   delta(scramble)")
    print(f"  offset-attn   {r['offset_ord'][0]*100:7.2f}%  {r['offset_scr'][0]*100:8.2f}%   "
          f"{(r['offset_scr'][0]-r['offset_ord'][0])*100:+.2f}pp")
    print(f"  bag-of-words  {r['bag_ord'][0]*100:7.2f}%  {r['bag_scr'][0]*100:8.2f}%   "
          f"{(r['bag_scr'][0]-r['bag_ord'][0])*100:+.2f}pp")
    print("\n2x2 perplexity:")
    print(f"                 ordered    scrambled")
    print(f"  offset-attn   {r['offset_ord'][1]:9.1f}  {r['offset_scr'][1]:9.1f}")
    print(f"  bag-of-words  {r['bag_ord'][1]:9.1f}  {r['bag_scr'][1]:9.1f}")

    # machine-readable dump for RESULTS.md
    print("\nRESULTS_DICT = " + repr({k: (round(v[0], 5), round(v[1], 2)) for k, v in r.items()}))
    print("IG_DICT = " + repr({d: round(float(oa.ig[d]), 5) for d in range(1, D + 1)}))


if __name__ == "__main__":
    main()

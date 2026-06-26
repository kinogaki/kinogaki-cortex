"""Exp W — ray-cortex: does proximity-gather (raytracing) finally earn its keep IN COMBINATION?

Resurrects the parked raytracing/proximity idea (Exp P) as a RARE-CONTEXT BACKOFF modulator inside an
integrated online predictor: offset-attention backbone + PMI-graph proximity-gather + leaky evidence +
online-leader-clustered topic prior. ALL online / count-based / no-backprop (see lib/raycortex.py header).

We judge on the RIGHT axes (fragile-ideas commandment 7), not just headline next-word perplexity:
  1. overall perplexity vs baselines (bigram, trigram, offset-attention-alone, full ray-cortex)
  2. RARE-CONTEXT slice — positions where the direct n-gram has FEW counts (proximity's intended axis)
  3. robustness — 15% context corruption: does evidence+proximity degrade more gracefully?
  4. ablation — full vs minus-proximity vs minus-evidence vs minus-topic (which piece carries weight)

Run: python experiments/raycortex/run.py   (from the repo root)
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import corpus, offsetattn, raycortex

TRAIN_BYTES = 15_000_000
VOCAB = 20000
N_GRAPH = 3000
D = 6
EVAL_POS = 30000          # held-out positions scored per config
SEED = 0


def ppl(logp):
    return float(np.exp(-logp.mean()))


def slice_ppl(logp, mask):
    return ppl(logp[mask]) if mask.any() else float("nan")


# ── word n-gram baselines (online counts, add-ALPHA backoff) ──────────────────────────────────

def ngram_logp(train, stream, positions, order, vocab):
    """P(w_t | w_{t-order+1..t-1}) with add-ALPHA backoff down to unigram. Fully VECTORIZED: per order
    build a sorted packed (ctx·base+tgt) key array + a sorted ctx-total array, then resolve all eval
    positions with batched np.searchsorted (high→low backoff; lower orders fill unresolved positions)."""
    V = vocab + 1; ALPHA = raycortex.ALPHA; base = V
    m = len(positions)
    pos = np.asarray(positions)
    tgt = stream[pos].astype(np.int64)

    tabs = {}
    for k in range(1, order + 1):
        a = train.astype(np.int64); L = len(a)
        ctx = np.zeros(L - k, np.int64)
        for jj in range(k):
            ctx = ctx * base + a[jj:L - k + jj]
        tok = a[k:]
        fullkey = ctx * base + tok
        fk, fc = np.unique(fullkey, return_counts=True)          # sorted full (ctx,tgt) keys + counts
        uctx, inv = np.unique(fk // base, return_inverse=True)    # sorted ctx keys
        tot = np.zeros(len(uctx)); np.add.at(tot, inv, fc)
        tabs[k] = (fk, fc.astype(np.float64), uctx, tot)

    uni = np.bincount(train, minlength=V).astype(np.float64)
    uni_lp = np.log((uni + ALPHA) / (uni.sum() + ALPHA * V))

    out = np.full(m, np.nan)
    # build the eval-side context key for each order, vectorized
    for k in range(order, 0, -1):
        ck = np.zeros(m, np.int64)
        ok = pos >= k
        for jj in range(k):
            ck = ck * base + np.where(ok, stream[np.clip(pos - k + jj, 0, len(stream) - 1)], 0).astype(np.int64)
        fk, fc, uctx, tot = tabs[k]
        ci = np.searchsorted(uctx, ck)
        seen_ctx = ok & (ci < len(uctx)) & (uctx[np.clip(ci, 0, len(uctx) - 1)] == ck)
        todo = seen_ctx & np.isnan(out)
        if not todo.any():
            continue
        full = ck * base + tgt
        fi = np.searchsorted(fk, full)
        hit = (fi < len(fk)) & (fk[np.clip(fi, 0, len(fk) - 1)] == full)
        c = np.where(hit, fc[np.clip(fi, 0, len(fk) - 1)], 0.0)
        denom = tot[np.clip(ci, 0, len(uctx) - 1)]
        lp = np.log((c + ALPHA) / (denom + ALPHA * V))
        out[todo] = lp[todo]
    unresolved = np.isnan(out)
    out[unresolved] = uni_lp[tgt[unresolved]]
    return out


def main():
    rng = np.random.default_rng(SEED)
    print("loading text8 …")
    ids = corpus.load_ids("text8", nbytes=TRAIN_BYTES)
    spans = corpus.split_words(ids)
    stream, vocab_list, UNK = offsetattn.build_word_stream(ids, spans, vocab_size=VOCAB)
    n = len(stream)
    ntr = int(n * 0.90)
    train, held = stream[:ntr], stream[ntr:]
    print(f"  words={n:,}  vocab={VOCAB}  UNK_frac={(stream==UNK).mean():.3f}  train={ntr:,} held={len(held):,}")

    # eval positions: a contiguous slice of the held-out stream (so leaky evidence has real adjacency)
    pos = np.arange(D, D + EVAL_POS)
    pos = pos[pos < len(held)]
    targets = held[pos]

    # ── fit the integrated model ONCE (train-only), reuse its sub-models for every ablation ──
    print("fitting ray-cortex (offset + proximity-graph + topic) …")
    t = time.time()
    full = raycortex.RayCortex(D=D, N_graph=N_GRAPH, w_off=1.0, w_prox=1.0, w_topic=0.6,
                               ev_clip=np.inf, use_prox=True, use_topic=True).fit(train)
    print(f"  fit {time.time()-t:.1f}s   topic clusters K={full.topic.K}  graph N={N_GRAPH}")
    gseq = full.topic.commit_over(held)            # committed topic G over held-out text (online)

    # corrupt 15% of the held-out CONTEXT (targets scored against the clean held stream)
    held_corr = held.copy()
    ncorr = int(0.15 * len(held))
    cidx = rng.choice(len(held), size=ncorr, replace=False)
    held_corr[cidx] = rng.integers(0, VOCAB + 1, size=ncorr).astype(held.dtype)

    EV_ON = 4.0    # evidence winsorization radius (nats) when evidence is enabled; np.inf = off

    def run_rc(rc, corrupt=None, clip=np.inf):
        rc.ev_clip = clip
        lp, dc = rc.predict_stream(held, pos, gseq=gseq, corrupt_ctx=corrupt)
        return lp, dc

    results = {}   # name -> (logp, direct_counts)

    # baselines
    print("baselines …")
    t = time.time()
    results["bigram"]  = (_bigram_held(train, held, pos, VOCAB), None)
    results["trigram"] = (ngram_logp(train, held, pos, 3, VOCAB), None)
    print(f"  baselines {time.time()-t:.1f}s")

    # ablations — share the SAME fitted sub-models, just flip weights / the evidence clip
    configs = {
        "offset-only":      dict(w_prox=0.0, w_topic=0.0, clip=np.inf),
        "+proximity":       dict(w_prox=1.0, w_topic=0.0, clip=np.inf),
        "+topic":           dict(w_prox=0.0, w_topic=0.6, clip=np.inf),
        "+evidence":        dict(w_prox=0.0, w_topic=0.0, clip=EV_ON),
        "full":             dict(w_prox=1.0, w_topic=0.6, clip=EV_ON),
        "full−proximity":   dict(w_prox=0.0, w_topic=0.6, clip=EV_ON),
        "full−topic":       dict(w_prox=1.0, w_topic=0.0, clip=EV_ON),
        "full−evidence":    dict(w_prox=1.0, w_topic=0.6, clip=np.inf),
    }
    for name, cfg in configs.items():
        full.w_prox = cfg["w_prox"]; full.w_topic = cfg["w_topic"]
        t = time.time()
        lp, dc = run_rc(full, corrupt=None, clip=cfg["clip"])
        results[name] = (lp, dc)
        print(f"  {name:16s} ppl={ppl(lp):7.2f}  ({time.time()-t:.1f}s)", flush=True)

    # robustness: full vs offset-only under 15% corruption
    print("robustness (15% context corruption) …")
    full.w_prox = 1.0; full.w_topic = 0.6
    lp_full_c, dc_c = run_rc(full, corrupt=held_corr, clip=EV_ON)
    full.w_prox = 0.0; full.w_topic = 0.0
    lp_off_c, _ = run_rc(full, corrupt=held_corr, clip=np.inf)
    lp_big_c = _bigram_held(train, held_corr, pos, VOCAB, target_stream=held)

    # ── reporting ──
    dc = results["full"][1]                        # direct-count per eval position (same ctx for all rc cfgs)
    rare = dc <= 2                                  # "rare context" = backbone has ≤2 direct successors
    veryrare = dc == 0
    common = dc >= 20
    print()
    print(f"eval positions: {len(pos):,}   rare(dc≤2): {rare.mean()*100:.1f}%   "
          f"unseen(dc=0): {veryrare.mean()*100:.1f}%   common(dc≥20): {common.mean()*100:.1f}%")

    def line(name, lp):
        return (f"{name:16s}  {ppl(lp):8.2f}  {slice_ppl(lp, rare):9.2f}  "
                f"{slice_ppl(lp, veryrare):9.2f}  {slice_ppl(lp, common):9.2f}")

    print("\n=== PERPLEXITY (overall / rare dc≤2 / unseen dc=0 / common dc≥20) ===")
    print(f"{'model':16s}  {'overall':>8s}  {'rare':>9s}  {'unseen':>9s}  {'common':>9s}")
    for name in ["bigram", "trigram", "offset-only", "+proximity", "+topic", "+evidence",
                 "full−proximity", "full−topic", "full−evidence", "full"]:
        print(line(name, results[name][0]))

    print("\n=== ROBUSTNESS (15% context corruption) — overall ppl ===")
    print(f"  bigram         {ppl(lp_big_c):8.2f}")
    print(f"  offset-only    {ppl(lp_off_c):8.2f}")
    print(f"  full ray-cortex{ppl(lp_full_c):8.2f}")
    print(f"  clean→corrupt degradation:  offset {ppl(lp_off_c)/ppl(results['offset-only'][0]):.2f}x   "
          f"full {ppl(lp_full_c)/ppl(results['full'][0]):.2f}x")

    # save a compact npz for RESULTS.md authoring
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.npz"),
             dc=dc, **{k.replace("−", "_minus_"): v[0] for k, v in results.items()},
             lp_full_c=lp_full_c, lp_off_c=lp_off_c, lp_big_c=lp_big_c,
             rare=rare, veryrare=veryrare, common=common)
    print("\nsaved results.npz")


def _bigram_held(train, held_ctx, pos, vocab, target_stream=None):
    """Bigram trained on `train`, scored on held positions; ctx from held_ctx, targets from target_stream
    (defaults to held_ctx — pass the clean held when held_ctx is corrupted)."""
    import math
    from collections import defaultdict
    V = vocab + 1; ALPHA = raycortex.ALPHA
    tgt_stream = target_stream if target_stream is not None else held_ctx
    big = defaultdict(lambda: defaultdict(int)); tot = defaultdict(int)
    for t in range(1, len(train)):
        big[int(train[t-1])][int(train[t])] += 1; tot[int(train[t-1])] += 1
    uni = np.bincount(train, minlength=V).astype(np.float64)
    uni_lp = np.log((uni + ALPHA) / (uni.sum() + ALPHA * V))
    out = np.empty(len(pos))
    for j, t in enumerate(pos):
        prev = int(held_ctx[t-1]); tgt = int(tgt_stream[t])
        if prev in big:
            out[j] = math.log((big[prev].get(tgt, 0) + ALPHA) / (tot[prev] + ALPHA * V))
        else:
            out[j] = uni_lp[tgt]
    return out


if __name__ == "__main__":
    main()

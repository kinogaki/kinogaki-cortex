#!/usr/bin/env python3
"""Exp AJ — less-is-more: validity-ordered, noncompensatory, early-stopping inference.

WORD level on text8, over the SAME offset count tables as Exp S (offsetattn). We stop pooling every cue
with a geometric mean and instead rank cues by their online VALIDITY, consult them one at a time
high-validity-first, and STOP at the first cue that clears a satisficing aspiration margin
(Gigerenzer/Goldstein take-the-best + recognition heuristic; Simon satisficing). Three questions:

  (1) FRUGAL vs FULL. Accuracy AND mean cues-consulted/compute of take-the-best+early-stop vs the
      full-integration baseline. Does frugal match-or-beat accuracy at a fraction of the compute?
  (2) LESS-IS-MORE (α>β). On sparse/noisy contexts, does IGNORING the weak soft channel (only letting
      it override a crisp count cue when v[soft] > v[count]) IMPROVE accuracy vs always integrating it?
  (3) BASE-RATE GUARD. Does assignment = argmax(similarity × clusterCount^γ) with a small γ>0 improve
      cluster STABILITY under context perturbation vs pure-similarity (γ=0)?

No neural net, no GPU, no gradients — counts, online hit/miss validity, a leaky aspiration, a count prior.
"""
import os, sys, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import corpus
from offsetattn import OffsetAttn, build_word_stream
import takethebest as ttb
import jepa

D = 8                    # offsets / context width (matches Exp S)
VOCAB = 40000
NBYTES = 18_000_000      # text8 prefix (~3M words) — keeps the run ~15-20MB / a few minutes
EVAL_WORDS = 80_000
PRIME_WINDOW = 200_000   # online/causal priming pass bounded to the last N train words (validity ≈ ecological)
SEED = 0


LAMBDA = 0.10           # unigram backoff weight: p = (1-λ)·cue_dist + λ·unigram (a fair, finite ppl)


def score_dist(dist, truth, M, uni):
    """(correct?, neg-log-prob-of-truth). The cue's (top-K) dist is BACKED OFF to the unigram — every
    combiner is scored as p = (1-λ)·dist[truth] + λ·unigram[truth], so a truth dropped by the top-K cap
    is never starved and perplexity is finite and comparable across combiners. Top-1 (accuracy) is read
    from the cue dist alone (the backoff is uniform-ish and never changes the argmax)."""
    up = uni[truth]
    if not dist:
        return False, -math.log(max(up, 1e-12))
    top = max(dist.items(), key=lambda kv: kv[1])[0]
    p = (1 - LAMBDA) * dist.get(truth, 0.0) + LAMBDA * up
    return top == truth, -math.log(max(p, 1e-12))


def eval_combiner(predict_fn, stream, eval_start, M, label, uni, mask=None):
    """Walk eval positions, predict_fn(ctx) -> (dist, n_cues). Score acc, perplexity, mean cues consulted.
    `mask` (len = #eval positions) restricts the tally to a subset (used for the sparse/noisy slices)."""
    correct = nll = cues = n = kept = 0.0
    t0 = time.time()
    for i, t in enumerate(range(eval_start, len(stream))):
        ctx = stream[t - D:t]
        truth = int(stream[t])
        dist, nc = predict_fn(ctx)
        n += 1; cues += nc
        c, l = score_dist(dist, truth, M, uni)
        if mask is None or mask[i]:
            correct += c; nll += l; kept += 1
    acc = correct / kept if kept else 0.0
    ppl = math.exp(nll / kept) if kept else float("inf")
    mean_cues = cues / n
    dt = time.time() - t0
    print(f"  {label:<40} acc={acc*100:6.2f}%  ppl={ppl:9.1f}  cues/step={mean_cues:5.2f}  "
          f"(n={int(kept)}, {dt:.1f}s)")
    return acc, ppl, mean_cues


def main():
    print(f"Exp AJ — take-the-best / less-is-more  (D={D}, vocab={VOCAB}, {NBYTES//1_000_000}MB text8)")
    print("loading text8 ...")
    ids = corpus.load_ids("text8", NBYTES)
    spans = corpus.split_words(ids)
    stream, vocab_list, UNK = build_word_stream(ids, spans, VOCAB)
    M = UNK + 1
    print(f"words: {len(stream):,}   vocab {VOCAB}+UNK   OOV(UNK) rate {np.mean(stream==UNK):.3f}")

    eval_start = len(stream) - EVAL_WORDS
    train = stream[:eval_start]
    print(f"train {len(train):,} words   /   eval {EVAL_WORDS:,} words")

    print("fitting offset tables ...")
    t0 = time.time()
    oa = OffsetAttn(D).fit(train)
    oa.build_bag()
    print(f"  fitted in {time.time()-t0:.1f}s")

    # take-the-best (with and without the less-is-more override) share the primed cue ranking
    print("priming cue validity + satisficing aspiration (online, causal over train) ...")
    t0 = time.time()
    lim = ttb.TakeTheBest(oa, less_is_more=True).prime(stream, eval_start, window=PRIME_WINDOW)
    print(f"  primed in {time.time()-t0:.1f}s   aspiration bar = {lim.asp.level:.4f}")

    print("\nper-cue ONLINE VALIDITY (descending — the order take-the-best scans):")
    for name, v, nseen in lim.cv.table():
        bar = "#" * int(round(v * 50))
        print(f"  {name:<6} v={v:.4f}  (n={nseen:>8})  {bar}")
    print(f"  v[soft]={lim.v['soft']:.4f}   v[recog]={lim.v['recog']:.4f}   "
          f"v[off1]={lim.v['off1']:.4f}  (α>β check uses these)")

    # unigram backoff (train frequencies) — the shared smoothing floor for a fair, finite perplexity
    fcnt = np.bincount(train.astype(np.int64), minlength=M).astype(np.float64) + 0.5
    uni = fcnt / fcnt.sum()

    # === Q1: FRUGAL vs FULL — accuracy AND compute ===
    print("\n=== Q1: take-the-best+early-stop vs full integration (accuracy AND compute) ===")
    r = {}
    r["full"] = eval_combiner(lambda c: ttb.full_integration(oa, c), stream, eval_start, M,
                              "full integration (all offsets, baseline)", uni)
    r["ttb"] = eval_combiner(lambda c: lim.predict(c), stream, eval_start, M,
                             "TAKE-THE-BEST (validity-ordered, early-stop)", uni)

    # === Q2: LESS-IS-MORE — ignore the weak channel on sparse/noisy contexts ===
    # Sparse = the most-recent context word is RARE (few train occurrences) → the soft pool is thin.
    print("\n=== Q2: less-is-more (α>β) — ignore the weak soft channel on sparse/noisy contexts ===")
    freq = np.bincount(train.astype(np.int64), minlength=M)
    sparse_mask = np.array([freq[int(stream[t - 1])] < 50 for t in range(eval_start, len(stream))])
    print(f"  sparse slice = positions whose t-1 word has <50 train occurrences: "
          f"{int(sparse_mask.sum())}/{len(sparse_mask)} ({sparse_mask.mean()*100:.1f}%)")

    lim_off = ttb.TakeTheBest(oa, less_is_more=False)   # reuse primed validity/aspiration
    lim_off.cv = lim.cv; lim_off.asp = lim.asp; lim_off.v = lim.v
    lim_off.order = lim.order; lim_off.crisp_order = lim.crisp_order

    print("  -- ALL contexts --")
    r["ttb_lim_all"] = eval_combiner(lambda c: lim.predict(c), stream, eval_start, M,
                                     "take-the-best + less-is-more (ignore weak)", uni)
    r["ttb_compensatory_all"] = eval_combiner(lambda c: lim_off.predict(c), stream, eval_start, M,
                                              "take-the-best, compensatory (always integrate)", uni)
    print("  -- SPARSE contexts only (the α>β prediction) --")
    r["ttb_lim_sp"] = eval_combiner(lambda c: lim.predict(c), stream, eval_start, M,
                                    "less-is-more (ignore weak)  [sparse]", uni, mask=sparse_mask)
    r["ttb_comp_sp"] = eval_combiner(lambda c: lim_off.predict(c), stream, eval_start, M,
                                     "compensatory (integrate)   [sparse]", uni, mask=sparse_mask)
    r["full_sp"] = eval_combiner(lambda c: ttb.full_integration(oa, c), stream, eval_start, M,
                                 "full integration            [sparse]", uni, mask=sparse_mask)

    # === Q3: BASE-RATE GUARD on clustering — stability under context perturbation ===
    print("\n=== Q3: base-rate guard (argmax similarity × clusterCount^γ) — cluster stability ===")
    seq = train.astype(np.int64).copy()
    seq[seq == UNK] = -1                              # treat UNK as OOV for signatures
    Ntop = min(VOCAB, int(seq.max()) + 1)
    seq = np.where((seq >= 0) & (seq < Ntop), seq, -1)
    print("  building online signatures ...")
    sig, scnt = jepa.online_signatures(seq, Ntop, D=64, window=5, seed=SEED)
    # cluster the most-frequent ripe words in first-appearance (stream) order — vectorized first-seen
    first_seen = np.full(Ntop, len(seq), np.int64)
    valid_pos = np.nonzero(seq >= 0)[0]
    np.minimum.at(first_seen, seq[valid_pos], valid_pos)   # earliest position of each word id
    order = np.argsort(first_seen)
    order = order[first_seen[order] < len(seq)]

    # perturb the context (corrupt 15% of words) → rebuild signatures → re-cluster; measure agreement
    seq_pert = jepa.corrupt(seq, 0.15, Ntop, seed=SEED + 1)
    sig_p, scnt_p = jepa.online_signatures(seq_pert, Ntop, D=64, window=5, seed=SEED)

    # Cmax raised so the prior can genuinely CONSOLIDATE (cluster count is free to fall with γ — the
    # whole point of a base rate). thresh kept at jepa's default; γ=0 reproduces pure similarity exactly.
    CMAX = 1500
    q3 = []
    for gamma in (0.0, 0.05, 0.10, 0.25, 0.50):
        clu_a, Ca = ttb.guarded_leader_cluster(sig, scnt, order, gamma=gamma, Cmax=CMAX)
        clu_b, Cb = ttb.guarded_leader_cluster(sig_p, scnt_p, order, gamma=gamma, Cmax=CMAX)
        stab = ttb.cluster_stability(clu_a, clu_b)
        nclust = int((clu_a >= 0).sum())
        print(f"  γ={gamma:<4}  clusters={Ca:>4}  words_clustered={nclust:>5}  "
              f"stability(perturb)={stab:.4f}")
        q3.append((gamma, Ca, nclust, round(stab, 4)))

    # machine-readable dump for RESULTS.md
    print("\nRESULTS_Q1Q2 = " + repr({k: (round(v[0], 5), round(v[1], 2), round(v[2], 3))
                                      for k, v in r.items()}))
    print("VALIDITY = " + repr({n: round(v, 5) for n, v in lim.v.items()}))
    print("ASPIRATION = " + repr(round(lim.asp.level, 5)))
    print("Q3 = " + repr(q3))


if __name__ == "__main__":
    main()

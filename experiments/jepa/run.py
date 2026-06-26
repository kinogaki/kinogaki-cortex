#!/usr/bin/env python3
"""Exp U — JEPA-style, ONLINE: predict in REPRESENTATION space (not input space) under masking, + inter-layer
sparsity. NO backprop, NO k-means, NO SVD/eigendecomposition — single streaming pass + online leader clustering.

JEPA's thesis (LeCun): don't reconstruct the raw input; mask part of it and predict the masked part's abstract
REPRESENTATION from the visible context; add sparsity so representations don't collapse. Translated to the
count-cortex with a hard ONLINE-ONLY rule:

  1. REPRESENTATION = a word's concept CLUSTER, built ONLINE: each word's signature is an online-accumulated
     hashed context-count vector (random sign-projection BY ACCUMULATION, no factorization); clusters are formed
     by ONLINE LEADER CLUSTERING in one pass (nearest running-mean prototype by cosine, or spawn). Grounded in
     counts -> the latent CANNOT collapse, with no collapse-prevention machinery.
  2. MASKED PREDICTION = hide a word, build BIDIRECTIONAL ±W offset-keyed context counts, predict either the
     masked WORD (input space) or its CLUSTER (rep space / JEPA). Report token-ppl, cluster-ppl, accuracy split
     by word frequency (frequent vs rare), and degradation under context corruption.
  3. INTER-LAYER SPARSITY = predict the masked cluster from a SPARSE top-k cluster code of the context; sweep k,
     watch the train/test overfit gap.

Corpus: text8 (clean a-z+space). Train ~12 MB, held-out test tail.
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from jepa import (online_signatures, leader_cluster, JepaCounts, context_at, SparseCluCounts, topk_code, corrupt)

print = functools.partial(print, flush=True)

# ── config ──
TRAIN_BYTES   = 12_000_000
N             = 12000      # top-N words get a representation (cluster); rest are OOV (-1)
D             = 64         # signature dims (hashed context features)
W             = 5          # context half-window (bidirectional masking)
SIG_WINDOW    = 5          # ±window used to build online signatures
MIN_EVIDENCE  = 40         # a word needs this much context evidence before it gets clustered (online "ripe")
COS_THRESH    = 0.75       # leader-clustering: join if cosine>=this, else spawn
CMAX          = 400        # cluster cap (the abstract vocabulary size)
EVAL_PROBES   = 60_000
RARE_THRESH   = 50         # train-count below this = "rare" word (frequency split)
SEED          = 0


def perplexity(p_at_truth):
    return float(np.exp(-np.mean(np.log(np.clip(p_at_truth, 1e-12, 1.0)))))


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN_BYTES)
    spans = split_words(ids)
    words = [ids_to_str(ids[s:e]) for s, e in spans]
    w2id, wids = {}, np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    cut = int(len(wids) * 0.85)
    train_g, test_g = wids[:cut], wids[cut:]
    print(f"{len(words):,} words, {len(w2id):,} types | train {len(train_g):,}  test {len(test_g):,}  "
          f"(load {time.time()-t0:.1f}s)")

    counts_g = np.bincount(train_g, minlength=len(w2id))
    top = np.argsort(counts_g)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    seq_tr = remap[train_g]
    seq_te = remap[test_g]
    tr_count = np.bincount(seq_tr[seq_tr >= 0], minlength=N)

    # ── 1. ONLINE REPRESENTATION: signatures (1 pass) + leader clustering (1 pass) ──
    t1 = time.time()
    sig, ev = online_signatures(seq_tr, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first_seen = np.full(N, len(seq_tr), np.int64)
    valid_pos = np.nonzero(seq_tr >= 0)[0]
    np.minimum.at(first_seen, seq_tr[valid_pos], valid_pos)        # earliest stream position per top-word
    order = np.argsort(first_seen)
    order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    sizes = np.bincount(clu[clu >= 0], minlength=C)
    n_clustered = int((clu >= 0).sum())
    print(f"online signatures + leader-cluster in {time.time()-t1:.1f}s | "
          f"C={C} clusters, {n_clustered:,}/{N:,} top-words clustered "
          f"(rest below evidence={MIN_EVIDENCE}) | cluster sizes min {sizes.min()} "
          f"max {sizes.max()} median {int(np.median(sizes))}")

    print("\n=== representation = ONLINE distributional cluster (grounded in counts -> non-collapsing) ===")
    shown = 0
    for c in np.argsort(sizes)[::-1]:
        members = list(np.nonzero(clu == c)[0])
        members.sort(key=lambda i: -tr_count[i])
        names = [topword[i] for i in members]
        if 4 <= len(names) <= 60:
            print(f"    cluster {c:3d} ({len(names):3d}w): " + ", ".join(names[:12]))
            shown += 1
        if shown >= 10:
            break

    # ── 2. MASKED PREDICTION: bidirectional offset-keyed context counts (vectorized) ──
    t2 = time.time()
    jc = JepaCounts(W=W, N=N, C=C, alpha=0.1)
    jc.fit(seq_tr, clu)
    print(f"\nfit JepaCounts (bidirectional +/-{W}) in {time.time()-t2:.1f}s")

    rng = np.random.default_rng(SEED)
    cand = np.nonzero(seq_te >= 0)[0]
    cand = cand[(cand >= W) & (cand < len(seq_te) - W)]
    cand = cand[clu[seq_te[cand]] >= 0]
    probe = rng.choice(cand, size=min(EVAL_PROBES, len(cand)), replace=False)

    def eval_probes(seq_ctx):
        ptok = np.empty(len(probe)); pclu = np.empty(len(probe))
        htok = np.empty(len(probe)); hclu = np.empty(len(probe))
        cnt = np.empty(len(probe), np.int64)
        for i, t in enumerate(probe):
            tgt = seq_te[t]; tc = clu[tgt]
            ctx = context_at(seq_ctx, t, W)
            dt = jc.predict_tok(ctx); dc = jc.predict_clu(ctx)
            if dt is None: ptok[i] = 1.0 / N; htok[i] = 0
            else:          ptok[i] = dt[tgt]; htok[i] = (dt.argmax() == tgt)
            if dc is None: pclu[i] = 1.0 / C; hclu[i] = 0
            else:          pclu[i] = dc[tc]; hclu[i] = (dc.argmax() == tc)
            cnt[i] = tr_count[tgt]
        return ptok, pclu, htok, hclu, cnt

    t3 = time.time()
    ptok, pclu, htok, hclu, cnt = eval_probes(seq_te)
    print(f"eval {len(probe):,} probes in {time.time()-t3:.1f}s")

    # rep->token: pick predicted cluster, argmax token within it (token decision THROUGH the latent)
    clu_members = [np.nonzero(clu == c)[0] for c in range(C)]
    uni = tr_count.astype(np.float64)
    href = np.empty(len(probe))
    for i, t in enumerate(probe):
        tgt = seq_te[t]; ctx = context_at(seq_te, t, W)
        dc = jc.predict_clu(ctx); dt = jc.predict_tok(ctx)
        if dc is None: href[i] = 0; continue
        mem = clu_members[int(dc.argmax())]
        if mem.size == 0: href[i] = 0; continue
        score = dt[mem] if dt is not None else uni[mem]
        href[i] = (mem[int(score.argmax())] == tgt)

    rare = cnt < RARE_THRESH; freq = ~rare
    print("\n=== masked prediction: input space (token) vs representation space (cluster) ===")
    print(f"    probes {len(probe):,}  | rare(<{RARE_THRESH}) {rare.mean()*100:.1f}%  freq {freq.mean()*100:.1f}%")
    print(f"    token   : ppl {perplexity(ptok):8.1f}   acc {htok.mean()*100:5.2f}%   "
          f"(rare {htok[rare].mean()*100:5.2f}%  freq {htok[freq].mean()*100:5.2f}%)")
    print(f"    cluster : ppl {perplexity(pclu):8.2f}   acc {hclu.mean()*100:5.2f}%   "
          f"(rare {hclu[rare].mean()*100:5.2f}%  freq {hclu[freq].mean()*100:5.2f}%)")
    print(f"    rep->token (decide via predicted cluster): acc {href.mean()*100:5.2f}%   "
          f"(rare {href[rare].mean()*100:5.2f}%  freq {href[freq].mean()*100:5.2f}%)  vs direct token {htok.mean()*100:.2f}%")
    tr_clu_lbl = clu[seq_tr[seq_tr >= 0]]
    base_clu = np.bincount(tr_clu_lbl[tr_clu_lbl >= 0], minlength=C).argmax()
    print(f"    cluster majority-baseline acc {np.mean(clu[seq_te[probe]] == base_clu)*100:5.2f}%   "
          f"(chance 1/C = {100.0/C:.2f}%)")

    # ── robustness: corrupt 10-20% of CONTEXT words, compare degradation ──
    print("\n=== robustness: corrupt context words, compare degradation (JEPA claims rep-space robust) ===")
    print(f"    {'corrupt%':>9}  {'tok acc':>8} {'tok ppl':>9}   {'clu acc':>8} {'clu ppl':>9}")
    clean_tok, clean_clu = htok.mean(), hclu.mean()
    clean_ptok, clean_pclu = perplexity(ptok), perplexity(pclu)
    last_ht = clean_tok; last_hc = clean_clu
    for frac in (0.0, 0.10, 0.20):
        seq_c = corrupt(seq_te, frac, N, seed=SEED + 1) if frac > 0 else seq_te
        pt, pc, ht, hc, _ = eval_probes(seq_c)
        print(f"    {frac*100:8.0f}%  {ht.mean()*100:7.2f}% {perplexity(pt):9.1f}   "
              f"{hc.mean()*100:7.2f}% {perplexity(pc):9.2f}")
        last_ht, last_hc = ht.mean(), hc.mean()
    print(f"    (at 20% noise: token acc retains {last_ht/max(clean_tok,1e-9)*100:.0f}% of clean, "
          f"cluster acc retains {last_hc/max(clean_clu,1e-9)*100:.0f}%)")

    # ── 3. INTER-LAYER SPARSITY: sparse top-k cluster code -> masked cluster. Sweep k. ──
    print("\n=== inter-layer sparsity: predict masked CLUSTER from a sparse top-k cluster code ===")
    print(f"    {'k':>5}  {'train acc':>9} {'test acc':>9} {'overfit gap':>11}  {'test ppl':>9}")

    def probe_set(seq, lo, hi, m):
        c = np.nonzero(seq >= 0)[0]; c = c[(c >= W) & (c < len(seq) - W)]; c = c[clu[seq[c]] >= 0]
        return rng.choice(c, size=min(m, len(c)), replace=False)
    probe_tr = probe_set(seq_tr, 0, 0, 40_000)
    probe_te2 = probe[:40_000]

    def ctx_codes(seq, probes):
        """For each probe: the list of visible context CLUSTER ids (deduped per position kept as multiset)."""
        out = []
        for t in probes:
            cl = [clu[seq[j]] for j in range(max(0, t - W), min(len(seq), t + W + 1))
                  if j != t and seq[j] >= 0 and clu[seq[j]] >= 0]
            out.append(np.array(cl, np.int64))
        return out
    cc_tr = ctx_codes(seq_tr, probe_tr); cc_te = ctx_codes(seq_te, probe_te2)
    tgt_tr = clu[seq_tr[probe_tr]]; tgt_te = clu[seq_te[probe_te2]]

    # training-fit codes come from ALL train positions (vectorized): build per-position context cluster lists once
    # (reuse the offset structure: for each train position, its window's clusters). To stay fast we sample a large
    # train subset for FITTING the sparse tables too — counting on a representative stream (still single-pass).
    fit_probes = probe_set(seq_tr, 0, 0, 400_000)
    cc_fit = ctx_codes(seq_tr, fit_probes); tgt_fit = clu[seq_tr[fit_probes]]

    sweep = [1, 3, 10, 30, None]
    rows = []
    for k in sweep:
        # build flat (active_cluster, target_cluster) pairs from the fit stream
        af, tf = [], []
        for cl, tc in zip(cc_fit, tgt_fit):
            code = topk_code(cl, C, k)
            if code.size:
                af.append(code); tf.append(np.full(code.size, tc))
        af = np.concatenate(af) if af else np.empty(0, np.int64)
        tf = np.concatenate(tf) if tf else np.empty(0, np.int64)
        sc = SparseCluCounts(C=C, alpha=0.1).fit_codes(af, tf)

        def acc_ppl(ccs, tgts):
            hit = 0; ps = np.empty(len(ccs))
            for i, cl in enumerate(ccs):
                d = sc.predict([int(a) for a in topk_code(cl, C, k)])
                if d is None: ps[i] = 1.0 / C; continue
                ps[i] = d[tgts[i]]; hit += (d.argmax() == tgts[i])
            return hit / len(ccs), perplexity(ps)
        atr, _ = acc_ppl(cc_tr, tgt_tr)
        ate, pte = acc_ppl(cc_te, tgt_te)
        kl = "dense" if k is None else str(k)
        rows.append((kl, atr, ate, atr - ate, pte))
        print(f"    {kl:>5}  {atr*100:8.2f}% {ate*100:8.2f}% {(atr-ate)*100:+10.2f}%  {pte:9.2f}")

    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(C=C, n_clustered=n_clustered, probes=len(probe),
                tok_acc=clean_tok, clu_acc=clean_clu, tok_ppl=clean_ptok, clu_ppl=clean_pclu,
                tok_rare=float(htok[rare].mean()), tok_freq=float(htok[freq].mean()),
                clu_rare=float(hclu[rare].mean()), clu_freq=float(hclu[freq].mean()),
                href=float(href.mean()), rare_frac=float(rare.mean()), rows=rows)


if __name__ == "__main__":
    main()

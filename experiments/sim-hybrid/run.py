#!/usr/bin/env python3
"""Exp Z — similarity reps as a REPRESENTATION FACTORY, projected INTO the count predictor. ONLINE, NO backprop.

Thesis (from Exp P/W): proximity/"raytracing" is a bad PREDICTOR but a good SIMILARITY tool. So use it as a
representation factory — build similarity reps HIERARCHICALLY (words, then phrases) the ONLINE way (jepa.py's
online co-occurrence signatures + online leader clustering), then PROJECT those reps INTO a count predictor:
when a context word/phrase is rare or unseen, back off onto its similarity CLUSTER's aggregated next-word
counts, so the rare item inherits its neighbours' statistics. Hybrids are the point.

Right axis (FRAGILE_IDEAS): judge on the RARE-CONTEXT slice — where direct counts are sparse and similarity is
supposed to help — not just headline perplexity on common text.

Pipeline (single streaming pass; no backprop, no k-means, no SVD/eigen, no PPMI factorization):
  1. L1 word reps  : word -> online signature -> online leader-cluster id.    (jepa.online_signatures/leader_cluster)
  2. L2 phrase reps: branching-entropy phrase units -> bag of word-reps -> online leader-cluster id.
  3. count predictor: bigram P(next|prev_word) + per-cluster aggregated next-word counts.
  4. hybrid eval   : bigram vs +word-rep backoff vs +hierarchy(phrase). Sliced overall / rare / unseen.

Corpus: text8, ~18 MB. 85/15 train/test. Fixed seed.
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from simhybrid import WordReps, PhraseReps, BigramCounts, SimBackoffLM

print = functools.partial(print, flush=True)

TRAIN_BYTES = 18_000_000
N           = 12000        # top-N words get a representation; rest OOV (-1)
D           = 96
SIG_WINDOW  = 4
MIN_EV      = 80
COS_THRESH  = 0.85         # sharper clusters: numbers cluster, countries cluster (the similarity sanity check)
CMAX        = 2000
ALPHA       = 0.1
KAPPA       = 20.0         # word-cluster backoff weight = kappa/(kappa+prev_count)
KAPPA_P     = 40.0
PHRASE_RATE = 0.5
PHRASE_MINEV= 8
RARE_THRESH = 20           # prev-word train-count below this = "rare context"
EVAL_PROBES = 120_000
SEED        = 0


def ppl(p_true):
    return float(np.exp(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0)))))


def ci95(mask, hits):
    """95% CI half-width for an accuracy over a boolean hit array restricted to mask."""
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0
    p = hits[mask].mean()
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-12) / n), n


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

    counts_g = np.bincount(train_g, minlength=len(w2id))
    top = np.argsort(counts_g)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    seq_tr = remap[train_g]
    seq_te = remap[test_g]
    print(f"{len(words):,} words, {len(w2id):,} types | train {len(seq_tr):,} test {len(seq_te):,} "
          f"| top-N {N} | load {time.time()-t0:.1f}s")

    # ── 1. L1 word reps (online) ──
    t1 = time.time()
    wr = WordReps(N=N, D=D, sig_window=SIG_WINDOW, min_evidence=MIN_EV,
                  cos_thresh=COS_THRESH, cmax=CMAX, seed=SEED).fit(seq_tr)
    sizes = np.bincount(wr.clu[wr.clu >= 0], minlength=wr.C)
    print(f"L1 word reps: C={wr.C} clusters, {(wr.clu>=0).sum():,}/{N} clustered "
          f"| sizes min {sizes.min()} med {int(np.median(sizes))} max {sizes.max()} | {time.time()-t1:.1f}s")
    tr_count = np.bincount(seq_tr[seq_tr >= 0], minlength=N)
    print("  similarity sanity check (does proximity = meaning hold up?):")
    word2top = {w: i for i, w in enumerate(topword)}
    for probe_w in ("three", "france", "january"):
        wid = word2top.get(probe_w)
        if wid is None or wr.clu[wid] < 0:
            continue
        c = int(wr.clu[wid])
        mem = list(np.nonzero(wr.clu == c)[0]); mem.sort(key=lambda i: -tr_count[i])
        print(f"    '{probe_w}' -> c{c} ({sizes[c]:3d}w): " + ", ".join(topword[i] for i in mem[:12]))

    # ── 2. L2 phrase reps (online) ──
    t2 = time.time()
    pr = PhraseReps(wr, min_evidence=PHRASE_MINEV, cos_thresh=COS_THRESH, cmax=300,
                    target_rate=PHRASE_RATE, seed=SEED).fit(seq_tr, N)
    print(f"L2 phrase reps: {len(pr.key2pid):,} phrase types, PC={pr.PC} phrase clusters | {time.time()-t2:.1f}s")
    if pr.PC > 0:
        psizes = np.bincount(pr.pid_clu[pr.pid_clu >= 0], minlength=pr.PC)
        shown = 0
        for c in np.argsort(psizes)[::-1]:
            pids = [p for p in range(len(pr.pid_clu)) if pr.pid_clu[p] == c]
            pids.sort(key=lambda p: -pr.pev[p])
            phr = [" ".join(topword[w] for w in pr.members[p]) for p in pids[:4]]
            if 3 <= len(pids) <= 200:
                print(f"    p{c:3d} ({len(pids):3d} phrases): " + " | ".join(phr))
                shown += 1
            if shown >= 6:
                break

    # ── 3. count predictor (project reps into it) ──
    t3 = time.time()
    bc = BigramCounts(N=N, clu=wr.clu, C=wr.C, alpha=ALPHA).fit(seq_tr)
    lm = SimBackoffLM(bc, wr, pr, kappa=KAPPA, kappa_p=KAPPA_P)
    print(f"count predictor + per-cluster aggregates fit | {time.time()-t3:.1f}s")

    # ── per-test-position recent phrase word-cluster (for the hierarchy expert) ──
    # The recent phrase = the branching-entropy phrase covering the position just before t in the TEST stream;
    # its coarse word-cluster = the majority word-cluster of that phrase's member words (the bag's cluster).
    te_spans = []
    from simhybrid import phrase_cuts as _pc
    te_for_cuts = np.where(seq_te < 0, N, seq_te)
    te_spans = _pc(te_for_cuts, N + 1, target_rate=PHRASE_RATE)
    # map each test position -> the word-cluster of the phrase ending at/just before it
    pos_pcluster = np.full(len(seq_te), -1, np.int64)
    for (s, e) in te_spans:
        ws = seq_te[s:e]; ws = ws[ws >= 0]
        cl = wr.clu[ws]; cl = cl[cl >= 0]
        if cl.size == 0:
            continue
        maj = np.bincount(cl).argmax()
        if e < len(seq_te):
            pos_pcluster[e] = maj                 # phrase before position e -> coarse cluster for predicting at e

    # ── 4. eval: probe random test positions, score the three models, slice by rare/unseen prev context ──
    rng = np.random.default_rng(SEED)
    cand = np.nonzero((seq_te[:-1] >= 0) & (seq_te[1:] >= 0))[0] + 1   # positions t with prev & target in-vocab
    probe = rng.choice(cand, size=min(EVAL_PROBES, len(cand)), replace=False)
    prev = seq_te[probe - 1]
    tgt = seq_te[probe]
    prev_tr_count = bc.prev_count[prev]                                # how much DIRECT evidence the context has

    p_big = np.empty(len(probe)); h_big = np.empty(len(probe), bool)
    p_wr  = np.empty(len(probe)); h_wr  = np.empty(len(probe), bool)
    p_hi  = np.empty(len(probe)); h_hi  = np.empty(len(probe), bool)
    unseen_tgt = np.empty(len(probe), bool)         # true target NOT among prev word's direct bigram next-set
    for i, t in enumerate(probe):
        pv = int(prev[i]); tg = int(tgt[i])
        ni, _ = bc.word_row(pv)
        unseen_tgt[i] = tg not in ni                # the backoff's true home: zero direct count for THIS bigram
        db = lm.prob_bigram(pv); p_big[i] = db[tg]; h_big[i] = (db.argmax() == tg)
        dw = lm.prob_wordrep(pv); p_wr[i] = dw[tg]; h_wr[i] = (dw.argmax() == tg)
        dh = lm.prob_hier(pv, int(pos_pcluster[t])); p_hi[i] = dh[tg]; h_hi[i] = (dh.argmax() == tg)

    rare = prev_tr_count < RARE_THRESH
    common = ~rare

    def line(name, p, h, m):
        cw, n = ci95(m, h)
        return (f"  {name:<22} ppl {ppl(p[m]):8.2f}  acc {h[m].mean()*100:6.3f}% "
                f"+/-{cw*100:.3f}  (n={n:,})")

    print(f"\n=== next-word: count-only vs +word-rep projection vs +hierarchy(phrase) ===")
    print(f"  probes {len(probe):,} | rare(prev<{RARE_THRESH}) {rare.mean()*100:.1f}% "
          f"| common {common.mean()*100:.1f}% | unseen-target {unseen_tgt.mean()*100:.1f}%")
    for nm, mask in (("OVERALL", np.ones(len(probe), bool)), ("RARE context (prev rare)", rare),
                     ("UNSEEN target (zero direct count)", unseen_tgt), ("COMMON context", common)):
        print(f"  -- {nm} --")
        print(line("bigram (count only)", p_big, h_big, mask))
        print(line("+word-rep backoff", p_wr, h_wr, mask))
        print(line("+hierarchy (phrase)", p_hi, h_hi, mask))

    # paired rare-slice deltas with CI
    def paired_delta(h_a, h_b, mask):
        d = (h_b.astype(float) - h_a.astype(float))[mask]
        n = len(d)
        return d.mean() * 100, 1.96 * d.std() / np.sqrt(max(n, 1)) * 100, n
    dm, dc, dn = paired_delta(h_big, h_wr, rare)
    print(f"\n  RARE-slice paired acc gain  (+word-rep − bigram): {dm:+.3f}% +/-{dc:.3f}  (n={dn:,})")
    dm2, dc2, dn2 = paired_delta(h_wr, h_hi, rare)
    print(f"  RARE-slice paired acc gain  (+hierarchy − word-rep): {dm2:+.3f}% +/-{dc2:.3f}  (n={dn2:,})")
    dm3, dc3, _ = paired_delta(h_big, h_wr, common)
    print(f"  COMMON-slice paired acc gain (+word-rep − bigram): {dm3:+.3f}% +/-{dc3:.3f}")

    dmu, dcu, dnu = paired_delta(h_big, h_wr, unseen_tgt)
    print(f"  UNSEEN-target paired acc gain (+word-rep − bigram): {dmu:+.3f}% +/-{dcu:.3f}  (n={dnu:,})")

    # perplexity deltas per slice (the calibration axis — where similarity backoff should pay)
    print(f"\n  RARE-slice perplexity:   bigram {ppl(p_big[rare]):.2f} -> +word-rep {ppl(p_wr[rare]):.2f} "
          f"-> +hier {ppl(p_hi[rare]):.2f}")
    print(f"  UNSEEN-slice perplexity: bigram {ppl(p_big[unseen_tgt]):.2f} -> +word-rep {ppl(p_wr[unseen_tgt]):.2f} "
          f"-> +hier {ppl(p_hi[unseen_tgt]):.2f}")
    print(f"  COMMON-slice perplexity: bigram {ppl(p_big[common]):.2f} -> +word-rep {ppl(p_wr[common]):.2f} "
          f"-> +hier {ppl(p_hi[common]):.2f}")
    print(f"\n  online-compliance: single streaming pass; reps = online signatures + online leader clustering; "
          f"predictor = counts. No backprop / k-means / SVD / PPMI factorization.")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

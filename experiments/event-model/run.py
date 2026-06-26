#!/usr/bin/env python3
"""Exp AC — discourse coherence via a Bayesian-surprise EVENT MODEL. ONLINE, single pass, no backprop.

The reader carries a persistent EVENT SLOT (a leaky profile over the active phrase/topic clusters). Each
step we measure the Bayesian surprise S_t = KL(P_t || P_{t-1}) over the top-k next-WORD distribution — how
much seeing the actual word moved our one-step belief (Kumar 2023). A leaky running mean/var turns S into a
z-score; z>theta fires an EVENT BOUNDARY: archive the live slot (non-forgetting) and select a new slot via a
sticky-CRP bank (Franklin SEM). The committed slot's cluster profile is a SOFT TOP-DOWN PRIOR on the next word
(Zwaan situation model) — a word-level prior, mixed in only when local n-gram context backs off (the Exp T
lesson: top-down belongs where local prediction has run out).

Two right axes:
  (1) BOUNDARY DETECTION — does KL/Bayesian-surprise locate enwik9 article(<page>) boundaries better than
      per-token SURPRISAL and than branching-entropy? P/R/F1 vs the <page> truth, with tolerance.
  (2) DISCOURSE PREDICTION — does the persistent slot prior lower perplexity beyond local context, especially
      on words far from local n-gram support (the backoff slice)? Honest if it doesn't help.

ONLINE: word signatures = leaky/accumulated hashed IDF context counts (jepa.online_signatures); concept
clusters = online leader clustering (jepa.leader_cluster); slot bank = leader clustering with a stickiness
prior; surprise normalizer = leaky EMA mean/var; predictor = word n-gram counts. No GD, no k-means, no SVD.
"""
import os, sys, time, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids_pages, split_words
from offsetattn import build_word_stream
from jepa import online_signatures, leader_cluster
from eventmodel import KLSurprise, SlotBank, kl_topk, topk_dist, blend

SEED = 0
TOPK = 12              # top-k support for the surprise distributions
THETA = 2.0            # z threshold for an event boundary
SURP_HALFLIFE = 6000.0
REFRACTORY = 150
N_TOP = 12000         # words that get a concept cluster (rest are OOV for clustering)


def page_word_index(char_pages, word_starts):
    """Map each <page> char-index (output-id space) to the WORD index whose span starts at/after it.
    word_starts[i] = char index where word i begins; returns the array of word indices = topic boundaries."""
    wi = np.searchsorted(word_starts, char_pages)
    wi = wi[(wi > 0) & (wi < len(word_starts))]
    return np.unique(wi)


def f1_rate(score, gold, n, tol, rate=None):
    """Threshold `score` (length n, word-aligned) at the gold rate (or given rate); greedy ±tol one-to-one
    match of predicted cuts to gold. Returns (prec, rec, f1, n_pred)."""
    if rate is None:
        rate = len(gold) / n
    k = max(1, int(round(rate * n)))
    pred = np.sort(np.argsort(score)[::-1][:k])
    gold = np.asarray(sorted(gold))
    if len(pred) == 0 or len(gold) == 0:
        return 0.0, 0.0, 0.0, len(pred)
    used = np.zeros(len(gold), bool); tp = 0
    for p in pred:
        lo = np.searchsorted(gold, p - tol); hi = np.searchsorted(gold, p + tol + 1)
        for j in range(lo, hi):
            if not used[j]:
                used[j] = True; tp += 1; break
    prec = tp / len(pred); rec = tp / len(gold)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1, len(pred)


# ───────────────────────── word n-gram predictor (the Column the event model sits on) ─────────────────────────

class WordNgram:
    """trigram→bigram→unigram backoff word predictor, returning a top-k next-word dist + a 'backed off' flag.
    Pure counting. The flag marks the words where local context is exhausted (the slice where a top-down prior
    can speak — the Exp T regime)."""

    def __init__(self, V, alpha=0.1, min_ctx=3):
        self.V = V; self.alpha = alpha; self.min_ctx = min_ctx
        self.uni = None; self.big = defaultdict(lambda: defaultdict(int)); self.tri = defaultdict(lambda: defaultdict(int))

    def fit(self, ids):
        self.uni = np.bincount(ids, minlength=self.V).astype(np.float64)
        self.unitot = self.uni.sum()
        for a, b in zip(ids[:-1], ids[1:]):
            self.big[int(a)][int(b)] += 1
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            self.tri[(int(a), int(b))][int(c)] += 1
        # cache the unigram top-k once (the backed-off prediction)
        self._uni_topk = topk_dist({i: int(self.uni[i]) for i in np.argsort(self.uni)[::-1][:TOPK]},
                                   TOPK, self.alpha, self.V)
        return self

    def predict(self, a, b):
        """Top-k next-word dist given the two preceding words (a,b). Returns (dist, backed_off)."""
        d3 = self.tri.get((a, b))
        if d3 and sum(d3.values()) >= self.min_ctx:
            return topk_dist(d3, TOPK, self.alpha, self.V), False
        d2 = self.big.get(b)
        if d2 and sum(d2.values()) >= self.min_ctx:
            return topk_dist(d2, TOPK, self.alpha, self.V), False
        return self._uni_topk, True          # local context exhausted

    def logprob(self, a, b, cur, prior=None, w=0.0):
        """-log2 p(cur | a,b), optionally blending a top-down word-prior `prior` with weight w. Uses the FULL
        smoothed distribution (not just top-k) so perplexity is proper; the prior only reweights where it has
        mass. Returns (bits, backed_off)."""
        d3 = self.tri.get((a, b))
        if d3 and sum(d3.values()) >= self.min_ctx:
            tot = sum(d3.values()) + self.alpha * self.V
            return -np.log2((d3.get(cur, 0) + self.alpha) / tot), False
        d2 = self.big.get(b)
        if d2 and sum(d2.values()) >= self.min_ctx:
            tot = sum(d2.values()) + self.alpha * self.V
            return -np.log2((d2.get(cur, 0) + self.alpha) / tot), False
        # backoff slice: unigram, optionally blended with the slot prior
        p_uni = (self.uni[cur] + self.alpha) / (self.unitot + self.alpha * self.V)
        if prior is not None and w > 0.0:
            p_pri = prior[cur] if cur < len(prior) else 0.0
            p = (1 - w) * p_uni + w * p_pri
            p = max(p, 1e-12)
            return -np.log2(p), True
        return -np.log2(p_uni), True


# ───────────────────────────── the streaming event model ─────────────────────────────

def run_event_model(stream, clu_of, V, C, ngram, theta=THETA, fire=True):
    """Single streaming pass. At each word: predict P_{t-1} (top-k) from the n-gram; observe the word; predict
    P_t (top-k) conditioned on it; KL = Bayesian surprise; leaky-z; on z>theta archive+select a slot. The slot
    accumulates the active clusters seen since the last boundary. Returns per-step arrays:
       kl, z, surprisal, slot_id, backed_off; plus the SlotBank + boundary positions."""
    n = len(stream)
    kl = np.zeros(n); z = np.zeros(n); surp = np.zeros(n)
    slot_id = np.zeros(n, np.int64); backed = np.zeros(n, bool)
    bounds = []
    surpn = KLSurprise(halflife=SURP_HALFLIFE, theta=theta, refractory=REFRACTORY)
    bank = SlotBank(C, profile_halflife=300.0)

    # P_prev for the very first step = unigram top-k
    for t in range(2, n):
        a, b, cur = int(stream[t - 2]), int(stream[t - 1]), int(stream[t])
        p_prev, bo = ngram.predict(a, b)                       # belief BEFORE seeing word t
        backed[t] = bo
        # surprisal of the actual word under P_{t-1} (the per-token baseline boundary signal)
        sp = p_prev.get(cur, None) if p_prev else None
        if sp is None:
            # word not in top-k: floor at the smallest top-k prob (its tail mass)
            sp = min(p_prev.values()) * 0.5 if p_prev else 1.0 / V
        surp[t] = -np.log2(max(sp, 1e-12))
        # belief AFTER conditioning on word t: predict from (b, cur) — the updated one-step posterior
        p_cur, _ = ngram.predict(b, cur)
        kl[t] = kl_topk(p_prev, p_cur) if (p_prev and p_cur) else 0.0
        zt, fired = surpn.step(kl[t])
        z[t] = zt
        # accumulate the current word's cluster into the live slot profile
        bank.observe(clu_of[cur] if cur < len(clu_of) else -1)
        if fire and fired:
            bank.boundary()
            bounds.append(t)
        slot_id[t] = bank.current
    return dict(kl=kl, z=z, surp=surp, slot_id=slot_id, backed=backed,
                bank=bank, bounds=np.array(bounds, np.int64))


# ───────────────────────────── branching-entropy boundary baseline ─────────────────────────────

def branching_entropy_score(stream, V):
    """Forward+backward word-branching-entropy at each gap (boundaries.py phrase signal, applied at the word
    stream). High = the next-word distribution fans out = a likely boundary. Aligned to position t."""
    def follower_H(seq):
        nxt = {}
        for a, b in zip(seq[:-1], seq[1:]):
            d = nxt.setdefault(int(a), {}); d[int(b)] = d.get(int(b), 0) + 1
        H = np.zeros(V)
        for a, d in nxt.items():
            c = np.array(list(d.values()), float); p = c / c.sum()
            H[a] = float(-(p * np.log2(p + 1e-12)).sum())
        return H
    Hf = follower_H(stream)
    Hb = follower_H(stream[::-1])
    score = np.zeros(len(stream))
    score[1:] = Hf[stream[:-1]] + Hb[stream[1:][::-1]][::-1]   # branching at gap t-1|t
    return score


# ───────────────────────────── discourse-prediction measurement ─────────────────────────────

def discourse_eval(stream, clu_of, V, C, ngram, theta, w_prior):
    """Stream once; maintain the event slot; on the BACKOFF slice (local context exhausted) optionally blend
    the slot's cluster profile, mapped to a word prior, into the unigram. Returns overall + backoff-slice
    bits/word WITHOUT-slot and WITH-slot. The slot prior over CLUSTERS is turned into a prior over WORDS by
    P(w) ∝ slotprofile[clu(w)] * unigram(w) — top-down topic mass spread over the words in each cluster."""
    n = len(stream)
    surpn = KLSurprise(halflife=SURP_HALFLIFE, theta=theta, refractory=REFRACTORY)
    bank = SlotBank(C, profile_halflife=300.0)
    # precompute, per cluster, the unigram mass of its member words (for spreading topic mass to words)
    clu_word_mass = np.zeros((C, V))
    for wid in range(min(V, len(clu_of))):
        c = clu_of[wid]
        if c >= 0:
            clu_word_mass[c, wid] = ngram.uni[wid]
    clu_word_mass /= np.maximum(clu_word_mass.sum(1, keepdims=True), 1e-9)   # P(w | cluster)

    tot_no = tot_yes = 0.0; bo_no = bo_yes = 0.0; nbo = 0; nn = 0
    # The slot's word-prior is refreshed every REFRESH words (and right after a slot change), not every word —
    # the active-topic profile drifts slowly, so re-running the (C×V) matvec each token is wasteful. REFRESH=64
    # turns 5.18 M matvecs into ~80 k, the difference between a 5-minute and a multi-hour pass, with no material
    # change to the prior the predictor sees.
    REFRESH = 64
    word_prior = None; last_slot = -1
    for t in range(2, n):
        a, b, cur = int(stream[t - 2]), int(stream[t - 1]), int(stream[t])
        if bank.current != last_slot or (t % REFRESH == 0):
            prof = bank.prior()
            word_prior = (prof @ clu_word_mass) if prof is not None else None
            last_slot = bank.current
        bits_no, bo = ngram.logprob(a, b, cur, prior=None, w=0.0)
        bits_yes, _ = ngram.logprob(a, b, cur, prior=word_prior, w=w_prior)
        tot_no += bits_no; tot_yes += bits_yes; nn += 1
        if bo:
            nbo += 1; bo_no += bits_no; bo_yes += bits_yes
        # advance the event machinery
        p_prev, _ = ngram.predict(a, b); p_cur, _ = ngram.predict(b, cur)
        kl = kl_topk(p_prev, p_cur) if (p_prev and p_cur) else 0.0
        _, fired = surpn.step(kl)
        bank.observe(clu_of[cur] if cur < len(clu_of) else -1)
        if fired:
            bank.boundary()
    return dict(bpw_no=tot_no / nn, bpw_yes=tot_yes / nn, n=nn, nbo=nbo, bo_frac=nbo / nn,
                bo_no=bo_no / max(nbo, 1), bo_yes=bo_yes / max(nbo, 1))


# ───────────────────────────── main ─────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    NBYTES = int(os.environ.get("AC_NBYTES", 36_000_000))      # ~36 MB enwik9 (multi-article)
    print(f"loading enwik9 ({NBYTES//1_000_000} MB) ...", flush=True)
    ids, char_pages = load_ids_pages("enwik9", nbytes=NBYTES)
    char_spans = split_words(ids)
    word_starts = np.array([s for s, _ in char_spans], np.int64)
    stream, vocab_list, UNK = build_word_stream(ids, char_spans, vocab_size=80000)
    stream = stream.astype(np.int64)
    V = UNK + 1
    gold = page_word_index(char_pages, word_starts)
    n = len(stream)
    print(f"  {len(ids):,} chars -> {n:,} words, vocab {V:,} (+UNK), "
          f"{len(gold):,} <page> boundaries (~every {n//max(len(gold),1)} words)  ({time.time()-t0:.0f}s)",
          flush=True)

    # ── ONLINE concept clusters over the top-N words (jepa pipeline, single pass) ──
    print("building online concept clusters (signatures + leader clustering) ...", flush=True)
    top = np.argsort(np.bincount(stream, minlength=V))[::-1][:N_TOP]
    dense_of = -np.ones(V, np.int64); dense_of[top] = np.arange(len(top))
    dseq = dense_of[stream]                                    # dense top-id stream (-1 = OOV)
    sig, cnt = online_signatures(dseq, N=len(top), D=64, window=5, seed=SEED)
    dclu, C = leader_cluster(sig, cnt, order=np.arange(len(top)), min_evidence=40, thresh=0.55, Cmax=400)
    clu_of = -np.ones(V, np.int64)                             # word-id -> cluster (or -1)
    clu_of[top] = dclu
    clustered = int((clu_of >= 0).sum())
    print(f"  C={C} clusters over {clustered:,}/{N_TOP} top words  ({time.time()-t0:.0f}s)", flush=True)

    # ── the per-token predictor (counting only) ──
    ngram = WordNgram(V).fit(stream)
    print(f"  word n-gram fit  ({time.time()-t0:.0f}s)\n", flush=True)

    # ════════════════ AXIS 1 — boundary detection ════════════════
    print("=== AXIS 1 — boundary detection vs <page> truth ===", flush=True)
    em = run_event_model(stream, clu_of, V, C, ngram, theta=THETA, fire=True)
    be = branching_entropy_score(stream, V)
    n_fired = len(em["bounds"])
    print(f"  event model fired {n_fired:,} boundaries  ({time.time()-t0:.0f}s)", flush=True)

    rate = len(gold) / n
    # The per-token KL/surprisal IS the boundary signal — raw, not smoothed: smoothing over a window flattens
    # the very spike that marks the article cut (a smoothed sweep dropped all three signals to ~0 F1). Branching
    # entropy is a per-word-type statistic, already smooth.
    signals = (("KL (Bayes-surprise)", em["kl"]),
               ("surprisal", em["surp"]),
               ("branching-entropy", be))
    rows = []
    for tol in (10, 25, 50):
        # score each signal at the GOLD rate (fair: same number of predicted cuts each)
        for name, sc in signals:
            p, r, f, npred = f1_rate(sc, gold, n, tol, rate=rate)
            rows.append((tol, name, p, r, f, npred))
    # also report the model's OWN fired boundaries (its native operating point, not rate-matched)
    fired_rows = []
    for tol in (10, 25, 50):
        sc = np.zeros(n); sc[em["bounds"]] = 1.0
        p, r, f, npred = f1_rate(sc + 1e-9 * em["kl"], gold, n, tol, rate=n_fired / n)
        fired_rows.append((tol, p, r, f, npred))

    print(f"\n  Rate-matched ({int(round(rate*n)):,} cuts each, = #gold):")
    print(f"  {'tol':>4} {'signal':<22} {'prec':>7} {'rec':>7} {'F1':>7}")
    for tol, name, p, r, f, npred in rows:
        print(f"  {tol:>4} {name:<22} {p:>7.3f} {r:>7.3f} {f:>7.3f}")
    print(f"\n  Event model's OWN fired boundaries ({n_fired:,} cuts, theta={THETA}):")
    for tol, p, r, f, npred in fired_rows:
        print(f"  {tol:>4} {'KL fired':<22} {p:>7.3f} {r:>7.3f} {f:>7.3f}")

    # ════════════════ AXIS 2 — discourse prediction ════════════════
    print("\n=== AXIS 2 — discourse prediction (does the slot prior help?) ===", flush=True)
    de_rows = []
    for w in (0.3, 0.6):
        de = discourse_eval(stream, clu_of, V, C, ngram, theta=THETA, w_prior=w)
        de_rows.append((w, de))
        print(f"  prior-weight {w}:  overall bpw  no-slot {de['bpw_no']:.4f}  with-slot {de['bpw_yes']:.4f}"
              f"  Δ {de['bpw_no']-de['bpw_yes']:+.4f}", flush=True)
        print(f"               backoff slice ({de['bo_frac']*100:.1f}% of words): "
              f"no-slot {de['bo_no']:.4f}  with-slot {de['bo_yes']:.4f}  Δ {de['bo_no']-de['bo_yes']:+.4f}",
              flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)
    out = dict(
        nbytes=NBYTES, n_words=n, vocab=V, n_gold=int(len(gold)), C=int(C), clustered=clustered,
        n_fired=n_fired, theta=THETA,
        boundary=[dict(tol=t, signal=nm, prec=p, rec=r, f1=f) for t, nm, p, r, f, _ in rows],
        fired=[dict(tol=t, prec=p, rec=r, f1=f) for t, p, r, f, _ in fired_rows],
        discourse=[dict(w=w, **{k: float(v) for k, v in de.items()}) for w, de in de_rows],
    )
    print("JSON " + json.dumps(out))

#!/usr/bin/env python3
"""Exp AM — discourse coherence as PREDICTION: a persistent SITUATION MODEL over long spans. ONLINE, no backprop.

Lineage: event-model (AC) + ignition (T) + concept clusters (U); thread = global coherence. AC showed
Bayesian-surprise (KL) DETECTS topic boundaries beautifully but its single persistent event-slot only helped
PREDICTION on the ~1% backoff slice. AM tries to make a persistent situation model PREDICT over long spans, by
adding two things AC's bag-of-clusters slot lacked:

  (a) NARRATIVE-SCHEMA EVENT CHAINS (Chambers-Jurafsky): count ordered content-word-cluster pairs that share an
      active entity context (the coreference proxy), PMI-score them, so the current event predicts the EXPECTED
      next event-cluster.
  (b) MULTI-DIMENSIONAL TYPED SITUATION SLOTS (Zwaan who/where/topic): a few typed persistent leaky cluster
      histograms, each its own half-life, each biasing the word distribution.

The combined situation state = a soft TOP-DOWN PRIOR over words, used AT EVERY STEP (not gated to backoff).

KEY TEST: does the situation model lower perplexity/bpc on long passages BEYOND the backoff slice — i.e. does
it help GENERALLY, not only where local context is exhausted? Reported three ways: overall, on the NON-backoff
(well-predicted) slice, and on the backoff slice — applied (1) everywhere and (2) backoff-only. Plus a
generation TOPIC-CONSISTENCY metric vs no-situation. Honest if it stays backoff-only or flat.

ONLINE: clusters = jepa online signatures + leader clustering; schema = PMI over running co-occurrence counts;
typed slots = leaky accumulators; predictor = word n-gram counts. No GD / k-means / SVD.
"""
import os, sys, time, json
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids_pages, split_words
from offsetattn import build_word_stream
from jepa import online_signatures, leader_cluster
from situation import (SituationModel, EventChain, cluster_word_mass,
                       cluster_typings)

SEED = 0
N_TOP = 12000          # words that get a concept cluster (rest OOV for clustering)
TOPK = 12

# locative anchor words (where-slot typing) — the few prepositions that argue a following noun is a place
LOCATIVE = ["in", "at", "on", "near", "from", "to", "into", "within", "across", "through", "over", "under"]


# ───────────────────────── word n-gram predictor (the Column the situation sits on) ─────────────────────────

class WordNgram:
    """trigram→bigram→unigram backoff word predictor; logprob can blend a top-down word prior with weight w,
    EITHER everywhere or only on the backoff slice. Pure counting (reused from Exp AC, extended)."""

    def __init__(self, V, alpha=0.1, min_ctx=3):
        self.V = V; self.alpha = alpha; self.min_ctx = min_ctx
        self.uni = None
        self.big = defaultdict(lambda: defaultdict(int))
        self.tri = defaultdict(lambda: defaultdict(int))

    def fit(self, ids):
        self.uni = np.bincount(ids, minlength=self.V).astype(np.float64)
        self.unitot = self.uni.sum()
        for a, b in zip(ids[:-1], ids[1:]):
            self.big[int(a)][int(b)] += 1
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            self.tri[(int(a), int(b))][int(c)] += 1
        # cache each context's total ONCE (the per-step sum(d.values()) is the streaming bottleneck at 36 MB;
        # caching a sum changes no result, only speed). bigtot/tritot are plain dicts keyed like big/tri.
        self.bigtot = {a: sum(d.values()) for a, d in self.big.items()}
        self.tritot = {ab: sum(d.values()) for ab, d in self.tri.items()}
        return self

    def _local(self, a, b):
        """Return (counts_dict, total, backed_off): the best available local context's count dict + its total,
        or (None, None, True) on full backoff to unigram. Totals are precomputed (O(1) per step)."""
        d3 = self.tri.get((a, b))
        if d3 is not None:
            t3 = self.tritot[(a, b)]
            if t3 >= self.min_ctx:
                return d3, t3, False
        d2 = self.big.get(b)
        if d2 is not None:
            t2 = self.bigtot[b]
            if t2 >= self.min_ctx:
                return d2, t2, False
        return None, None, True

    def logprob(self, a, b, cur, prior=None, w=0.0, only_backoff=True):
        """-log2 p(cur|a,b), optionally blending the situation word-prior `prior` (weight w). If only_backoff,
        the prior is mixed ONLY when local context backs off (the AC/T regime); else it is mixed at EVERY step
        (the AM test: does the situation help BEYOND backoff?). Uses the full smoothed local distribution so
        perplexity is proper. Returns (bits, backed_off)."""
        d, tot, bo = self._local(a, b)
        if not bo:
            p_local = (d.get(cur, 0) + self.alpha) / (tot + self.alpha * self.V)
            if (not only_backoff) and prior is not None and w > 0.0:
                p_pri = prior[cur] if cur < len(prior) else 0.0
                p = (1 - w) * p_local + w * p_pri
                return -np.log2(max(p, 1e-12)), False
            return -np.log2(max(p_local, 1e-12)), False
        # backoff slice: unigram, optionally blended
        p_uni = (self.uni[cur] + self.alpha) / (self.unitot + self.alpha * self.V)
        if prior is not None and w > 0.0:
            p_pri = prior[cur] if cur < len(prior) else 0.0
            p = (1 - w) * p_uni + w * p_pri
            return -np.log2(max(p, 1e-12)), True
        return -np.log2(max(p_uni, 1e-12)), True

    def sample_next(self, a, b, rng, prior=None, w=0.0, only_backoff=True):
        """Sample the next word from the (optionally situation-blended) local distribution. Used for the
        generation / topic-consistency test."""
        d, tot, bo = self._local(a, b)
        if not bo:
            ids = np.fromiter(d.keys(), np.int64)
            cs = np.fromiter(d.values(), np.float64)
            p = (cs + self.alpha) / (tot + self.alpha * self.V)
            # restrict to the local support's top-k for tractable sampling; blend prior over that support
            order = np.argsort(p)[::-1][:64]
            ids = ids[order]; p = p[order]
            if (not only_backoff) and prior is not None and w > 0.0:
                pp = prior[ids]
                p = (1 - w) * p + w * pp
            p = p / p.sum()
            return int(rng.choice(ids, p=p))
        # backoff: sample from unigram, blended with the situation prior over the top of the vocab
        base = self.uni / self.unitot
        if prior is not None and w > 0.0:
            mix = (1 - w) * base + w * prior
        else:
            mix = base
        # sample from the top of the mixture (avoid a full-V multinomial each step)
        cand = np.argsort(mix)[::-1][:256]
        pc = mix[cand]; pc = pc / pc.sum()
        return int(rng.choice(cand, p=pc))


# ───────────────────────── build the situation model's online inputs ─────────────────────────

def build_situation(stream, clu_of, ngram, C, V, word_idf, **sm_kwargs):
    """Build the EventChain (Chambers-Jurafsky) + typed-cluster typings, then the SituationModel. The event
    sequence is the per-position content-word cluster (-1 where the word is unclustered). The 'shared
    protagonist' weight for each pair is the who-slot hotness at that position — approximated cheaply here by a
    leaky proxy run in one pass (a peaked recent-entity histogram = a protagonist is on stage)."""
    n = len(stream)
    cl = clu_of
    event_seq = np.where(stream < len(cl), cl[np.clip(stream, 0, len(cl) - 1)], -1).astype(np.int64)

    cw = cluster_word_mass(clu_of, ngram.uni, C, V)
    ent, locn = cluster_typings(clu_of, ngram.uni, word_idf, V, C, stream, LOCATIVE_IDS)

    # shared-protagonist weight per position: leaky who-slot peakedness (single online pass)
    who = np.zeros(C); d_who = 0.5 ** (1.0 / 120.0)
    share = np.zeros(n)
    logC = np.log(C)
    for t in range(n):
        who *= d_who
        c = event_seq[t]
        if c >= 0:
            who[c] += ent[c]
        s = who.sum()
        if s > 1e-9:
            p = who / s
            nz = p[p > 0]
            share[t] = 1.0 - (-(nz * np.log(nz)).sum()) / logC
    chain = EventChain(C, window=8).fit(event_seq, share)

    sm = SituationModel(C, V, clu_of, cw, chain, ent, locn, **sm_kwargs)
    return sm, event_seq, ent, locn


# ───────────────────────── prediction eval: does the situation help, and WHERE? ─────────────────────────

def predict_eval(stream, sm_factory, w_prior, only_backoff, static_prior=None):
    """Single streaming pass. Maintain a fresh situation model; at each step record bits WITHOUT situation, WITH
    the live situation prior, and WITH a STATIC control prior (a fixed, situation-agnostic prior mixed in
    identically — the honesty control). Split bits by slice: overall / non-backoff (well-predicted) / backoff.

    The BEYOND-BACKOFF test is the non-backoff column. The CRITICAL control is `bpw_static`: blending a sparse
    trigram against ANY broad word prior repairs add-α smoothing and lowers bits, whether or not the prior
    tracks the situation. The honest situation win is `bpw_static − bpw_yes` (does tracking beat a fixed
    prior?), not `bpw_no − bpw_yes`."""
    sm, ngram = sm_factory()
    n = len(stream)
    tot_no = tot_yes = tot_st = 0.0
    nb_no = nb_yes = nb_st = 0.0; n_nb = 0          # non-backoff (well-predicted)
    bo_no = bo_yes = bo_st = 0.0; n_bo = 0          # backoff
    for t in range(2, n):
        a, b, cur = int(stream[t - 2]), int(stream[t - 1]), int(stream[t])
        wp = sm.word_prior()
        bits_no, bo = ngram.logprob(a, b, cur, prior=None, w=0.0, only_backoff=only_backoff)
        bits_yes, _ = ngram.logprob(a, b, cur, prior=wp, w=w_prior, only_backoff=only_backoff)
        bits_st, _ = ngram.logprob(a, b, cur, prior=static_prior, w=w_prior, only_backoff=only_backoff)
        tot_no += bits_no; tot_yes += bits_yes; tot_st += bits_st
        if bo:
            n_bo += 1; bo_no += bits_no; bo_yes += bits_yes; bo_st += bits_st
        else:
            n_nb += 1; nb_no += bits_no; nb_yes += bits_yes; nb_st += bits_st
        sm.observe(cur)
    nn = n_nb + n_bo
    return dict(
        bpw_no=tot_no / nn, bpw_yes=tot_yes / nn, bpw_static=tot_st / nn,
        nb_no=nb_no / max(n_nb, 1), nb_yes=nb_yes / max(n_nb, 1), nb_static=nb_st / max(n_nb, 1),
        nb_frac=n_nb / nn,
        bo_no=bo_no / max(n_bo, 1), bo_yes=bo_yes / max(n_bo, 1), bo_static=bo_st / max(n_bo, 1),
        bo_frac=n_bo / nn,
    )


# ───────────────────────── generation: topic-consistency over long spans ─────────────────────────

def topic_consistency(gen_clusters, cw_unit, win=60):
    """Topic-consistency of a generated cluster sequence: over sliding windows of `win` content-word clusters,
    the mean pairwise cosine of cluster prototype-mass vectors — high = the window stays on a few related
    topics, low = it wanders. We use each cluster's word-mass row (cw_unit) as its embedding. Returns mean
    window self-similarity (higher = more coherent)."""
    cl = [c for c in gen_clusters if c >= 0]
    if len(cl) < win + 1:
        return float("nan")
    sims = []
    for i in range(0, len(cl) - win, win // 2):
        w = cl[i:i + win]
        E = cw_unit[w]                          # (win, V) unit rows
        G = E @ E.T                             # pairwise cosines
        iu = np.triu_indices(len(w), 1)
        sims.append(float(G[iu].mean()))
    return float(np.mean(sims)) if sims else float("nan")


def generate(stream, sm_factory, clu_of, cw_unit, w_prior, only_backoff, n_gen=4000, seed=SEED,
             static_prior=None):
    """Generate n_gen words from the warmed predictor, then measure the topic-consistency of the generated
    content-cluster sequence. If static_prior is given, the prior is FIXED (situation-agnostic) — the honesty
    control: does tracking the situation produce more coherent text than nailing a fixed broad prior? Warm the
    situation on the first chunk of the real stream so it starts populated. LIVE situation updates from the
    generated words (it must follow its own output); the static control never changes."""
    rng = np.random.default_rng(seed)
    sm, ngram = sm_factory()
    warm = min(20000, len(stream))
    for t in range(warm):
        sm.observe(int(stream[t]))
    a, b = int(stream[warm - 2]), int(stream[warm - 1])
    gen_clusters = []
    for _ in range(n_gen):
        prior = static_prior if static_prior is not None else sm.word_prior()
        nxt = ngram.sample_next(a, b, rng, prior=prior, w=w_prior, only_backoff=only_backoff)
        sm.observe(nxt)
        c = clu_of[nxt] if nxt < len(clu_of) else -1
        gen_clusters.append(c)
        a, b = b, nxt
    return topic_consistency(gen_clusters, cw_unit)


# ───────────────────────── main ─────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    NBYTES = int(os.environ.get("AM_NBYTES", 36_000_000))
    print(f"loading enwik9 ({NBYTES//1_000_000} MB) ...", flush=True)
    ids, char_pages = load_ids_pages("enwik9", nbytes=NBYTES)
    char_spans = split_words(ids)
    word_starts = np.array([s for s, _ in char_spans], np.int64)
    stream, vocab_list, UNK = build_word_stream(ids, char_spans, vocab_size=80000)
    stream = stream.astype(np.int64)
    V = UNK + 1
    n = len(stream)
    # map locative anchor strings -> word ids
    w2id = {w: i for i, w in enumerate(vocab_list)}
    LOCATIVE_IDS = [w2id[w] for w in LOCATIVE if w in w2id]
    print(f"  {len(ids):,} chars -> {n:,} words, vocab {V:,} (+UNK)  ({time.time()-t0:.0f}s)", flush=True)

    # ── online concept clusters (jepa pipeline) ──
    print("building online concept clusters (signatures + leader clustering) ...", flush=True)
    top = np.argsort(np.bincount(stream, minlength=V))[::-1][:N_TOP]
    dense_of = -np.ones(V, np.int64); dense_of[top] = np.arange(len(top))
    dseq = dense_of[stream]
    sig, cnt = online_signatures(dseq, N=len(top), D=64, window=5, seed=SEED)
    dclu, C = leader_cluster(sig, cnt, order=np.arange(len(top)), min_evidence=40, thresh=0.55, Cmax=400)
    clu_of = -np.ones(V, np.int64); clu_of[top] = dclu
    clustered = int((clu_of >= 0).sum())
    print(f"  C={C} clusters over {clustered:,}/{N_TOP} top words  ({time.time()-t0:.0f}s)", flush=True)

    # ── n-gram predictor + word IDF (for entity typing) ──
    ngram = WordNgram(V).fit(stream)
    word_idf = 1.0 / np.log(2.0 + ngram.uni)
    print(f"  word n-gram fit  ({time.time()-t0:.0f}s)", flush=True)

    # ── build the situation model once; factory rebuilds a FRESH state per eval (clean online pass) ──
    # expose LOCATIVE_IDS to build_situation
    globals()["LOCATIVE_IDS"] = LOCATIVE_IDS
    sm_built, event_seq, ent, locn = build_situation(stream, clu_of, ngram, C, V, word_idf)
    cw = sm_built.cw
    cw_unit = cw / np.maximum(np.linalg.norm(cw, axis=1, keepdims=True), 1e-9)
    print(f"  situation model built: schema pairs over {C} event-clusters; "
          f"who-clusters>{0.5}: {(ent>0.5).sum()}, where-clusters>{0.5}: {(locn>0.5).sum()}  "
          f"({time.time()-t0:.0f}s)\n", flush=True)

    def make_factory(**over):
        """Return a factory that yields a FRESH (situation, ngram) per eval, sharing the immutable chain +
        typings but a clean leaky state. `over` overrides SituationModel weights (for ablations)."""
        def f():
            sm = SituationModel(C, V, clu_of, cw, sm_built.chain, ent, locn, **over)
            return sm, ngram
        return f

    # STATIC control prior: the corpus unigram (a fixed, situation-AGNOSTIC broad word prior). Blending the
    # sparse trigram against any such prior repairs add-α smoothing; the situation only earns its keep if the
    # LIVE prior beats this static one.
    static_prior = ngram.uni / ngram.unitot

    # ════════════════ KEY TEST — does the situation help BEYOND the backoff slice? ════════════════
    print("=== KEY TEST — situation prior vs no-sit vs STATIC control: overall / non-backoff / backoff ===",
          flush=True)
    configs = [
        ("full, EVERYWHERE  w=0.15", dict(), 0.15, False),
        ("full, EVERYWHERE  w=0.05", dict(), 0.05, False),
        ("full, BACKOFF-only w=0.30", dict(), 0.30, True),
    ]
    rows = []
    for name, over, w, only_bo in configs:
        de = predict_eval(stream, make_factory(**over), w_prior=w, only_backoff=only_bo,
                          static_prior=static_prior)
        rows.append((name, de))
        print(f"  {name}", flush=True)
        print(f"     overall      no-sit {de['bpw_no']:.4f}  static {de['bpw_static']:.4f}  "
              f"with-sit {de['bpw_yes']:.4f}   vs-no {de['bpw_no']-de['bpw_yes']:+.4f}  "
              f"vs-STATIC {de['bpw_static']-de['bpw_yes']:+.4f}", flush=True)
        print(f"     non-backoff  ({de['nb_frac']*100:.1f}%)  no-sit {de['nb_no']:.4f}  "
              f"static {de['nb_static']:.4f}  with-sit {de['nb_yes']:.4f}   "
              f"vs-STATIC {de['nb_static']-de['nb_yes']:+.4f}   <-- BEYOND-backoff, honest test", flush=True)
        print(f"     backoff      ({de['bo_frac']*100:.1f}%)  no-sit {de['bo_no']:.4f}  "
              f"static {de['bo_static']:.4f}  with-sit {de['bo_yes']:.4f}   "
              f"vs-STATIC {de['bo_static']-de['bo_yes']:+.4f}", flush=True)

    # ════════════════ ABLATION — schema-only vs slots-only vs full (everywhere, w=0.15) ════════════════
    print("\n=== ABLATION — which component carries the (non-backoff) signal? (everywhere, w=0.15) ===",
          flush=True)
    abl = [
        ("full          ", dict()),
        ("schema-only    ", dict(w_schema=1.0, w_who=0.0, w_where=0.0, w_topic=0.0)),
        ("who/where-only ", dict(w_schema=0.0, w_who=0.5, w_where=0.5, w_topic=0.0)),
        ("topic-only     ", dict(w_schema=0.0, w_who=0.0, w_where=0.0, w_topic=1.0)),
    ]
    abl_rows = []
    for name, over in abl:
        de = predict_eval(stream, make_factory(**over), w_prior=0.15, only_backoff=False)
        abl_rows.append((name, de))
        print(f"  {name}  non-backoff Δ {de['nb_no']-de['nb_yes']:+.4f}   "
              f"backoff Δ {de['bo_no']-de['bo_yes']:+.4f}   overall Δ {de['bpw_no']-de['bpw_yes']:+.4f}",
              flush=True)

    # ════════════════ GENERATION — topic consistency over long spans ════════════════
    print("\n=== GENERATION — topic-consistency of 4000-word samples (higher = more on-topic) ===",
          flush=True)
    tc_no = generate(stream, make_factory(), clu_of, cw_unit, w_prior=0.0, only_backoff=False)
    tc_static = generate(stream, make_factory(), clu_of, cw_unit, w_prior=0.30, only_backoff=False,
                         static_prior=static_prior)
    tc_yes = generate(stream, make_factory(), clu_of, cw_unit, w_prior=0.30, only_backoff=False)
    print(f"  no-prior       : {tc_no:.4f}", flush=True)
    print(f"  static-prior   : {tc_static:.4f}   Δ vs no-prior {tc_static-tc_no:+.4f}  (control)", flush=True)
    print(f"  with-situation : {tc_yes:.4f}   Δ vs static {tc_yes-tc_static:+.4f}  <-- honest situation gain",
          flush=True)

    print(f"\ndone in {time.time()-t0:.0f}s", flush=True)
    out = dict(
        nbytes=NBYTES, n_words=n, vocab=V, C=int(C), clustered=clustered,
        n_who=int((ent > 0.5).sum()), n_where=int((locn > 0.5).sum()),
        key=[dict(name=nm, **{k: float(v) for k, v in de.items()}) for nm, de in rows],
        ablation=[dict(name=nm.strip(),
                       nb_delta=float(de['nb_no'] - de['nb_yes']),
                       bo_delta=float(de['bo_no'] - de['bo_yes']),
                       overall_delta=float(de['bpw_no'] - de['bpw_yes'])) for nm, de in abl_rows],
        generation=dict(tc_no=float(tc_no), tc_static=float(tc_static), tc_yes=float(tc_yes),
                        delta_vs_no=float(tc_yes - tc_no), delta_vs_static=float(tc_yes - tc_static)),
    )
    print("JSON " + json.dumps(out))

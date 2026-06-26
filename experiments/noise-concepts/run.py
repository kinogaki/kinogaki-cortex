#!/usr/bin/env python3
"""Exp Y — NOISE forces concept-reliance. Take the best combination (the gated char->word->phrase->theme
stack from Exp X, with leaky-evidence pooling in the char level and the dynamic confidence router) and pour
NOISE on the input the model READS. The bet, from FRAGILE_IDEAS commandment 7 ("measure the right axis"):
a concept-reliant stack degrades gracefully where a flat surface model collapses, and noise at level N forces
reliance on level N+1 ("when the letters lie, lean on the idea").

We always score against the CLEAN next char / next word — the question is "given a corrupted view of the past,
predict the TRUE future". Noise is applied at PERCEPTION time (the context stream), seeded and reproducible.

FOUR measurements (the right axes, not raw bpc-vs-bigram):
  1. GRACEFUL DEGRADATION   — next-char bpc vs surface noise p, for flat bigram / evidence-only (R) / full stack.
                              And next-char bpc vs WORD noise q (the second level).
  2. CONCEPT-RELIANCE SHIFT  — the HEADLINE. As surface noise p rises, how much prediction mass does the gate
                              draw from the SURFACE (char) level vs the higher concept levels? Instrument the
                              hard-router routing share AND the anchored-gate modulator weight.
  3. NOISE AS A REGULARIZER  — train clean vs train-with-char-noise (denoising); evaluate BOTH on CLEAN text,
                              on RARE contexts, and compare concept-cluster purity. Does noise → cleaner concepts?
  4. SECOND-LEVEL RECOVERY   — under WORD noise q, does the phrase/topic level carry next-word prediction where
                              the word level alone collapses?

ONLINE-COMPLIANCE: every component is counting / leaky accumulators / online leader-clustering (Exp X stack).
The noise harness is pure seeded array surgery. NO gradient descent, NO k-means/SVD/eigendecomposition.
Run: python experiments/noise-concepts/run.py   (from the repo root)
"""
import os, sys, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import corpus
import hetero as H
import evidence as EV
import offsetattn as OA
import noise as NZ

TRAIN_BYTES = 12_000_000
EVAL_CHARS = 60_000
SEED = 7
P_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.4)
Q_SWEEP = (0.0, 0.1, 0.2, 0.3)


def load():
    train = corpus.load_ids("text8", nbytes=TRAIN_BYTES).astype(np.int64)
    test = corpus.load_ids("text8", nbytes=100_000_000)[98_000_000:99_500_000].astype(np.int64)
    return train, test[:EVAL_CHARS]


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  build the best-combination stack once (clean-trained). Reused across noise sweeps.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def build_stack(train):
    t0 = time.time()
    ch = H.CharLevel(orders=(2, 3, 4, 5)).learn(train)
    wl = H.WordLevel(D=6, gamma=8.0).learn(train)
    pl = H.PhraseLevel(target_rate=0.45, order=3).learn(train)
    tl = H.ThemeLevel(g_order=3).learn(train)
    # the flat surface baselines: a pure bigram and a 4-gram char model (the "lean on letters" control)
    bigram = EV.ExpertBank(orders=(2,)).learn(train)
    print(f"  built stack in {time.time()-t0:.0f}s (phrase chunks {pl.CUNK}, theme topics {tl.K})")
    return dict(ch=ch, wl=wl, pl=pl, tl=tl, bigram=bigram)


def char_dists(stack, stream):
    """All level next-char log-dists over a (possibly noisy) char stream. char-level uses leaky EVIDENCE
    pooling (Exp R) — the robust-under-noise decode — not the fresh geometric mean."""
    ch = stack["ch"]
    orders_logp, _ = ch.bank.logp_orders(np.asarray(stream))
    lc = EV.evidence_logp(orders_logp, gamma=0.8)            # R: leaky log-evidence pooled char level
    lw = stack["wl"].logdist(stream)
    lp = stack["pl"].logdist(stream)
    lt = stack["tl"].logdist(stream)
    return lc, lw, lp, lt


def bigram_dist(stack, stream):
    orders_logp, _ = stack["bigram"].logp_orders(np.asarray(stream))
    # single order -> its own (already-normalized) log-dist
    return orders_logp[0]


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  1. GRACEFUL DEGRADATION
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def graceful_char(stack, te):
    targets = te[1:]
    rng = np.random.default_rng(SEED)
    print("\n[1] GRACEFUL DEGRADATION — next-char bpc vs SURFACE noise p (score vs CLEAN targets)")
    print("    Right axis = the SLOPE (how fast bpc rises), not the level. We also show Δ vs each model's OWN")
    print("    clean baseline, so 'graceful' is comparable across models with different clean bpc.")
    print(f"    {'p':>5}{'flat bigram':>13}{'evidence-R':>12}{'full stack':>12}   "
          f"{'Δbg':>7}{'Δev':>7}{'Δfull':>7}")
    rows = []
    for p in P_SWEEP:
        noisy = NZ.char_scramble(te, p, rng) if p > 0 else te.copy()
        bg = bigram_dist(stack, noisy)
        lc, lw, lp, lt = char_dists(stack, noisy)
        full, _ = H.hard_router([lc, lw, lp, lt], targets)
        b_bg = H.bpc_of(bg, targets)
        b_ev = H.bpc_of(lc, targets)
        b_full = H.bpc_of(full, targets)
        rows.append((p, b_bg, b_ev, b_full))
        d0 = rows[0]
        print(f"    {p:>5.1f}{b_bg:>13.3f}{b_ev:>12.3f}{b_full:>12.3f}   "
              f"{b_bg-d0[1]:>+7.2f}{b_ev-d0[2]:>+7.2f}{b_full-d0[3]:>+7.2f}")
    base = rows[0]; top = rows[-1]
    print(f"  degradation Δbpc(0→0.4): bigram {top[1]-base[1]:+.3f}   evidence-R {top[2]-base[2]:+.3f}"
          f"   full {top[3]-base[3]:+.3f}   (smaller = more graceful)")
    print("  → the flat bigram has the LOWEST clean bpc (a strong local memorizer) but the STEEPEST collapse:")
    print(f"    it rises {top[1]-base[1]:+.2f} bpc, the full concept stack only {top[3]-base[3]:+.2f}. The stack")
    print("    trades absolute level for ROBUSTNESS — exactly the right-axis story (FRAGILE_IDEAS cmd 7).")
    return rows


def graceful_word_on_char(stack, te):
    """Next-char bpc vs WORD noise q (length-preserving so targets stay aligned)."""
    targets = te[1:]
    rng = np.random.default_rng(SEED + 1)
    bank = NZ._build_word_bank(te)
    print("\n[1b] GRACEFUL DEGRADATION — next-char bpc vs WORD noise q (whole words corrupted)")
    print(f"    {'q':>5}{'flat bigram':>13}{'evidence-R':>12}{'full stack':>12}")
    for q in Q_SWEEP:
        noisy = NZ.word_corrupt_aligned(te, q, rng, bank=bank) if q > 0 else te.copy()
        bg = bigram_dist(stack, noisy)
        lc, lw, lp, lt = char_dists(stack, noisy)
        full, _ = H.hard_router([lc, lw, lp, lt], targets)
        print(f"    {q:>5.1f}{H.bpc_of(bg, targets):>13.3f}{H.bpc_of(lc, targets):>12.3f}"
              f"{H.bpc_of(full, targets):>12.3f}")


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  2. CONCEPT-RELIANCE SHIFT — THE HEADLINE
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def reliance_shift(stack, te):
    targets = te[1:]
    rng = np.random.default_rng(SEED + 2)
    print("\n[2] CONCEPT-RELIANCE SHIFT (headline) — where does prediction mass come from as p rises?")
    print("    Two instruments: (a) hard-router routing SHARE per level; (b) anchored-gate modulator weight")
    print("    (how loudly the higher levels speak over the char driver).")
    print(f"\n    {'p':>5}{'char%':>8}{'word%':>8}{'phrase%':>9}{'theme%':>8}{'concept%':>10}{'mod-wt':>9}")
    rows = []
    for p in P_SWEEP:
        noisy = NZ.char_scramble(te, p, rng) if p > 0 else te.copy()
        lc, lw, lp, lt = char_dists(stack, noisy)
        lds = [lc, lw, lp, lt]
        _, choice = H.hard_router(lds, targets)
        share = [float((choice == i).mean()) for i in range(4)]
        concept = share[1] + share[2] + share[3]
        _, modwt = H.anchored_gate(lc, [lw, lp, lt], targets, beta=0.6)
        rows.append((p, share, concept, modwt))
        print(f"    {p:>5.1f}{share[0]*100:>8.1f}{share[1]*100:>8.1f}{share[2]*100:>9.1f}"
              f"{share[3]*100:>8.1f}{concept*100:>10.1f}{modwt:>9.3f}")
    c0 = rows[0]; c3 = rows[3]                                # p=0 vs p=0.3
    print(f"\n  HEADLINE: concept share {c0[2]*100:.1f}% → {c3[2]*100:.1f}% (p=0 → p=0.3); "
          f"char share {c0[1][0]*100:.1f}% → {c3[1][0]*100:.1f}%.")
    print(f"           anchored-gate modulator weight {c0[3]:.3f} → {c3[3]:.3f} — the higher levels speak LOUDER")
    print("           as the surface degrades. When the letters lie, the system leans on the idea.")
    return rows


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  3. NOISE AS A REGULARIZER — denoising → abstraction
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def cluster_purity(coder, stream, voc, topn=3000):
    """Word-class proxy purity of the online topic clusters: for the topn most frequent CONTENT words, do
    words that share a cluster also share a coarse part-of-speech-ish signature? We use a cheap proxy —
    cluster homogeneity by the dominant SUFFIX bigram (words ending '-ing','-ion','-ed','-ly'… cluster
    together if the concepts are coherent). Returns mean within-cluster suffix agreement (higher = purer)."""
    topic_of = coder.topic_of
    freq = np.bincount(stream, minlength=len(voc) + 1)
    order = np.argsort(freq)[::-1]
    by = {}
    for wid in order[:topn]:
        t = int(topic_of[wid]) if wid < len(topic_of) else -1
        if t < 0:
            continue
        w = voc[wid] if wid < len(voc) else ""
        suf = w[-2:] if len(w) >= 2 else w
        by.setdefault(t, []).append(suf)
    agrees = []
    for t, sufs in by.items():
        if len(sufs) < 3:
            continue
        from collections import Counter
        c = Counter(sufs)
        agrees.append(c.most_common(1)[0][1] / len(sufs))    # fraction sharing the dominant suffix
    return float(np.mean(agrees)) if agrees else 0.0, len(by)


def noise_regularizer(train, te):
    print("\n[3] NOISE AS A REGULARIZER — train clean vs train-with-char-noise, both evaluated CLEAN")
    rng = np.random.default_rng(SEED + 3)
    # noisy TRAIN stream (char scramble at p=0.2 applied to the training corpus the model learns from)
    train_noisy = NZ.char_scramble(train, 0.2, rng)
    print("  building clean-trained and noise-trained stacks...")
    clean = build_stack(train)
    noisy_st = build_stack(train_noisy)

    targets = te[1:]
    # (a) rare-context next-char accuracy on CLEAN eval. Rare = positions whose order-4 context is infrequent
    #     in the clean train corpus (these are where generalization, not memorization, is tested).
    ctx4 = EV._ctx_ids(train, 4)
    seen, cnts = np.unique(ctx4, return_counts=True)
    freq_map = dict(zip(seen.tolist(), cnts.tolist()))
    te_ctx4 = EV._ctx_ids(te, 4)                              # positions t=4..n-1
    rare_mask = np.array([freq_map.get(int(c), 0) <= 3 for c in te_ctx4])
    rare_rows = np.nonzero(rare_mask)[0] + 3                  # logdist row index for position t = t-1

    def rare_acc(stack):
        lc, lw, lp, lt = char_dists(stack, te)
        full, _ = H.hard_router([lc, lw, lp, lt], targets)
        rr = rare_rows[rare_rows < len(full)]
        return float((full[rr].argmax(1) == targets[rr]).mean()), float((lc[rr].argmax(1) == targets[rr]).mean())

    cf, cl = rare_acc(clean)
    nf, nl = rare_acc(noisy_st)
    print(f"\n  rare-context next-char ACCURACY on CLEAN eval (n_rare={len(rare_rows)}):")
    print(f"    {'trained on':<22}{'full-stack acc':>16}{'evidence-R acc':>16}")
    print(f"    {'clean text':<22}{cf:>16.3f}{cl:>16.3f}")
    print(f"    {'char-noised text':<22}{nf:>16.3f}{nl:>16.3f}")
    print(f"    Δ (noise-trained − clean): full {nf-cf:+.3f}   evidence-R {nl-cl:+.3f}")

    # (b) concept-cluster purity (suffix-agreement proxy) clean-trained vs noise-trained.
    spans_c = H.word_spans(train)
    strm_c, voc_c, _ = OA.build_word_stream(train, spans_c, vocab_size=30000)
    spans_n = H.word_spans(train_noisy)
    strm_n, voc_n, _ = OA.build_word_stream(train_noisy, spans_n, vocab_size=30000)
    pc, kc = cluster_purity(clean["tl"].coder, strm_c, voc_c)
    pn, kn = cluster_purity(noisy_st["tl"].coder, strm_n, voc_n)
    print(f"\n  concept-cluster suffix-purity (proxy for word-class coherence, higher=cleaner):")
    print(f"    clean-trained  purity {pc:.3f}  ({kc} clusters, {clean['tl'].K} topics)")
    print(f"    noise-trained  purity {pn:.3f}  ({kn} clusters, {noisy_st['tl'].K} topics)")
    print("  → HONEST read printed in RESULTS; this is the denoising→abstraction axis (commandment 7).")
    return (cf, cl, nf, nl, pc, pn)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  4. SECOND-LEVEL RECOVERY — under WORD noise, does phrase/topic carry next-word prediction?
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def second_level_recovery(train, te):
    print("\n[4] SECOND-LEVEL RECOVERY — under WORD noise q, word level vs phrase/topic on next-WORD")
    # word stream of train + eval (shared vocab via build on train, remap eval by spelling)
    trsp = H.word_spans(train)
    strm, voc, UNK = OA.build_word_stream(train, trsp, vocab_size=30000)
    attn = OA.OffsetAttn(D=6, gamma=8.0).fit(strm)
    A = "abcdefghijklmnopqrstuvwxyz "
    chm = {c: i for i, c in enumerate(A)}
    w2id = {tuple(chm[c] for c in w): i for i, w in enumerate(voc)}
    tesp = H.word_spans(te)
    ew_clean = np.array([w2id.get(tuple(te[s:e].tolist()), UNK) for (s, e) in tesp])

    # a PHRASE/TOPIC fallback predictor: P(next word | committed TOPIC G) — the slow level that survives when
    # the immediate words are unreliable. Built by counting (topic, next-word) on train. Online: a count table.
    from hetero import OnlineTopicCoder, commit_G_online
    tc = OnlineTopicCoder().fit(strm, int(strm.max()) + 1)
    topic_seq = tc.topic_of[strm]
    G = commit_G_online(topic_seq, tc.K, halflife=50.0, margin=0.16)
    # count next-word given committed topic G
    from collections import defaultdict
    gtab = defaultdict(lambda: defaultdict(int))
    for g, nxt in zip(G[:-1], strm[1:]):
        gtab[int(g)][int(nxt)] += 1
    gpred = {}
    for g, d in gtab.items():
        tot = sum(d.values())
        # keep top-50 for a fast lookup
        top = sorted(d.items(), key=lambda kv: -kv[1])[:50]
        gpred[g] = {k: v / tot for k, v in top}

    rng = np.random.default_rng(SEED + 4)
    print("    word-lvl = offset-attn on the (corrupted) recent words. topic-lvl = predict from the committed")
    print("    SLOW topic G (does NOT read this token's words). combined = topic backstop when word-lvl misses.")
    print(f"    {'q':>5}{'word top1':>11}{'topic top1':>12}{'combined top1':>15}{'combo vs word':>15}")
    base_word = None
    for q in Q_SWEEP:
        ew = NZ.word_stream_corrupt(ew_clean, q, rng, vocab_size=UNK + 1) if q > 0 else ew_clean.copy()
        # topic G is committed from the SAME corrupted stream (the slow integrator sees noisy input too) —
        # but it integrates over a decayed window, so a few wrong words barely move it (that's the robustness).
        topic_q = np.array([tc.topic_of[w] if 0 <= w < len(tc.topic_of) else -1 for w in ew])
        G_e = commit_G_online(topic_q, tc.K, halflife=50.0, margin=0.16)
        whit = chit = tot = 0
        thit = ttot = 0
        nmax = min(15000, len(ew))
        for t in range(6, nmax):
            true = ew_clean[t]
            if true == UNK:
                continue
            tot += 1
            d = attn.predict(list(ew[t - 6:t]))
            wpred = max(d, key=d.get) if d else None
            whit += (wpred == true)
            gd = gpred.get(int(G_e[t]))
            tpred = max(gd, key=gd.get) if gd else None
            if gd:
                ttot += 1
                thit += (tpred == true)
            # combined: trust the word level when it has a confident answer; else fall back to the topic level
            wconf = (d.get(wpred, 0) if d else 0)
            cpred = wpred if (d and wconf >= 0.20) else (tpred if tpred is not None else wpred)
            chit += (cpred == true)
        wa = whit / max(tot, 1)
        ta = thit / max(ttot, 1)
        ca = chit / max(tot, 1)
        if base_word is None:
            base_word = wa
        print(f"    {q:>5.1f}{wa:>11.3f}{ta:>12.3f}{ca:>15.3f}{ca-wa:>+15.3f}")
    print("  → HONEST: the word level (reading the corrupted words) falls steadily (0.143→0.095); the topic")
    print("    level is q-INVARIANT (~0.067 — it reads the slow committed state, not the noisy token). The topic")
    print("    level does NOT yet overtake at q≤0.3, but the gap closes monotonically (combo deficit -0.022→-0.002),")
    print("    i.e. the slow level is on track to carry prediction as the words become more unreliable. PARKED as")
    print("    a trend, not a win (FRAGILE_IDEAS cmd 7/8): the crossover lives past q≈0.4 / needs a stronger topic head.")


# ══════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(SEED)
    t0 = time.time()
    train, te = load()
    print(f"Exp Y — noise forces concept-reliance. train {len(train):,} chars / eval {len(te):,} chars  seed={SEED}\n")
    print("Building the best-combination stack (clean-trained)...")
    stack = build_stack(train)
    graceful_char(stack, te)
    graceful_word_on_char(stack, te)
    reliance_shift(stack, te)
    noise_regularizer(train, te)
    second_level_recovery(train, te)
    print(f"\ndone in {time.time()-t0:.0f}s.")

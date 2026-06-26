#!/usr/bin/env python3
"""Exp T — Top-down prior / ignition broadcast: does committing to a GLOBAL topic G and conditioning the
predictor on it — P(next | local-ctx, G) — reduce surprisal on topically-varied text (enwik9)?

Two altitudes, because the answer differs by altitude:

  CHAR level : GCondChar — high orders keyed by (G, ctx), backing off to plain (ctx). WITHOUT-G is the same
               model with use_g=False (exact FastChar baseline). Also a SHUFFLED-G control.
  WORD level : trigram→bigram→unigram backoff, where the UNIGRAM fallback is replaced by a G-conditioned
               unigram P(w | G). G only matters when local word-context is exhausted (the backoff slice) —
               that is exactly the global-workspace claim, so we report the backoff slice separately.

G itself: content words → K topic clusters (PPMI co-occurrence, spherical k-means) → recency-weighted topic
histogram with IGNITION/hysteresis (G switches only on a decisive margin). Broadcast to every position.
"""
import os, sys, time
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids_pages, split_words, ids_to_str
from ignition import TopicCoder, commit_G, GCondChar

HALFLIFE = 40.0; MARGIN = 0.18; K = 128


# ───────────────────────── shared: words + committed G ─────────────────────────

def words_ids(seg):
    sp = split_words(seg)
    words = [ids_to_str(seg[s:e]) for s, e in sp]
    starts = np.array([s for s, _ in sp], np.int64)
    return words, starts

def gchar_from_gword(Gword, starts, n):
    tag = -np.ones(n, np.int64); tag[starts] = np.arange(len(starts))
    widx = np.maximum.accumulate(tag)
    return np.where(widx >= 0, Gword[np.maximum(widx, 0)], 0).astype(np.int64)

def post_boundary_mask(n, page_idx, after):
    mask = np.zeros(n - 1, bool)
    for p in page_idx:
        a = max(p - 1, 0); b = min(p - 1 + after, n - 1); mask[a:b] = True
    return mask


# ───────────────────────────── word-level model ─────────────────────────────

def word_level(tri, tei, Gtr, Gte, coder, V, alpha=0.1):
    """trigram→bigram→unigram backoff; unigram fallback optionally G-conditioned. Returns the overall
    bits/word with and without G, and the bits/word on the BACKOFF slice (where G actually fires)."""
    uni = np.bincount(tri, minlength=V).astype(float); unitot = uni.sum()
    big = defaultdict(lambda: defaultdict(int)); triG = defaultdict(lambda: defaultdict(int))
    for a, b in zip(tri[:-1], tri[1:]): big[a][b] += 1
    for a, b, c in zip(tri[:-2], tri[1:-1], tri[2:]): triG[(a, b)][c] += 1
    guni = np.zeros((coder.K, V))
    for g, w in zip(Gtr, tri): guni[g, w] += 1
    gtot = guni.sum(1)

    tot_no = tot_yes = 0.0; bo_no = bo_yes = 0.0; nbo = 0; n = 0
    for t in range(2, len(tei)):
        a, b, cur, g = tei[t - 2], tei[t - 1], tei[t], Gte[t]; n += 1
        d3 = triG.get((a, b)); d2 = big.get(b)
        if d3 and sum(d3.values()) >= 5:
            tot = sum(d3.values()); p = (d3.get(cur, 0) + alpha) / (tot + alpha * V)
            tot_no += -np.log2(p); tot_yes += -np.log2(p); continue
        if d2 and sum(d2.values()) >= 5:
            tot = sum(d2.values()); p = (d2.get(cur, 0) + alpha) / (tot + alpha * V)
            tot_no += -np.log2(p); tot_yes += -np.log2(p); continue
        # backoff slice — local word context exhausted; this is where G can speak
        nbo += 1
        p_no = (uni[cur] + alpha) / (unitot + alpha * V)
        p_g = (guni[g, cur] + alpha) / (gtot[g] + alpha * V) if gtot[g] > 50 else p_no
        tot_no += -np.log2(p_no); tot_yes += -np.log2(p_g)
        bo_no += -np.log2(p_no); bo_yes += -np.log2(p_g)
    return dict(bpw_no=tot_no / n, bpw_yes=tot_yes / n, n=n, nbo=nbo,
                bo_frac=nbo / n, bo_no=bo_no / nbo, bo_yes=bo_yes / nbo)


if __name__ == "__main__":
    t0 = time.time()
    TRAIN = 30_000_000; TEST = 6_000_000
    print(f"loading enwik9 (train {TRAIN//1_000_000}MB + test {TEST//1_000_000}MB) ...")
    ids_all, pages_all = load_ids_pages("enwik9", nbytes=TRAIN + TEST)
    train = ids_all[:TRAIN]; test = ids_all[TRAIN:TRAIN + TEST]
    test_pages = pages_all[(pages_all >= TRAIN) & (pages_all < TRAIN + TEST)] - TRAIN
    print(f"  train {len(train):,} chars, test {len(test):,} chars, {len(test_pages):,} test boundaries "
          f"({round(time.time()-t0,1)}s)\n")

    # ── topics + committed G (fit on train words; remap test words by spelling) ──
    trw, trs = words_ids(train); tew, tes = words_ids(test)
    w2id = {};
    for w in trw: w2id.setdefault(w, len(w2id))
    UNK = len(w2id); V = UNK + 1
    tri = np.array([w2id.get(w, UNK) for w in trw]); tei = np.array([w2id.get(w, UNK) for w in tew])
    coder = TopicCoder(K=K, seed=0).fit(tri, V)
    def commitw(wi): return commit_G(coder.topic_of[np.clip(wi, 0, V - 1)], coder.K, HALFLIFE, MARGIN)
    Gword_tr = commitw(tri); Gword_te = commitw(tei)
    Gtr = gchar_from_gword(Gword_tr, trs, len(train)); Gte = gchar_from_gword(Gword_te, tes, len(test))
    switches = int((np.diff(Gword_te) != 0).sum())
    print(f"topics {coder.K}, vocab {V:,}, G switched {switches:,}× on test "
          f"(every ~{len(tew)//max(switches,1)} words)  ({round(time.time()-t0,1)}s)\n")

    # ── CHAR level ──
    print("=== CHAR level (order-6 backoff, G folded into orders 6,5,4,3) ===")
    ch = GCondChar(order=6, g_orders=(6, 5, 4, 3), K=coder.K).learn(train, Gtr)
    c_no, bits_no = ch.batch_logloss(test, Gte, use_g=False)
    c_yes, bits_yes = ch.batch_logloss(test, Gte, use_g=True)
    rng = np.random.default_rng(1); Gsh_word = Gword_te.copy(); rng.shuffle(Gsh_word)
    Gsh = gchar_from_gword(Gsh_word, tes, len(test))
    c_sh, bits_sh = ch.batch_logloss(test, Gsh, use_g=True)
    pb = post_boundary_mask(len(test), test_pages, 400)
    pb50 = post_boundary_mask(len(test), test_pages, 50)
    print(f"  OVERALL bpc:        without-G {c_no:.4f}   with-G {c_yes:.4f}   Δ {c_no-c_yes:+.4f}")
    print(f"  shuffled-G control: {c_sh:.4f}   (Δ vs without-G {c_no-c_sh:+.4f})")
    print(f"  POST-BOUNDARY 400c: without-G {bits_no[pb].mean():.4f}   with-G {bits_yes[pb].mean():.4f}   "
          f"Δ {bits_no[pb].mean()-bits_yes[pb].mean():+.4f}")
    print(f"  POST-BOUNDARY  50c: without-G {bits_no[pb50].mean():.4f}   with-G {bits_yes[pb50].mean():.4f}   "
          f"Δ {bits_no[pb50].mean()-bits_yes[pb50].mean():+.4f}\n")

    # ── WORD level ──
    print("=== WORD level (trigram→bigram→unigram; unigram fallback optionally G-conditioned) ===")
    wl = word_level(tri, tei, Gword_tr, Gword_te, coder, V)
    print(f"  OVERALL bits/word:  without-G {wl['bpw_no']:.4f}   with-G {wl['bpw_yes']:.4f}   "
          f"Δ {wl['bpw_no']-wl['bpw_yes']:+.4f}")
    print(f"  BACKOFF slice ({wl['bo_frac']*100:.1f}% of words, local ctx exhausted):")
    print(f"     plain-unigram {wl['bo_no']:.4f}   G-unigram {wl['bo_yes']:.4f}   "
          f"Δ {wl['bo_no']-wl['bo_yes']:+.4f}\n")

    print(f"done in {round(time.time()-t0,1)}s")
    import json
    print("JSON " + json.dumps(dict(
        char_bpc_no=c_no, char_bpc_yes=c_yes, char_bpc_sh=c_sh,
        char_pb400_no=float(bits_no[pb].mean()), char_pb400_yes=float(bits_yes[pb].mean()),
        char_pb50_no=float(bits_no[pb50].mean()), char_pb50_yes=float(bits_yes[pb50].mean()),
        word_bpw_no=wl["bpw_no"], word_bpw_yes=wl["bpw_yes"],
        word_bo_frac=wl["bo_frac"], word_bo_no=wl["bo_no"], word_bo_yes=wl["bo_yes"],
        topics=coder.K, switches=switches)))

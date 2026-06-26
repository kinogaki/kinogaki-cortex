#!/usr/bin/env python3
"""Exp X — the HETEROGENEOUS, specialized stack vs the uniform Column (the opposite axis from Exp I).

Exp I proved a UNIFORM Column wired bigger works. This tests the other hypothesis: the brain is not one
repeated part — it's SPECIALIZED TOPOLOGIES at different levels (sensory center-surround, cortical columns,
thalamic gating, basal-ganglia selection, hippocampal memory), each with its own connection RANGE (proximal
local vs distal long-range) and TIMESCALE (fast sensory vs slow integrative). So give each level a different
column type + range + timescale, then GATE/ARBITRATE which level speaks per token. Does specialization win?

Four specialized levels, every one predicting the SAME next CHARACTER (so bpc is comparable end-to-end):
  L0 CHAR   — dense LOCAL char n-gram (proximal, fast)                          [evidence.ExpertBank]
  L1 WORD   — offset-keyed attention (distal, mid timescale) + lexicon          [offsetattn.OffsetAttn]
  L2 PHRASE — branching-entropy CHUNKS + change/trajectory model (slower)       [boundaries + trajectory]
  L3 THEME  — ONLINE topic state (leader-clustering) + G-conditioned chars (slow)[ignition idea, online]

Three measurements (judged on the RIGHT axis per FRAGILE_IDEAS, not bpc-vs-bigram alone):
  1. ABLATION   — uniform-Column stack vs heterogeneous stack vs each level removed: bpc + right-axis metrics.
  2. GATING     — dynamic confidence/surprise gate vs static geometric-mean pool.
  3. IDEA-WALK  — generate by walking the association graph guided by a topic state + surprise (honest demo).

ONLINE-COMPLIANCE: all counting / leaky accumulators / online leader-clustering. No backprop, no k-means/SVD.
Run: python experiments/heterogeneous-stack/run.py   (from the repo root)
"""
import os, sys, time, math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
import corpus
import hetero as H
import offsetattn as OA
from cortex import Cortex, CH

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_BYTES = 10_000_000
EVAL_CHARS = 80_000


def load():
    train = corpus.load_ids("text8", nbytes=TRAIN_BYTES).astype(np.int64)
    test = corpus.load_ids("text8", nbytes=100_000_000)[98_000_000:99_500_000].astype(np.int64)
    return train, test[:EVAL_CHARS]


def char_str(ids):
    A = "abcdefghijklmnopqrstuvwxyz "
    return "".join(A[i] for i in ids)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  1. ABLATION
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def build_levels(train):
    t0 = time.time()
    ch = H.CharLevel(orders=(2, 3, 4, 5)).learn(train)
    wl = H.WordLevel(D=6, gamma=8.0).learn(train)
    pl = H.PhraseLevel(target_rate=0.45, order=3).learn(train)
    tl = H.ThemeLevel(g_order=3).learn(train)
    print(f"  built 4 specialized levels in {time.time()-t0:.1f}s "
          f"(phrase chunk-vocab {pl.CUNK}, theme topics {tl.K})")
    return ch, wl, pl, tl


def level_dists(levels, te):
    ch, wl, pl, tl = levels
    return [ch.logdist(te), wl.logdist(te), pl.logdist(te), tl.logdist(te)]


def ablation(train, te):
    targets = te[1:]
    print("\n[1] ABLATION — uniform vs heterogeneous, on bpc + right-axis metrics")
    levels = build_levels(train)
    lc, lw, lp, lt = level_dists(levels, te)

    # uniform baseline: the Exp I Column cortex (char+word), scored on the SAME eval chars.
    t0 = time.time()
    train_s = char_str(train[:4_000_000])          # Cortex.fit wants a string; cap for speed
    te_s = char_str(te)
    uni = Cortex(char_orders=[2, 4, 6], word_orders=[1, 2, 3]).fit(train_s)
    uni_bpc = float(np.mean([-math.log2(uni.dist(te_s[max(0, t - 64):t])[CH[te_s[t]]] + 1e-12)
                             for t in range(1, len(te_s))]))
    print(f"  uniform Column cortex (Exp I) bpc {uni_bpc:.3f}  [{time.time()-t0:.0f}s]")

    rows = []
    rows.append(("uniform Column stack (Exp I)", uni_bpc, float("nan")))
    rows.append(("char only (L0)", H.bpc_of(lc, targets), H.acc_of(lc, targets)))

    # heterogeneous = char + higher levels under the BEST dynamic gate (hard confidence router), and
    # ablations dropping one level. The hard router is the strongest gate (see block [2]); use it here so the
    # ablation reflects the best the stack can do, not a hobbled pool.
    def gated(higher):
        lds = [lc] + higher
        hr, _ = H.hard_router(lds, targets)
        return H.bpc_of(hr, targets), H.acc_of(hr, targets)

    full_b, full_a = gated([lw, lp, lt])
    rows.append(("HETERO full (char+word+phrase+theme)", full_b, full_a))
    rows.append(("  − word", *gated([lp, lt])))
    rows.append(("  − phrase", *gated([lw, lt])))
    rows.append(("  − theme", *gated([lw, lp])))
    rows.append(("  char + word only", *gated([lw])))

    print(f"\n    {'configuration':<40}{'bpc':>9}{'char-acc':>10}")
    for name, b, a in rows:
        astr = "    —" if a != a else f"{a:>10.3f}"
        print(f"    {name:<40}{b:>9.3f}{astr}")

    print("\n  → On aggregate bpc the fast LOCAL char model already wins; the distal levels can't beat it as")
    print("    CHAR predictors (their projections are noisy). The next blocks judge each on its RIGHT axis.")
    return levels, (lc, lw, lp, lt)


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  RIGHT-AXIS metrics: each specialization on the axis it was built to win.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def right_axes(train, te):
    print("\n[1b] RIGHT-AXIS — each specialized level on the axis it was designed for")
    # WORD level: next-word top-1 + perplexity (calibration). offset-attn vs bag vs bigram.
    trsp = H.word_spans(train)
    strm, voc, UNK = OA.build_word_stream(train, trsp, vocab_size=30000)
    attn = OA.OffsetAttn(D=6, gamma=8.0).fit(strm)
    attn.build_bag()
    A = "abcdefghijklmnopqrstuvwxyz "
    chm = {c: i for i, c in enumerate(A)}
    w2id = {tuple(chm[c] for c in w): i for i, w in enumerate(voc)}
    tesp = H.word_spans(te)
    ew = np.array([w2id.get(tuple(te[s:e].tolist()), UNK) for (s, e) in tesp])

    def eval_word(predfn, nmax=12000):
        hit = tot = 0; nll = 0.0
        for t in range(6, min(nmax, len(ew))):
            if ew[t] == UNK:
                continue
            d = predfn(ew[t - 6:t])
            if not d:
                continue
            tot += 1
            pred = max(d, key=d.get)
            hit += (pred == ew[t])
            p = d.get(ew[t], 0) or 1e-6
            nll += -math.log2(p)
        return hit / max(tot, 1), 2 ** (nll / max(tot, 1)), tot

    def bigram(c):
        dd = attn.tab[1].get(int(c[-1]))
        if not dd:
            return None
        tot = sum(dd.values())
        return {k: v / tot for k, v in dd.items()}

    o_a, o_p, n = eval_word(lambda c: attn.predict(list(c)))
    b_a, b_p, _ = eval_word(lambda c: attn.predict_bag(list(c)))
    g_a, g_p, _ = eval_word(bigram)
    print(f"  WORD level (next-word, n={n}) — offset-attn's axis is CALIBRATION (perplexity), not top-1:")
    print(f"    {'model':<16}{'top-1':>9}{'perplexity':>13}")
    print(f"    {'offset-attn':<16}{o_a:>9.3f}{o_p:>13.0f}")
    print(f"    {'bag-of-words':<16}{b_a:>9.3f}{b_p:>13.0f}")
    print(f"    {'bigram (d=1)':<16}{g_a:>9.3f}{g_p:>13.0f}")
    print(f"    → offset-attn KILLS the bag (order matters) and is best-calibrated; ties bigram on top-1 (Exp S).")

    # PHRASE level: phrase-coherence of what its chunk vocabulary captures (real multi-word units found).
    pl = H.PhraseLevel(target_rate=0.45).learn(train)
    multi = [nm for nm in pl.chunk2id if len(nm) >= 2]
    id2w = {i: w for i, w in enumerate(voc)}
    samples = []
    for nm in sorted(pl.chunk2id, key=lambda k: -len(k))[:6]:
        samples.append(" ".join(id2w.get(w, "?") for w in nm))
    print(f"\n  PHRASE level — branching-entropy discovered {len(pl.chunk2id)} chunks, "
          f"{len(multi)} multi-word ({100*len(multi)/len(pl.chunk2id):.0f}%). Sample units:")
    for s in samples[:4]:
        print(f"    · {s[:70]}")

    # THEME level: do online topics form coherent word groups? show a few topic clusters.
    tl = H.ThemeLevel(g_order=3).learn(train)
    print(f"\n  THEME level — ONLINE leader-clustering found {tl.K} topics (no k-means). Sample topics:")
    topic_of = tl.coder.topic_of
    by_topic = {}
    freq = np.bincount(strm)
    for wid in np.argsort(freq)[::-1]:
        t = topic_of[wid]
        if t < 0:
            continue
        by_topic.setdefault(int(t), [])
        if len(by_topic[int(t)]) < 6:
            by_topic[int(t)].append(id2w.get(int(wid), "?"))
    shown = 0
    for t, words in sorted(by_topic.items(), key=lambda kv: -len(kv[1])):
        if len(words) >= 5 and shown < 4:
            print(f"    topic {t:>3}: {' '.join(words)}")
            shown += 1


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  2. GATING vs STATIC POOLING
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def gating(dists, te, levels):
    lc, lw, lp, lt = dists
    targets = te[1:]
    print("\n[2] GATING — dynamic confidence/surprise routing vs static geometric-mean pool")
    lds = [lc, lw, lp, lt]
    eq = H.static_pool(lds)
    cw = H.static_pool(lds, weights=[4, 1, 0.5, 0.5])
    sg, w = H.soft_gate(lds, targets)
    hr, choice = H.hard_router(lds, targets)
    ag, mw = H.anchored_gate(lc, [lw, lp, lt], targets, beta=0.6)
    rows = [
        ("char only (reference)", H.bpc_of(lc, targets)),
        ("static pool — equal weight", H.bpc_of(eq, targets)),
        ("static pool — fixed char-heavy weight", H.bpc_of(cw, targets)),
        ("DYNAMIC soft gate (confidence softmax)", H.bpc_of(sg, targets)),
        ("DYNAMIC hard router (argmax confidence)", H.bpc_of(hr, targets)),
        ("DYNAMIC anchored gate (driver+modulator)", H.bpc_of(ag, targets)),
    ]
    print(f"\n    {'gate / pool':<44}{'bpc':>9}")
    for name, b in rows:
        print(f"    {name:<44}{b:>9.3f}")
    # how often each level wins the hard router (where the gate routes)
    frac = [float((choice == i).mean()) for i in range(4)]
    print(f"\n  hard-router routing share: char {frac[0]:.2f}  word {frac[1]:.2f}  "
          f"phrase {frac[2]:.2f}  theme {frac[3]:.2f}")
    print("  → DYNAMIC routing beats the naive equal-weight static pool by a wide margin (it stops the weak")
    print("    distal levels from dragging the strong local one down); it recovers ~char-level bpc.")

    # ROBUSTNESS — the gate's RIGHT axis. Corrupt the local CONTEXT (random char flips); the fast char model
    # degrades, the slow word/theme levels (longer range) are less hurt. Does gated routing now BEAT char-only?
    import evidence as EV
    ch, wl, pl, tl = levels
    rng = np.random.default_rng(0)
    print("\n  [2b] ROBUSTNESS — corrupt local context, score vs CLEAN targets (the gate's right axis):")
    print(f"    {'noise':>7}{'char-only':>12}{'hard-router':>13}{'gate−char':>11}")
    for frac in (0.0, 0.05, 0.10, 0.20):
        noisy = EV.corrupt_context(te, frac, rng)
        nc = ch.logdist(noisy)
        nw = wl.logdist(noisy)
        nt = tl.logdist(noisy)
        hb = H.bpc_of(H.hard_router([nc, nw, nt], targets)[0], targets)
        cb = H.bpc_of(nc, targets)
        print(f"    {frac:>7.2f}{cb:>12.3f}{hb:>13.3f}{cb-hb:>+11.3f}")
    print("    → HONEST: the gate does NOT beat char-only even under heavy noise — the distal levels' CHAR")
    print("      projections are too noisy to rescue a corrupted local model. The fast char model degrades")
    print("      gracefully on its own; robustness lives WITHIN the char level (leaky evidence, Exp R), not in")
    print("      routing to noisier distal char-projections. Parked: gate on a per-level leaky-evidence decode.")
    return hr


# ══════════════════════════════════════════════════════════════════════════════════════════════════
#  3. IDEA-WALK — generate by walking the association graph, guided by topic + surprise.
# ══════════════════════════════════════════════════════════════════════════════════════════════════

def idea_walk(train, seed_word="science", steps=24):
    print("\n[3] IDEA-WALK — walking the concept/association graph (honest exploratory demo)")
    import graph as G
    trsp = H.word_spans(train)
    strm, voc, UNK = OA.build_word_stream(train, trsp, vocab_size=8000)
    top, remap, A = G.build_graph(strm, N=2000, window=5, k_edges=20)
    id2w = {i: w for i, w in enumerate(voc)}
    w2id = {w: i for i, w in enumerate(voc)}
    node2word = {ni: id2w.get(int(wid), "?") for ni, wid in enumerate(top)}
    word2node = {id2w.get(int(wid), "?"): ni for ni, wid in enumerate(top)}

    rng = np.random.default_rng(1)

    def walk(start, goal=None, steps=steps, surprise_temp=0.6):
        """Walk the PMI graph: keep an activation bump, spread it, step to a high-activation neighbour, but
        bias toward a GOAL topic node and inject SURPRISE (temperature) so it explores, not loops."""
        n = A.shape[0]
        cur = word2node.get(start)
        if cur is None:
            return [start]
        path = [start]
        goal_v = None
        if goal and goal in word2node:
            goal_v = np.zeros(n); goal_v[word2node[goal]] = 1.0
            goal_v = G.spread(goal_v, A, alpha=0.5, hops=2)
        visited = {cur}
        for _ in range(steps):
            a0 = np.zeros(n); a0[cur] = 1.0
            act = G.spread(a0, A, alpha=0.5, hops=1)
            if goal_v is not None:
                act = act * (1.0 + 2.0 * goal_v)               # pull toward the goal topic
            for v in visited:
                act[v] = 0.0                                    # don't revisit (novelty / anti-loop)
            if act.sum() <= 0:
                break
            p = act ** (1.0 / surprise_temp)
            p = p / p.sum()
            cur = int(rng.choice(n, p=p))
            visited.add(cur)
            path.append(node2word.get(cur, "?"))
        return path

    free = walk("science", goal=None)
    goaled = walk("science", goal="music")
    print(f"  free walk from 'science':   {' → '.join(free[:18])}")
    print(f"  goal walk science→'music':  {' → '.join(goaled[:18])}")

    # coherence number: mean PMI-edge weight traversed (are consecutive ideas actually associated?) vs a
    # random-word sequence baseline.
    def edge_coherence(path):
        ws = 0.0; m = 0
        for a, b in zip(path[:-1], path[1:]):
            ia, ib = word2node.get(a), word2node.get(b)
            if ia is None or ib is None:
                continue
            ws += float(A[ia, ib]); m += 1
        return ws / max(m, 1)

    rand_path = [node2word[int(x)] for x in rng.integers(0, A.shape[0], size=steps)]
    print(f"\n  coherence (mean traversed PMI-edge weight, higher = more associated):")
    print(f"    free walk      {edge_coherence(free):.3f}")
    print(f"    goal walk      {edge_coherence(goaled):.3f}")
    print(f"    random words   {edge_coherence(rand_path):.3f}")
    print("  → the walk stays on associated ideas (non-zero traversed weight) where random jumps don't.")
    print("  HONEST: this is associative idea-roaming, NOT grounded reasoning — a navigation demo, not thought.")


# ══════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(0)
    train, te = load()
    print(f"Exp X — heterogeneous specialized stack. train {len(train):,} chars / eval {len(te):,} chars\n")
    levels, dists = ablation(train, te)
    right_axes(train, te)
    gating(dists, te, levels)
    idea_walk(train)
    print("\ndone.")

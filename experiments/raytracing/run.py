#!/usr/bin/env python3
"""Exp P — raytracing / proximity columns: real columns placed in a space, connected by proximity.

Build a semantic space for the top-N word-columns (PPMI co-occurrence + eigen-embed — a count factorization, not
backprop), so related columns are near. Then FEED DATA as a trajectory and predict two ways:
  A. spatial gather  — near columns' followers vote (distance-weighted): spatial generalization / backoff.
  B. ray extrapolation — next point = current + velocity, read off nearest column: literal "cast a ray".
Baseline: plain bigram. Metric: next-word top-1 accuracy (restricted to the top-N column vocab). Plus an
inspection of the learned space (does proximity = meaning?). Corpus: text8 (clean, good semantics).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from spatial import build_embedding, neighbors, knn
from graph import build_graph

N = 2500

if __name__ == "__main__":
    ids = load_ids("text8", nbytes=20_000_000)
    spans = split_words(ids); words = [ids_to_str(ids[s:e]) for s, e in spans]
    w2id = {}; wids = np.empty(len(words), np.int64)
    for i, w in enumerate(words): wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    cut = int(len(wids) * 0.85); train, testw = wids[:cut], wids[cut:]
    top, remap, pos = build_embedding(train, N=N, window=5, D=16)
    if remap.size < len(w2id):                                # pad: words only in held-out → OOV (-1)
        remap = np.concatenate([remap, np.full(len(w2id) - remap.size, -1, np.int64)])
    topword = [id2word[t] for t in top]
    print(f"{len(words):,} words, embedding top-{N} in 16-D\n")

    # ── is proximity meaningful? (nearest neighbors in column-space) ──
    print("=== learned space: nearest columns (proximity = meaning?) ===")
    for w in ["three", "king", "war", "water", "france", "music", "city", "she"]:
        if w in w2id and remap[w2id[w]] >= 0:
            i = remap[w2id[w]]
            print(f"    {w:<8} → " + ", ".join(topword[j] for j in neighbors(pos, i, 8)))

    # ── forward bigram among top-N columns ──
    a = remap[train[:-1]]; b = remap[train[1:]]; m = (a >= 0) & (b >= 0)
    F = np.zeros((N, N), np.float32); np.add.at(F, (a[m], b[m]), 1.0)
    base_pred = F.argmax(1)                                   # baseline: most likely next column

    # ── spatial gather: W (proximity-weighted kNN) @ F → smoothed followers ──
    K = 10; W = np.zeros((N, N), np.float32)
    for i in range(N):
        nb = knn(pos, pos[i], K); d = np.linalg.norm(pos[nb] - pos[i], axis=1)
        W[i, nb] = np.exp(-d / (d.mean() + 1e-9))
    W /= W.sum(1, keepdims=True)
    G = W @ F; gather_pred = G.argmax(1)

    # ── method C: graph spreading (the sources' endorsed form) — blend a word's followers with its PMI-graph
    #    association-neighbors' followers (1-hop spreading activation), no Euclidean coords ──
    _, _, Ag = build_graph(train, N=N, window=5, k_edges=20)
    graph_pred = (F + 0.5 * (Ag @ F)).argmax(1)

    # ── evaluate next-word top-1 accuracy on held-out (restricted to top-N) ──
    ta = remap[testw[:-1]]; tb = remap[testw[1:]]; mm = (ta >= 0) & (tb >= 0)
    ca, cb = ta[mm], tb[mm]
    acc_base = float(np.mean(base_pred[ca] == cb))
    acc_gather = float(np.mean(gather_pred[ca] == cb))
    acc_graph = float(np.mean(graph_pred[ca] == cb))

    # ── ray extrapolation: need triples (prev, cur, next) all in top-N ──
    t3 = remap[testw]; hit = ray = 0
    for i in range(1, len(t3) - 1):
        p, c, nx = t3[i - 1], t3[i], t3[i + 1]
        if p < 0 or c < 0 or nx < 0: continue
        pred = knn(pos, pos[c] + (pos[c] - pos[p]), 1)[0]
        ray += 1; hit += (pred == nx)
        if ray >= 20000: break
    acc_ray = hit / max(ray, 1)

    print("\n=== next-word top-1 accuracy (held-out, top-N vocab) ===")
    print(f"    bigram baseline        {acc_base*100:5.2f}%")
    print(f"    A. spatial gather      {acc_gather*100:5.2f}%   (euclidean-near columns' followers vote)")
    print(f"    B. ray extrapolation   {acc_ray*100:5.2f}%   (current + velocity → nearest column)")
    print(f"    C. graph spreading     {acc_graph*100:5.2f}%   (PMI-graph neighbors' followers, endorsed form)")
    print(f"    random (1/N)           {100.0/N:5.3f}%")

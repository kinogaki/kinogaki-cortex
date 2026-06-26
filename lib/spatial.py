"""Spatial / "raytracing" columns — real columns placed in a space, connected by PROXIMITY.

The idea (the user's): instead of an index of columns, give each column a POSITION so that related columns are
physically near, connect by proximity, and predict by GATHERING from near columns — the raytracing/reference-
frame picture (a query is a point; near columns vote, distance-weighted, like a photon/kNN gather; "motor" =
moving through the space).

HOW TO FEED DATA (the open question): a token stream becomes a TRAJECTORY through column-space. Each word is a
point (its column's position). The positions are LEARNED from co-occurrence (PPMI + eigen-embed — a one-shot
factorization of the count matrix, not backprop), so semantically related words cluster. Two ways to predict:
  A. spatial gather  — near columns' follower-distributions vote (distance-weighted) → spatial generalization.
  B. ray extrapolation — predict the NEXT point as current + velocity (recent movement), read off the nearest
     column. Literally casting a ray through the trajectory. (Bold; may be weak — we measure honestly.)
"""
import numpy as np

def build_embedding(wids, N=2500, window=5, D=16):
    """Place the top-N words in D-dim space by PPMI co-occurrence + eigen-embedding (related words → near)."""
    counts = np.bincount(wids)
    top = np.argsort(counts)[::-1][:N]
    remap = -np.ones(counts.size, np.int64); remap[top] = np.arange(len(top))
    n = len(top)
    cooc = np.zeros(n * n)
    for g in range(1, window + 1):
        a = remap[wids[:-g]]; b = remap[wids[g:]]
        m = (a >= 0) & (b >= 0)
        cooc += np.bincount(a[m] * n + b[m], minlength=n * n)
    cooc = cooc.reshape(n, n); cooc = cooc + cooc.T
    tot = cooc.sum(); rs = cooc.sum(1) + 1e-9
    pmi = np.log((cooc * tot) / np.outer(rs, rs) + 1e-12)
    ppmi = np.maximum(pmi, 0.0)
    vals, vecs = np.linalg.eigh(ppmi)                        # symmetric → real eigals
    idx = np.argsort(vals)[::-1][:D]
    pos = vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 1e-9))
    return top, remap, pos                                   # pos[i] = position of top-word i

def neighbors(pos, i, k=8):
    d = np.linalg.norm(pos - pos[i], axis=1); return np.argsort(d)[1:k + 1]

def knn(pos, q, k=8):
    d = np.linalg.norm(pos - q, axis=1); return np.argsort(d)[:k]

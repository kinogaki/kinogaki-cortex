"""Graph proximity columns — the form of "raytracing/proximity" the sources endorse over Euclidean coords.

Multiple sources (HTM forums, GOFAI) converge: a Euclidean reference frame for language is a dead end (no
natural origin/unit for symbolic space). The buildable replacement = SPREADING ACTIVATION over a learned
association GRAPH: proximity = co-occurrence/PMI edge weight (not distance). "Cast a ray" = activate the
context nodes, spread activation a few hops, and let each reached column vote (weighted by its activation ×
base-level recency·frequency, à la ACT-R). All counts, no backprop.
"""
import numpy as np

def build_graph(wids, N=2500, window=5, k_edges=20):
    counts = np.bincount(wids)
    top = np.argsort(counts)[::-1][:N]
    remap = -np.ones(counts.size, np.int64); remap[top] = np.arange(len(top)); n = len(top)
    cooc = np.zeros(n * n)
    for g in range(1, window + 1):
        a = remap[wids[:-g]]; b = remap[wids[g:]]; m = (a >= 0) & (b >= 0)
        cooc += np.bincount(a[m] * n + b[m], minlength=n * n)
    cooc = cooc.reshape(n, n); cooc = cooc + cooc.T
    tot = cooc.sum(); rs = cooc.sum(1) + 1e-9
    ppmi = np.maximum(np.log((cooc * tot) / np.outer(rs, rs) + 1e-12), 0.0)
    np.fill_diagonal(ppmi, 0.0)
    A = np.zeros((n, n), np.float32)                          # top-k_edges PMI neighbors per node, row-normalized
    for i in range(n):
        nb = np.argsort(ppmi[i])[::-1][:k_edges]
        A[i, nb] = ppmi[i, nb]
    A /= (A.sum(1, keepdims=True) + 1e-9)
    return top, remap, A

def spread(a0, A, alpha=0.5, hops=1):
    a = a0.copy()
    for _ in range(hops):
        a = a + alpha * (A.T @ a)                             # pump activation along association edges
    s = a.sum(); return a / s if s > 0 else a

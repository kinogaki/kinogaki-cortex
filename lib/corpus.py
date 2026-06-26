"""Fast corpus loading in ID-SPACE — never build a Python string for GB-scale data.

Alphabet matches lib/metrics / lib/cortex: a..z = 0..25, space = 26 (V=27). text8 is already clean lowercase
[a-z ]; enwik9 is raw Wikipedia XML, crudely normalized here (lowercase, keep a-z, everything else → space,
collapse runs of space). Good enough for scaling experiments (we care about throughput + does-more-data-help,
not perfect cleaning). Returns an int8 numpy array of ids — feeds FastColumn/FastChar directly.
"""
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
A = "abcdefghijklmnopqrstuvwxyz "; SPACE = 26

def load_ids(name, nbytes=None):
    with open(os.path.join(DATA, name), "rb") as f:
        raw = f.read(nbytes) if nbytes else f.read()
    b = np.frombuffer(raw, dtype=np.uint8).copy()
    up = (b >= 65) & (b <= 90); b[up] += 32                  # A-Z → a-z
    az = (b >= 97) & (b <= 122)
    ids = np.where(az, b - 97, SPACE).astype(np.int8)        # a-z → 0..25, else → space
    sp = ids == SPACE                                        # collapse consecutive spaces
    keep = ~(sp & np.concatenate([[False], sp[:-1]]))
    return np.ascontiguousarray(ids[keep])

def load_ids_pages(name, nbytes=None):
    """Like load_ids, but also return the output-index of every <page> start (enwik9 article = topic boundary).
    Ground truth for topic-boundary detection — the boundaries survive into id-space via a kept-position map."""
    import re
    with open(os.path.join(DATA, name), "rb") as f:
        raw = f.read(nbytes) if nbytes else f.read()
    page_off = np.array([m.start() for m in re.finditer(b"<page>", raw)], dtype=np.int64)
    b = np.frombuffer(raw, dtype=np.uint8).copy()
    up = (b >= 65) & (b <= 90); b[up] += 32
    az = (b >= 97) & (b <= 122)
    ids = np.where(az, b - 97, SPACE).astype(np.int8)
    sp = ids == SPACE; keep = ~(sp & np.concatenate([[False], sp[:-1]]))
    out = np.ascontiguousarray(ids[keep])
    out_index = np.cumsum(keep) - 1                          # raw byte → output id index (for kept bytes)
    pages = np.unique(out_index[np.clip(page_off, 0, len(keep) - 1)])
    return out, pages[pages > 0]

def ids_to_str(ids):
    return "".join(A[i] for i in ids)                        # only for small slices / sampling

def split_words(ids):
    """Boundaries at space (26): returns list of (start,end) word spans over the id array — id-space words."""
    sp = np.nonzero(ids == SPACE)[0]
    bounds = np.concatenate([[-1], sp, [len(ids)]])
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i] + 1]

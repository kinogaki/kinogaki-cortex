"""varsets.py — the variation-set minimal-pair miner (M16). ONLINE, NO backprop, BOUNDED ring buffer.

A child does not hear sentences in isolation. Caregiver speech comes in VARIATION SETS — runs of adjacent
utterances that say nearly the same thing with one thing swapped:

    "put it on the table"   "put the cup on the table"   "put it there"

The aligned, near-repeated frame ("put __ on the table") and the one swapped span (the FILLER: it / the cup)
are handed to the child for free — a minimal pair, on a plate. M16 mines exactly this from the stream with no
gradient: keep a small RING BUFFER of recent utterances; align each new utterance against the previous one
(token-level longest-common-subsequence). When two adjacent utterances overlap enough, the AGREEING tokens are
a frame and each DISAGREEING run is a (slot, filler) substitution. Two products fall out:

  1. CONSTRUCTION evidence — every (frame-word, filler-word) pair the diff exposes gets an EXTRA count into the
     usage-based frame tables (constructions.py): the variation set is teaching the open slot directly, so the
     filler-in-this-frame evidence is up-weighted relative to plain reading. This is the count-native form of
     "reactive/structured input teaches syntax faster" (Haga: helps syntax, not world knowledge).

  2. BOUNDARY evidence — the align/disagree transition points (where the frame stops matching and the filler
     begins) are phrase boundaries, harvested without branching entropy. A new boundary source for boundaries.py.

Everything is ONLINE and BOUNDED: one LCS diff per utterance against a fixed-size ring of the last N (the buffer
caps memory); the counts land in the same bounded AF tables under the usual eviction. No gradient, no k-means.

  RingBuffer(n)                        the bounded window of recent utterances.
  lcs_align(a, b)                      token LCS alignment -> matched anchor positions (the diff backbone).
  diff_spans(a, b)                     -> (agree_pairs, sub_pairs, boundary_gaps): the harvested frame/filler.
  VariationMiner(n, overlap)           .feed(utterance) -> mined evidence; .frame_bonus accumulates the extra
                                       (frame, filler) counts; .boundary_hits accumulates diff-derived cut points.
"""
import numpy as np


# ── token-level longest-common-subsequence: the alignment backbone (no gradient, O(len_a*len_b)) ──

def lcs_align(a, b):
    """Longest common subsequence of two token lists. Returns the list of (i, j) matched index pairs (a[i]==b[j])
    forming the alignment backbone. Classic DP — the 'anchored diff' M16 asks for, done by counting matches."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return []
    dp = np.zeros((la + 1, lb + 1), np.int32)
    for i in range(la - 1, -1, -1):
        ai = a[i]
        row, nrow = dp[i], dp[i + 1]
        for j in range(lb - 1, -1, -1):
            row[j] = nrow[j + 1] + 1 if ai == b[j] else max(nrow[j], row[j + 1])
    # backtrace the matched pairs
    pairs = []
    i = j = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def diff_spans(a, b):
    """Diff utterance `b` against the PREVIOUS utterance `a` via LCS. Returns:
        overlap     : |LCS| / max(|a|,|b|)  (the variation-set test: high = near-repeat).
        sub_pairs   : list of (frame_token, filler_token) — for each disagreeing run in b, pair the new
                      filler tokens with the agreeing FRAME tokens that bracket the slot (left + right anchor).
        boundary_b  : set of indices in b where a substitution span begins/ends (the diff-derived phrase cuts).
    The frame/filler pairing is exactly AF's "X ___" evidence, but sourced from the adjacent-utterance diff
    instead of raw adjacency: the swapped word is *known* to be a filler of the surrounding frame."""
    pairs = lcs_align(a, b)
    overlap = len(pairs) / max(len(a), len(b), 1)
    matched_b = set(j for _, j in pairs)
    sub_pairs = []
    boundary_b = set()
    lb = len(b)
    # walk b; runs of unmatched tokens are fillers; the nearest matched tokens on each side are the frame anchors
    t = 0
    while t < lb:
        if t in matched_b:
            t += 1; continue
        s = t
        while t < lb and t not in matched_b:
            t += 1
        # b[s:t] is a filler run; left anchor = b[s-1] if matched, right anchor = b[t] if matched
        left = b[s - 1] if s - 1 >= 0 and (s - 1) in matched_b else None
        right = b[t] if t < lb and t in matched_b else None
        for fi in range(s, t):
            if left is not None:
                sub_pairs.append((left, b[fi]))          # "left-anchor ___ = filler" (AF 1-gram frame)
        boundary_b.add(s)
        if t <= lb:
            boundary_b.add(t)
    return overlap, sub_pairs, boundary_b


# ── the miner: a bounded ring buffer + the online diff, accumulating frame/filler and boundary evidence ──

class RingBuffer:
    """Fixed-size FIFO of the last n utterances (each a tuple of token ids). Bounded memory by construction."""
    __slots__ = ("n", "buf")
    def __init__(self, n):
        self.n = n; self.buf = []
    def push(self, u):
        self.buf.append(u)
        if len(self.buf) > self.n:
            self.buf.pop(0)
    def prev(self):
        return self.buf[-2] if len(self.buf) >= 2 else None


class VariationMiner:
    """Stream utterances through; mine variation-set minimal pairs online.

    feed(u): diff u against the previous utterance. If overlap >= `overlap_min`, the swap is a genuine
    variation-set substitution: accumulate its (frame, filler) pairs into `frame_bonus` (extra AF counts) and
    record the diff cut points into `boundary_hits` (a per-stream-position boundary vote). BOUNDED: only the
    ring buffer + the (already-bounded) AF tables grow; the bonus dict is keyed on the same frame ids.

    `frame_bonus[(frame_word, filler_word)]` -> extra count to fold into constructions.build_frame_counts.
    `boundary_hits` -> list of stream positions (global token index) the diff marked as a phrase boundary.
    """
    def __init__(self, n=6, overlap_min=0.60, bonus=2.0):
        self.ring = RingBuffer(n)
        self.overlap_min = overlap_min
        self.bonus = bonus
        self.frame_bonus = {}
        self.boundary_hits = []
        self.n_varsets = 0          # utterances that registered as a variation set (overlap >= min)
        self.n_subs = 0             # total substitution (frame,filler) pairs harvested
        self._pos = 0               # running global token offset (start of the current utterance)

    def feed(self, u):
        u = tuple(int(x) for x in u)
        prev = self.ring.buf[-1] if self.ring.buf else None
        self.ring.push(u)
        if prev is not None:
            overlap, subs, bnd = diff_spans(prev, u)
            if overlap >= self.overlap_min:
                self.n_varsets += 1
                for fr, fl in subs:
                    if fr >= 0 and fl >= 0:
                        self.frame_bonus[(fr, fl)] = self.frame_bonus.get((fr, fl), 0.0) + self.bonus
                        self.n_subs += 1
                for bi in bnd:
                    self.boundary_hits.append(self._pos + bi)
        self._pos += len(u)
        return self

    def apply_to_frame_counts(self, frame_counts, N):
        """Fold the mined (frame,filler) bonuses into AF's frame_counts dict (frame_key -> (filler_ids, counts)).
        ONLINE-compatible: this is additive counting — identical to having incremented each pair `bonus` extra
        times as the diff exposed it. Returns a NEW dict (leaves the baseline untouched for the A/B contrast)."""
        out = {}
        # start from a copy of the baseline counts
        for fk, (fids, cnt) in frame_counts.items():
            out[fk] = (fids.copy(), cnt.copy())
        # bucket the bonuses by frame
        by_frame = {}
        for (fr, fl), b in self.frame_bonus.items():
            by_frame.setdefault(int(fr), {})[int(fl)] = b
        for fk, addl in by_frame.items():
            if fk in out:
                fids, cnt = out[fk]
                idx = {int(w): i for i, w in enumerate(fids)}
                new_w, new_c = [], []
                for fl, b in addl.items():
                    if fl in idx:
                        cnt[idx[fl]] += b
                    else:
                        new_w.append(fl); new_c.append(b)
                if new_w:
                    fids = np.concatenate([fids, np.array(new_w, fids.dtype)])
                    cnt = np.concatenate([cnt, np.array(new_c, cnt.dtype)])
                out[fk] = (fids, cnt)
            else:
                ws = np.array(list(addl.keys()), np.int64)
                cs = np.array(list(addl.values()), np.float64)
                out[fk] = (ws, cs)
        return out

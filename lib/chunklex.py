"""chunklex.py — the PARSER/Isbilen organ: a chunk lexicon with sub-unit interference.

The cortex spine learns FIXED-order n-grams. A child does not: she commits to *whole units* of
variable length ("thedog", "onceuponatime") and, having committed, stops tracking the transitions
*inside* them. Isbilen's splice result is the signature: once "cup" is a chunk, the within-chunk
c-u-p transition decays — the learner hears the unit, not its parts. That is the one thing a
backoff n-gram cannot do: it always keeps every sub-transition.

This module is that organ, count-native and online:

  - ChunkLexicon : a count table over VARIABLE-LENGTH unit-strings (not fixed-order contexts).
      * observe(stream) : greedily COVER the buffer left-to-right with the highest-weight chunks
        (take-the-best = longest-confident-first, AJ shape). Each covering chunk +1.
      * mint : when two adjacent covering chunks RECUR together, mint their concatenation as a new
        chunk (leader-clustering spawn-on-novelty). The lexicon grows its own vocabulary.
      * sub-unit interference (THE NEW PART) : as a longer chunk's weight grows, LEAK weight from
        the constituent sub-chunks it was minted from (a leaky subtraction, Exp R shape). The model
        commits to the whole and the parts decay — the Isbilen splice effect, by counting.
      * LFU eviction (Exp AI) : bounded — when the lexicon overflows, drop the least-frequent chunks.

  - The committed chunks become an emission vocabulary and a segmenter (cover points = boundaries).

Two dials the FRAGILE budget sweeps: DECAY (how hard the whole leaks from its parts) and the COVER
policy (longest-first vs highest-weight-first; whether single chars are always allowed).

ONLINE single streaming pass. NO gradient / k-means / SVD / backprop. BOUNDED (cap + LFU + the decay
that FREES sub-unit mass). Tokens are ints in [0..V); a chunk is a tuple of ints.
"""
import math
import numpy as np


class ChunkLexicon:
    """Online variable-length chunk lexicon with leaky sub-unit interference and LFU eviction."""

    def __init__(self, vocab, *, decay=0.5, mint_thresh=4, cover="longest",
                 max_chunks=20000, max_len=8, seed=0):
        self.V = vocab
        self.decay = decay              # fraction of a recurrence's credit leaked from each sub-chunk
        self.mint_thresh = mint_thresh  # adjacent-pair recurrences before minting the concatenation
        self.cover = cover              # "longest" | "weight" — the take-the-best tie policy
        self.max_chunks = max_chunks
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        # weight table over chunks (tuples). seed with the single tokens so cover always succeeds.
        self.w = {(i,): 1.0 for i in range(vocab)}
        self.parent = {}                # chunk -> (left, right) it was minted from (its sub-units)
        self.pair = {}                  # (left_chunk, right_chunk) -> recurrence count, until minted
        self.n_mint = 0
        self.n_evict = 0

    # ── cover: greedily segment a buffer into known chunks, left-to-right, take-the-best ──
    def _best_chunk_at(self, buf, i):
        """The chunk to commit at position i: the longest (or highest-weight) known chunk matching here.
        Single tokens always match (they are seeded), so cover never stalls."""
        best = None; best_key = -1.0
        hi = min(self.max_len, len(buf) - i)
        for L in range(hi, 0, -1):
            cand = tuple(buf[i:i + L])
            wv = self.w.get(cand)
            if wv is None:
                continue
            if self.cover == "longest":
                key = L + 1e-6 * wv          # longest-confident-first; weight breaks ties
            else:                            # "weight" — highest-weight-first, length breaks ties
                key = wv + 1e-6 * L
            if key > best_key:
                best_key = key; best = cand
        return best if best is not None else (buf[i],)

    def cover_buffer(self, buf):
        """Return the list of chunks that greedily cover buf left-to-right (the segmentation)."""
        out = []; i = 0; n = len(buf)
        while i < n:
            ch = self._best_chunk_at(buf, i)
            out.append(ch); i += len(ch)
        return out

    # ── observe: cover, count, mint, interfere ──
    def observe(self, stream):
        buf = list(int(x) for x in stream)
        chunks = self.cover_buffer(buf)
        prev = None
        for ch in chunks:
            self.w[ch] = self.w.get(ch, 0.0) + 1.0
            # sub-unit interference: a covered chunk that is itself a minted whole leaks from its parts.
            par = self.parent.get(ch)
            if par is not None and self.decay > 0.0:
                left, right = par
                if left in self.w:
                    self.w[left] = max(0.0, self.w[left] - self.decay)
                if right in self.w:
                    self.w[right] = max(0.0, self.w[right] - self.decay)
            # adjacent-pair recurrence → leader-spawn a new chunk
            if prev is not None:
                cat = prev + ch
                if cat not in self.w and len(cat) <= self.max_len:
                    key = (prev, ch)
                    c = self.pair.get(key, 0) + 1
                    if c >= self.mint_thresh:
                        self.w[cat] = float(self.mint_thresh)
                        self.parent[cat] = (prev, ch)
                        self.pair.pop(key, None)
                        self.n_mint += 1
                    else:
                        self.pair[key] = c
            prev = ch
        if len(self.w) > self.max_chunks:
            self._evict()

    def _evict(self):
        """LFU: drop the least-frequent chunks, but NEVER the seeded single tokens (cover must survive)."""
        target = int(self.max_chunks * 0.9)
        multis = [(w, ch) for ch, w in self.w.items() if len(ch) > 1]
        multis.sort()
        drop = len(self.w) - target
        for w, ch in multis[:drop]:
            self.w.pop(ch, None); self.parent.pop(ch, None); self.n_evict += 1

    # ── readouts ──
    def transition_weight(self, left, right):
        """How committed is the splice point between two single tokens, read off the lexicon? Returns the
        mass on the 2-chunk (left,right) — the 'do we hear them as one unit' signal. This is what the
        sub-unit-interference decay is supposed to PUSH DOWN for an internal (within-learned-chunk) pair."""
        return self.w.get((left, right), 0.0)

    def chunk_stats(self):
        multis = [(ch, w) for ch, w in self.w.items() if len(ch) > 1 and w > 0]
        types = len(multis)
        tokens = sum(w for _, w in multis)
        return dict(types=types, tokens=tokens, n_mint=self.n_mint, n_evict=self.n_evict,
                    lex_size=len(self.w))


class PureTP:
    """Saffran null: a pure forward transitional-probability tracker. Counts adjacent bigrams ONLY and
    NEVER chunks — so its within-word transition NEVER decays. The baseline the splice test kills against:
    if ChunkLexicon's spliced internal transition does not fall BELOW PureTP's, sub-unit interference did
    nothing."""
    def __init__(self, vocab):
        self.V = vocab
        self.uni = np.zeros(vocab)
        self.bi = {}
    def observe(self, stream):
        s = list(int(x) for x in stream)
        for t in range(len(s)):
            self.uni[s[t]] += 1
            if t > 0:
                k = (s[t - 1], s[t]); self.bi[k] = self.bi.get(k, 0) + 1
    def tp(self, left, right):
        """P(right | left) — the forward transitional probability across the pair."""
        c = self.bi.get((left, right), 0)
        return c / self.uni[left] if self.uni[left] > 0 else 0.0


class ChunkAgent:
    """A bpc/generation agent that predicts the next char FROM THE LEXICON, not a parallel n-gram. At each
    position it knows the partial chunk it is mid-way through covering (the suffix since the last cover
    boundary); the next char is voted by every lexicon chunk that EXTENDS that partial chunk, weighted by
    its committed lexicon weight — completion-by-chunk. Sub-unit interference therefore shows up directly:
    a committed whole sharpens its own completion and starves the sub-chunk continuations. Blended with a
    low-order char backoff so unseen prefixes never floor. `.K`/`.dist` adapt to lib/metrics unchanged.

    The point of the comparison: does WHOLE-UNIT completion buy held-out bpc over a fixed-order n-gram that
    only ever sees the last `order` chars and keeps every sub-transition?"""
    def __init__(self, lex, order=2, alpha=0.05, lam=0.6):
        self.lex = lex; self.V = lex.V; self.K = 64; self.order = order; self.alpha = alpha; self.lam = lam
        self.ctx = [dict() for _ in range(order + 1)]            # low-order char backoff (the floor)
        self.A = "abcdefghijklmnopqrstuvwxyz "[:self.V]
        self.CH = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        # index chunks by prefix for fast completion lookup
        self._index()
    def _index(self):
        self.by_prefix = {}
        for ch, w in self.lex.w.items():
            if w <= 0:
                continue
            for L in range(len(ch)):
                self.by_prefix.setdefault(ch[:L], {})[ch[L]] = self.by_prefix.setdefault(ch[:L], {}).get(ch[L], 0.0) + w
    def train(self, ids):
        s = list(int(x) for x in ids)
        for t in range(len(s)):
            nx = s[t]
            for k in range(min(self.order, t) + 1):
                d = self.ctx[k].setdefault(tuple(s[t - k:t]), {}); d[nx] = d.get(nx, 0) + 1
        self._index()
    def _backoff(self, ids):
        for k in range(min(self.order, len(ids)), -1, -1):
            d = self.ctx[k].get(tuple(ids[len(ids) - k:]) if k else ())
            if d:
                p = np.full(self.V, self.alpha); tot = self.alpha * self.V
                for tok, c in d.items():
                    p[tok] += c; tot += c
                return p / tot
        return np.full(self.V, 1.0 / self.V)
    def _chunk_completion(self, ids):
        """The lexicon's vote: cover the context, then over the trailing PARTIAL chunk look up which chars
        extend it (longest matching partial first). Returns a normalized dist or None if no chunk extends."""
        chunks = self.lex.cover_buffer(ids)
        partial = ()
        if chunks:
            # the last covered chunk consumed the tail; the partial is what's left mid-stream — but cover is
            # greedy-complete, so re-derive the partial as the longest suffix that is a known chunk PREFIX.
            pass
        for L in range(min(self.lex.max_len - 1, len(ids)), -1, -1):
            cand = tuple(ids[len(ids) - L:]) if L else ()
            ext = self.by_prefix.get(cand)
            if ext:
                p = np.full(self.V, 0.0); tot = 0.0
                for tok, w in ext.items():
                    if tok < self.V:
                        p[tok] += w; tot += w
                if tot > 0:
                    return p / tot
        return None
    def _dist_ids(self, ids):
        back = self._backoff(ids)
        comp = self._chunk_completion(ids)
        if comp is None:
            return back
        p = self.lam * comp + (1 - self.lam) * back
        return p / p.sum()
    def dist(self, suffix):
        ids = [self.CH[c] for c in suffix if c in self.CH][-self.K:]
        return self._dist_ids(ids)

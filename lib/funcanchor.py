"""funcanchor.py — Exp AX: the function-word anchor voter (free top-k frequency bootstrap).

The bet (Zipf's closed-class generalization; the determiner/preposition as the cheapest possible POS
seed). A toddler does not arrive with a list of "function words"; what arrives free, from raw counting,
is FREQUENCY RANK. The handful of most-frequent tokens in any language IS its closed class — "the",
"of", "to", "and", "a" — and the closed class is exactly the set of items whose RIGHT NEIGHBOUR'S
CATEGORY they predict ("the ___" → a noun; "to ___" → a verb). So: take the top-k tokens by raw leaky
count as anchors, NO labels; per anchor keep a right-neighbour CATEGORY tally; at predict time the cue
"follows-determiner-anchor" carries a counted ecological validity v = hits/(hits+misses), fed into the
take-the-best (AJ) scan beside the AF frame cue. The cheapest category bootstrap there is: a frequency
threshold plus adjacency counts.

Everything is ONLINE single-pass over counts and BOUNDED (~k anchors): no gradient, no k-means/SVD, no
backprop. The category labels come from jepa's online leader clustering (the same substrate AF uses);
the anchor set is just the argmax-k of a running word-frequency bincount.

  AnchorVoter(clu, C, k=20)
    .fit(seq)                  one pass: pick the top-k anchors by freq; tally each anchor's right- and
                               left-neighbour CATEGORY counts.
    .anchors                   the discovered closed-class band (top-k word-ids), no labels.
    .predict_right(prev)       P(next-word CATEGORY | prev is an anchor) → dense (C,) or None (prev not
                               an anchor / no evidence).
    .predict_left(nxt)         P(prev-word CATEGORY | nxt is an anchor) → dense (C,) or None.

The anchor cue is deliberately NARROW: it fires only when the immediately preceding word is one of the
~k anchors. That is its strength (the closed class is where adjacency is most categorial) and its honest
limit (it is silent everywhere else — AF carries the rest).
"""
import numpy as np


def top_k_anchors(seq, k=20, N=None):
    """The free closed-class set: the top-`k` word-ids by raw frequency over the dense id stream (-1 = OOV,
    skipped). This is the whole 'labelling' — a bincount argmax, no linguistic resource. Returns the
    anchor ids sorted by descending frequency, plus the running freq vector (the leaky count, here the
    batch count — order-independent, identical to a streaming leaky tally at the end of the pass)."""
    if N is None:
        N = int(seq.max()) + 1
    freq = np.bincount(seq[seq >= 0], minlength=N).astype(np.int64)
    order = np.argsort(freq)[::-1]
    anchors = order[:k][freq[order[:k]] > 0]
    return anchors.astype(np.int64), freq


class AnchorVoter:
    """The anchor → right/left-neighbour CATEGORY voter. ONLINE, bounded to k anchors.

    For each of the top-k anchor words we keep a length-C tally of the CATEGORY (jepa cluster) of the
    word that immediately FOLLOWS it (right) and the word that immediately PRECEDES it (left). The right
    tally is the productive cue ("the/a/this ___ is a NOUN-category"); the left tally is its mirror
    (object-of-preposition etc.). Counts only — no gradient.
    """

    def __init__(self, clu, C, k=20, alpha=0.1):
        self.clu = np.asarray(clu)
        self.C = int(C)
        self.k = int(k)
        self.alpha = float(alpha)
        self.anchors = None
        self.is_anchor = None        # (N,) bool
        self.right = {}              # anchor word-id -> dense (C,) right-neighbour category counts
        self.left = {}               # anchor word-id -> dense (C,) left-neighbour category counts
        self.freq = None

    def fit(self, seq):
        seq = np.asarray(seq)
        N = len(self.clu)
        self.anchors, self.freq = top_k_anchors(seq, self.k, N=N)
        self.is_anchor = np.zeros(N, bool)
        self.is_anchor[self.anchors] = True
        # right: prev is anchor -> category of next ; left: next is anchor -> category of prev
        prev = seq[:-1]; nxt = seq[1:]
        catn = np.where(nxt >= 0, self.clu[np.clip(nxt, 0, N - 1)], -1)
        catp = np.where(prev >= 0, self.clu[np.clip(prev, 0, N - 1)], -1)
        for a in self.anchors:
            a = int(a)
            rmask = (prev == a) & (catn >= 0)
            if rmask.any():
                self.right[a] = np.bincount(catn[rmask], minlength=self.C).astype(np.float64)
            lmask = (nxt == a) & (catp >= 0)
            if lmask.any():
                self.left[a] = np.bincount(catp[lmask], minlength=self.C).astype(np.float64)
        return self

    def predict_right(self, prev):
        """P(category of the next word | prev), defined ONLY when prev is an anchor with evidence."""
        v = self.right.get(int(prev))
        if v is None:
            return None
        v = v + self.alpha
        return v / v.sum()

    def predict_left(self, nxt):
        v = self.left.get(int(nxt))
        if v is None:
            return None
        v = v + self.alpha
        return v / v.sum()

    def fires_right(self, prev):
        return int(prev) in self.right


# ── category-validity readout: how cleanly a (C,)-vote maps onto a gold POS lexicon ────────────────

def category_pos_purity(votes, gold_pos):
    """POS-cluster purity of a set of (predicted_category, gold_pos) observations.

    `votes` is a list of (cat, pos) pairs: each time a voter PREDICTED category `cat` for a position
    whose word carries gold POS `pos` (gold-tagged words only). Purity = the standard clustering purity:
    sum over predicted categories of (the count of the most common gold POS in that category) / total.
    1.0 = every predicted category is POS-pure (the dream of a category bootstrap); 1/|POS| = chance.
    Returns (purity, n_scored, n_categories_used)."""
    from collections import defaultdict
    by_cat = defaultdict(lambda: defaultdict(int))
    for cat, pos in votes:
        if pos is None:
            continue
        by_cat[cat][pos] += 1
    total = 0; pure = 0
    for cat, posc in by_cat.items():
        tot = sum(posc.values())
        total += tot
        pure += max(posc.values())
    purity = pure / total if total else 0.0
    return purity, total, len(by_cat)

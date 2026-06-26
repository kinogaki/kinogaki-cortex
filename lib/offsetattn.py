"""Offset-keyed count-attention — the principled, COUNT-BASED, no-backprop form of attention.

Real attention asks "which earlier positions predict the next token, and how much?". This is the
count-based answer, with NO learned query/key/value and NO backprop. We keep ONE count table per
relative OFFSET d: T_d[a] = {b: count} = how often word b followed when the word d steps back was a.
Each offset is an expert predicting the SAME next position from a DIFFERENT earlier slot
(position-transformed voting). We weight each offset by its INFORMATIVENESS — the information gain
IG(d) = H(next) - H(next | word_at_offset_d), read straight off the counts. Prediction pools the
per-offset experts (log-linear / geometric-mean, like cortex.vote_sparse) weighted by IG(d), backing
off / abstaining when an offset has never seen the context word.

The whole thing is the Column idea (an associative count-dict) replicated across relative positions and
voted with a data-derived weight per position. That's attention without a single gradient step.

  OffsetAttn(D)               D per-offset count tables + per-offset IG weights.
    .fit(ids)                 stream of int word-ids (UNK = a reserved id is fine, it's just a token).
    .predict(ctx)             ctx = the D preceding word-ids (oldest..newest); -> {wid: prob} or None.
    .ig                       np.array(D) of IG(d) in bits, the learned per-offset weights.

A BAG control reuses the SAME D tables but POOLS THEM OFFSET-AGNOSTICALLY: it asks each context word
"what tends to follow you, at ANY distance?" by merging all offsets into one distance-blind table. The
bag is order-invariant by construction; the offset model is not — that gap is the headline claim.
"""
import math
import numpy as np

ALPHA = 0.05            # matches cortex.ALPHA — add-alpha smoothing for the geometric-mean pool
_TOPK_MEMO = {}         # id(count_dict) -> cached top-K key list (dicts are never mutated post-fit)


class OffsetAttn:
    """D per-offset associative count tables, voted with an information-gain weight per offset."""

    def __init__(self, D=8, gamma=8.0):
        self.D = D
        self.gamma = gamma                          # pooling temperature: weight ∝ IG(d)**gamma
        self.tab = [dict() for _ in range(D + 1)]   # tab[d][a] -> {next_word: count}; index 0 unused
        self.ig = np.zeros(D + 1)                    # IG(d) in bits; index 0 unused
        self.w = np.zeros(D + 1)                     # pooling weight per offset = IG(d)**gamma

    def fit(self, ids):
        """ids: 1-D int array, the word stream. Count next-word given word-at-offset-d, for d=1..D."""
        ids = np.asarray(ids)
        n = len(ids)
        for d in range(1, self.D + 1):
            a = ids[: n - d]            # the word d steps back
            b = ids[d:]                 # the next word (the SAME target position for every d)
            tab = self.tab[d]
            # vectorized accumulation: sort by (a,b), run-length the unique pairs.
            order = np.argsort(a.astype(np.int64) * (ids.max() + 1) + b, kind="stable")
            asd, bsd = a[order], b[order]
            pair = np.stack([asd, bsd], axis=1)
            uniq, cnt = _unique_pairs(pair)
            for (ca, cb), c in zip(uniq, cnt):
                dd = tab.setdefault(int(ca), {})
                dd[int(cb)] = dd.get(int(cb), 0) + int(c)
        self._compute_ig()
        return self

    def _compute_ig(self):
        """IG(d) = H(next) - H(next | word_at_offset_d), both estimated from this offset's counts.

        H(next) is the marginal entropy of the target column from THIS table (so the two terms use the
        identical sample). H(next|a) is the weighted average of the conditional entropies. IG >= 0 always."""
        for d in range(1, self.D + 1):
            tab = self.tab[d]
            marg = {}
            Hcond_weighted = 0.0
            N = 0
            for a, dd in tab.items():
                na = sum(dd.values())
                N += na
                h = 0.0
                for b, c in dd.items():
                    p = c / na
                    h -= p * math.log2(p)
                    marg[b] = marg.get(b, 0) + c
                Hcond_weighted += na * h
            if N == 0:
                self.ig[d] = 0.0
                continue
            Hcond = Hcond_weighted / N
            Hmarg = 0.0
            for b, c in marg.items():
                p = c / N
                Hmarg -= p * math.log2(p)
            self.ig[d] = max(0.0, Hmarg - Hcond)
        # Pooling weight = IG(d)**gamma. A FLAT geometric mean (gamma=1) over-weights the many weakly
        # informative far offsets and blurs the sharp near one (hurts the argmax); sharpening by gamma
        # concentrates the pool on the informative offsets. gamma is the one free knob — still no backprop,
        # the weights are read off entropy. As gamma->inf the model degenerates to the d=1 bigram.
        self.w = self.ig ** self.gamma

    def predict(self, ctx):
        """ctx: length-D sequence of word-ids, oldest..newest (ctx[-1] is t-1, ctx[-D] is t-D).
        Pool the per-offset experts, log-linear, weighted by IG(d)**gamma. Returns {wid: prob} or None."""
        experts = []      # (weight, count_dict)
        for d in range(1, self.D + 1):
            a = ctx[-d]                       # the word d steps back
            dd = self.tab[d].get(int(a))
            if dd:
                experts.append((self.w[d], dd))
        return _weighted_pool(experts)

    # --- BAG control: same tables, offset-agnostic merge -----------------------------------------

    def build_bag(self):
        """Merge ALL offset tables into ONE distance-blind table: bag[a] = {next: count summed over d}.
        This is the offset-AGNOSTIC predictor — give it any context word, it returns what follows that
        word at ANY of the D distances. Order of the context words cannot change its answer set."""
        bag = {}
        for d in range(1, self.D + 1):
            for a, dd in self.tab[d].items():
                tgt = bag.setdefault(a, {})
                for b, c in dd.items():
                    tgt[b] = tgt.get(b, 0) + c
        self.bag = bag
        return self

    def predict_bag(self, ctx):
        """Bag prediction: pool the bag-experts for every distinct context word, UNWEIGHTED (no offset
        info to weight by). Set of experts depends only on WHICH words are in ctx, not their order."""
        experts = []
        for a in set(int(x) for x in ctx):
            dd = self.bag.get(a)
            if dd:
                experts.append((1.0, dd))
        return _weighted_pool(experts)


def _unique_pairs(pair):
    """Unique rows of an (N,2) int array + counts. (np.unique axis=0 is fine at our scale.)"""
    uniq, cnt = np.unique(pair, axis=0, return_counts=True)
    return uniq, cnt


def _weighted_pool(experts, topk=64):
    """Weighted log-linear (geometric-mean) pool over sparse count-dicts, à la cortex.vote_sparse.
    experts: list of (weight, {token: count}). Returns {token: prob} or None. Weights need not sum to 1
    (we normalize); an all-zero-weight set falls back to unweighted.

    Candidates = the union of each expert's TOP-`topk` tokens by count. Low-count tokens never top the
    geometric-mean pool (a single expert's tiny count is overwhelmed by the others' ALPHA floor), so the
    cap is a near-lossless sparse approximation that keeps the union small and the loop fast."""
    if not experts:
        return None
    W = sum(w for w, _ in experts)
    if W <= 0:
        experts = [(1.0, d) for _, d in experts]
        W = float(len(experts))
    keys = set()
    for _, d in experts:
        if len(d) <= topk:
            keys.update(d)
        else:
            cached = _TOPK_MEMO.get(id(d))
            if cached is None:
                cached = [k for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:topk]]
                _TOPK_MEMO[id(d)] = cached
            keys.update(cached)
    out = {}
    for k in keys:
        lp = 0.0
        for w, d in experts:
            lp += w * math.log(d.get(k, 0) + ALPHA)
        out[k] = lp / W          # weighted geometric mean = calibrated pool
    m = max(out.values())
    z = sum(math.exp(v - m) for v in out.values())
    return {k: math.exp(v - m) / z for k, v in out.items()}


def build_word_stream(ids, char_spans, vocab_size=40000):
    """text8 id-array + split_words spans -> (word_id_stream, vocab_list, UNK_id).

    Map each word's char-id span to a string key, keep the top `vocab_size` by frequency, rest -> UNK.
    Returns int32 stream, the kept-word list (index = id), and the UNK id (= len(vocab_list))."""
    from collections import Counter
    A = "abcdefghijklmnopqrstuvwxyz "
    words = []
    for s, e in char_spans:
        words.append(ids[s:e].astype(np.uint8).tobytes())   # raw bytes = a fast hashable word key
    cnt = Counter(words)
    keep = [w for w, _ in cnt.most_common(vocab_size)]
    w2id = {w: i for i, w in enumerate(keep)}
    UNK = len(keep)
    stream = np.fromiter((w2id.get(w, UNK) for w in words), dtype=np.int32, count=len(words))
    vocab_list = ["".join(A[b] for b in w) for w in keep]    # for display only
    return stream, vocab_list, UNK

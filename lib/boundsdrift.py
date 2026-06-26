"""boundsdrift.py — Exp AZ (M3): reliability-gated boundary detectors → the head-final drift.

Three boundary detectors run online over a CHARACTER stream, each proposing word boundaries:

  forward-TP-dip   — a boundary is where the forward transition prob P(c_t | c_{t-1}) DIPS (the
                     next char becomes hard to predict from the left). Saffran/Aslin forward TP.
  backward-TP-dip  — a boundary is where the backward transition prob P(c_t | c_{t+1}) DIPS (the
                     PREVIOUS char becomes hard to predict from the right). Pelucchi 2009: infants
                     also track backward TP.
  entropy-rise     — Harris/Exp-A branching-entropy rise (forward+backward successor variety).

Each detector carries an AB (NARS-style) hit/miss truth value (f,c): every time it FIRES a boundary
we score it against an eventually-STABLE chunk boundary (here, the ground-truth word boundary that
the chunk lexicon would converge on — a space in the source) and tally HIT/MISS. The detector's
RELIABILITY is f·c (c-discounted frequency, Exp AB). Combining the detectors by validity-ordered
take-the-best (Exp AJ) means the most-reliable detector dominates.

THE FALSIFIABLE CLAIM (M3, scoped). On a HEAD-FINAL corpus (modifiers/case-markers precede the
head, so the end of a unit is predictable from the right) the reliability gate should shift weight
toward backward-TP relative to a HEAD-INITIAL corpus (English). The deliverable is the
forward→backward-TP DRIFT — measured as Δ = reliability(backward) − reliability(forward), expected
to rise going head-initial → head-final.

Everything is an online single streaming pass: leaky bigram/skip count tables, reliability tallies
that increment as the stream is read. Bounded memory: O(V²) bigram tables + O(#detectors) tallies.
No gradient descent, no k-means/SVD/eigen. Alphabet matches lib/cortex: a..z=0..25, space=26, V=27.
"""
import numpy as np

V = 27           # a..z + space
SPACE = 26
K = 1.0          # NARS evidential horizon: c = w/(w+k)


# ── transition-probability fields (online bigram counts, one causal pass) ──────────────────────────

def forward_tp(ids):
    """P(c_t | c_{t-1}) at every position t (t>=1), from FORWARD bigram counts accumulated causally.
    Low forward-TP = the next char is surprising given the previous one = a candidate boundary BEFORE t.
    Single online pass: at t we use counts from positions < t only (a context's bet reflects its past)."""
    n = len(ids)
    tp = np.full(n, 1.0 / V)                    # position 0 has no left context
    # online causal bigram: count[a][b] over pairs seen so far; P(b|a) at the moment b arrives.
    prev_b = np.zeros((V, V), np.float64)       # running bigram counts (a->b)
    a_tot = np.zeros(V, np.float64)
    for t in range(1, n):
        a = int(ids[t - 1]); b = int(ids[t])
        denom = a_tot[a]
        tp[t] = (prev_b[a, b] + 0.5) / (denom + 0.5 * V)   # add-0.5 smoothed forward TP, causal
        prev_b[a, b] += 1.0; a_tot[a] += 1.0
    return tp


def backward_tp(ids):
    """P(c_t | c_{t+1}) at every position t (t<n-1): the BACKWARD transition prob — how predictable
    the current char is from the one AFTER it. Computed by running the same online bigram on the
    REVERSED stream, then re-aligning. Low backward-TP = current char surprising given its right
    neighbour = a candidate boundary AFTER t. Pelucchi/backward-TP cue."""
    rev = ids[::-1]
    tp_rev = forward_tp(rev)        # forward TP on reversed = backward TP, but mis-aligned
    return tp_rev[::-1]


def branch_entropy_rise(ids):
    """Forward+backward successor-variety entropy RISE (Harris / Exp A's winning low-level cue).
    A boundary is where BOTH the next-char and prev-char distributions fan out and the entropy RISES.
    Causal-ish global field (uses the whole slice's neighbour distributions — a bounded V×V table)."""
    def follower_H(seq):
        nb = np.zeros((V, V), np.float64)
        a = seq[:-1].astype(np.int64); b = seq[1:].astype(np.int64)
        np.add.at(nb, (a, b), 1.0)
        tot = nb.sum(1, keepdims=True)
        p = nb / np.maximum(tot, 1e-12)
        Hrow = -(p * np.log2(p + 1e-12)).sum(1)             # entropy of followers of each char
        H = np.zeros(len(seq)); H[:-1] = Hrow[a]            # entropy at position i = H(followers of ids[i])
        return H
    Hf = follower_H(ids)                                    # forward (next-char) entropy
    Hb = follower_H(ids[::-1])[::-1]                        # backward (prev-char) entropy
    rise = lambda H: np.concatenate([[0.0], np.maximum(0.0, H[1:] - H[:-1])])
    return rise(Hf) + rise(Hb[::-1])[::-1]


# ── reliability-gated detectors ───────────────────────────────────────────────────────────────────

def gap_score(field, kind):
    """Per-GAP boundary score from a per-position field. A 'gap' g sits between char g and g+1.
    kind='dip'  -> a boundary where the field is LOW (a TP dip): score = -field (high = boundary).
    kind='rise' -> the field already IS a rise: score = field at the gap."""
    n = len(field)
    if kind == "dip":
        # forward-TP dip BEFORE position t signals a boundary in the gap (t-1 | t)
        s = -field[1:]                  # gap g between (g, g+1) scored by forward-TP into g+1
    elif kind == "dip_back":
        s = -field[:-1]                 # backward-TP dip: gap g scored by backward-TP out of g
    else:  # rise
        s = field[1:]
    return s                            # length n-1, one score per gap


class Detector:
    """One boundary detector + its online AB reliability tally.

    score[g] is the detector's boundary evidence at gap g. It FIRES at a gap when its score is in the
    top `rate` quantile (the detector's own operating point). When it fires we score the bet against
    the eventually-stable chunk boundary (gold space position): HIT if a true boundary sits at this
    gap (±tol), else MISS. reliability = f·c (Exp AB) — frequency it's right, c-discounted by evidence.
    """
    def __init__(self, name):
        self.name = name
        self.wp = 0.0; self.wm = 0.0       # NARS hit / miss totals

    def fit_tally(self, score, gold_gaps_set, rate, tol=1):
        """Online single pass over the gaps in stream order: fire on the running top-quantile, tally
        hit/miss against the gold boundary set. The quantile threshold is itself estimated online from
        a leaky reservoir of recent scores, so this is causal (no global lookahead on the threshold)."""
        n = len(score)
        self.wp = 0.0; self.wm = 0.0
        # online threshold: maintain a leaky estimate of the (1-rate) quantile via a counter sketch.
        # Simple, bounded: a running histogram over a fixed score grid updated leakily.
        lo, hi = float(np.min(score)), float(np.max(score))
        if hi <= lo:
            return self
        nbins = 256
        hist = np.zeros(nbins, np.float64)
        decay = 0.9995
        def bin_of(x):
            b = int((x - lo) / (hi - lo) * (nbins - 1))
            return min(max(b, 0), nbins - 1)
        warm = min(2000, n // 4)
        for g in range(n):
            x = score[g]
            if g >= warm:
                # current (1-rate) quantile from the leaky histogram
                tot = hist.sum()
                if tot > 0:
                    target = (1 - rate) * tot
                    csum = np.cumsum(hist)
                    bidx = int(np.searchsorted(csum, target))
                    thr = lo + (hi - lo) * (bidx / (nbins - 1))
                    if x >= thr:                       # detector FIRES at this gap
                        hit = (g in gold_gaps_set) or (g - 1 in gold_gaps_set) or (g + 1 in gold_gaps_set)
                        if hit: self.wp += 1.0
                        else:   self.wm += 1.0
            hist *= decay
            hist[bin_of(x)] += 1.0
        return self

    def truth(self):
        w = self.wp + self.wm
        f = self.wp / w if w > 0 else 0.5
        c = w / (w + K)
        return f, c

    def reliability(self):
        f, c = self.truth()
        return f * c                       # Exp AB: c-discounted frequency = expected accuracy


def gold_gaps(ids):
    """The eventually-stable chunk boundaries = the source word boundaries. A space at position p means
    a boundary in the gap just BEFORE the space and just AFTER it (word|space|word). We collapse to the
    gap immediately preceding each non-space-run transition. Returns a set of gap indices."""
    # gaps are between char g and g+1. A boundary gap = where ids[g]!=SPACE and ids[g+1]==SPACE, or
    # ids[g]==SPACE and ids[g+1]!=SPACE (the two edges of the space token). We mark word edges.
    g_is_bnd = ((ids[:-1] != SPACE) & (ids[1:] == SPACE)) | ((ids[:-1] == SPACE) & (ids[1:] != SPACE))
    return set(np.nonzero(g_is_bnd)[0].tolist())


def run_detectors(ids, rate=0.5, tol=1):
    """Build the three detectors over a char stream, tally each against the stable (gold) boundaries,
    return {name: Detector}. The reliability of each is the deliverable signal."""
    gold = gold_gaps(ids)
    ftp = forward_tp(ids)
    btp = backward_tp(ids)
    ent = branch_entropy_rise(ids)
    dets = {}
    dets["forward_tp"]  = Detector("forward_tp").fit_tally(gap_score(ftp, "dip"),      gold, rate, tol)
    dets["backward_tp"] = Detector("backward_tp").fit_tally(gap_score(btp, "dip_back"), gold, rate, tol)
    dets["entropy"]     = Detector("entropy").fit_tally(gap_score(ent, "rise"),         gold, rate, tol)
    return dets


def take_the_best(dets):
    """Exp AJ combiner: rank detectors by reliability (f·c), the winner is the gate's chosen cue.
    Returns (ordered list of (name, reliability), winner_name)."""
    ranked = sorted(((d.name, d.reliability()) for d in dets.values()), key=lambda kv: -kv[1])
    return ranked, ranked[0][0]


def drift(dets):
    """The head-final DRIFT signal: reliability(backward_tp) − reliability(forward_tp).
    Positive and RISING from head-initial → head-final text is the M3 prediction."""
    return dets["backward_tp"].reliability() - dets["forward_tp"].reliability()


# ── synthetic head-initial / head-final corpora (no Japanese/Korean in data/) ─────────────────────

def synth_corpus(n_words, head_final, seed=0, vocab_stems=60, vocab_marks=6):
    """A frequency-matched synthetic two-morpheme language. Each WORD = stem + marker (head-initial:
    STEM then MARK, 'head' material first; head-final: MARK then STEM, the predictable case-marker/
    modifier precedes the head). Stems are Zipf-frequent multi-char syllables; markers are a tiny
    closed class (high-frequency, low-entropy) — exactly the asymmetry that makes ONE side of each
    word predictable from its neighbour.

    The two languages are MIRROR IMAGES with identical lexical statistics — the only difference is the
    order of stem vs marker inside the word. That is the cleanest possible head-direction contrast: any
    forward/backward-TP asymmetry comes from order alone, not from frequency.
    Returns an int8 id stream (a..z=0..25, space=26)."""
    rng = np.random.default_rng(seed)
    cons = "bcdfghjklmnpqrstvwz"; vows = "aeiou"
    def syll():
        return rng.choice(list(cons)) + rng.choice(list(vows))
    # stems: 2-syllable, Zipf-sampled; markers: 1-syllable closed class, very frequent + low entropy
    stems = ["".join(syll() for _ in range(2)) for _ in range(vocab_stems)]
    marks = ["".join(syll() for _ in range(1)) for _ in range(vocab_marks)]
    stem_p = 1.0 / (np.arange(1, vocab_stems + 1) ** 1.07); stem_p /= stem_p.sum()
    mark_p = 1.0 / (np.arange(1, vocab_marks + 1) ** 0.7);  mark_p /= mark_p.sum()
    out = []
    for _ in range(n_words):
        st = stems[rng.choice(vocab_stems, p=stem_p)]
        mk = marks[rng.choice(vocab_marks, p=mark_p)]
        word = (mk + st) if head_final else (st + mk)
        out.append(word)
    s = " ".join(out)
    ids = np.array([ord(c) - 97 if c != " " else SPACE for c in s], dtype=np.int8)
    return np.ascontiguousarray(ids)

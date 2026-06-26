"""Usage-based CONSTRUCTION induction — grammar as counting, made productive. ONLINE, NO backprop.

The thesis (Bybee frequency effects; Goldberg constructions; Tomasello usage-based acquisition;
statistical preemption). A "grammar" is not a rule system handed down from above; it is what falls out of
COUNTING usage, if you count two things at once for every context (frame):

  - TOKEN count  N(frame)      = how often this exact frame occurred, and N(frame -> w) per specific filler.
  - TYPE  count  V(frame)      = how many DISTINCT fillers followed the frame.

Two ratios drive two opposite fates (Bybee's two kinds of frequency):

  (a) ENTRENCHMENT — high TOKEN count + low type-richness => the frame + its dominant filler FREEZE into a
      single chunk unit (a frozen idiom / collocation). It predicts the SPECIFIC continuation.
  (b) ABSTRACTION — high TYPE/TOKEN ratio (many distinct fillers, none dominant) => the frame SPAWNS an
      OPEN-SLOT CONSTRUCTION: it predicts the filler CATEGORY (a cluster) rather than any specific filler.
      This is the productive schema; it can fire for fillers never seen in this frame, if they fall in the
      slot's category. That is compositional generalization, induced by counting alone.

  (c) STATISTICAL PREEMPTION — when two frames compete to express the same slot+category (the construction
      and a near-synonym frame), the OBSERVED pairing is up-weighted and the UNOBSERVED competitor's link is
      DOWN-WEIGHTED (count-based inhibition). "He gave me the ball" preempts "*He gave me the ball to."

Everything here is ONLINE: per-frame running token/type counts + the filler categories come from the same
online leader-clustering used in jepa.py (single pass, nearest running-mean prototype or spawn). No gradient
descent, no k-means, no SVD. The vectorized builders are batched order-independent accumulation — identical
counts to a token-at-a-time online update.

  ConstructionGrammar(C, fillers_clu, ...)
    .fit(seq)                  stream of dense word-ids (-1 = OOV, skipped); count every frame's fillers.
    .classify()                label each frame: 'frozen' | 'open-slot' | 'mixed' by token/type counts.
    .predict_open(frame)       open-slot prediction: P(filler-CATEGORY | frame) -> dense (C,) or None.
    .predict_specific(frame)   frozen prediction:    P(specific filler | frame) -> {wid: p} or None.
    .preempt()                 apply count-based inhibition between competing frames for the same category.

A FRAME here is the preceding word (1-gram frame, "X ___") — the simplest construction slot. The same
machinery extends to 2-gram frames ("X Y ___"); we count both and let the longer, sparser frame back off to
the shorter when it has too little evidence (usage-based backoff).
"""
import numpy as np


# ── per-frame filler bookkeeping: token count (specific) + type count (distinct, via filler CATEGORY) ──

def build_frame_counts(seq, order=1):
    """One pass: for every `order`-gram frame, accumulate {filler_word -> count}. Vectorized (np.unique over
    packed (frame, filler) keys) = batched order-independent accumulation, identical to a token-at-a-time
    online update. Returns dict frame_key(int) -> (filler_ids np.array, counts np.array), plus the packer.

    order=1: frame is the single preceding word id. order=2: frame is packed prev2*N + prev1 (N = vocab)."""
    n = len(seq)
    if order == 1:
        frame = seq[:-1].astype(np.int64)
        filler = seq[1:].astype(np.int64)
        m = (frame >= 0) & (filler >= 0)
    else:
        N = int(seq.max()) + 1
        frame = seq[:-2].astype(np.int64) * N + seq[1:-1].astype(np.int64)
        filler = seq[2:].astype(np.int64)
        m = (seq[:-2] >= 0) & (seq[1:-1] >= 0) & (filler >= 0)
    frame, filler = frame[m], filler[m]
    if frame.size == 0:
        return {}
    F = int(filler.max()) + 1
    key = frame * F + filler
    uk, uc = np.unique(key, return_counts=True)
    uf = uk // F            # frame
    uw = (uk % F).astype(np.int64)   # filler word
    out = {}
    edges = np.nonzero(np.diff(uf))[0] + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(uk)]])
    for s, e in zip(starts, ends):
        out[int(uf[s])] = (uw[s:e], uc[s:e].astype(np.float64))
    return out


class FrameStats:
    """Token/type statistics for one frame, computed from its (filler_ids, counts) and a filler->category map.

    token  = total occurrences (sum of counts)            -> Bybee token frequency (entrenchment driver)
    types  = number of DISTINCT fillers                   -> Bybee type frequency (productivity driver)
    cats   = number of DISTINCT filler CATEGORIES (clusters) covered; productivity is really about how many
             distinct *categories* the slot accepts, not just distinct words (Goldberg/Tomasello).
    domfrac= fraction of tokens taken by the single most frequent filler (1.0 = fully frozen).
    """
    __slots__ = ("frame", "fids", "cnt", "token", "types", "cats", "domfrac", "dom", "cat_counts")

    def __init__(self, frame, fids, cnt, clu_of):
        self.frame = frame
        self.fids = fids
        self.cnt = cnt
        self.token = float(cnt.sum())
        self.types = int(len(fids))
        cats = clu_of[fids]                       # category (cluster) of each filler
        valid = cats >= 0
        self.cat_counts = {}
        for c, w in zip(cats[valid], cnt[valid]):
            self.cat_counts[int(c)] = self.cat_counts.get(int(c), 0.0) + float(w)
        self.cats = int(len(self.cat_counts))
        i = int(cnt.argmax())
        self.dom = int(fids[i])
        self.domfrac = float(cnt[i] / self.token) if self.token > 0 else 0.0


# ── the grammar: classify frames into frozen idioms vs open-slot constructions; predict accordingly ──

class ConstructionGrammar:
    """Induce constructions from frame counts. ONLINE: every count is accumulated in one pass; categories come
    from jepa-style online leader clustering (passed in as clu_of)."""

    def __init__(self, clu_of, C, alpha=0.1,
                 min_token=40, freeze_dom=0.55, open_types=8, open_ttr=0.30):
        """clu_of: (N,) filler-word -> category id (-1 = unclustered). C: #categories.
        min_token: a frame needs this many tokens before we judge it (online 'ripe').
        freeze_dom: dominant-filler fraction at/above which a (ripe) frame FREEZES to that filler (idiom).
        open_types / open_ttr: a frame with >= open_types distinct fillers AND type/token-richness above the
            schema bar SPAWNS an open-slot construction. (We use a category-spread test, see classify.)"""
        self.clu_of = np.asarray(clu_of)
        self.C = C
        self.alpha = alpha
        self.min_token = min_token
        self.freeze_dom = freeze_dom
        self.open_types = open_types
        self.open_ttr = open_ttr
        self.frames = {}            # frame_key -> FrameStats
        self.label = {}             # frame_key -> 'frozen' | 'open-slot' | 'mixed' | 'sparse'
        self.cat_tab = {}           # frame_key -> dense (C,) category distribution (open-slot head)
        self.inhib = {}             # (frame_key, cat) -> multiplicative down-weight from preemption (<=1)

    def fit(self, frame_counts):
        """frame_counts: dict frame_key -> (filler_ids, counts) from build_frame_counts (one streaming pass)."""
        for fk, (fids, cnt) in frame_counts.items():
            self.frames[fk] = FrameStats(fk, fids, cnt, self.clu_of)
        return self

    def classify(self):
        """Label every frame by its token/type/category profile (Bybee's two frequency effects, mechanized).

          frozen     : ripe, and one filler owns >= freeze_dom of the tokens -> entrenched chunk (idiom).
          open-slot  : ripe, many distinct fillers spread over several categories, no single dominator ->
                       a productive schema predicting the CATEGORY.
          mixed      : ripe but neither extreme (a dominant filler AND a productive tail).
          sparse     : not enough tokens yet to judge (stays a plain count, no construction claim).
        """
        for fk, fs in self.frames.items():
            if fs.token < self.min_token:
                self.label[fk] = "sparse"; continue
            ttr = fs.types / fs.token
            if fs.domfrac >= self.freeze_dom:
                self.label[fk] = "frozen"
            elif fs.types >= self.open_types and fs.cats >= 3 and ttr >= self.open_ttr / 10:
                # productive: spreads over >=3 categories with no single filler dominating. (ttr bar is gentle
                # — high-token frames have naturally low raw TTR; the category-spread test carries the weight.)
                self.label[fk] = "open-slot"
            else:
                self.label[fk] = "mixed"
        # Build the open-slot CATEGORY head for every non-frozen frame (these predict the slot category).
        for fk, fs in self.frames.items():
            if self.label[fk] in ("open-slot", "mixed"):
                v = np.zeros(self.C)
                for c, w in fs.cat_counts.items():
                    v[c] = w
                self.cat_tab[fk] = v
        return self.label

    # ── prediction heads ──────────────────────────────────────────────────────────────────────────

    def predict_specific(self, frame):
        """Frozen/specific head: smoothed P(specific filler | frame) as {wid: p}. None if frame unseen."""
        fs = self.frames.get(int(frame))
        if fs is None:
            return None
        tot = fs.token + self.alpha * (self.C if self.C else 1)
        return {int(w): float((c + self.alpha) / tot) for w, c in zip(fs.fids, fs.cnt)}

    def predict_open(self, frame):
        """Open-slot head: P(filler CATEGORY | frame) as a dense (C,) array, with preemption inhibition applied.
        Returns None if the frame has no category head (sparse / fully frozen)."""
        v = self.cat_tab.get(int(frame))
        if v is None:
            return None
        v = v.copy()
        for (ifk, c), mult in self.inhib.items():
            if ifk == int(frame):
                v[c] *= mult
        v = v + self.alpha
        return v / v.sum()

    def predict_filler_via_category(self, frame, cat_prior=None):
        """COMPOSITIONAL head: predict the next FILLER WORD through the open slot's CATEGORY, so it can score a
        filler this frame has NEVER hosted. P(w | frame) = P(category(w) | frame, open-slot) * P(w | category),
        where P(w|category) is the GLOBAL category-internal filler frequency (counted once, frame-independent).
        This is the productive generalization: the frame supplies the slot's category distribution, the category
        supplies which words live in it. Returns {wid: p} over the union of the active categories' members, or
        None if no open-slot head. `cat_prior` (C,) optional override of P(category|frame) (e.g. preemption-free).
        """
        pc = self.predict_open(frame) if cat_prior is None else cat_prior
        if pc is None:
            return None
        out = {}
        active = np.nonzero(pc > (pc.min() + 1e-12))[0] if pc.size else []
        for c in (active if len(active) else range(self.C)):
            pw = self._cat_word_prob.get(int(c))
            if pw is None:
                continue
            for w, p in pw.items():
                out[w] = out.get(w, 0.0) + pc[c] * p
        if not out:
            return None
        z = sum(out.values())
        return {w: p / z for w, p in out.items()}

    def build_category_lexicon(self, frame_counts):
        """P(word | category): the global frequency of each word WITHIN its category, counted across ALL frames
        (frame-independent). This is the 'what fills this slot type' knowledge the compositional head reuses.
        One pass over the frame counts (order-independent accumulation)."""
        cat_word = {}
        for fk, (fids, cnt) in frame_counts.items():
            cats = self.clu_of[fids]
            for c, w, n in zip(cats, fids, cnt):
                if c < 0:
                    continue
                d = cat_word.setdefault(int(c), {})
                d[int(w)] = d.get(int(w), 0.0) + float(n)
        self._cat_word_prob = {}
        for c, d in cat_word.items():
            z = sum(d.values())
            self._cat_word_prob[c] = {w: n / z for w, n in d.items()}
        return self

    # ── statistical preemption: count-based inhibition between competing frames ──────────────────────

    def preempt(self, strength=0.5):
        """For each category, frames compete to express it. The frame that uses the category MOST (the
        conventional/observed expression) inhibits the SAME category's link in frames that use it far less for
        their token volume (the unattested-by-comparison competitor). Multiplicative down-weight in [strength,1]
        proportional to how under-attested the competitor is. Pure counting — no gradient.

        Concretely: for category c, let r(frame) = cat_count[frame,c]/token[frame] be how strongly that frame
        commits to c. The strongest committer wins; a competitor with r much smaller has its (frame,c) link
        scaled toward `strength`. This reproduces preemption: a near-synonym frame that COULD take filler-class
        c but is rarely observed doing so gets its c-link suppressed, reducing over-generation."""
        # gather, per category, the committing frames and their commitment ratios
        by_cat = {}
        for fk, fs in self.frames.items():
            if self.label.get(fk) not in ("open-slot", "mixed"):
                continue
            for c, w in fs.cat_counts.items():
                r = w / fs.token
                by_cat.setdefault(c, []).append((fk, r))
        for c, lst in by_cat.items():
            if len(lst) < 2:
                continue
            rmax = max(r for _, r in lst)
            if rmax <= 0:
                continue
            for fk, r in lst:
                rel = r / rmax                      # 1.0 for the leader, ->0 for an under-attested competitor
                if rel < 1.0:
                    # down-weight toward `strength` the more under-attested it is
                    self.inhib[(fk, c)] = strength + (1 - strength) * rel
        return self


# ── n-gram baselines for the right-axis comparison ───────────────────────────────────────────────

class NgramBackoff:
    """Plain smoothed n-gram next-word model over the SAME frames — the control the constructions must beat on
    compositional generalization. predict(frame) -> {wid: p}. By construction it can ONLY predict fillers it has
    literally seen after this frame; on a held-out (frame, filler) combo it assigns the smoothing floor. That is
    exactly the gap an open-slot construction is meant to close."""
    def __init__(self, frame_counts, vocab, alpha=0.1):
        self.tab = frame_counts
        self.V = vocab
        self.alpha = alpha
        # unigram backoff
        tot = np.zeros(vocab)
        for fids, cnt in frame_counts.values():
            np.add.at(tot, fids, cnt)
        self.uni = (tot + alpha) / (tot.sum() + alpha * vocab)

    def predict(self, frame):
        e = self.tab.get(int(frame))
        if e is None:
            return None
        fids, cnt = e
        tot = cnt.sum() + self.alpha * self.V
        return {int(w): float((c + self.alpha) / tot) for w, c in zip(fids, cnt)}

    def prob_of(self, frame, w):
        """P(w | frame) with backoff to unigram when the (frame,w) pair was never seen (the held-out case)."""
        e = self.tab.get(int(frame))
        if e is None:
            return float(self.uni[w])
        fids, cnt = e
        tot = cnt.sum() + self.alpha * self.V
        hit = np.where(fids == w)[0]
        if len(hit):
            return float((cnt[hit[0]] + self.alpha) / tot)
        return float(self.alpha / tot)             # held-out filler -> pure smoothing floor

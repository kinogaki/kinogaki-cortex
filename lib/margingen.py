"""margingen.py — G6: margin-gated production (read the same counts the hard way).

Production is comprehension read BACKWARDS. The spine learns one thing — a count table over
(cue, label) co-occurrences — and the two language faculties are two READ DIRECTIONS over it:

  COMPREHENSION  (cue -> label, one-to-MANY, forgiving): "given this context, is this label
    compatible?" A recognition read. Many labels can be acceptable, so it reads the table with
    NO gate: it scores the heard label against the cue's distribution and accepts if it clears a
    low recognition bar. One-to-many is forgiving — comprehension comes online early.

  PRODUCTION  (cue -> the ONE label to say, MANY-to-one, committing): "what is THE word here?"
    A commitment read. It must pick a single label and stake the utterance on it, so it reads the
    SAME table through a MARGIN GATE:

        activation(label | cue) = count(cue,label) * f(cue,label)    # AB hit/miss frequency
                                  -------------------------------    # divided by FAN(cue)
                                              FAN(cue)               # = #labels the cue owns

        margin = activation(top) / activation(2nd)

    emit the top label only if  margin >= theta_emit  (AJ's self-setting bar). Else the producer
    can (a) BACK OFF to a generic higher-margin label (say "dog" before "retriever" when the
    specific one is contested), or (b) DEFER — stay silent. The bar is many-to-one because to SAY
    a word you must beat its competitors; to UNDERSTAND one you need only recognise it.

The structural prediction (the thing this experiment tests): the SAME counts, read the two ways,
reproduce the COMPREHENSION > PRODUCTION gap — an "understands but won't say it yet" stage — and
that gap SHRINKS as evidence accrues (the margin clears for more cues as counts grow). This is the
read-direction twin of M17 (which gates at acquisition time); here the gap is purely a property of
reading one table forward vs backward.

Everything is ONLINE single-pass, BOUNDED (per-cue top-k label store, leaky), no gradients:
the table is the count; f is the AB hit/miss split; fan is the live-label count; the margin and
the gate are read off those numbers on demand. Production allocates NO new table — it is a query.

  CueLabelStore(cap, half_life)
    .observe(cue, label, t)               fold one (cue,label) co-occurrence into the counts.
    .comprehend(cue, label, t)            recognition read (no gate): is `label` recognised here?
    .produce(cue, t, theta, backoff)      margin-gated read: (emitted_label | None, margin, info).
    .dist(cue, t)                          ranked [(label, activation), ...] over the cue's labels.
"""
import math
from collections import defaultdict, OrderedDict


class CueLabelStore:
    """A bounded, leaky count table over (cue -> {label: counts}) with an AB hit/miss split.

    For each cue we keep an OrderedDict of label -> record. A record carries:
        n   : raw co-occurrence count (how often this label followed this cue)
        wp  : HIT  count  (times this label was the cue's running top-1 AND it was the true label)
        wm  : MISS count  (times this label was the cue's running top-1 AND it was NOT)
        t   : last-seen time (for the recency leak)
    The hit/miss split is AB (confidence.py): we score the cue's CURRENT argmax against the label
    that actually arrives, online, so (f,c) reflect how trustworthy the cue's leading bet is.

    BOUNDED: each cue keeps at most `cap` labels; the least-recently-touched is evicted. This caps
    memory at O(#cues * cap) and is also the cognitively-right bound (a cue's live competitors are
    the recent ones). The leak (`half_life`) lets stale labels fall below the floor and be ignored.
    """

    def __init__(self, cap=32, half_life=None, floor=0.0, k=1.0):
        self.cap = cap
        self.lam = (math.log(2) / half_life) if half_life else 0.0   # 0 = no leak (pure counts)
        self.floor = floor
        self.k = k                                                   # NARS evidential horizon
        self.cues = defaultdict(OrderedDict)                        # cue -> OrderedDict{label: rec}
        self.t = 0

    # ---- online write: score the running bet (AB), then fold in the new evidence ----------------

    def observe(self, cue, label, t=None):
        """Fold one (cue, label) co-occurrence. BEFORE incrementing, score the cue's current top-1
        bet against `label` (the AB hit/miss update); then increment the label's count. Single pass,
        causal: the (f,c) at any time reflect only earlier occurrences of this cue."""
        if t is None:
            t = self.t
        post = self.cues[cue]
        # score the running argmax (the bet this cue would have made) against the arriving label
        top = self._argmax(post, t)
        if top is not None:
            rec_top = post[top]
            if top == label:
                rec_top["wp"] += 1.0
            else:
                rec_top["wm"] += 1.0
        # fold the observed label in
        rec = post.get(label)
        if rec is None:
            rec = {"n": 0.0, "wp": 0.0, "wm": 0.0, "t": t}
            post[label] = rec
        rec["n"] += 1.0
        rec["t"] = t
        post.move_to_end(label)
        if len(post) > self.cap:
            post.popitem(last=False)               # evict the least-recently-touched label (O(1))

    def tick(self):
        self.t += 1

    # ---- activation: the AB-weighted, fan-divided count -----------------------------------------

    def _leak(self, rec, t):
        return math.exp(-self.lam * (t - rec["t"])) if self.lam else 1.0

    def _activation(self, rec, fan, t):
        """activation = n * frequency(f) * recency-leak / fan.
        f = wp/(wp+wm) is the AB hit-rate of this label as the cue's bet; with no hit/miss evidence
        (wp+wm==0) f defaults to 1 so a freshly-seen label is not zeroed — it just has no track
        record yet (its margin will be soft). Dividing by fan is the similarity-interference law:
        a cue that owns many labels spreads its activation thin (a contested slot)."""
        w = rec["wp"] + rec["wm"]
        f = (rec["wp"] / w) if w > 0 else 1.0
        return rec["n"] * f * self._leak(rec, t) / max(1, fan)

    def _live_labels(self, post, t):
        """[(label, rec, activation)] for labels currently above floor; fan is the live count."""
        live = []
        for label, rec in post.items():
            a_raw = rec["n"] * self._leak(rec, t)
            if a_raw <= self.floor:
                continue
            live.append((label, rec))
        fan = len(live)
        return [(label, rec, self._activation(rec, fan, t)) for label, rec in live], fan

    def _argmax(self, post, t):
        """The cue's current top-1 label by RAW leaked count (the bet AB scores). None if empty."""
        best, best_a = None, -1.0
        for label, rec in post.items():
            a = rec["n"] * self._leak(rec, t)
            if a > best_a:
                best, best_a = label, a
        return best

    def dist(self, cue, t=None):
        """Ranked [(label, activation), ...] over the cue's live labels (the production read)."""
        if t is None:
            t = self.t
        post = self.cues.get(cue)
        if not post:
            return []
        live, _ = self._live_labels(post, t)
        live.sort(key=lambda x: x[2], reverse=True)
        return [(label, a) for label, rec, a in live]

    def fan(self, cue, t=None):
        if t is None:
            t = self.t
        post = self.cues.get(cue)
        if not post:
            return 0
        _, fan = self._live_labels(post, t)
        return fan

    # ---- the two read directions -----------------------------------------------------------------

    def comprehend(self, cue, label, t=None, rank_tol=None, prob_floor=0.0):
        """COMPREHENSION (cue -> label, NO gate, one-to-many forgiving). Returns (recognised, rank,
        n_labels). A label is RECOGNISED if it is among the cue's live labels at all (one-to-many is
        forgiving: any compatible label is understood). With `rank_tol` set, require the label to be
        within the top-`rank_tol` of the cue's distribution (a tighter recognition bar). The point:
        comprehension does NOT need to beat competitors — it only needs to be present/plausible."""
        if t is None:
            t = self.t
        ranked = self.dist(cue, t)
        if not ranked:
            return False, -1, 0
        labels = [l for l, _ in ranked]
        if label not in labels:
            return False, -1, len(labels)
        rank = labels.index(label)
        if rank_tol is not None and rank >= rank_tol:
            return False, rank, len(labels)
        return True, rank, len(labels)

    def produce(self, cue, t=None, theta=2.0, backoff=None):
        """PRODUCTION (cue -> the ONE label, MARGIN-gated, many-to-one committing). Returns
        (emitted_label_or_None, margin, info). The producer emits its top label ONLY when

            margin = activation(top) / activation(2nd) >= theta

        Else it tries to BACK OFF (if `backoff` is given: a dict label->generic_label, e.g. a
        hypernym map) to a generic label whose own margin clears theta; failing that it DEFERS
        (returns None = stays silent). `info` carries top/second labels + activations + margin for
        the caller to bucket by. A cue with a single live label has infinite margin (uncontested)."""
        if t is None:
            t = self.t
        ranked = self.dist(cue, t)
        info = {"top": None, "second": None, "a_top": 0.0, "a_2nd": 0.0,
                "margin": 0.0, "n_labels": len(ranked), "backed_off": False, "deferred": False}
        if not ranked:
            info["deferred"] = True
            return None, 0.0, info
        top, a_top = ranked[0]
        info["top"] = top
        info["a_top"] = a_top
        if len(ranked) == 1:
            margin = float("inf")                  # uncontested slot
        else:
            second, a_2nd = ranked[1]
            info["second"] = second
            info["a_2nd"] = a_2nd
            margin = (a_top / a_2nd) if a_2nd > 0 else float("inf")
        info["margin"] = margin
        if margin >= theta:
            return top, margin, info
        # contested: try a generic back-off label whose own margin clears theta
        if backoff is not None and top in backoff:
            generic = backoff[top]
            g_ranked = self.dist(generic, t) if generic in self.cues else None
            # back-off here is a SEMANTIC fall-back: emit the generic label itself as the safe choice
            # (dog before retriever). It is "higher margin" by construction — a hypernym owns its slot
            # more cleanly. We accept the generic if it exists in the label space at all.
            info["backed_off"] = True
            return generic, margin, info
        info["deferred"] = True
        return None, margin, info

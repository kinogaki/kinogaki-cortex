"""Representational REDESCRIPTION — turning an implicit count into an explicit, manipulable concept. ONLINE, NO backprop.

Karmiloff-Smith, *Beyond Modularity* (the RR model). The named gap in this cortex: our counts and clusters are
IMPLICIT — a black box that maps input→output, whose PARTS are not separately addressable. You can ask the
construction "what comes after `gave` ?" but you cannot ask it "show me everything that fills a RECIPIENT slot
regardless of the verb" — the slot isn't a thing you can name, only a side effect of a count.

RR's claim: knowledge that already WORKS implicitly gets spontaneously RE-DESCRIBED into a more EXPLICIT format
whose parts ARE addressable and recombinable — and the trigger is **STABILITY / mastery, NOT error**. A behaviour
that has stopped changing is "done"; the system then re-represents it so System 2 can bind, compare, and recombine
its parts. (KS calls the levels Implicit → E1 → E2/E3; we model the Implicit→E1 step: the first explicit redescription.)

Both KS signatures are testable as COUNTS:
  (1) MANIPULABILITY — the explicit form answers a COMPOSITIONAL query the raw count cannot (cross-frame slot
      lookup, substitution, analogy). We build an explicit slot-structured object and run a query the implicit
      table structurally cannot answer.
  (2) The U-SHAPED DIP — KS predicts a transient REGRESSION right after redescription: the smooth implicit form
      briefly hands over to the cruder, just-promoted explicit form, accuracy dips, then recovers. We measure
      prediction accuracy over the exposures around each promotion event.

EVERYTHING here is online, count-native, no gradients:
  - StabilityMonitor: a leaky per-construction detector of "has my leader-count + next-token distribution stopped
    moving over the last N exposures" — a count-native MASTERY signal (no error term anywhere).
  - redescribe(): on stability (not error) promote the frozen co-firing pattern into an explicit SlotObject — a
    named, slot-structured node with separately-addressable ROLE (the frame) and FILLER (the slot category +
    its members) — WITHOUT touching the underlying counts. Promotion is a re-description, not a re-training.
  - SlotRegistry: the explicit layer. Holds promoted SlotObjects and an INVERTED index slot-category → frames,
    so System-2 queries ("who fills this slot anywhere", "substitute the frame", "complete the analogy") are
    O(lookup) over explicit parts — a capability the flat implicit count does not expose.
"""
import numpy as np


# ── STABILITY MONITOR: a count-native mastery detector (stability triggers, NOT error) ──

class StabilityMonitor:
    """Per construction (frame), watch two things stop moving over the last `window` exposures:

      - the LEADER: which filler-category the frame commits to most (argmax of its running category counts).
      - the next-token DISTRIBUTION: the frame's category distribution (L1-normalized running counts).

    A construction is STABLE / mastered when, across the last `window` exposures, (a) the leader category never
    changed and (b) the distribution's total-variation drift per exposure stayed below `tv_eps`. This is mastery
    by *settling*, not by getting an answer right — there is no target and no error term. Leaky/bounded: we keep
    only a small ring of recent snapshots per frame, so memory is O(frames · window), not O(stream).

    `note(frame, dist)` is called once per exposure with the frame's CURRENT (post-update) category distribution
    — a running count vector the construction already maintains. The monitor does not own the counts; it only
    observes their trajectory (the RR trigger reads the implicit system, it does not alter it).
    """
    __slots__ = ("window", "tv_eps", "leader", "last", "stable_run", "exposures", "promoted_at")

    def __init__(self, window=6, tv_eps=0.02):
        self.window = window          # exposures the behaviour must hold steady
        self.tv_eps = tv_eps          # max per-exposure total-variation drift to count as "not moving"
        self.leader = {}              # frame -> last leader category
        self.last = {}                # frame -> last distribution (dense float, sparse-ish but kept small)
        self.stable_run = {}          # frame -> consecutive stable exposures
        self.exposures = {}           # frame -> total exposures seen
        self.promoted_at = {}         # frame -> exposure index at promotion (None until promoted)

    def note(self, frame, dist):
        """Observe one exposure of `frame` with current category distribution `dist` (dense, sums≈1)."""
        self.exposures[frame] = self.exposures.get(frame, 0) + 1
        lead = int(dist.argmax())
        prev = self.last.get(frame)
        steady = False
        if prev is not None:
            tv = 0.5 * float(np.abs(dist - prev).sum())     # total-variation distance between successive dists
            same_leader = (self.leader.get(frame) == lead)
            steady = same_leader and (tv <= self.tv_eps)
        self.leader[frame] = lead
        self.last[frame] = dist
        self.stable_run[frame] = (self.stable_run.get(frame, 0) + 1) if steady else 0
        return self.stable_run[frame]

    def is_stable(self, frame):
        """Mastered: held leader + sub-eps drift for `window` consecutive exposures, and not yet promoted."""
        return (self.stable_run.get(frame, 0) >= self.window
                and frame not in self.promoted_at)

    def mark_promoted(self, frame):
        self.promoted_at[frame] = self.exposures.get(frame, 0)


# ── THE EXPLICIT LAYER: a slot-structured object with separately-addressable parts ──

class SlotObject:
    """An EXPLICIT redescription of one stable construction. Where the implicit count is a flat map
    frame→{filler:count}, this names the parts so System 2 can address them:

      - role     : the frame word (the construction's identity, e.g. "gave", "to", "number").
      - slot_cat : the dominant filler CATEGORY id this slot selects for (the leader, now frozen + named).
      - fillers  : the category's member word-ids + their global P(word|category) — the slot's *type*, reusable
                   across every construction that selects the same category. SEPARATELY ADDRESSABLE from the role.
      - dist     : a SNAPSHOT of the implicit category distribution at promotion (frozen; the counts keep moving).

    Crucially the object is built BY COPYING the stable counts — promotion does not alter the implicit table.
    The point of E1 is not better counts; it is parts you can *name and recombine*.
    """
    __slots__ = ("role", "slot_cat", "fillers", "filler_p", "dist", "promoted_at", "_specific_top")

    def __init__(self, role, slot_cat, fillers, filler_p, dist, promoted_at, specific_top=-1):
        self.role = role
        self.slot_cat = slot_cat
        self.fillers = fillers          # np.array of word-ids in the slot category
        self.filler_p = filler_p        # np.array P(word|category), aligned with fillers
        self.dist = dist                # frozen category distribution snapshot at promotion
        self.promoted_at = promoted_at
        self._specific_top = specific_top   # this role's own dominant filler (the specifics it RE-BINDS at E2)


class SlotRegistry:
    """The EXPLICIT knowledge layer: the promoted SlotObjects + an INVERTED index that the implicit count cannot
    provide. Implicit counts are indexed BY FRAME (you must name the verb to see its fillers). The registry adds
    the inverse — indexed BY SLOT CATEGORY (name the slot, get every construction that uses it) — which is what
    makes cross-frame substitution and analogy O(lookup). This index is the manipulability KS predicts.
    """
    def __init__(self):
        self.objects = {}               # frame(role) -> SlotObject
        self.by_cat = {}                # slot_cat -> set(frames that select it)  (the INVERTED index)

    def add(self, obj):
        self.objects[obj.role] = obj
        self.by_cat.setdefault(obj.slot_cat, set()).add(obj.role)

    # ── the compositional queries the FLAT IMPLICIT COUNT structurally cannot answer ──

    def frames_filling(self, slot_cat):
        """"Which constructions fill THIS slot, regardless of their role?"  Inverted lookup — impossible on a
        frame-keyed count without scanning + re-deriving every frame's leader (the parts aren't addressable)."""
        return sorted(self.by_cat.get(slot_cat, set()))

    def slot_of(self, role):
        o = self.objects.get(role)
        return None if o is None else o.slot_cat

    def substitute(self, role, new_role):
        """Analogical SUBSTITUTION: keep this construction's SLOT (filler type) but swap its ROLE. Returns the
        new role bound to the same slot's fillers — a recombination of separately-addressable parts. The implicit
        count has no 'slot' to carry over; it can only report new_role's own raw counts."""
        o = self.objects.get(role)
        if o is None:
            return None
        return dict(role=new_role, slot_cat=o.slot_cat, fillers=o.fillers, filler_p=o.filler_p)

    def analogy(self, a, b, c):
        """a : b :: c : ?  over slots. If a fills slot S and c also fills slot S, then the analogy completes to
        a construction that fills b's slot — pure part-binding over the explicit index. Returns candidate roles
        that fill b's slot (the answer set), or None if the relation doesn't hold over named slots."""
        sa, sc = self.slot_of(a), self.slot_of(c)
        sb = self.slot_of(b)
        if sa is None or sb is None or sc is None:
            return None
        if sa != sc:                    # a and c must share a slot for the analogy to be slot-structured
            return None
        return self.frames_filling(sb)


def redescribe(frame, frames_stats, clu, C, cat_word_prob, exposure_idx):
    """The REDESCRIPTION PASS — promote one STABLE construction's frozen co-firing pattern into an explicit
    SlotObject, WITHOUT altering any count. Reads the implicit construction (its running category counts + the
    global category lexicon) and re-describes its dominant slot as a named, slot-structured object.

    frames_stats : FrameStats for this frame (has .cat_counts, the running per-category token counts).
    clu, C       : the online leader-clustering labels (word→category) and category count.
    cat_word_prob: {cat -> {word: P(word|cat)}} the global, frame-independent slot lexicon (already counted).
    Returns a SlotObject or None (no categorical leader to name).
    """
    cc = frames_stats.cat_counts
    if not cc:
        return None
    tot = sum(cc.values())
    dist = np.zeros(C)
    for c, w in cc.items():
        dist[c] = w
    dist = dist / max(tot, 1e-9)
    slot_cat = int(dist.argmax())
    pw = cat_word_prob.get(slot_cat, {})
    if not pw:
        return None
    fillers = np.array(sorted(pw.keys()), dtype=np.int64)
    filler_p = np.array([pw[int(w)] for w in fillers], dtype=np.float64)
    # this role's OWN dominant filler (the specific knowledge re-bound at E2, separately addressable from the slot)
    specific_top = int(frames_stats.fids[int(frames_stats.cnt.argmax())]) if len(frames_stats.fids) else -1
    return SlotObject(role=frame, slot_cat=slot_cat, fillers=fillers,
                      filler_p=filler_p, dist=dist.copy(), promoted_at=exposure_idx,
                      specific_top=specific_top)

"""dualroute.py — M19/BA: a dual-route inflection head with f·c blocking.

The bet (Pinker's *Words and Rules*; Rumelhart & McClelland's single-route net; Marcus et al.'s
over-regularization corpus; Weissweiler et al.'s *graded* productivity). Past-tense inflection is the
canonical battleground: do children store irregulars whole (*went*) and compute regulars by rule
(*walk+ed*), or does ONE associative system do both? The count-native claim is that the two "routes" are
not two systems — they are two reads of the SAME counter, fused by one tunable gate, and the famous
U-shaped over-regularization curve falls out of that gate for free, item by item, never as one macro dip.

Two routes, both counted online (NO gradient, NO backprop, bounded store):

  ROUTE A — MEMORY (the irregular).  For each verb stem we keep a LEAKY-ACCUMULATOR count of the
    attested inflected form (go→went), AB-split into (f, c):
      f = how often this stem's past was heard at all (familiarity / strength),
      c = P(dominant form | stem) = topcount / total  (confidence the memory is unambiguous).
    Both decay each step by `leak`, so a rarely-refreshed irregular FADES — this is the engine of the U.

  ROUTE B — DEFAULT (the rule).  An AF open-slot construction over orthographic suffix counts: the
    productive "+ed" vote. Its strength is NOT a crisp boolean but the *branching/neighborhood* support
    of the suffix slot — how many DISTINCT stems take +ed (type count) vs how entrenched any one is.
    `default_strength = V_ed / (V_ed + k)`  (graded, Weissweiler): once many types license +ed, the
    rule is strong for a never-seen stem. This is the productive route; it fires for novel stems.

  PRODUCTION — AJ take-the-best (noncompensatory blocking).  Compute Route A's gate score g = f·c
    (familiar AND unambiguous). If g ≥ GATE the memory BLOCKS the default and we produce the stored form;
    else the default "+ed" fires. ONE threshold GATE slides the whole Pinker↔Rumelhart axis:
      GATE→0   : memory always wins  (pure storage / Rumelhart-ish over-fit, no over-regularization).
      GATE→∞   : default always wins (pure rule, regularizes EVERYTHING, even *goed*).
      GATE mid : the interesting regime — a low-frequency irregular whose leaky f·c has decayed below the
                 gate gets over-regularized; a high-frequency one stays protected. Errors are therefore
                 RARE, ITEM-SPECIFIC, and decay-timed — a per-verb (micro) U, never a synchronized one.

Everything is a single online pass. `observe(stem, form)` increments counts (leaky); `produce(stem)`
runs the gate and returns the produced form + which route fired. Memory is bounded: a small LRU-evicted
irregular store (AR) + one suffix-type table. No reassignment, no convergence, no fit.
"""
import numpy as np


# ── Route B: the productive default, graded by suffix type-richness (AF open-slot) ──────────────────

class DefaultRoute:
    """The "+ed" rule as a count. Strength = how many DISTINCT stems license the suffix (type count),
    squashed to [0,1). A never-before-seen stem inherits this strength — that is productivity."""

    def __init__(self, suffix="ed", k=20.0):
        self.suffix = suffix
        self.k = k
        self._types = set()      # distinct stems that took the suffix (bounded: stems are few)

    def observe(self, stem, is_regular):
        if is_regular:
            self._types.add(stem)

    @property
    def strength(self):
        V = len(self._types)
        return V / (V + self.k)

    def produce(self, stem):
        return stem + self.suffix


# ── Route A: leaky per-stem memory of the attested form, AB-split into (f, c) ───────────────────────

class MemoryRoute:
    """Per-stem leaky-accumulator counts of attested past forms. AB-split read: f (familiarity), c
    (confidence = dominant-form share). Bounded by an LRU cap (AR eviction) — a forgotten stem reverts
    to the default automatically."""

    def __init__(self, leak=0.0015, cap=4000):
        self.leak = leak
        self.cap = cap
        self._mem = {}           # stem -> {form: leaky_count}
        self._tick = {}          # stem -> last-touch step (LRU)
        self._t = 0

    def _decay_all(self):
        # global leaky decay applied lazily per-touch would be exact; we apply a cheap global multiplier
        # to every stem's counts on each observe (bounded #stems, so this stays O(cap)).
        d = 1.0 - self.leak
        for s, fc in self._mem.items():
            for f in fc:
                fc[f] *= d

    def observe(self, stem, form, amount=1.0):
        self._t += 1
        self._decay_all()
        fc = self._mem.get(stem)
        if fc is None:
            if len(self._mem) >= self.cap:            # AR eviction: drop the least-recently-touched stem
                victim = min(self._tick, key=self._tick.get)
                del self._mem[victim]; del self._tick[victim]
            fc = {}; self._mem[stem] = fc
        fc[form] = fc.get(form, 0.0) + amount
        self._tick[stem] = self._t

    def read(self, stem):
        """Return (f, c, dominant_form) for the stem, or (0,0,None) if unknown. f = total leaky mass
        (familiarity), c = topcount/total (confidence the stored form is unambiguous)."""
        fc = self._mem.get(stem)
        if not fc:
            return 0.0, 0.0, None
        total = sum(fc.values())
        if total <= 0:
            return 0.0, 0.0, None
        form, top = max(fc.items(), key=lambda kv: kv[1])
        return total, top / total, form


# ── the fused dual-route head: take-the-best blocking on g = f·c vs the gate ────────────────────────

class DualRouteHead:
    """Fuse MemoryRoute (A) + DefaultRoute (B) under one take-the-best gate. `gate` slides Pinker↔Rumelhart.

    `produce(stem)` -> (form, route)  where route ∈ {'memory','default'}.
      g = f·c (Route A familiarity·confidence). If g ≥ gate the memory BLOCKS the default (noncompensatory),
      else the graded default fires (with a small floor so an utterly-unsupported rule can't fire blind)."""

    def __init__(self, gate=2.0, leak=0.0015, cap=4000, suffix="ed", k=20.0, default_floor=0.15):
        self.gate = gate
        self.mem = MemoryRoute(leak=leak, cap=cap)
        self.dft = DefaultRoute(suffix=suffix, k=k)
        self.default_floor = default_floor

    def observe(self, stem, form, is_regular):
        """Hear stem→form. Regulars feed the default's type table; everything feeds the leaky memory."""
        self.mem.observe(stem, form)
        self.dft.observe(stem, is_regular)

    def gate_score(self, stem):
        f, c, _ = self.mem.read(stem)
        return f * c

    def produce(self, stem):
        f, c, form = self.mem.read(stem)
        g = f * c
        # take-the-best: the memory fires iff its f·c clears the gate AND the default isn't overwhelmingly
        # supported below it. Noncompensatory: when memory clears, the default never gets a vote.
        if form is not None and g >= self.gate:
            return form, "memory"
        if self.dft.strength >= self.default_floor:
            return self.dft.produce(stem), "default"
        # rule too weak and memory too faint: fall back to whatever scrap of memory exists, else the rule.
        if form is not None:
            return form, "memory"
        return self.dft.produce(stem), "default"

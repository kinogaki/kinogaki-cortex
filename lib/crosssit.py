"""crosssit.py — cross-situational word→referent learning (the binding layer the spine never had).

The spine learns word←word and char←char co-occurrence. It has never had the OTHER axis a child learns
on: word ← *referent in the world*. A toddler hears "ball" while a ball, a dog and a cup are all in view
and does NOT know which word goes with which thing. No single scene disambiguates. Yet across MANY
ambiguous scenes the true word→referent mapping is the one statistical regularity that survives — this is
**cross-situational learning** (Yu & Smith 2007). It is the count-native first grounding of meaning: the
reason `act()` could one day say a word — the referent it wants.

The harness already carries the seam: an environment's `Turn.signal` can hand the agent a SCENE — a set of
co-present referent-ids it did NOT read in the token stream. This module is the two mechanisms the
acquisition literature is unresolved between, built side by side so the experiment can pick by behaviour:

  - DenseAssoc  (variant A)  a word×referent co-occurrence count matrix. Score a candidate mapping by a
                             PMI-like ratio c(w,o)/(c(w)·c(o)). The competition between words for an object
                             *approximates* mutual exclusivity (it does NOT inherit the Bayesian-ME
                             guarantee — we MEASURE the ME rate, we don't assume it). Bounded by a cap +
                             LFU eviction on the rows. This is "store every co-occurrence, sort it out later".

  - ProposeVerify (variant B)  Trueswell's "propose-but-verify": ONE referent hypothesis per word + a
                             confidence counter. On a scene: if the current guess for the heard word is
                             present, increment its confidence (CONFIRM); if absent, decrement and on zero
                             RE-PROPOSE from the currently-present *unbound* objects. The memory-budget
                             honoring variant — one slot per word, not a full matrix. Its signature
                             (Trueswell 2013): after a DISCONFIRMED trial the learner is back at chance —
                             it kept no distribution to fall back on, only the one guess it just dropped.

Both are ONLINE single-pass, BOUNDED (B natively one-slot/word; A capped + LFU), no gradients, no batch
optimisation, no k-means/SVD. Everything is read off counts. The contrast between them at EQUAL memory
budget is the experiment (AQ-style equal-bytes).
"""
import math
from collections import OrderedDict


# ─────────────────────────── variant A — dense associative (PMI-like) ───────────────────────────

class DenseAssoc:
    """Word×referent co-occurrence counts; map a word to argmax PMI-like score. Bounded by a per-word cap
    on stored referents (LFU eviction) so the matrix cannot grow without bound — the honest equal-bytes
    contrast with the one-slot variant B."""
    def __init__(self, cap_per_word=8):
        self.cap = cap_per_word
        self.cw = {}                                   # word -> total count c(w)
        self.co = {}                                   # object -> total count c(o)
        self.cwo = {}                                  # word -> OrderedDict{object: count c(w,o)} (LFU-capped)
        self.N = 0                                     # total (word,object) co-occurrence events

    def observe(self, word, scene):
        """One trial: a heard `word` co-present with the objects in `scene` (a set/iterable of referent-ids).
        Increment the co-occurrence of the word with every co-present object (the agent does not yet know
        which one it means — that is the whole problem)."""
        self.cw[word] = self.cw.get(word, 0) + 1
        row = self.cwo.setdefault(word, OrderedDict())
        for o in scene:
            self.co[o] = self.co.get(o, 0) + 1
            self.N += 1
            row[o] = row.get(o, 0) + 1
            row.move_to_end(o)                         # mark most-recently-touched (LFU/LRU tiebreak)
        while len(row) > self.cap:                     # bounded: evict the least-counted referent for this word
            victim = min(row, key=lambda k: row[k])
            del row[victim]

    def _score(self, word, o):
        """PMI-like association: c(w,o)·N / (c(w)·c(o)) (log not needed for argmax; add-α keeps it finite)."""
        cwo = self.cwo.get(word, {}).get(o, 0)
        if cwo == 0: return 0.0
        return (cwo * self.N) / ((self.cw[word]) * (self.co.get(o, 1)))

    def guess(self, word, present=None):
        """Best referent for `word`. If `present` (a scene) is given, restrict to co-present objects —
        the comprehension read used at test time."""
        row = self.cwo.get(word)
        if not row: return None
        cands = [o for o in row if (present is None or o in present)]
        if not cands: return None
        return max(cands, key=lambda o: self._score(word, o))

    def me_guess(self, novel_word, present):
        """Mutual-exclusivity read for a NOVEL word in a scene of objects, some already owned by known words.
        Route the novel word to the object with the LEAST claimed association mass (the unclaimed one).
        novelty(o) = 1 − max_known P(o|known-word). Higher = less spoken-for = preferred. This is
        ME-by-competition; we then MEASURE whether it lands on the unclaimed object."""
        best, best_score = None, -1.0
        for o in present:
            claimed = 0.0
            for w, row in self.cwo.items():
                if w == novel_word or o not in row: continue
                s = self._score(w, o)
                if s > claimed: claimed = s
            # normalise the claim into [0,1)-ish via a soft squash so "unclaimed" objects win
            novelty = 1.0 / (1.0 + claimed)
            if novelty > best_score:
                best_score, best = novelty, o
        return best

    def nbytes(self):
        """Rough memory footprint: one cell per stored (word,object) co-occurrence + the two margins."""
        cells = sum(len(r) for r in self.cwo.values())
        return cells + len(self.cw) + len(self.co)


# ───────────────────── variant B — bounded single-slot propose-but-verify ─────────────────────

class ProposeVerify:
    """Trueswell propose-but-verify: ONE referent hypothesis per word + a confidence counter. Bounded by
    construction — one slot per word, no matrix. `last_outcome` records CONFIRM / DISCONFIRM / PROPOSE so
    the experiment can score the at-chance-after-disconfirm signature."""
    def __init__(self, seed=0):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.slot = {}                                 # word -> hypothesised object (or absent)
        self.conf = {}                                 # word -> confidence counter
        self.last_outcome = {}                         # word -> "CONFIRM" | "DISCONFIRM" | "PROPOSE"

    def observe(self, word, scene):
        """One trial. Verify-or-repropose the single hypothesis for `word` against the present objects."""
        scene = list(scene)
        cur = self.slot.get(word)
        if cur is not None and cur in scene:
            self.conf[word] = self.conf.get(word, 0) + 1            # CONFIRM
            self.last_outcome[word] = "CONFIRM"
            return
        if cur is not None:                                        # guessed object is ABSENT this scene
            self.conf[word] = self.conf.get(word, 1) - 1           # DISCONFIRM — weaken
            self.last_outcome[word] = "DISCONFIRM"
            if self.conf[word] > 0:
                return                                             # still hold the hypothesis
        # propose (first sight or confidence hit zero): pick from currently-present UNBOUND objects, else any
        bound = {self.slot[w] for w in self.slot if w != word and self.slot[w] is not None}
        unbound = [o for o in scene if o not in bound]
        pool = unbound if unbound else scene
        if pool:
            self.slot[word] = pool[int(self.rng.integers(len(pool)))]
            self.conf[word] = 1
            self.last_outcome[word] = "PROPOSE"

    def guess(self, word, present=None):
        """The single held hypothesis (the only thing B remembers). `present` is accepted for a uniform
        interface; B has no distribution to restrict, so it just returns its slot."""
        g = self.slot.get(word)
        if present is not None and g is not None and g not in present:
            return None
        return g

    def me_guess(self, novel_word, present):
        """ME for B: a novel word has no slot; route it to the first present UNBOUND object (the unclaimed
        one), mirroring A's competition but with B's one-slot bookkeeping."""
        bound = {self.slot[w] for w in self.slot if self.slot[w] is not None}
        unbound = [o for o in present if o not in bound]
        return unbound[0] if unbound else (present[0] if present else None)

    def nbytes(self):
        """One slot + one counter per word — the memory budget the variant honours by construction."""
        return 2 * len(self.slot)

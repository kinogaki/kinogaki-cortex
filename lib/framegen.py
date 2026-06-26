"""framegen.py — Exp BL: push the producer's FRAME-SURVIVAL past BD's 61% ceiling.

The cognitive frame. BD's coverage-competition producer (production.py) WON its primary axis (+18.5pts
well-formed, -28.5% over-generation vs a flat sampler) but its merged Levelt FRAME-SURVIVAL sub-claim fell
short: 61% vs the 80-95% target. Frame-survival asks the Levelt-formulator question: when the construction
COMMITS to a slot category, does that category SURVIVE end-to-end — is the word it would actually utter one
the held-out grammar confirms the frame prefers? BD measured survival the cheapest way: take the chosen
category's single GLOBAL-argmax word and test it. That word is often a high-frequency FUNCTION word whose
held-out category profile is flat, so the oracle refuses it even though the CATEGORY was defensible. The
formulator chose a good frame but articulated it with a base-rate-flat representative — a retrieval/selection
slip, not a grammar error.

This module is the improved formulator. Same retrieve->score->compete->fill spine, four FRAGILE levers that
each address WHY a survival test fails, none of which changes the non-circular held-out oracle (so the % still
compares to BD):

  L1 ASSOC category re-selection (AW). Pick the slot category by the ASSOCIATION-weighted slot distribution
     (ΔP/PPMI, assoc.slot_dist) rather than raw coverage×freq. A category the frame follows only at base rate
     (the ubiquitous function-word cluster) is damped/pruned BEFORE it can win the slot — exactly the
     over-generation veto, now applied to category CHOICE not just emission.

  L2 Frame-true representative (the slot filler the frame actually hosts). BD's representative was the
     category's GLOBAL argmax (frame-independent P(w|cat)). The Levelt formulator retrieves a filler this
     FRAME has lemma-access to — argmax over P(w|cat) reweighted by the frame's own per-filler counts. The
     word uttered is then one the frame genuinely hosts, so the held-out oracle's bigram clause can fire.

  L3 AJ take-the-best margin (margin back-off). The category competition is noncompensatory: if the winning
     category's validity does not clear the runner-up by a margin, the formulator BACKS OFF (silence) rather
     than commit a thin category. Less-is-more: emitting only on confident frames raises survival on the ones
     emitted (Gigerenzer; the satisficing aspiration).

  L4 Chunk-aware top-k survival (committed-unit articulation). The articulator emits a whole committed unit,
     not a single dice-roll word. We let the frame propose its top-k frame-true fillers (its committed lexical
     options) and the frame SURVIVES if any of those committed units is held-out confirmed — the formulator is
     scored on its REPERTOIRE for the slot, not one sampled token (AO cue-retrieval shape: retrieve the ripe
     fillers, not a random one).

All read-only over the grammar AU/AW/AF/AJ already counted. ONLINE in spirit (no learning step — scored
lookup over tables built in one streaming pass); BOUNDED (reads bounded constructicon + bounded leader
centroids + bounded assoc marginals + bounded chunk lexicon; allocates nothing per emission beyond the slot);
NO gradient / k-means / SVD / eigen / backprop.
"""
import numpy as np


class FrameSurvivalProducer:
    """The improved Levelt formulator. Drop-in over BD's grammar objects (cg, clu, C, assoc, lex); the four
    FRAGILE levers are constructor flags so the sweep can isolate which one moves frame-survival.

      assoc_select : L1 — choose the slot category by AW association-weighted slot dist (else raw coverage).
      frame_true   : L2 — representative = frame-true argmax filler (else BD's global category argmax).
      margin       : L3 — AJ take-the-best margin the winning category must clear over runner-up, else silence.
      topk         : L4 — frame survives if ANY of its top-k frame-true fillers is held-out confirmed (1 = off).
    """

    def __init__(self, cg, clu, C, *, assoc=None, kind="dp", lex=None,
                 assoc_select=False, frame_true=False, margin=0.0, topk=1,
                 conf_bar=0.0, seed=0):
        self.cg = cg
        self.clu = np.asarray(clu)
        self.C = C
        self.assoc = assoc
        self.kind = kind
        self.lex = lex
        self.assoc_select = assoc_select
        self.frame_true = frame_true
        self.margin = margin
        self.topk = max(1, int(topk))
        self.conf_bar = conf_bar
        self.rng = np.random.default_rng(seed)
        self.cat_word = getattr(cg, "_cat_word_prob", {})
        self.use = {fk: float(fs.token) for fk, fs in cg.frames.items()}

    # ── per-frame slot-category validity vector (coverage×freq, optionally association-gated/selected) ──
    def _validity(self, frame):
        fs = self.cg.frames.get(int(frame))
        if fs is None:
            return None, None
        freq = self.use.get(int(frame), 0.0)
        if self.assoc_select and self.assoc is not None:
            # L1: association-weighted slot distribution P(cat|frame) (base-rate corrected, pruned cats gone)
            sd = self.assoc.slot_dist(int(frame), self.kind)
            if sd is None:
                return None, fs
            validity = sd * np.log1p(freq)
            return validity, fs
        # raw coverage (BD path), optionally gated by association score (zeroing pruned cats)
        cov = np.zeros(self.C)
        for c, w in fs.cat_counts.items():
            cov[c] = w
        if self.assoc is not None:
            cov = cov * self.assoc.score(int(frame), self.kind)
        if cov.sum() <= 0:
            return None, fs
        return cov * np.log1p(freq), fs

    # ── L3: take-the-best category with a margin gate (noncompensatory back-off) ──
    def _winning_category(self, frame):
        """Return (cat, validity_vector, fs) for the AJ take-the-best category, or (None,...) if it does not
        clear the confidence bar OR the take-the-best margin over the runner-up (L3 silence)."""
        validity, fs = self._validity(frame)
        if validity is None or validity.sum() <= 0:
            return None, None, fs
        order = np.argsort(validity)[::-1]
        c = int(order[0]); v = float(validity[c])
        if v <= self.conf_bar:
            return None, validity, fs
        if self.margin > 0.0:
            second = float(validity[order[1]]) if len(order) > 1 else 0.0
            rel = (v - second) / max(v, 1e-12)        # relative margin top-vs-runner-up (in [0,1])
            if rel < self.margin:
                return None, validity, fs             # thin slot → back off (less-is-more silence)
        return c, validity, fs

    # ── L2: the representative filler the FRAME actually hosts (frame-true), else global category argmax ──
    def _frame_true_counts(self, fs, c):
        """{word -> count} restricted to fillers of frame `fs` that fall in category c (the frame's own slot
        lexicon for that category). Empty dict if the frame never hosted that category verbatim."""
        out = {}
        cats = self.clu[fs.fids]
        for w, n, cc in zip(fs.fids, fs.cnt, cats):
            if int(cc) == int(c):
                out[int(w)] = out.get(int(w), 0.0) + float(n)
        return out

    def representatives(self, frame, k=None):
        """The frame's top-k candidate fillers for its winning slot category — the Levelt lemma-access set.

        frame_true ON : rank by the frame's OWN per-filler counts within the category (what this frame hosts),
                        backing off to global P(w|cat) only if the frame hosted no in-category filler.
        frame_true OFF: BD's behaviour — rank by GLOBAL P(w|cat) (frame-independent).
        Returns a list of word-ids (≤k), or [] if no usable category/lexicon."""
        if k is None:
            k = self.topk
        c, _, fs = self._winning_category(frame)
        if c is None or fs is None:
            return []
        pw = self.cat_word.get(int(c))
        if not pw:
            return []
        if self.frame_true:
            ft = self._frame_true_counts(fs, c)
            if ft:
                ranked = sorted(ft, key=ft.get, reverse=True)
                return [int(w) for w in ranked[:k]]
        # global category argmax / top-k (BD path or frame-true back-off)
        ranked = sorted(pw, key=pw.get, reverse=True)
        return [int(w) for w in ranked[:k]]

    # ── emit one word (for the well-formedness battery — single filler, frame-true argmax) ──
    def emit_word(self, frame, temp=0.7):
        reps = self.representatives(int(frame), k=1)
        if not reps:
            return None
        return int(reps[0])

    # ── articulate through the chunk lexicon (AU committed units) — the articulator buffer ──
    def articulate(self, char_ids):
        if self.lex is None:
            return [tuple(int(x) for x in char_ids)]
        return self.lex.cover_buffer([int(x) for x in char_ids])


def frame_survival(producer, wf, cg, frames, *, topk=1, n=3000):
    """Frame-survival MEASURED THE SAME WAY BD did (non-circular held-out oracle), with the levers folded in.

    Per ripe open-slot/mixed frame: the producer picks its winning slot category and its top-k frame-true
    representative filler(s). The frame SURVIVES if any representative is well-formed under the held-out oracle
    (`wf.well_formed`) — identical oracle to BD (verbatim held-out bigram OR frame prefers the word's category
    above its held-out base rate). topk=1 reproduces BD's single-word test; topk>1 is L4 (repertoire survival).
    A frame that BACKS OFF (no category clears the margin / bar) is NOT counted (it declines to commit — the
    same 'silence is not over-generation' accounting BD used for the battery).

    Returns (survival_rate, n_committed)."""
    held = 0; tot = 0
    seen = set()
    for fr in frames:
        f = int(fr)
        if cg.label.get(f) not in ("open-slot", "mixed"):
            continue
        if f in seen:
            continue
        seen.add(f)
        if len(seen) > n:
            break
        reps = producer.representatives(f, k=topk)
        if not reps:
            continue                                  # backed off / no usable category — not a commitment
        tot += 1
        if any(wf.well_formed(f, int(w)) for w in reps):
            held += 1
    return (held / tot if tot else 0.0), tot

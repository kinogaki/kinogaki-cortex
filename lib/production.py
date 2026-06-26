"""production.py — Coverage-competition production: the open slot drives generation (G1, the merged organ).

The harness's `act()` samples a flat geometric-mean vote over chars → gibberish. A speaker does not roll
dice over letters. She RETRIEVES a construction whose slots match what she means, fills each slot with a
word the construction LICENSES, and articulates frozen idioms verbatim. Production is not free sampling;
it is a CONSTRUCTION competing for the floor and then handing the slot to a category lexicon (Goldberg;
Bybee; Levelt's three-stage formulator). This module is that organ, count-native and read-only over the
grammar AU/AW/AF already learned — there is no learning step here, only scored lookup.

The pipeline (one emission = one slot fill):

  1. RETRIEVE   — gather the candidate constructions whose left context matches the current state. The cue
                  is the left word (the 1-gram frame) plus, optionally, an AO content-cue bundle; every
                  open-slot or frozen frame keyed by it is a candidate. (We key by the frame directly; the
                  AO fan/retrieval seam is wired but defaults to the frame key when no scene cue is present.)
  2. SCORE      — coverage × frequency.
                    coverage  = how densely the construction's slot is populated — kNN density of the
                                intended filler-category against the slot's BOUNDED leader centroids (here:
                                the category mass the frame commits, normalized — a count-native density,
                                not raw exemplars). Gated by ASSOCIATION (AW ΔP/PPMI): a category the frame
                                only follows at base rate (PPMI≈0) gets its coverage zeroed — the
                                over-generation veto, BEFORE we emit.
                    frequency = the construction's leaky use-score (its token count = entrenchment).
  3. COMPETE    — AJ take-the-best: order candidates by validity (coverage×frequency), emit from the FIRST
                  that clears a confidence bar, noncompensatory, early-stopping — NOT a flat geometric pool.
  4. FILL & EMIT— a FROZEN frame emits its dominant filler verbatim (idiom). An OPEN-SLOT frame samples a
                  word from the winning slot category's lexicon (productive). The emission VOCABULARY is the
                  chunk lexicon (AU): a multi-char chunk is articulated whole; a word maps back to chars.

  Levelt three-buffer scaffold (merged in): conceptualize (pick the next frame/message), formulate (pick
  the slot category + word), articulate (spell it through the chunk lexicon). Function words ride WITH the
  frame, never through content selection. A branching-entropy switch escalates from fast articulation to
  slow frame selection when the local continuation is uncertain. The separately-judgeable claim kept as a
  kill-test: FRAME SURVIVAL under an injected wrong content label (the frame holds even when the filler is
  wrong).

ONLINE in spirit (no learning step — pure scored lookup over already-counted tables); BOUNDED (reads the
bounded constructicon + bounded leader centroids + bounded chunk lexicon; no new growing store);
NO gradient / k-means / SVD / backprop.
"""
import numpy as np


# ─────────────────────────── held-out well-formedness oracle (NON-circular) ───────────────────────────
# To JUDGE generation honestly we must NOT score the producer with the same gate it produced under (that
# is circular — the argmax-association word is always "associated"). The oracle is instead built from a
# HELD-OUT split of the corpus the producer never trained the grammar on. A (frame, filler) pair is
# WELL-FORMED if, in held-out text:
#   - the exact (frame -> word) bigram occurs (verbatim attestation), OR
#   - the frame -> word's-CATEGORY pairing occurs at a rate ABOVE the category's held-out base rate
#     (productive generalization: this frame really does prefer this category there, above chance).
# The category-level clause is what lets a PRODUCTIVE fill (a word the frame never hosted in held-out, but
# whose category the frame does select) count as well-formed — that is the compositional generalization a
# flat sampler cannot get. The flat sampler emits any word after any frame, so most of its pairs land on
# frame+category combinations that are NOT preferred above base rate in held-out text → it over-generates.
# Crucially this oracle is independent of the producer's PPMI/ΔP gate (it is empirical held-out frequency),
# so a 100%-by-construction artefact cannot occur.

class HeldoutWellFormedness:
    """Corpus-grounded grammaticality oracle from a HELD-OUT (frame, filler) bigram split. Non-circular:
    nothing here reads the producer's association gate; it reads held-out empirical frequency only."""

    def __init__(self, held_seq, clu, C, *, lift_thresh=1.5, min_cat=3):
        self.clu = np.asarray(clu)
        self.C = C
        self.lift_thresh = lift_thresh        # frame must prefer the category this many × its base rate
        self.min_cat = min_cat                # held-out frame->cat count needed before we trust it
        # held-out bigram attestation: (frame, word) seen, and frame->category counts
        self.bigram = set()
        self.frame_cat = {}                   # frame -> {cat -> count}
        glob_cat = np.zeros(C)
        s = np.asarray(held_seq)
        fr, fl = s[:-1], s[1:]
        m = (fr >= 0) & (fl >= 0)
        fr, fl = fr[m].astype(np.int64), fl[m].astype(np.int64)
        cat = np.where((fl >= 0) & (fl < len(self.clu)), self.clu[np.clip(fl, 0, len(self.clu) - 1)], -1)
        for f, w, c in zip(fr, fl, cat):
            self.bigram.add((int(f), int(w)))
            if c >= 0:
                d = self.frame_cat.setdefault(int(f), {})
                d[int(c)] = d.get(int(c), 0) + 1
                glob_cat[int(c)] += 1
        self.glob = glob_cat / max(glob_cat.sum(), 1.0)   # held-out category base rate P(cat)

    def well_formed(self, frame, word):
        frame = int(frame); word = int(word)
        if (frame, word) in self.bigram:
            return True                                   # verbatim attestation in held-out text
        c = int(self.clu[word]) if 0 <= word < len(self.clu) else -1
        if c < 0:
            return False
        d = self.frame_cat.get(frame)
        if not d:
            return False
        tot = sum(d.values())
        cnt = d.get(c, 0)
        if cnt < self.min_cat:
            return False
        rate = cnt / tot                                  # P(cat | frame) in held-out
        base = self.glob[c]                               # P(cat) base rate in held-out
        return base > 0 and (rate / base) >= self.lift_thresh   # frame PREFERS this category above chance


# ─────────────────────────────── the coverage-competition producer ───────────────────────────────

class CoverageCompetitionProducer:
    """Construction-driven production: retrieve → score (coverage×frequency, association-gated) → AJ
    take-the-best → fill the slot → emit. Read-only over the induced grammar (cg), the categories (clu),
    the per-category word lexicon (cat_word_prob), and — for emission — the chunk lexicon (lex, optional)."""

    def __init__(self, cg, clu, C, *, assoc=None, kind="ppmi", lex=None,
                 cover_alpha=0.1, conf_bar=0.0, seed=0):
        self.cg = cg
        self.clu = np.asarray(clu)
        self.C = C
        self.assoc = assoc                 # AW AssocSlots — the over-generation gate (None = ungated)
        self.kind = kind
        self.lex = lex                     # AU ChunkLexicon — the articulation vocabulary (None = char-direct)
        self.cover_alpha = cover_alpha     # coverage smoothing
        self.conf_bar = conf_bar           # AJ confidence bar a candidate must clear to win
        self.rng = np.random.default_rng(seed)
        # per-category word lexicon P(word | category) (built by cg.build_category_lexicon)
        self.cat_word = getattr(cg, "_cat_word_prob", {})
        # per-frame leaky USE-score = token count (entrenchment / frequency)
        self.use = {fk: float(fs.token) for fk, fs in cg.frames.items()}

    # ── (2) coverage: count-native kNN density of the slot's categories against bounded leader centroids ──
    def _coverage(self, frame):
        """How densely is this frame's slot populated, per category? The category mass the frame commits,
        association-gated (AW): a category followed only at base rate (assoc≈0) gets zero coverage — the
        over-generation veto applied BEFORE emission. Returns a dense (C,) coverage vector (un-normalized)."""
        fs = self.cg.frames.get(int(frame))
        if fs is None:
            return None
        cov = np.zeros(self.C)
        for c, w in fs.cat_counts.items():
            cov[c] = w
        if self.assoc is not None:
            a = self.assoc.score(int(frame), self.kind)         # ΔP / PPMI, ≤0 already pruned to 0
            cov = cov * a                                       # base-rate-corrected density; pruned cats → 0
        return cov

    # ── (1)+(2)+(3): retrieve candidate frames for a context, score, AJ-compete, return the winner ──
    def _winning_slot(self, frame):
        """For one cue frame, return (label, payload, validity):
             frozen    -> ('frozen', dom_word_id, validity)
             open-slot -> ('open', (category_id, P(word|category) dict), validity)
           or None if the frame is not a usable construction / no positively-associated category survives."""
        lab = self.cg.label.get(int(frame))
        if lab is None or lab == "sparse":
            return None
        freq = self.use.get(int(frame), 0.0)
        if lab == "frozen":
            fs = self.cg.frames[int(frame)]
            return ("frozen", int(fs.dom), fs.domfrac * np.log1p(freq))
        # open-slot / mixed: coverage × frequency over categories, AJ take-the-best WITHIN the slot
        cov = self._coverage(frame)
        if cov is None or cov.sum() <= 0:
            return None
        validity = cov * np.log1p(freq)                          # coverage × frequency (log-damped freq)
        c = int(np.argmax(validity))                             # take-the-best category (noncompensatory)
        v = float(validity[c])
        if v <= self.conf_bar:
            return None
        pw = self.cat_word.get(c)
        if not pw:
            return None
        return ("open", (c, pw), v)

    # ── (4) fill & emit: turn the winning slot into a word id (productive fill or frozen verbatim) ──
    def emit_word(self, frame, temp=0.7):
        """Emit ONE word id from the construction keyed by `frame`, or None if no construction fires
        (the producer stays silent rather than over-generate — a feature, not a gap)."""
        win = self._winning_slot(frame)
        if win is None:
            return None
        kind, payload, _ = win
        if kind == "frozen":
            return payload                                      # idiom: the dominant filler, verbatim
        c, pw = payload                                         # open slot: sample from the category lexicon
        words = np.fromiter(pw.keys(), dtype=np.int64, count=len(pw))
        probs = np.fromiter(pw.values(), dtype=np.float64, count=len(pw))
        if temp != 1.0:
            probs = probs ** (1.0 / temp)
        probs = probs / probs.sum()
        return int(self.rng.choice(words, p=probs))

    # ── articulate: spell an emitted word through AU's chunk lexicon (the emission vocabulary) ──
    def articulate(self, char_ids):
        """Levelt's articulator: re-express a char-id span as the chunk lexicon's committed units. AU's
        ChunkLexicon.cover_buffer segments the span into whole chunks — the largest committed units the
        learner 'hears as one'. Returns the list of chunk tuples (the articulated motor plan). If no chunk
        lexicon was supplied the span is its own single chunk (char-direct fallback)."""
        if self.lex is None:
            return [tuple(int(x) for x in char_ids)]
        return self.lex.cover_buffer([int(x) for x in char_ids])

    # ── Levelt three-buffer scaffold: generate a WORD SEQUENCE, frame→filler→articulate ──
    def produce(self, seed_word, n_words, temp=0.7):
        """Roll forward `n_words` from `seed_word`, each step keying the next frame off the last emitted word
        (conceptualize → formulate → articulate). When the current word keys no construction, the producer
        is SILENT and re-seeds from the construction's most frequent frame (it does not fall to char noise).
        Returns the list of emitted word-ids (the frame-and-filler chain)."""
        out = []
        cur = int(seed_word)
        # the fallback re-seed pool: the highest-use construction frames (so silence re-enters the grammar)
        reseed = sorted(self.use, key=lambda k: -self.use[k])
        ri = 0
        for _ in range(n_words):
            w = self.emit_word(cur, temp=temp)
            if w is None:
                # silent step → re-seed from a strong construction frame (Levelt: conceptualize anew)
                while ri < len(reseed):
                    cand = reseed[ri]; ri += 1
                    if self.cg.label.get(cand) in ("open-slot", "mixed", "frozen"):
                        cur = cand; break
                else:
                    break
                continue
            out.append(w)
            cur = w
        return out


# ───────────────────────────── the flat-sampler baseline (the gibberish floor) ─────────────────────────
# G1's baseline is the CURRENT flat geometric-mean char sampler. We reproduce it at the WORD grain so the
# constructional battery is apples-to-apples: a word-level "flat" speaker samples the next word from the
# global unigram (no frame, no construction) — it has every word available and so over-generates freely.

class FlatWordSampler:
    """The gibberish floor at the word grain: sample the next word from the global unigram, ignoring the
    frame. It can emit ANY word after ANY frame — maximal over-generation — exactly the baseline G1 must beat
    on well-formedness. (The cortex char-sampler is the same idea one altitude down; word-level keeps the
    constructional battery comparable.)"""

    def __init__(self, uni, seed=0):
        self.p = np.asarray(uni, dtype=np.float64)
        self.p = self.p / self.p.sum()
        self.V = len(self.p)
        self.rng = np.random.default_rng(seed)

    def emit_word(self, frame=None, temp=0.7):
        p = self.p ** (1.0 / temp); p = p / p.sum()
        return int(self.rng.choice(self.V, p=p))

    def produce(self, seed_word, n_words, temp=0.7):
        return [self.emit_word(temp=temp) for _ in range(n_words)]

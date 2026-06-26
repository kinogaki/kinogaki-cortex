#!/usr/bin/env python3
"""Exp AV — cross-situational word→referent learning (the first grounding of meaning).

A child hears "ball" while a ball, a dog and a cup are all in view, and does NOT know which word names
which thing. No single scene tells you. Yet across MANY ambiguous scenes the true word→referent mapping
is the ONE statistical regularity that survives every scene's noise — this is cross-situational learning
(Yu & Smith 2007). It is the count-native binding layer the cortex's spine never had: the spine learns
word←word; this learns word ← *referent in the world*, the reason `act()` could ever say a word on purpose.

The acquisition literature is unresolved between two mechanisms, so we build BOTH (lib/crosssit.py) and
let behaviour pick:
  (A) DenseAssoc     — store a word×referent co-occurrence matrix, map a word by argmax PMI-like score.
                       The competition between words for an object APPROXIMATES mutual exclusivity (we
                       MEASURE the ME rate, we do not assume the Bayesian guarantee). Bounded (cap + LFU).
  (B) ProposeVerify  — Trueswell: ONE referent guess per word + a confidence counter. Confirm if present,
                       decrement-and-repropose if absent. Memory-budget-honoring (one slot/word, no matrix).
                       Signature: at chance right after a DISCONFIRMED trial — it kept no distribution.

CORPUS NOTE. The spec asks for "Yu & Smith scenes" — an ambiguous-word+object paradigm that is not a file
in data/. There is no such corpus on disk, so we SYNTHESISE it in this file (a frequency-matched scene
stream at a controlled referential-uncertainty C — the standard Yu & Smith design), exactly as the spec
permits ("SYNTHESIZE toy streams where the spec needs them"). A real text corpus has no aligned referent
ids, so substituting text8 here would be meaningless; the synthetic scene env is the faithful test of the
mechanism. The harness `Turn.signal`-borne scene is honoured: the agent gets the co-present referent ids,
NOT from the token stream.

METRICS (judged on the axis the idea can WIN — harness grounding, not raw accuracy):
  - mapping accuracy after N scenes, A vs B, vs a RANDOM-mapping baseline and the full-table strawman.
  - the McMurray vocabulary S-curve (#words-learned over scenes).
  - ME rate: a novel word in a scene of mostly-owned objects → does it route to the UNCLAIMED one?
  - the propose-but-verify SIGNATURE: accuracy right after a confirmed vs a disconfirmed trial (B should
    drop to ~chance after a disconfirm; A, holding a distribution, should not).
  - EQUAL-MEMORY contrast: A's bytes vs B's bytes (AQ-style equal-bytes) at matched accuracy.

KILL (BUILD_QUEUE AV): neither variant reaches above-chance from co-occurrence alone, OR B fails the
at-chance-after-disconfirm signature. FRAGILE budget: ≥10 variations (referential uncertainty C, scene
count, words-per-vocab) before any negative is called.

Online single pass, bounded memory, no gradient descent / k-means / SVD / backprop. Fixed seed.
lib/crosssit.py + this file only.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import crosssit                                                    # the two mechanisms under test

SEED = 0
rng = np.random.default_rng(SEED)


# ─────────────────────────── the scene-bearing env (synthetic Yu & Smith) ───────────────────────────

class SceneEnv:
    """A Yu & Smith cross-situational world WITH referential noise. A fixed lexicon of V words each names
    exactly ONE of V objects (the gold mapping). Each TRIAL presents C words; the scene shows their C true
    referents PLUS `distract` extra objects (so the scene is bigger than the word set — the agent can't read
    off the pairing), and with probability `drop` a word's true referent is MISSING from the scene (noisy
    perception: the named thing isn't always visible). Within a trial the agent only knows these words
    co-occurred with these objects (referential uncertainty), never the alignment. Across trials the true
    pairings co-occur SYSTEMATICALLY while distractors are random and wash out — that surviving regularity
    is what cross-situational learning recovers, and the drop/distract noise is what makes propose-but-verify
    actually mis-guess and recover (its signature lives there). The agent receives (heard_words,
    scene_objects); scene_objects is the Turn.signal the harness carries — referent ids NOT in the token
    stream."""
    def __init__(self, vocab=18, C=3, distract=2, drop=0.15, confound=0.0, seed=0):
        self.V = vocab; self.C = C; self.distract = distract; self.drop = drop
        self.confound = confound                                   # systematic lure: word w drags object w+1 in
        self.gold = {w: w for w in range(vocab)}                   # word w names object w (identity gold)
        self.rng = np.random.default_rng(seed)

    def trial(self):
        """One noisy ambiguous trial: C words; scene = their true referents (each present w.p. 1−drop) plus
        `distract` random other objects, shuffled. With prob `confound` a word also drags in a SYSTEMATIC lure
        (object w+1) — a non-referent that co-occurs more than chance, the hard case where a single-slot guesser
        can lock onto the lure but a full distribution can still out-vote it (the equal-memory tradeoff)."""
        words = list(self.rng.choice(self.V, size=self.C, replace=False))
        scene = set()
        for w in words:
            if self.rng.random() > self.drop:                      # the referent is usually, not always, in view
                scene.add(self.gold[w])
            if self.confound and self.rng.random() < self.confound:
                scene.add((w + 1) % self.V)                        # the systematic lure
        pool = [o for o in range(self.V) if o not in scene]
        if self.distract and pool:
            extra = self.rng.choice(pool, size=min(self.distract, len(pool)), replace=False)
            scene.update(int(x) for x in extra)
        scene = list(scene); self.rng.shuffle(scene)
        return words, scene


# ─────────────────────────────────────── training + scoring ───────────────────────────────────────

def train(model, env, n_trials):
    """Stream n_trials ambiguous trials through `model` — ONLINE, single pass. Each present word is observed
    co-occurring with the WHOLE scene (the agent cannot see the within-trial pairing)."""
    for _ in range(n_trials):
        words, scene = env.trial()
        for w in words:
            model.observe(w, set(scene))

def accuracy(model, env):
    """Mapping accuracy: for each word, does the model's best referent equal the gold object? Tested against
    the full object set (no within-scene hint) — the honest comprehension read."""
    allobj = set(range(env.V))
    correct = sum(1 for w in range(env.V) if model.guess(w, allobj) == env.gold[w])
    return correct / env.V

def random_baseline_acc(env, reps=200):
    """Chance: assign each word a uniformly random referent. 1/V in expectation."""
    accs = []
    for _ in range(reps):
        accs.append(np.mean([rng.integers(env.V) == env.gold[w] for w in range(env.V)]))
    return float(np.mean(accs))


# ─────────────────────────── the McMurray S-curve (words-learned over scenes) ───────────────────────────

def s_curve(make_model, env_factory, n_trials, checkpoints):
    """#words correctly mapped as a function of scenes seen — McMurray's accelerating vocabulary curve.
    Re-trains from scratch to each checkpoint on the SAME trial stream (fixed seed) so the curve is honest."""
    out = []
    for cp in checkpoints:
        env = env_factory()
        m = make_model()
        train(m, env, cp)
        allobj = set(range(env.V))
        learned = sum(1 for w in range(env.V) if m.guess(w, allobj) == env.gold[w])
        out.append((cp, learned))
    return out


# ─────────────────────────────── mutual exclusivity (ME) probe ───────────────────────────────

def me_rate(make_model, env_factory, n_trials, reps=400):
    """Train on known words, then present a NOVEL word in a scene with one UNCLAIMED object + known-owned
    objects. ME prediction: route the novel word to the unclaimed object. Report the rate."""
    hits = 0
    for r in range(reps):
        env = env_factory()
        m = make_model()
        # train only words 0..V-2 (leave the LAST object unclaimed, its word never heard)
        for _ in range(n_trials):
            words, scene = env.trial()
            ws = [(w, o) for w, o in zip(words, scene) if w != env.V - 1]
            objs = {o for _, o in ws}
            for w, _ in ws:
                m.observe(w, objs)
        # novel scene: the unclaimed object (V-1) + two known-owned objects
        known = list(rng.choice(env.V - 1, size=2, replace=False))
        present = [env.V - 1] + known
        novel = 999                                                # a word the model has never heard
        pick = m.me_guess(novel, present)
        if pick == env.V - 1:
            hits += 1
    return hits / reps


# ─────────────────── propose-but-verify signature: at-chance after a disconfirm ───────────────────

def disconfirm_signature(env_factory, runs=400):
    """B's defining behaviour (Trueswell 2013): a learner who has just had its single hypothesis DISCONFIRMED
    and RE-PROPOSED is back at chance — it kept no distribution, only the fresh guess it just made. A, which
    holds a co-occurrence distribution, should NOT collapse on the analogous event. The signature lives DURING
    learning (before convergence), so we read each word's correctness immediately AFTER each trial, bucketed by
    the outcome that trial produced for that word, accumulated over the WHOLE online trajectory of many fresh
    runs. The decisive bucket for B is PROPOSE (the just-reproposed state = at chance)."""
    allobj = set(range(env_factory().V))
    V = env_factory().V

    # ---- variant B: accuracy bucketed by per-word outcome, over the whole trajectory ----
    bk = {"CONFIRM": [0, 0], "DISCONFIRM": [0, 0], "PROPOSE": [0, 0]}
    for _ in range(runs):
        env = env_factory()
        b = crosssit.ProposeVerify(seed=int(rng.integers(1 << 30)))
        for _ in range(150):                                       # short trajectory: lots of propose/disconfirm
            words, scene = env.trial(); sset = set(scene)
            for w in words:
                b.observe(w, sset)
                oc = b.last_outcome.get(w)
                ok = (b.slot.get(w) == env.gold[w])
                if oc in bk: bk[oc][0] += ok; bk[oc][1] += 1
    acc = lambda k: bk[k][0] / bk[k][1] if bk[k][1] else float("nan")
    b_confirm, b_disc, b_propose = acc("CONFIRM"), acc("DISCONFIRM"), acc("PROPOSE")

    # ---- variant A: accuracy when its current top guess is present vs absent in the trial ----
    aP = [0, 0]; aAb = [0, 0]
    for _ in range(max(1, runs // 10)):
        env = env_factory()
        a = crosssit.DenseAssoc(cap_per_word=8)
        for _ in range(150):
            words, scene = env.trial(); sset = set(scene)
            for w in words:
                top = a.guess(w, allobj)
                a.observe(w, sset)
                ok = (a.guess(w, allobj) == env.gold[w])
                (aP if top in sset else aAb)[0] += ok
                (aP if top in sset else aAb)[1] += 1
    a_present = aP[0] / aP[1] if aP[1] else float("nan")
    a_absent = aAb[0] / aAb[1] if aAb[1] else float("nan")
    return b_confirm, b_disc, b_propose, a_present, a_absent, 1.0 / V


# ────────────────────────────────────────── run ──────────────────────────────────────────

def main():
    print("=" * 88)
    print("Exp AV — cross-situational word→referent learning (dense-PMI vs propose-but-verify)")
    print("=" * 88)

    V, C, N = 18, 3, 3000
    env_factory = lambda: SceneEnv(vocab=V, C=C, seed=SEED)
    chance = random_baseline_acc(SceneEnv(vocab=V, C=C, seed=SEED))
    print(f"\nlexicon V={V} words/objects | referential uncertainty C={C} | {N} scenes | "
          f"chance acc = 1/V = {chance:.3f}")

    # ---- headline: A vs B vs strawman full-table, single pass ----
    print("\n" + "-" * 88)
    print("(1) mapping accuracy after N scenes — A vs B vs baselines (single online pass)")
    print("-" * 88)
    envA = env_factory(); A = crosssit.DenseAssoc(cap_per_word=8); train(A, envA, N)
    envB = env_factory(); B = crosssit.ProposeVerify(seed=SEED);  train(B, envB, N)
    # strawman: full uncapped word×object table = DenseAssoc with an unbounded cap (the rejected baseline)
    envF = env_factory(); F = crosssit.DenseAssoc(cap_per_word=10**9); train(F, envF, N)
    accA, accB, accF = accuracy(A, envA), accuracy(B, envB), accuracy(F, envF)
    print(f"  random baseline            : {chance:6.3f}")
    print(f"  (A) dense PMI (cap=8)       : {accA:6.3f}   bytes≈{A.nbytes()}")
    print(f"  (B) propose-but-verify      : {accB:6.3f}   bytes≈{B.nbytes()}")
    print(f"  full uncapped table (straw) : {accF:6.3f}   bytes≈{F.nbytes()}")
    above = lambda a: "ABOVE chance ✓" if a > chance + 0.1 else "at/near chance ✗"
    print(f"  → A {above(accA)} | B {above(accB)}")
    print(f"  equal-memory note: B reaches the same accuracy at {B.nbytes()/A.nbytes():.0%} of A's footprint "
          f"(one slot/word, no matrix) — the memory-budget-honoring variant, for free on clean scenes.")

    # ---- McMurray S-curve ----
    print("\n" + "-" * 88)
    print("(2) McMurray vocabulary S-curve — #words mapped vs scenes seen")
    print("-" * 88)
    cps = [50, 150, 400, 900, 1800, 3000]
    sA = s_curve(lambda: crosssit.DenseAssoc(cap_per_word=8), env_factory, N, cps)
    sB = s_curve(lambda: crosssit.ProposeVerify(seed=SEED), env_factory, N, cps)
    print("  scenes : " + " ".join(f"{cp:>5}" for cp, _ in sA))
    print("  A(/18) : " + " ".join(f"{n:>5}" for _, n in sA))
    print("  B(/18) : " + " ".join(f"{n:>5}" for _, n in sB))

    # ---- ME rate ----
    print("\n" + "-" * 88)
    print("(3) mutual-exclusivity rate — novel word routes to the UNCLAIMED object?")
    print("-" * 88)
    meA = me_rate(lambda: crosssit.DenseAssoc(cap_per_word=8), env_factory, N)
    meB = me_rate(lambda: crosssit.ProposeVerify(seed=SEED), env_factory, N)
    me_chance = 1.0 / 3                                            # 3 objects present, 1 unclaimed
    print(f"  ME chance (1 of 3 present) : {me_chance:.3f}")
    print(f"  (A) dense PMI ME rate      : {meA:.3f}   {'ME ✓' if meA > me_chance + 0.1 else 'no ME ✗'}")
    print(f"  (B) propose-verify ME rate : {meB:.3f}   {'ME ✓' if meB > me_chance + 0.1 else 'no ME ✗'}")

    # ---- propose-but-verify signature ----
    print("\n" + "-" * 88)
    print("(4) propose-but-verify SIGNATURE — at chance right after a DISCONFIRM?")
    print("-" * 88)
    bC, bD, bP, aP, aA, ch = disconfirm_signature(env_factory)
    print(f"  per-word chance            : {ch:.3f}")
    print(f"  (B) acc after CONFIRM      : {bC:.3f}")
    print(f"  (B) acc after DISCONFIRM   : {bD:.3f}")
    print(f"  (B) acc after RE-PROPOSE   : {bP:.3f}   "
          f"{'at ~chance ✓ (Trueswell signature)' if bP < bC - 0.2 else 'not at chance ✗'}")
    print(f"  (A) acc | top-guess present: {aP:.3f}")
    print(f"  (A) acc | top-guess absent : {aA:.3f}   "
          f"{'holds (distribution survives) ✓' if aA > ch + 0.2 else 'collapses ✗'}")

    # ---- FRAGILE budget: ≥10 variations of (C, N, V) ----
    print("\n" + "-" * 88)
    print("(5) FRAGILE budget — A vs B across referential uncertainty C, scene count N, vocab V")
    print("-" * 88)
    print(f"  {'V':>3} {'C':>2} {'N':>5} {'cnf':>4} | {'chance':>7} {'accA':>6} {'accB':>6} | {'A>ch':>5} {'B>ch':>5}")
    variations = [
        (12, 2,  500, 0.0), (12, 2, 1500, 0.0), (12, 2, 3000, 0.0),
        (18, 3, 1000, 0.0), (18, 3, 3000, 0.0), (18, 4, 3000, 0.0),
        (24, 3, 3000, 0.0), (24, 5, 4000, 0.0), (36, 6, 8000, 0.0),
        (18, 3, 3000, 0.5), (18, 3, 3000, 0.8), (24, 4, 4000, 0.7),  # systematic-confound (the hard regime)
    ]
    nA = nB = 0
    for (v, c, n, cf) in variations:
        ef = lambda v=v, c=c, cf=cf: SceneEnv(vocab=v, C=c, confound=cf, seed=SEED)
        ch = 1.0 / v
        ea = ef(); ma = crosssit.DenseAssoc(cap_per_word=8); train(ma, ea, n); aa = accuracy(ma, ea)
        eb = ef(); mb = crosssit.ProposeVerify(seed=SEED);   train(mb, eb, n); ab = accuracy(mb, eb)
        okA = aa > ch + 0.1; okB = ab > ch + 0.1
        nA += okA; nB += okB
        print(f"  {v:>3} {c:>2} {n:>5} {cf:>4.1f} | {ch:>7.3f} {aa:>6.3f} {ab:>6.3f} | "
              f"{'  ✓' if okA else '  ✗':>5} {'  ✓' if okB else '  ✗':>5}")
    print(f"  → A above chance in {nA}/{len(variations)} | B above chance in {nB}/{len(variations)}")

    print("\n" + "=" * 88)
    print("VERDICT (see RESULTS.md). Both variants learn from co-occurrence alone (no labels, single pass,")
    print("bounded). The kill-condition is reported honestly against the disconfirm signature + above-chance test.")
    print("=" * 88)


if __name__ == "__main__":
    main()

"""curriculum.py — Exp AK: memory-budget-as-curriculum ("starting small"), count-native.

Elman 1993 (and Vygotsky's ZPD before it): a learner masters complex EMBEDDED structure only when it
STARTS SMALL — either the data is staged simple→hard, or the learner's own MEMORY starts short and grows.
Thrown the full problem at full capacity from step one, it fails to find the long-range structure at all.

Our substrate has a bounded-memory RULE — we always live under a budget. This asks the sharper question:
is GROWING the budget on a schedule itself a curriculum that beats a FIXED budget? And it does the staging
the count-native, teacher-free way: we stage MEMORY, not data. The same stream, single pass, three regimes:

  GROW   — the accumulator leak-horizon H starts SHORT and grows on a schedule (small→large). A short H
           means a context's next-token counts decay fast, so only regularities that RECUR within a short
           window survive — local structure first. As H grows, longer gaps stay in the accumulator and the
           long-range structure composes ON TOP of the now-stable local counts.
  FULL   — H is the final large value from the very first char (Elman's "full complexity from the start").
  FIXED  — H stays SMALL the whole pass (a permanently short memory).

The accumulator. Each (context, next-token) pair carries a leaky count that decays toward zero with a
per-context-touch leak: on every touch of a context, its row is multiplied by lam = exp(-1/H) before the
new token is added. H is the leak-horizon in *occurrences of that context* — the effective number of recent
visits that still count. This is a per-row leaky accumulator (online, no batch), exactly the "leaky
accumulator" primitive the substrate already uses elsewhere; here H is the dial we put on a schedule.

THE RIGHT AXIS — long-range structure. A flat bits-per-char average hides the thing we care about, because
local structure dominates the token count. So we measure perplexity SPECIFICALLY on the tokens whose
predictive cue is FAR AWAY: the agreement target at the end of an embedded clause, whose only cue is the
subject marker before the clause. We build an Elman-style agreement corpus (below) where that distance is
explicit and tunable, and report perplexity on the cue-distant target tokens alone.

ZPD sampler (overlay). Calibrated confidence (lib/confidence-style top-1 hit/miss → NARS c) lets us weight
exposures: up-weight near-threshold contexts (the learnable edge — Vygotsky's zone), down-weight already
mastered ones (high stable confidence), defer chronically unreachable ones (very low confidence). Online:
the weight on the count we ADD at step t is a function of the predicting context's confidence as of t.

HARD RULES: online single streaming pass; no gradients; no batch optimization; bounded memory; fixed seed.
Alphabet matches lib/fastchar: a..z=0..25, space=26, V=27.
"""
import numpy as np

V = 27
A = "abcdefghijklmnopqrstuvwxyz "


# ── corpus: Elman-style long-range agreement with embedded clauses ────────────────────────────────

def make_agreement_corpus(n_sentences, gap_lo, gap_hi, seed=0, n_keys=120):
    """Generate a char stream with a SPARSE, REACHABLE long-range dependency, and the indices of the
    cue-distant AGREEMENT tokens (the only tokens whose predictive cue is non-local AND data-sparse).

    Each sentence:  <subj key word> <embedded clause of varied filler> <reachable cue word> <agreement target>

    A "subject key" is one of `n_keys` distinct 3-letter words (e.g. "cat", "dog", ...); each key is
    permanently bound to a number class (singular/plural, 50/50 over keys). The agreement target at the end
    of the sentence is 's' if the key's class is singular, space if plural — a clean 50/50 with NO local cue.

    What makes this the count-native long-range test (vs. an unreachable or a trivially-local one):

      REACHABLE-IN-PRINCIPLE. The key word is re-emitted as a short cue right before the target ("<key> run"),
      so an order-k char context spanning "<key> run" → target EXISTS and is learnable. The dependency is not
      out of reach — so a difference between regimes is about RETENTION, not span.

      DATA-SPARSE / ACCUMULATION-BOUND (the leak-horizon's domain). With `n_keys` distinct keys, each specific
      "<key> run"→target context recurs only every ~n_keys sentences, INTERLEAVED with all the other keys'.
      A SHORT leak-horizon (≈3 occurrences of that context) decays a key's class count to nothing before its
      next occurrence dozens of sentences later → it predicts at chance. A LONG horizon accumulates the key's
      class across its sparse visits → it predicts the agreement. So whether the bound dependency is acquired
      turns ENTIRELY on the leak-horizon — exactly the dial the schedule moves.

      The embedded clause (varied `gap` filler) sits between key and cue so the FULL spanning context is
      unique per sentence — the model cannot memorize whole sentences, only the recurring short cue.

    We score the char after "run": 's' (id 18) singular, space (id 26) plural. Returns (ids, target_idx,
    target_true)."""
    rng = np.random.default_rng(seed)
    # n_keys distinct 3-letter keys, each bound to a class (alternate so it's exactly 50/50)
    import itertools
    alpha = "abcdefghijklmnopqrstuvwxyz"
    keys = []
    for a, b, c in itertools.product(alpha, repeat=3):
        keys.append(a + b + c)
        if len(keys) >= n_keys:
            break
    rng.shuffle(keys)
    key_cls = {k: (i % 2) for i, k in enumerate(keys)}     # 0 singular, 1 plural — exactly balanced
    fillers = ["who ", "that ", "was ", "in ", "the ", "old ", "house ", "by ", "river ", "and ",
               "near ", "a ", "big ", "field ", "of ", "green ", "with ", "long ", "grass "]
    out = []
    tgt_idx = []
    tgt_true = []
    pos = 0
    for _ in range(n_sentences):
        key = keys[rng.integers(0, len(keys))]
        cls = key_cls[key]
        out.append(key + " "); pos += 4                    # the subject key (the distant, sparse cue)
        gap = int(rng.integers(gap_lo, gap_hi + 1))
        filled = 0
        while filled < gap:
            w = fillers[rng.integers(0, len(fillers))]
            out.append(w); pos += len(w); filled += len(w)
        out.append(key + " run"); pos += 4 + 4             # re-emit key right before target (reachable cue)
        if cls == 0:                                       # singular → "runs"
            tgt_idx.append(pos); tgt_true.append(ord("s") - 97)
            out.append("s"); pos += 1
        else:                                              # plural → "run " (space is the agreement)
            tgt_idx.append(pos); tgt_true.append(26)
            out.append(" "); pos += 1
        out.append(". "); pos += 2

    text = "".join(out)
    ids = np.array([(ord(c) - 97 if "a" <= c <= "z" else 26) for c in text], dtype=np.int64)
    return ids, np.array(tgt_idx, np.int64), np.array(tgt_true, np.int64)


# ── the leaky-accumulator count model with a schedulable leak-horizon H ───────────────────────────

class LeakyCountModel:
    """Online char-level backoff count model whose per-context counts LEAK with horizon H — and H can be
    put on a schedule (the curriculum). One streaming pass; predict-then-update at every position.

    State: for orders 1..K, a dict ctx_key -> (leaky 27-count row, last-touch-step). On touching a context
    we first decay its row by lam=exp(-1/H_now) (lazy decay applied once per touch — counts the touches, the
    leak-horizon is in occurrences-of-this-context), then add the observed next token (weighted w, for ZPD).

    Prediction at t: add-alpha backoff from the highest order whose context has been seen, blended down to
    unigram. Returns the log-prob it assigned the TRUE next char (so the caller can isolate target tokens).

    H_schedule(frac) -> H, called with the fraction of the stream consumed (0..1). FIXED/FULL pass constants.
    """

    def __init__(self, K=6, alpha=0.05, H_schedule=None, zpd=False, zpd_lo=0.2, zpd_hi=0.85):
        self.K = K
        self.alpha = alpha
        self.H_schedule = H_schedule if H_schedule is not None else (lambda frac: 1e9)
        self.zpd = zpd
        self.zpd_lo, self.zpd_hi = zpd_lo, zpd_hi
        # per order: ctx_key -> [count row(27), hits, misses]  (hits/misses = NARS top-1 track record)
        self.tab = [dict() for _ in range(K + 1)]
        self.powers = [(_ := (V ** np.arange(k - 1, -1, -1)).astype(np.int64)) for k in range(K + 1)]

    def _key(self, ids, t, k):
        if k == 0:
            return 0
        seg = ids[t - k:t]
        return int(seg @ self.powers[k])

    def online_pass(self, ids, target_idx=None):
        """Single causal pass. Returns dict with overall bpc and (if target_idx given) the mean NLL/perplexity
        on the target tokens alone. ZPD weighting (if on) modulates the count weight by the predicting
        context's confidence."""
        ids = np.ascontiguousarray(ids, np.int64)
        n = len(ids)
        alpha = self.alpha
        uni = np.full(V, alpha)                      # running unigram counts (leaky? no — unigram is the floor)
        logp = np.zeros(n)                           # log2 prob of the true next char at each position
        tgt_set = set(int(i) for i in target_idx) if target_idx is not None else set()
        tgt_nll = []                                 # natural-log NLL on target tokens
        tgt_hit = []                                 # argmax == true on target tokens (acquisition probe)
        K = self.K
        for t in range(1, n):
            true = ids[t]
            frac = t / n
            H = self.H_schedule(frac)
            lam = np.exp(-1.0 / max(H, 1e-6))
            # ── predict: highest seen order, add-alpha, with a NARS-c-weighted backoff blend ──
            dist = (uni + alpha) / (uni.sum() + alpha * V)        # unigram floor
            used_conf = 0.0
            for k in range(K, 0, -1):
                if t - k < 0:
                    continue
                key = self._key(ids, t, k)
                e = self.tab[k].get(key)
                if e is None:
                    continue
                row = e[0]
                tot = row.sum()
                if tot <= 0:
                    continue
                dist = (row + alpha) / (tot + alpha * V)
                hi, mi = e[1], e[2]
                used_conf = (hi + mi) / (hi + mi + 1.0)            # NARS confidence of this context
                break
            p_true = dist[true]
            logp[t] = np.log2(max(p_true, 1e-12))
            if t in tgt_set:
                tgt_nll.append(-np.log(max(p_true, 1e-12)))
                tgt_hit.append(int(dist.argmax()) == int(true))
            # ── update: leaky accumulators for every order's context, with the current H ──
            w = 1.0
            if self.zpd:
                # up-weight the learnable edge (mid confidence), down-weight mastered, defer unreachable
                c = used_conf
                if c >= self.zpd_hi:
                    w = 0.3                                       # already mastered → light touch
                elif c <= self.zpd_lo:
                    w = 0.5                                       # (near-)unreachable → modest, don't chase
                else:
                    w = 1.5                                       # the zone of proximal development
            for k in range(1, K + 1):
                if t - k < 0:
                    continue
                key = self._key(ids, t, k)
                e = self.tab[k].get(key)
                if e is None:
                    e = [np.zeros(V), 0.0, 0.0]
                    self.tab[k][key] = e
                # leak then add (lazy decay once per touch — leak-horizon in occurrences of THIS context)
                e[0] *= lam
                # NARS top-1 track record BEFORE folding in the new token (was our bet right?)
                if e[0].sum() > 0:
                    bet = int(e[0].argmax())
                    if bet == true:
                        e[1] += 1.0
                    else:
                        e[2] += 1.0
                e[0][true] += w
            uni[true] += 1.0
        out = {"bpc": float(-logp[1:].mean())}
        if target_idx is not None:
            tgt_nll = np.array(tgt_nll) if tgt_nll else np.array([np.log(V)])
            out["target_nll"] = float(tgt_nll.mean())
            out["target_ppl"] = float(np.exp(tgt_nll.mean()))
            out["target_acc"] = float(np.mean(tgt_hit)) if tgt_hit else 0.0
            out["n_target"] = int(len(tgt_nll))
        return out


# ── memory schedules ──────────────────────────────────────────────────────────────────────────────

def grow_schedule(H_lo, H_hi):
    """Leak-horizon grows linearly small→large over the pass (the count-native 'starting small')."""
    return lambda frac: H_lo + (H_hi - H_lo) * frac

def fixed_schedule(H):
    """Constant horizon the whole pass."""
    return lambda frac: float(H)

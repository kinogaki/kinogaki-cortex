"""extmemory.py — Exp AQ: environment-as-memory ("writing it down"), the bounded-memory rule's 3rd coping route.

The bounded-memory rule (Exps AE no-forget, AA sleep, AI power-law) keeps asking the same question: under a FIXED
memory budget, what survives eviction? Two coping routes were already tested — (1) evict the right tail well
(AE's LTI/ART, AI's ACT-R exp(B)), and (2) consolidate the head offline (AA's sleep replay). This file tests the
THIRD route humans actually use most: we DON'T hold everything in our heads — we WRITE IT DOWN. A small fast
internal memory, plus a big slow EXTERNAL store reached by retrieval cues. Ericsson & Kintsch's long-term working
memory: experts keep a tiny set of *cues* in their narrow focus and the *content* in an external store (on paper,
on the board, in the file) that the cues retrieve on demand.

The architectures compared, at EQUAL TOTAL budget:

  AllInternal — one bounded backoff count table. When an order's table overflows it EVICTS its tail (reservoir-
    sampled lowest leaky use-score). The frequent head stays; the long tail is forgotten. Everything the model
    will ever answer from lives in this one bounded store. (This is Exp AE's FLAT baseline, reused honestly.)

  IntExt — a SMALL fast internal table (the narrow focus) PLUS an EXTERNAL store (paper). On overflow the internal
    table doesn't just drop its tail — it WRITES the evicted context's counts to the external store (a hash-keyed
    on-disk-shaped table), then drops it from internal. At PREDICT time the internal table answers when it is
    CONFIDENT (its top context's distribution is peaky enough); when it is UNCERTAIN (high entropy / no high-order
    context) it pays a RETRIEVAL: look the context up in the external store and use those counts instead. The
    external store is the long tail the internal model already evicted — kept on paper, re-read only when needed.

ACCOUNTING (stated honestly). "Budget" = number of stored context entries (each entry is a 27-float count vector
plus scalars — same unit for both architectures). AllInternal gets the WHOLE budget internal. IntExt splits it:
a small internal table (`int_cap` per order) + an external store capped at `ext_cap` per order, with
int_cap + ext_cap == AllInternal's cap PER ORDER. So at equal total entry budget the only difference is the SPLIT
and the WRITE-DOWN/RE-READ policy. We DO charge the external store against the budget (the strict, fair test). We
ALSO report a "slow/cheap external" variant where the external store is larger (paper is cheaper than skull) so
the reader can see both the strict equal-budget verdict and the realistic asymmetric-cost one.

The external store is NOT free in TIME either: every read is counted (retrieval rate, hit rate). A real on-disk
index would pay I/O per consult; we keep it in a separate dict (same data layout an on-disk table would have) and
report how often it is consulted, so the cost is visible.

HARD RULES honored: ONLINE single streaming pass; NO gradients / batch optimization; bounded memory is the whole
point; leaks are per-entry O(1) recurrences; eviction is reservoir-sampled (no global sort). Alphabet matches
lib/fastchar / lib/retention: a..z = 0..25, space = 26, V = 27. Fixed seed.
"""
import os, re
import numpy as np

V = 27
ALPHA = 0.1
A = "abcdefghijklmnopqrstuvwxyz "
_RNG = np.random.default_rng(0)


# ── data ──────────────────────────────────────────────────────────────────────────────────────

def clean_file(path):
    """Project-gutenberg-aware char cleaner (same as lib/retention): strip the boilerplate, lowercase,
    everything non-letter → space, collapse runs. Returns an int64 id array in our space."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S)
    m2 = re.search(r"\*\*\* END OF", raw, re.S)
    if m1 and m2:
        raw = raw[m1.end():m2.start()]
    raw = raw.lower()
    raw = re.sub(r"[^a-z]+", " ", raw)
    raw = re.sub(r" +", " ", raw).strip()
    return np.array([ord(c) - 97 if c != " " else 26 for c in raw], dtype=np.int64)


def _ctx_key(ctx_ids):
    k = len(ctx_ids)
    powers = (V ** np.arange(k - 1, -1, -1)).astype(np.int64)
    return int(ctx_ids @ powers)


def _sample_victim(d, score, n=24):
    """Reservoir-style eviction: sample ~n keys, return the lowest-score one. O(n), online — no global sort."""
    keys = list(d.keys())
    if len(keys) <= n:
        cand = keys
    else:
        cand = [keys[i] for i in _RNG.integers(0, len(keys), size=n)]
    return min(cand, key=lambda kk: score(d[kk]))


def _add_alpha(c):
    return (c + ALPHA) / (c.sum() + ALPHA * V)


def _entropy(p):
    return float(-(p * np.log2(p + 1e-12)).sum())


# ── A: all-internal — one bounded count table, evict the tail (Exp AE's FLAT, reused) ───────────

class AllInternal:
    """Bounded backoff counts, single timescale. Entry = [counts(27), use, last]. Leaky recency use-score;
    evict the lowest effective use. cap is PER ORDER. The long tail is FORGOTTEN — there is nowhere else for it."""

    def __init__(self, K=5, cap=4000, leak=0.9995):
        self.K = K; self.cap = cap; self.leak = leak
        self.tab = [dict() for _ in range(K + 1)]
        self.clock = 0

    def train_stream(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        for ti in range(1, n):
            t = self.clock + ti
            tok = int(ids[ti]); kmax = min(self.K, ti)
            for k in range(kmax, 0, -1):
                key = _ctx_key(ids[ti - k:ti])
                d = self.tab[k]; e = d.get(key)
                if e is None:
                    if len(d) >= self.cap:
                        del d[_sample_victim(d, lambda x, now=t: x[1] * (self.leak ** (now - x[2])))]
                    e = [np.zeros(V, np.float64), 0.0, t]; d[key] = e
                e[1] = e[1] * (self.leak ** (t - e[2])) + 1.0
                e[2] = t
                e[0][tok] += 1.0
        self.clock += n - 1

    def predict(self, ctx):
        """Highest-order-seen backoff over the internal table. Returns (dist, used_order)."""
        for k in range(min(self.K, len(ctx)), 0, -1):
            e = self.tab[k].get(_ctx_key(ctx[-k:]))
            if e is not None and e[0].sum() > 0:
                return _add_alpha(e[0]), k
        return np.full(V, 1.0 / V), 0

    def eval_bpc(self, ids, mask=None):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); s = 0.0; cnt = 0
        for t in range(1, n):
            if mask is not None and not mask[t]:
                continue
            p, _ = self.predict(ids[max(0, t - self.K):t])
            s += -np.log2(p[int(ids[t])] + 1e-12); cnt += 1
        return s / max(cnt, 1)

    def size(self):
        return sum(len(d) for d in self.tab[1:])

    # uniform reporting hooks (no external store here)
    consults = 0
    hits = 0
    def ext_size(self):
        return 0


# ── B: bounded-internal + external store — write it down, re-read on uncertainty ────────────────

class IntExt:
    """Small fast INTERNAL table + an EXTERNAL store the model WRITES evicted contexts to and RE-READS when the
    internal answer is uncertain.

      Internal — bounded per order at `int_cap`; same leaky-recency eviction as AllInternal. BUT on overflow,
        before dropping the victim, its accumulated counts are WRITTEN DOWN to the external store (merged if the
        context is already on paper). The internal table holds the live, frequent head.

      External — capped per order at `ext_cap`; entry = [counts(27), use, last]. It receives the internal table's
        evicted tail. When it overflows it evicts its own lowest-use entry (paper has a margin too). This is the
        long tail, on disk-shaped storage, only ever READ on demand.

      Predict — try the internal table's highest-order-seen context. If it is CONFIDENT (entropy of its
        distribution ≤ `conf_h` bits, i.e. it has a peaky opinion), answer from internal — no retrieval. If it is
        UNCERTAIN (no high-order internal context, or its distribution is flat), pay a RETRIEVAL: look the SAME
        context up in the external store; if found, BLEND internal+external counts (the cue retrieves the
        written-down content) and answer from the richer evidence; else fall back to internal. We count every
        consult and every hit.

    Total entry budget = int_cap + ext_cap per order; set equal to AllInternal.cap for the strict fair test.
    """

    def __init__(self, K=5, int_cap=1000, ext_cap=3000, leak=0.9995, conf_h=2.0):
        self.K = K; self.int_cap = int_cap; self.ext_cap = ext_cap
        self.leak = leak; self.conf_h = conf_h
        self.itab = [dict() for _ in range(K + 1)]           # internal (focus)
        self.etab = [dict() for _ in range(K + 1)]           # external store (paper)
        self.clock = 0
        self.consults = 0                                    # external reads attempted (since last reset)
        self.hits = 0                                        # external reads that found the context
        self.writes = 0                                      # contexts written down (over training)

    def reset_counters(self):
        """Zero the consult/hit tallies so a reported eval pass measures retrieval over THAT pass only
        (eval_bpc is run several times per model — overall/rare/common — and we don't want to sum them)."""
        self.consults = 0; self.hits = 0

    def _write_down(self, k, key, victim, t):
        """Evicted internal entry → external store (merge counts if already on paper)."""
        ed = self.etab[k]; ee = ed.get(key)
        if ee is None:
            if len(ed) >= self.ext_cap:
                del ed[_sample_victim(ed, lambda x, now=t: x[1] * (self.leak ** (now - x[2])))]
            ee = [np.zeros(V, np.float64), 0.0, t]; ed[key] = ee
        ee[0] += victim[0]                                   # accumulate the written-down counts
        ee[1] = max(ee[1], victim[1]); ee[2] = t
        self.writes += 1

    def train_stream(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        for ti in range(1, n):
            t = self.clock + ti
            tok = int(ids[ti]); kmax = min(self.K, ti)
            for k in range(kmax, 0, -1):
                key = _ctx_key(ids[ti - k:ti])
                d = self.itab[k]; e = d.get(key)
                if e is None:
                    if len(d) >= self.int_cap:
                        vkey = _sample_victim(d, lambda x, now=t: x[1] * (self.leak ** (now - x[2])))
                        self._write_down(k, vkey, d[vkey], t)   # WRITE IT DOWN before forgetting
                        del d[vkey]
                    e = [np.zeros(V, np.float64), 0.0, t]; d[key] = e
                e[1] = e[1] * (self.leak ** (t - e[2])) + 1.0
                e[2] = t
                e[0][tok] += 1.0
        self.clock += n - 1

    def predict(self, ctx):
        """Internal first; on uncertainty, RE-READ the external store and blend. Returns (dist, used_order, src)
        where src in {'int','ext','blend','prior'}."""
        # highest-order-seen INTERNAL context
        ik = 0; ie = None
        for k in range(min(self.K, len(ctx)), 0, -1):
            e = self.itab[k].get(_ctx_key(ctx[-k:]))
            if e is not None and e[0].sum() > 0:
                ik = k; ie = e; break

        if ie is not None:
            ip = _add_alpha(ie[0])
            if _entropy(ip) <= self.conf_h:                  # CONFIDENT — no need to consult paper
                return ip, ik, "int"

        # UNCERTAIN — pay a retrieval. Look the highest-order context up on paper.
        self.consults += 1
        best_ext = None; ek = 0
        for k in range(min(self.K, len(ctx)), 0, -1):
            ee = self.etab[k].get(_ctx_key(ctx[-k:]))
            if ee is not None and ee[0].sum() > 0:
                best_ext = ee; ek = k; break
        if best_ext is not None:
            self.hits += 1
            if ie is not None and ik >= ek:                  # blend internal + written-down content
                return _add_alpha(ie[0] + best_ext[0]), max(ik, ek), "blend"
            return _add_alpha(best_ext[0]), ek, "ext"
        if ie is not None:                                   # paper had nothing — fall back to internal
            return _add_alpha(ie[0]), ik, "int"
        return np.full(V, 1.0 / V), 0, "prior"

    def eval_bpc(self, ids, mask=None):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); s = 0.0; cnt = 0
        for t in range(1, n):
            if mask is not None and not mask[t]:
                continue
            p, _, _ = self.predict(ids[max(0, t - self.K):t])
            s += -np.log2(p[int(ids[t])] + 1e-12); cnt += 1
        return s / max(cnt, 1)

    def size(self):
        return sum(len(d) for d in self.itab[1:])

    def ext_size(self):
        return sum(len(d) for d in self.etab[1:])

    def total_size(self):
        return self.size() + self.ext_size()


# ── the rare slice: which eval positions need an evicted (rare) context ─────────────────────────

def rare_mask(train_ids, eval_ids, K=5, rare_max=3):
    """A held-out position is RARE iff its highest-order context (order K, the one a bounded table evicts first)
    appeared at most `rare_max` times in the TRAIN stream. These are exactly the entries an all-internal bounded
    table is most likely to have evicted — the slice where 'writing it down' should pay off. Returns a bool mask
    over eval positions (True = rare). Built by an offline count of the train stream (this is measurement of the
    test set's difficulty, not part of either model)."""
    counts = {}
    tr = np.ascontiguousarray(train_ids, np.int64)
    for ti in range(K, len(tr)):
        key = _ctx_key(tr[ti - K:ti])
        counts[key] = counts.get(key, 0) + 1
    ev = np.ascontiguousarray(eval_ids, np.int64)
    mask = np.zeros(len(ev), dtype=bool)
    for t in range(K, len(ev)):
        key = _ctx_key(ev[t - K:t])
        c = counts.get(key, 0)
        mask[t] = (1 <= c <= rare_max)                       # seen, but rare (so it COULD have been kept)
    return mask

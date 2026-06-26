"""powerlaw_word.py — Exp AR: power-law (ACT-R) memory at the WORD/CONCEPT level, under a budget, on a
NON-STATIONARY stream. The parked resurrection from Exp AI.

Exp AI built the ACT-R base-level activation B = ln(Σ_k age_k^(−d)) (Anderson & Schooler 1991) and tested it as a
budgeted-eviction policy for DENSE char n-grams. It LOST to raw-count LFU at every cap, for a clean reason: LFU is
the power law's d→0 limit, and a char-gram's predictive value is almost entirely its TOTAL COUNT — there is no
"useful-then-stale" structure at order ≤5 over one corpus, so recency is variance, not signal. But Exp AI made a
prediction: the power law should WIN where frequency STOPS ranking usefulness — SPARSE, NON-STATIONARY memory, i.e.
the WORD/CONCEPT level. This file tests exactly there.

── Why words are the opposite case from char-grams ──────────────────────────────────────────────────────────────
A WORD-bigram context (prev word → next word) is:
  · SPARSE — millions of distinct contexts, most seen a handful of times; a memory cap bites hard and often.
  · NON-STATIONARY — "species"/"selection" predict richly inside Darwin and then go STALE when the stream shifts to
    Shakespeare; "thou"/"thee" do the reverse. A word's usefulness is tied to the current TOPIC/REGISTER, not its
    lifetime frequency. This is the "useful-then-stale" structure char-grams lack.
Under a cap on a topic-shifting stream, the right thing to keep is the rare-but-recently-relevant context over the
high-frequency-but-stale one. LFU keeps the stale high-count word; the power law (frequency × recency × spacing)
should keep the recently-relevant one. THE TEST: does evict-lowest-B now BEAT evict-lowest-count?

We REUSE the exact ACT-R accumulator from Exp AI (lib.powerlaw.actr_weight) — same incremental Petrov/Anderson
approximation, per-entry O(1), no stored timestamps. Only the TOKENS change: char ids → word ids.

HARD RULES honored: single streaming pass; no gradients; no batch optimization; bounded memory (the cap is the
point). Eviction is reservoir-sampled lowest-score (O(1) amortized, no global sort). Fixed seed.
"""
import os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lib.powerlaw import actr_weight   # the Exp AI ACT-R accumulator — reused verbatim

_RNG = np.random.default_rng(0)


# ── data: word-id streams from the register files (Gutenberg-cleaned, shared shape with Exp AI/AE) ──────────────

def _clean_text(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", raw, re.S)
    m2 = re.search(r"\*\*\* END OF", raw, re.S)
    if m1 and m2:
        raw = raw[m1.end():m2.start()]
    raw = raw.lower()
    raw = re.sub(r"[^a-z]+", " ", raw)
    return [w for w in raw.split(" ") if w]


class Vocab:
    """Streaming word→id map. New words get the next id (online — no pre-pass over the corpus). An OOV id is
    reserved at 0 so eval on a held-out slice never KeyErrors on a word train never saw."""
    OOV = 0

    def __init__(self):
        self.w2i = {}
        self.i2w = ["<oov>"]

    def encode(self, words, grow=True):
        out = np.empty(len(words), dtype=np.int64)
        for j, w in enumerate(words):
            i = self.w2i.get(w)
            if i is None:
                if grow:
                    i = len(self.i2w); self.w2i[w] = i; self.i2w.append(w)
                else:
                    i = self.OOV
            out[j] = i
        return out

    def __len__(self):
        return len(self.i2w)


def load_words(path, vocab, n=None, grow=True):
    words = _clean_text(path)
    if n:
        words = words[:n]
    return vocab.encode(words, grow=grow)


# ── reservoir eviction (same shape as Exp AI) ──────────────────────────────────────────────────────────────────

def _sample_victim(d, score, n=24):
    keys = list(d.keys())
    if len(keys) <= n:
        cand = keys
    else:
        cand = [keys[i] for i in _RNG.integers(0, len(keys), size=n)]
    return min(cand, key=lambda kk: score(d[kk]))


# ── the word-level count model: one context order, four eviction policies under a cap ──────────────────────────

class WordCountModel:
    """Bounded add-α word model. context = previous `order` word(s); predict the next word. The context table is
    capped; policies differ ONLY in what they EVICT on overflow (identical counting + prediction, so peak quality
    is comparable). Per-context entry holds a SUCCESSOR-COUNT dict (sparse — a word context has few successors),
    plus the ACT-R state [n, t_create, t_last, ema_use].

    The cap is on the NUMBER OF CONTEXTS (distinct prev-word keys). This is the budget the policies fight over.

    policy:
      'powerlaw' — evict lowest ACT-R weight exp(B): frequency × recency × spacing. The bet.
      'lfu'      — evict smallest n (total use count). The Exp AI char-gram winner.
      'lru'      — evict largest age (oldest t_last).
      'ema'      — evict lowest exponential leaky use-score (geometric recency; the Exp AE FLAT baseline).
      'none'     — unbounded (cap ignored); upper bound.
    """

    def __init__(self, order=1, cap=20000, d=0.5, ema_leak=0.999, alpha=0.05, policy="powerlaw"):
        self.order = order; self.cap = cap; self.d = d; self.ema_leak = ema_leak
        self.alpha = alpha; self.policy = policy
        self.tab = {}        # context-key -> [succ:dict(word->count), n, t_create, t_last, ema_use]
        self.clock = 0
        self.vocab_seen = set([Vocab.OOV])   # words ever seen as a successor (for add-α denominator support size)

    def _ctx_key(self, prev_ids):
        if self.order == 1:
            return int(prev_ids[-1])
        return tuple(int(x) for x in prev_ids[-self.order:])

    def _score(self, e, now):
        p = self.policy
        if p == "powerlaw":
            # actr_weight expects e[1]=n, e[2]=t_create, e[3]=t_last (e[0] unused for the weight).
            return actr_weight([None, e[1], e[2], e[3], None], now, self.d)
        if p == "lfu":
            return e[1]
        if p == "lru":
            return -(now - e[3])
        if p == "ema":
            return e[4] * (self.ema_leak ** (now - e[3]))
        return 0.0

    def _new_entry(self, t):
        return [dict(), 0.0, float(t), float(t), 0.0]   # succ, n, t_create, t_last, ema_use

    def train_stream(self, ids):
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids)
        o = self.order
        for ti in range(o, n):
            t = self.clock + ti
            tok = int(ids[ti])
            self.vocab_seen.add(tok)
            key = self._ctx_key(ids[ti - o:ti])
            e = self.tab.get(key)
            if e is None:
                if self.policy != "none" and len(self.tab) >= self.cap:
                    sc = self._score
                    del self.tab[_sample_victim(self.tab, lambda x, now=t: sc(x, now))]
                e = self._new_entry(t); self.tab[key] = e
            e[0][tok] = e[0].get(tok, 0.0) + 1.0
            e[1] += 1.0
            e[4] = e[4] * (self.ema_leak ** (t - e[3])) + 1.0
            e[3] = float(t)
        self.clock += n - o

    def _logprob(self, prev_ids, nxt):
        """log2 P(nxt | ctx) under add-α over the seen-successor vocabulary; backs off to a uniform-over-seen
        prior when the context is absent (evicted or never seen). V = number of distinct successor words seen so
        far (bounded, grows online) — keeps the add-α denominator honest under streaming."""
        V = max(len(self.vocab_seen), 2)
        key = self._ctx_key(prev_ids)
        e = self.tab.get(key)
        a = self.alpha
        if e is None:
            # no context: uniform-over-seen-vocabulary prior.
            return np.log2(1.0 / V)
        c = e[0].get(nxt, 0.0)
        tot = sum(e[0].values())
        return np.log2((c + a) / (tot + a * V))

    def eval_bpw(self, ids):
        """held-out bits-per-WORD (lower=better) AND next-word accuracy (argmax hit-rate). eval is frozen — no
        counting, no clock advance — so a register's score reflects only what currently survives in the table."""
        ids = np.ascontiguousarray(ids, np.int64); n = len(ids); o = self.order
        s = 0.0; hits = 0; tot = 0
        for ti in range(o, n):
            prev = ids[ti - o:ti]; nxt = int(ids[ti])
            s += -self._logprob(prev, nxt)
            # accuracy: argmax successor of the context (if present), else count as miss.
            e = self.tab.get(self._ctx_key(prev))
            if e is not None and e[0]:
                pred = max(e[0].items(), key=lambda kv: kv[1])[0]
                if pred == nxt:
                    hits += 1
            tot += 1
        return s / tot, hits / tot

    def size(self):
        return len(self.tab)

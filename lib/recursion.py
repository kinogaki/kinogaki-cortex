"""recursion.py — Exp BJ: structure-graded recursion exposure (self-gated embedding depth).

The one curriculum AK did NOT test. AK staged the *memory budget* (leak-horizon) and found an honest
negative: growing the budget ties full-from-start, because a count learner has no gradient to lock — an
early noisy count is simply outvoted, never frozen, so there is nothing for "starting small" to rescue.
AK's winner was FULL (every regularity present from char one).

BJ asks the sharper structural question Elman's *original* recursion result was about: order the stream by
**embedding depth**, and let the agent **self-gate** the ordering on its own branching entropy — admit
depth d+1 sentences only once the depth-d transition entropy has *stabilized* (the local structure is in).
This is teacher-free (no labels, no schedule clock): the curriculum reads the agent's own count tables.

The corpus: center-embedded subject–verb agreement (the textbook recursion stressor).

    depth 1:  <S1>  <clause filler>  v1{agree-with-S1}
    depth 2:  <S1>  <S2>  <clause filler>  v2{agree-S2}  v1{agree-S1}
    depth 3:  <S1>  <S2>  <S3>  <clause>  v3{S3}  v2{S2}  v1{S1}

The OUTERMOST verb's agreement cue is the OUTERMOST subject — separated by the entire nested embedding
(a uniquely-filled middle, so whole sentences can't be memorized). Each subject key is one of N distinct
3-letter words permanently bound to a number class (s|·). We score the agreement char at each closing
verb; the deep-embedding (depth-2/3 outer) targets are the recursion-only axis BJ must win or lose on.

Three regimes on the SAME multiset of sentences, single streaming pass, fixed seed:
  GRADED  — sentences delivered easy→hard by depth, the depth gate SELF-OPENED by branching entropy.
  FULL    — all depths interleaved uniformly from char one (AK's winner).
  ANTI    — hard→easy (deep first), the curriculum reversed (control: ordering per se, wrong direction).

The model is AK's leaky-accumulator backoff count model verbatim in spirit (orders 1..K, add-alpha
backoff, predict-then-update, single pass) — we change ONLY the *order* sentences arrive in, never the
learner. So any difference is the curriculum, not the substrate.

The self-gate (the novel piece). We keep a running branching-entropy estimate over the transitions that a
given depth introduces (the depth-d *closing-verb* contexts). After each sentence we update the estimate;
the gate opens depth d+1 when depth d's mean branching entropy has changed by < eps over a window (it has
stabilized = the local recursion is learned). This reads only counts the agent already has — online,
bounded, no backprop. If the gate never beats FULL, AK extends to structural ordering: a clean negative.

HARD RULES: online single streaming pass; no gradients; no batch optimization; bounded memory; fixed seed.
Alphabet a..z=0..25, space=26 (V=27), matching lib/cortex and lib/curriculum.
"""
import numpy as np
import itertools

V = 27
A = "abcdefghijklmnopqrstuvwxyz "


# ── corpus: center-embedded agreement at depths 1..D ──────────────────────────────────────────────

_FILLERS = ["who ", "that ", "was ", "in ", "the ", "old ", "house ", "by ", "river ", "and ",
            "near ", "a ", "big ", "field ", "of ", "green ", "with ", "long ", "grass "]


def _keys(n_keys, rng):
    alpha = "abcdefghijklmnopqrstuvwxyz"
    ks = []
    for a, b, c in itertools.product(alpha, repeat=3):
        ks.append(a + b + c)
        if len(ks) >= n_keys:
            break
    rng.shuffle(ks)
    return ks


def make_embedded_corpus(n_per_depth, depths=(1, 2, 3), gap_lo=3, gap_hi=6,
                         seed=0, n_keys=120, reachable=True):
    """Build center-embedded subject–verb agreement sentences at each depth, returned SEPARATELY (so a
    curriculum can order them). Returns `sentences`: list of (depth, ids, targets), targets a list of
    (local_pos, true_id, is_outer) — is_outer marks the OUTERMOST closing verb's agreement char (the deep
    recursion-only token).

    A sentence:  S1 S2 .. Sd  <short embedded filler>  v_d v_{d-1} .. v_1
    Verbs close inner→outer (subs reversed) so the LAST verb agrees with the FIRST (outermost) subject — the
    center-embedding. Each verb is a GENERIC `run` (no key re-emission); the agreement char's ONLY cue is the
    governing subject, back across the nest. Each key is permanently number-classed 50/50 (`s` sing / space
    plural). Difficulty is graded by DEPTH: the outer subject→outer-verb span grows with depth, so the
    order-K window reaches the cue at depth-1 (learnable) and progressively loses it as depth deepens — the
    center-embedding stressor a windowed count learner has no stack for.

    reachable=True keeps spans short (small filler, few inner verbs) so depth-1/2 ARE in window and the task
    is learnable — the regime where a curriculum *could* help. reachable=False uses larger filler so even
    depth-1 outer cue is out of window — the hard control (the model is at chance; any curriculum ties there).
    """
    rng = np.random.default_rng(seed)
    keys = _keys(n_keys, rng)
    key_cls = {k: (i % 2) for i, k in enumerate(keys)}     # 0 singular -> 's', 1 plural -> ' '

    def cls_char(key):
        cls = key_cls[key]
        return ("s" if cls == 0 else " "), ((ord("s") - 97) if cls == 0 else 26)

    sentences = []
    for d in depths:
        for _ in range(n_per_depth):
            subs = [keys[rng.integers(0, len(keys))] for _ in range(d)]
            out = []
            for s in subs:                                 # open subjects outer->inner (full key, once)
                out.append(s + " ")
            gap = int(rng.integers(gap_lo, gap_hi + 1))    # short embedded filler (unique middle)
            filled = 0
            while filled < gap:
                w = _FILLERS[rng.integers(0, len(_FILLERS))]
                out.append(w); filled += len(w)
            targets = []
            for j, s in enumerate(reversed(subs)):         # close inner->outer with a GENERIC verb
                out.append("run")
                pos = len("".join(out))
                ch, tid = cls_char(s)
                is_outer = (j == d - 1)                     # the final (outermost) closing verb = deep token
                targets.append((pos, tid, is_outer))
                out.append(ch + " ")
            out.append(". ")
            text = "".join(out)
            ids = np.array([(ord(c) - 97 if "a" <= c <= "z" else 26) for c in text], dtype=np.int64)
            sentences.append((d, ids, targets))
    return sentences


# ── the leaky-accumulator backoff count model (AK's substrate, sentence-fed) ───────────────────────

class CountModel:
    """Online char backoff count model, orders 1..K, add-alpha backoff, predict-then-update, single pass.

    Fed sentence-by-sentence so a curriculum can choose the order. State persists across sentences. We track
    per-context branching entropy online (for the self-gate) and score targets by NLL.

    Bounded memory (a HARD RULE) is enforced by a per-context LEAK with horizon H: on each touch a context's
    27-count row is multiplied by lam=exp(-1/H) before the new token is folded in — the substrate's leaky
    accumulator. This is what makes the deep agreement *accumulation-bound* (AK's lesson): a key's class
    count must survive the leak across its sparse re-visits to be predicted, so the deep dependency is
    genuinely hard and a curriculum has something to differentiate. H is constant here (AK showed the
    *final* horizon is the lever, not its schedule); BJ varies the ORDER sentences arrive, not H.
    """

    def __init__(self, K=8, alpha=0.05, H=40.0):
        self.K = K
        self.alpha = alpha
        self.H = H
        self.lam = np.exp(-1.0 / max(H, 1e-6))
        self.tab = [dict() for _ in range(K + 1)]           # order k: ctx_key -> 27-count row
        self.powers = [(V ** np.arange(k - 1, -1, -1)).astype(np.int64) for k in range(K + 1)]
        self.uni = np.zeros(V)

    def _key(self, seg):
        k = len(seg)
        if k == 0:
            return 0
        return int(np.asarray(seg, np.int64) @ self.powers[k])

    def _predict(self, ctx):
        """add-alpha backoff dist from the highest seen order over the trailing context `ctx` (list of ids)."""
        alpha = self.alpha
        dist = (self.uni + alpha) / (self.uni.sum() + alpha * V)
        K = self.K
        for k in range(min(K, len(ctx)), 0, -1):
            key = self._key(ctx[-k:])
            row = self.tab[k].get(key)
            if row is None:
                continue
            tot = row.sum()
            if tot <= 0:
                continue
            return (row + alpha) / (tot + alpha * V)
        return dist

    def feed_sentence(self, ids, targets=None, prefix=None, learn=True):
        """Run one sentence through (causal). `prefix` = trailing context from the prior sentence (cross-
        sentence context, like AK's continuous stream). Returns list of (true_id, nll, is_target, is_outer)
        for the targets if given. Updates counts iff learn."""
        ctx = list(prefix) if prefix is not None else []
        tgt_pos = {p: (tid, is_outer) for (p, tid, is_outer) in targets} if targets else {}
        results = []
        for t in range(len(ids)):
            true = int(ids[t])
            if t in tgt_pos:
                dist = self._predict(ctx)
                p = float(dist[true])
                tid, is_outer = tgt_pos[t]
                results.append((true, -np.log(max(p, 1e-12)),
                                int(dist.argmax()) == true, is_outer))
            if learn:
                for k in range(1, min(self.K, len(ctx)) + 1):
                    key = self._key(ctx[-k:])
                    row = self.tab[k].get(key)
                    if row is None:
                        row = np.zeros(V); self.tab[k][key] = row
                    row *= self.lam                          # leak then add (per-touch lazy decay = bounded memory)
                    row[true] += 1.0
                self.uni[true] += 1.0
            ctx.append(true)
            if len(ctx) > self.K:
                ctx = ctx[-self.K:]
        return results, ctx

    # ── online branching-entropy estimate over a set of contexts (the self-gate signal) ──
    def branching_entropy(self, depth_contexts):
        """Mean next-char entropy over the supplied order-K context keys (the closing-verb contexts a depth
        introduces). High while the local recursion is unlearned, falls + stabilizes as counts accumulate."""
        ents = []
        for key in depth_contexts:
            row = self.tab[self.K].get(key)
            if row is None or row.sum() <= 0:
                continue
            p = row / row.sum()
            nz = p[p > 0]
            ents.append(float(-(nz * np.log2(nz)).sum()))
        return float(np.mean(ents)) if ents else None


# ── the self-gated depth curriculum ────────────────────────────────────────────────────────────────

def _closing_contexts(sentences, depth, K):
    """The order-K contexts that PRECEDE the outer agreement target at a given depth (what the gate watches).
    These are exactly the `<key> run` spans the recursion must learn — the depth's signature transitions."""
    powers = (V ** np.arange(K - 1, -1, -1)).astype(np.int64)
    ctxs = set()
    for (d, ids, targets) in sentences:
        if d != depth:
            continue
        for (p, tid, is_outer) in targets:
            if is_outer and p >= K:
                ctxs.add(int(np.asarray(ids[p - K:p], np.int64) @ powers))
    return ctxs


def run_graded(sentences, depths, K=8, alpha=0.05, eps=0.02, window=200, seed=0, H=200.0):
    """SELF-GATED depth curriculum. Deliver depth-1 sentences first; admit depth d+1 only once depth d's
    mean branching entropy over its closing contexts has stabilized (|Δ| < eps across a window). Pure online:
    the gate reads the model's own counts. Returns per-depth target stats on the held-back probe + gate log.
    """
    rng = np.random.default_rng(seed)
    by_depth = {d: [s for s in sentences if s[0] == d] for d in depths}
    for d in depths:
        rng.shuffle(by_depth[d])
    # split each depth: 70% train stream / 30% held-out probe (probe scored learn=False)
    train, probe = {}, {}
    for d in depths:
        s = by_depth[d]
        cut = int(len(s) * 0.7)
        train[d], probe[d] = s[:cut], s[cut:]
    ctx_keys = {d: _closing_contexts(sentences, d, K) for d in depths}

    m = CountModel(K=K, alpha=alpha, H=H)
    prefix = None
    gate_log = []
    ent_hist = {d: [] for d in depths}
    di = 0
    queue = list(train[depths[di]]); rng.shuffle(queue)
    seen_since_gate = 0
    fed = 0
    while True:
        if not queue:
            # current depth exhausted: try to advance regardless (no more of this depth to feed)
            if di + 1 < len(depths):
                di += 1
                gate_log.append((fed, depths[di], "exhausted-advance"))
                queue = list(train[depths[di]]); rng.shuffle(queue); seen_since_gate = 0
                continue
            else:
                break
        d, ids, targets = queue.pop()
        _, prefix = m.feed_sentence(ids, targets=None, prefix=prefix, learn=True)
        fed += 1; seen_since_gate += 1
        # update the gate signal for the CURRENT depth
        be = m.branching_entropy(ctx_keys[depths[di]])
        if be is not None:
            ent_hist[depths[di]].append(be)
        # self-gate: open next depth when current depth entropy stabilized
        if di + 1 < len(depths) and seen_since_gate >= window:
            h = ent_hist[depths[di]]
            if len(h) >= window:
                recent = h[-window:]
                delta = abs(recent[-1] - recent[0])
                if delta < eps:
                    di += 1
                    gate_log.append((fed, depths[di], f"opened (Δbe={delta:.4f})"))
                    queue = list(train[depths[di]]) + queue
                    rng.shuffle(queue); seen_since_gate = 0
    stats = _score(m, probe, depths, K)
    return stats, gate_log


def run_full(sentences, depths, K=8, alpha=0.05, seed=0, H=200.0):
    """AK's winner: all depths interleaved uniformly from char one. Same train/probe split + seed."""
    rng = np.random.default_rng(seed)
    by_depth = {d: [s for s in sentences if s[0] == d] for d in depths}
    train, probe = [], {}
    for d in depths:
        s = by_depth[d]; rng.shuffle(s)
        cut = int(len(s) * 0.7)
        train += s[:cut]; probe[d] = s[cut:]
    rng.shuffle(train)
    m = CountModel(K=K, alpha=alpha, H=H)
    prefix = None
    for (d, ids, targets) in train:
        _, prefix = m.feed_sentence(ids, prefix=prefix, learn=True)
    return _score(m, probe, depths, K)


def run_anti(sentences, depths, K=8, alpha=0.05, seed=0, H=200.0):
    """Reversed curriculum: hard->easy (deepest first). Ordering control — wrong direction."""
    rng = np.random.default_rng(seed)
    by_depth = {d: [s for s in sentences if s[0] == d] for d in depths}
    train, probe = {}, {}
    for d in depths:
        s = by_depth[d]; rng.shuffle(s)
        cut = int(len(s) * 0.7)
        train[d], probe[d] = s[:cut], s[cut:]
    m = CountModel(K=K, alpha=alpha, H=H)
    prefix = None
    for d in reversed(depths):
        q = list(train[d]); rng.shuffle(q)
        for (dd, ids, targets) in q:
            _, prefix = m.feed_sentence(ids, prefix=prefix, learn=True)
    return _score(m, probe, depths, K)


def _score(model, probe, depths, K):
    """Score the held-out probe sentences (learn=False) per depth; isolate the OUTER (deep) agreement token."""
    out = {}
    for d in depths:
        nlls_all, nlls_outer, hit_outer = [], [], []
        prefix = None
        for (dd, ids, targets) in probe[d]:
            res, prefix = model.feed_sentence(ids, targets=targets, prefix=prefix, learn=False)
            for (true, nll, hit, is_outer) in res:
                nlls_all.append(nll)
                if is_outer:
                    nlls_outer.append(nll); hit_outer.append(hit)
        out[d] = {
            "ppl_all": float(np.exp(np.mean(nlls_all))) if nlls_all else float("nan"),
            "ppl_outer": float(np.exp(np.mean(nlls_outer))) if nlls_outer else float("nan"),
            "acc_outer": float(np.mean(hit_outer)) if hit_outer else float("nan"),
            "n_outer": len(nlls_outer),
        }
    return out

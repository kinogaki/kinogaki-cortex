"""deliberate.py — Exp AL: a MULTI-STEP serial workspace over explicit concept-slots. ONLINE, NO backprop.

Exp AG built a count-native System 2 — a confidence+conflict GATE plus a capacity-4 serial workspace
(inhibition-of-return / cognitive decoupling / a leaky-accumulator race). The verdict was split: the GATE
wins the Engle signature cleanly, but the elaborate serial WORKSPACE did NOT beat a trivial one-step
"defer to the wider context". The honest reason (AG's own diagnosis): a char next-token probe is a
ONE-STEP decision — there is nothing to hold and manipulate across cycles, so a single deferral reaches
the answer and the workspace's serial machinery is dead weight. AG parked the workspace (Fragile-Ideas
§7/§8) and named its untested winning axis: *multi-step problems where a single deferral can't reach the
answer*.

This module gives the workspace that fair test. The probe (built in exp_al_multistep) is a 2-HOP
RELATIONAL inference over explicit, slot-addressable concepts: from a count-derived relation R we are
told R(X)=Y and R(Y)=Z implicitly (the relation is a function over concept ids), and asked to resolve
R(R(X)). A one-step operator can apply R once — it lands on Y, the WRONG answer (the classic prepotent
trap: the near, salient associate). Only a workspace that HOLDS the intermediate concept Y in a slot,
then APPLIES the relation operator a SECOND time, reaches Z. That is the manipulation AG's workspace
never got to do.

Everything is counts + leaky accumulators, serial, bounded:

  Relation R as counts. A relation is a count-derived map id->id (e.g. "the strongest associate", or a
  one-hop edge in a co-occurrence graph). It is a lookup table, no gradient. apply_relation() reads it.

  The serial workspace (reused machinery from System 2, generalized to >1 step). A bounded FOCUS of
  capacity k holds concept SLOTS (handles to explicit concept ids — the redescription layer's parts, not
  raw chars). Each serial CYCLE runs one OPERATOR — here APPLY-RELATION: take the slot currently in the
  focus of attention, look up R, write the result into a fresh slot, and shift attention to it
  (Oberauer's focus moves; the just-used slot gets inhibition-of-return so we don't loop on it). After
  >=2 cycles the focus holds the composed concept R(R(X)). A CONFIDENCE+CONFLICT gate decides whether to
  spend cycles at all (cheap problems ship the one-step answer); a STEP BUDGET bounds the chain; and
  SUPPRESS-NOT-ERASE keeps the one-step answer live so a budget-0 / unknown-relation fallback is graceful.

  The three contestants this module exposes (scored against each other in run.py):
    system1_answer        — the prepotent associate of X (no relation applied, no workspace): the salient
                            distractor. (The fast field's argmax.)
    one_step_answer       — apply the relation ONCE: R(X)=Y. The trivial "defer to the wider context"
                            operator that WON in AG on the one-step probe. On a 2-hop query it lands on
                            the intermediate, not the target.
    multistep_answer      — the serial workspace: hold X, apply R, hold the result, apply R again, read
                            the focus. Reaches R(R(X))=Z when the chain is resolvable; falls back
                            gracefully (to the one-step or System-1 answer) when it is not.

Alphabet/ids: concept ids are dense word-ids (top-N), the same ids redescribe.py/constructions.py use.
The relation table and the gate fields are built once from online counts; the deliberate pass is a
per-query serial loop over them — no second training pass, no labels, no gradient.
"""
import numpy as np


# ── relations as counts: a map concept-id -> concept-id, plus its confidence ─────────────────────────

def build_relation(frame_counts, N, min_token=20):
    """A count-derived RELATION R: concept -> its single strongest successor concept (the prepotent
    next-associate), built from the same per-frame filler counts the construction grammar uses.

    frame_counts : {frame_id -> (filler_ids, counts)} from constructions.build_frame_counts(order=1).
    Returns
      nxt   : (N,) int — R(x) = argmax filler after x  (-1 if x has no ripe frame).
      conf  : (N,) float — P(R(x) | x) = the dominant filler's share (the relation's per-source confidence).
      seen  : (N,) bool — x has a ripe frame (>= min_token tokens) so R(x) is defined.
    This is a pure lookup table — counting only, no gradient. It is intentionally the SAME 'strongest
    associate' the prepotent System-1 voter would pick, so applying R once == the one-step deferral."""
    nxt = -np.ones(N, np.int64)
    conf = np.zeros(N, np.float64)
    seen = np.zeros(N, bool)
    for fk, (fids, cnt) in frame_counts.items():
        if fk < 0 or fk >= N:
            continue
        tot = float(cnt.sum())
        if tot < min_token:
            continue
        i = int(cnt.argmax())
        w = int(fids[i])
        if 0 <= w < N:
            nxt[fk] = w
            conf[fk] = float(cnt[i]) / tot
            seen[fk] = True
    return nxt, conf, seen


def build_content_relation(seq, N, stop, window=4, min_token=20, seed=0):
    """A count-derived CONTENT relation R: concept -> its strongest CONTENT co-associate within ±window,
    excluding stop concepts as TARGETS. The raw 'strongest successor' bigram relation collapses to function
    words (the/of/and — the known leader-clustering artefact), so chains over it are contentless. This builds
    the relation the probe actually wants: a co-occurrence-derived SEMANTIC edge over content concepts.

    seq    : dense top-id stream (-1 = OOV).
    stop   : (N,) bool — concepts excluded as relation TARGETS (the function-word floor). Sources may be any
             content concept; we just refuse to point AT a stopword (so Y and Z are real concepts).
    window : ±window co-occurrence (bidirectional) — the relation is 'most associated content neighbour'.
    Returns (nxt, conf, seen) exactly like build_relation. Pure counting (np.add.at over offsets) — online,
    order-independent, no gradient. PMI-style: down-weight a target by its global frequency so ubiquitous
    (but non-stop) words don't dominate; the relation keeps the DISCRIMINATIVE associate."""
    freq = np.bincount(seq[seq >= 0], minlength=N).astype(np.float64) + 1.0
    # accumulate weighted (source, target) co-occurrence counts, vectorized by np.unique over packed keys
    # (batched order-independent accumulation = identical to a token-at-a-time online update).
    src_l = []; tgt_l = []; wgt_l = []
    for g in range(1, window + 1):
        for a, b in ((seq[:-g], seq[g:]), (seq[g:], seq[:-g])):
            m = (a >= 0) & (b >= 0)
            wa, wb = a[m], b[m]
            keep = ~stop[wb]                             # never point AT a stop concept
            wa, wb = wa[keep], wb[keep]
            if wa.size == 0:
                continue
            src_l.append(wa.astype(np.int64)); tgt_l.append(wb.astype(np.int64))
            # PMI-ish weight: a co-occurrence counts more if the target is globally rarer (discriminative).
            wgt_l.append(1.0 / np.log(2.0 + freq[wb]))
    src = np.concatenate(src_l); tgt = np.concatenate(tgt_l); wgt = np.concatenate(wgt_l)
    key = src * N + tgt
    order = np.argsort(key, kind="stable")
    key, wgt, src, tgt = key[order], wgt[order], src[order], tgt[order]
    uk, idx = np.unique(key, return_index=True)
    sums = np.add.reduceat(wgt, idx)                     # summed weight per (source,target)
    us = uk // N; ut = (uk % N).astype(np.int64)
    nxt = -np.ones(N, np.int64); conf = np.zeros(N, np.float64); seen = np.zeros(N, bool)
    src_tot = np.zeros(N, np.float64)
    np.add.at(src_tot, us, sums)
    # per source, the argmax-weight target (uk is sorted by source then target, so scan runs of equal source)
    edges = np.nonzero(np.diff(us))[0] + 1
    starts = np.concatenate([[0], edges]); ends = np.concatenate([edges, [len(us)]])
    for a_, b_ in zip(starts, ends):
        s = int(us[a_])
        if freq[s] - 1.0 < min_token or src_tot[s] < 1.0:
            continue
        j = a_ + int(sums[a_:b_].argmax())
        nxt[s] = int(ut[j]); conf[s] = float(sums[j] / src_tot[s]); seen[s] = True
    return nxt, conf, seen


def apply_relation(x, nxt):
    """One hop: R(x). Returns -1 if undefined (graceful — the caller floors to a default)."""
    if x < 0 or x >= len(nxt):
        return -1
    return int(nxt[x])


# ── the gate (reused from System 2): decide whether to spend serial cycles ────────────────────────────

def should_deliberate(x, conf, seen, nxt, theta=0.5):
    """Default-interventionist trigger, count-native. Deliberate (spend cycles) on a query rooted at x
    when the FIRST hop is confident enough to be worth chaining AND a second hop exists — i.e. there is a
    real multi-step structure to manipulate. If R(x) is unseen or the chain dies after one hop, there is
    nothing to deliberate over: ship the one-step answer (graceful). Returns (fire, reason)."""
    if x < 0 or not seen[x]:
        return False, "no-relation"          # nothing to apply; fall back to System 1
    y = int(nxt[x])
    if y < 0 or not seen[y]:
        return False, "one-hop-only"          # chain dies after one hop; one-step IS the answer
    if conf[x] < theta:
        return False, "low-confidence"        # first hop untrustworthy; don't build on sand
    return True, "deliberate"


# ── the serial multi-step workspace: hold a concept-slot, apply the operator, shift attention ─────────

def multistep(x, nxt, conf, seen, k=4, budget=4, ior=0.7, floor=0.02, hops=2):
    """Run the deliberate, capacity-bounded, SERIAL workspace on ONE query rooted at concept x.

    The job: compose the relation R hops-many times — reach R(R(x)) for hops=2 — by HOLDING the running
    concept in the focus of attention and re-applying the APPLY-RELATION operator each cycle. This is the
    manipulation a one-step deferral cannot do.

    x      : the query's root concept id.
    nxt    : the relation table R (concept -> concept).
    conf   : per-source relation confidence (used to credit the chain + floor the default).
    seen   : which concepts have a defined R.
    k      : FOCUS capacity (Oberauer/Cowan ~4) — the workspace holds at most k concept-slots; older slots
             age out (bounded memory).
    budget : STEP BUDGET — max serial cycles. budget=0 ⇒ no deliberation ⇒ graceful one-step/System-1.
    ior    : inhibition-of-return — once a slot has been the operand it is suppressed so attention ADVANCES
             to the freshly-written slot instead of re-applying R to the same concept (the loop must move).
    floor  : suppress-not-erase — the one-step answer R(x) keeps a floor of activation so if the chain
             dies mid-way we still ship it, never an empty answer.
    hops   : how many times to compose R (the probe's required depth; 2 = the 2-hop query).

    Returns (answer_id, cycles_used, status). status ∈ {'composed','partial','fallback'}:
      composed  — the chain resolved all `hops` applications; answer = R^hops(x).
      partial   — the chain died after >=1 hop; answer = the last good concept reached (graceful, the
                  suppress-not-erase default = the one-step answer).
      fallback  — nothing applied (budget 0 or x undefined); answer = x's one-step/System-1 default.
    """
    one_step = apply_relation(x, nxt)                       # R(x): the prepotent one-step answer (the default)
    default = one_step if one_step >= 0 else int(x)
    if budget <= 0 or x < 0 or not seen[x]:
        return default, 0, "fallback"

    # the focus of attention: a bounded list of concept-SLOTS with leaky activations. We seed it with the
    # root x (fully active — it is what we are attending) and a suppress-not-erase trace of the default.
    slots = [int(x)]                                        # concept ids currently held
    act = {int(x): 1.0}                                     # leaky activation per held slot
    if default not in act:
        act[default] = floor                               # the one-step answer stays live (suppress-not-erase)
        slots.append(default)
    attended = int(x)                                      # the slot currently in the focus of attention
    reached = int(x)                                       # the deepest concept the chain has reached so far
    depth = 0                                              # how many hops composed so far
    cycles = 0
    status = "fallback"

    for _ in range(budget):
        cycles += 1
        # SERIAL OPERATOR = apply-relation to the attended slot. (One operator per cycle — the workspace
        # bottleneck: System 2 can only manipulate the single item in the focus of attention.)
        y = apply_relation(attended, nxt)
        if y < 0:
            status = "partial" if depth >= 1 else "fallback"
            break
        # write the result into a fresh concept-slot; it becomes the new focus of attention.
        if y not in act:
            slots.append(y)
            act[y] = 0.0
        # the chain CREDITS the freshly-reached concept by the hop's confidence (count-derived evidence
        # that this composition is supported) — the time-integrated selection signal, as in AG's race.
        act[y] += float(conf[attended]) if attended < len(conf) else 0.5
        # inhibition-of-return on the just-used operand: suppress it so attention does not snap back and
        # re-apply R to the same concept (the loop must ADVANCE to the new slot).
        act[attended] *= (1.0 - ior)
        reached = y
        depth += 1
        attended = y                                        # shift the focus to the freshly-composed concept
        # CAPACITY BOUND: keep only the k most-active slots (older intermediates age out of working memory).
        if len(slots) > k:
            slots = sorted(slots, key=lambda s: act[s], reverse=True)[:k]
            act = {s: act[s] for s in slots}
            if attended not in act:                          # never drop the slot we're attending
                attended = max(act, key=act.get)
                reached = attended
        if depth >= hops:
            status = "composed"
            break
    else:
        # budget exhausted before reaching `hops`: graceful partial (we got somewhere, just not all the way).
        status = "composed" if depth >= hops else ("partial" if depth >= 1 else "fallback")

    if status == "composed":
        answer = reached
    elif status == "partial":
        # the chain stalled: ship the deepest concept we reached (>= the one-step answer). Suppress-not-erase
        # guarantees this is never worse than the default.
        answer = reached if reached != int(x) else default
    else:
        answer = default
    return int(answer), cycles, status


# ── the three contestants, vectorized over a batch of queries ─────────────────────────────────────────

def answer_batch(xs, nxt, conf, seen, k=4, budget=4, ior=0.7, floor=0.02, hops=2, theta=0.5):
    """Score the three contestants on a batch of query-root concepts `xs` (each needs R(R(x)) resolved).

    Returns dict of (m,) int answer arrays + bookkeeping:
      s1       : System-1 prepotent answer = R(x) is NOT applied; the salient associate IS the trap, so
                 System-1's 'answer' for a 2-hop query is just to emit the most-associated concept of x,
                 which for this probe we take as R(x)'s competitor... see run.py for how the probe frames
                 the prepotent distractor. Here s1 = x's strongest associate = R(x) (the one-step result is
                 ALSO what a System-1 reader would blurt). We expose the one-step explicitly below.
      one_step : R(x) — apply the relation once (the trivial deferral that won in AG).
      multi    : the serial workspace's R(R(x)).
      cycles   : (m,) serial cycles each multi-step query used.
      status   : (m,) status strings.
      fired    : (m,) bool — did the gate choose to deliberate.
    """
    m = len(xs)
    one_step = np.array([apply_relation(int(x), nxt) for x in xs], np.int64)
    multi = one_step.copy()
    cycles = np.zeros(m, np.int64)
    status = np.array(["fallback"] * m, dtype=object)
    fired = np.zeros(m, bool)
    for i, x in enumerate(xs):
        x = int(x)
        go, _ = should_deliberate(x, conf, seen, nxt, theta=theta)
        fired[i] = go
        if go:
            ans, cy, st = multistep(x, nxt, conf, seen, k=k, budget=budget,
                                    ior=ior, floor=floor, hops=hops)
            multi[i] = ans; cycles[i] = cy; status[i] = st
        else:
            # gate declined: the multi-step model degrades to the one-step answer (graceful).
            multi[i] = one_step[i]; cycles[i] = 0; status[i] = "fallback"
    return dict(one_step=one_step, multi=multi, cycles=cycles, status=status, fired=fired)

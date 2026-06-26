"""system2.py — Exp AG: a count-native SYSTEM 2 (the deliberate pass).

Everything built so far is SYSTEM 1: fast, parallel, associative — the char-order experts vote, a
leader pops out, we ship it. That is the prepotent, local-argmax response. The cognitive-science
convergence (Evans/Stanovich two-process theory, Botvinick conflict monitoring, Engle working-memory-
capacity, Oberauer's focus of attention, Kahneman's default-interventionist arrangement, Sloman's
suppress-not-erase) says SYSTEM 2 is NOT a separate smarter model — it is a small, capacity-bounded,
SERIAL workspace that is *triggered by conflict*, decouples from the immediate input, and OVERRIDES
System 1 only when System 1 is wrong. The Engle signature: the override must help where the prepotent
answer is wrong and do no harm where there is no conflict.

We build that out of counts only — no gradient, no batch optimization. The pieces:

  SYSTEM 1 (the default). A product-of-experts over SHORT char orders (the fast, local, prepotent
  voter). Its leader is the local argmax; we read its top-two leaky-accumulator vote activations.

  THE DUAL TRIGGER (default-interventionist). At each position compute:
    (a) calibrated confidence c — the NARS f·c of the deciding context (lib/confidence), how much we
        should trust the fast answer;
    (b) conflict — Botvinick's energy: HIGH only when the top-TWO answers are BOTH strongly on
        (product / min of the top-two normalized vote activations). One dominant answer ⇒ ~0 conflict;
        two co-active answers ⇒ high conflict.
  Deploy System 2 when  c < theta  OR  conflict > kappa.  Otherwise emit the System-1 leader at once.

  THE DELIBERATE PASS (decoupled, serial, capacity-bounded). A FOCUS of capacity k≈4 holds handles to
  the most active candidate continuations (Oberauer's focus / Cowan's 4±1). Selection is a RACE over
  leaky accumulators seeded from System-1's votes PLUS a top-down GOAL bias added into the race — the
  wider-context (long-order) expert's opinion, which the fast local voter under-weights. Each serial
  CYCLE (one operator) picks the current racer, reads its evidence, and applies INHIBITION-OF-RETURN
  (decrement it) so the loop advances. We run up to a STEP BUDGET of cycles. SUPPRESS-NOT-ERASE: the
  System-1 leader's accumulator is floored above zero, never deleted, so if the budget runs out we
  still ship a graceful answer. The override COMMITS only if the deliberate winner beats the System-1
  default (else we keep the default — System 2 cannot make an un-conflicted case worse).

Alphabet matches lib/confidence / lib/cortex: a..z = 0..25, space = 26, V = 27. Single causal pass:
the System-1 tables are the online (w+,w-) counts from confidence.CountTruth; the deliberate pass is a
per-position serial loop over those same counts — no second training pass, no labels beyond next char.
"""
import numpy as np

V = 27


# ── System 1: a short-order product-of-experts with readable vote activations ──────────────────────

def system1_votes(order_lds_short):
    """Fast/parallel System-1 vote field. Sum the SHORT-order log-dists (product of experts), exponentiate
    to a per-position activation over the 27 next-chars, normalize. Returns (m,27) vote activations a in
    [0,1] summing to 1 per row — these are the leaky accumulators System 1 hands to System 2."""
    s = np.zeros_like(order_lds_short[0])
    for ld in order_lds_short:
        s = s + ld
    s = s - s.max(1, keepdims=True)
    a = np.exp(s)
    a = a / a.sum(1, keepdims=True)
    return a


def top_two(a):
    """Per row, the (top-1 id, top-1 activation, top-2 id, top-2 activation)."""
    idx = np.argsort(a, axis=1)[:, ::-1]
    rows = np.arange(a.shape[0])
    i1 = idx[:, 0]; i2 = idx[:, 1]
    a1 = a[rows, i1]; a2 = a[rows, i2]
    return i1, a1, i2, a2


def conflict_energy(a1, a2):
    """Botvinick conflict / Hopfield energy WITHIN System 1: HIGH only when its top two are BOTH strongly
    on. Normalize the pair to (p1,p2) and take 4*p1*p2 ∈ [0,1] (max 1 at a 50/50 tie, →0 when one answer
    dominates). The 'two answers fighting' signal — a single confident leader ⇒ ~0."""
    s = a1 + a2 + 1e-12
    p1 = a1 / s; p2 = a2 / s
    return 4.0 * p1 * p2


def cross_conflict(a, goal):
    """Botvinick conflict ACROSS subsystems: the fast System-1 leader and the reflective top-down GOAL
    leader point different ways, and BOTH are strongly on. Per row: g_on = goal mass on System-1's leader
    vs goal's own leader. Conflict is high when System-1 is confident in X while the goal is confident in
    Y≠X — exactly the case where a deliberate check is worth its cost. Returns (m,) in [0,1]."""
    i1 = a.argmax(1)
    gl = goal.argmax(1)
    rows = np.arange(a.shape[0])
    a_lead = a[rows, i1]                          # System-1 confidence in its leader
    g_lead = goal[rows, gl]                        # goal confidence in its leader
    disagree = (i1 != gl).astype(float)
    # both subsystems strongly committed AND they disagree
    return disagree * np.sqrt(a_lead * g_lead)


# ── the deliberate pass: a serial, capacity-bounded race with inhibition-of-return ─────────────────

def deliberate(a_row, goal_row, default_id, k=4, budget=6, ior=0.6, goal_gain=1.0, floor=0.02,
               decouple=0.6):
    """Run System 2 on ONE position. Pure counts + leaky accumulators, serial.

    a_row     : (27,) System-1 vote activations (the fast, local, prepotent field).
    goal_row  : (27,) the top-down GOAL/context bias — the wider (long-order) expert's normalized opinion,
                what the fast voter under-weights. Added into the race (top-down increments).
    default_id: System-1's prepotent leader (the answer we ship if we never beat it).
    k         : FOCUS capacity (Oberauer/Cowan ~4) — only the k most active candidates are decoupled into
                the workspace; everything else is out of mind.
    budget    : STEP BUDGET — number of serial deliberate CYCLES. budget=0 ⇒ ship the default (graceful).
    ior       : inhibition-of-return — fraction by which a candidate is decremented after it's visited,
                so the serial loop advances instead of locking onto the leader.
    goal_gain : how hard top-down context drives the race.
    floor     : suppress-not-erase floor — the default's accumulator never falls below this (Sloman:
                System 1's answer stays live, just suppressed).

    Returns (committed_id, used_system2). The committed id is the deliberate winner ONLY if it strictly
    beats the default's final accumulator; otherwise the default (the override is conservative)."""
    if budget <= 0:
        return default_id, False

    # decouple: bring the k most active candidates into the bounded FOCUS (a few handles, not all 27).
    focus = list(np.argsort(a_row)[::-1][:k])
    did = int(default_id)
    if did not in [int(c) for c in focus]:              # the default is always kept live in the workspace
        focus.append(did)
    cand = [int(c) for c in focus]
    # leaky accumulators, seeded from System-1's fast vote (the prepotent field).
    acc = {c: float(a_row[c]) for c in cand}
    # standing top-down GOAL drive (the wider-context opinion) per candidate.
    goal = {c: float(goal_row[c]) for c in cand}
    # time-integrated SELECTION evidence: how much sustained support each candidate gathered across the
    # serial cycles. This — not the instantaneous accumulator that IOR keeps knocking down — is the
    # deliberate verdict (Botvinick/Usher–McClelland: the racer that crosses threshold most over time).
    won = {c: 0.0 for c in cand}

    for _ in range(budget):
        # cognitive decoupling (Stanovich): the prepotent System-1 seed FADES as deliberation proceeds,
        # so the sustained top-down GOAL drive — not the first impression — decides a long race.
        for c in cand:
            acc[c] *= decouple
            acc[c] += goal_gain * goal[c]               # top-down drive: the wider context speaks
        # suppress-not-erase: floor the default so inhibition can never delete it.
        if acc[did] < floor:
            acc[did] = floor
        # serial operator: select the current racer (one item per cycle — the workspace bottleneck).
        cur = max(acc, key=acc.get)
        won[cur] += acc[cur]                            # credit the selected racer's current strength
        # inhibition-of-return: decrement the visited racer so the loop advances next cycle.
        acc[cur] *= (1.0 - ior)
        if cur == did and acc[cur] < floor:
            acc[cur] = floor

    # the deliberate winner = the candidate with the most time-integrated selection evidence. Commit it
    # only if it strictly beats the default (the no-harm rule: System 2 can't worsen an un-conflicted
    # call). The default always has its suppress-not-erase floor of selection mass.
    won[did] = max(won[did], floor)
    winner = max(won, key=won.get)
    if winner != did and won[winner] > won[did]:
        return winner, True
    return did, False


# ── the gated model: System 1 by default, System 2 on conflict, vectorized harness ─────────────────

def gated_predict(a, goal, c_conf, theta, kappa, k=4, budget=6, ior=0.6, goal_gain=1.0, floor=0.02,
                  decouple=0.6, kappa_x=None):
    """Run the full default-interventionist model over all m positions.

    a       : (m,27) System-1 vote activations.
    goal    : (m,27) top-down context bias (normalized long-order opinion).
    c_conf  : (m,) calibrated confidence of the deciding context (NARS f·c).
    theta   : confidence floor — below it, deliberate.
    kappa   : within-System-1 conflict ceiling — above it, deliberate.
    kappa_x : cross-subsystem conflict ceiling (fast leader vs goal leader disagree, both strong).
              None ⇒ off.

    Returns (pred_ids[m], fired_mask[m], override_mask[m]). pred is the committed next-char id.
    fired = the gate deployed System 2; override = System 2 actually changed the answer."""
    i1, a1, i2, a2 = top_two(a)
    conflict = conflict_energy(a1, a2)
    fire = (c_conf < theta) | (conflict > kappa)
    if kappa_x is not None:
        fire = fire | (cross_conflict(a, goal) > kappa_x)
    m = a.shape[0]
    pred = i1.copy()
    override = np.zeros(m, bool)
    fired = fire.copy()
    for t in np.nonzero(fire)[0]:
        cid, used = deliberate(a[t], goal[t], int(i1[t]), k=k, budget=budget,
                               ior=ior, goal_gain=goal_gain, floor=floor, decouple=decouple)
        pred[t] = cid
        override[t] = used
    return pred, fired, override


# ── building the System-2 inputs from confidence.CountTruth tables ─────────────────────────────────

def context_confidence(hi_table, ctx_hi):
    """Calibrated confidence of the deciding (high-short-order) context: NARS f·c (the c-discounted
    frequency from lib/confidence). Unseen context ⇒ 0 (maximally untrusted ⇒ always deliberate)."""
    from confidence import truth_of
    rows = hi_table.lookup(ctx_hi)
    m = len(ctx_hi)
    out = np.zeros(m)
    seen = rows >= 0
    if seen.any():
        r = rows[seen]
        f, c = truth_of(hi_table.wp[r], hi_table.wm[r])
        out[seen] = f * c
    return out


def goal_field(order_lds_long):
    """The top-down GOAL/context bias: the wider (long-order) experts' product-of-experts opinion,
    normalized per row to (m,27). This is the 'broader context' the fast local voter under-weights."""
    s = np.zeros_like(order_lds_long[0])
    for ld in order_lds_long:
        s = s + ld
    s = s - s.max(1, keepdims=True)
    g = np.exp(s)
    g = g / g.sum(1, keepdims=True)
    return g

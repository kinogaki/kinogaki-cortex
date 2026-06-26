#!/usr/bin/env python3
"""Exp BE — Contingency-gated learning rate (the temporal-contingency dial + the yoked ablation).

A child does not learn equally from every word in the air. A reply that *answers* you — one that lands
while your own utterance is still warm — teaches more than the identical words overheard from across the
room. Goldstein & Schwade showed this with the cleanest possible control: babies given **contingent**
caregiver feedback advanced their babbling; babies given the **yoked** feedback (the exact same sounds,
re-timed to ignore them) did not. Same input, different *contingency*. That is the dial G2 installs.

The mechanism (lib/contingency.py): in `observe()`, multiply each count increment by a soft gain
`g = exp(−Δt/τ)`, Δt = steps since the agent last spoke, into two bounded registers — `tab_hot` (warm
replies, loud) + `tab_cold` (background, never silenced). Prediction pools both. ONLINE, BOUNDED, no
backprop. The dial only ever *adds* emphasis to contingent tokens; cold input always still updates.

The world here makes timing carry information, the way a conversation does. The agent lives in a stream
that ALTERNATES: right after it speaks, the teacher replies with a chunk of REAL ENGLISH (the signal we
want it to learn — held-out bpc is measured on held-out English). In the gaps, the room is full of
DISTRACTOR babble — same alphabet, scrambled, no structure. With the contingency clock running, the
real-English replies arrive WARM (small Δt → big g) and the babble arrives COLD; the dial up-weights the
signal and down-weights the noise *for free, from timing alone* — no content label, no supervision.

The load-bearing control, registered BEFORE the run: **YOKED**. Identical token stream, identical loop,
but g is drawn from the SCRAMBLED timeline (same multiset of gains, re-paired to the wrong chunks). Same
tokens, same total count mass — only the timing→content alignment is destroyed. Plus two more baselines:
PASSIVE (gain≡1, single table — the AT floor, no dial) and an idealized CONTENT-ORACLE ceiling.

KILL (from BUILD_QUEUE / G2): contingency-ON matches YOKED on bpc AND on the turn-overlap contingency
metric, at matched tokens → the dial is inert; surface the honest negative. FRAGILE: sweep τ and the
hot/cold weights (≥10 variations) and check turn-overlap before declaring death.

Online throughout; Columns + calibrated vote; no gradients/k-means/SVD. lib/contingency.py + this file.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import harness as H
import metrics
import contingency
from cortex import V0
from corpus import load_ids, ids_to_str

SEED = 0
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

# ─────────────────────────── corpus: real English signal + a babble distractor ───────────────────────────
# SUBSTITUTION (said up front): G2's spec names "CDS/dialogue" (child-directed speech, e.g. CHILDES).
# CHILDES is not in data/. We substitute text8 (clean lowercase Wikipedia) as the structured "teacher
# reply" signal, and a frequency-matched SCRAMBLE of the same alphabet as the non-contingent babble — so
# the ONLY difference between warm and cold input is structure, isolating the dial. The mechanism is
# corpus-agnostic; the contingency claim is about timing, not about this particular text.
N = 1_500_000
ids = load_ids("text8", nbytes=N)
signal = np.asarray(ids[:-40_000], dtype=np.int64)     # the real-English stream the replies are drawn from
held = ids_to_str(ids[-40_000:])                       # held-out REAL English — what we score bpc on
print(f"signal {len(signal):,} chars | held-out {len(held):,} chars | alphabet {V0}")

# a babble distractor: the marginal char distribution of English, but i.i.d. (no n-gram structure)
counts = np.bincount(signal, minlength=V0).astype(float); pmarg = counts / counts.sum()


def babble(n):
    return RNG.choice(V0, size=n, p=pmarg)


# ─────────────────────────── the contingent conversation, as a fixed transcript ───────────────────────────
# We build ONE transcript of (chunk_ids, is_signal) so every condition sees the EXACT same tokens in the
# EXACT same order — the conditions differ only in how the dial weights them. Pattern per "turn":
#   the agent speaks  →  WARM real-English reply (signal)  →  a run of COLD babble (the noisy room).
# This is the reactive contract made explicit: the warm slot is contingent on the emission; babble is not.
CHUNK = 96          # chars per reply / per babble burst
N_BABBLE = 2        # babble bursts between replies (the room is mostly noise)
N_TURNS = 700

def build_transcript(reliability=1.0, rng=None):
    """One conversation as a list of (ids, is_warm_slot). The WARM slot (right after the agent speaks) is
    real English with probability `reliability`, else babble; the COLD gap slots are babble with the same
    probability (so an UNRELIABLE world leaks signal into cold slots too). reliability=1.0 = perfect
    contingency (the idealized split); lower = realistic, noisy contingency where timing is only a *cue*.
    `is_warm_slot` marks the slot's TIMING (warm/cold), NOT whether it is signal — that's the point."""
    rng = rng or np.random.default_rng(SEED)
    t = []; sp = 0
    for _ in range(N_TURNS):
        if sp + CHUNK >= len(signal):
            break
        # warm slot: signal w.p. reliability
        if rng.random() < reliability:
            t.append((signal[sp:sp + CHUNK].copy(), True)); sp += CHUNK
        else:
            t.append((babble(CHUNK), True))
        # cold gap slots: signal LEAKS in w.p. (1-reliability)
        for _ in range(N_BABBLE):
            if rng.random() < (1.0 - reliability) and sp + CHUNK < len(signal):
                t.append((signal[sp:sp + CHUNK].copy(), False)); sp += CHUNK
            else:
                t.append((babble(CHUNK), False))
    return t


transcript = build_transcript(reliability=1.0)
n_signal_tok = sum(len(c) for c, s in transcript if s)
n_total_tok = sum(len(c) for c, s in transcript)
print(f"transcript {len(transcript)} chunks | signal {n_signal_tok:,} / {n_total_tok:,} tokens "
      f"({100*n_signal_tok/n_total_tok:.0f}% structured, rest babble)")


# ─────────────────────────── the driver: one transcript, one dial setting ───────────────────────────
# We drive observe() directly so the contingency CLOCK is explicit and identical across conditions. The
# agent "speaks" before each warm reply (Δt→0); babble bursts cool the clock. For YOKED we first record
# the real gain timeline, scramble it, and replay the SAME tokens with the scrambled gains.

def real_run(agent, tr=None):
    """Contingency-ON: emit before each WARM slot; let the clock decay over the cold gaps. Returns the
    realized per-chunk gain timeline (for building the yoked control)."""
    tr = transcript if tr is None else tr
    gains = []
    for chunk_ids, is_warm in tr:
        if is_warm:
            agent.act(1)                                # the agent spoke → its reply is warm (Δt→0)
        g = float(np.exp(-agent.dt / agent.tau))
        gains.append(g)
        agent.observe(chunk_ids)                        # counted with the real gain inside observe()
    return gains


def yoked_run(agent, scrambled_gains, tr=None):
    """YOKED: identical tokens, identical loop, but the gain for each chunk is INJECTED from the scrambled
    timeline (timing→content alignment destroyed). Same multiset of gains, same total mass."""
    tr = transcript if tr is None else tr
    for (chunk_ids, _), g in zip(tr, scrambled_gains):
        agent.observe(chunk_ids, gain_override=g)


def passive_run(agent, tr=None):
    """The AT floor: no dial. Single table, every increment weight 1 — read everything equally."""
    tr = transcript if tr is None else tr
    for chunk_ids, _ in tr:
        agent.observe(chunk_ids, gain_override=None)    # PassiveAgent ignores the dial entirely


# ─────────────────────────── turn-overlap: the contingency metric (NOT bpc) ───────────────────────────
# Reported separately from bpc, per the spec. After learning, prompt with the tail of a held-out REAL
# reply and let the agent continue; turn-overlap = fraction of generated chars that the model assigns
# above-chance probability under the held-out continuation — i.e. how well it tracks the structured
# (contingent) channel vs the babble it also heard. We measure it as: mean prob the model puts on the
# TRUE next char of held-out English, minus mean prob it puts on babble's next char. A model that locked
# onto the contingent signal scores high; one that drowned in babble scores ~0.
def turn_overlap(agent):
    real_lp, babb_lp = [], []
    probe = held[:8000]
    pb = RNG.choice(V0, size=len(probe), p=pmarg)
    for t in range(8, len(probe)):
        p = agent.dist(probe[max(0, t - 16):t])
        real_lp.append(p[H.CharCodec().encode(probe[t])[0]] if probe[t] in "abcdefghijklmnopqrstuvwxyz " else 0)
        babb_lp.append(p[int(pb[t])])
    return float(np.mean(real_lp) - np.mean(babb_lp))


# ─────────────────────────── a no-dial passive agent (the AT floor) ───────────────────────────
class PassiveAgent(contingency.ContingencyAgent):
    """Gain≡1, single effective table: cold_floor carries everything at weight 1, hot off. The AT baseline:
    reads babble and signal with equal weight — no contingency dial at all."""
    def __init__(self, **kw):
        super().__init__(hot_w=0.0, cold_w=0.0, cold_floor=1.0, hot_pool=1.0, **kw)


def score(agent):
    return metrics.bpc(agent, held), turn_overlap(agent)


# ───────────────────────────── 1. the three load-bearing conditions ─────────────────────────────
print("\n" + "=" * 92)
print("(1) ON vs YOKED vs PASSIVE — same tokens, same loop; only the timing→gain alignment differs.")
print("=" * 92)

DIAL = dict(tau=2.0, hot_w=1.0, cold_w=0.15, cold_floor=0.04, hot_pool=2.5)

a_on = contingency.ContingencyAgent(seed=SEED, **DIAL)
real_gains = real_run(a_on)
bpc_on, ov_on = score(a_on)

# YOKED: scramble the realized gains, replay identical tokens
scrambled = contingency.yoked_gains(real_gains, seed=SEED)
a_yk = contingency.ContingencyAgent(seed=SEED, **DIAL)
yoked_run(a_yk, scrambled)
bpc_yk, ov_yk = score(a_yk)

a_pa = PassiveAgent(seed=SEED, tau=2.0)
passive_run(a_pa)
bpc_pa, ov_pa = score(a_pa)

print(f"  {'condition':<10} {'held-bpc':>10} {'turn-overlap':>14}   note")
print(f"  {'PASSIVE':<10} {bpc_pa:>10.3f} {ov_pa:>14.4f}   no dial (gain≡1, single table) — the AT floor")
print(f"  {'YOKED':<10} {bpc_yk:>10.3f} {ov_yk:>14.4f}   same tokens, SCRAMBLED timing → random gain")
print(f"  {'ON':<10} {bpc_on:>10.3f} {ov_on:>14.4f}   real contingency: warm replies up-weighted")
print(f"\n  ON vs YOKED:  Δbpc = {bpc_yk - bpc_on:+.3f} (positive = ON better)   "
      f"Δturn-overlap = {ov_on - ov_yk:+.4f} (positive = ON better)")
print(f"  ON vs PASSIVE: Δbpc = {bpc_pa - bpc_on:+.3f}   "
      f"hot/cold mass at end: {a_on.hot_mass:,.0f} / {a_on.cold_mass:,.0f}")


# ───────────────────────────── 2. FRAGILE sweep — ≥10 variations ─────────────────────────────
# Do not kill on one weak point. Sweep τ (warmth window) and the hot/cold weighting; for EACH, the
# ON-vs-YOKED gap on bpc is the dial's winning axis. We want to know if ANY setting makes timing pay.
print("\n" + "=" * 92)
print("(2) FRAGILE sweep — τ × (cold_w, hot_pool). ON vs YOKED Δbpc at each setting (the dial's axis).")
print("=" * 92)
print(f"  {'tau':>4} {'cold_w':>7} {'hotP':>5} | {'bpc_ON':>8} {'bpc_YK':>8} {'Δbpc':>7} | "
      f"{'ov_ON':>7} {'ov_YK':>7} {'Δov':>7}")

variations = []
best = None
for tau in (1.0, 2.0, 4.0, 8.0):
    for cold_w, hot_pool in ((0.1, 3.0), (0.2, 2.0), (0.4, 1.5)):
        d = dict(tau=tau, hot_w=1.0, cold_w=cold_w, cold_floor=0.04, hot_pool=hot_pool)
        on = contingency.ContingencyAgent(seed=SEED, **d); rg = real_run(on); b_on, o_on = score(on)
        yk = contingency.ContingencyAgent(seed=SEED, **d)
        yoked_run(yk, contingency.yoked_gains(rg, seed=SEED)); b_yk, o_yk = score(yk)
        dbpc = b_yk - b_on; dov = o_on - o_yk
        variations.append((tau, cold_w, hot_pool, b_on, b_yk, dbpc, o_on, o_yk, dov))
        print(f"  {tau:>4.0f} {cold_w:>7.2f} {hot_pool:>5.1f} | {b_on:>8.3f} {b_yk:>8.3f} {dbpc:>+7.3f} | "
              f"{o_on:>7.4f} {o_yk:>7.4f} {dov:>+7.4f}")
        if best is None or dbpc > best[5]:
            best = variations[-1]

n_win = sum(1 for v in variations if v[5] > 0.005)        # ON beats YOKED on bpc by a clear margin
n_ov_win = sum(1 for v in variations if v[8] > 0.001)
print(f"\n  settings where ON beats YOKED on bpc (Δ>0.005): {n_win}/{len(variations)}")
print(f"  settings where ON beats YOKED on turn-overlap (Δ>0.001): {n_ov_win}/{len(variations)}")
print(f"  best Δbpc: {best[5]:+.3f} at tau={best[0]:.0f} cold_w={best[1]:.2f} hotP={best[2]:.1f}")


# ─────────────────── 2b. ROBUSTNESS — graded contingency (timing as a noisy cue) ───────────────────
# Section 1's perfect split (warm≡signal, cold≡babble) is the easy case; a real conversation is leaky:
# some warm replies are off-topic, and some real structure arrives in the gaps. Here timing is only a
# PROBABILISTIC cue. We sweep `reliability` r: warm slot is signal w.p. r, and signal LEAKS into cold
# slots w.p. (1−r). r=1.0 = perfect; r=0.5 = timing carries NO information (warm and cold equally likely
# to be signal) → the dial SHOULD collapse to YOKED there. The honest question: where does it break?
print("\n" + "=" * 92)
print("(2b) ROBUSTNESS — graded contingency. r=reliability of the timing cue (0.5 = timing uninformative).")
print("=" * 92)
print(f"  {'r':>5} | {'bpc_ON':>8} {'bpc_YK':>8} {'Δbpc':>7} | {'ov_ON':>7} {'ov_YK':>7} {'Δov':>7}")
DG = dict(tau=2.0, hot_w=1.0, cold_w=0.2, cold_floor=0.04, hot_pool=2.0)
robust = []
for r in (1.0, 0.9, 0.75, 0.6, 0.5):
    tr = build_transcript(reliability=r, rng=np.random.default_rng(SEED + 7))
    on = contingency.ContingencyAgent(seed=SEED, **DG); rg = real_run(on, tr); b_on, o_on = score(on)
    yk = contingency.ContingencyAgent(seed=SEED, **DG)
    yoked_run(yk, contingency.yoked_gains(rg, seed=SEED), tr); b_yk, o_yk = score(yk)
    robust.append((r, b_on, b_yk, b_yk - b_on, o_on, o_yk, o_on - o_yk))
    print(f"  {r:>5.2f} | {b_on:>8.3f} {b_yk:>8.3f} {b_yk-b_on:>+7.3f} | "
          f"{o_on:>7.4f} {o_yk:>7.4f} {o_on-o_yk:>+7.4f}")
print("  expected: Δ shrinks toward 0 as r→0.5 (the dial only pays when timing actually predicts content).")


# ───────────────────────────── 3. verdict on the kill-condition ─────────────────────────────
print("\n" + "=" * 92)
print("VERDICT")
print("=" * 92)
kill_fired = (n_win == 0) and (n_ov_win == 0)
if kill_fired:
    print("  KILL FIRED: at no setting does contingency-ON beat YOKED on bpc OR turn-overlap.\n"
          "  The dial is inert — scrambled-timing gains learn just as well. Honest NEGATIVE.")
else:
    print(f"  KILL DID NOT FIRE: ON beats YOKED on bpc at {n_win}/{len(variations)} settings "
          f"(turn-overlap at {n_ov_win}/{len(variations)}).\n"
          f"  Timing-aligned gain pays off: best Δbpc {best[5]:+.3f}. The contingency dial is real here.")
print(f"\n  ON vs PASSIVE-floor Δbpc {bpc_pa - bpc_on:+.3f}  (does the dial beat reading everything equally?)")
print("=" * 92)

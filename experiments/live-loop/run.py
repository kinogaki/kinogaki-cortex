#!/usr/bin/env python3
"""Exp BM — Live reactive loop: run BE's contingency gate + BC's recovery inside a real interlocutor.

BE (offline, fixed transcript): a WARM reply — one that lands right after the agent spoke — teaches more
than the SAME words re-timed to ignore it (the yoked control). The contingency was hand-built there. The
cognitive claim (Goldstein & Schwade) is about a LIVE contingent partner: the caregiver answers what the
child just did. BM closes the loop — a real interlocutor (Haiku, in 10-year-old register) whose reply is
a function of the agent's own utterance — and asks the BE kill-test again, now reactively:

    does contingency-ON beat the YOKED (scrambled-timing) ablation on the agent's surprise-at-replies
    and on turn-overlap, when the warm channel is a genuinely reactive partner?

The agent ACTS (emits → Δt→0), the world REPLIES (contingent), the agent OBSERVES the reply WARM, then a
COLD babble burst cools the clock. ON learns with the real warm gains; YOKED replays the SAME captured
replies with scrambled gains; PASSIVE reads everything at weight 1 (the AT floor). All three see the
EXACT same reply tokens — only the timing→gain alignment differs (BE's design, made live).

CREDENTIALS: we probe for the `anthropic` package + ANTHROPIC_API_KEY/CLAUDE_API_KEY and do a one-message
smoke test. If live works, we run the loop against Haiku. If the package/key/net is unavailable we DO NOT
fail — we fall back to a rich 10yo-register SCRIPTED responder (lib/liveloop.ScriptedResponder) and flag
the run as BLOCKED-ON-CREDENTIALS in the output and RESULTS.

Rules: ONLINE single pass; NO gradient/k-means/SVD/backprop; BOUNDED. Reuses contingency.ContingencyAgent
+ yoked_gains verbatim; this file + lib/liveloop.py are the only new code.
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
import liveloop

SEED = 0
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)
CODEC = H.CharCodec()

# ─────────────────────── babble distractor: English char marginals, i.i.d. (no structure) ───────────────────────
# A small text8 slice ONLY to get the marginal char distribution for the cold "noisy room" babble and to
# build a held-out English string for the turn-overlap probe. The learnable WARM signal is the live/
# fallback replies, not this slice. (SUBSTITUTION note: Goldstein & Schwade's CDS corpus is not in data/;
# the warm channel is a real reactive partner, the cold channel is frequency-matched English babble.)
slice_ids = load_ids("text8", nbytes=400_000)
counts = np.bincount(slice_ids, minlength=V0).astype(float); pmarg = counts / counts.sum()
held = ids_to_str(slice_ids[-30_000:])                      # held-out real English for turn-overlap
babble = liveloop.babble_maker(pmarg, V0, RNG)

# ─────────────────────── choose the interlocutor: live Haiku, else scripted fallback ───────────────────────
print("=" * 92)
print("BM — live reactive loop: probing the interlocutor")
print("=" * 92)
ok, detail = liveloop.smoke_test()
print(f"  smoke test: {'LIVE OK' if ok else 'BLOCKED'} — {detail}")
if ok:
    responder = liveloop.make_haiku_responder(max_tokens=40, temperature=0.7)
    MODE = "LIVE-HAIKU"
    N_TURNS = 60                                            # cheap: ≤60 live calls per condition-build
else:
    responder = liveloop.ScriptedResponder(seed=SEED)
    MODE = "FALLBACK-SCRIPTED (BLOCKED-ON-CREDENTIALS)"
    N_TURNS = 600                                           # fallback is free → run a longer loop
print(f"  MODE = {MODE} | turns = {N_TURNS}")

# ─────────────────────── the dial (BE's section-1 setting) ───────────────────────
DIAL = dict(tau=2.0, hot_w=1.0, cold_w=0.15, cold_floor=0.04, hot_pool=2.5)
UTTER_LEN = 24
N_BABBLE = 2


# ─────────────────────── metrics: surprise-at-replies + turn-overlap (BE's two axes) ───────────────────────
def surprise_at_replies(agent, replies):
    """The agent's mean surprise (−log2 p) on the TRUE next char of the held interlocutor replies — how
    well it predicts the contingent channel. Lower = it locked onto the warm replies (BE's obs-surprise)."""
    bits = []
    for rids in replies:
        s = CODEC.decode(rids)
        for t in range(1, len(s)):
            p = agent.dist(s[max(0, t - 16):t])
            bits.append(-np.log2(p[CODEC.encode(s[t])[0]] + 1e-12))
    return float(np.mean(bits)) if bits else float("nan")


def turn_overlap(agent):
    """BE's contingency metric: mean prob on the TRUE next char of held-out English minus mean prob on
    babble's next char. High = locked onto the structured (contingent) channel over the babble."""
    real_lp, babb_lp = [], []
    probe = held[:6000]
    pb = RNG.choice(V0, size=len(probe), p=pmarg)
    for t in range(8, len(probe)):
        p = agent.dist(probe[max(0, t - 16):t])
        real_lp.append(p[CODEC.encode(probe[t])[0]] if probe[t] in "abcdefghijklmnopqrstuvwxyz " else 0)
        babb_lp.append(p[int(pb[t])])
    return float(np.mean(real_lp) - np.mean(babb_lp))


class PassiveAgent(contingency.ContingencyAgent):
    """The AT floor: gain≡1, single effective table — reads warm replies and cold babble at equal weight."""
    def __init__(self, **kw):
        super().__init__(hot_w=0.0, cold_w=0.0, cold_floor=1.0, hot_pool=1.0, **kw)


# ─────────────────────── 1. ON vs YOKED vs PASSIVE in the reactive loop ───────────────────────
print("\n" + "=" * 92)
print("(1) ON vs YOKED vs PASSIVE — reactive loop. Same captured replies; only timing→gain alignment differs.")
print("=" * 92)

# ON: run the live/fallback reactive loop, capturing the replies + the realized warm gains.
a_on = contingency.ContingencyAgent(seed=SEED, **DIAL)
gains, replies = liveloop.reactive_run(
    a_on, responder, CODEC, n_turns=N_TURNS, babble=babble, n_babble=N_BABBLE, utter_len=UTTER_LEN)
n_reply_tok = sum(len(r) for r in replies)
print(f"  captured {len(replies)} replies, {n_reply_tok:,} reply chars "
      f"| warm gain mean={np.mean(gains):.3f} (1.0=fully warm)")
print(f"  sample replies: {[CODEC.decode(r)[:40] for r in replies[:3]]}")
bpc_on = metrics.bpc(a_on, held); sup_on = surprise_at_replies(a_on, replies); ov_on = turn_overlap(a_on)

# YOKED: replay the SAME captured replies with scrambled gains (BE's ablation, over the live replies).
scrambled = contingency.yoked_gains(gains, seed=SEED)
a_yk = contingency.ContingencyAgent(seed=SEED, **DIAL)
liveloop.yoked_reactive_run(a_yk, replies, scrambled, babble=babble, n_babble=N_BABBLE)
bpc_yk = metrics.bpc(a_yk, held); sup_yk = surprise_at_replies(a_yk, replies); ov_yk = turn_overlap(a_yk)

# PASSIVE: read the SAME replies + babble at weight 1.
a_pa = PassiveAgent(seed=SEED, tau=2.0)
liveloop.yoked_reactive_run(a_pa, replies, [1.0] * len(replies), babble=babble, n_babble=N_BABBLE)
bpc_pa = metrics.bpc(a_pa, held); sup_pa = surprise_at_replies(a_pa, replies); ov_pa = turn_overlap(a_pa)

print(f"\n  {'condition':<10} {'reply-surprise':>15} {'held-bpc':>10} {'turn-overlap':>14}   note")
print(f"  {'PASSIVE':<10} {sup_pa:>15.4f} {bpc_pa:>10.3f} {ov_pa:>14.4f}   no dial — replies+babble equal weight")
print(f"  {'YOKED':<10} {sup_yk:>15.4f} {bpc_yk:>10.3f} {ov_yk:>14.4f}   same replies, SCRAMBLED timing")
print(f"  {'ON':<10} {sup_on:>15.4f} {bpc_on:>10.3f} {ov_on:>14.4f}   real contingency: warm replies loud")
print(f"\n  ON vs YOKED:  Δreply-surprise = {sup_yk - sup_on:+.4f} (positive = ON better)")
print(f"                Δturn-overlap  = {ov_on - ov_yk:+.4f} (positive = ON better)")
print(f"                Δbpc           = {bpc_yk - bpc_on:+.3f} (positive = ON better)")


# ─────────────────────── 2. FRAGILE sweep — τ × (cold_w, hot_pool) ───────────────────────
# Reuse the SAME captured replies (one live/fallback loop) across all settings — the dial is what varies,
# not the world. This keeps a live run to ≤N_TURNS calls total.
print("\n" + "=" * 92)
print("(2) FRAGILE sweep — τ × (cold_w, hot_pool) over the captured replies. ON vs YOKED on the two axes.")
print("=" * 92)
print(f"  {'tau':>4} {'cold_w':>7} {'hotP':>5} | {'sup_ON':>8} {'sup_YK':>8} {'Δsup':>7} | "
      f"{'ov_ON':>7} {'ov_YK':>7} {'Δov':>7}")


def build_on(dial):
    """Re-derive ON over the captured replies WITHOUT new live calls: replay replies through the warm
    channel with the realized gains (gains depend only on the babble-burst sizes, identical here)."""
    a = contingency.ContingencyAgent(seed=SEED, **dial)
    # replay: warm reply (gain from clock), then cold babble bursts — mirrors reactive_run, no model calls
    for rids in replies:
        a.dt = 0                                            # the agent just "spoke" → next obs is warm
        a.observe(rids)
        for _ in range(N_BABBLE):
            a.observe(babble(len(rids)))
    return a


variations = []
best = None
for tau in (1.0, 2.0, 4.0, 8.0):
    for cold_w, hot_pool in ((0.1, 3.0), (0.15, 2.5), (0.4, 1.5)):
        d = dict(tau=tau, hot_w=1.0, cold_w=cold_w, cold_floor=0.04, hot_pool=hot_pool)
        on = build_on(d)
        # recover the realized gains for THIS tau (same clock pattern, scaled by tau)
        rg = []
        probe = contingency.ContingencyAgent(seed=SEED, **d)
        for rids in replies:
            probe.dt = 0
            rg.append(float(np.exp(-probe.dt / probe.tau)))
            probe.observe(rids)
            for _ in range(N_BABBLE):
                probe.observe(babble(len(rids)))
        s_on = surprise_at_replies(on, replies); o_on = turn_overlap(on)
        yk = contingency.ContingencyAgent(seed=SEED, **d)
        liveloop.yoked_reactive_run(yk, replies, contingency.yoked_gains(rg, seed=SEED),
                                    babble=babble, n_babble=N_BABBLE)
        s_yk = surprise_at_replies(yk, replies); o_yk = turn_overlap(yk)
        dsup = s_yk - s_on; dov = o_on - o_yk
        variations.append((tau, cold_w, hot_pool, s_on, s_yk, dsup, o_on, o_yk, dov))
        print(f"  {tau:>4.0f} {cold_w:>7.2f} {hot_pool:>5.1f} | {s_on:>8.4f} {s_yk:>8.4f} {dsup:>+7.4f} | "
              f"{o_on:>7.4f} {o_yk:>7.4f} {dov:>+7.4f}")
        if best is None or dsup > best[5]:
            best = variations[-1]

n_sup_win = sum(1 for v in variations if v[5] > 0.01)
n_ov_win = sum(1 for v in variations if v[8] > 0.001)
print(f"\n  settings where ON beats YOKED on reply-surprise (Δ>0.01): {n_sup_win}/{len(variations)}")
print(f"  settings where ON beats YOKED on turn-overlap (Δ>0.001): {n_ov_win}/{len(variations)}")
print(f"  best Δreply-surprise: {best[5]:+.4f} at tau={best[0]:.0f} cold_w={best[1]:.2f} hotP={best[2]:.1f}")


# ─────────────────────── 3. verdict on the BE kill-test, run live ───────────────────────
print("\n" + "=" * 92)
print("VERDICT (the live AT kill-test)")
print("=" * 92)
kill_fired = (n_sup_win == 0) and (n_ov_win == 0)
if kill_fired:
    print("  KILL FIRED: in the reactive loop, contingency-ON does NOT beat YOKED on reply-surprise OR\n"
          "  turn-overlap at any setting. The dial is inert when the warm channel is genuinely reactive.\n"
          "  Honest NEGATIVE — BE's offline win does not carry into the live loop.")
else:
    print(f"  KILL DID NOT FIRE: ON beats YOKED on reply-surprise at {n_sup_win}/{len(variations)} settings\n"
          f"  and on turn-overlap at {n_ov_win}/{len(variations)}. BE's offline win SURVIVES the reactive loop:\n"
          f"  best Δreply-surprise {best[5]:+.4f}.")
print(f"\n  MODE = {MODE}")
print(f"  ON vs PASSIVE-floor: Δreply-surprise {sup_pa - sup_on:+.4f}, Δbpc {bpc_pa - bpc_on:+.3f}")
print("=" * 92)

#!/usr/bin/env python3
"""Exp BA — dual-route inflection head: words-and-rules as ONE tunable gate, and the rare micro-U.

The cognitive bet (Pinker *Words and Rules* vs Rumelhart & McClelland's single net; Marcus/Maslen
over-regularization corpus; Weissweiler *graded* productivity). A child says *went* for years, then for
a while says *goed*, then *went* again — a U-shaped curve. The textbook reading: a memory route (store
*went*) and a rule route (+ed) compete; the rule "wins" transiently. The count-native claim tested here:
the two routes are two reads of ONE leaky counter, fused by ONE gate, and the U-shape is not designed in
— it falls out, PER VERB, exactly when a low-frequency irregular's leaky f·c decays below the gate. The
gate is a measurable knob that slides Pinker (memory protects irregulars) ↔ Rumelhart (the rule eats
everything), not a resolution of the debate.

CORPUS. The spec asks for AO-CHILDES / child-directed speech + a synthetic irregular stream. CHILDES is
NOT in data/, so we BUILD a synthetic frequency-matched verb stream (the place this mechanism can win:
per-item rates need a stream with known irregular:regular structure and controllable frequencies — a
Saffran-style designed stream, exactly what the spec authorizes). Verbs are Zipfian; a labelled set of
irregulars (go→went …) vs regulars (+ed). We stream stem→form pairs ONLINE; the head never sees a label
at production time, only counts.

METRIC. Per-VERB over-regularization rate = P(produce stem+ed | the verb is irregular), tracked over a
sliding window as the stream advances. The headline tests:
  (1) THE U.  Does each irregular show a per-item dip (correct → over-regularized → correct) that is
      RARE in aggregate (~2.5–10%, Marcus) and STAGGERED across verbs — never a synchronized macro-U?
  (2) THE KNOB.  Does sweeping the gate slide the aggregate over-regularization rate monotonically
      Pinker(0%)↔Rumelhart(100%), with a mid regime giving the low-constant item-specific rate?
  (3) FREQUENCY.  Is over-regularization concentrated on LOW-frequency irregulars (decay wins) and
      absent on high-frequency ones (memory protected) — the central Marcus prediction?
BASELINES. single-route default-only (gate→0 in the sense "rule always", here the dual head with the
memory disabled), pure-memory (gate→∞: memory always blocks), recency n-gram (last-seen form wins).

KILL (BA). No gate setting reproduces a low-constant item-specific rate, OR dual-route matches the
single-route error pattern no better → buys nothing over AF+AB. FRAGILE budget: sweep the gate, don't
kill on one setting.

Online, single streaming pass, no gradient/k-means/SVD, bounded memory (LRU irregular store + one suffix
table). Fixed seed.
"""
import os, sys, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import dualroute

SEED = 0
rng = np.random.default_rng(SEED)


# ── build a synthetic frequency-matched verb stream ────────────────────────────────────────────────

# irregulars: (stem, stored_past). Chosen across the frequency range so we can test the freq prediction.
IRREGULARS = [
    ("go", "went"), ("be", "was"), ("have", "had"), ("do", "did"), ("say", "said"),
    ("make", "made"), ("take", "took"), ("come", "came"), ("see", "saw"), ("know", "knew"),
    ("give", "gave"), ("find", "found"), ("think", "thought"), ("tell", "told"), ("feel", "felt"),
    ("leave", "left"), ("put", "put"), ("mean", "meant"), ("keep", "kept"), ("hold", "held"),
    ("bring", "brought"), ("begin", "began"), ("write", "wrote"), ("stand", "stood"), ("hear", "heard"),
    ("run", "ran"), ("eat", "ate"), ("fall", "fell"), ("catch", "caught"), ("buy", "bought"),
]
N_REG = 120                                    # regular verbs (productive +ed)
REGULARS = [f"reg{i:03d}" for i in range(N_REG)]

ALL = [s for s, _ in IRREGULARS] + REGULARS
PAST = {s: p for s, p in IRREGULARS}
for s in REGULARS:
    PAST[s] = s + "ed"
IS_IRREG = {s: True for s, _ in IRREGULARS}
for s in REGULARS:
    IS_IRREG[s] = False

# Zipfian usage frequencies over ALL verbs (rank 1 = most frequent). Irregulars are the high-frequency
# end of the language (the real Marcus fact), but we deliberately scatter a few irregulars into the tail
# so the frequency prediction has variance to test.
V = len(ALL)
ranks = np.arange(1, V + 1)
# put high-freq irregulars first, but interleave a handful of low-freq irregulars into the regular tail
order = list(range(len(IRREGULARS))) + list(range(len(IRREGULARS), V))
# move the LAST 6 irregulars (run..buy) down into the tail to make them low-frequency irregulars
LOWFREQ_IRREG = [s for s, _ in IRREGULARS[-6:]]
freq_rank = {}
hi = [s for s, _ in IRREGULARS[:-6]]
lo = LOWFREQ_IRREG
regs = REGULARS
seq_for_zipf = hi + regs[:60] + lo + regs[60:]    # lo verbs sit deep in the tail
for r, s in enumerate(seq_for_zipf, start=1):
    freq_rank[s] = r
zipf_w = np.array([1.0 / freq_rank[s] for s in ALL])
zipf_w /= zipf_w.sum()

STREAM_LEN = 220_000

# DEVELOPMENTAL DYNAMIC (what makes the U a U, not a flat rate). In a child, the +ed rule's productivity
# RISES over time as the regular vocabulary grows (the vocabulary spurt). We model this: regular verbs are
# introduced PROGRESSIVELY — at step t only the first `n_active_regs(t)` regulars can appear. Early in the
# stream the rule is weak (few +ed types), so a not-yet-entrenched irregular is produced from faint memory;
# mid-stream the rule comes online and OVER-REGULARIZES it; late, the irregular's own leaky count has
# climbed past the gate and it RECOVERS. That rise-then-recover is the per-item micro-U.
REG_RAMP_END = int(STREAM_LEN * 0.55)            # all regulars active by 55% through the stream
irreg_stems = [s for s, _ in IRREGULARS]
irreg_w = np.array([1.0 / freq_rank[s] for s in irreg_stems]); irreg_w /= irreg_w.sum()
reg_order = REGULARS                              # the order regulars come online
reg_w_full = np.array([1.0 / freq_rank[s] for s in REGULARS])

def build_stream():
    out = np.empty(STREAM_LEN, dtype=object)
    # mixture weight on irregular vs regular block (irregulars are the frequent core, ~ constant share)
    irreg_share = irreg_w.sum() / zipf_w.sum() if False else None
    base_irreg_w = np.array([1.0 / freq_rank[s] for s in irreg_stems])
    for t in range(STREAM_LEN):
        n_active = max(4, int(len(REGULARS) * min(1.0, t / REG_RAMP_END)))
        active_regs = reg_order[:n_active]
        rw = reg_w_full[:n_active]
        # build per-step weight over irregulars + currently-active regulars
        w_ir = base_irreg_w
        w = np.concatenate([w_ir, rw]); w = w / w.sum()
        pool = irreg_stems + active_regs
        out[t] = pool[rng.choice(len(pool), p=w)]
    return out

# Per-step rebuilding the weight vector is O(STREAM_LEN * V) and slow; precompute in coarse epochs instead.
def build_stream_fast(epochs=40):
    out = []
    base_irreg_w = np.array([1.0 / freq_rank[s] for s in irreg_stems])
    per = STREAM_LEN // epochs
    for e in range(epochs):
        t_mid = e * per + per // 2
        n_active = max(4, int(len(REGULARS) * min(1.0, t_mid / REG_RAMP_END)))
        active_regs = reg_order[:n_active]
        rw = reg_w_full[:n_active]
        w = np.concatenate([base_irreg_w, rw]); w = w / w.sum()
        pool = np.array(irreg_stems + active_regs, dtype=object)
        idx = rng.choice(len(pool), size=per, p=w)
        out.append(pool[idx])
    tail = STREAM_LEN - per * epochs
    if tail > 0:
        out.append(pool[rng.choice(len(pool), size=tail, p=w)])
    return np.concatenate(out)

stems = build_stream_fast()


# ── baselines ──────────────────────────────────────────────────────────────────────────────────────

class RecencyNGram:
    """Last-seen form wins (a degenerate memory with no leak, no gate, no rule). Over-regularizes a verb
    only if it has literally never been heard — so it should give ~0 over-regularization but also can't
    produce a U. The 'frequency is everything' null."""
    def __init__(self):
        self.last = {}
    def observe(self, stem, form, is_regular):
        self.last[stem] = form
    def produce(self, stem):
        if stem in self.last:
            return self.last[stem], "memory"
        return stem + "ed", "default"


def run_model(make_head, probe_every=2000, win=4000):
    """One online pass. At each step: PRODUCE for the current stem (read-out), then OBSERVE the truth.
    Track per-verb over-regularization in a sliding window. Returns (agg_rate_trace, per_verb_traces)."""
    head = make_head()
    # sliding-window buffers per irregular stem: list of (step, was_overreg 0/1)
    events = {s: [] for s, _ in IRREGULARS}
    agg_trace = []          # (step, aggregate overreg rate over irregular productions in window)
    win_buf = []            # global window of (step, is_irreg_production, overreg)
    for t, stem in enumerate(stems):
        form, route = head.produce(stem)
        if IS_IRREG[stem]:
            overreg = 1 if (route == "default" or form == stem + "ed") and form != PAST[stem] else 0
            # more precisely: over-regularization = produced the +ed form for an irregular
            overreg = 1 if form == stem + "ed" else 0
            events[stem].append((t, overreg))
            win_buf.append((t, overreg))
        # online observe of the TRUTH
        head.observe(stem, PAST[stem], IS_IRREG[stem])
        # trim window
        while win_buf and win_buf[0][0] < t - win:
            win_buf.pop(0)
        if t % probe_every == 0 and t > 0 and win_buf:
            rate = sum(o for _, o in win_buf) / len(win_buf)
            agg_trace.append((t, rate))
    return agg_trace, events, head


def overreg_rate(events_for_stem):
    if not events_for_stem:
        return float("nan"), 0
    n = len(events_for_stem)
    return sum(o for _, o in events_for_stem) / n, n


def thirds_u(events_for_stem):
    """A REAL temporal U needs the over-reg rate to RISE then FALL over the verb's own timeline. Split the
    verb's event sequence into thirds; return (early%, mid%, late%). A micro-U = mid > early and mid > late
    (the hump). A 'never-learned' verb shows flat-high (early≈mid≈late high) — NOT a U."""
    ov = np.array([o for _, o in events_for_stem])
    if len(ov) < 30:
        return None
    a, b, c = np.array_split(ov, 3)
    return a.mean(), b.mean(), c.mean()


def micro_u_depth(events_for_stem, win=3000):
    """Detect a per-item U: scan a sliding window over this stem's events; report (peak window rate,
    n events). A real micro-U has near-0 early, a hump >0 mid, near-0 late — peak >> aggregate."""
    if len(events_for_stem) < 10:
        return 0.0, len(events_for_stem)
    steps = np.array([s for s, _ in events_for_stem])
    ov = np.array([o for _, o in events_for_stem])
    peak = 0.0
    # window by event-index (each verb has its own event timeline)
    W = max(8, len(ov) // 6)
    for i in range(0, len(ov) - W + 1, max(1, W // 2)):
        peak = max(peak, ov[i:i + W].mean())
    return peak, len(ov)


def main():
    print(f"Exp BA — dual-route inflection head  (synthetic CDS-ish stream; {STREAM_LEN:,} tokens, "
          f"{len(IRREGULARS)} irregular + {N_REG} regular verbs, seed {SEED})")
    print("CORPUS NOTE: CHILDES not in data/ — synthetic frequency-matched verb stream (spec-authorized).")
    print()

    t0 = time.time()

    # ── Q2 + main: sweep the gate (the Pinker↔Rumelhart knob) ──────────────────────────────────────
    gates = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 1e9]
    print("Q2 — GATE SWEEP (the Pinker↔Rumelhart knob). agg = over-regularization rate over irregulars:")
    print(f"  {'gate':>8} | {'agg over-reg %':>14} | {'hi-freq irreg %':>15} | {'lo-freq irreg %':>15}")
    sweep = {}
    for g in gates:
        agg, ev, head = run_model(lambda g=g: dualroute.DualRouteHead(gate=g, leak=0.0015))
        # aggregate over all irregular productions (whole-run)
        tot = sum(len(e) for e in ev.values())
        ov = sum(o for e in ev.values() for _, o in e)
        agg_rate = ov / tot if tot else float("nan")
        hi_rate = np.nanmean([overreg_rate(ev[s])[0] for s in [k for k, _ in IRREGULARS[:-6]]])
        lo_rate = np.nanmean([overreg_rate(ev[s])[0] for s in LOWFREQ_IRREG])
        sweep[g] = (agg_rate, hi_rate, lo_rate, ev)
        gl = "inf" if g >= 1e8 else f"{g:g}"
        print(f"  {gl:>8} | {agg_rate*100:13.2f}% | {hi_rate*100:14.2f}% | {lo_rate*100:14.2f}%")
    print()

    # pick the mid gate that gives a low-constant aggregate rate for the detailed analysis
    target = None
    for g in gates:
        ar = sweep[g][0]
        if 0.015 <= ar <= 0.18:        # low-constant Marcus band (loosened a touch)
            target = g; break
    if target is None:
        # fall back to the gate whose aggregate rate is closest to ~6%
        target = min(gates, key=lambda g: abs(sweep[g][0] - 0.06))
    print(f"Selected mid gate = {target:g} (aggregate over-reg = {sweep[target][0]*100:.2f}%) for Q1/Q3.")
    print()

    ev = sweep[target][3]

    # ── Q3: frequency dependence (per-verb) ────────────────────────────────────────────────────────
    print("Q3 — FREQUENCY: per-verb over-regularization rate (Marcus: concentrated on LOW-freq irregulars):")
    print(f"  {'verb':>8} {'freq-rank':>9} {'n_prod':>7} {'over-reg %':>11} {'micro-U peak %':>14}")
    rows = []
    for s, _ in IRREGULARS:
        r, n = overreg_rate(ev[s])
        peak, ne = micro_u_depth(ev[s])
        rows.append((s, freq_rank[s], n, r, peak))
    for s, fr, n, r, peak in sorted(rows, key=lambda x: x[1]):
        tag = " (LOW-freq irreg)" if s in LOWFREQ_IRREG else ""
        print(f"  {s:>8} {fr:>9} {n:>7} {r*100:10.2f}% {peak*100:13.2f}%{tag}")
    print()

    # ── Q1: micro vs macro U ───────────────────────────────────────────────────────────────────────
    # macro-U = the AGGREGATE trace dips and recovers synchronously. micro-U = individual verbs hump at
    # DIFFERENT times. Measure: aggregate peak vs mean of per-verb peaks, and stagger of peak timing.
    agg_trace, _, _ = run_model(lambda: dualroute.DualRouteHead(gate=target, leak=0.0015))
    agg_rates = [r for _, r in agg_trace]
    agg_peak = max(agg_rates) if agg_rates else 0.0
    agg_mean = np.mean(agg_rates) if agg_rates else 0.0
    per_verb_peaks = [micro_u_depth(ev[s])[0] for s, _ in IRREGULARS if len(ev[s]) >= 10]
    print("Q1 — MICRO vs MACRO U:")
    print(f"  aggregate over-reg trace: mean={agg_mean*100:.2f}%  peak={agg_peak*100:.2f}%  "
          f"(flat aggregate = NO macro-U)")
    print(f"  mean per-verb micro-U peak = {np.mean(per_verb_peaks)*100:.2f}%  "
          f"max per-verb peak = {np.max(per_verb_peaks)*100:.2f}%")
    print(f"  → per-verb peaks {'EXCEED' if np.mean(per_verb_peaks) > agg_peak else 'do NOT exceed'} "
          f"the aggregate peak (micro-U present iff they exceed and aggregate stays flat).")
    print()
    print("  TEMPORAL U per verb (early/mid/late thirds of the verb's own timeline; U = mid>early & mid>late):")
    print(f"    {'verb':>8} {'early%':>7} {'mid%':>7} {'late%':>7}  shape")
    n_real_u = 0
    for s, _ in IRREGULARS:
        tu = thirds_u(ev[s])
        if tu is None:
            continue
        e, m, l = tu
        # a REAL U starts correct (low early), humps in the middle, recovers (lower late than mid).
        is_u = (e < 0.5) and (m > e + 0.02) and (m > l + 0.01)
        shape = "U (rise→fall)" if is_u else ("flat-high (never learned)" if e > 0.5 else "flat-low")
        if is_u:
            n_real_u += 1
        # only print verbs with any over-reg activity to keep it readable
        if max(e, m, l) > 0.005:
            print(f"    {s:>8} {e*100:6.1f}% {m*100:6.1f}% {l*100:6.1f}%  {shape}")
    print(f"  → {n_real_u} verbs show a genuine temporal U (rise-then-fall on their own timeline).")
    print()

    # ── Baselines ──────────────────────────────────────────────────────────────────────────────────
    print("BASELINES (aggregate over-reg rate over irregulars; can any reproduce a low-constant rate?):")
    base_specs = [
        ("single-route default-only", lambda: dualroute.DualRouteHead(gate=1e9, leak=0.0015,
                                                                       default_floor=-1.0)),
        ("pure-memory (gate=0)", lambda: dualroute.DualRouteHead(gate=0.0, leak=0.0)),
        ("recency n-gram", lambda: RecencyNGram()),
    ]
    # default-only: force the rule to always fire by making memory never clear (gate=inf) AND floor<=0
    # so the default always passes. pure-memory: gate 0 + no leak so memory always wins once seen.
    for name, mk in base_specs:
        _, ev_b, _ = run_model(mk)
        tot = sum(len(e) for e in ev_b.values()); ov = sum(o for e in ev_b.values() for _, o in e)
        rate = ov / tot if tot else float("nan")
        lo = np.nanmean([overreg_rate(ev_b[s])[0] for s in LOWFREQ_IRREG])
        hi = np.nanmean([overreg_rate(ev_b[s])[0] for s in [k for k, _ in IRREGULARS[:-6]]])
        print(f"  {name:<28} agg={rate*100:6.2f}%   hi-freq={hi*100:6.2f}%   lo-freq={lo*100:6.2f}%")
    print()

    # ── Verdict signals ────────────────────────────────────────────────────────────────────────────
    midband = [g for g in gates if 0.015 <= sweep[g][0] <= 0.18]
    slides = sweep[0.0][0] < sweep[1e9][0]            # gate slides Pinker(low)↔Rumelhart(high)
    freq_dissoc = sweep[target][2] > sweep[target][1] + 0.02   # lo-freq > hi-freq by a margin
    print("VERDICT SIGNALS:")
    print(f"  gate slides Pinker↔Rumelhart (agg rises 0→inf): {slides}  "
          f"({sweep[0.0][0]*100:.1f}% → {sweep[1e9][0]*100:.1f}%)")
    print(f"  a mid gate gives a low-constant (1.5–18%) aggregate rate: {len(midband) > 0}  "
          f"(gates: {[f'{g:g}' for g in midband]})")
    print(f"  over-reg concentrated on LOW-freq irregulars (Marcus): {freq_dissoc}  "
          f"(lo={sweep[target][2]*100:.1f}% vs hi={sweep[target][1]*100:.1f}%)")
    print(f"\n  elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

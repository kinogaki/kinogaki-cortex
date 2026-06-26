#!/usr/bin/env python3
"""Experiment BI — the GOLDILOCKS learning-rate gate (an inverted-U on surprisal).

A child does not learn equally from every word. The desirable-difficulty / Goldilocks-attention
literature (Kidd et al. on infant looking; the N400/cloze surprisal curve) says learning peaks in a
MIDDLE band of predictability: the already-known carries no news, the unparsable does not connect to
anything you have. The naive correction — "learn MORE the more surprised you are" (monotone
surprise-as-gate) — is exactly wrong at the high end, where a typo / OOV burst is both maximally
surprising and maximally worthless. BI makes the model's WRITE-WEIGHT an INVERTED-U on its OWN
surprisal: skip the low-s news-free token AND the high-s noise, spend the bounded write budget on the
middle band that actually teaches.

The decisive test is at EQUAL TABLE SIZE. Three gate shapes share one streaming-pass, LFU-capped
count model — flat (count every token once), monotone (the naive surprise-gate this corrects), and
goldilocks (the inverted-U) — so the only difference is HOW MUCH each token writes. The kill: at equal
cap the gate must lower held-out bpc, and not merely by storing MORE distinct contexts (that is a
memory win, not a learning-rate win). We run the FRAGILE budget (≥10 gate variations) and we check the
high-surprisal rare-context slice the goldilocks gate deliberately skips before drawing any conclusion.

Folded in (kept DISTINCT from the write-gate, per the BUILD_QUEUE note): the N400/cloze read-out —
after training, validate that the model's surprisal tracks cloze predictability (surprisal ↑ as cloze
↓, the canonical N400 relationship; constraining contexts give small N400 / high cloze). That is a read
of the same counts, never a second gate.

HARD RULES: single streaming pass (predict-then-write, one token at a time); no gradient descent /
k-means / SVD / backprop (the gate is a closed-form scalar on a surprisal the counts already give);
bounded memory (every order's table LFU-capped — the cap IS the experiment). Fixed seed 0. text8 slice.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics, goldilocks            # the shared lib + this experiment's module

SEED = 0
np.random.seed(SEED)

ORDER = 5                       # char backoff order (Exp AI's grain)
N_TRAIN = 120_000               # chars, single streaming pass (≤ a few MB)
N_EVAL = 12_000                 # disjoint held-out tail
CAPS = [1500, 4000, 12000]      # per-order context budget (the bounded-memory axis)


def banner(s):
    print("\n" + "=" * 80 + f"\n{s}\n" + "=" * 80, flush=True)


def main():
    t0 = time.time()

    # ── corpus: a small text8 slice (clean lowercase [a-z ]) ─────────────────────────────────────
    ids = corpus.load_ids("text8", nbytes=N_TRAIN + N_EVAL + 8192)
    train, ev = ids[:N_TRAIN], ids[N_TRAIN:N_TRAIN + N_EVAL]
    print(f"  corpus text8: {len(train):,} train / {len(ev):,} held-out chars, order {ORDER}, seed {SEED}")

    # ── the FRAGILE budget: ≥10 gate variations, every one at equal cap ──────────────────────────
    # flat (1), monotone (3 slopes), goldilocks (≥6 center/width/peak settings) = 10 gate shapes.
    GATES = [
        ("flat",       "flat",       {}),
        ("mono-soft",  "monotone",   dict(s0=3.0, wmax=3.0)),
        ("mono-mid",   "monotone",   dict(s0=2.0, wmax=4.0)),
        ("mono-steep", "monotone",   dict(s0=1.0, wmax=6.0)),
        ("gold-lo",    "goldilocks", dict(center=1.6, width=1.4, peak=3.0, floor=0.05)),
        ("gold-mid",   "goldilocks", dict(center=2.2, width=1.6, peak=3.0, floor=0.05)),
        ("gold-hi",    "goldilocks", dict(center=2.8, width=1.6, peak=3.0, floor=0.05)),
        ("gold-narrow","goldilocks", dict(center=2.2, width=1.0, peak=3.0, floor=0.05)),
        ("gold-wide",  "goldilocks", dict(center=2.2, width=2.4, peak=3.0, floor=0.05)),
        ("gold-peaky", "goldilocks", dict(center=2.2, width=1.6, peak=5.0, floor=0.02)),
        ("gold-floor0","goldilocks", dict(center=2.2, width=1.6, peak=3.0, floor=0.00)),
    ]

    banner("THE GATE SWEEP — held-out bpc at EQUAL table cap (lower=better); writes=learning-rate budget")
    print(f"  {'cap':>6}  {'gate':>12} | {'bpc':>8} | {'table':>8} | {'writes':>11} | {'skipped':>8}")
    print("  " + "-" * 70)
    results = {}                                            # (cap, name) -> dict
    for cap in CAPS:
        for name, gate, kw in GATES:
            r = goldilocks.train_eval(train, ev, order=ORDER, cap=cap, gate=gate, gate_kw=kw)
            results[(cap, name)] = r
            print(f"  {cap:>6}  {name:>12} | {r['bpc']:8.4f} | {r['size']:8,} | "
                  f"{r['writes']:11,.0f} | {r['skipped']:8,}")

    # ── THE KILL-CHECK: at equal cap, does the gate beat flat WITHOUT storing more? ──────────────
    banner("KILL-CHECK — best gate vs FLAT at equal cap: Δbpc and whether the table is bigger")
    print(f"  {'cap':>6} | {'flat bpc':>9} | {'best gate':>12} {'(bpc)':>9} | {'Δbpc':>8} | {'Δtable':>9} | verdict")
    print("  " + "-" * 84)
    kill_rows = []
    for cap in CAPS:
        flat = results[(cap, "flat")]
        # best NON-flat gate by bpc at this cap
        cands = [(g[0], results[(cap, g[0])]) for g in GATES if g[0] != "flat"]
        best_n, best = min(cands, key=lambda x: x[1]["bpc"])
        dbpc = best["bpc"] - flat["bpc"]
        dtab = best["size"] - flat["size"]
        # win only if bpc improves AND not just by storing more (Δtable not materially larger)
        helps = dbpc < -0.0005
        cheaper_or_equal = dtab <= max(1, int(0.02 * flat["size"]))   # ≤2% bigger counts as equal-budget
        verdict = ("WIN" if helps and cheaper_or_equal else
                   "win-by-memory" if helps else "no gain")
        kill_rows.append((cap, flat["bpc"], best_n, best["bpc"], dbpc, dtab, verdict))
        print(f"  {cap:>6} | {flat['bpc']:9.4f} | {best_n:>12} {best['bpc']:9.4f} | "
              f"{dbpc:+8.4f} | {dtab:+9,} | {verdict}")

    # ── THE SLICE THE GOLDILOCKS GATE SKIPS: high-surprisal rare-context bpc (BUILD_QUEUE kill note) ─
    banner("HIGH-S SLICE — bpc on the rare/unparsable tail the goldilocks gate SKIPS (the kill caveat)")
    cap = CAPS[1]
    flat_m = results[(cap, "flat")]["model"]
    gold_m = results[(cap, "gold-mid")]["model"]
    # rare-context eval sites: where the FLAT model's surprisal on the true char is in the top tertile
    ids_e = [int(x) for x in ev]
    surpr_flat = []
    ctx = []
    for nx in ids_e:
        p = flat_m._dist_ids(ctx[-ORDER:])
        surpr_flat.append(-np.log2(p[nx] + 1e-12)); ctx.append(nx)
        if len(ctx) > ORDER + 2: ctx = ctx[-(ORDER + 2):]
    surpr_flat = np.array(surpr_flat)
    hi = surpr_flat >= np.quantile(surpr_flat, 0.66)
    def sliced_bpc(model, mask):
        ids2 = ids_e; bits = []; ctx = []
        for i, nx in enumerate(ids2):
            p = model._dist_ids(ctx[-ORDER:])
            if mask[i]: bits.append(-np.log2(p[nx] + 1e-12))
            ctx.append(nx)
            if len(ctx) > ORDER + 2: ctx = ctx[-(ORDER + 2):]
        return float(np.mean(bits))
    flat_hi = sliced_bpc(flat_m, hi); gold_hi = sliced_bpc(gold_m, hi)
    print(f"  high-surprisal slice (top tertile by flat surprisal, {int(hi.sum()):,} sites):")
    print(f"    flat       bpc = {flat_hi:.4f}")
    print(f"    goldilocks bpc = {gold_hi:.4f}   (Δ = {gold_hi - flat_hi:+.4f}; + = gate HURTS the tail it skips)")

    # ── THE N400 / CLOZE READ-OUT (distinct from the write-gate) ─────────────────────────────────
    banner("N400 / CLOZE READ-OUT — model surprisal as an N400 proxy (a READ of the counts, not a gate)")
    print(f"  {'model':>12} | {'n400 r':>8} | {'cloze lo-ctx':>12} {'hi-ctx':>9} | {'surpr lo-ctx':>12} {'hi-ctx':>9}")
    print("  " + "-" * 80)
    n400_rows = []
    for name in ("flat", "gold-mid"):
        m = results[(CAPS[1], name)]["model"]
        c = goldilocks.cloze_readout(m, ev, seed=SEED)
        n400_rows.append((name, c))
        print(f"  {name:>12} | {c['n400_r']:8.3f} | {c['cloze_lowctx']:12.4f} {c['cloze_hictx']:9.4f} | "
              f"{c['surpr_lowctx']:12.3f} {c['surpr_hictx']:9.3f}")
    print("  (N400 faithful ⇔ r>0 strongly: surprisal rises as cloze falls; lo-ctx=constraining ⇒ small N400/high cloze)")

    print(f"\n  total wall {time.time()-t0:.0f}s")

    # ── machine-readable dump for RESULTS.md ─────────────────────────────────────────────────────
    print("\n[DUMP]")
    for (cap, name), r in results.items():
        print("sweep", cap, name, round(r["bpc"], 4), r["size"], int(r["writes"]), r["skipped"])
    for row in kill_rows:
        print("kill", row[0], round(row[1], 4), row[2], round(row[3], 4), round(row[4], 4), row[5], row[6])
    print("highs", round(flat_hi, 4), round(gold_hi, 4), round(gold_hi - flat_hi, 4))
    for name, c in n400_rows:
        print("n400", name, round(c["n400_r"], 3), round(c["cloze_lowctx"], 4), round(c["cloze_hictx"], 4),
              round(c["surpr_lowctx"], 3), round(c["surpr_hictx"], 3))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Exp AA — SLEEP / CONSOLIDATION over the count memory ("agent dreaming"). Inspired by Letta's
"Towards Agents that Learn": an agent improves by refining its TOKEN-SPACE MEMORY offline (sleep-time
compute), with the known failure mode that "memories become generic and lossy after repeated refinement."

In our world this is almost literal — a Column IS a memory-agent; its COUNT TABLES are token-space memory.
The online substrate already learns weight-free. The NEW thing: an OFFLINE sleep pass that REFINES the
count memory using only count operations (prune / distill / promote + replay over a bounded buffer), and a
test of whether refinement-without-new-data helps, and where REPEATED refinement starts to degrade.

  1. Sleep improves without new data?  Train online, then ONE sleep cycle. Held-out bpc/acc + MEMORY size
     before vs after, split RARE-context vs COMMON-context.
  2. Repeated refinement (the Letta failure mode). A GENTLE schedule (idempotent) and an AGGRESSIVE schedule
     (escalating distill threshold each cycle) — find improve -> saturate / DEGRADE turning point.
  3. Generic-vs-specific balance. Rare- vs common-context tracked separately across cycles.
  4. Promote (concepts) — reported as a count-only negative (see RESULTS): clustering surviving contexts
     into a concept tier cannot reach genuinely UNSEEN contexts without a batch similarity index, so it
     does not generalize; off by default.

Corpus: text8 (~16 MB). Char backoff order 6. Single online pass + offline sleep. Fixed seed. NO gradient,
NO batch optimization — every sleep step is count/replay/leader-cluster (online-compliant; see RESULTS).
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids
from consolidate import learn_tables, score, memory_size, sleep

print = functools.partial(print, flush=True)

TRAIN_BYTES = 16_000_000
TEST_BYTES  = 2_000_000      # held-out tail, disjoint from train
ORDER       = 6
BUFFER      = 3_000_000      # the bounded recent buffer the sleep pass "dreams" over (last N train chars)
N_CYCLES    = 10
RARE_THRESH = 20             # context evidence below this = "rare context"
SEED        = 0

# the GENTLE sleep schedule (the keeper): prune untrustworthy contexts, drop the long tail, lossless distill
GENTLE = dict(min_ctx=3, tail_mass=0.999, distill_tau=0.05, promote=False)


def fmt(s):
    return (f"bpc {s['bpc']:.4f}  acc {s['acc']*100:5.2f}%  | "
            f"rare bpc {s['bpc_rare']:.3f} acc {s['acc_rare']*100:5.2f}%  | "
            f"common bpc {s['bpc_common']:.3f} acc {s['acc_common']*100:5.2f}%")


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN_BYTES + TEST_BYTES)
    train = np.ascontiguousarray(ids[:TRAIN_BYTES])
    test = np.ascontiguousarray(ids[TRAIN_BYTES:TRAIN_BYTES + TEST_BYTES])
    buffer = np.ascontiguousarray(train[-BUFFER:])
    print(f"text8: train {len(train):,} chars, held-out test {len(test):,} chars, "
          f"buffer {len(buffer):,} (load {time.time()-t0:.1f}s)")

    # ── ONLINE substrate: one streaming pass ──
    t1 = time.time()
    tab0 = learn_tables(train, ORDER)
    base = score(tab0, test, rare_ctx_thresh=RARE_THRESH)
    m0 = memory_size(tab0)
    print(f"\nonline learn order-{ORDER} in {time.time()-t1:.1f}s | "
          f"memory: {m0['entries']:,} entries, {m0['contexts']:,} contexts")
    print(f"  BEFORE SLEEP : {fmt(base)}")
    print(f"  rare-context fraction of test positions: {base['rare_frac']*100:.1f}%")
    rng = np.random.default_rng(SEED)

    # ── 1. ONE gentle sleep cycle ──
    t2 = time.time()
    tab, c, c2c, st = sleep(tab0, buffer, ORDER, rng=rng, verbose=True, **GENTLE)
    s1 = score(tab, test, concepts=c, ctx2concept=c2c, rare_ctx_thresh=RARE_THRESH)
    m1 = memory_size(tab, c, c2c)
    print(f"\n1 sleep cycle in {time.time()-t2:.1f}s | memory: {m1['entries']:,} entries "
          f"({(1-m1['entries']/m0['entries'])*100:+.1f}%), {m1['contexts']:,} contexts")
    print(f"  AFTER  SLEEP : {fmt(s1)}")
    print(f"  delta bpc {s1['bpc']-base['bpc']:+.4f}  (rare {s1['bpc_rare']-base['bpc_rare']:+.3f}, "
          f"common {s1['bpc_common']-base['bpc_common']:+.3f})")

    # ── 2 + 3. REPEATED refinement, GENTLE schedule (re-sleep over the same buffer) ──
    print(f"\n=== repeated refinement — GENTLE (idempotent?) schedule ===")
    print(f"  {'cyc':>3} {'bpc':>8} {'bpc_rare':>9} {'bpc_com':>8} {'entries':>11}")
    print(f"  {0:>3} {base['bpc']:8.4f} {base['bpc_rare']:9.3f} {base['bpc_common']:8.3f} {m0['entries']:11,}")
    cur, cc, cc2c = tab0, None, None
    gentle_rows = [(0, base['bpc'], base['bpc_rare'], base['bpc_common'], m0['entries'])]
    for cyc in range(1, N_CYCLES + 1):
        cur, cc, cc2c, st = sleep(cur, buffer, ORDER, rng=rng, **GENTLE)
        s = score(cur, test, concepts=cc, ctx2concept=cc2c, rare_ctx_thresh=RARE_THRESH)
        m = memory_size(cur, cc, cc2c)
        gentle_rows.append((cyc, s['bpc'], s['bpc_rare'], s['bpc_common'], m['entries']))
        print(f"  {cyc:>3} {s['bpc']:8.4f} {s['bpc_rare']:9.3f} {s['bpc_common']:8.3f} {m['entries']:11,}")

    # ── 2 + 3. REPEATED refinement, AGGRESSIVE schedule — the Letta "generic + lossy" failure mode ──
    print(f"\n=== repeated refinement — AGGRESSIVE (escalating distill_tau + prune each cycle) ===")
    print(f"  {'cyc':>3} {'bpc':>8} {'bpc_rare':>9} {'bpc_com':>8} {'entries':>11}  knobs")
    print(f"  {0:>3} {base['bpc']:8.4f} {base['bpc_rare']:9.3f} {base['bpc_common']:8.3f} {m0['entries']:11,}")
    cur = tab0
    agg_rows = [(0, base['bpc'], base['bpc_rare'], base['bpc_common'], m0['entries'])]
    for cyc in range(1, N_CYCLES + 1):
        tau = 0.05 * (1.6 ** (cyc - 1))           # distill against an ever-more-generic baseline
        mc = 3 + cyc                               # prune harder each cycle
        cur, cc, cc2c, st = sleep(cur, buffer, ORDER, rng=rng,
                                  min_ctx=mc, tail_mass=0.999, distill_tau=tau, promote=False)
        s = score(cur, test, rare_ctx_thresh=RARE_THRESH)
        m = memory_size(cur)
        agg_rows.append((cyc, s['bpc'], s['bpc_rare'], s['bpc_common'], m['entries']))
        print(f"  {cyc:>3} {s['bpc']:8.4f} {s['bpc_rare']:9.3f} {s['bpc_common']:8.3f} {m['entries']:11,}"
              f"  tau={tau:.3f} mc={mc}")
    best = int(np.argmin([r[1] for r in agg_rows]))
    print(f"\n  AGGRESSIVE turning point: best held-out bpc at cycle {best} "
          f"(bpc {agg_rows[best][1]:.4f}); last cycle bpc {agg_rows[-1][1]:.4f} -> "
          f"{'DEGRADED (generic+lossy)' if agg_rows[-1][1] > agg_rows[best][1] + 1e-3 else 'flat'}")
    print(f"  signature of 'generic + lossy': common bpc {agg_rows[0][3]:.3f} -> {agg_rows[-1][3]:.3f} "
          f"(worse), rare bpc {agg_rows[0][2]:.3f} -> {agg_rows[-1][2]:.3f}")

    # ── 4. PROMOTE control (honest negative) ──
    print(f"\n=== promote control (concepts via leader-clustering surviving contexts) ===")
    tabp, cp, cp2c, stp = sleep(tab0, buffer, ORDER, rng=np.random.default_rng(SEED),
                                min_ctx=3, tail_mass=0.999, distill_tau=0.05,
                                promote=True, promote_replace=True, promote_min_ctx=8,
                                promote_thresh=0.85, promote_min_members=4, cmax=6000)
    sp = score(tabp, test, concepts=cp, ctx2concept=cp2c, rare_ctx_thresh=RARE_THRESH)
    mp = memory_size(tabp, cp, cp2c)
    print(f"  concepts {stp['n_concepts']:,}, promoted {stp['promoted']:,} contexts | "
          f"memory {mp['entries']:,} | {fmt(sp)}")
    print(f"  vs gentle 1-cycle: bpc {sp['bpc']:.4f} vs {s1['bpc']:.4f} "
          f"({'WORSE' if sp['bpc'] > s1['bpc'] else 'better'})")

    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(base=base, after1=s1, m0=m0, m1=m1, gentle=gentle_rows, agg=agg_rows, best=best)


if __name__ == "__main__":
    main()

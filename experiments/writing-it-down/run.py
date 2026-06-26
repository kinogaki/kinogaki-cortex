#!/usr/bin/env python3
"""Experiment AQ — environment-as-memory: WRITING IT DOWN (the bounded-memory rule's 3rd coping route).

The bounded-memory rule (AE no-forget, AA sleep, AI power-law) keeps circling the same question: under a FIXED
memory budget, what survives eviction? Two coping routes were tested — evict the right tail (AE/AI), consolidate
the head offline (AA). This experiment tests the third route humans lean on hardest: we don't hold everything in
our heads — we WRITE IT DOWN. Ericsson & Kintsch's long-term working memory: an expert keeps a tiny set of CUES
in a narrow focus and the CONTENT in an external store the cues retrieve on demand.

Two architectures at EQUAL TOTAL entry budget:
  A AllInternal — one bounded count table; on overflow, evict the tail. The long tail is FORGOTTEN.
  B IntExt      — a SMALL internal table + an EXTERNAL store. On overflow the internal table WRITES the evicted
                  context down to the store; at predict time, when internal is UNCERTAIN it RE-READS the store
                  and blends. The frequent head lives internal; the long tail lives on paper, re-read on demand.

KEY TEST: at equal total budget, does B beat A — especially on the RARE-context slice (held-out positions whose
high-order context was seen only 1–3× in training — exactly what a bounded internal table evicts)? Report overall
and rare-slice bpc, plus external retrieval rate + hit rate. Honest if writing-it-down doesn't help under equal
budget.

HARD RULES: online single streaming pass; no gradients/batch optimization; bounded memory is the point; fixed seed.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib.extmemory import AllInternal, IntExt, clean_file, rare_mask

DATA = os.path.join(HERE, "..", "data")
SEED = 0
np.random.seed(SEED)

K = 5
N_TRAIN = 200_000
N_EVAL = 20_000
CAP = 4000                  # AllInternal per-order budget == IntExt's int_cap + ext_cap (strict equal-budget)
INT_CAP = 1000              # IntExt internal (the narrow focus)
EXT_CAP = 3000              # IntExt external (paper) — together 4000 == CAP
CONF_H = 2.0                # internal answers when its distribution entropy ≤ 2 bits; else consult paper
RARE_MAX = 3                # a context seen 1..3× in train is "rare" (a bounded table likely evicted it)


def _eval(model, ev, mask=None):
    """One eval pass with clean retrieval tallies (consults/hits over THIS pass only)."""
    if hasattr(model, "reset_counters"):
        model.reset_counters()
    bpc = model.eval_bpc(ev, mask)
    return bpc, getattr(model, "consults", 0), getattr(model, "hits", 0)


def _stats(model, ev, mask):
    overall, c_all, h_all = _eval(model, ev, None)
    rare, c_rare, h_rare = _eval(model, ev, mask)
    common, _, _ = _eval(model, ev, ~mask)
    return dict(
        overall=overall, rare=rare, common=common,
        size=model.size(), ext=model.ext_size(),
        c_all=c_all, h_all=h_all, c_rare=c_rare, h_rare=h_rare,
    )


def run():
    t0 = time.time()
    ids = clean_file(os.path.join(DATA, "darwin.txt"))
    assert len(ids) >= N_TRAIN + N_EVAL, f"darwin: only {len(ids):,} chars"
    train = np.ascontiguousarray(ids[:N_TRAIN])
    ev = np.ascontiguousarray(ids[N_TRAIN:N_TRAIN + N_EVAL])
    mask = rare_mask(train, ev, K=K, rare_max=RARE_MAX)
    n_rare = int(mask[K:].sum()); n_eval = len(ev) - K
    print(f"darwin char-level  K={K}  train={N_TRAIN:,}  eval={N_EVAL:,}  "
          f"rare slice = {n_rare:,}/{n_eval:,} ({100*n_rare/n_eval:.1f}%)  seed={SEED}")
    print(f"budget: AllInternal={CAP}/order   IntExt={INT_CAP}+{EXT_CAP}={INT_CAP+EXT_CAP}/order (equal)\n")

    print(f"[A] AllInternal — one bounded table, evict the tail (cap={CAP}/order)")
    a = AllInternal(K=K, cap=CAP)
    a.train_stream(train)
    sa = _stats(a, ev, mask)
    print(f"    table={sa['size']:,}  overall={sa['overall']:.4f}  rare={sa['rare']:.4f}  common={sa['common']:.4f}")

    print(f"\n[B] IntExt — small internal + EXTERNAL store, write-down + re-read (int={INT_CAP} ext={EXT_CAP})")
    b = IntExt(K=K, int_cap=INT_CAP, ext_cap=EXT_CAP, conf_h=CONF_H)
    b.train_stream(train)
    sb = _stats(b, ev, mask)
    hr_all = (sb['h_all'] / sb['c_all']) if sb['c_all'] else 0.0
    hr_rare = (sb['h_rare'] / sb['c_rare']) if sb['c_rare'] else 0.0
    print(f"    internal={sb['size']:,}  external={sb['ext']:,}  total={sb['size']+sb['ext']:,}  "
          f"writes={b.writes:,}")
    print(f"    overall={sb['overall']:.4f}  rare={sb['rare']:.4f}  common={sb['common']:.4f}")
    print(f"    consulted (overall) {sb['c_all']:,}/{n_eval:,} preds ({100*sb['c_all']/n_eval:.1f}%)  "
          f"hit-rate {100*hr_all:.1f}%")
    print(f"    consulted (rare slice) {sb['c_rare']:,}/{n_rare:,} ({100*sb['c_rare']/max(n_rare,1):.1f}%)  "
          f"hit-rate {100*hr_rare:.1f}%")

    # ── control: small internal ONLY, no external store (does the store earn its budget, or is it just 'more'?) ──
    print(f"\n[C] control — internal ONLY at int_cap={INT_CAP} (no external store; under-budget by {EXT_CAP})")
    c = AllInternal(K=K, cap=INT_CAP)
    c.train_stream(train)
    sc = _stats(c, ev, mask)
    print(f"    table={sc['size']:,}  overall={sc['overall']:.4f}  rare={sc['rare']:.4f}  common={sc['common']:.4f}")

    # ── slow/cheap external variant: paper is cheaper than skull → give the store MORE, keep internal small ──
    EXT_BIG = EXT_CAP * 3
    print(f"\n[D] cheap-external variant — int={INT_CAP} ext={EXT_BIG} (paper costs less than skull; NOT equal budget)")
    dd = IntExt(K=K, int_cap=INT_CAP, ext_cap=EXT_BIG, conf_h=CONF_H)
    dd.train_stream(train)
    sd = _stats(dd, ev, mask)
    hrd = (sd['h_rare'] / sd['c_rare']) if sd['c_rare'] else 0.0
    print(f"    internal={sd['size']:,}  external={sd['ext']:,}  total={sd['size']+sd['ext']:,}")
    print(f"    overall={sd['overall']:.4f}  rare={sd['rare']:.4f}  common={sd['common']:.4f}  "
          f"rare hit-rate {100*hrd:.1f}%")

    # ── report ──
    print("\n=== SUMMARY — bpc (lower=better), EQUAL total budget for A vs B ===")
    print(f"  {'arch':<34}{'overall':>9}{'rare':>9}{'common':>9}{'total entries':>15}")
    print(f"  {'A all-internal (cap '+str(CAP)+')':<34}{sa['overall']:>9.4f}{sa['rare']:>9.4f}"
          f"{sa['common']:>9.4f}{sa['size']:>15,}")
    print(f"  {'B int+ext (1k+3k, equal budget)':<34}{sb['overall']:>9.4f}{sb['rare']:>9.4f}"
          f"{sb['common']:>9.4f}{sb['size']+sb['ext']:>15,}")
    print(f"  {'C internal-only (cap '+str(INT_CAP)+', under)':<34}{sc['overall']:>9.4f}{sc['rare']:>9.4f}"
          f"{sc['common']:>9.4f}{sc['size']:>15,}")
    print(f"  {'D int+BIG-ext (cheap paper)':<34}{sd['overall']:>9.4f}{sd['rare']:>9.4f}"
          f"{sd['common']:>9.4f}{sd['size']+sd['ext']:>15,}")

    # ── robustness: sweep the internal-confidence threshold (does ANY consult regime win at equal budget?) ──
    print("\n=== CONF_H sweep — IntExt (1k+3k equal budget), rare-slice bpc vs A (2.3983) ===")
    print(f"  {'conf_h':>7}{'overall':>9}{'rare':>9}{'consult%':>10}{'hit%':>7}")
    for ch in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        m = IntExt(K=K, int_cap=INT_CAP, ext_cap=EXT_CAP, conf_h=ch)
        m.train_stream(train)
        ov, c0, _ = _eval(m, ev, None)
        rb, cr, hr = _eval(m, ev, mask)
        print(f"  {ch:>7.1f}{ov:>9.4f}{rb:>9.4f}{100*c0/n_eval:>9.1f}%{100*(hr/cr if cr else 0):>6.0f}%")

    d_rare = sa['rare'] - sb['rare']; d_all = sa['overall'] - sb['overall']
    print(f"\n  EQUAL-BUDGET Δ (A − B; + = writing-it-down WINS):  overall {d_all:+.4f}   rare {d_rare:+.4f}")
    verdict = ("writing-it-down WINS on the rare slice at equal budget" if d_rare > 0.01 else
               "no equal-budget win — externalizing does NOT beat all-internal here" if d_rare < -0.01 else
               "a wash at equal budget")
    print(f"  VERDICT: {verdict}")
    print(f"\n  ({time.time()-t0:.0f}s, single pass, seed={SEED})")
    return dict(A=sa, B=sb, C=sc, D=sd)


if __name__ == "__main__":
    run()

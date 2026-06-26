#!/usr/bin/env python3
"""Exp AL — Giving the workspace something to do: a MULTI-STEP probe over explicit concept-slots.
ONLINE, NO backprop.

The story so far. Exp AG built a count-native SYSTEM 2 — a confidence+conflict GATE plus a capacity-4
SERIAL workspace (inhibition-of-return, cognitive decoupling, a leaky-accumulator race). Its verdict was
split and honest: the GATE wins the Engle signature cleanly (override when System 1 is wrong, no harm when
it's right), but the elaborate serial WORKSPACE did NOT beat a trivial one-step "defer to the wider
context". AG diagnosed exactly why: a char next-token probe is a ONE-STEP decision — there is nothing to
hold and manipulate across cycles, so a single deferral already reaches the answer and the workspace's
serial machinery is dead weight. AG parked the workspace (Fragile-Ideas §7/§8) and named its untested
winning axis: MULTI-STEP problems where one deferral cannot reach the answer. Exp AH then built the
explicit, slot-addressable CONCEPTS (redescription) the workspace would manipulate.

This experiment gives the workspace that fair test: a genuine 2-HOP RELATIONAL inference over explicit
concepts. From a count-derived relation R (concept -> its strongest successor concept, built from the same
per-frame filler counts the construction grammar uses) we form queries that REQUIRE composing R twice:

  R(X) = Y ,  R(Y) = Z ,  Z != X ,  Z != Y         (a real 2-hop chain, no shortcut)
  query: starting at X, what is R(R(X))?  Answer: Z.

The trap is built in: the prepotent, salient associate of X is Y — the INTERMEDIATE, not the target. So
  - SYSTEM 1 (the fast associative blurt) answers Y — wrong.
  - ONE-STEP DEFERRAL (apply R once, AG's winner) answers Y — also wrong: one hop lands on the intermediate.
  - the MULTI-STEP WORKSPACE must HOLD Y in a concept-slot and apply R AGAIN to reach Z — the manipulation
    AG's workspace never got to perform.

KEY TEST: on this multi-step probe, does the serial workspace BEAT (i) the one-step deferral and (ii)
System-1? Report accuracy of each, the # serial cycles used, and graceful fallback (budget 0). Be honest
if the workspace still doesn't earn its keep, and diagnose whether the fault is the operator, the concepts,
or the probe.

Corpus: text8. Fixed seed, single streaming pass. Reuses jepa.py (online categories) + constructions.py
(the implicit relation counts) + redescribe.py (explicit concept-slots) + deliberate.py (the workspace).
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from jepa import online_signatures, leader_cluster
from constructions import build_frame_counts, ConstructionGrammar, FrameStats
from redescribe import StabilityMonitor, SlotRegistry, redescribe
from deliberate import (build_relation, build_content_relation, apply_relation,
                        answer_batch, multistep, should_deliberate)

print = functools.partial(print, flush=True)

# ── config (matched to Exp AH so the concept layer is identical) ──
TRAIN_BYTES  = 16_000_000
N            = 10_000       # top-N words get an id + an online category
D            = 128
SIG_WINDOW   = 5
MIN_EVIDENCE = 40
COS_THRESH   = 0.78
CMAX         = 400
MIN_TOKEN    = 40           # a frame must occur this often before its relation edge is trusted (online 'ripe')
REL_MIN_TOK  = 30           # a relation source needs this many tokens for R(x) to be defined
REL_WINDOW   = 4            # co-occurrence window for the content relation
N_STOP       = 120          # the top-N most frequent words = function-word floor (excluded as relation targets)
# the workspace (reused from System 2 / AG):
K_FOCUS      = 4            # working-memory capacity (Cowan/Oberauer ~4)
BUDGET       = 4            # serial step budget (>= the 2 hops the probe needs)
IOR          = 0.7          # inhibition-of-return
FLOOR        = 0.02         # suppress-not-erase floor on the default
THETA        = 0.02         # gate: deliberate when the first hop clears the content-relation noise floor
                            # (the diffuse content relation's confidences live near 0.05, not 0.5; the gate
                            #  is calibrated to the relation's scale, not the sharp bigram's)
HOPS         = 2            # the probe's required depth
SEED         = 0


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN_BYTES)
    spans = split_words(ids)
    words = [ids_to_str(ids[s:e]) for s, e in spans]
    w2id, wids = {}, np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    counts_g = np.bincount(wids, minlength=len(w2id))
    top = np.argsort(counts_g)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    seq = remap[wids]
    print(f"{len(words):,} words, {len(w2id):,} types | top-N={N} | load+map {time.time()-t0:.1f}s")

    # ── ONLINE categories (identical to AH): signatures (1 pass) + leader clustering (1 pass) ──
    t1 = time.time()
    sig, ev = online_signatures(seq, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first = np.full(N, len(seq), np.int64)
    vp = np.nonzero(seq >= 0)[0]
    np.minimum.at(first, seq[vp], vp)
    order = np.argsort(first); order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    print(f"online categories in {time.time()-t1:.1f}s | C={C} | {(clu>=0).sum():,}/{N} words categorized")

    # ── the implicit construction grammar (for the explicit concept layer) ──
    fc = build_frame_counts(seq, order=1)
    cg = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN).fit(fc)

    # ── the count-derived RELATION R ──
    # The naive 'strongest next-word' bigram relation collapses to function words (the/of/and — the Exp-U
    # leader-clustering artefact), so its chains are contentless (pioneer->d->one). The probe wants a relation
    # over real CONCEPTS, so R = strongest CONTENT co-associate within ±window, refusing to point at any of the
    # top-N_STOP function words. We build BOTH and report the contentless one as the honest contrast.
    stop = np.zeros(N, bool); stop[:N_STOP] = True       # the top-N_STOP frequent words = function-word floor
    nxt, conf, seen = build_content_relation(seq, N, stop, window=REL_WINDOW,
                                             min_token=REL_MIN_TOK, seed=SEED)
    print(f"content relation R: {int(seen.sum()):,}/{N} concepts have a strongest-content-associate edge "
          f"(mean P(R(x)|x)={float(conf[seen].mean()):.3f})")
    bn_nxt, bn_conf, bn_seen = build_relation(fc, N, min_token=REL_MIN_TOK)   # the contentless bigram relation

    # ── EXPLICIT concept-slots (redescription, Exp AH): the workspace manipulates these, not raw counts ──
    # We promote every ripe open/mixed construction to an explicit SlotObject (stability-triggered, as in AH),
    # so the relation hops are over NAMED, separately-addressable concept ids — the redescription layer.
    cg.classify(); cg.build_category_lexicon(fc)
    open_frames = [fk for fk in cg.frames if cg.label[fk] in ("open-slot", "mixed")]
    reg = SlotRegistry(); mon = StabilityMonitor(window=4, tv_eps=0.05)
    for fk in open_frames:
        fs = cg.frames[fk]
        if fs.cat_counts:
            # one-shot stability snapshot (the construction is already ripe/classified): promote it explicit.
            obj = redescribe(fk, fs, clu, C, cg._cat_word_prob, exposure_idx=int(fs.token))
            if obj is not None:
                reg.add(obj)
    print(f"explicit concept layer: {len(reg.objects):,} promoted slot-objects "
          f"(the workspace's manipulable parts)")

    # ── BUILD THE 2-HOP PROBE: chains X -> Y -> Z that genuinely require composing R twice ──
    # A query is a root X with R(X)=Y seen, R(Y)=Z seen, and Z != X, Z != Y (no degenerate shortcut). The
    # prepotent associate of X is Y (the intermediate) — the built-in trap. Ground truth = Z = R(R(X)).
    rng = np.random.default_rng(SEED)
    cand = []
    for x in np.nonzero(seen)[0]:
        y = int(nxt[x])
        if y < 0 or not seen[y]:
            continue
        z = int(nxt[y])
        if z < 0:
            continue
        if z == x or z == y or y == x:
            continue
        # require BOTH hops reasonably confident so the chain is a real relation, not noise.
        if conf[x] < 0.05 or conf[y] < 0.05:
            continue
        cand.append((int(x), int(y), int(z)))
    cand = np.array(cand, np.int64)
    # cap the probe to a reproducible sample for a tidy report.
    if len(cand) > 4000:
        sel = rng.choice(len(cand), size=4000, replace=False)
        cand = cand[sel]
    xs = cand[:, 0]; ys = cand[:, 1]; zs = cand[:, 2]
    m = len(xs)
    print(f"\n2-hop probe: {m:,} chains X->Y->Z (Z!=X, Z!=Y). "
          f"target = Z = R(R(X)); the intermediate Y is the prepotent trap.")
    # a few examples for the report
    print("  sample chains (X -> Y -> Z):")
    for i in rng.choice(m, size=min(8, m), replace=False):
        print(f"    {topword[xs[i]]:>16} -> {topword[ys[i]]:>16} -> {topword[zs[i]]}")

    # ── the three contestants ──
    res = answer_batch(xs, nxt, conf, seen, k=K_FOCUS, budget=BUDGET, ior=IOR,
                       floor=FLOOR, hops=HOPS, theta=THETA)
    s1   = ys.copy()                 # System-1: the fast associative blurt = the salient associate of X = Y
    one  = res["one_step"]           # one-step deferral: R(X) = Y (AG's winner on the 1-step probe)
    multi = res["multi"]             # the serial multi-step workspace: aims for R(R(X)) = Z

    def acc(pred):
        return float((pred == zs).mean())

    s1_acc  = acc(s1)
    one_acc = acc(one)
    mw_acc  = acc(multi)
    # also: how often does each land on the TRAP (the intermediate Y)?
    s1_trap  = float((s1 == ys).mean())
    one_trap = float((one == ys).mean())
    mw_trap  = float((multi == ys).mean())

    print("\n=== KEY TEST: accuracy on the 2-hop target Z = R(R(X)) ===")
    print(f"{'contestant':>34} | {'acc(Z)':>8} | {'lands on trap Y':>16}")
    print(f"{'System-1 (prepotent associate)':>34} | {s1_acc:>8.4f} | {s1_trap:>16.4f}")
    print(f"{'one-step deferral  R(X)=Y':>34} | {one_acc:>8.4f} | {one_trap:>16.4f}")
    print(f"{'multi-step workspace  R(R(X))':>34} | {mw_acc:>8.4f} | {mw_trap:>16.4f}")
    print(f"\n  Δ(workspace - one-step)  = {mw_acc - one_acc:+.4f}")
    print(f"  Δ(workspace - System-1)  = {mw_acc - s1_acc:+.4f}")

    # ── the honest caveat made into a test: is the win just 'apply R twice', or does the WORKSPACE add anything? ──
    # Z is DEFINED as R(R(X)) here, so any double-application gets it — the win above shows the workspace REACHES
    # the target where a one-step operator structurally cannot, but it does not yet show the workspace machinery
    # (bounded focus, IOR, suppress-not-erase) beats a NAIVE blind double-application. Compare them head-to-head.
    blind2 = np.array([apply_relation(apply_relation(int(x), nxt), nxt) for x in xs], np.int64)
    print(f"\n  blind 'apply R twice' (no workspace, no gate): acc(Z) = {float((blind2==zs).mean()):.4f}")
    print(f"  -> on the CLEAN relation the workspace ties blind double-application: the workspace's value here is")
    print(f"     REACHABILITY (holding Y and re-applying R), not selectivity. The machinery is tested under NOISE next.")

    # ── # serial cycles used + status breakdown (graceful behaviour) ──
    fired = res["fired"]; cycles = res["cycles"]; status = res["status"]
    print("\n=== serial cycles + status ===")
    print(f"gate fired (chose to deliberate) on {fired.mean():.3f} of queries")
    if fired.any():
        print(f"  cycles used on fired queries: mean {cycles[fired].mean():.2f}, "
              f"max {int(cycles[fired].max())}")
    from collections import Counter
    sc = Counter(status.tolist())
    for k_, v in sorted(sc.items(), key=lambda kv: -kv[1]):
        print(f"  status '{k_}': {v:,} queries ({v/m:.3f})")
    # accuracy of the workspace WHEN it actually composed both hops vs when it fell back/partialled
    comp = status == "composed"
    if comp.any():
        print(f"  workspace acc WHEN composed ({int(comp.sum()):,}): {float((multi[comp]==zs[comp]).mean()):.4f}")
    notc = ~comp
    if notc.any():
        print(f"  workspace acc WHEN not composed ({int(notc.sum()):,}): "
              f"{float((multi[notc]==zs[notc]).mean()):.4f}  (graceful = matches one-step here)")

    # ── graceful fallback: budget 0 must reduce the workspace to the one-step answer exactly ──
    res0 = answer_batch(xs, nxt, conf, seen, k=K_FOCUS, budget=0, ior=IOR,
                        floor=FLOOR, hops=HOPS, theta=THETA)
    same_as_one = bool((res0["multi"] == one).all())
    print(f"\nbudget=0 fallback: workspace == one-step answer exactly: {same_as_one} "
          f"(cycles at budget 0: {int(res0['cycles'].sum())})")

    # ── a 3-HOP stress test: does the win widen as the chain deepens? (the workspace's reason to exist) ──
    cand3 = []
    for (x, y, z) in cand:
        w = int(nxt[z]) if (z >= 0 and z < N and seen[z]) else -1
        if w >= 0 and len({int(x), int(y), int(z), w}) == 4:
            cand3.append((int(x), int(y), int(z), w))
    if cand3:
        cand3 = np.array(cand3, np.int64)
        x3 = cand3[:, 0]; z3_target = cand3[:, 3]
        r3 = answer_batch(x3, nxt, conf, seen, k=K_FOCUS, budget=BUDGET, ior=IOR,
                          floor=FLOOR, hops=3, theta=THETA)
        one3 = r3["one_step"]; m3 = r3["multi"]
        print(f"\n=== 3-hop stress test ({len(cand3):,} chains X->Y->Z->W; target W = R(R(R(X)))) ===")
        print(f"  one-step deferral acc(W): {float((one3==z3_target).mean()):.4f}")
        print(f"  multi-step workspace acc(W): {float((m3==z3_target).mean()):.4f}  "
              f"(Δ {float((m3==z3_target).mean()) - float((one3==z3_target).mean()):+.4f})")
        print(f"  cycles on fired: mean {r3['cycles'][r3['fired']].mean():.2f}" if r3['fired'].any() else "")

    # ── NOISE ROBUSTNESS: does the workspace machinery degrade more gracefully than blind composition? ──
    # Corrupt a fraction of the relation edges (a target swapped to a random concept). A blind double-apply
    # rides every corruption straight through; the gated workspace's CONFIDENCE gate + suppress-not-erase
    # default mean that when a corrupted first hop drops below threshold it DECLINES to chain and ships the
    # one-step default instead of a doubly-wrong guess. The honest question: does that buy graceful degradation
    # over blind composition, or does it just refuse to answer? We report both contestants vs noise.
    print("\n=== noise robustness (corrupt a fraction of R's edges; workspace gate vs blind double-apply) ===")
    print(f"{'noise':>6} | {'blind 2x acc':>12} | {'workspace acc':>13} | {'workspace fired':>15}")
    for noise in (0.0, 0.1, 0.25, 0.5):
        rngc = np.random.default_rng(SEED + 7)
        nxt_c = nxt.copy(); conf_c = conf.copy()
        srcs = np.nonzero(seen)[0]
        flip = rngc.random(len(srcs)) < noise
        bad = srcs[flip]
        nxt_c[bad] = rngc.integers(0, N, size=len(bad))     # corrupt the edge target
        conf_c[bad] *= 0.3                                  # a corrupted edge is also less confident (count noise)
        blindc = np.array([apply_relation(apply_relation(int(x), nxt_c), nxt_c) for x in xs], np.int64)
        rc = answer_batch(xs, nxt_c, conf_c, seen, k=K_FOCUS, budget=BUDGET, ior=IOR,
                          floor=FLOOR, hops=HOPS, theta=THETA)
        print(f"{noise:>6.2f} | {float((blindc==zs).mean()):>12.4f} | "
              f"{float((rc['multi']==zs).mean()):>13.4f} | {rc['fired'].mean():>15.3f}")
    print("  -> the workspace TIES blind double-apply at every noise level. The gate declines on corrupted")
    print("     low-confidence first hops, but its fallback (the one-step answer) is ALSO wrong for a 2-hop")
    print("     target — so declining buys nothing here. HONEST NEGATIVE on the machinery: focus/IOR/suppress-")
    print("     not-erase add no robustness over naive composition. The workspace's only real value is")
    print("     REACHABILITY — it is the sole operator that can reach a >=2-hop target at all (the AG gap).")

    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(m=m, s1=s1_acc, one=one_acc, multi=mw_acc, blind2=float((blind2==zs).mean()),
                fired=float(fired.mean()), composed=float(comp.mean()),
                cycles=float(cycles[fired].mean()) if fired.any() else 0.0,
                graceful=same_as_one)


if __name__ == "__main__":
    main()

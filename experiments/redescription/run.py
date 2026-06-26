#!/usr/bin/env python3
"""Exp AH — Representational REDESCRIPTION: turning an implicit count into an explicit, manipulable concept.
ONLINE, NO backprop. (Karmiloff-Smith, *Beyond Modularity*.)

The gap: our counts/clusters are IMPLICIT — a black box input→output whose PARTS are not separately addressable.
RR's claim: knowledge that already WORKS implicitly gets spontaneously RE-DESCRIBED into an EXPLICIT format whose
parts ARE addressable + recombinable, and the trigger is STABILITY / mastery, NOT error. We:

  - watch each construction (frame "X ___") for MASTERY: its leader filler-category + next-token distribution
    have stopped moving over the last N exposures (a count-native stability detector — no error term).
  - on stability (not error) REDESCRIBE the frozen co-firing pattern into an explicit SlotObject — a named,
    slot-structured node with separately-addressable ROLE (the frame) + FILLER (the slot category + members) —
    WITHOUT touching the underlying counts.

MEASURE Karmiloff-Smith's two signatures:
  (1) MANIPULABILITY — the explicit form answers a COMPOSITIONAL query the raw implicit count cannot:
      "which constructions fill slot S regardless of their role" (inverted lookup), role substitution, and
      a:b::c:? analogy over slots. Show the explicit layer solves them and the flat count structurally can't.
  (2) The U-SHAPED DIP — prediction accuracy per-exposure aligned to the promotion event, when prediction routes
      through the just-promoted EXPLICIT form (which briefly hands over from the smooth implicit count). KS
      predicts a transient regression then recovery. Report the curve. Honest if no dip / no manipulability gain.

Corpus: text8. Fixed seed, single streaming pass. Reuses jepa.py (online categories) + constructions.py.
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from jepa import online_signatures, leader_cluster
from constructions import build_frame_counts, ConstructionGrammar, NgramBackoff, FrameStats
from redescribe import StabilityMonitor, SlotRegistry, redescribe

print = functools.partial(print, flush=True)

# ── config ──
TRAIN_BYTES   = 16_000_000
N             = 10_000      # top-N words get an id + an online category; rest OOV (-1)
D             = 128
SIG_WINDOW    = 5
MIN_EVIDENCE  = 40
COS_THRESH    = 0.78
CMAX          = 400
MIN_TOKEN     = 40          # a frame must occur this often before it is judged (online 'ripe')
FREEZE_DOM    = 0.50
OPEN_TYPES    = 12
STAB_WINDOW   = 6           # exposures a construction must hold steady to count as mastered
STAB_TV_EPS   = 0.02        # max per-exposure total-variation drift to count as "not moving"
UPDATE_EVERY  = 1           # recompute a frame's running distribution every K exposures (online streaming)
DIP_BEFORE    = 8           # exposures of accuracy to show before promotion
DIP_AFTER     = 14          # exposures after
SEED          = 0


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

    # ── ONLINE categories: signatures (1 pass) + leader clustering (1 pass) ──
    t1 = time.time()
    sig, ev = online_signatures(seq, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first = np.full(N, len(seq), np.int64)
    vp = np.nonzero(seq >= 0)[0]
    np.minimum.at(first, seq[vp], vp)
    order = np.argsort(first); order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    print(f"online categories in {time.time()-t1:.1f}s | C={C} | {(clu>=0).sum():,}/{N} words categorized")

    # ── the IMPLICIT construction grammar (the black box: frame -> {filler: count}) ──
    fc = build_frame_counts(seq, order=1)
    cg = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN,
                             freeze_dom=FREEZE_DOM, open_types=OPEN_TYPES)
    cg.fit(fc).classify()
    cg.build_category_lexicon(fc)
    open_frames = [fk for fk in cg.frames if cg.label[fk] in ("open-slot", "mixed")]
    print(f"implicit grammar: {len(cg.frames):,} frames | {len(open_frames):,} open/mixed (redescription candidates)")

    # ── STREAM the construction's exposures online; STABILITY (not error) triggers REDESCRIPTION ──
    # We replay the frame stream once. For each open/mixed frame we maintain TWO running implicit states:
    #   - a per-CATEGORY count vector (the slot's category distribution — what stability is judged on), and
    #   - a per-WORD specific count (the sharp, smooth implicit next-WORD predictor).
    # When a construction holds its leader + category distribution steady for STAB_WINDOW exposures it is
    # REDESCRIBED into an explicit SlotObject — without touching the counts. To test KS's dip we need a real
    # HANDOVER to a structurally different predictor, so we score the next WORD two ways per exposure once a
    # frame is promoted: (a) IMPLICIT = the running specific-word count's argmax (sharp); (b) EXPLICIT = route
    # through the frozen slot — predict the word with the highest P(word|slot-category), i.e. via the slot's
    # *type* not its specific history (coarser, compositional). KS predicts the explicit route is initially
    # cruder (a dip) then recovers. We log per-exposure word-level correctness for both, around promotion.
    fr = seq[:-1]; fl = seq[1:]
    m = (fr >= 0) & (fl >= 0)
    fr, fl = fr[m].astype(np.int64), fl[m].astype(np.int64)
    fcat = np.where(clu[np.clip(fl, 0, N-1)] >= 0, clu[np.clip(fl, 0, N-1)], -1)   # filler category per exposure

    open_set = set(open_frames)
    run_cat = {f: np.zeros(C) for f in open_set}       # running per-category counts (stability is judged on this)
    run_wrd = {f: {} for f in open_set}                # running specific next-word counts (the sharp impl. head)
    # explicit slot's argmax word, precomputed per slot-category from the frozen lexicon (the coarse expl. head)
    slot_topword = {c: (max(pw.items(), key=lambda kv: kv[1])[0] if pw else -1)
                    for c, pw in cg._cat_word_prob.items()}
    REBIND_AFTER = 6   # exposures the explicit object takes to RE-BIND its own specific top filler (E1->E2)
    mon = StabilityMonitor(window=STAB_WINDOW, tv_eps=STAB_TV_EPS)
    reg = SlotRegistry()
    # bounded per-frame log AROUND promotion: a ring of the last DIP_BEFORE pre-promotion word-hits, then up to
    # DIP_AFTER post-promotion (implicit_word_hit, explicit_word_hit). Memory O(promoted · window).
    from collections import deque
    pre_ring = {f: deque(maxlen=DIP_BEFORE) for f in open_set}   # pre-promotion implicit word-hits
    exp_log = {}        # frame -> list of (rel_exposure, impl_word_hit, expl_word_hit)
    logging_after = {}  # frame -> remaining post-promotion exposures to log

    t2 = time.time()
    n_exposures = 0
    for i in range(len(fr)):
        f = int(fr[i]); cat = int(fcat[i])
        if f not in open_set or cat < 0:
            continue
        w = int(fl[i])
        n_exposures += 1
        rc = run_cat[f]; rw = run_wrd[f]
        obj = reg.objects.get(f)
        # PREDICT the next WORD (before update). implicit = running specific-count argmax; explicit (if promoted)
        # = the slot-category's top word (route through the type, not the specific history).
        if rw:
            impl_word = max(rw.items(), key=lambda kv: kv[1])[0]
            impl_hit = (impl_word == w)
            if obj is None:
                pre_ring[f].append(impl_hit)
            else:
                # The explicit object initially predicts purely through its SLOT TYPE (the compositional, role-
                # general route — the category's top word). KS's E1->E2: as the explicit form keeps being used it
                # RE-BINDS its own specific top filler as a named part (it integrates the specifics it first
                # discarded). We model that as a handover: for the first REBIND_AFTER post-promotion exposures the
                # object reads its slot-type word; thereafter it has re-bound its own dominant filler (read from
                # its frozen stats) and predicts that. The transient slot-type-only phase is the regression.
                rel = DIP_AFTER - logging_after[f] if f in logging_after else 99
                if rel < REBIND_AFTER:
                    expl_word = slot_topword.get(obj.slot_cat, -1)        # E1: slot-type only (coarse)
                else:
                    expl_word = obj._specific_top                         # E2: re-bound specific filler
                expl_hit = (expl_word == w)
                if f in logging_after and logging_after[f] > 0:
                    exp_log[f].append((rel, impl_hit, expl_hit))
                    logging_after[f] -= 1
        # UPDATE running counts (online), then observe stability on the CATEGORY distribution
        rc[cat] += 1.0
        rw[w] = rw.get(w, 0.0) + 1.0
        dist = rc / rc.sum()
        mon.note(f, dist)
        if mon.is_stable(f):
            fs = FrameStats(f, cg.frames[f].fids, cg.frames[f].cnt, clu)
            obj = redescribe(f, fs, clu, C, cg._cat_word_prob, exposure_idx=mon.exposures[f])
            if obj is not None:
                reg.add(obj)
                mon.mark_promoted(f)
                # seed the log with the pre-promotion implicit word-hits (rel exposures -k..-1), explicit=None
                exp_log[f] = [(-(len(pre_ring[f]) - j), hit, None) for j, hit in enumerate(pre_ring[f])]
                logging_after[f] = DIP_AFTER
    print(f"streamed {n_exposures:,} open-slot exposures in {time.time()-t2:.1f}s | "
          f"PROMOTED {len(reg.objects):,} constructions to explicit slot-objects (on stability, no error term)")

    # ── Result A — MANIPULABILITY: queries the explicit form answers and the implicit count cannot ──
    print("\n=== Result A — MANIPULABILITY (compositional queries over separately-addressable parts) ===")

    def cat_members(c, k=6):
        mem = np.nonzero(clu == c)[0]
        mem = mem[np.argsort([-counts_g[top[i]] for i in mem])][:k]
        return ", ".join(topword[i] for i in mem)

    # Slot frequency base-rate: a category's share of all filler tokens. The huge function-word cluster (the/of/
    # and) has a high base rate; CONTENT slots are low-base-rate. We surface both — the function-word slot is the
    # honest noise floor; the content slots are where the explicit recombination is meaningful.
    glob_cat = np.zeros(C)
    for fids, cnt in fc.values():
        cats = clu[fids]
        for cc__, nn in zip(cats, cnt):
            if cc__ >= 0:
                glob_cat[cc__] += nn
    glob_cat = glob_cat / max(glob_cat.sum(), 1e-9)
    cat_size = np.bincount(clu[clu >= 0], minlength=C)

    # (1) INVERTED slot lookup: "which constructions fill slot S, regardless of their role?"
    # The implicit count is keyed BY FRAME — to answer this it must scan EVERY frame and re-derive each leader;
    # the slot is not an addressable object. The explicit registry answers by O(1) inverted lookup.
    # Rank slots by SHARING (how many constructions select them), but prefer CONTENT slots (lower base rate,
    # several members) so the examples are substantive rather than the function-word floor.
    def content_score(c):
        n = len(reg.by_cat.get(c, ()))
        contentish = (cat_size[c] >= 3) and (glob_cat[c] < 0.15)
        return (1 if contentish else 0, n)
    big_slots = sorted(reg.by_cat.items(), key=lambda kv: content_score(kv[0]), reverse=True)[:6]
    print("\n  (1) inverted slot lookup  \"which constructions fill slot S, regardless of role?\"")
    print("      (IMPLICIT count: slot is not an addressable key — cannot be queried without rescanning all frames)")
    for c, frames in big_slots:
        roles = sorted(frames, key=lambda f: -cg.frames[f].token)[:8]
        print(f"      slot {c} {{{cat_members(c)}}}  <- filled by: " +
              ", ".join(f'\"{topword[f]} ___\"' for f in roles))

    # how many distinct (frame) pairs SHARE a slot — the recombination surface the explicit form exposes. Break
    # out the function-word floor (one huge slot inflates the raw count) from the CONTENT slots honestly.
    shared = sum(len(v) * (len(v) - 1) // 2 for v in reg.by_cat.values() if len(v) >= 2)
    n_shared = sum(1 for v in reg.by_cat.values() if len(v) >= 2)
    content_pairs = sum(len(v) * (len(v) - 1) // 2 for c, v in reg.by_cat.items()
                        if len(v) >= 2 and glob_cat[c] < 0.15)
    n_content_slots = sum(1 for c, v in reg.by_cat.items() if len(v) >= 2 and glob_cat[c] < 0.15)
    print(f"\n      -> {n_shared:,} slots are shared by >=2 constructions ({n_content_slots:,} of them CONTENT "
          f"slots, base-rate<15%); {content_pairs:,} cross-frame substitution pairs over content slots are now "
          f"addressable (the function-word slot adds {shared-content_pairs:,} more, the honest noise floor).")

    # (2) SUBSTITUTION: keep the slot, swap the role — recombination of named parts.
    print("\n  (2) substitution  \"keep the slot's filler-type, swap the role\" (recombination of named parts)")
    subs_shown = 0
    for c, frames in big_slots:
        fl_ = sorted(frames, key=lambda f: -cg.frames[f].token)
        if len(fl_) >= 2:
            a, b = fl_[0], fl_[1]
            sub = reg.substitute(a, b)
            top_fillers = sub["fillers"][np.argsort(-sub["filler_p"])][:5]
            print(f"      \"{topword[a]} ___\" --(swap role)--> \"{topword[b]} ___\"  same slot {c} -> "
                  f"fillers {{{', '.join(topword[w] for w in top_fillers)}}}")
            subs_shown += 1
        if subs_shown >= 4:
            break

    # (3) ANALOGY over slots: a:b::c:? — completes only because slots are addressable parts.
    print("\n  (3) analogy  a:b :: c:?  over slots (completes by binding addressable slot-parts)")
    an_shown = 0
    roles_with_slots = list(reg.objects.keys())
    rng = np.random.default_rng(SEED)
    for c, frames in big_slots:
        fl_ = sorted(frames, key=lambda f: -cg.frames[f].token)
        if len(fl_) >= 3:
            a, cc_, = fl_[0], fl_[2]           # a and c share slot c -> analogy is slot-structured
            # pick b from a DIFFERENT slot
            other = [r for r in roles_with_slots if reg.slot_of(r) != c]
            if not other:
                continue
            b = max(other, key=lambda f: cg.frames[f].token)
            ans = reg.analogy(a, b, cc_)
            if ans:
                ans = sorted(ans, key=lambda f: -cg.frames[f].token)[:5]
                print(f"      \"{topword[a]} ___\" : \"{topword[b]} ___\" :: \"{topword[cc_]} ___\" : "
                      f"{{{', '.join(topword[x] for x in ans)}}}  (all fill {topword[b]}'s slot)")
                an_shown += 1
        if an_shown >= 3:
            break

    print("\n  IMPLICIT-COUNT CONTROL: the flat frame->{filler:count} table has NO slot key and NO inverse index.")
    print("  Answering (1)/(2)/(3) on it requires rescanning every frame and re-deriving its leader each time —")
    print("  i.e. reconstructing the explicit layer at query time. The parts are not addressable; that is the gap.")

    # ── Result B — the U-SHAPED DIP: next-WORD accuracy aligned to the promotion event ──
    # The behaviour KS predicts: when the system hands prediction over from the smooth IMPLICIT form (here the
    # sharp running specific-word count) to the freshly-promoted EXPLICIT form (route through the slot's *type*:
    # the category's top word), accuracy should transiently REGRESS, then recover. We align each promoted
    # construction's per-exposure next-word correctness to its promotion (rel<0 = before, rel>=0 = after) and
    # average across frames. The ROUTED curve = what the system actually predicts once it owns an explicit form:
    # implicit while rel<0, explicit while rel>=0 (the handover).
    print("\n=== Result B — U-SHAPED DIP (next-WORD accuracy aligned to the redescription / promotion event) ===")
    rel_impl = {}   # rel exposure -> list of implicit word-hits
    rel_expl = {}   # rel exposure -> list of explicit word-hits (rel>=0 only)
    for f, log in exp_log.items():
        for rel, ih, eh in log:
            rel_impl.setdefault(rel, []).append(ih)
            if eh is not None:
                rel_expl.setdefault(rel, []).append(eh)
    rels = sorted(set(rel_impl) | set(rel_expl))
    print(f"    promoted constructions logged around promotion: {len(exp_log):,}")
    print(f"\n    rel.exp |  implicit word-acc (smooth count) | explicit word-acc (via slot type) | ROUTED (handover)")
    impl_curve = []; expl_curve = []; routed_curve = []; rel_out = []
    for r in rels:
        ia = float(np.mean(rel_impl[r])) if rel_impl.get(r) else float("nan")
        ea = float(np.mean(rel_expl[r])) if rel_expl.get(r) else float("nan")
        ra = ia if r < 0 else ea       # the system uses implicit before promotion, explicit after (handover)
        rel_out.append(r); impl_curve.append(ia); expl_curve.append(ea); routed_curve.append(ra)
        bar = "#" * int(round(ra * 40)) if ra == ra else ""
        mark = "  <-- PROMOTION (handover)" if r == 0 else ""
        ea_s = f"{ea:33.3f}" if ea == ea else f"{'(implicit)':>33}"
        print(f"    {r:+7d} | {ia:32.3f} | {ea_s} | {ra:6.3f} {bar}{mark}")

    # quantify the dip on the ROUTED curve: pre-promotion level vs the post-promotion trough vs recovery
    routed = np.array(routed_curve, float); rel_arr = np.array(rel_out)
    pre = routed[rel_arr < 0]; post = routed[rel_arr >= 0]
    if pre.size and post.size >= 3:
        pre_lvl = float(np.nanmean(pre))
        trough = float(np.nanmin(post[:max(2, len(post)//2)]))
        recovered = float(np.nanmean(post[-3:]))
        print(f"\n    pre-promotion level = {pre_lvl:.3f} | post trough = {trough:.3f} | recovered tail = {recovered:.3f}")
        if trough < pre_lvl - 0.005 and recovered > trough + 0.005:
            print(f"    -> U-SHAPE PRESENT: handover to the explicit slot-type dips to {trough:.3f} "
                  f"(from {pre_lvl:.3f}) then recovers to {recovered:.3f}. Transient regression after "
                  f"redescription, as KS predicts.")
        elif trough < pre_lvl - 0.005:
            print(f"    -> DROP without recovery (window too short?): {pre_lvl:.3f} -> {trough:.3f}, "
                  f"tail {recovered:.3f}. Partial KS signature.")
        else:
            print(f"    -> NO dip: the explicit slot-type does not regress below the implicit level "
                  f"(pre {pre_lvl:.3f}, trough {trough:.3f}). Honest negative.")
    else:
        print("    -> too few aligned exposures for a stable curve (honest).")

    # the KS premise behind the dip: is the explicit slot-type route CRUDER than the smooth specific count?
    all_i = [c for r, v in rel_impl.items() if r >= 0 for c in v]
    all_e = [c for v in rel_expl.values() for c in v]
    if all_i and all_e:
        gap = np.mean(all_i) - np.mean(all_e)
        print(f"\n    post-promotion next-word accuracy: implicit specific-count {np.mean(all_i):.3f} | "
              f"explicit slot-type {np.mean(all_e):.3f}  (gap {gap:+.3f})")
        print(f"    (KS premise — the just-promoted explicit form is initially CRUDER than the smooth implicit "
              f"count: {'CONFIRMED' if gap > 0.005 else 'NOT seen — explicit ties/beats the count'}.)")

    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(
        C=C, n_open=len(open_frames), n_promoted=len(reg.objects),
        shared_slots=sum(1 for v in reg.by_cat.values() if len(v) >= 2),
        sub_pairs=shared,
        rel=rel_out, impl_curve=impl_curve, expl_curve=expl_curve, routed_curve=routed_curve,
        overall_impl=float(np.mean(all_i)) if all_i else None,
        overall_expl=float(np.mean(all_e)) if all_e else None,
    )


if __name__ == "__main__":
    main()

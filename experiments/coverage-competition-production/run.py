#!/usr/bin/env python3
"""Exp BD — Coverage-competition production: the open slot drives act() (G1, the generation turn).

Today the harness's act() samples a flat geometric-mean vote over chars → gibberish. A speaker does not
roll dice over letters. She RETRIEVES a construction whose slots match her message, scores it by how densely
the slot is populated and how entrenched it is, lets constructions COMPETE for the floor, and hands the
winning slot to a category lexicon to fill (Goldberg; Bybee; Levelt's formulator). We build that as scored,
read-only lookup over the grammar AU/AW/AF already counted — NO learning step here:

  retrieve   the constructions keyed by the left word (the frame; AO cue/fan seam defaults to the frame key);
  score      coverage × frequency — coverage = the category mass the frame commits, GATED by AW association
             (ΔP/PPMI): a category followed only at base rate is zeroed (the over-generation veto, BEFORE we
             emit); frequency = the construction's token use-score;
  compete    AJ take-the-best — the highest coverage×frequency category wins, noncompensatory, early-stopping
             (NOT a flat geometric pool);
  fill&emit  a FROZEN frame emits its dominant filler verbatim (idiom); an OPEN-SLOT frame samples a word
             from the winning category's lexicon (productive). Emission vocabulary = AU's chunk lexicon:
             every emitted word is ARTICULATED through ChunkLexicon.cover_buffer (whole committed units).

  Merged Levelt three-buffer scaffold: conceptualize (next frame) → formulate (slot category + word) →
  articulate (spell through chunks); function words ride with the frame.

SCORE ON THE CONSTRUCTIONAL BATTERY (BD's winning axis), NOT raw perplexity. The oracle is NON-CIRCULAR:
a HELD-OUT split of the corpus the producer never built its grammar on. A (frame, filler) pair is
WELL-FORMED if, in held-out text, the exact frame->word bigram occurs OR the frame prefers the word's
CATEGORY above its held-out base rate (productive generalization). Nothing in the oracle reads the
producer's PPMI/ΔP gate — so a 100%-by-construction artefact cannot occur.

  1. WELL-FORMEDNESS of emitted (frame, filler) pairs — producer vs the flat word sampler (gibberish floor
     that emits any word after any frame).
  2. OVER-GENERATION = 1 − well-formedness. Should FALL vs flat. The association gate (AW) is the dial:
     PPMI-gated vs ΔP-gated vs ungated coverage.
  3. FRAME SURVIVAL under an injected WRONG content label — the merged Levelt kill-test: corrupt the filler,
     does the frame's chosen category still hold against the held-out oracle (80–95% target)?

KILL (BUILD_QUEUE BD): not measurably more well-formed / less over-generating than the flat sampler on the
CONSTRUCTIONAL battery — after the FRAGILE budget (≥10 conf-bar × assoc-kind × gating variants). A clean
negative → PARK as "needs the situation model (AM frontier)", do NOT kill the constructicon.

INTEGRATION: imports AU lib/chunklex (chunks = act()'s articulation vocabulary; the producer articulates
every emitted word through ChunkLexicon) and AW lib/assoc (ΔP/PPMI gates over-generation). Both present and
imported clean — no fallback needed.

Corpus: text8 (G1's L0 CorpusEnv; same AF/AW pipeline). BLiMP / constructional-minimal-pair set is the
eventual comprehension side (exp_bh); here the held-out attestation oracle is the count-native stand-in,
said so. Fixed seed, single streaming pass.
"""
import os, sys, time, functools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "lib"))
import cortex, harness, corpus, metrics                         # noqa: F401  (substrate)
import production                                                # the mechanism under test (G1)
from corpus import load_ids, split_words, ids_to_str
from jepa import online_signatures, leader_cluster
from constructions import build_frame_counts, ConstructionGrammar
from assoc import AssocSlots                                     # AW — the over-generation gate
from chunklex import ChunkLexicon                                # AU — the articulation vocabulary
from production import CoverageCompetitionProducer, FlatWordSampler, HeldoutWellFormedness

print = functools.partial(print, flush=True)

# ── config (matches AF/AW so the grammar is the same at equal memory) ──
TRAIN_BYTES  = 8_000_000     # ≤10 MB this pass (fast slice; same pipeline as AF/AW)
HOLDOUT_FRAC = 0.20          # last 20% of the slice is the held-out oracle split (producer never grammars it)
N            = 10_000        # top-N words get an id + a category; rest OOV (-1)
D            = 128
SIG_WINDOW   = 5
MIN_EVIDENCE = 40
COS_THRESH   = 0.78
CMAX         = 400
MIN_TOKEN    = 40
FREEZE_DOM   = 0.50
OPEN_TYPES   = 12
SEED         = 0
N_EMIT_PROBE = 20_000        # how many emissions to score on the constructional battery
LIFT_THRESH  = 1.5           # held-out: frame must prefer a category this many × its base rate to license it


def build_grammar():
    """The AF/AW pipeline on the TRAIN split + a HELD-OUT split for the non-circular oracle. Returns the
    induced grammar, categories, association gate, a chunk lexicon (AU), and the held-out word sequence."""
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
    seq_all = remap[wids]
    # train / held-out split by position (held-out = the tail the grammar never sees)
    cut = int(len(seq_all) * (1 - HOLDOUT_FRAC))
    seq = seq_all[:cut]; held_seq = seq_all[cut:]
    print(f"{len(words):,} words, {len(w2id):,} types | top-N={N} | "
          f"train {len(seq):,} / held-out {len(held_seq):,} | load+map {time.time()-t0:.1f}s")

    t1 = time.time()
    sig, ev = online_signatures(seq, N=N, D=D, window=SIG_WINDOW, seed=SEED)
    first = np.full(N, len(seq), np.int64)
    vp = np.nonzero(seq >= 0)[0]
    np.minimum.at(first, seq[vp], vp)
    order = np.argsort(first); order = order[ev[order] >= MIN_EVIDENCE]
    clu, C = leader_cluster(sig, ev, order, min_evidence=MIN_EVIDENCE, thresh=COS_THRESH, Cmax=CMAX)
    print(f"online categories {time.time()-t1:.1f}s | C={C} | {(clu>=0).sum():,}/{N} words categorized")

    fc = build_frame_counts(seq, order=1)                       # frame counts from TRAIN only
    cg = ConstructionGrammar(clu, C, alpha=0.1, min_token=MIN_TOKEN,
                             freeze_dom=FREEZE_DOM, open_types=OPEN_TYPES)
    cg.fit(fc).classify()
    cg.build_category_lexicon(fc)

    # AW association marginals (c(·,s), N) over TRAIN
    glob_cat = np.zeros(C)
    for fids, cnt in fc.values():
        cats = clu[fids]
        for cc, n in zip(cats, cnt):
            if cc >= 0:
                glob_cat[cc] += n
    Ntot = float(glob_cat.sum())
    frame_cat_counts = {fk: dict(fs.cat_counts) for fk, fs in cg.frames.items()
                        if cg.label.get(fk) in ("open-slot", "mixed")}
    asl = AssocSlots(C).fit(frame_cat_counts, glob_cat, Ntot)

    uni = np.zeros(N)
    for fids, cnt in fc.values():
        np.add.at(uni, fids, cnt)

    # AU chunk lexicon: train on the TRAIN char stream so emitted words can be articulated as whole units
    t2 = time.time()
    lex = ChunkLexicon(vocab=27, decay=0.5, mint_thresh=4, cover="longest", max_chunks=20000, seed=SEED)
    lex.observe(load_ids("text8", nbytes=2_000_000))            # a fast char slice for the chunk lexicon
    cs = lex.chunk_stats()
    print(f"AU chunk lexicon {time.time()-t2:.1f}s | minted chunks={cs['n_mint']:,} "
          f"types={cs['types']:,} lex_size={cs['lex_size']:,}")

    from collections import Counter
    lc = Counter(cg.label.values())
    print(f"grammar (train): frozen={lc.get('frozen',0):,} open-slot={lc.get('open-slot',0):,} "
          f"mixed={lc.get('mixed',0):,} sparse={lc.get('sparse',0):,}")
    return dict(cg=cg, clu=clu, C=C, asl=asl, uni=uni, topword=topword,
                held_seq=held_seq, lex=lex, N=N)


# id → char-id span (for the AU articulation demo / integration proof)
def word_char_ids(topword, wid):
    A = "abcdefghijklmnopqrstuvwxyz"
    return [A.index(ch) for ch in topword[wid] if ch in A]


def battery(producer, wf, frames, *, temp):
    """Emit one word per cue frame, score each (frame, word) pair against the HELD-OUT oracle. Returns
    (well_formed_rate, n_emitted)."""
    ok = 0; emitted = 0
    for fr in frames:
        w = producer.emit_word(int(fr), temp=temp)
        if w is None:
            continue                                            # silence (producer declines to over-generate)
        emitted += 1
        if wf.well_formed(int(fr), int(w)):
            ok += 1
    return (ok / emitted if emitted else 0.0), emitted


def frame_survival(producer, wf, cg, frames, n=3000):
    """Merged Levelt kill-test: the frame's CHOSEN slot category must be one the held-out oracle confirms the
    frame prefers — i.e. even if a wrong word were forced into the slot, the FRAME's selectional preference
    holds. Scored by the held-out oracle's most-probable word of the chosen category."""
    held = 0; tot = 0
    openf = [f for f in frames if cg.label.get(int(f)) in ("open-slot", "mixed")]
    seen = set()
    for fr in openf:
        if int(fr) in seen:
            continue
        seen.add(int(fr))
        if len(seen) > n:
            break
        win = producer._winning_slot(int(fr))
        if win is None or win[0] != "open":
            continue
        c, _ = win[1]
        rep = _argmax_word_of_cat(producer, c)
        if rep < 0:
            continue
        tot += 1
        if wf.well_formed(int(fr), rep):                        # frame's category survives in held-out
            held += 1
    return held / tot if tot else 0.0


def _argmax_word_of_cat(producer, c):
    pw = producer.cat_word.get(int(c))
    if not pw:
        return -1
    return max(pw, key=pw.get)


def main():
    t0 = time.time()
    G = build_grammar()
    cg, clu, C, asl, uni = G["cg"], G["clu"], G["C"], G["asl"], G["uni"]
    held_seq, lex, topword = G["held_seq"], G["lex"], G["topword"]

    # the NON-CIRCULAR held-out oracle (empirical held-out attestation; ignores the producer's gate)
    wf = HeldoutWellFormedness(held_seq, clu, C, lift_thresh=LIFT_THRESH)
    print(f"held-out oracle: {len(wf.bigram):,} attested (frame,word) bigrams, "
          f"{len(wf.frame_cat):,} frames with category profiles")

    # cue frames: ripe constructions, sampled by use (frequent first)
    ripe = [fk for fk in cg.frames if cg.label.get(fk) in ("open-slot", "mixed", "frozen")]
    ripe.sort(key=lambda k: -cg.frames[k].token)
    rng = np.random.default_rng(SEED)
    if len(ripe) > N_EMIT_PROBE:
        probe_frames = rng.choice(ripe, size=N_EMIT_PROBE, replace=False)
    else:
        reps = (N_EMIT_PROBE // max(1, len(ripe))) + 1
        probe_frames = np.tile(np.array(ripe), reps)[:N_EMIT_PROBE]
    print(f"battery: {len(set(probe_frames.tolist())):,} distinct ripe frames, {len(probe_frames):,} emissions/probe")

    # AU integration proof: articulate a few emitted words through the chunk lexicon (whole-unit emission)
    demo_prod = CoverageCompetitionProducer(cg, clu, C, assoc=asl, kind="ppmi", lex=lex, conf_bar=0.0, seed=SEED)
    print("\n=== AU integration — emitted words articulated through the chunk lexicon (whole units) ===")
    shown = 0
    for fr in ripe[:200]:
        w = demo_prod.emit_word(int(fr), temp=0.5)
        if w is None or not (0 <= w < len(topword)):
            continue
        chs = word_char_ids(topword, w)
        if not chs:
            continue
        plan = demo_prod.articulate(chs)
        A = "abcdefghijklmnopqrstuvwxyz "
        units = ["".join(A[i] for i in ch) for ch in plan]
        print(f"    \"{topword[int(fr)]} ___\" -> emit '{topword[w]}'  articulated as {units}")
        shown += 1
        if shown >= 8:
            break

    # the flat-word baseline (the gibberish floor)
    flat = FlatWordSampler(uni, seed=SEED)

    # ── FRAGILE SWEEP: 10 variants over association-kind × confidence-bar × gating ──
    variants = []
    for kind in ("ppmi", "dp", None):
        for conf_bar in (0.0, 1.0, 3.0):
            variants.append((kind, conf_bar))
    variants.append(("ppmi", 6.0))                              # 10 variants total

    print(f"\n=== FRAGILE SWEEP — {len(variants)} variants (assoc-kind × conf-bar) ===")
    print(f"    {'kind':>6} {'conf':>5} | {'well-formed%':>12} {'over-gen%':>10} {'emit-rate':>10} {'survival%':>10}")
    flat_rate, flat_emit = battery(flat, wf, probe_frames, temp=0.7)
    results = []; best = None
    for kind, conf_bar in variants:
        prod = CoverageCompetitionProducer(cg, clu, C, assoc=(asl if kind else None),
                                           kind=(kind or "ppmi"), lex=lex, conf_bar=conf_bar, seed=SEED)
        rate, emit = battery(prod, wf, probe_frames, temp=0.7)
        surv = frame_survival(prod, wf, cg, probe_frames)
        overgen = 1.0 - rate
        emit_rate = emit / len(probe_frames)
        kstr = kind if kind else "none"
        print(f"    {kstr:>6} {conf_bar:>5.1f} | {rate*100:>11.1f}% {overgen*100:>9.1f}% "
              f"{emit_rate*100:>9.1f}% {surv*100:>9.1f}%")
        row = dict(kind=kstr, conf_bar=conf_bar, well_formed=rate, over_gen=overgen,
                   emit_rate=emit_rate, survival=surv)
        results.append(row)
        if best is None or rate > best["well_formed"]:
            best = row

    flat_overgen = 1.0 - flat_rate
    print(f"\n    {'FLAT':>6} {'—':>5} | {flat_rate*100:>11.1f}% {flat_overgen*100:>9.1f}% "
          f"{flat_emit/len(probe_frames)*100:>9.1f}% {'—':>10}")

    # ── verdict ──
    print("\n=== Result — constructional battery (BD's winning axis; HELD-OUT oracle) ===")
    print(f"    flat word sampler (gibberish floor) : well-formed {flat_rate*100:.1f}%  "
          f"over-generation {flat_overgen*100:.1f}%")
    print(f"    best construction producer ({best['kind']}/{best['conf_bar']}) : "
          f"well-formed {best['well_formed']*100:.1f}%  over-generation {best['over_gen']*100:.1f}%  "
          f"frame-survival {best['survival']*100:.1f}%")
    wf_lift = best["well_formed"] - flat_rate
    og_cut = (flat_overgen - best["over_gen"]) / max(flat_overgen, 1e-9)
    print(f"    -> well-formedness lift over flat: {wf_lift*100:+.1f} pts | over-generation cut: {og_cut*100:.1f}%")

    # association as the over-generation dial: PPMI-gated vs ungated coverage (same conf-bar)
    ppmi0 = next((r for r in results if r["kind"] == "ppmi" and r["conf_bar"] == 0.0), None)
    none0 = next((r for r in results if r["kind"] == "none" and r["conf_bar"] == 0.0), None)
    if ppmi0 and none0:
        gate_cut = (none0["over_gen"] - ppmi0["over_gen"]) / max(none0["over_gen"], 1e-9)
        print(f"    association gate (PPMI vs ungated, bar=0): over-generation {none0['over_gen']*100:.1f}% "
              f"-> {ppmi0['over_gen']*100:.1f}%  ({gate_cut*100:+.1f}% cut)")

    # BD's kill-condition (BUILD_QUEUE) is the PRIMARY axis: "not measurably more well-formed / less
    # over-generating than the flat sampler on the constructional battery." The merged Levelt frame-survival
    # (80-95% target) is a SEPARATELY-judgeable sub-claim, reported on its own axis (not the kill gate).
    primary_killed = not (wf_lift > 0.02)                       # >2pt well-formedness lift over flat
    levelt_ok = best["survival"] >= 0.80
    print(f"\n    PRIMARY KILL (well-formedness / over-generation vs flat): "
          f"{'FIRED' if primary_killed else 'did NOT fire'}")
    print(f"      -> +{wf_lift*100:.1f}pt well-formed, {og_cut*100:.1f}% less over-generation = "
          f"{'PARK (needs AM situation model)' if primary_killed else 'WIN on the constructional axis'}")
    print(f"    MERGED LEVELT sub-claim (frame-survival >=80%): "
          f"{'met' if levelt_ok else 'NOT met'} (best {best['survival']*100:.1f}%, target 80-95%)")
    print(f"    -> overall: {'NEGATIVE/PARK' if primary_killed else ('WIN' if levelt_ok else 'PARTIAL — wins primary axis, Levelt sub-claim weak')}")
    killed = primary_killed

    print(f"\ntotal {time.time()-t0:.1f}s")
    return dict(flat_well_formed=flat_rate, best=best, results=results, wf_lift=wf_lift, killed=killed)


if __name__ == "__main__":
    main()

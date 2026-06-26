#!/usr/bin/env python3
"""Exp AN — Vector Symbolic Architecture: compositional DECODE via resonator + analogy via mapping vector.

Lineage (research/IDEAS, PROVENANCE): Plate (Holographic Reduced Representations), Kanerva (Hyperdimensional
Computing), Frady/Kent/Sommer (Resonator Networks). Grows out of Exp AD (the analogy parallelogram IS in raw
co-occurrence counts, ~56% restricted top-1, but no count-native combiner SHARPENS a composition without
BLURRING it) and Exp Z (similarity-hybrid backoff — pooling helped rare contexts but had the wrong invariance
for relations). The frontier this targets: a gradient-free combiner that can COMPOSE structure AND be READ
BACK OUT — the sharpening-combiner the project lacks.

Banned list (the project's four rules): single online pass, NO gradients / batch-opt / SVD / eigen / word2vec.
A VSA respects all of it — atoms are fixed random ±1 hashes (random projection, allowed); BIND/BUNDLE/PERMUTE
are elementwise multiply/add/shift; cleanup is nearest-neighbor against a codebook (the leader-cluster step
the cortex already runs). Nothing is trained.

(1) COMPOSITIONAL DECODE (resonator). Mine (subject, verb, object)-ish triples from text8 by left/right
    co-occurrence counts (the most-associated left neighbor = subject role, right neighbor = object role —
    a dependency-ish proxy, all from counts). Encode each triple as T = role_s⊛S + role_v⊛V + role_o⊛O.
      (a) slot decode: unbind a known role, clean up — recover S/V/O. accuracy vs #roles, vs dimension D.
      (b) resonator factoring: encode a record as a single PRODUCT bind(perm0 S, perm1 V, perm2 O) and let a
          resonator factor it with NO known roles. accuracy vs #factors F, vs D — the capacity ceiling.

(2) ANALOGY VIA MAPPING VECTOR. T = a⊛b; is c⊛T ≈ d for a:b::c:d? Compare to AD's raw-count 3CosAdd on the
    IDENTICAL four families (capital-country, currency, plural, gender), restricted + open vocab.

KEY AXIS: does VSA give compositional READING (structure recovery) and/or SHARPER analogy than raw counts?
Honest about capacity: a sum/product of too many noisy atoms drowns the signal — we report where it breaks.
Corpus text8, fixed seed, single pass.
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from vsa import (Codebook, bind, unbind, bundle, permute, encode_triple, decode_slot,
                 Resonator, mapping_vector, apply_mapping)
from reasoning import CountProfiles, AnalogyCounts

print = functools.partial(print, flush=True)

TRAIN_BYTES = 16_000_000
N           = 30000        # top-N words (analogy target vocab + triple atom pool)
M           = 6000         # context vocab for the AD count-analogy comparison
WINDOW      = 4
SEED        = 0
N_TRIPLES   = 4000         # held-out triples mined for the decode test
DIMS        = [512, 1024, 2048, 4096, 8192]   # hypervector dimension sweep
TRIPLE_POOL = 4000         # restrict triple atoms to the top-4000 words (keeps cleanup honest + fast)

# same four families as Exp AD — for the apples-to-apples analogy comparison
CAPITAL = [("france","paris"),("germany","berlin"),("italy","rome"),("japan","tokyo"),
           ("china","beijing"),("russia","moscow"),("spain","madrid"),("england","london"),
           ("greece","athens"),("egypt","cairo"),("portugal","lisbon"),("austria","vienna"),
           ("poland","warsaw"),("norway","oslo"),("sweden","stockholm"),("ireland","dublin"),
           ("turkey","ankara"),("iran","tehran"),("cuba","havana"),("canada","ottawa")]
CURRENCY = [("usa","dollar"),("japan","yen"),("russia","ruble"),("india","rupee"),
            ("mexico","peso"),("italy","lira"),("germany","mark"),("britain","pound"),
            ("denmark","krone"),("israel","shekel")]
PLURAL   = [("year","years"),("car","cars"),("dog","dogs"),("city","cities"),("hand","hands"),
            ("book","books"),("road","roads"),("color","colors"),("animal","animals"),
            ("computer","computers"),("river","rivers"),("game","games"),("king","kings"),
            ("bird","birds"),("eye","eyes"),("word","words")]
GENDER   = [("man","woman"),("king","queen"),("boy","girl"),("father","mother"),("son","daughter"),
            ("brother","sister"),("husband","wife"),("uncle","aunt"),("nephew","niece"),
            ("sir","madam"),("prince","princess"),("actor","actress"),("god","goddess")]
FAMILIES = {"capital-country": CAPITAL, "currency": CURRENCY, "plural": PLURAL, "gender": GENDER}


def main():
    t0 = time.time()
    ids = load_ids("text8", nbytes=TRAIN_BYTES)
    spans = split_words(ids)
    words = [ids_to_str(ids[s:e]) for s, e in spans]
    w2id, wids = {}, np.empty(len(words), np.int64)
    for i, w in enumerate(words):
        wids[i] = w2id.setdefault(w, len(w2id))
    id2word = {v: k for k, v in w2id.items()}
    counts = np.bincount(wids, minlength=len(w2id))
    top = np.argsort(counts)[::-1][:N]
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    word2dense = {w: i for i, w in enumerate(topword)}
    seq = remap[wids]                                       # dense top-word stream, -1 = OOV
    print(f"text8 {TRAIN_BYTES//1_000_000}MB | {len(words):,} words, {len(w2id):,} types "
          f"| top-N {N} | load {time.time()-t0:.1f}s")

    rng = np.random.default_rng(SEED)

    # ════════════════════════════════════════════════════════════════════════════════════════════════
    # (1) COMPOSITIONAL DECODE — mine triples, encode as role⊛filler bundles, read structure back out
    # ════════════════════════════════════════════════════════════════════════════════════════════════
    # Triple mining (all from counts): for a high-frequency VERB-ish center word v, its most-associated
    # LEFT neighbor = subject role, most-associated RIGHT neighbor = object role. We just take adjacent
    # trigrams (w[t-1], w[t], w[t+1]) where all three are in the top-TRIPLE_POOL — a dependency-ish
    # (subject, verb, object) proxy straight from the stream, no parser. These are the structured records.
    t1 = time.time()
    s_all, v_all, o_all = seq[:-2], seq[1:-1], seq[2:]
    valid = (s_all >= 0) & (s_all < TRIPLE_POOL) & (v_all >= 0) & (v_all < TRIPLE_POOL) \
            & (o_all >= 0) & (o_all < TRIPLE_POOL) & (s_all != v_all) & (v_all != o_all) & (s_all != o_all)
    S, V, O = s_all[valid], v_all[valid], o_all[valid]
    pick = rng.choice(len(S), size=min(N_TRIPLES, len(S)), replace=False)
    triples = np.stack([S[pick], V[pick], O[pick]], 1)      # (n,3) dense ids, all in top-TRIPLE_POOL
    print(f"mined {len(triples):,} (subj,verb,obj) trigram records from top-{TRIPLE_POOL} | {time.time()-t1:.1f}s")

    # three fixed role hypervectors (shared across all triples) + a codebook of TRIPLE_POOL atom fillers
    print("\n" + "=" * 80)
    print("(1a) SLOT DECODE — T = role_s⊛S + role_v⊛V + role_o⊛O; unbind a role, clean up, recover filler")
    print("=" * 80)
    print("  accuracy = fraction of the 4000 records whose S/V/O is the TOP cleanup hit. cleanup over the")
    print("  full top-4000 atom codebook (open). bundle = sign(sum), the HDC default.\n")
    print(f"  {'dim D':>7} | {'subj':>7} {'verb':>7} {'obj':>7} | {'all-3':>7}")
    decode_by_dim = {}
    for D in DIMS:
        atoms = Codebook(TRIPLE_POOL, D, seed=SEED)
        roles = Codebook(3, D, seed=SEED + 1).V                # role_s, role_v, role_o
        ok = np.zeros(3, np.int64); all3 = 0
        for (s, v, o) in triples:
            T = encode_triple(roles, [atoms.V[s], atoms.V[v], atoms.V[o]], binarize=True)
            got = [atoms.cleanup(decode_slot(T, roles[r]), topk=1)[0] for r in range(3)]
            hit = [got[0] == s, got[1] == v, got[2] == o]
            ok += np.array(hit, np.int64)
            all3 += int(all(hit))
        n = len(triples)
        decode_by_dim[D] = (ok / n, all3 / n)
        print(f"  {D:>7} | {ok[0]/n*100:6.1f}% {ok[1]/n*100:6.1f}% {ok[2]/n*100:6.1f}% | {all3/n*100:6.1f}%")

    # how does it scale with the NUMBER of role-filler pairs bundled? (capacity of a sum)
    print("\n  capacity vs #role-filler pairs bundled (D=4096, recover slot 0; extra fillers = random atoms):")
    print(f"  {'#pairs':>7} | {'slot-0 acc':>11}")
    D = 4096
    atoms = Codebook(TRIPLE_POOL, D, seed=SEED)
    maxk = 8
    bigroles = Codebook(maxk, D, seed=SEED + 1).V
    nfac_curve = {}
    for k in range(1, maxk + 1):
        ok = 0
        for (s, v, o) in triples[:1500]:
            fillers = [s, v, o] + list(rng.integers(0, TRIPLE_POOL, size=max(0, k - 3)))
            fillers = fillers[:k]
            if len(fillers) < k:    # k<3 case: truncate
                fillers = ([s, v, o])[:k]
            vecs = [atoms.V[fi] for fi in fillers]
            T = encode_triple(bigroles[:k], vecs, binarize=True)
            got = atoms.cleanup(decode_slot(T, bigroles[0]), topk=1)[0]
            ok += int(got == fillers[0])
        nfac_curve[k] = ok / 1500
        print(f"  {k:>7} | {ok/1500*100:10.1f}%")

    # ── (1b) RESONATOR: factor a single PRODUCT bind with NO known roles ──
    print("\n" + "=" * 80)
    print("(1b) RESONATOR — factor s = perm0(S)⊛perm1(V)⊛perm2(O) into its 3 atoms, no roles known")
    print("=" * 80)
    print("  the hard one: a single product of F atoms, factored by iterative unbind+cleanup over a product")
    print(f"  space of {TRIPLE_POOL}^F. accuracy = all-F atoms recovered. reports lock-rate (converged) too.\n")
    print(f"  {'dim D':>7} {'F':>3} | {'all-F acc':>10} {'per-slot':>9} {'lock%':>7} {'avg-iters':>10}")
    reson_rows = []
    for D in [1024, 2048, 4096]:
        for F in (2, 3, 4):
            # one codebook per factor. Each factor occupies a distinct slot via PERMUTE-by-index, so the
            # resonator's codebook for factor g is the SET OF PERMUTED atoms rho^g(atom) — order is baked
            # into the code, and the cleanup space matches what was bound (a fair resonator capacity test).
            base = [Codebook(TRIPLE_POOL, D, seed=SEED + 10 + g) for g in range(F)]
            cbs = []
            for g in range(F):
                cb = Codebook.__new__(Codebook)
                cb.D = D; cb._unit = None
                cb.V = np.roll(base[g].V, g, axis=1)            # rho^g applied to every atom of factor g
                cbs.append(cb)
            res = Resonator(cbs, max_iter=40, seed=SEED)
            ok = 0; slot_ok = 0; locked = 0; iters = 0
            sub = triples[:300]
            for tr in sub:
                fac = list(tr[:F]) + list(rng.integers(0, TRIPLE_POOL, size=max(0, F - 3)))
                fac = fac[:F]
                if len(fac) < F:
                    continue
                s = cbs[0].V[fac[0]].copy()                     # already-permuted atoms (order in the code)
                for g in range(1, F):
                    s = bind(s, cbs[g].V[fac[g]])
                got, nit, lock = res.factor(s)
                hits = [got[g] == fac[g] for g in range(F)]
                ok += int(all(hits)); slot_ok += sum(hits); locked += int(lock); iters += nit
            ns = len(sub)
            reson_rows.append((D, F, ok / ns, slot_ok / (ns * F), locked / ns, iters / ns))
            print(f"  {D:>7} {F:>3} | {ok/ns*100:9.1f}% {slot_ok/(ns*F)*100:8.1f}% "
                  f"{locked/ns*100:6.1f}% {iters/ns:10.1f}")

    # WHERE does the resonator break? sweep codebook SIZE per factor at F=3, D=4096 — the operational
    # capacity curve. (Frady/Kent: the product space the resonator can factor scales ~linearly with D, so a
    # 4000^3 space at D=4096 is hopeless, but a small space is reliable. This locates the cliff.)
    print("\n  resonator operational capacity: codebook size per factor (F=3, D=4096) — where it cliffs:")
    print(f"  {'atoms/factor':>13} | {'all-3 acc':>10} {'lock%':>7}")
    D = 4096
    for npa in (20, 50, 100, 200, 400, 800, 1600):
        cbs = [Codebook(npa, D, seed=SEED + 30 + g) for g in range(3)]
        res = Resonator(cbs, max_iter=80, seed=SEED)
        ok = 0; lk = 0
        for _ in range(200):
            fac = [int(rng.integers(0, npa)) for _ in range(3)]
            s = cbs[0].V[fac[0]].copy()
            for g in (1, 2):
                s = bind(s, cbs[g].V[fac[g]])
            got, _, lock = res.factor(s)
            ok += int(tuple(got) == tuple(fac)); lk += int(lock)
        print(f"  {npa:>13} | {ok/200*100:9.1f}% {lk/200*100:6.1f}%")

    # ════════════════════════════════════════════════════════════════════════════════════════════════
    # (2) ANALOGY VIA MAPPING VECTOR — T=a⊛b; c⊛T≈d? vs AD's raw-count 3CosAdd, same families
    # ════════════════════════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("(2) ANALOGY — VSA mapping vector  T=a⊛b, c⊛T≈d   vs   AD raw-count 3CosAdd (PPMI)")
    print("=" * 80)

    # assemble the same analogy items as AD (all 4-distinct ordered pairs within a family)
    def to_dense(p):
        return word2dense.get(p[0]), word2dense.get(p[1])
    items = {}
    for fam, pairs in FAMILIES.items():
        dp = [to_dense(p) for p in pairs]
        dp = [(a, b) for (a, b) in dp if a is not None and b is not None]
        fam_items = []
        for i in range(len(dp)):
            for j in range(len(dp)):
                if i != j:
                    (a, b), (c, d) = dp[i], dp[j]
                    if len({a, b, c, d}) == 4:
                        fam_items.append((a, b, c, d))
        items[fam] = fam_items
    fam_targets = {fam: np.array(sorted({d for (_, _, _, d) in items[fam]}), np.int64)
                   for fam in items if items[fam]}
    total_items = sum(len(v) for v in items.values())
    print(f"  analogy items (same as AD): " + ", ".join(f"{f} {len(v)}" for f, v in items.items())
          + f" | total {total_items}\n")

    # --- AD baseline: raw-count PPMI 3CosAdd (the number to beat: ~56% restricted top-1) ---
    t2 = time.time()
    prof = CountProfiles(N=N, M=M, window=WINDOW, mode="ppmi").fit(seq)
    ad_solver = AnalogyCounts(prof.P)
    print(f"  AD raw-count PPMI profiles ({N}x{M}) built | {time.time()-t2:.1f}s")

    def eval_ad(restrict):
        res = {}
        for fam, fi in items.items():
            if not fi:
                continue
            t1c = t5c = 0
            r = fam_targets[fam] if restrict else None
            for (a, b, c, d) in fi:
                cand = ad_solver.solve(a, b, c, topk=5, restrict=r)
                t1c += int(bool(cand) and cand[0] == d); t5c += int(d in cand)
            res[fam] = (t1c / len(fi), t5c / len(fi), len(fi))
        return res

    # --- VSA mapping vector: each word = its fixed ±1 hypervector; T=a⊛b; cleanup(c⊛T) ---
    # VSA atoms are RANDOM and INDEPENDENT — they carry NO co-occurrence structure, so binding two random
    # atoms cannot encode a semantic relation. We test it anyway (the honest null), AND a count-grounded
    # variant where the hypervector is the SIGN of a random projection of the word's PPMI profile (a
    # random-projection sketch = allowed) so the relation actually lives in the geometry.
    def make_vsa_atoms(kind, D):
        if kind == "random":
            return Codebook(N, D, seed=SEED + 5).V
        # count-grounded: sign(PPMI_profile @ random_gaussian) — a ±1 sketch of the co-occurrence row
        rp = np.random.default_rng(SEED + 7).standard_normal((M, D)).astype(np.float32)
        proj = prof.P.astype(np.float32) @ rp                  # (N,D)
        v = np.sign(proj); v[v == 0] = 1.0
        return v.astype(np.float32)

    def eval_vsa(atoms, restrict):
        unit = atoms / np.maximum(np.linalg.norm(atoms, axis=1, keepdims=True), 1e-9)
        res = {}
        for fam, fi in items.items():
            if not fi:
                continue
            t1c = t5c = 0
            r = fam_targets[fam] if restrict else None
            U = unit if r is None else unit[r]
            for (a, b, c, d) in fi:
                T = mapping_vector(atoms[a], atoms[b])
                q = apply_mapping(atoms[c], T)
                qn = q / max(np.linalg.norm(q), 1e-9)
                sims = U @ qn
                order = np.argsort(sims)[::-1]
                cand = (order if r is None else r[order])[:5]
                cand = [int(x) for x in cand if x not in (a, b, c)][:5]
                t1c += int(bool(cand) and cand[0] == d); t5c += int(d in cand)
            res[fam] = (t1c / len(fi), t5c / len(fi), len(fi))
        return res

    def macro(d, idx):
        vals = [d[f][idx] for f in d]
        return float(np.mean(vals)) if vals else 0.0

    for restrict, label in ((True, "RESTRICTED candidate set (d among family's own b-words)"),
                            (False, "OPEN vocabulary")):
        print(f"  --- {label} ---")
        ad = eval_ad(restrict)
        vr = eval_vsa(make_vsa_atoms("random", 4096), restrict)
        vg = eval_vsa(make_vsa_atoms("grounded", 4096), restrict)
        print(f"  {'family':<16} {'n':>4} | {'AD ppmi t1/t5':>14} | {'VSA-rand t1/t5':>15} | {'VSA-ground t1/t5':>17}")
        for fam in items:
            if not items[fam]:
                continue
            n = ad[fam][2]
            print(f"  {fam:<16} {n:>4} | {ad[fam][0]*100:5.0f}/{ad[fam][1]*100:3.0f}      "
                  f"| {vr[fam][0]*100:5.0f}/{vr[fam][1]*100:3.0f}       "
                  f"| {vg[fam][0]*100:5.0f}/{vg[fam][1]*100:3.0f}")
        print(f"  {'MACRO-AVG':<16} {'':>4} | {macro(ad,0)*100:5.0f}/{macro(ad,1)*100:3.0f}      "
              f"| {macro(vr,0)*100:5.0f}/{macro(vr,1)*100:3.0f}       "
              f"| {macro(vg,0)*100:5.0f}/{macro(vg,1)*100:3.0f}\n")

    # dimension sweep for the count-grounded VSA (does more dimension sharpen the analogy?)
    print("  count-grounded VSA analogy vs dimension (restricted macro top-1 / top-5):")
    print(f"  {'dim D':>7} | {'macro t1/t5':>12}")
    for D in DIMS:
        vg = eval_vsa(make_vsa_atoms("grounded", D), True)
        print(f"  {D:>7} | {macro(vg,0)*100:5.0f}/{macro(vg,1)*100:3.0f}")

    print(f"\n  online-compliance: atoms = fixed random ±1 hashes (or a sign-of-random-projection sketch of the "
          f"count profile — a random projection, allowed); BIND/BUNDLE/PERMUTE = elementwise multiply/add/shift; "
          f"cleanup = nearest-neighbor over a codebook (the leader-cluster step). No gradient / batch-opt / "
          f"SVD / eigen / word2vec. Single pass, fixed seed.")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

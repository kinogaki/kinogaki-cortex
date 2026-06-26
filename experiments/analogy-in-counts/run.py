#!/usr/bin/env python3
"""Exp AD — count-only compositional reasoning + analogy. ONLINE, single pass, NO backprop.

Two ideas converge here (research/IDEAS, PROVENANCE):
  - The PARALLELOGRAM is already in raw co-occurrence counts (PMC11493305): word-analogy structure is
    recoverable from the unfactorized count matrix; SVD/word2vec only SMOOTH it to human parity.
  - NARS (Wang): derive new links by syllogism with a truth value (f,c) — induction/abduction from
    observed counts, no training. Turney LRA: relations live in pair co-occurrence patterns.

Hard online rule: a single streaming pass of order-independent counting + leaky accumulators + online
leader clustering. NO gradient descent, NO batch optimization, and CRITICALLY no SVD / eigendecomposition
/ PMI-matrix factorization and no word2vec (the banned smoothing steps).

(A) ANALOGY FROM RAW COUNTS. Per-word co-occurrence profile; relation r(a->b)=logcount(.|b)-logcount(.|a);
    solve a:b::c:? by 3CosAdd in log-count space. Headline RISK (the research flagged it): word2vec/LRA get
    parity via SVD smoothing we forbid — so test whether ONLINE LEADER CLUSTERING over context profiles can
    substitute. Report accuracy WITH vs WITHOUT the leader-cluster smoothing, vs random/frequency baselines.

(B) INDUCED LINKS (NARS). From L->A and A->B counts, INDUCE L->B through bridges A with a derived (f,c).
    Measure held-out prediction on probes that NEVER co-occurred directly (compositional, transitive).

Corpus: text8, ~16 MB. Fixed seed, single pass.
"""
import os, sys, time, functools
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lib"))
from corpus import load_ids, split_words, ids_to_str
from reasoning import CountProfiles, LeaderSmoothing, AnalogyCounts, InducedLinks

print = functools.partial(print, flush=True)

TRAIN_BYTES = 16_000_000
N           = 30000        # top-N words get a target profile
M           = 6000         # context vocab = top-M words (the profile's feature space)
WINDOW      = 4
SEED        = 0

# Leader-smoothing knobs (the SVD substitute)
SIG_D       = 96
COS_THRESH  = 0.8
CMAX        = 800
MIN_EV      = 40
BETAS       = [0.0, 0.3, 0.6, 0.9]   # 0.0 = raw counts (no smoothing); sweep the smoothing strength

# ── Standard analogy item families, built from text8 vocab (capital-country, currency, plural, gender) ──
# Each family is a list of (term_a, term_b) pairs sharing ONE relation; analogy items = all ordered pairs
# of pairs within a family: (a1,b1):(a2,b2) means a1:b1::a2:?->b2.  Drawn from the classic Mikolov/Google set,
# trimmed to entries present in text8.
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
            ("road","roads"),("bird","birds"),("eye","eyes"),("road","roads"),("word","words")]
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
    top = np.argsort(counts)[::-1][:N]                  # frequency-ranked: id 0..N-1 = most..least frequent
    remap = -np.ones(len(w2id), np.int64); remap[top] = np.arange(len(top))
    topword = [id2word[t] for t in top]
    word2dense = {w: i for i, w in enumerate(topword)}  # surface word -> dense id (0..N-1)
    seq = remap[wids]
    print(f"text8 {TRAIN_BYTES//1_000_000}MB | {len(words):,} words, {len(w2id):,} types "
          f"| top-N {N} ctx-M {M} | load {time.time()-t0:.1f}s")

    # ── build co-occurrence count profiles in BOTH count-native modes (log-count and positive-PMI) ──
    # PPMI is the representation the analogy literature says the parallelogram lives in (PMC11493305); it is
    # pure per-cell counting (a ratio of counts), NOT a factorization — so it is on the allowed list.
    t1 = time.time()
    profs = {m: CountProfiles(N=N, M=M, window=WINDOW, mode=m).fit(seq) for m in ("log", "ppmi")}
    print(f"count profiles ({N}x{M}, window +/-{WINDOW}, modes log+ppmi) built | {time.time()-t1:.1f}s")

    # ── online leader-cluster smoothing (the SVD substitute), per profile mode ──
    t2 = time.time()
    smooths = {m: LeaderSmoothing(N=N, sig_D=SIG_D, sig_window=WINDOW, min_evidence=MIN_EV,
                                  cos_thresh=COS_THRESH, cmax=CMAX, seed=SEED).fit(seq, profs[m])
               for m in profs}
    print(f"online leader smoothing: C={smooths['ppmi'].C} clusters, "
          f"{(smooths['ppmi'].clu>=0).sum():,}/{N} clustered | {time.time()-t2:.1f}s")

    # ── assemble analogy items, keeping only those whose 4 words are all in the top-N profile vocab ──
    def to_dense(p):
        return word2dense.get(p[0]), word2dense.get(p[1])
    items = {}      # family -> list of (a,b,c,d) dense ids
    for fam, pairs in FAMILIES.items():
        dp = [to_dense(p) for p in pairs]
        dp = [(a, b) for (a, b) in dp if a is not None and b is not None]
        fam_items = []
        for i in range(len(dp)):
            for j in range(len(dp)):
                if i == j:
                    continue
                (a, b), (c, d) = dp[i], dp[j]
                if len({a, b, c, d}) == 4:
                    fam_items.append((a, b, c, d))
        items[fam] = fam_items
    total_items = sum(len(v) for v in items.values())
    print(f"analogy items: " + ", ".join(f"{f} {len(v)}" for f, v in items.items()) + f" | total {total_items}")

    # restrict candidate sets to the family's own b-words (the standard category-restricted analogy eval) and
    # ALSO report open-vocab (top-N) accuracy — the honest, harder number.
    fam_targets = {fam: np.array(sorted({d for (_, _, _, d) in items[fam]}), np.int64)
                   for fam in items if items[fam]}

    rng = np.random.default_rng(SEED)

    # ── (A) ANALOGY: 3CosAdd raw vs leader-smoothed vs random/frequency baselines ──
    print("\n" + "=" * 78)
    print("(A) ANALOGY FROM RAW COUNTS — 3CosAdd in log-count space  (top-1 / top-5 acc)")
    print("=" * 78)
    print("baselines: RANDOM = pick uniformly from the restricted candidate set; "
          "FREQ = pick the most frequent restricted candidate.\n")

    # precompute analogy solvers per (profile mode, smoothing beta). beta=0 = raw counts (no smoothing).
    solvers = {}
    for mode in profs:
        for beta in BETAS:
            P = smooths[mode].smooth_matrix(profs[mode], beta)
            solvers[(mode, beta)] = AnalogyCounts(P)

    # evaluation helpers
    def eval_solver(solver, restrict_by_family):
        """Return dict family -> (top1, top5, n) over that family's items. restrict_by_family: use the
        family's own b-words as the candidate set (True) or open top-N vocab (False)."""
        res = {}
        for fam, fam_items in items.items():
            if not fam_items:
                continue
            t1c = t5c = 0
            restrict = fam_targets[fam] if restrict_by_family else None
            for (a, b, c, d) in fam_items:
                cand = solver.solve(a, b, c, topk=5, restrict=restrict)
                if cand and cand[0] == d:
                    t1c += 1
                if d in cand:
                    t5c += 1
            n = len(fam_items)
            res[fam] = (t1c / n, t5c / n, n)
        return res

    # baselines (restricted candidate set). dense id IS the frequency rank (0 = most frequent), so the
    # FREQ baseline = the lowest-id candidate that isn't one of a,b,c; RANDOM = a uniform pick from the pool.
    def baseline(kind):
        res = {}
        for fam, fam_items in items.items():
            if not fam_items:
                continue
            t1c = 0
            cands = fam_targets[fam]
            for (a, b, c, d) in fam_items:
                pool = sorted(x for x in cands if x not in (a, b, c))
                if not pool:
                    continue
                pick = int(rng.choice(pool)) if kind == "random" else pool[0]   # pool[0] = most frequent
                if pick == d:
                    t1c += 1
            res[fam] = (t1c / len(fam_items), None, len(fam_items))
        return res

    base_rand = baseline("random")
    base_freq = baseline("frequency")

    def macro(d, idx):
        vals = [d[f][idx] for f in d if d[f][idx] is not None]
        return float(np.mean(vals)) if vals else 0.0

    # cache every (mode,beta,restrict) evaluation once
    ev_cache = {}
    def ev(mode, beta, restrict):
        key = (mode, beta, restrict)
        if key not in ev_cache:
            ev_cache[key] = eval_solver(solvers[(mode, beta)], restrict)
        return ev_cache[key]

    # one table per profile mode (log-count vs positive-PMI), columns = beta sweep, rows = families.
    # The SVD-substitution question is read off the beta columns: does beta>0 (leader smoothing) BEAT beta=0?
    for restrict, label in ((True, "RESTRICTED candidate set (pick d among the family's own b-words)"),
                            (False, "OPEN vocabulary (pick d among ALL top-N words)")):
        print(f"--- {label} ---")
        for mode in ("log", "ppmi"):
            print(f"  [{mode}-count profile]  baselines: random {macro(base_rand,0)*100:.0f}% "
                  f"freq {macro(base_freq,0)*100:.0f}%  (restricted-set top-1)" if restrict
                  else f"  [{mode}-count profile]")
            print(f"  {'family':<16} {'n':>4} | " + "  ".join(f"b={b:<3}t1/t5" for b in BETAS))
            for fam in items:
                if not items[fam]:
                    continue
                n = ev(mode, BETAS[0], restrict)[fam][2]
                row = f"  {fam:<16} {n:>4} |"
                for beta in BETAS:
                    t1, t5, _ = ev(mode, beta, restrict)[fam]
                    row += f"  {t1*100:4.0f}/{t5*100:3.0f}"
                print(row)
            row = f"  {'MACRO-AVG':<16} {'':>4} |"
            for beta in BETAS:
                e = ev(mode, beta, restrict)
                row += f"  {macro(e,0)*100:4.0f}/{macro(e,1)*100:3.0f}"
            print(row)
        print()

    # a few worked examples (open vocab, best mode) for the eyeball check
    print("  worked examples (open vocab, ppmi):  a:b :: c:?   ->  raw(b=0) top-3  |  smoothed(b=0.6) top-3")
    sraw, ssm = solvers[("ppmi", 0.0)], solvers[("ppmi", 0.6)]
    examples = [("france","paris","japan","tokyo"), ("man","woman","king","queen"),
                ("car","cars","dog","dogs"), ("japan","tokyo","italy","rome")]
    for (wa, wb, wc, wd) in examples:
        a, b, c = word2dense.get(wa), word2dense.get(wb), word2dense.get(wc)
        if None in (a, b, c):
            continue
        r = [topword[i] for i in sraw.solve(a, b, c, topk=3)]
        s = [topword[i] for i in ssm.solve(a, b, c, topk=3)]
        gold = "*" if wd in r or wd in s else " "
        print(f"   {wa}:{wb} :: {wc}:?  (gold {wd}{gold})  raw {r}  | smooth {s}")

    # ── (B) INDUCED LINKS (NARS): held-out compositional prediction ──
    print("\n" + "=" * 78)
    print("(B) INDUCED LINKS (NARS) — transitive count composition on NEVER-CO-OCCURRED probes")
    print("=" * 78)
    t3 = time.time()
    il = InducedLinks(N=M, M=M, near=1, far=2, k=1.0).fit(np.where(seq < M, seq, -1))
    print(f"induced-link tables (L->A gap1, A->B gap2, L->B gap3) built over top-{M} | {time.time()-t3:.1f}s")

    # probe set: positions t in the stream where L = word at t-3, B = word at t (the true gap-3 next word),
    # all three of L, (bridge region), B in top-M. The HELD-OUT slice = (L,B) pairs whose DIRECT gap-3 count
    # is ZERO on the rest of the corpus EXCEPT this occurrence — i.e. L and B essentially never co-occur at
    # this offset, so a direct counter is blind. Induction must answer through bridges A.
    gap = il.near + il.far
    L_all = seq[:-gap]; B_all = seq[gap:]
    valid = (L_all >= 0) & (L_all < M) & (B_all >= 0) & (B_all < M)
    Lv, Bv = L_all[valid], B_all[valid]
    # direct co-occurrence count for each (L,B) pair across the corpus
    pair_key = Lv * M + Bv
    uk, inv, uc = np.unique(pair_key, return_inverse=True, return_counts=True)
    pair_count = uc[inv]                                  # how many times THIS (L,B) co-occurs at gap-3
    held_out = pair_count <= 1                            # never (or barely) co-occurred directly
    print(f"probes {valid.sum():,} (L,B) at gap {gap}; held-out (direct co-occ <=1): {held_out.mean()*100:.1f}%")

    # Three predictors, all count-only: DIRECT gap-3 counts (≈blind on held-out by construction), a UNIGRAM
    # baseline (always rank B-words by global frequency — the order-blind floor), and NARS INDUCED (transitive
    # bridge composition). Right axis = top-k recovery + mean-rank on the held-out (never-co-occurred) slice.
    uni = np.bincount(Bv, minlength=M).astype(np.float64)            # global target frequency (count baseline)
    uni_order = np.argsort(uni)[::-1]
    uni_rank = np.empty(M, np.int64); uni_rank[uni_order] = np.arange(M)   # rank of each B under unigram

    held_idx = np.nonzero(held_out)[0]
    sample = rng.choice(held_idx, size=min(4000, len(held_idx)), replace=False)
    d_t1 = d_t5 = u_t1 = u_t5 = i_t1 = i_t5 = 0
    d_rank = u_rank = i_rank = 0.0
    n = 0
    for s in sample:
        L = int(Lv[s]); B = int(Bv[s])
        pi = il.induced(L)
        if pi.sum() == 0:
            continue
        n += 1
        pd = il.direct(L)
        d_order = np.argsort(pd)[::-1]; i_order = np.argsort(pi)[::-1]
        d_t1 += int(d_order[0] == B); d_t5 += int(B in d_order[:5])
        u_t1 += int(uni_order[0] == B); u_t5 += int(B in uni_order[:5])
        i_t1 += int(i_order[0] == B);  i_t5 += int(B in i_order[:5])
        d_rank += int(np.nonzero(d_order == B)[0][0]) + 1
        u_rank += int(uni_rank[B]) + 1
        i_rank += int(np.nonzero(i_order == B)[0][0]) + 1
    print(f"\n  held-out compositional prediction ({n:,} scored probes; true target B never co-occurs at gap {gap}).")
    print(f"  mean-rank: LOWER is better (1 = perfect); chance = {M/2:.0f}.")
    print(f"  {'model':<26} {'top-1':>8} {'top-5':>8} {'mean-rank':>10}")
    print(f"  {'unigram frequency':<26} {u_t1/n*100:7.2f}% {u_t5/n*100:7.2f}% {u_rank/n:10.1f}")
    print(f"  {'direct gap-3 counts':<26} {d_t1/n*100:7.2f}% {d_t5/n*100:7.2f}% {d_rank/n:10.1f}")
    print(f"  {'NARS induced (bridges)':<26} {i_t1/n*100:7.2f}% {i_t5/n*100:7.2f}% {i_rank/n:10.1f}")
    print(f"  LIFT induced - direct:       {(i_t1-d_t1)/n*100:+7.2f}% {(i_t5-d_t5)/n*100:+7.2f}% "
          f"{(d_rank-i_rank)/n:+9.1f}")
    print(f"  LIFT induced - unigram:      {(i_t1-u_t1)/n*100:+7.2f}% {(i_t5-u_t5)/n*100:+7.2f}% "
          f"{(u_rank-i_rank)/n:+9.1f}")

    print(f"\n  online-compliance: single streaming pass; profiles + offset tables = counts; smoothing = online "
          f"leader clustering (running-mean prototypes); NARS (f,c) = count ratios. "
          f"No backprop / batch-opt / SVD / eigen / PMI-factorization / word2vec.")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

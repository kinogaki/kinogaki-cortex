#!/usr/bin/env python3
"""Experiment AR — power-law (ACT-R) memory at the WORD level, under a budget, on a NON-STATIONARY stream.
The parked resurrection from Exp AI.

Exp AI showed ACT-R power-law eviction (evict lowest B = ln Σ age_k^(−d)) LOSES to raw-count LFU for DENSE
char-grams, because a char-gram's value ≈ its total count (no useful-then-stale structure → recency is variance,
not signal; LFU = the power law's d→0 limit). But AI PREDICTED the power law wins where frequency stops ranking
usefulness: SPARSE, NON-STATIONARY memory — the WORD/CONCEPT level. This experiment tests exactly there.

Stream three truly different REGISTERS in ONE pass — Darwin (Victorian science) → Shakespeare (Early-Modern verse)
→ KJV Bible (archaic scripture) — through a WORD-bigram count table capped at N contexts. A later register's flood
forces eviction; the four policies (powerlaw / lfu / lru / ema) differ ONLY in WHAT they keep. After the full
stream, eval held-out bits-per-word + next-word accuracy on EVERY register — backward retention is the whole game.

KEY TEST: at the word level, under non-stationarity + a budget, does power-law eviction now BEAT LFU — keeping the
rare-but-recently-relevant word over the high-frequency-but-stale one when the topic shifts? (The OPPOSITE of AI's
char-gram result.) Honest if it still loses — then LFU is universal and that's the finding.

HARD RULES: single streaming pass; no gradients; no batch optimization; bounded memory; reservoir eviction; seed 0.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from lib.powerlaw_word import WordCountModel, Vocab, load_words

DATA = os.path.join(HERE, "..", "data")
SEED = 0
np.random.seed(SEED)

ORDER = 2                       # word-trigram context (prev TWO words → next): genuinely sparse + register-specific.
D = 0.5                         # ACT-R decay
N_TRAIN = 120_000               # words per register (single streaming pass)
N_EVAL = 8_000                  # disjoint held-out tail per register
# order-2 has ~52k/71k/34k distinct contexts per register (~155k total) — a small cap forces HARD eviction, the
# bounded-memory + non-stationary regime Exp AI predicted is where the power law should finally pay.
CAPS = [10_000, 30_000, 80_000]
POLICIES = ["powerlaw", "lfu", "lru", "ema"]
REGISTERS = ["darwin", "shakespeare", "bible"]


def banner(s):
    print("\n" + "=" * 84 + f"\n{s}\n" + "=" * 84, flush=True)


def load_split(vocab):
    """Load train+held-out word-id slices for each register, GROWING one shared vocab as we go (online)."""
    train, ev = {}, {}
    for r in REGISTERS:
        ids = load_words(os.path.join(DATA, f"{r}.txt"), vocab, N_TRAIN + N_EVAL, grow=True)
        train[r] = ids[:N_TRAIN]; ev[r] = ids[N_TRAIN:N_TRAIN + N_EVAL]
    return train, ev


def run_policy(policy, cap, train, ev):
    """Stream darwin → shakespeare → bible through ONE capped model, snapshotting the full retention matrix
    (bpw + acc on every register) after EACH phase. Returns (matrix, kept_sizes)."""
    m = WordCountModel(order=ORDER, cap=cap, d=D, policy=policy)
    matrix = {}   # after_register -> {eval_register: (bpw, acc)}
    for after in REGISTERS:
        m.train_stream(train[after])
        matrix[after] = {r: m.eval_bpw(ev[r]) for r in REGISTERS}
    return matrix, m.size()


def main():
    t0 = time.time()
    banner("Exp AR — POWER-LAW WORD-LEVEL EVICTION under register shift (darwin → shakespeare → bible)")
    vocab = Vocab()
    train, ev = load_split(vocab)
    print(f"  registers: {', '.join(REGISTERS)}   order={ORDER} (word-bigram)   d={D}")
    print(f"  {N_TRAIN:,} train + {N_EVAL:,} held-out words / register   shared vocab |V|={len(vocab):,}")
    # how many distinct order-N contexts each register really has (why the cap bites):
    for r in REGISTERS:
        ctxs = set(tuple(int(x) for x in train[r][i - ORDER:i]) for i in range(ORDER, len(train[r])))
        print(f"    {r:11s}: {len(ctxs):,} distinct order-{ORDER} contexts in train")

    # full sweep: every (policy, cap) gives a retention matrix.
    results = {}     # (policy, cap) -> (matrix, size)
    for cap in CAPS:
        for pol in POLICIES:
            results[(pol, cap)] = run_policy(pol, cap, train, ev)
            print(f"  [{time.time()-t0:6.0f}s] done cap={cap:>6,} policy={pol}", flush=True)

    # ── REPORT 1: held-out bpw on EACH register AFTER THE FULL STREAM (the budget+shift headline) ───────────────
    banner("RESULT 1 — held-out bits-per-word after the FULL stream (darwin+shakespeare+bible), lower=better")
    for cap in CAPS:
        print(f"\n  cap = {cap:,} contexts")
        print(f"    {'policy':<10}" + "".join(f"{r:>13}" for r in REGISTERS) + f"{'mean':>11}")
        rows = {}
        for pol in POLICIES:
            mat = results[(pol, cap)][0]["bible"]   # "after bible" = after the whole stream
            bpws = [mat[r][0] for r in REGISTERS]
            rows[pol] = bpws
            print(f"    {pol:<10}" + "".join(f"{b:>13.4f}" for b in bpws) + f"{np.mean(bpws):>11.4f}")
        # who wins per column + mean
        best_mean = min(POLICIES, key=lambda p: np.mean(rows[p]))
        print(f"    → best mean bpw: {best_mean}")

    # ── REPORT 2: BACKWARD RETENTION of DARWIN — the first register, most flooded by later ones ─────────────────
    banner("RESULT 2 — backward retention of DARWIN (bpw on darwin held-out as later registers flood the cap)")
    print("  Δ = bpw(after bible) − bpw(after darwin); + = forgot darwin. The non-stationarity test.")
    for cap in CAPS:
        print(f"\n  cap = {cap:,}")
        print(f"    {'policy':<10}{'after darwin':>14}{'after shake':>14}{'after bible':>14}{'Δ forgot':>12}")
        for pol in POLICIES:
            mat = results[(pol, cap)][0]
            peak = mat["darwin"]["darwin"][0]
            mid = mat["shakespeare"]["darwin"][0]
            fin = mat["bible"]["darwin"][0]
            print(f"    {pol:<10}{peak:>14.4f}{mid:>14.4f}{fin:>14.4f}{fin-peak:>12.4f}")

    # ── REPORT 3: next-word ACCURACY after the full stream (a second, scale-free lens) ──────────────────────────
    banner("RESULT 3 — next-word accuracy after the full stream (argmax hit-rate, higher=better)")
    for cap in CAPS:
        print(f"\n  cap = {cap:,}")
        print(f"    {'policy':<10}" + "".join(f"{r:>13}" for r in REGISTERS) + f"{'mean':>11}")
        for pol in POLICIES:
            mat = results[(pol, cap)][0]["bible"]
            accs = [mat[r][1] for r in REGISTERS]
            print(f"    {pol:<10}" + "".join(f"{a:>12.2%} " for a in accs) + f"{np.mean(accs):>10.2%}")

    # ── REPORT 4: head-to-head — power-law minus LFU, the one number that decides the bet ───────────────────────
    banner("RESULT 4 — THE BET: power-law − LFU mean bpw (negative = power-law WINS, the opposite of Exp AI)")
    for cap in CAPS:
        pl = np.mean([results[("powerlaw", cap)][0]["bible"][r][0] for r in REGISTERS])
        lfu = np.mean([results[("lfu", cap)][0]["bible"][r][0] for r in REGISTERS])
        verdict = "POWER-LAW WINS" if pl < lfu else "LFU wins"
        print(f"    cap={cap:>6,}:  powerlaw={pl:.4f}  lfu={lfu:.4f}  Δ(pl−lfu)={pl-lfu:+.4f}   → {verdict}")

    print(f"\n  table sizes held at cap (sanity): "
          f"{ {f'{p}@{c}': results[(p,c)][1] for p in ['powerlaw','lfu'] for c in CAPS} }")
    print(f"\n  total runtime {time.time()-t0:.0f}s   seed={SEED}")


if __name__ == "__main__":
    main()

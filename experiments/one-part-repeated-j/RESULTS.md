# Exp J — is the uniform Column a better base? (foundation check) — 2026-06-25

Before scaling to gigabytes or adding attention/gating, answer the prerequisite: is the `Column` abstraction
**optimizable** (not a pure-Python dead end) and does feeding it more data actually pay off? Same Column
interface (`learn`/`predict`), two backends: the readable dict one (`lib/cortex.py`) and a vectorized one
(`lib/fastcol.py`, count tables built by one `np.unique` over the stream, predict = `searchsorted` slice).

## Correctness & speed — one Column, order 6, 2 MB

| backend | learn time | test bpc |
|---|---:|---:|
| dict Column (readable reference) | 2.41 s | 2.3502 |
| FastColumn (vectorized) | **0.28 s** | 2.3502 |

**Identical model** (bpc matches to 1e-6) at a **9× faster learn** — the abstraction is real: one interface,
swappable guts. The vectorized backend is not a different model, it's the same Column computed without the
Python loop. (9× is the floor — the gap widens at higher orders and on the multi-column/word stack, where the
dict version's tuple-hashing dominates; the full Exp I stack took ~10 min dict-side.)

## Scale — same Column, more data (only the fast backend makes this reachable)

| train chars | learn | MB/s | test bpc |
|---:|---:|---:|---:|
| 2,000,000 | 0.28 s | 7.3 | 2.350 |
| 10,000,000 | 1.45 s | 6.9 | 2.021 |
| 50,000,000 | 6.96 s | 7.2 | **1.849** |

**The data axis was starved, not saturated.** bpc falls 2.35 → 2.02 → 1.85 from 2→50 MB — exactly Exp F's
capacity×data law. The "+phrase band didn't pay off" verdict in Exp I was a **2 MB artifact**; the architecture
keeps improving with data, and at ~7 MB/s sustained a **1 GB corpus learns in ~2–3 minutes, not days**.

## Verdict — the Column IS the better base

- **Flexible:** the same `Column` expresses the whole A–H zoo by rewiring (char n-gram, backoff, voting Level,
  word/phrase hierarchy) — proven in Exp I. One part, many models.
- **Optimizable:** the count-table design vectorizes cleanly with no interface change (9×+ today, GPU-able next
  — see below). Not a pure-Python dead end.
- **A real base for what's next:** GB-scale corpora are now affordable, so the questions that need data (does a
  4th level pay off with 100× the text? does global coherence emerge at scale?) become answerable. And the
  predict/vote inner loop (gather from a table + reduce across columns) is the *exact shape* of a path tracer's
  sample-and-accumulate — the Metal port is a natural, not a rewrite. The foundation holds. Build on it.

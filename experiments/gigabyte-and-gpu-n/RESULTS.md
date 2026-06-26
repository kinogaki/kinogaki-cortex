# Exp N — gigabyte char scaling — 2026-06-25 (autonomous)

Push the char Column to a real gigabyte (enwik9, 1 GB Wikipedia → 827 M chars in id-space) and ask: does
prediction keep improving, and can we learn it fast? Order-5 char backoff, pure id-space (no Python strings),
**batch bpc** (vectorized — no per-position loop; that was Exp K's bottleneck). Fixed 1 M-char held-out.

**Correctness first:** batch eval == per-position eval (2.3889 == 2.3889) — the fast path is the same model.

## Scaling (order-5 char, fixed held-out)

| train chars | learn | MB/s | test bpc |
|---:|---:|---:|---:|
| 10,000,000 | 1.2 s | 8.6 | 1.9973 |
| 100,000,000 | 12.0 s | 8.3 | 1.8208 |
| 300,000,000 | 40.9 s | 7.3 | 1.7734 |
| **822,304,732** | 155.9 s | 5.3 | **1.7443** |

**More data still helps at a gigabyte** — bpc falls monotonically 1.997 → 1.744, ~85× the data buying −0.25 bpc.
Diminishing (the last 2.7× of data buys only −0.03) but not flat: at 1 GB the order-5 *char* cortex is near its
own capacity ceiling (Exp F said optimal order grows with data — a gigabyte wants order 6–8, which Exp O's dense/
GPU path can now afford). enwik9 loads to id-space in **3.6 s**; the whole gigabyte learns in **2.6 min** on CPU
(seconds on GPU — see Exp O).

## Takeaways

- The data axis was never the problem — feeding the same Column more text just works, all the way to 1 GB.
- The remaining char-bpc gap to SOTA (~1.0–1.4) is **capacity** (order, concept levels), not data — and capacity
  is exactly what the GPU now makes cheap. The interesting science (boundaries, attention, global coherence)
  rides on top of this fast, well-fed substrate.

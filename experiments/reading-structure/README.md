# Reading structure back out of a sum

**Experiment AN · 2026-06-26 · split**

A three-way split. HRR role-filler decode **with known roles** = **100%** (robust to 8
bound pairs, even at D=512) — compositional reading is solved when structure is supplied.
The blind **resonator** (factor a product with unknown roles) **fails** at affordable
dimension (cliff at 200–400 atoms/factor, ≈0% over a 4000-word codebook — the capacity
wall). And **analogy** via a bound mapping vector is a clean negative: **24 / 60** restricted
vs raw counts' **56 / 94** — binarized binding discards the graded magnitudes the
parallelogram needs. The payoff: a VSA is a *reader* of structure you supply (which
redescription mints and the System-2 workspace moves), not a discoverer of structure you
don't — don't factor blindly, and keep the count parallelogram as the analogy organ.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/reading-structure/run.py
```

**Blog post:** ["Reading structure back out of a sum"](https://cortex.kinogaki.com/blog/reading-structure/)

Credit: Plate (HRR); Kanerva (HDC); Frady/Kent/Olshausen/Sommer (resonator networks).

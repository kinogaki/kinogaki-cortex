# Reading structure back out of a sum

**Experiment AN · 2026-06-26 · split**

HRR role-filler decode **with known roles** = **100%** (robust to 8 bound pairs, even
at D=512) — compositional reading is solved when structure is supplied. But the blind
**resonator** (factor a product with unknown roles) **fails** at affordable dimension
(≈0% all-factor recovery over a 4000-word codebook at D ≤ 2048 — the capacity wall).
The payoff: VSA-decode works *given* structure (which redescription supplies and the
System-2 workspace manipulates) — don't factor blindly. (Analogy probe not recovered;
flagged as a follow-up.)

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/reading-structure/run.py
```

**Blog post:** ["Reading structure back out of a sum"](https://cortex.kinogaki.com/blog/reading-structure/)

Credit: Plate (HRR); Kanerva (HDC); Frady/Kent/Olshausen/Sommer (resonator networks).

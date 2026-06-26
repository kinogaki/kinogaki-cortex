# Exp AN — resonator / HRR compositional decode + VSA analogy — 2026-06-26

Vector Symbolic Architecture, gradient-free. text8 16 MB, 2.73 M words; 4,000 (subject, verb, object) trigram
records mined from the top-4,000 words. Atoms = fixed random ±1 hypervectors; bind = elementwise product;
bundle = sign(sum) (the HDC default); permute (cyclic shift) tags slot order. Online, single pass, no gradients,
no SVD. Three tests: decode *when the frame is known* (HRR), decode *blindly* (resonator), and *analogy* via a
bound mapping vector (head-to-head with Exp AD's raw-count parallelogram).

## (1a) Slot decode with KNOWN roles — `T = role_s⊛S + role_v⊛V + role_o⊛O`, unbind a role, clean up → PERFECT

| dim D | subj | verb | obj | all-3 |
|---:|---:|---:|---:|---:|
| 512 | 100% | 100% | 100% | 100% |
| 1024–8192 | 100% | 100% | 100% | 100% |

Capacity (D=4096, recover slot-0 from a bundle of *k* role-filler pairs, extra fillers random): **100% at every
k from 1 to 8.** Structure survives the sum and comes back out exactly — even at low dimension (512) and high
load (8 bound pairs). **When the roles/frame are supplied, HRR compositional reading is solved.**

## (1b) Resonator — factor `s = ρ⁰(S)⊛ρ¹(V)⊛ρ²(O)` with NO roles known → FAILS at affordable dimension

| dim D | F | all-F acc | per-slot | lock% |
|---:|---:|---:|---:|---:|
| 1024 | 2 | 0.0% | 0.3% | 0% |
| 2048 | 2 | 0.7% | 4.3% | 1% |
| 4096 | 2 | 13.0% | 25.2% | 13% |
| 4096 | 3 | 0.0% | 0.0% | 19% (spurious) |
| 4096 | 4 | 0.0% | 0.0% | 0% |

The resonator **converges** to the **wrong factors** over a 4,000-atom codebook. Sweeping atoms-per-factor at
F=3, D=4096 locates the capacity cliff exactly:

| atoms / factor | all-3 acc | lock% |
|---:|---:|---:|
| 20 | 99.5% | 99.5% |
| 50 | 94.0% | 95.5% |
| 100 | 71.5% | 72.5% |
| 200 | 16.5% | 17.0% |
| 400 | 0.0% | 0.0% |
| 800 | 0.0% | 1.0% |

At D=4096 the resonator factors a `~100³` space reliably and **falls off a cliff between 200 and 400 atoms per
factor** — two orders of magnitude short of a 4,000-word vocabulary, exactly where Frady/Kent say it will.
Blind structure discovery does not pay on real text vocabularies at affordable cost.

## (2) Analogy via mapping vector — VSA does NOT sharpen the count parallelogram (clean negative)

`T = a⊛b`; is `c⊛T ≈ d` for `a:b::c:d`? Same four families / protocol as Exp AD (744 items). Atoms either pure
random ±1 (the honest null) or **grounded** = `sign(PPMI_profile @ gaussian)` (a ±1 random-projection sketch, so
a relation *can* live in the geometry). Restricted candidate set, macro top-1 / top-5:

| family | AD raw-count ppmi | VSA-rand | VSA-ground |
|---|---:|---:|---:|
| capital-country | 64 / 92 | 7 / 26 | 8 / 29 |
| currency | 30 / 100 | 27 / 93 | 30 / 87 |
| plural | 88 / 98 | 8 / 33 | 42 / 72 |
| gender | 43 / 87 | 4 / 30 | 17 / 52 |
| **MACRO** | **56 / 94** | 11 / 45 | **24 / 60** |

Grounded VSA lands at **24 / 60**, less than half of AD's **56 / 94** from the same counts. Dimension helps the
grounded sketch (restricted top-1: 17% @ D=512 → 28% @ D=8192) but never approaches the raw count. Sign-binarizing
a PPMI row into ±1 discards the *graded* co-occurrence weights the parallelogram rides on, so multiplicative `a⊛b`
binding is strictly lossier than additive 3CosAdd over the full profile. **The count parallelogram, read with a
cosine, remains the best gradient-free analogy organ we have.**

## Verdict — the three-way split is the finding

- **(1a) keeper.** HRR role-filler binding + cleanup recovers subject/verb/object perfectly (100%, robust to 8
  bound pairs, even at D=512). Structure survives a sum, gradient-free, online — the count side could *compose*
  (AD) but never *read a composition back out*. This is genuinely new, **iff a structure source supplies the slots.**
- **(1b) parked negative.** The resonator factors only tiny alphabets (cliff at 200–400 atoms/factor); do not
  expect word-scale blind factorization at D ≤ 8192.
- **(2) negative.** VSA mapping-vector analogy is strictly worse than AD's raw-count 3CosAdd (24 vs 56). Keep the
  count parallelogram as the analogy organ.
- **The architectural payoff:** VSA gives compositional decode **iff a structure source supplies the roles/slots**
  — and we *have* one. **Exp AH (representational redescription)** produces explicit, slot-addressable concepts;
  **Exp AL's System-2 workspace** manipulates exactly these role-filler bundles. The real path is *VSA-decode over
  redescribed slots*, not blind factorization — AN + AH + AL compose into one route to count-native compositional
  structure. Don't ask the resonator to discover the frame; let redescription hand it over.

Online throughout: random projections + cleanup over the atom codebook, no gradients, no SVD, no backprop.
`lib/vsa.py`, `exp_an_resonator/run.py`. Whole run ~22 min CPU, single pass, fixed seed.

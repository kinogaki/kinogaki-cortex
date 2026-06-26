# Exp AN — resonator / HRR compositional decode — 2026-06-26

Vector Symbolic Architecture, gradient-free. text8 16 MB, 2.73 M words; 4,000 (subject, verb, object) trigram
records mined from the top-4,000 words. Atoms = fixed random ±1 hypervectors; bind = elementwise product;
bundle = sign(sum) (the HDC default); permute (cyclic shift) tags slot order. Online, single pass, no gradients,
no SVD. Two tests: decode *when the frame is known* (HRR), and decode *blindly* (resonator).

*(Recovered from the run output: the experiment agent looped on re-running and never wrote this file; the numbers
below are from the completed run, captured before the loop. The analogy-via-mapping-vector probe was not recovered
and is flagged as a follow-up.)*

## (1a) Slot decode with KNOWN roles — `T = role_s⊛S + role_v⊛V + role_o⊛O`, unbind a role, clean up → PERFECT

| dim D | subj | verb | obj | all-3 |
|---:|---:|---:|---:|---:|
| 512 | 100% | 100% | 100% | 100% |
| 1024–8192 | 100% | 100% | 100% | 100% |

Capacity (D=4096, recover slot-0 from a bundle of *k* role-filler pairs, extra fillers random): **100% at every
k from 1 to 8.** Structure survives the sum and comes back out exactly — even at low dimension (512) and high
load (8 bound pairs). **When the roles/frame are supplied, HRR compositional reading is solved.**

## (1b) Resonator — factor `s = ρ⁰(S)⊛ρ¹(V)⊛ρ²(O)` with NO roles known → FAILS at affordable dimension

| D | F | all-F acc | per-slot | lock% | avg-iters |
|---:|---:|---:|---:|---:|---:|
| 1024 | 2 | 0.0% | 0.2% | 0.0% | 60.0 |
| 1024 | 3 | 0.0% | 0.0% | 51.5% | 47.4 |
| 1024 | 4 | 0.0% | 0.0% | 1.2% | 59.9 |
| 2048 | 2 | 0.5% | 3.9% | 0.7% | 59.8 |
| 2048 | 3 | 0.0% | 0.0% | 60.5% | 46.9 |
| 2048 | 4 | 0.0% | 0.0% | 6.0% | 59.1 |

The resonator **converges** (lock-rate up to 60% at F=3) but to the **wrong factors** — ≈0% all-F recovery over a
4,000-atom codebook at D ≤ 2048. This is the documented resonator capacity wall: factoring a product of F *unknown*
atoms over a vocabulary this size needs an impractically high dimension. Blind structure discovery does not pay
on real text vocabularies at affordable cost.

## Verdict — the split is the finding

- **Compositional reading is solved when structure is supplied.** HRR role-filler binding + cleanup recovers
  subject/verb/object perfectly (100%, robust to 8 bound pairs, even at D=512). Structure genuinely survives a
  sum — gradient-free, online. The representation works.
- **Blind structure discovery (resonator) fails at affordable cost** — it converges to the wrong factorization
  over a real codebook at D ≤ 2048 (the known capacity limit).
- **The architectural payoff (why this matters):** VSA gives compositional decode **iff a structure source
  supplies the roles/slots** — and we *have* one. **Exp AH (representational redescription) produces exactly the
  explicit, slot-addressable concepts** this decode needs; and **Exp AL's System-2 workspace manipulates exactly
  these role-filler bundles.** So the real path is *VSA-decode over redescribed slots*, not blind factorization —
  AN + AH + AL compose into one route to count-native compositional structure. Don't ask the resonator to
  discover the frame; let redescription hand it over.
- **Honest gaps:** the analogy-via-mapping-vector comparison to Exp AD (raw-count 3CosAdd, ~56%) was not recovered
  from this run. Follow-up: that probe, and the AN+AH integration (decode redescribed slot-objects).

Online throughout: random projections + cleanup over the atom codebook, no gradients, no SVD, no backprop.
`lib/vsa.py`, `exp_an_resonator/run.py`.

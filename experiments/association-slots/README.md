# Streaming-association slot strength (ΔP / PPMI)

**Experiment AW · 2026-06-26 · refines AF**

AF scores an open construction slot's filler-categories by raw commitment ratio `r = c(f,s)/c(f,·)` — the
base-rate-blind conditional `P(s|f)`. AW (M5) swaps that substrate for streaming **association**: four additive
marginals per `(frame, category)` — `c(f,s), c(f,·), c(·,s), N` — yield `ΔP = P(s|f)−P(s|¬f)` and
`PPMI = max(0, log(c(f,s)·N/(c(f,·)·c(·,s))))` on demand, discounting each category by its global base rate. On
text8 (AF's pipeline, equal memory) this is a **partial/park**: association does **not** beat AF's −39.5 %
over-generation veto (PPMI's hard veto zeroes 27 % of over-generation links but also 19 % of real ones), and ΔP
edges raw counts on held-out compositional perplexity by only **0.7 %** (PPMI is 11 % worse). Park as "raw counts
suffice for English text," with the caveat that ΔP never hurts and prunes the slot table to 75 % of raw — its home
is CDS / free-word-order corpora and a hard-veto generation guardrail, not English-text perplexity. Online,
single pass, bounded, no backprop. Mechanism in `lib/assoc.py`.

**Run it** (the only python with numpy):

```sh
cd experiments && exp_a_boundary/.venv/bin/python exp_aw_assoc/run.py
```

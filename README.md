# kinogaki-cortex

An **online, count-based, no-backprop, inspectable** model of text — brain-inspired.

Today's language models are one enormous frozen function that has memorized the
statistics of text. kinogaki-cortex is the opposite: a model that reads the way a
person does. It learns from every sentence as it arrives, never retrains, never
forgets, and writes down what it learns as things you can open and read. Thousands
of tiny predictors each guess the next character from their own vantage point and
vote; when their agreement lurches, the system carves a boundary and mints a
concept; concepts stack into higher concepts. The whole mind stays a readable
file, not a black box.

There is one hard rule across every experiment here: **no gradient descent, no
backprop, no batch optimization that revisits the data.** Every model is a single
streaming pass of counters — it sees a piece of text once, updates its counts, and
predicts by looking them up. That is the property a transformer cannot match: it
learns online and a sparse update barely touches what it already knew, so it does
not forget.

This repository is the lab, preserved as rerunnable code. The narrated version
lives at **[cortex.kinogaki.com](https://cortex.kinogaki.com/)**, which is now in
two layers: the root **[/](https://cortex.kinogaki.com/)** is the living theory —
one coherent, refined statement of where the work stands, with a changelog of the
findings that shifted it; the **[/blog/](https://cortex.kinogaki.com/blog/)** is
the findings journal — every experiment as a dated post (question, attempt, the
one number that mattered, the honest lesson), newest first.

## Standing rules

Four rules hold across every experiment. Each is a fact about how a mind works,
and together they describe the only kind of intelligence we are trying to build:
one that learns as it lives, on a finite budget, the way a person does. The
narrated version is the blog's [How we work](https://cortex.kinogaki.com/blog/how-we-work/).

1. **Online only.** A single streaming pass, learn-while-it-lives. No gradient
   descent, no backprop, no batch optimization that revisits the data — no
   k-means, no SVD. Counting, leaky accumulators, and online leader-clustering
   only. The model learns from every sentence as it arrives and a sparse update
   barely touches what it already knew, so it does not forget.
2. **Fragile ideas.** "No better than a bigram" is the *normal* first result of a
   real idea, not a verdict — the space is high-dimensional, so wins compound
   late; nurture an idea 10–20 steps before any kill, check the *other* metric
   dials (a win often hides on calibration, rare-context, or transfer, not the
   headline), and shelve to a graveyard with a note, not the trash. See
   [research/FRAGILE_IDEAS.md](research/FRAGILE_IDEAS.md).
3. **Bounded memory.** The model runs on an explicit memory budget — no unlimited
   counts. Like a mind in the real world it copes through generalization, sleep
   cycles (offline consolidation), and using the environment as external memory.
   This re-elevates the mechanisms that "vanished" at unbounded scale: under a
   budget they are how a bounded model approximates the unbounded one. See
   [research/MEMORY_CONSTRAINT.md](research/MEMORY_CONSTRAINT.md).
4. **Human cognition is the guiding model.** Human intelligence, flaws included, is
   the existence proof of general intelligence under exactly our constraints, so we
   follow it — System 1 vs System 2, working memory, metacognition — and treat
   biases as adaptive features, not bugs. Everything built so far is System 1
   (fast, associative); a count-native System 2 (slow, serial, working-memory-bound,
   deliberate) is now a primary direction. See
   [research/COGNITION_AS_GUIDE.md](research/COGNITION_AS_GUIDE.md).

And one practice that follows from the first rule: **we publish the negatives.** A
research log that only shows wins is a sales brochure. The raytracing idea that
lost to a bigram, the topic cache that helped then hurt, the boundary signal that
scored below random — each has a post. A clean negative closes a door, names why,
and points at the door worth trying.

The through-line that holds across every experiment: **each concept level helps
predict the level it operates on** — word concepts help characters, phrases and
topics help words — learned online by counting, combined by a product of expert
opinions, kept inspectable throughout. The frontier we have not crossed is
**global coherence**: the model writes fluent local word-salad. The next swings aim
there.

## The experiments

In order, A through AK. "Verdict" is the honest call, negatives included. Each row
links to its narrated blog post and its code folder. The full lineage — why each
experiment grew from an earlier one — is in
[research/PROVENANCE.md](research/PROVENANCE.md).

| code | title | date | verdict / one-line result | post | folder |
|---|---|---|---|---|---|
| Exp A | Finding where one word ends | 2026-06-25 | qualified win — branching-entropy recovers word boundaries (F1 0.775); Bayesian surprise scores below random at the char scale | [link](https://cortex.kinogaki.com/boundaries-from-chars/) | [experiments/boundaries-from-chars](experiments/boundaries-from-chars) |
| Exp B | The counter beat the neural net | 2026-06-25 | win — a plain online counter beat both gradient nets (2.3 vs ~3.5 bpc); became the substrate | [link](https://cortex.kinogaki.com/associative-vs-gradient/) | [experiments/associative-vs-gradient](experiments/associative-vs-gradient) |
| Exp C | Words that lower the cost of letters | 2026-06-25 | win — word concepts cut char cost 22% (given) / 17% (discovered, no labels) | [link](https://cortex.kinogaki.com/concepts/) | [experiments/concepts](experiments/concepts) |
| Exp D | When combining the experts made it worse | 2026-06-25 | negative — full 5-expert vote lost to simple char+lexicon mix; char prediction saturates | [link](https://cortex.kinogaki.com/voting/) | [experiments/voting](experiments/voting) |
| Exp E | The hierarchy pays off at the right altitude | 2026-06-25 | win — measured at the word level, perplexity nearly halved (476→247) | [link](https://cortex.kinogaki.com/word-level-compounding/) | [experiments/word-level-compounding](experiments/word-level-compounding) |
| Exp F | How big a brain the data wants | 2026-06-25 | win — best capacity grows with the corpus; small saturates, large keeps learning | [link](https://cortex.kinogaki.com/scaling-law/) | [experiments/scaling-law](experiments/scaling-law) |
| Exp G | Building the dials bits-per-char hides | 2026-06-25 | win — new rulers (overfit, real-word rate, phrase coherence); named the global-coherence frontier | [link](https://cortex.kinogaki.com/the-scorecard/) | [experiments/the-scorecard-g](experiments/the-scorecard-g) |
| Exp H | Scoring the levels on the new dials | 2026-06-25 | win — word concepts halve the overfit gap (real-word 77%→89%); phrases lift coherence to 82% | [link](https://cortex.kinogaki.com/the-scorecard/) | [experiments/the-scorecard-h](experiments/the-scorecard-h) |
| Exp I | One part, repeated, wired bigger | 2026-06-25 | win — uniform Column wired wider+deeper improves cost and coherence (81%→94%) | [link](https://cortex.kinogaki.com/one-part-repeated/) | [experiments/one-part-repeated-i](experiments/one-part-repeated-i) |
| Exp J | The combiner is the hinge | 2026-06-25 | win — calibrated geometric-mean pool fixes overconfidence while keeping fluency | [link](https://cortex.kinogaki.com/one-part-repeated/) | [experiments/one-part-repeated-j](experiments/one-part-repeated-j) |
| Exp K | The level that reaches past the last few words | 2026-06-25 | clarifying — data is the lever; more fixed local levels saturate (the "deep didn't pay" wall was data starvation) | [link](https://cortex.kinogaki.com/depth-at-scale/) | [experiments/depth-at-scale](experiments/depth-at-scale) |
| Exp L | Associative attention vs fixed n-grams | 2026-06-25 | predecessor to Exp S — content-based count-attention, the rougher first cut; no blog post | — | [experiments/associative-attention](experiments/associative-attention) |
| Exp M | Finding phrases the way you'd guess them | 2026-06-26 | mixed — phrases are a clear unsupervised win; topic boundaries real but weak | [link](https://cortex.kinogaki.com/boundaries/) | [experiments/boundaries](experiments/boundaries) |
| Exp N | More data helps, all the way to a gigabyte | 2026-06-26 | win — cost falls to 822 MB (1.997→1.744); the "deep didn't pay" was a 2 MB artifact | [link](https://cortex.kinogaki.com/gigabyte-and-gpu/) | [experiments/gigabyte-and-gpu-n](experiments/gigabyte-and-gpu-n) |
| Exp O | Three engines, one answer (incl. GPU) | 2026-06-26 | win — sorted / dense / Metal-GPU columns agree; the GPU makes the gigabyte affordable | [link](https://cortex.kinogaki.com/gigabyte-and-gpu/) | [experiments/gigabyte-and-gpu-o](experiments/gigabyte-and-gpu-o) |
| Exp P | Meaning is a map, not a road | 2026-06-26 | negative — the meaning-space is real and beautiful but does not predict the next word | [link](https://cortex.kinogaki.com/raytracing/) | [experiments/raytracing](experiments/raytracing) |
| Exp R | A vote that remembers what it just saw | 2026-06-26 | qualified — fresh vote wins clean; the leaky accumulator wins past 10% noise | [link](https://cortex.kinogaki.com/evidence/) | [experiments/evidence](experiments/evidence) |
| Exp S | Attention, but counted instead of trained | 2026-06-26 | clear win — count-keyed offset attention cuts perplexity 3x under a bigram, no gradients | [link](https://cortex.kinogaki.com/offset-attention/) | [experiments/offset-attention](experiments/offset-attention) |
| Exp T | When the whole room agrees on a topic | 2026-06-26 | qualified — topic hurts at the char level, helps words where local context ran out; beats shuffled topic | [link](https://cortex.kinogaki.com/ignition/) | [experiments/ignition](experiments/ignition) |
| Exp U | Predicting the kind, not the word | 2026-06-26 | qualified — counted-cluster head beats the token head 65.4% vs 6.7%; cannot collapse | [link](https://cortex.kinogaki.com/jepa/) | [experiments/jepa](experiments/jepa) |
| Exp V | You can't write your signature backwards | 2026-06-26 | clear win — a memory of change transfers to unseen words (+25% vs +114%), runs one direction | [link](https://cortex.kinogaki.com/trajectory-memory/) | [experiments/trajectory-memory](experiments/trajectory-memory) |
| Exp W | We gave the map its best shot | 2026-06-26 | negative — the fair rematch for proximity (Exp P): given the graph form, the best stack, and the rare-context slice, proximity still has no prediction niche (rare gap −4.8 ppl, not significant); evidence earns its keep there (+12.5 ppl, significant). Parked deeper | [link](https://cortex.kinogaki.com/raycortex/) | [experiments/raycortex](experiments/raycortex) |
| Exp X | One brain part, or many? | 2026-06-26 | negative — specialization-by-level loses on bpc; the lone win is dynamic routing > static pool (~0.9 bpc) | [link](https://cortex.kinogaki.com/heterogeneous-stack/) | [experiments/heterogeneous-stack](experiments/heterogeneous-stack) |
| Exp Y | When the letters lie, it leans on the idea | 2026-06-26 | qualified — under input-only noise the concept stack degrades 2.7x slower than a flat bigram, and the gate routes prediction mass from letters to concepts (86%→95%) with no noise signal given | [link](https://cortex.kinogaki.com/noise-concepts/) | [experiments/noise-concepts](experiments/noise-concepts) |
| Exp Z | Use the map to read, not to walk | 2026-06-26 | qualified win — proximity failed as a predictor, so we poured it into the counter as a backoff prior; on the unseen-pair slice the similarity cluster cuts perplexity ~20x without changing the top guess | [link](https://cortex.kinogaki.com/sim-hybrid/) | [experiments/sim-hybrid](experiments/sim-hybrid) |
| Exp AA | What an agent learns while it dreams | 2026-06-26 | qualified win — one offline sleep pass (prune + distil) cuts memory 37% and improves rare-context bpc with no new data; keep dreaming and it goes generic and lossy (Letta's failure mode) | [link](https://cortex.kinogaki.com/sleep-consolidation/) | [experiments/sleep-consolidation](experiments/sleep-consolidation) |
| Exp AB | How sure is a count? | 2026-06-26 | qualified win — a NARS truth value (split a count into hits/misses) calibrates for free (ECE 0.280→0.027, 10x) and cuts perplexity 12.4→4.3 as an expert weight; the knob-free gate wins only on the rare slice | [link](https://cortex.kinogaki.com/calibrated-confidence/) | [experiments/calibrated-confidence](experiments/calibrated-confidence) |
| Exp AC | What the model thinks is happening | 2026-06-26 | qualified win — Bayesian surprise KL(Pt‖Pt-1) beats per-token surprisal 5.7x and branching-entropy ~120x at finding real article boundaries (F1 0.154 @±25), lead grows with scale; event-slot prior helps only on the 1% backoff slice | [link](https://cortex.kinogaki.com/event-model/) | [experiments/event-model](experiments/event-model) |
| Exp AD | Is the analogy already in the counts? | 2026-06-26 | qualified win — raw PPMI 3CosAdd solves a:b::c:? at 56/94% (~4x baseline), no SVD/word2vec; two negatives: leader-clustering blurs the relation axes and NARS induction spreads mass too broadly to beat a direct counter | [link](https://cortex.kinogaki.com/analogy-in-counts/) | [experiments/analogy-in-counts](experiments/analogy-in-counts) |
| Exp AE | Learning the new without losing the old | 2026-06-26 | qualified win — under a real register shift (Darwin→Shakespeare→Bible) at bounded memory, the brain-inspired stack forgets ~21x less than a recency cache (+0.021 vs +0.454) and has the better peak; ART resonance was load-bearing | [link](https://cortex.kinogaki.com/non-forgetting/) | [experiments/non-forgetting](experiments/non-forgetting) |
| Exp AF | Grammar is just counting, made productive | 2026-06-26 | qualified win — count token + type frequency and a flat n-gram turns compositional; on held-out unseen (frame,filler) pairs the open-slot construction beats the n-gram 4.3x on perplexity (5405 vs 23461) | [link](https://cortex.kinogaki.com/constructions/) | [experiments/constructions](experiments/constructions) |
| Exp AG | Thinking slow, by counting | 2026-06-26 | **theory update** · win + honest negative — a count-native System 2: a dual gate (calibrated confidence + Botvinick conflict) deploys a deliberate pass that overrides System 1 only when wrong (+0.38 acc on conflict cases, 0.000 harm on no-conflict, bit-for-bit fallback at zero budget); the gate is the win, the elaborate serial workspace loses to a trivial "defer to wider context" (0.15 vs 0.39) and is parked for the multi-step task | [link](https://cortex.kinogaki.com/blog/deliberate-pass/) | [experiments/deliberate-pass](experiments/deliberate-pass) |
| Exp AH | When a habit becomes a thought | 2026-06-26 | **theory update** · qualified win — stability (not error) redescribes a mastered construction into an explicit, slot-addressable concept answering queries the flat count can't (inverted slot lookup, role substitution, slot analogy); KS U-shaped dip confirmed (0.155 → trough 0.051 → recovered 0.181 above baseline); the manipulable operands System 2 deliberates over | [link](https://cortex.kinogaki.com/blog/redescription/) | [experiments/redescription](experiments/redescription) |
| Exp AI | The shape of forgetting | 2026-06-26 | honest negative — ACT-R's power law is the right shape (spacing: spaced 8.96x more accessible; EMA can't represent it) but raw-count LFU wins eviction for dense char-grams at every cap (LFU = the d→0 limit; decay sweep degrades monotonically) and power-law-weighted prediction loses (+0.68 bpc); right shape, wrong place — reserve the power law for word/concept level | [link](https://cortex.kinogaki.com/blog/shape-of-forgetting/) | [experiments/shape-of-forgetting](experiments/shape-of-forgetting) |
| Exp AJ | Less is more, and you can prove it | 2026-06-26 | **theory update** · clear win — validity-ordered, noncompensatory, early-stopping take-the-best beats full geometric-mean integration on every axis (acc 15.00% vs 9.71%, ppl 1918 vs 7160, 4.56 vs 8 cues/step); less-is-more (α>β) confirmed on sparse contexts; base-rate prior γ>0 lowers clustering stability (honest negative); revises the standing combiner | [link](https://cortex.kinogaki.com/blog/less-is-more/) | [experiments/less-is-more](experiments/less-is-more) |
| Exp AK | Starting small, on purpose | 2026-06-26 | **theory update** · honest negative — growing the memory budget does NOT beat full-from-start (FULL 2.744 vs GROW 2.751, robust; fixed-small loses 30%); "starting small" was a property of the gradient optimizer, not of learning — a count learner can't get stuck, so no curriculum needed, only enough final memory; ZPD overlay hurts (−5.8%) | [link](https://cortex.kinogaki.com/blog/starting-small/) | [experiments/starting-small](experiments/starting-small) |

## Cross-cutting threads

- **The surprise signal climbs the hierarchy.** One branching-entropy signal carves
  word boundaries (A), phrases (M), a memory of change (V), and a vote that
  remembers (R).
- **Saturation is the recurring teacher.** Character prediction saturates, which
  killed the all-expert vote (D), redirected the win to the word level (E), and
  explained why more fixed local levels stop paying (K).
- **Counting replaces trained machinery.** Attention without gradients (S), a JEPA
  that cannot collapse because it is counted (U), and a meaning-space built by
  co-occurrence (P) — each rebuilds a neural idea out of counts.
- **The frontier is global coherence.** Named by the scorecard (G/H); the later
  swings (S, T, U, V) all aim past the local context toward it.

See [research/PROVENANCE.md](research/PROVENANCE.md) for the full edge-by-edge
lineage, and [research/LAB_NOTEBOOK.md](research/LAB_NOTEBOOK.md) for the running
log.

## Scaling studies

Every experiment above was first measured on a few megabytes. We re-ran them all
at 30–200× the data — half a billion words, three billion characters — to learn
which verdicts hold when data is no longer the bottleneck. The line was sharp:
mechanisms that compete with local counts on already-seen prediction **vanish**
(topic prior +0.34 bits/word → 0.0), and mechanisms that do what counting *can't*
**hold or grow** (Bayesian-surprise boundaries F1 0.154 → 0.447). The raw batches
and the dichotomy are in [scaling/](scaling/); the narrated synthesis is the post
**[What survives scale](https://cortex.kinogaki.com/what-survives-scale/)**.

| study | corpus | what it asks | report |
|---|---|---|---|
| Batch 1 — single mechanisms at scale | LM1B (~526 M words / 3.03 B chars) | does each single mechanism keep paying to 500 M–3 B units? | [scaling/REPORT.md](scaling/REPORT.md) · [RESULTS.tsv](scaling/RESULTS.tsv) |
| Batch 2 — synthesis experiments re-run at scale | LM1B + enwik9 + register files | does each synthesis verdict survive on its own right-axis metric? | [scaling/REPORT2.md](scaling/REPORT2.md) · [RESULTS2.tsv](scaling/RESULTS2.tsv) |

## Layout

```
experiments/<slug>/   one folder per experiment: run.py, RESULTS.md, README.md
lib/                  shared modules (corpus, columns, cortex, attention, …)
data/                 datasets are NOT committed — run data/get-data.sh
research/             VISION, FRAGILE_IDEAS, PROVENANCE, LAB_NOTEBOOK, and more
```

## Running

See [SETUP.md](SETUP.md). In short: Python 3.x, `pip install numpy` (MLX optional,
Apple-Silicon GPU; experiments fall back to numpy), `bash data/get-data.sh`, then
`python experiments/<slug>/run.py`.

## License

Apache-2.0. Copyright 2026 Kinogaki LLC. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

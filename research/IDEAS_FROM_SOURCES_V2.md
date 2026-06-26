# Buildable ideas from the sources (V2) — mined 2026-06-25 (autonomous)

Re-read of ALL 167 Thousand Brains Project transcripts + papers/brain/gofai/grounding/forums (10 parallel
readers), filtered for **count-based, online, no-backprop** mechanisms buildable on our `Column`. Ranked by
cross-source consensus. Each idea names what it helps and (where sharp) the source.

## The convergence (what nearly every reader independently surfaced)

**1. Evidence accumulation with decay — the single most-recommended idea.** Don't eliminate hypotheses on one
mismatch; keep a leaky log-evidence score per hypothesis: `e ← γ·e + log(count(obs|h)+α)`. Properties everyone
flagged: (a) robust to noisy/garbled context (one bad token can't zero a hypothesis); (b) **bounded vs unbounded
sum = an explicit memory-horizon knob** (past+present weights =1 → finite memory; >1 → never forget); (c)
**prediction-error = the drop in the winner's evidence = a content-free boundary signal** (complements branching
entropy); (d) update only the **top-k% hypotheses** within δ of the max → fixed compute as the model commits.
→ *Helps: voting, temporal-memory, boundaries, global-coherence. The robust core of every Level's decode.*

**2. Offset/displacement as a first-class count KEY → "count-attention".** A skip-gram that ignores position is
a bag-of-words (scrambled mug still votes "mug"). Fix: count `(B | A, signed-offset-bucket)` and **vote
position-transformed** — a Column at offset d shifts its prediction by d into a common coordinate before pooling.
Multiple readers independently derive that this **IS self-attention with unlearned weights** (key = offset
compatibility, value = count distribution, query = which offsets are informative). Kills bag-of-words, stays
backprop-free, GPU-cheap (one extra small axis on the count tensor).
→ *Helps: attention, voting, global-coherence. The load-bearing upgrade to what we already have.*

**3. Top-down prior / ignition broadcast — the global-coherence lever.** When a higher Level commits a winner
(its evidence crosses an ignition threshold), broadcast it to a small shared context `G`; lower Levels condition
their counts on `G` (`count(next | local-ctx, G)`). Two channels (papers/brain + TBP agree): **L1 = broad global
id** (topic/word we're in) and **L6 = point-specific** (expect *this* next element). Quantify exactly: plot
per-token surprisal with vs without the prior — expect a spike at boundaries that the prior flattens.
→ *Helps: global-coherence, hierarchy, attention. The most concrete attack on our named frontier.*

## Tier 2 — strong, build after the core

**4. Proximity/raytracing = spreading activation over a PMI association GRAPH, NOT Euclidean coords.** The
forums document (and two senior HTM voices confirm) that a metric reference frame for *language* is a dead end —
symbolic categories have no natural origin/unit. The buildable form everyone converges on: edge weight =
PMI/co-occurrence; "cast a ray" = activate context nodes, spread 1–2 hops (`a' = A·a`), each reached column votes
weighted by activation × base-level recency (ACT-R). Variants: overlapping-SDR / voxel-hash so near contexts
share buckets (built-in smoothing); superior-colliculus "place a bump, gather nearby, move the bump = motor".
*(Our Exp P tests both the Euclidean and the graph form head-to-head.)* → *proximity/raytracing, attention, voting.*

**5. Information-gain weighting + expected-disambiguation attention.** Weight each context word by
`H(next) − H(next|word)` (validates our existing informativeness attention — TBP derives the same). Make it
*dynamic*: given the top-2 hypotheses, attend the offset whose count-distributions diverge most
(`argmax KL(p_h1‖p_h2)`) — "read where it splits the candidates", a count-based active-attention policy, no RL.
→ *attention, voting, global-coherence.*

**6. Chunking / K-lines — how phrase-columns should be BORN.** When the vote settles low-entropy/high-agreement
(our branching-entropy cliff), mint a new Column for that coalition / recurring subsequence; trigger on **high
contextual diversity** (a unit seen in many distinct contexts → factor it out, MDL). Replaces fixed n-gram
"phrase" levels with discovered units. → *architecture, boundaries, global-coherence.*

**7. Generic + specific two-tier (backoff, both active).** A shared low-order "morphology" model predicts
everywhere cheaply; a sparse high-order "specific" model overrides only at high-confidence/high-surprise points
(= Kneser-Ney backoff, reframed; specific *inhibits* generic only where it has mass). Also the fast/slow split:
fast-decaying episodic scratch-pad over recent text + slow durable counts. → *architecture, global-coherence, temporal-memory.*

**8. Entropy-gated write + variable-rate sampling.** Only commit a count / mint a context where prediction is
surprised; stride past low-entropy runs. Compressed, informative-where-it-matters memory; surprise also gates
*learning rate* (predicted → tiny increment, surprising → large). → *boundaries, temporal-memory, attention, efficiency.*

## Tier 3 — infrastructure & architecture hygiene

- **Divisive-normalization pooling + precision-weighted votes** (`P(w)=Σ A_c·p_c(w) / (Σ+σ)`, `A_c`=column
  reliability=1/recent-NLL): a calibrated drop-in for geometric-mean pooling with a universal gain knob. *(brain)*
- **Multi-scale coprime phase code as a position key** (grid cells): tuple of `pos mod p_i` localizes in long
  text cheaply; union-of-bumps = keep several position hypotheses, collapse on a cue. *(papers)*
- **Voxel-hash / LSH count store** — O(1) incremental, similar contexts share buckets; the GPU count substrate.
- **Distributional classes for free** — cluster words by KL of their successor-distributions → emergent POS/topic
  classes, multiple overlapping ones from different context widths. No new mechanism.
- **Uniform messaging protocol** — every Column/Level emits `{log-count dist, confidence=neg-entropy, offset,
  sender}`; votes/goals/observations all the same struct → composable stacking. *(Monty CMP)*
- **Goal-states, not actions** — "motor = moving through text" = a controller sets a target next-state; the lower
  Column picks the move (which token to emit / where to jump) that reduces divergence. Stop = state match.
- **Driver vs modulator edges** — type edges: content (adds to the vote) vs gain (only sets another column's
  precision). Never mix. Keeps the proximity field a *modulator*, not content. *(brain)*

## Skeptical guardrails (what the sources warn against)

- **Euclidean reference frame for language = dead end** (forums, repeatedly). Use the association graph; derive
  geometry (SOM) only as an emergent layout if it pays off.
- **Don't pass models/SDRs between columns, only states**; **don't do O(n²) pairwise reference-frame transforms**
  — use sparse broadcast + 1–2 settle iterations (TBP abandoned the transforms as implausible/expensive).
- **TBP's own unsolved problems** (don't expect recipes): infinite-loop/event segmentation; movement through
  abstract/conceptual space; learning behavior efficiently with one column. Treat as research, not blueprints.
- SDR-overlap-as-similarity makes Hawkins himself uncomfortable across several talks — validate emergent classes
  help prediction before depending on them.

## Recommended build queue (consensus order)

1. **Evidence-accumulation Level** (#1) — robust leaky-log-evidence decode + evidence-slope boundary. Smallest
   change, biggest leverage; gives a label-free quality metric too.
2. **Offset-keyed count-attention** (#2) — add a signed-offset axis; position-transformed geometric-mean vote.
3. **Top-down prior / ignition** (#3) — commit higher-level winner to `G`, condition lower counts on it; measure
   the boundary-surprisal spike it flattens. ← the global-coherence experiment.
4. **Graph spreading proximity** (#4) — the endorsed raytracing; compare to Euclidean (Exp P) and skip-gram.
5. Then: chunking-born columns (#6), generic+specific backoff (#7), entropy-gated write (#8).

Per-reader raw outputs are preserved in this session's transcript; this doc is the synthesis.

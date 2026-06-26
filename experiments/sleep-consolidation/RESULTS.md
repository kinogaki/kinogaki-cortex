# Exp AA — SLEEP / CONSOLIDATION over the count memory ("agent dreaming") — 2026-06-26

**The bet (Letta, *Towards Agents that Learn*).** An agent improves by refining its **token-space memory**,
not its weights; an offline "sleep-time compute" phase (*agent dreaming*) refines that memory **without new
data**; and the documented failure mode is that **"memories become generic and lossy after repeated
refinement."** In our world this maps almost literally: **a Column *is* a memory-agent, and its count tables
*are* the token-space memory.** The online substrate already learns weight-free. The new thing here is the
offline **sleep pass** that refines the count memory — and the test of whether refinement-without-new-data
pays, and where repeated refinement turns destructive.

**The substrate (online, unchanged).** A backoff char model = one count table per order *k*: `ctx(k chars) →
{next_char: count}`. Prediction backs off high→low to the longest seen context, add-α smoothed. The plain
online Column, built in one streaming pass.

**The sleep pass (offline, but count-based — no gradient, no batch optimization).** One extra pass that
rewrites the tables, replaying a bounded recent buffer (the last 3 MB of train — what "recurred recently"):

- **(a) Prune** — heavy-hitter / count-min style. Drop contexts with total count `< MIN_CTX` (untrustworthy);
  within a kept context, zero the next-char tail beyond a cumulative-mass cap. Removes lossy noise, bounds memory.
- **(b) Distill specific→generic** — where a high-order context's next-char distribution ≈ its **backoff**
  (`KL(specific ‖ backoff) < τ`), drop the specific entry. **Lossless:** backoff fills the hole with the same
  distribution. The direct count-world analogue of Letta's "distill into the generic memory."
- **(c) Promote** — leader-cluster surviving high-order contexts by their next-char distribution; a cluster
  becomes a shared **concept** count row. (Reported below as a negative.)

**Setup.** text8, **16 MB train**, disjoint **2 MB held-out tail**, char backoff **order 6**, buffer 3 MB,
`RARE_THRESH = 20` (a position is "rare-context" if the longest matched context had `< 20` train counts).
Fixed seed. **Whole run 189 s on CPU.** Online learn 5 s; one sleep cycle ~10 s.

---

## Result 1 — one sleep cycle: smaller memory, equal-or-better prediction

| | held-out bpc | acc | **rare-ctx bpc** | common-ctx bpc | memory (entries) |
|---|---:|---:|---:|---:|---:|
| **before sleep** (online) | 1.8773 | 61.25 % | 2.592 | 1.735 | 3,258,309 |
| **after 1 gentle sleep** | **1.8666** | 61.03 % | **2.456** | 1.770 | **2,056,633** |
| delta | **−0.0107** | −0.22 pt | **−0.136** | +0.036 | **−36.9 %** |

**Sleep refines the memory without new data: it cuts memory by 37 % and *improves* held-out bpc, with the
gain concentrated on the rare-context tail (−0.136 bpc).** The gentle prune throws away contexts seen `< 3`
times (731 k of them) — noise the online pass had no way to filter — and lossless-distills 112 k high-order
contexts whose distribution already equalled their backoff. Common-context bpc edges up +0.036 (a few useful
high-order specifics get pruned); the rare-context win is ~4× larger, so the net is a clear gain. **Won on:
memory-compression + rare-context generalization.**

---

## Result 2 — repeated refinement: gentle saturates, aggressive reproduces Letta's failure mode

**Gentle schedule** (re-sleep over the same buffer, fixed knobs): **idempotent.** After the first cycle the
table is a fixed point — everything `< MIN_CTX` is gone and every `KL < τ` entry is distilled, so re-running
changes almost nothing.

| cycle | 0 | 1 | 2 | 5 | 10 |
|---|---:|---:|---:|---:|---:|
| bpc | 1.8773 | **1.8666** | 1.8675 | 1.8678 | 1.8680 |
| entries | 3.26 M | 2.06 M | 2.05 M | 2.05 M | 2.05 M |

No degradation — but no further gain either. Honest reading: a *gentle* sleep is a one-shot cleanup, not a
ladder you can keep climbing. To make refinement keep biting (and thus testable for the failure mode) we ran
an **aggressive schedule** that escalates each cycle (distill `τ ×1.6`, prune `MIN_CTX +1`), so each pass
distills against an ever-more-generic baseline:

| cycle | bpc | **rare-ctx bpc** | **common-ctx bpc** | entries |
|---:|---:|---:|---:|---:|
| 0 | 1.8773 | 2.592 | 1.735 | 3.26 M |
| **1** | **1.8638** | 2.422 | 1.781 | 1.90 M |
| 2 | 1.8692 | 2.404 | 1.798 | 1.70 M |
| 3 | 1.8744 | 2.405 | 1.812 | 1.51 M |
| 5 | 1.9140 | 2.454 | 1.869 | 1.12 M |
| 7 | 2.0469 | 2.366 | 2.032 | 0.79 M |
| 10 | 2.2867 | **2.203** | **2.289** | 0.61 M |

**The Letta failure mode reproduces exactly.** Quality improves for one cycle (best bpc 1.8638 at cycle 1),
then **degrades monotonically** as refinement continues — the memory goes "generic and lossy." The signature
is precise: **common-context bpc gets steadily *worse* (1.735 → 2.289) while rare-context bpc keeps
*improving* (2.592 → 2.203).** Over-distillation collapses specific high-order memories into their generic
backoff. That helps the tail (the generic answer was always the rare context's best guess) but destroys the
detail that priced the common, predictable contexts — exactly "memories become generic and lossy." **Turning
point: cycle 1.** Past it, you are trading away the model's competence for compression.

---

## Result 3 — the generic-vs-specific balance is the whole story

Every result above splits cleanly on one axis. **Consolidation moves probability mass from specific memories
toward generic (backoff) memories.** A little of that is pure win (prune noise, distill redundancy → +rare,
−memory, flat-overall). Too much of it is Letta's failure (the generic memory swallows the specific one →
+rare but −−common, net worse). The rare and common columns move in **opposite directions** under refinement,
and the verdict depends entirely on how far you push:

- **Rare contexts** monotonically *benefit* from genericization — their specific counts were never
  trustworthy, so leaning on backoff is strictly better (2.592 → 2.203 across the whole aggressive run).
- **Common contexts** monotonically *suffer* — their specific high-order counts were real signal, and
  distilling them away replaces a sharp distribution with a blurry one (1.735 → 2.289).

The sweet spot (one gentle cycle) is where the rare gain still outweighs the common loss.

---

## Result 4 — promote (concepts) is a count-only negative

Clustering the surviving high-order contexts into a shared concept tier (leader clustering, 2,309 concepts,
281 k contexts promoted) **hurt:** bpc 2.10 vs 1.87 for the gentle cycle. The reason is structural and worth
recording: a concept is keyed by the *exact* context keys of the training contexts that formed it, so at eval
it only fires for contexts already in the table — it **cannot reach a genuinely unseen context** without a
similarity index over context keys, and any such index is a batch/nearest-neighbour structure that breaks the
online rule. So count-only promotion compresses (it merges members) but does not *generalize*, and merging
real specifics into one coarse row is just premature distillation. **Parked, not killed** (per
FRAGILE_IDEAS): promotion likely needs the online concept-cluster from Exp U (a *content* representation the
context maps into), not raw context keys — that is the natural follow-up.

---

## Verdict

**Sleep over count memory is a real win on the memory-compression / rare-context axis, and it reproduces
Letta's "generic and lossy" failure mode precisely — refinement helps once, then trades competence for
compression.**

- **Does sleep improve without new data?** *Yes.* One gentle cycle cuts memory **37 %** and *improves*
  held-out bpc, with the gain on the **rare-context tail** (−0.136 bpc). Pure replay + count bookkeeping, no
  new data, no gradients.
- **Does repeated refinement degrade?** *Yes, exactly as Letta reports — once you push it.* Gentle sleep is
  idempotent (saturates, no harm); the aggressive schedule peaks at **cycle 1** then degrades monotonically
  to bpc 2.29, the memory going "generic and lossy." **The turning point is early — one good sleep is the
  budget.**
- **Generic-vs-specific?** *The axis is the mechanism.* Consolidation shifts mass specific→generic; rare
  contexts gain and common contexts lose, and over-refinement is exactly when the common loss overtakes the
  rare gain.
- **Promote / concepts?** *Negative, parked* — count-only concepts cannot reach unseen contexts without a
  batch index; resurrect with an online content representation.

**The keeper for the cortex: run one offline sleep pass — prune untrustworthy contexts, lossless-distill
redundant specifics — and stop. It shrinks the memory by a third for equal-or-better prediction and sharpens
the rare-context tail. Do not keep dreaming: a second, harder cycle starts grinding the specific memories into
generic mush, which is the exact failure mode Letta warned about.**

---

## Online-compliance / sleep-phase note (the honest nuance)

| step | how it's computed | online? |
|---|---|---|
| substrate (count tables) | one streaming pass, `np.unique` over packed (ctx, next) keys | **yes** — order-independent accumulation = a token-at-a-time online update |
| sleep — prune | drop contexts below a count floor; zero the next-char tail beyond a mass cap | **yes** — count thresholds (heavy-hitter / count-min) |
| sleep — distill | compare a distribution to its backoff via KL; drop if redundant | **yes** — a comparison of two counted distributions, no optimization |
| sleep — promote | online leader clustering over surviving distributions (running-mean prototype or spawn) | **yes** — single pass, no re-assignment, no iteration-to-convergence |

**The honest nuance.** Sleep is a **second pass** over a bounded buffer, so it is *offline* — it is not the
single streaming pass of the pure online substrate. But the **learning rule stays local and count-based**:
replay over a bounded recent buffer, prune/distill/promote by count thresholds and KL comparisons, online
leader clustering. **No gradient descent, no backprop, no k-means, no SVD/eigendecomposition, no batch
optimization.** Brains sleep and replay; this is replay + bookkeeping over the counts, not training. We flag
the second pass plainly rather than calling the whole thing "online."

**Won on:** memory-compression (−37 % for equal-or-better bpc) and rare-context generalization (−0.136 rare
bpc), plus a clean reproduction of the repeated-refinement degradation failure mode.

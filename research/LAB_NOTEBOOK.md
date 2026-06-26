# kinogaki-cortex — lab notebook

A running log of experiments toward the north star (`research/VISION.md`): an online, non-forgetting, sparse,
voting, concept-growing, **inspectable** model of text. Honest record — what worked, what didn't, what we
learned, what's next. Newest at the bottom.

The north star's most defensible, *transformer-can't-do-this* claim is **online learning without catastrophic
forgetting**. So that's where we hunt for a real, demonstrable result first.

---

## Exp A — boundary kill-test (`exp_a_boundary/`) — DONE, qualified PASS

**Q:** does prediction-error segmentation of raw chars recover word boundaries, and does *transient Bayesian
surprise* beat surprisal/entropy (Kumar & Zacks 2023)?
**Result:** boundaries ARE recoverable (forward+backward **branching-entropy rise**, F1 **0.775**, lit-grade) —
but **Bayesian surprise scored *below random*** at the char level. **Worked:** branching-entropy rise; the
online no-GPU loop; dogfooding kinogaki to persist discovered "words" as a `.prism` doc. **Didn't:** Bayesian
surprise as a low-level cue (it's a *semantic-event* signal, peaks mid-word at char scale).
**Learned / design change:** the boundary signal is **level-dependent** — branching at chars→words, Bayesian
surprise reserved for sentences→themes. Core "ambiguity=boundary" bet holds at the low level.

---

## Exp B — catastrophic forgetting (`exp_b_forgetting/`) — RUNNING

**Q (the make-or-break):** does a sparse, online, locally-updated learner avoid catastrophic forgetting where a
dense backprop net fails — on text, at comparable capacity? This is the core value-prop, isolated.

**Design (ENGLISH-ONLY — per the steer).** Char-level next-char prediction, streamed across four distinct
English *registers* in sequence — **Austen prose → Shakespeare verse → Darwin science → KJV Bible** — one
pass, **no replay**. All one domain (English), but enough topic/style shift to make forgetting observable.
After each register's training phase, evaluate bits-per-char + top-1 accuracy on a held-out slice of **every**
register → a retention matrix.
- **Dense** model: char-embedding → tanh hidden → softmax. SGD updates *all* weights every step → interference.
- **Sparse** model (our design): fixed random projection → **k-winner-take-all sparse code** → softmax readout.
  Because the code is sparse, each step's gradient only touches the *k active* readout columns → updates for a
  new domain barely touch the columns that carry an old domain → **localized learning, low interference**.
- **Count n-gram**: trivially non-forgetting reference (accumulates, never overwrites).
**Forgetting metric:** drop in domain-1 (english) accuracy after learning the later domains. Hypothesis: dense
forgets a lot; sparse retains most; at comparable peak accuracy. If clean → a real, useful demonstration.

**Result — NEGATIVE but useful.** No catastrophic forgetting appeared: austen bpc *improved* after later
registers (positive backward transfer), because 4 English registers ≈ one stationary char-level distribution —
nothing to forget. The differentiator test needs real task shift, which English-only-char-level doesn't give.
**Key insight that fell out:** the **online associative count model beat both gradient nets** (bpc 2.3 vs 3.5)
and is inherently online/non-forgetting/backprop-free → **our substrate should be sparse-associative, not SGD
nets**. The north star and the better predictor coincide.

---

## Exp C — does the concept hierarchy earn its keep? (`exp_c_hierarchy/`) — DONE, **POSITIVE** ✅

**Q:** does conditioning prediction on a learned WORD concept layer lower bits-per-char vs a flat char model
(pure associative substrate, no backprop)?
**Result (the real win so far):** YES. Flat char 8-gram **2.124 bpc** → **+ word concepts 1.653 (+22.2%)**;
**fully unsupervised** (branching-entropy-discovered words) **1.773 (+16.6%)**. Pure association, online,
non-forgetting, **inspectable** — the discovered concept store is real English words (the, a, to, and, her,
was, she, not…) in a `.prism` doc.
**Didn't compound:** word→word context (L2) added nothing (char-8gram already spans into the previous word);
a recency **cache** adds only +0.1% (mechanism works, tiny at this scale — only touches word-first-chars).
**Learned:** level-1 concepts are real and valuable; **compounding needs a level beyond the char window's
reach (phrases / topic), not word-context**; unsupervised discovery over-segments (the +16.6%→+22% gap).

---

## Exp D — integrated multi-level VOTING cortex (`exp_d_voting/`) — DONE, instructive NEGATIVE

**Q:** combine multi-level experts (char8, char3, word-lexicon, phrase=word-context, topic=cache) by adaptive
gated voting / product-of-experts — does it compound below Exp C's char+lex 1.653 bpc?
**Result:** NO. Gated linear voting 1.707, product-of-experts 1.767 — both **worse than the simple tuned
char+lex mix (1.653)**. **Why (the key insight):** *character* prediction **saturates** with local+lexical
context — the char-8gram already determines most chars, so phrase/topic structure barely moves char-bpc.
**Bits-per-char is the wrong metric for higher-level concepts.** Also learned: linear voting → winner-take-all
(picks best expert, doesn't blend); **product-of-experts (constraints multiply, uniform experts auto-abstain)
is the right combiner** — but only where the experts carry complementary info the base level lacks.

## Exp E — does the hierarchy compound at the RIGHT granularity? (`exp_e_wordlevel/`) — DONE, **POSITIVE** ✅

**Q:** test higher levels where context matters — predict the next WORD (not char). Experts: word-unigram,
bigram (phrase), trigram (longer phrase), cache (topic). Product-of-experts.
**Result:** YES, it compounds. **Word perplexity 476 → 247** (−48%), bits-per-word 8.90 → 7.95 (+10.7%).
Learned product weights put the mass on bigram (0.94) + trigram (0.67) — the phrase levels. **Each concept
level helps predict the level it operates on**: word-concepts→char (Exp C +22%), phrase/topic→word (Exp E,
perplexity halved). The architecture (multi-level concepts + product-of-experts voting, online, associative,
inspectable) **works** — measured at the right granularity.

## Exp F — scaling dynamics: data × cortex capacity (`exp_f_scaling/`) — DONE ✅

**Q:** before extending, characterize the system — how does prediction scale with training data (1→90×) and
cortex capacity (n-gram order)? On text8 (100MB English), vectorized count model, fixed held-out.
**Result (the grid, bits-per-char):** small cortex (K=2) is **capacity-saturated** — 90× data buys only
−0.057; large cortex (K=5) keeps improving (−0.587, still −0.064 on the last 3× step, **not saturated**).
And capacity helps **only with enough data**: K=5 *overfits* at 1MB (worse than K=4) but wins at ≥3MB →
**optimal cortex size grows with data** (the capacity×data scaling law). Order-5 = 1.82 bpc (PPM territory;
char-LSTM ≈1.4, SOTA ≈1.0). **We are in the productive, not-saturated regime** — more data + bigger cortex
(order + concept vocabulary + levels) will keep paying off.
**Implication:** cheapest next wins are *scale* (data + capacity), not new mechanisms. Concept layers (Exp C/E)
are a second capacity axis that compounds on top.

## Exp G — generalization metric suite (`exp_g_metrics/`) — DONE ✅

Built the measurement dials on text8 (backoff char model, train 10MB / test 2MB): **(1) overfitting** =
train/test bpc gap — grows with capacity (K=8 *overfits*: test 2.18 > K=6's 2.01, train 1.27); **(2) prediction
horizon** = greedy run ~1.8 chars, next-char acc ~60%; **(3) text-vs-gibberish** = % real words: real 95%,
random 6.6%, char-gen 85% (t0.7)→56% (t1.0). Verdict: char substrate masters *local* English, lacks *global*
coherence. These three are now the standard scorecard.

## Exp H — concept hierarchy (C/E) on the metrics (`exp_h_concept_metrics/`, reusable `lib/metrics.py`) — DONE ✅

Scored char-only vs +word-concepts (C) vs +phrases/word-level (E):
| level | test bpc | overfit gap | real-word% | phrase-coh% |
|---|---|---|---|---|
| char-only | 2.004 | +0.528 | 77 | 57 |
| + concepts (C) | 1.897 | **+0.315** | **89** | 46 |
| + phrases (E) | — | — | 100 | **82** |

**Each level buys exactly what it models:** word-concepts → generalization (overfit gap halved — the lexicon
is a *regularizer*) + word-validity; phrases → phrase-coherence (82%). **Neither yet gives GLOBAL coherence**
(all produce locally-plausible word-salad). North-star claim validated with real metrics, not just bpc; the
explicit frontier is now **discourse/global coherence** — the next level up, where the reasoning/movement
ambitions must deliver, and we have the dials to detect it.

## Exp I — the UNIFORM-COMPONENT cortex (`exp_i_uniform_cortex/`, `lib/cortex.py`) — DONE ✅

**Q (the steer):** stop adding mechanisms — build the whole system from ONE repeated part (a `Column`) and show
that *wiring it bigger* makes it better. "cortex small vs big." A `Column` = online associative backoff
predictor over a token stream; a `Level` = N Columns voting; a `Cortex` = stacked Levels (char-Columns predict
chars; word-Columns predict the current word from the previous words and hand a top-down char prior back down).
Grow by **WIDTH** (more char Columns voting) and **DEPTH** (stack a word Level → widen it to a phrase band).

**Result — the thesis holds, and the COMBINER is the hinge.** Two findings:
1. **The pooling rule decides everything.** Raw **product-of-experts** sharpens with every added column →
   fluent generation (phrase-coh 98%) but **overconfident, overfit likelihood** (bpc blows up to 3.4, overfit
   +2.9). Calibrated **geometric-mean (log-linear) pooling** → honest bpc (2.11) but tempered generation. Same
   product-vs-linear tension as Exp D, now isolated as a knob. **Recipe that gets both:** geometric-mean pool
   for scoring + **sharpen at SAMPLING time** (low temp) for coherence.
2. **Same Column, wired bigger → better, each step attributable** (geometric-mean pool, temp-0.5 sampling):
   | config | test bpc | overfit | real-word% | phrase-coh% |
   |---|---:|---:|---:|---:|
   | 1 char col | 2.401 | +1.028 | 98.9 | 81.5 |
   | 3 char cols (**wider**) | **2.119** | **+0.566** | 97.6 | 73.4 |
   | + word level (**deeper**) | **2.103** | +1.151 | 99.0 | **92.4** |
   | + phrase band | 2.112 | +1.200 | **99.2** | **93.6** |
   - **WIDTH buys calibration + generalization:** 1→3 char cols, bpc 2.40→2.12 and **overfit gap halved** — the
     voting columns are an ensemble that regularizes (pure replication of the one part).
   - **DEPTH buys coherence:** the word level lifts phrase-coherence 73→92% (the big jump) + best bpc.
   - **4th band = diminishing returns at 2 MB** — exactly Exp F's capacity×data law (bigger cortex needs more
     data); char-bpc itself is near its local ceiling (Exp D), so that dial saturates while coherence still climbs.

**Learned / honest read:** "better" is multi-dial — width→calibration, depth→coherence — and the geometric-mean-
pool + low-temp-sample recipe gets honest likelihood *and* fluent generation from the same component. Still
locally-fluent **word-salad**: global/discourse coherence (the Exp G/H frontier) is untouched. Biggest sample:
*"argentina argentine nation of the absence of autistic people s states or the common ancestor of the formal
naming convention and reducing government official."*

## Exp J — is the uniform Column a better BASE? (`exp_j_foundation/`, `lib/fastcol.py`) — DONE ✅

**Q (the "but first"):** before scaling to GBs or adding attention, is the `Column` abstraction *optimizable*
(not a pure-Python dead end), and does more data actually pay off? Same Column interface, two backends: dict
(readable) vs vectorized (`np.unique` builds the count table, `searchsorted` predicts).
**Result:**
- **Correctness + speed:** vectorized FastColumn computes the **identical model** (test bpc 2.3502 == 2.3502)
  at **9× faster learn** (2.41 s → 0.28 s, order 6, 2 MB). The abstraction is real — one interface, swappable
  guts; 9× is the floor (gap widens at higher orders / the word stack where tuple-hashing dominates).
- **Scale (the big one):** bpc **2.35 → 2.02 → 1.85** from 2 → 10 → 50 MB at ~7 MB/s. The data axis was
  **starved, not saturated** — Exp I's "4th band didn't pay off" was a 2 MB artifact. A **1 GB corpus learns in
  ~2–3 min**, so the data-hungry questions (4th level at 100× text? global coherence at scale?) are now affordable.
**Verdict:** the Column is the better base — flexible (expresses the A–H zoo by rewiring), optimizable (vectorizes
with no interface change; GPU-able next), and its predict/vote inner loop (gather-from-table + reduce-across-columns)
is the *same shape* as a path tracer's sample-and-accumulate → the Metal port is natural, not a rewrite.

---
# Autonomous session (2026-06-25 night) — scale, GPU, boundaries, attention

## Exp K — does depth pay off at scale? (`exp_k_scale/`) — DONE (10/40 MB; 80 MB ran too slow, killed)
3/4/5-level cortex × data. **More data helps a lot** (3-lvl bpc 1.94→1.77 from 10→40 MB; 5-lvl 1.83→1.70).
**4th level (word trigram) stays flat** (1.767 vs 1.770) — local word context saturates like char-bpc did.
**5th level (topic recency cache) helps bpc** (−0.07 to −0.11) and lowers overfit, but the edge is a *constant*,
not growing with data, and it slightly hurts phrase-coherence. **Lesson:** the payoff isn't *more fixed local
levels* — it's a level that reaches *beyond local context* (→ motivates attention + boundaries, not more n-grams).
The per-position Python eval here was the bottleneck → fixed with batch eval (Exp N).

## Exp N — gigabyte char scaling (`exp_n_gigabyte/`, `lib/fastchar.py`,`corpus.py`) — DONE ✅
enwik9 (1 GB → 827 M chars, id-space, loads in 3.6 s). Batch bpc == per-position bpc (2.3889, verified).
Order-5 char: **bpc 1.997 → 1.821 → 1.773 → 1.744** at 10/100/300/822 MB. **More data helps all the way to a
gigabyte** (diminishing: last 2.7× buys −0.03 — char order-5 near its capacity ceiling, wants order 6–8 which
GPU now affords). Gigabyte learns in **2.6 min** (CPU). The data axis was never the problem.

## Exp O — really fast columns + GPU (`exp_o_gpu/`, `lib/densechar.py`,`gpuchar.py`) — DONE ✅ (headline)
Same Column, same bpc, three backends: **np.unique 11.2 s → dense bincount 5.45 s → MLX/Metal GPU 0.33 s**
(100 MB); **36.9 → 18.3 → 0.74 s** (300 MB). Two wins: (1) dense histogram beats sorted-unique 2× on CPU (and
is what the GPU wants); (2) **MLX/Metal scatter-add = ~25–50× on learn** — a gigabyte column in ~2–3 s vs 156 s.
Capacity (order, many voting columns) is now effectively free. GPU's bigger payoff is still ahead: voting/attention
(many experts gathered+reduced in parallel). MLX (Metal under the hood) was the right call; C++/Metal only if a
fused gather-vote kernel outgrows it.

## Exp M — discover phrases & topics by surprise (`exp_m_boundaries/`, `lib/boundaries.py`) — DONE, mixed
Exp A's branching-entropy signal, one level up. **Phrases (qualitative win):** unsupervised word-stream cuts
recover real units — *united states, such as, see also, list of, according to, th century, to be, part of* (plus
markup junk from the crude enwik9 cleaner — a corpus issue). **Topics (real but weak):** TextTiling surprise vs
enwik9 `<page>` truth → F1 0.148 vs random 0.066 = **2.2× over chance**, but over-segments 2×. Surprise *does*
carry topic-boundary info; absolute strength needs markup-cleaning + rate-tuning + a predictive (not bag-of-words)
content signature. Payoff when solid: topic boundary resets the attention cache → segment-scoped attention.

## Exp L — associative attention (`exp_l_attention/`, `lib/attention.py`) — BUILT, eval too slow (killed)
Skip-gram associations + per-word learned informativeness weight (content words dominate, function words self-
suppress) + long context window; "attention but online/count-based, no Q/K/V." Implementation works, but the
per-position eval (400-char window × ~60 context words × searchsorts) is the same Python-loop bottleneck the char
models had — a single 10 MB run didn't finish in ~23 min. **Killed; needs a batched eval** (the fix that worked
for FastChar). The *concept* is strongly validated by the source-mining: it's the #2 consensus idea there
("offset-keyed count-attention = unlearned self-attention"). Re-run once batched.

## Exp P — raytracing / proximity columns (`exp_p_raytracing/`, `lib/spatial.py`,`graph.py`) — DONE, clarifying
Place word-columns in a space (PPMI + eigen-embed) connected by proximity; predict 3 ways.
**The space is real** (proximity=meaning: three→four,five,six; king→prince,son,daughter; france→spain,italy).
**But proximity does NOT predict next word:** bigram 20.85% vs spatial-gather 17.07%, **ray-extrapolation 1.79%**
(the "sentence = linear trajectory in embedding space" hypothesis is false), graph-spreading 20.73% (≈bigram).
Confirms the forums' warning: Euclidean RF for language is a dead end *for prediction* — the embedding is a
**similarity/generalization** structure, not a sequential predictor. Keep proximity as a *modulator* (rare-context
backoff prior + inspection), not content. The fair untested test: does gather/spreading beat bigram on RARE
contexts (where bigram has no counts)?

## Source mining (10 parallel readers over ALL 167 TBP transcripts + papers/brain/gofai/grounding/forums)
Synthesis in `research/IDEAS_FROM_SOURCES_V2.md`. Convergent, count-based, no-backprop build queue:
**(1) Evidence accumulation with decay** (leaky log-evidence per hypothesis; bounded=memory-horizon knob; top-k%
update; evidence-slope-drop = a 2nd boundary signal) — the most-recommended idea, the robust core of decode.
**(2) Offset-keyed "count-attention"** (count `(B|A,offset)`, position-transformed voting = unlearned self-
attention; kills bag-of-words). **(3) Top-down prior / ignition broadcast** (commit higher-level winner to shared
context G; condition lower counts on it — the global-coherence lever; L1 global + L6 specific channels).
Then: graph-spreading proximity (#4, Exp P confirms use it as modulator not predictor), info-gain + expected-
disambiguation attention (#5), chunking/K-lines born columns (#6), generic+specific backoff (#7), entropy-gated
write (#8). Guardrails: no Euclidean RF for language; pass states not models; no O(n²) transforms (sparse
broadcast + settle); TBP's own unsolved = event-segmentation loops, abstract-space movement.

## State of the research (2026-06-25)

**The coherent A–E story (the architecture works, validated piece by piece):**
- **A** — prediction-error boundaries recover words (branching-entropy rise, F1 0.775).
- **B** — associative > gradient substrate (count model beat the nets, and is online/non-forgetting). Forgetting
  itself isn't testable in English-only (positive transfer, nothing to forget).
- **C** — word concepts help *char* prediction: +22% (unsupervised +16.6%), inspectable `.prism` lexicon.
- **D** — char-bits-per-char **saturates** with local+lexical → higher levels don't compound *there*
  (bits-per-char is the wrong metric); and **product-of-experts is the right combiner**, not linear voting.
- **E** — at the *word* level, phrase+topic concepts **compound**: perplexity 476 → 247 (−48%).
- **The through-line:** *each concept level helps predict the level it operates on*, online, associatively,
  inspectably, combined by product-of-experts (consensus = intersection of views). The north-star architecture's
  building blocks **work and combine sensibly**.
- **I** — the whole thing **collapses to ONE repeated part**: a `Column`, replicated (width) and stacked (depth)
  into a `Cortex`. Wired bigger → better, attributably (width→calibration+generalization, depth→coherence), with
  the **combiner as the hinge** (geometric-mean pool for honest bpc + low-temp sampling for fluent generation).
  This is the "cortex small vs big" build: `lib/cortex.py` is now the single-component spine the rest extends.

**Honest calibration:** a real, working, end-to-end validation of the architecture — **not an AGI breakthrough.**
The individual mechanisms (branching-entropy segmentation, lexical/n-gram prediction, cache, product pooling)
are each known; the contribution is the **integrated, online, associative, inspectable, multi-level** system on
the Prism substrate, shown to compound at the right granularity. A genuine breakthrough would need a capability
a plain LM lacks — the live candidates: (1) the no-forgetting/online property under *real* task shift (needs
more than English), (2) compositional reasoning / analogy via the explicit concept+movement structure
(untested), (3) the inspectability/editability itself as the product. Those are the next swings.

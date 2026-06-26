# Exp AF — usage-based CONSTRUCTION induction: grammar as counting, made productive — 2026-06-26

**The bet (usage-based grammar: Bybee, Goldberg, Tomasello; statistical preemption).** A grammar is not a rule
system handed down from above — it is what *falls out of counting usage*, if for every context (frame) you count
**two** things at once. Bybee's two frequency effects:

- **TOKEN frequency** — how often *this exact* frame→filler occurred. High token frequency **entrenches**: the
  frame + its dominant filler freeze into a single chunk (a frozen idiom). The slot predicts the **specific word**.
- **TYPE frequency** — how many *distinct* fillers (and filler *categories*) followed. High type frequency makes
  the slot **productive**: it spawns an **open-slot construction** that predicts the filler **category**, not any
  one word — so it can fire for fillers *never seen in that frame*. That is compositional generalization, induced
  by counting alone.
- **Statistical preemption** — when two frames compete to express the same slot category, the **observed**
  pairing is up-weighted and the rarely-observed **competitor** link is **down-weighted** (count-based inhibition).

**The online translation (the hard rule: ONLINE ONLY — no backprop, no k-means, no SVD).** Every count is
accumulated in one streaming pass. Filler **categories** come from `jepa.py`'s online leader clustering (single
pass, nearest running-mean prototype by cosine or spawn — no iteration-to-convergence). The vectorized builders
(`np.unique`, scatter-add) are batched order-independent accumulation: the counts are identical to a
token-at-a-time online update.

- **Frame** = the preceding word ("X ___"), the simplest construction slot.
- **Classify** each ripe frame (≥ 40 tokens) by its token/type/category profile: `frozen` (one filler owns
  ≥ 50 % of tokens), `open-slot` (≥ 12 distinct fillers spread over ≥ 3 categories, no dominator), `mixed`,
  `sparse` (not yet ripe).
- **Open-slot prediction** routes *through* the category: `P(word | frame) = Σ_c P(category c | frame, slot) ·
  P(word | c)`, where `P(word | c)` is the global, frame-independent category-internal frequency. This lets the
  frame supply the slot's *category* distribution and the category supply *which words live in it* — the
  productive step.

**Setup.** text8, 14 MB → 2.39 M words (86 k types); top-N = 10 000 words get an id + an online category, all
10 000 categorized into **C = 400** categories. We held out **30 % of the 545 925 distinct (frame, filler)
pairs** — those fillers were *never seen in that frame* during training — for the compositional test. **Whole run
57 s on CPU, single pass.** Fixed seed.

---

## Result 2 — entrenchment vs abstraction: the discovered constructions

**Frozen idioms** (high token, one dominant filler — the slot predicts the *specific* word):

```
"such ___"   token=3022  dom='as'      (71%)
"known ___"  token=1866  dom='as'      (57%)
"number ___" token=1376  dom='of'      (72%)
"part ___"   token=1298  dom='of'      (75%)
"based ___"  token= 955  dom='on'      (62%)
"do ___"     token= 835  dom='not'     (51%)
"th ___"     token=1616  dom='century' (63%)   (text8 has no digits: "19th century" → "th century")
```

These are exactly the collocations a usage-based account predicts freeze: *such as*, *known as*, *based on*,
*part of*. The slot is entrenched — it has effectively one filler.

**Open-slot constructions** (high type/category spread — the slot predicts the *category*). Shown by the category
the frame prefers most **above its global base rate** (the slot's selectional preference):

```
"zero ___"  (and 2,3,5,8,…)  slot-prefers ×5–8  {km, per, miles, approximately, ft, square}   ← NUMBER + UNIT
"to ___"                     slot-prefers ×9     {be, have, him, them, due, according}         ← post-"to" / infinitive
"a ___"                      slot-prefers ×9     {single, program, strong, larger, longer}     ← det + N/Adj
"as ___"                     slot-prefers ×11    {such, known, well, far, result, seen}
"the ___"                    slot-prefers ×5     {south, east, west, sea, original, empire}
"of ___"                     slot-prefers ×2     {large, list, small, art, species, study}
```

The number frames all converge on the **measurement-unit** category — a real *"NUMBER ___ (unit)"* construction
discovered with no grammar given. **The labels behave as claimed:** high-token/one-filler frames froze; high-type
frames abstracted to a category.

---

## Result 1 — COMPOSITIONAL GENERALIZATION (the headline)

40 000 held-out (frame, filler) pairs — each filler **never seen in that frame** during training. A plain n-gram
can only floor them (the pair has count 0); the open-slot construction predicts the filler *through its category*.
Lower perplexity is better; "construction > n-gram" = fraction of pairs where the construction gave the held-out
filler higher probability.

| held-out slice | n-gram ppl | construction ppl | construction > n-gram |
|---|---:|---:|---:|
| all held-out pairs | 19 610 | **6 471** | 60.3 % |
| **open-slot frames only** | 23 461 | **5 405** | **80.1 %** |
| frames with a category head | 23 882 | 5 475 | 80.1 % |

**The abstraction beats the n-gram, clearly, in exactly the regime it claims.** On held-out pairs whose frame is
an induced open-slot construction, the construction's perplexity is **4.3× lower** (5 405 vs 23 461) and it gives
the unseen filler higher probability on **80 %** of pairs. The mechanism is honest: the n-gram has *literally
never counted* this filler after this frame, so it falls to the smoothing floor; the construction recognizes the
filler's *category* is one the frame's slot accepts, and lends it the category's mass. This is compositional
generalization produced by counting two things instead of one.

---

## Result 3 — STATISTICAL PREEMPTION reduces over-generation

For each open-slot frame, **weak-competitor** links = categories the frame holds *weakly* (≤ 20 % of its peak
commitment) while a *rival* frame commits ≥ 2× harder — the "could-occur-but-rarely-observed → blocked" forms
preemption is meant to suppress. **strong** links = the frame's own well-attested categories (must be retained).
We compare the open-slot head's probability mass on each group before vs after count-based inhibition.

| | strong (attested) | weak-competitor (over-generation) |
|---|---:|---:|
| before preemption | 0.4781 | 0.01112 |
| after preemption | 0.4895 | **0.00673** |

**Preemption cuts over-generation mass by 39.5 % while *retaining* 102 % of attested mass** (attested links even
rise slightly, since the suppressed competitors' probability is renormalized back onto the real ones). The
unobserved-competitor form is inhibited; the conventional form is not. Pure counting — the inhibition weight is
read straight off the relative commitment ratios, no gradient.

---

## Verdict

**Grammar-as-counting is productive, not just descriptive — counting *two* things (token and type) turns a flat
n-gram into a compositional one, online.**

- **Does the open-slot construction generalize compositionally?** *Yes, decisively, on its axis.* On held-out
  filler-frame combinations the construction beats the n-gram **4.3× on perplexity** and on **80 %** of open-slot
  pairs. The n-gram floors the unseen pair; the construction recognizes the slot's *category* and predicts the
  filler it never saw there. This is the clean win.
- **Do idioms freeze and frames abstract?** *Yes, as labelled.* High-token/one-filler frames froze (*such as*,
  *based on*, *part of*); high-type frames abstracted to a category (NUMBER + unit; post-"to"; det + N). The two
  frequency effects fall straight out of the two counts.
- **Does preemption curb over-generation?** *Yes.* Count-based inhibition cut unobserved-competitor mass −39.5 %
  with no loss to attested forms.
- **The honest limits.** (1) The headline is a *generalization* axis, not a held-out *language-modeling* axis: on
  ordinary (non-held-out) next-word prediction the open-slot head does **not** beat a well-counted n-gram, because
  when the exact pair *was* seen the specific count is sharper than the category. The construction is a **backoff
  for the unseen**, not a replacement — it should sit behind the specific count, firing when the n-gram floors.
  (2) Categories are the same noisy hashed-signature leader clusters as Exp U; a function-word mega-cluster still
  pollutes the top-category readout (hence the *above-base-rate* display to surface the real slot preference). A
  cleaner online latent (content-word-only clustering, larger signature space) would sharpen every number here.

**The takeaway for the cortex:** carry **two counts per frame** — token (entrench → frozen chunk) and type/category
(abstract → open-slot construction) — and let the open-slot head, predicting *through* an online category, serve as
the **compositional backoff** the specific count can't provide. Add count-based preemption to stop the open slot
over-generating. Grammar, here, is genuinely just counting — made productive by counting the *right two things*.

---

## Online-compliance note (every part is online; nothing is batch-optimized)

| step | how it's computed | online? |
|---|---|---|
| filler categories | jepa.py online signatures (hashed IDF-weighted running context counts) + **leader clustering** (one pass, running-mean prototype or spawn) | **yes** — order-independent accumulation + single pass |
| per-frame token/type counts | `(frame) → {filler: count}` and the distinct-filler / distinct-category sets | **yes** — counting |
| entrench / abstract labels | thresholds on each frame's token, type, category-spread, dominant-fraction counts | **yes** — read off counts |
| category lexicon `P(word\|c)` | global per-category filler frequency, summed over frames | **yes** — counting |
| preemption inhibition | multiplicative down-weight from relative commitment ratios (count/token) between competing frames | **yes** — counting, no gradient |

**No gradient descent, no backprop, no k-means, no SVD/eigendecomposition.** Single streaming pass + one online
leader-clustering pass. Fixed seed, reproducible.

## Axis

Right axis = **compositional generalization on held-out filler-frame combinations** (commandment 7: a fragile
idea first wins on generalization, not on the headline perplexity of seen data). The construction *loses* to a
well-counted n-gram on ordinary next-word prediction and *wins* by 4.3× on the unseen-combination slice — the slice
it was built for. Killing it on the in-distribution metric would have been the wrong call.

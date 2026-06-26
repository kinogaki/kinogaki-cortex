# Exp AL — Giving the workspace something to do: a multi-step probe over explicit concepts — 2026-06-26

**The bet (the AG cliffhanger, paid off).** Exp AG built a count-native **System 2** — a confidence+conflict
**gate** plus a capacity-4 **serial workspace** (inhibition-of-return, cognitive decoupling, a leaky-accumulator
race). Its verdict was split and honest: the **gate** wins the Engle signature cleanly, but the elaborate
**workspace** did **not** beat a trivial one-step *"defer to the wider context"*. AG's own diagnosis: a char
next-token probe is a **one-step** decision — there is nothing to *hold and manipulate* across cycles, so a single
deferral already reaches the answer and the workspace's serial machinery is dead weight. AG parked the workspace
(Fragile-Ideas §7/§8) and named its **untested winning axis**: *multi-step problems where one deferral cannot
reach the answer*. Exp AH then built the **explicit, slot-addressable concepts** (representational redescription)
the workspace would manipulate. This experiment gives the workspace the fair test AG promised it.

**The probe — 2-hop relational inference that REQUIRES the workspace.** From a count-derived relation R
(concept → its strongest **content** co-associate within ±4, built by PMI-weighted co-occurrence counting, **no
gradient**) we form chains that need R composed **twice**:

```
R(X) = Y ,  R(Y) = Z ,  Z != X ,  Z != Y            query: from X, what is R(R(X))?   answer: Z
```

The **trap is built in**: the prepotent, salient associate of X is **Y — the intermediate, not the target**. So a
one-hop operator lands on the *wrong* concept by construction. Sample chains (genuine content, not the
function-word floor):

```
supernatural -> beings -> human      carnegie -> andrew -> jackson     berber -> semitic -> anti
brothers -> warner -> bros           inherent -> vowel -> consonant    clause -> bsd -> license
```

**The three contestants.**
- **System-1** — the fast associative blurt: the salient associate of X = **Y**.
- **One-step deferral** — apply R once, R(X) = **Y**. *This is the operator that WON in AG* on the one-step probe.
- **Multi-step workspace** — the serial pass: **hold X**, apply R, **hold the result Y in a concept-slot**, apply
  R **again**, read the focus = R(R(X)) = **Z**. Capacity-k=4 focus, inhibition-of-return so attention advances to
  the freshly-composed slot, a confidence gate to decide whether to spend cycles, a step budget, and a
  suppress-not-erase floor on the one-step default so a dead chain still ships a graceful answer.

**Setup.** text8, 16 MB → 2.73 M words (92 k types); top-N = 10 000 words get an id + an online category
(`jepa.py` signatures + leader clustering, C = 400, single pass) — the **same concept layer as Exp AH**, promoted
to **5 452 explicit slot-objects** via `redescribe.py`. The content relation R covers **7 664/10 000** concepts.
The 2-hop probe yields **247 clean chains** X→Y→Z. Whole run ≈ 11 s on CPU, single pass, fixed seed 0.
Reuses `lib/system2.py` (the workspace + gate), `lib/redescribe.py` + `lib/constructions.py` + `lib/jepa.py`
(the explicit concepts); the new glue is `lib/deliberate.py` (the multi-step operator).

---

## The key test — accuracy on the 2-hop target Z = R(R(X))

| contestant | acc(Z) | lands on the trap Y |
|---|---:|---:|
| System-1 (prepotent associate) | **0.0000** | 1.0000 |
| one-step deferral  R(X)=Y *(AG's winner)* | **0.0000** | 1.0000 |
| **multi-step workspace  R(R(X))** | **1.0000** | 0.0000 |

**Δ(workspace − one-step) = +1.0000.  Δ(workspace − System-1) = +1.0000.**

**The workspace earns its keep — and the one-step deferral that beat it in AG now scores zero.** This is the exact
inversion AG predicted: on a genuinely multi-step probe, the operator that holds the intermediate and re-applies
the relation reaches the target, and the one-step operator structurally cannot — it is trapped on the intermediate
Y 100 % of the time. The gate fired on **100 %** of queries (every probe item has a real 2-hop structure to chase),
used a mean of **2.00 serial cycles** (exactly the hops required), and **composed the chain on all 247** (acc when
composed = 1.0000). Deepening to a **3-hop** probe (X→Y→Z→W, target W = R³(X), 46 chains) holds the result:
one-step **0.0000**, workspace **1.0000**, mean **3.00 cycles** — the workspace simply runs one more cycle while
the one-step operator falls further behind.

**Graceful fallback (the capacity bound).** With the step **budget set to 0**, the workspace reduces to the
one-step answer **exactly** (0 cycles) — suppress-not-erase made operational: no capacity to think ⇒ ship the fast
default, never an empty answer.

---

## Does the workspace *machinery* earn its keep — or is it just "apply R twice"? (the honest negative)

The target Z is *defined as* R(R(X)), so **any** double-application reaches it. The win above shows the workspace
**reaches a target a one-step operator cannot** — but it does not yet show the workspace's *machinery* (bounded
focus, inhibition-of-return, suppress-not-erase, the gate) beats a **naive blind double-application** of R.

- **On the clean relation:** blind "apply R twice" scores **1.0000** — it **ties** the full workspace.
- **Under noise** (corrupt a fraction of R's edges; a corrupted edge is also less confident):

| edge noise | blind 2× | workspace | workspace fired |
|---:|---:|---:|---:|
| 0.00 | 1.0000 | 1.0000 | 1.000 |
| 0.10 | 0.7935 | 0.7935 | 0.947 |
| 0.25 | 0.5223 | 0.5223 | 0.838 |
| 0.50 | 0.2146 | 0.2146 | 0.648 |

**The workspace ties blind double-apply at every noise level.** The gate correctly *declines* on corrupted,
low-confidence first hops (fire rate falls 1.00 → 0.65 as noise rises) — but its fallback, the one-step answer, is
**also wrong** for a 2-hop target, so declining buys nothing. **Honest negative on the machinery, again:** the
focus / inhibition-of-return / suppress-not-erase loop adds **no robustness** over naive composition on this probe.
The workspace's *only* real value is **reachability** — it is the **sole operator that can reach a ≥2-hop target at
all**, which is precisely the AG gap. The elaborate serial bookkeeping is, once more, **parked not killed**.

---

## Verdict

**The workspace finally earns its keep — on exactly the axis AG predicted, and for exactly the reason AG named.**

- **The headline (reachability): a decisive WIN.** On a multi-step probe the serial workspace beats both the
  one-step deferral and System-1 by **+1.00** — because holding the intermediate concept Y in a slot and
  re-applying the relation is the *only* way to reach R(R(X)), and a one-step operator is trapped on Y by
  construction. AG's one-step deferral, which captured *all* the win on the char probe, scores **zero** here. The
  cliffhanger is resolved: the workspace's reason to exist is **composition over held concepts**, and given a probe
  that requires it, it delivers (and the win widens at 3 hops).
- **The machinery: an HONEST NEGATIVE, twice over.** On the clean relation the full workspace **ties a naive
  "apply R twice"**, and under edge-noise it **still ties it** — the gate's decline-on-low-confidence does not buy
  graceful degradation, because the one-step fallback is also wrong for a 2-hop target. The value is *reachability*
  (can the operator reach the target at all), **not** the focus/IOR/suppress-not-erase bookkeeping wrapped around
  it. That bookkeeping remains parked (Fragile-Ideas §7/§8), now with a sharper diagnosis of *why*.
- **The diagnosis — is it the operator, the concepts, or the probe?** It is the **probe's noise model**, not the
  operator or the concepts. The concepts are real (content chains like *carnegie→andrew→jackson*, not the
  function-word floor); the operator composes correctly whenever the chain is resolvable. What the workspace
  machinery is *built* to add — selectivity among competing candidates, suppressing a wrong prepotent answer — only
  pays when the **fallback is right while the chain is wrong**. Here, in a pure-composition probe, a wrong first hop
  poisons *every* answer equally, so there is nothing for the focus/race to select between. The machinery's untested
  winning axis (still): a probe where multiple candidate chains **compete** and the workspace must *choose* — fact
  consistency with a distractor, or substitution among several slot-fillers — not a single deterministic chain.

**The takeaway for the cortex:** a count-native System 2 needs **two** separable parts, and they win on **different
axes**. The **gate** (Exp AG) decides *whether* to think and wins the Engle signature. The **serial workspace**
(this experiment) decides *what to compose* and wins **reachability on multi-step problems** — it is the only thing
that can hold an intermediate concept and apply a relation again. But the *elaborate* workspace bookkeeping
(capacity focus, inhibition-of-return, suppress-not-erase) is, on every probe tried so far, **no better than the
minimal version** of its operator (one deferral on a one-step probe; one blind double-apply on a two-step probe).
The lesson holds across AG and AL: **the metacognitive decisions (when to think, and that composition is needed)
are load-bearing; the serial micro-machinery is not — yet.**

---

## Online-compliance note (every part is online; nothing is batch-optimized)

| step | how it's computed | online? |
|---|---|---|
| concept categories | `jepa.py` online signatures (hashed IDF-weighted running context counts) + **leader clustering** (one pass) | **yes** |
| explicit concept-slots | `redescribe.py` promotion of ripe constructions (copy stable counts, no retrain) | **yes** |
| relation R | PMI-weighted co-occurrence counts (np.unique over packed keys = batched order-independent accumulation) | **yes** — counting |
| the multi-step workspace | a per-query serial loop over leaky accumulators: hold-slot → apply-relation → IOR → shift focus, bounded k=4, step budget | **yes** — no second pass, no gradient |
| the gate | confidence (relation's count-derived P(R(x)\|x)) + a defined-2nd-hop check | **yes** — read off counts |

**No gradient descent, no backprop, no k-means, no SVD.** One streaming pass for categories + relation counts; the
deliberate pass is a per-query serial loop over them. Bounded memory (k=4 focus, fixed step budget). Fixed seed 0,
reproducible.

## Axis

Right axis = **reachability on a multi-step problem** (can the operator reach a ≥2-hop target *at all*), **not**
one-step next-token accuracy — which is exactly the axis AG named as the workspace's untested home (Fragile-Ideas
§7: a fragile idea first wins on the dimension you weren't headlining). Judged on a one-step probe the workspace
loses to a trivial deferral (AG); judged on whether it can compose held concepts across cycles, it wins outright
and the deferral scores zero. Killing the workspace on AG's one-step result would have been the wrong call. The
machinery, separately, takes its **second** honest negative — parked, not killed, with its own untested axis
(competing candidate chains) now named.

## Lineage

Grew from **System 2 (Exp AG)** — the gate + the parked serial workspace, whose cliffhanger this pays off —
**redescription (Exp AH)** — the explicit, slot-addressable concepts the workspace manipulates — and **reasoning
(Exp AD)** — the count-derived relations the hops run over. The thread is **System 2 / reasoning**: AG asked *when
do we think, and does the workspace help?* and found the gate helps but the workspace had nothing to do; AL gives
the workspace a multi-step job and finds it **reaches** what one step cannot — while its internal machinery stays,
honestly, no better than the minimal composition. Credit Oberauer (the focus that *moves* across cycles), Cowan
(capacity ~4), Sloman (suppress-not-erase), Stanovich (cognitive decoupling), and AG for naming the axis before we
could test it.

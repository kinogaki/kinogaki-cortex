# Exp AT — the harness substrate — 2026-06-26

**Not a result — a foundation.** Every experiment A→AS was *offline*: read a fixed corpus, score it. To get
from **modelling** text to **producing** it, the cortex has to live inside a world its own output can change —
read, speak, hear a reply, learn from the reply. That one property — *the context you learn from was shaped by
what you said* — is what turns a predictor into a speaker, and it is how a child learns to talk: not by reading
more, but by saying something and getting an answer back. This experiment builds and checks the **shell** that
makes that loop possible. It deliberately does **not** test generation quality or comprehension (both are the
next phase) — only that the substrate composes and is honest.

**The four pieces (`lib/harness.py`).**
- **Codec** — the *only* text-aware piece. Maps a domain object to opaque int tokens and back. Text is one codec
  (`CharCodec`, [a-z ]); the loop above it is **encoding-agnostic** — it would run over bytes, words, or anything
  a codec can map. ("It's up to encoding.")
- **Environment** — a reactive world. `reset() → Turn`, `step(action) → Turn`. The action is the agent's
  utterance; *what comes back may depend on it.* Two rungs ship: **L0 `CorpusEnv`** (passive reading — the
  baseline floor: a reactive *interface* with non-reactive *behaviour*) and **L2 `InterlocutorEnv`** (the agent
  speaks, a pluggable `responder` replies, the reply is the next observation — a live model like Haiku slots into
  that hook unchanged).
- **Agent** — the cortex as a sensorimotor policy. `observe(ids)` counts **online** (each new token counted once,
  carried tail as left-context — the Column's own update driven incrementally); `act(k)` samples the calibrated
  geometric-mean vote to emit an utterance (a deliberately *minimal* speaker — maturing it is the next phase).
  Exposes `.K` / `.dist(str)` so the **existing** metrics suite scores it unchanged.
- **`run()`** — drives Agent through Environment, firing metric **Probes** into a **ProgressReport** on a
  schedule. The intrinsic probes (`bpc_probe`, `surprise_probe`) ship; CBT / LAMBADA / bAbI / BLiMP plug in later.

---

## (1) L0 — passive reading through the loop *learns online*

enwik9 slice, 1.69 M chars streamed in 2,048-char chunks, held-out 20 k tail scored every ~140 steps. The agent
is the Column band (orders 0–6), pooled by the calibrated `vote`. Held-out bpc as the world goes by:

| step | held-out bpc | obs-surprise |
|---:|---:|---:|
| 0 | 4.63 | 0.87 |
| 136 | 2.51 | 1.74 |
| 408 | 2.37 | 1.94 |
| 816 | **2.29** | 1.82 |

**bpc 4.63 → 2.29.** The loop reproduces ordinary online learning with zero new machinery — the substrate doesn't
distort the model, it just hosts it. (obs-surprise rises early because the held-out floor falls faster than the
streaming-chunk surprise settles; both are intrinsic signals, not the headline.)

## (2) L2 — the reactive contract holds

The warm agent now **speaks** (48 chars/turn), an offline `canned_teacher` **replies in a way that depends on the
utterance** (a 10yo register; a Haiku-backed responder replaces this hook with no other change), and the agent
**learns from the reply**. Surprise at the world's replies across 40 turns:

| turn | obs-surprise |
|---:|---:|
| 0 | 2.36 |
| 16 | 1.86 |
| 39 | **1.77** |

**2.36 → 1.77 — falling.** Action flows out, a dependent reply flows back, the agent observes it, and its surprise
at the interlocutor drops as it adapts to that interlocutor's (tiny) distribution. The active-inference signal —
*how well did I predict the world I'm living in* — is wired and moving in the right direction.

**The honest caveat the substrate makes visible.** The agent's utterances are **gibberish** (`'r sk and that
are creaturn hi with the univen wr'`). That is *expected and correct here*: generation maturity (the
deliberate-gate deciding when to think, stopping, choosing an altitude) was explicitly deferred. The substrate's
job is to make that weakness measurable and give it a place to be fixed — not to hide it.

---

## Verdict

**Substrate check: pass.** The four pieces compose; the loop runs over both rungs; the agent learns online (L0,
bpc 4.63 → 2.29) and the reactive contract is real (L2 — action out, dependent reply in, learned from, surprise
falling 2.36 → 1.77). It is encoding-agnostic by construction and scores with the existing metrics suite unchanged.
The two things it deliberately does *not* yet have — a **generation organ** worth the name, and a **comprehension
battery** (CBT / LAMBADA / bAbI / BLiMP) — are the next two phases, and both have a clean slot to plug into. This
is the shell from which "read, then speak, then learn from the reply" becomes a measurable research program.

Online throughout: Columns + calibrated vote, no gradients, no new model. `lib/harness.py`, `exp_at_harness/run.py`
(~14 s CPU, single pass, fixed seed).

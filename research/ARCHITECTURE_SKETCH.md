# kinogaki-cortex — architecture sketch (the boxes & the pipeline) — 2026-06-25

The concrete machine: components, the data that flows between them, and how text is ingested. The "why" is in
`KINOGAKI_CORTEX_ARCHITECTURE.md`; this is the "what does it actually look like." A sketch to iterate on.

## The one data type that flows everywhere: `State`

Everything emits and consumes the same message (Monty's hardest-won lesson — "never pass models, only
states-of-the-world"):

```
State {
  loc:        vector      # position in the current reference frame (level 0: position-in-window; higher: a point in the learned concept space)
  features:   SDR         # sparse code (e.g. 2048 bits, ~2% active) — WHAT is here
  conf:       float       # this sender's confidence
  sender:     {id, level, view}   # which column/level/vantage produced it
  t:          int         # stream step
}
```

A consumer can't tell if a `State` came from the raw stream or from another column. That uniformity is what
buys stacking, parallelism, and cross-view voting.

## The components (boxes)

```
                ┌─────────────────────────────  CONCEPT STORE  ─────────────────────────────┐
                │   a .prism Document (via `pip install kinogaki`)                            │
                │   each concept = an Element (frame): typed slots + default priors +         │
                │   activation(recency×freq decay) + edges: evidence↑  predict↓  vote↔  move→ │
                │   learning = promote a settled coalition → a new Element; persist deltas.   │
                └───────▲────────────────────────▲───────────────────────────▲───────────────┘
          promote/bind  │            top-down predictions │        movement operators │
                        │                                  │                            │
   LEVEL 2  themes      │   [ ColumnField ]──vote──▶[ VotingBus ]──consensus──▶[ BoundaryDetector ]
   (slow, few)          │        ▲                                                   │ FLUSH summary ↑
                        │        │ States in                                          │
   LEVEL 1  words/      │   [ ColumnField ]──vote──▶[ VotingBus ]──consensus──▶[ BoundaryDetector ]
   phrases              │        ▲      ▲                              ▲              │ FLUSH summary ↑
                        │        │      │ Gate (default-deny)          │ broadcast    │
   LEVEL 0  chars       │   [ ColumnField : 1000s of Columns ]◀── consensus (context) ┘
                        │        ▲
                        │        │ State(loc+SDR)
   raw UTF-8 ──▶ [ Cursor/Reader ] ──▶ [ Encoder ] ──▶ level-0 input
                  (the "motor")          (char→SDR)
                        ▲
                        └──── next-action (step / jump-to-boundary / scan-back) chosen by curiosity + goal
```

Plus two cross-cutting controllers that don't sit in the stream:
- **Gate / Attention** — broadcasts a scope and *default-denies* who may vote each step (biased-competition +
  divisive normalization to combine; an ignition threshold commits a winning coalition and broadcasts it back).
- **Neuromodulator / LearnController** — sets each column's learning-rate + precision from surprise magnitude
  and a regime-shift detector. The anti-forgetting lever: learn fast only on big, real surprise; on a regime
  shift **allocate new capacity instead of overwriting**.

### Component responsibilities

| Component | Holds | Does |
|---|---|---|
| **Cursor / Reader** (the *motor*) | a position in the text | emits the symbol at the cursor; **chooses the next move** — step +1, jump to a learned boundary, or scan back. Navigation *is* the action space. |
| **Encoder** | a char→SDR map | turns the raw character (+ local position) into the level-0 `State`. **Tokenizer-free** — no pre-segmentation; words/phrases are *discovered*, not given. |
| **Column** (the *unit*) | a **private** SDR vocabulary + a small local sequence predictor; a *view* (level, direction, feature-projection) | predict the next `State`; compare to actual → **own surprise**; emit a **vote** (best-guess State + conf); update **locally** (Hebbian / ACT-R activation). Thousands per level. |
| **ColumnField** | all columns at a level | fan State in, collect votes, host diversity (timescale/direction/feature views). |
| **VotingBus** | — | **associative consensus over private codes** (no shared dictionary; ~20–30 active cells settle the population). Outputs a consensus `State` **and the disagreement** (an uncertainty/boundary signal). |
| **BoundaryDetector** (per level) | a running belief-distribution + background | fire on **transient Bayesian surprise** (KL belief-shift vs background) and/or **evidence-slope** drop and/or **disagreement**. On fire → **FLUSH**: summarize the finished segment into one `State`, send it **up** a level. |
| **ConceptStore** | the `.prism` Document | bind a settled State to an existing concept (by SDR overlap) or **mint a new Element (frame)**; maintain activation, slots, and edges; **persist after each observation** (online, never a full rewrite). |
| **MovementOperators** | learned transition functions keyed to verbs/relations | `(operator, {agent, patient, source, goal, force,…})` applied to the current state in concept-space — the "thinking is movement" step. Shares the frame/Element grammar. |

## The read/learn loop (one input step)

1. **Sense.** Cursor emits the symbol at its position (the cursor may have *jumped* — that was the previous
   step's action). Encoder → level-0 `State` (loc = position-in-window, features = char SDR).
2. **Predict + vote.** Each eligible Column predicts the next `State`, compares to the actual, computes its
   surprise, and emits a vote. (Top-down consensus from the previous step is available as context.)
3. **Gate + settle.** The Gate (attention/default-deny) selects who votes; the VotingBus produces a **consensus
   State** + **disagreement**.
4. **Segment.** BoundaryDetector combines transient-surprise + disagreement. If below threshold → continue
   within the segment (COPY/UPDATE). If a boundary fires → **FLUSH**: summarize the segment, emit it as one
   `State` to the level above (which repeats steps 2–4 at its slower timescale).
5. **Learn (online).** Neuromodulator sets learning rates from surprise×regime-shift. Columns update locally.
   ConceptStore binds the settled State to an existing frame (high SDR overlap) or **mints a new Element**;
   activation/edges updated; **deltas persisted to the `.prism` doc**.
6. **Move + broadcast.** If the consensus names a relation/verb, apply its MovementOperator to the current
   concept-space state. Broadcast the consensus back down as top-down context.
7. **Act.** The Cursor chooses the next move — usually step +1, but **curiosity** (go where prediction is most
   uncertain) or a **goal-state** can make it jump/scan. This closes the agency loop (the minimal "environment").

## Two modes, one machine

- **Read** (above): predict → vote → segment → learn → move.
- **Generate**: run the **top-down predictions** forward and *sample* — "move" through concept-space via
  operators and emit the symbols each predicted State decodes to. Generation is the read loop run in reverse-ish
  (predict-then-emit instead of predict-then-compare). This is why generation *emerges* from a good predictor.

## How it lands on Core (kinogaki / Prism)

- **Elements/paths** = frames/concepts (and the level tree). **Connections** = evidence↑ / predict↓ (driver) +
  vote↔ + move→, with gain/precision as modulator edges. **Time-samples** = each concept's activation/belief
  *trajectory* over stream steps. **Evaluator** = the gated consensus resolve. **The Document** = the durable,
  diff-able, inspectable mind. The **numeric learners live outside Core** and mutate the Document through one
  surface (the Atlas pattern) — Core stores/serves structure, it doesn't train.

## The minimal first slice (Experiment A — what we actually build first)

Strip it to the smallest thing that can be killed:
- **Cursor** (step +1 only — no jumps yet) + **Encoder** (char → simple sparse code).
- **One** level-0 predictor (not the full field yet) producing a next-char distribution.
- **BoundaryDetector** computing **transient Bayesian surprise** (normalized KL belief-shift).
- Output: segment boundaries written as Elements into a `.prism` doc.
- **Test:** do those boundaries recover word/phrase boundaries on space-stripped text, beating surprisal /
  entropy / BPE / a learned HM-RNN detector? If yes → add the field + voting (the rest of the boxes). If no →
  the core "ambiguity = boundary" bet is weak; stop before building the field.

Everything above the first slice is gated on this surviving — wide before deep, cheap before heavy.

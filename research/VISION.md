# kinogaki-cortex — the vision

> The north-star story. The architecture spec (`KINOGAKI_CORTEX_ARCHITECTURE.md`) and the component sketch
> (`ARCHITECTURE_SKETCH.md`) are how; this is *why*.

Today's language models are one enormous frozen function that has memorized the statistics of text. We're
building the opposite: a living, growing **society of mind** that reads the way a person does — continuously,
curiously, and never forgetting — and writes down what it learns as things you can actually look at.

Here's the idea as a story of what happens when it reads a sentence.

A character arrives. Not one model but **thousands** of tiny, sparse predictors each take a guess at what comes
next — each from its own vantage point (one watches letters, one watches words, one reads backward, one tracks
syntax). They **vote**, and the consensus is the system's current belief about "what is being said right now."
Most of the time the stream is unsurprising and they just flow along. But every so often the predictions
*shift* — not merely "a rare word appeared," but the whole field's confidence lurches, the way your
understanding clicks at the end of a phrase. That lurch is the chisel. Right there, the system carves a
boundary and **mints a concept** — and writes it down as a durable, named object with slots, in a `.prism`
document. A thing it can keep, reuse, inspect, and even let you edit.

Two moves make it more than chunking. First, **movement is thought**: the system doesn't passively stare at
text, it *moves through* a conceptual space it's slowly learning, and the operators of that movement are the
verbs and relations. "The cat *sat on* the mat" isn't three nouns and a verb — it's a *transformation* that
moves a state from one place in concept-space to another. Reasoning becomes a *path* through that space.
Second, **concepts stack themselves**: when a coalition of voters reliably settles on the same thing, that
settled agreement *becomes* a new, higher concept — so hierarchy isn't designed in, it grows from the bottom,
the way meaning actually accretes.

And the whole machine has four properties no transformer has at once: it learns **online** (every sentence
teaches it, no retraining), it **never forgets** (sparse codes don't overwrite each other — new knowledge is
*added*, not smeared over old), it's **cheap** (local updates, no global backprop), and it's **transparent**
(its entire mind is a readable document, not a black box).

**So how is this a road to AGI?** The bet isn't "we'll out-predict GPT" — we won't, at first. The bet is that
what stands between today's models and general intelligence isn't *more knowledge*; it's *continual,
compositional, grounded, inspectable learning* — the things scaling a frozen function can't give you. The
brain reaches general intelligence with one algorithm, massively parallel, learning every second of its life
without ever hitting "retrain." We copy *that* architecture — many models voting, prediction as the only
teacher, movement as thought, concepts as durable structure — rather than copying the *output* of intelligence
(fluent text). If meaning is something a mind grows by moving through the world and chunking what repeats, then
a system that grows meaning the same way is a genuinely different, and possibly truer, road than an
ever-bigger lookup table.

**And the part that keeps us honest:** this is a hypothesis, not a victory lap. Nobody — including Numenta —
has made this work for language; HTM-for-text hit a wall. What's new is the *specific fusion*: surprise-carved
concepts + a learned movement-space + thousands of voters + a persistent inspectable model. We've designed the
cheapest possible experiment to tell us, in days, whether the core bet even holds — before we pave the road.

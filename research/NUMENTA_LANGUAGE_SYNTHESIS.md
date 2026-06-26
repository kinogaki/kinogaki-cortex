# Does Numenta/TBT offer a language-rooted brain substrate? — synthesis (2026-06-25)

Mined from Numenta Discourse forums, Hawkins/Numenta talks & interviews, and the primary papers + book.
Source corpus saved under `research/sources/{forums,talks,papers}/`. This answers the user's bet directly:
*TBT is based on real cortical principles, so there must be a real brain-like substrate rooted in language.*

## The honest answer: the bet is RIGHT about the shape, and nobody — including Numenta — has filled in the content

**What is genuinely brain-rooted and real (the steel-man).** TBT rests on Mountcastle's "one cortical
algorithm everywhere" premise, and Hawkins makes one move, applied uniformly: keep the cortical-column
machinery, swap the inputs. The architecture-level claim is concrete and repeated:

- *"All knowledge is learned and stored in the context of locations and location spaces … and 'thinking' is
  movement through those location spaces."* (Frontiers 2019, grid-cell framework)
- The primitives: **concept = location**, **operation = movement** (path integration shifts a grid-cell
  bump), **composition = displacement cells**, **agreement = voting across columns**. In the Lex Fridman
  interview Hawkins makes it explicit for math: a proof is *"literally trying to discover a path from one
  location to another location in a space of mathematics."*
- **Crucially — and this is the part that most helps us — Numenta's current (2024) thinking says the space
  need NOT be 3-D Euclidean; its structure can be LEARNED:** *"the exact structure of space can potentially
  be learned, such that the lower-dimensional space of a melody, or the abstract space of a family tree, can
  be represented."* (TBP, arXiv 2412.18354). Their implementation embeds graphs of ≤3-D structure (strings,
  edge-graphs, point clouds).
- One thin but real empirical thread that a navigation-style code exists for a *non-spatial* continuum:
  Constantinescu/Behrens 2016 found a hexagonal (grid-cell-like) fMRI code over a learned **2-D conceptual**
  space (bird shapes). Suggestive, population-level, low-D — but real.
- **TBP's own operational definition of language IS the "motor = moving through text" instinct** — this is
  not our invention. The project FAQ states: *"reading and producing language can be framed as a sensorimotor
  task where the sensor moves through the sentence space,"* grounded in models learned through other senses.
  So the user's framing is *aligned with Numenta's own (still-aspirational) plan* — which both validates the
  instinct and inherits the same unfilled content gap (what is "sentence space," what generates the movement).
  As of the most recent material ("Meet Monty 2026" Q&A, Dec 2025), language remains explicitly **"next, not
  now"** — gated behind shipping the concrete sensorimotor core first.

**What is missing everywhere — the load-bearing content (why it's "aspiration" today).** Across forums,
talks, papers, and Numenta's own code, three gaps are never filled, and the participants say so themselves:

1. **No axes for meaning.** The theory never says what the *dimensions* of a conceptual/linguistic space are.
   "Discover the dimensionality" is stated as the problem, with no learning rule or worked non-spatial example.
2. **No movement generator for language.** *"A verb is a movement"* is a label, not a mechanism — nothing
   says how a word produces a location-update vector. A forum regular's sharp objection: an abstract concept
   *"is basically a set,"* and a set has no inherent metric, while grids encode a periodic metric over a
   continuous space — never resolved.
3. **Numenta itself deferred it.** Monty (the 2024 reference implementation) uses **3-D Cartesian pose
   graphs**, lists *"modeling language"* and abstract concepts as **future work**, and admits: *"How Monty
   would learn to generalize a mapping between these levels of representations remains outstanding … We are
   still figuring out exactly how this would work in a simpler case like the family-tree."* If a family tree
   is unsolved, recursive language is far out. Hawkins concedes the abstract-concept extension is *"highly
   likely"* and *"we haven't really proven"* it.

**The one thing that shipped for text — and what it was.** Cortical.io Semantic Folding built word-SDRs as
sparse 2-D topographic "fingerprints" (overlap = similarity) on HTM. Endorsed by Numenta ("it's topological"),
but it is a **static similarity embedding (SOM/word2vec-flavored), not reference frames, not grid cells, not
composition or grammar.** It never became competitive language tech. The grammar/meaning was supposed to fall
to HTM sequence memory downstream, which never matured for language — and HTM has *no hierarchy*, which the
NuPIC NLP post-mortem names as exactly why it *"will not be able to formulate a deep understanding of text."*

## What this means for kinogaki-cortex (refines, doesn't kill, the earlier verdict)

This *upgrades* the earlier "motor = navigating text is a category error" point into something more useful:

- The **naive** version (motor = literal cursor movement through 1-D characters) is still too literal — it's
  not the load-bearing thing.
- But TBT does hand us the **shape of a defensible "reference frame for text"**, and it's an *unfilled* shape,
  which is precisely the open research contribution: **a LEARNED low-dimensional conceptual coordinate space +
  a transition/"movement" operator + feature-at-location binding + sequence memory for serial order + voting
  across views.** Numenta gives the architecture-level bet and one empirical thread; *the content — what the
  coordinates of meaning are, how a word generates a movement — is original work nobody has done.*
- This dovetails with the project's real value prop (online, non-forgetting, inspectable): the SDR / local-
  Hebbian / grid-cell substrate is exactly the part that gives continual learning without catastrophic
  forgetting. So the sharpest framing of the bet is: **can we build the learned conceptual-reference-frame
  substrate for text that Numenta described but never built, inheriting HTM's online/non-forgetting
  properties, and represent it as an inspectable `.prism` document?** That is genuinely novel and genuinely hard.

**Verdict update:** "there's something there" is now *better supported* — but the something is a research
frontier with a clear target shape and zero off-the-shelf content, not a mechanism you inherit. Proceed
experiment-first (Experiments A/B in `KINOGAKI_CORTEX_RESEARCH.md`), and treat "learn a conceptual coordinate
space + transition operator for text" as the distinctive, high-risk/high-reward core — not the
prediction-error chunking (which is solved) and not literal cursor-motor (which is decorative).

## Saved source corpus (index)

**Forums** (`research/sources/forums/`): `egocentric-reference-frame-abstract-space.md`,
`grid-cells-and-abstract-ideas.md`, `language-capabilities-according-to-htm.md`,
`cortical-io-semantic-folding.md`, `nupic-nlp-attempts.md`, `tbp-faq-abstract-concepts.md`,
`tbp-arxiv-abstract-concepts.md`.

**Talks/interviews** (`research/sources/talks/`): `hawkins-frontiers-grid-cells-cortical-function.md`,
`hawkins-companion-paper-grid-cells-framework.md`, `hawkins-lex-fridman-thousand-brains.md` (real transcript),
`tbp-arxiv-2412-paradigm-abstract-concepts.md`, `thousand-brains-project-faq-language-status.md`,
`hawkins-2023-qa-chatgpt-language.md`, `byrnes-hawkins-talk-notes-abstract-cognition.md` (secondary).

**Papers** (`research/sources/papers/`): `grid-cells-framework-2019.md`,
`sequence-memory-thousands-of-synapses-2016.md`, `locations-in-the-neocortex-2019.md`,
`a-thousand-brains-2021.md`, `conceptual-gridlike-code-constantinescu-2016.md`,
`thousand-brains-project-2024.md`.

**Known gaps:** YouTube auto-transcripts for the NAISys 2020 talk and the official "Thousand Brains Theory"
video could not be fetched (extraction sites returned 403); the papers/interviews cover the same claims with
citable text. *A Thousand Brains* book quotes are via the Frontiers/companion papers (same claims, citable).

# Exp M — discover phrases and topics by surprise — 2026-06-25 (autonomous)

Reuse Exp A's branching-entropy boundary signal one level up. Corpus: enwik9 first 60 MB → 8.66 M words,
210 k vocab, **7,514 real article (`<page>`) boundaries** as topic ground truth (avg 1152 words/article).

## Phrases — branching entropy over the WORD stream (qualitative)

Cutting the word stream where forward+backward next-word entropy rises recovers real multi-word units, unsupervised:

> united states · such as · see also · list of · according to · th century · to be · can be · part of ·
> there are · there is · number of · due to · under the · has been

**Worked:** genuine collocations and fixed phrases emerge from pure prediction-uncertainty, no phrase list given.
**Caveat (honest):** the crude enwik9 normalizer leaves XML/wiki markup, so markup n-grams ("lt td gt",
"http www", "amp ndash") also rank high — a *corpus-cleaning* artifact, not a method failure. A real markup
stripper would clean these out.

## Topics — predictive surprise vs article boundaries (quantitative)

TextTiling-style surprise (1 − cosine between adjacent 120-content-word windows), peaks = topic boundaries,
scored vs the `<page>` truth (tol ±30 words):

| method | precision | recall | F1 |
|---|---:|---:|---:|
| surprise (TextTiling) | 0.107 | 0.239 | **0.148** |
| random baseline | 0.048 | 0.105 | 0.066 |

**2.2× over random** — the surprise signal carries **real** topic-boundary information. But absolute F1 is low:
it over-segments (15.5 k detected vs 7.5 k true → precision suffers), and TF-cosine on noisy, markup-laden,
short-article Wikipedia is a weak content representation.

## Honest read + improvement path

- **Phrases: a clean qualitative win** — the Exp A mechanism transfers up a level. Discovered phrase units can
  become a phrase-column's tokens (genuine stacking, un-cheating the fixed-bigram phrase level).
- **Topics: real but immature (2.2× random, F1 0.15).** The bet (surprise marks topic shifts) is *confirmed
  above chance*, not yet strong. Clear levers, untried tonight: (1) strip markup for a clean stream; (2) match
  the detection rate to the true rate (tune threshold — kills the 2× over-segmentation); (3) a better content
  signature than raw TF-cosine (e.g. the cortex's own next-content-word predictive distribution = true Bayesian
  surprise, not bag-of-words). (3) is the principled one and ties topics back into the predictive substrate.
- **The payoff once topics are solid:** a topic boundary resets the attention/topic cache → attention scoped to
  the current segment, the most promising route to the global-coherence frontier.

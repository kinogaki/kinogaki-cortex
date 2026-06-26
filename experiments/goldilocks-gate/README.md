# Exp BI — the Goldilocks learning-rate gate (an inverted-U on surprisal)

A child does not learn equally from every word it hears: learning peaks in a *middle* band of
predictability (Kidd's Goldilocks-attention curve; the N400/cloze surprisal relationship). The
already-known carries no news; the unparsable does not connect to anything you have. The naive
correction — "learn MORE the more surprised you are" (a monotone surprise-as-gate) — is wrong at the
high end, where a typo or OOV burst is both maximally surprising and worthless. This experiment makes
an online count model's **write-weight** an **inverted-U on its own surprisal** (predict-then-write),
so the bounded write budget is spent on the middle band that actually teaches. The decisive test is at
**equal table size** (LFU-capped): the gate must lower held-out bpc *without* simply storing more
distinct contexts — otherwise it is a memory win, not a learning-rate win. We run the FRAGILE budget
(11 gate shapes: flat, three monotone, seven goldilocks) across three caps, check the high-surprisal
rare-context slice the gate deliberately skips, and fold in the **N400/cloze read-out** (kept distinct
from the write-gate): model surprisal validated as a cloze/N400 proxy on the same frozen counts. Online
single pass, no backprop, bounded memory, text8 slice, seed 0. See `RESULTS.md`.

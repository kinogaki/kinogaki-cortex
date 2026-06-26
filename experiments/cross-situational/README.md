# Cross-situational word→referent learning

**Experiment AV · 2026-06-26 · acquisition (the first grounding of meaning)**

The spine learns word←word; this learns word ← *referent in the world* — the binding layer the cortex never
had. A child hears "ball" while a ball, a dog and a cup are all in view and can't tell which word names which
thing; across many ambiguous scenes the true mapping is the one regularity that survives (Yu & Smith 2007). We
build a scene-bearing env (the referent ids arrive via the harness `Turn.signal`, **not** the token stream)
and ship **both** mechanisms the literature is split between (`lib/crosssit.py`): **(A) DenseAssoc**, a
word×referent PMI-like co-occurrence matrix, and **(B) ProposeVerify**, Trueswell's one-slot
propose-but-verify. Both recover the full mapping from co-occurrence alone, online and bounded (above-chance in
12/12 variations). B reproduces the **Trueswell at-chance-after-disconfirm signature** cleanly
(0.93→0.27 after a re-propose) where A's distribution holds (0.67) — a real behavioural dissociation; and under
a systematic confound the dense matrix out-votes the lure (0.89–1.00) while the one-slot guesser only partly
recovers (0.39–0.94) — the equal-memory tradeoff. **Verdict: win** (mechanism + dissociation; headline at
ceiling because the toy lexicon is clean). Neither kill-condition fired — keep both variants.

**Run it** (no data files; synthetic scenes, ~30s):

```sh
python exp_av_crosssit/run.py
```

Online single pass, bounded memory, no gradients / k-means / SVD / backprop. New files only:
`lib/crosssit.py`, `exp_av_crosssit/run.py`, `RESULTS.md`, this file.

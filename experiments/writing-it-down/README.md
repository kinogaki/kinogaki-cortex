# Writing it down

**Experiment AQ · 2026-06-26 · negative, with a principle**

At **equal** memory budget a bounded-internal + external store does **not** beat one
bigger internal table (evidence fragmentation: the confident-internal path answers from
the internal fragment alone). It wins only in the cost-asymmetric regime — cheap/big
external → **−0.23 bpc**. Externalizing is a **cost arbitrage** (the page is cheaper
than the skull), not a better use of the same bytes.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/writing-it-down/run.py
```

**Blog post:** ["Writing it down"](https://cortex.kinogaki.com/blog/writing-it-down/)

Credit: Ericsson & Kintsch (long-term working memory).

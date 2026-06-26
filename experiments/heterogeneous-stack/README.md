# The heterogeneous specialized stack

**Experiment X · 2026-06-26 · negative result**

The opposite-axis test to Exp I: instead of one uniform Column repeated, give each LEVEL a different column type, connection range, and timescale, with a thalamus/basal-ganglia gate arbitrating which level speaks. Headline: specialization-by-level lost on char-bpc (each piece won only on the axis it was built for), but one clean architectural win emerged — dynamic confidence routing beats static pooling by ~0.9 bpc. When levels are unequal, the arbitration mechanism is load-bearing.

**Run it** (from the repo root, after `bash data/get-data.sh`):

```sh
python experiments/heterogeneous-stack/run.py
```

**Blog post:** ["One brain part, or many? We gave each level its own job"](https://cortex.kinogaki.com/heterogeneous-stack/)

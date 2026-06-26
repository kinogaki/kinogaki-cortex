# Exp AU — chunk lexicon with sub-unit interference (the PARSER/Isbilen organ)

Build a `ChunkLexicon` that greedily **covers** a stream with the highest/longest-confident
variable-length chunks (take-the-best), **mints** the concatenation of two adjacent chunks when they
recur (leader-spawn), and — the new part — **leaks weight from a minted whole's sub-chunks as the
whole commits** (sub-unit interference, the Isbilen splice effect by counting). Bounded by LFU
eviction. The chunks become a segmenter (cover points = boundaries) and an emission/completion
vocabulary. Judged on three axes: the **splice test** (does the within-word B–C transition decay
below pure-forward-TP, the Saffran null?), **boundary F1** on space-stripped text8 (vs Exp A's
0.775), and **held-out bpc** vs a fixed-order n-gram. Online single pass; no backprop/k-means/SVD;
bounded. The mechanism lives in `lib/chunklex.py`; run with the Exp A venv.

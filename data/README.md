# data/

The corpora the experiments read. **Nothing here is committed** — the datasets
are large and public. Fetch them with the script:

```sh
bash data/get-data.sh
```

That downloads and unzips into this directory:

| file | size | what | used by |
|---|---|---|---|
| `text8` | 100 MB | clean lowercase Wikipedia (`a-z` + space) | most char/word experiments |
| `enwik9` | 1 GB | raw Wikipedia with markup and real `<page>` boundaries | gigabyte, boundaries, ignition |

The experiments read these by name through `lib/corpus.py`, which resolves the
path as `<repo>/data/<name>`. Most runs take only a slice (`nbytes=...`), so you
can start an experiment as soon as the relevant file finishes downloading.

A few of the earliest experiments also read smaller single-author corpora
(`english.txt`, `shakespeare.txt`, `darwin.txt`, `bible.txt` — plain-English
Project Gutenberg books). Drop any such texts here under those names to run
`associative-vs-gradient`, `concepts`, `voting`, and `word-level-compounding`.

`boundaries-from-chars` (Exp A) reads its own `experiments/boundaries-from-chars/data/raw.txt` —
put any plain-English book there.

Sources: text8 and enwik9 are from Matt Mahoney's
[Large Text Compression Benchmark](https://mattmahoney.net/dc/textdata.html).

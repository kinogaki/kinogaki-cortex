# Setup

Everything here is a single streaming pass of counters in plain Python. There is
no build step and no training framework.

## Requirements

- **Python 3.x** (3.9+).
- **numpy** — required.
- **mlx** — optional. The GPU experiments (the Metal path in `gigabyte-and-gpu-o`)
  use Apple's MLX for Apple-Silicon acceleration. Without it, the experiments
  fall back to numpy.

```sh
python -m venv .venv && source .venv/bin/activate
pip install numpy
pip install mlx        # optional, Apple Silicon only
```

## Data

The datasets are not committed. Fetch them once:

```sh
bash data/get-data.sh
```

See `data/README.md` for what gets downloaded and which experiments need what.

## Running an experiment

From the repo root:

```sh
python experiments/<slug>/run.py
```

Each experiment folder has a `README.md` (what it tests, the headline result, the
exact command, and a link to its blog post) and, where one exists, the original
`RESULTS.md`. The shared modules live in `lib/`; experiments add it to
`sys.path` automatically.

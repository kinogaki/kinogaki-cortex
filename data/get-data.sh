#!/usr/bin/env bash
# Fetch the text corpora the experiments read. Datasets are NOT committed.
# Run from anywhere; files land next to this script (the repo's data/ dir).
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DATA_DIR"

fetch() {  # url  outfile
  local url="$1" out="$2"
  if [ -f "$out" ]; then echo "have $out"; return; fi
  echo "fetching $out from $url"
  if command -v curl >/dev/null 2>&1; then curl -L -o "$out" "$url"
  else wget -O "$out" "$url"; fi
}

# text8 — 100 MB of clean lowercase Wikipedia (a-z + space). Most char/word experiments.
fetch "http://mattmahoney.net/dc/text8.zip" "text8.zip"
[ -f text8 ] || unzip -o text8.zip

# enwik9 — 1 GB raw Wikipedia (with markup, real <page> boundaries). Gigabyte/topic experiments.
fetch "https://mattmahoney.net/dc/enwik9.zip" "enwik9.zip"
[ -f enwik9 ] || unzip -o enwik9.zip

echo
echo "done. data/ now holds: text8 (100 MB) and enwik9 (1 GB)."
echo "note: experiment 'boundaries-from-chars' (Exp A) reads its own data/raw.txt —"
echo "      drop any plain-English Project Gutenberg book there to run it."

"""noise.py — a PERCEPTION-TIME noise harness for the concept stack (Exp Y).

The thesis under test: corrupt the SURFACE so a flat letter model can't lean on exact n-grams, and the system
is FORCED up the stack — when the letters lie, lean on the idea. Two levels of corruption:

  * CHAR scramble (surface) — with prob p, damage each char (swap-adjacent / random-substitute / small-window
    shuffle). Defeats exact char n-grams; the word/concept level (which only needs the word roughly recoverable,
    or the surrounding words) should hold up better.
  * WORD unreliability (the second level) — with prob q, damage whole words (substitute a random in-vocab word /
    drop / swap-adjacent words). Defeats the word level; the phrase/topic level above words should carry on.

HARD ONLINE RULE: noise here is pure array surgery (vectorized, seeded). It is applied to the INPUT STREAM the
model reads (context). The PREDICTION TARGET is always the CLEAN next char/word — we are measuring "given a
corrupted view of the past, how well do you predict the true future?". No optimization, no learning here.

Alphabet (corpus.py): a..z = 0..25, space = 26, V = 27. Words = runs between spaces.
All functions take a numpy int array `ids` and an `np.random.Generator` (seed it for reproducibility).
"""
import numpy as np

V = 27
SPACE = 26


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  CHAR scramble — surface corruption (level 0). Defeats exact letter n-grams.
# ════════════════════════════════════════════════════════════════════════════════════════════════

def char_scramble(ids, p, rng, swap=0.34, sub=0.33, shuffle=0.33, win=3, protect_space=True):
    """With probability p, corrupt each (non-space) char. Three damage modes, chosen per hit:
      - swap   : swap this char with its right neighbour (transposition — the classic typo).
      - sub    : replace with a uniformly random letter (a..z).
      - shuffle: shuffle this char into a small window of `win` chars around it (local scramble).
    Spaces are protected by default so WORD BOUNDARIES survive (the higher levels still see the token grid;
    the point is that the *letters inside* lie, not that segmentation is destroyed). Returns a fresh array.

    Vectorized: pick the hit positions once, assign each a mode, apply each mode as a batch."""
    out = ids.copy()
    n = len(out)
    if p <= 0:
        return out
    hit = rng.random(n) < p
    if protect_space:
        hit &= (ids != SPACE)
    idx = np.nonzero(hit)[0]
    if idx.size == 0:
        return out
    mode = rng.choice(3, size=idx.size, p=np.array([swap, sub, shuffle]) / (swap + sub + shuffle))

    # --- substitute: random letter (0..25), avoid space so we don't fake a boundary ---
    sub_i = idx[mode == 1]
    if sub_i.size:
        out[sub_i] = rng.integers(0, 26, size=sub_i.size).astype(out.dtype)

    # --- swap with right neighbour (skip if neighbour is a space or out of range) ---
    sw_i = idx[mode == 0]
    sw_i = sw_i[(sw_i + 1 < n)]
    if protect_space and sw_i.size:
        sw_i = sw_i[ids[sw_i + 1] != SPACE]
    if sw_i.size:
        tmp = out[sw_i].copy()
        out[sw_i] = out[sw_i + 1]
        out[sw_i + 1] = tmp

    # --- small-window shuffle: permute chars in [i, i+win) among themselves (letters only) ---
    sh_i = idx[mode == 2]
    for i in sh_i:                                   # few positions; windows are tiny
        e = min(i + win, n)
        seg = out[i:e]
        keep = seg != SPACE if protect_space else np.ones(len(seg), bool)
        letters = seg[keep]
        if len(letters) > 1:
            seg[keep] = rng.permutation(letters)
            out[i:e] = seg
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  WORD unreliability — second-level corruption (level 1). Defeats the word/spelling level.
# ════════════════════════════════════════════════════════════════════════════════════════════════

def _word_spans(ids):
    sp = np.nonzero(ids == SPACE)[0]
    bounds = np.concatenate([[-1], sp, [len(ids)]])
    return [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i] + 1]


def _build_word_bank(ids, max_words=20000):
    """A pool of real word spellings (char-id tuples) to substitute from — the 'plausible but wrong word'.
    Sampled by frequency so substitutes look like real text (function words common, etc.)."""
    spans = _word_spans(ids)
    seen = {}
    for s, e in spans:
        w = ids[s:e].tobytes()
        seen[w] = seen.get(w, 0) + 1
    items = sorted(seen.items(), key=lambda kv: -kv[1])[:max_words]
    bank = [np.frombuffer(w, dtype=ids.dtype) for w, _ in items]
    return bank


def word_corrupt(ids, q, rng, sub=0.34, drop=0.33, swap=0.33, bank=None):
    """With probability q, corrupt each WORD span. Three modes:
      - sub  : replace the whole word with a random in-vocab word (same surface stats, wrong meaning).
      - drop : delete the word (and one adjoining space) — the word vanishes from perception.
      - swap : swap this word with the next word (word-order scramble).
    Boundaries (spaces) are otherwise preserved. Targets stay aligned to the CLEAN stream because we
    DON'T change length on the eval path — see word_corrupt_aligned. This builder MAY change length and is
    for the *length-preserving* substitute/swap-heavy regime; pass it through char-position re-alignment.

    Returns a NEW id array (length may differ when drops happen)."""
    if bank is None:
        bank = _build_word_bank(ids)
    spans = _word_spans(ids)
    if not spans:
        return ids.copy()
    nw = len(spans)
    hit = rng.random(nw) < q
    mode = rng.choice(3, size=nw, p=np.array([sub, drop, swap]) / (sub + drop + swap))
    # rebuild the stream span by span
    pieces = []
    wi = 0
    while wi < nw:
        s, e = spans[wi]
        word = ids[s:e]
        if not hit[wi]:
            pieces.append(word)
            wi += 1
        elif mode[wi] == 0:                          # substitute
            pieces.append(bank[rng.integers(0, len(bank))])
            wi += 1
        elif mode[wi] == 1:                          # drop
            wi += 1                                   # emit nothing
        else:                                        # swap with next word
            if wi + 1 < nw:
                s2, e2 = spans[wi + 1]
                pieces.append(ids[s2:e2])
                pieces.append(word)
                wi += 2
            else:
                pieces.append(word)
                wi += 1
    if not pieces:
        return ids[:0].copy()
    sp = np.array([SPACE], dtype=ids.dtype)
    out = [pieces[0]]
    for p in pieces[1:]:
        out.append(sp)
        out.append(p)
    return np.concatenate(out)


def word_corrupt_aligned(ids, q, rng, sub=0.5, swap=0.5, bank=None):
    """Length-PRESERVING word corruption so the per-char target array stays aligned to the clean stream
    (the char-level eval harness needs out has the same length as ids). Only substitute (same-length pad/clip)
    and swap-adjacent (length-preserving) are used; drops are excluded here. Substitutes are padded/truncated
    with the SUBSTITUTE word's own letters by tiling, so within-word char stats still come from a real word.

    This is the WORD-noise channel used for next-CHAR scoring (q sweep) where we must keep alignment."""
    if bank is None:
        bank = _build_word_bank(ids)
    out = ids.copy()
    spans = _word_spans(ids)
    nw = len(spans)
    if nw == 0:
        return out
    hit = rng.random(nw) < q
    mode = rng.choice(2, size=nw, p=np.array([sub, swap]) / (sub + swap))
    wi = 0
    while wi < nw:
        s, e = spans[wi]
        L = e - s
        if not hit[wi]:
            wi += 1
            continue
        if mode[wi] == 0:                            # substitute, fit to length L by tiling/clipping
            w = bank[rng.integers(0, len(bank))]
            if len(w) >= L:
                out[s:e] = w[:L]
            else:
                reps = int(np.ceil(L / len(w)))
                out[s:e] = np.tile(w, reps)[:L]
            wi += 1
        else:                                        # swap-adjacent: exchange equal-length prefixes
            if wi + 1 < nw:
                s2, e2 = spans[wi + 1]
                k = min(e - s, e2 - s2)
                a = out[s:s + k].copy()
                out[s:s + k] = out[s2:s2 + k]
                out[s2:s2 + k] = a
                wi += 2
            else:
                wi += 1
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  word-stream corruption (for next-WORD scoring): operate directly on a word-id stream.
# ════════════════════════════════════════════════════════════════════════════════════════════════

def word_stream_corrupt(stream, q, rng, vocab_size, sub=0.5, swap=0.5):
    """Corrupt a WORD-ID stream in place (for the next-word eval): with prob q, substitute a random vocab id
    or swap with the next word. Length-preserving so word targets stay aligned to the clean stream."""
    out = stream.copy()
    n = len(out)
    hit = rng.random(n) < q
    idx = np.nonzero(hit)[0]
    if idx.size == 0:
        return out
    mode = rng.choice(2, size=idx.size, p=np.array([sub, swap]) / (sub + swap))
    sub_i = idx[mode == 0]
    if sub_i.size:
        out[sub_i] = rng.integers(0, vocab_size, size=sub_i.size).astype(out.dtype)
    sw_i = idx[(mode == 1)]
    sw_i = sw_i[sw_i + 1 < n]
    if sw_i.size:
        tmp = out[sw_i].copy()
        out[sw_i] = out[sw_i + 1]
        out[sw_i + 1] = tmp
    return out

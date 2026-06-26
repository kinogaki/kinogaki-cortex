"""Emergent boundaries — discover phrases and topics by SURPRISE, no delimiters, reusing Exp A's signal at
higher levels.

  phrase_cuts()  — branching entropy over the WORD stream (Harris/Exp A, now one level up): a phrase ends where
                   the next-word distribution fans out (high forward+backward branching entropy). Discovers
                   collocations / multi-word units without spaces between them being given.
  topic_cuts()   — Bayesian/predictive surprise over the meaning stream: adjacent windows of content words whose
                   bag-of-words distributions diverge (low cosine) mark a topic shift (TextTiling, framed as
                   "the topic model is surprised"). Scored against enwik9 article (<page>) boundaries = real F1.
"""
import numpy as np

def _follower_entropy(seq, vocab):
    """For each token id, entropy of the distribution of tokens that immediately follow it (forward branching)."""
    nxt = {}
    for a, b in zip(seq[:-1], seq[1:]):
        d = nxt.setdefault(a, {}); d[b] = d.get(b, 0) + 1
    H = np.zeros(vocab)
    for a, d in nxt.items():
        c = np.array(list(d.values()), float); p = c / c.sum()
        H[a] = float(-(p * np.log2(p + 1e-12)).sum())
    return H

def phrase_cuts(wids, vocab, target_rate=0.5):
    """Cut the word stream into phrases where forward+backward branching entropy rises. Returns list of (start,
    end) spans. target_rate ≈ fraction of gaps that become cuts."""
    Hf = _follower_entropy(wids, vocab)                      # high after token = uncertain what's next
    Hb = _follower_entropy(wids[::-1], vocab)                # backward
    score = Hf[wids[:-1]] + Hb[wids[1:][::-1]][::-1]         # score at each gap t|t+1
    thr = np.quantile(score, 1 - target_rate)
    cut = np.concatenate([[True], score >= thr, [True]])     # gap after position i
    edges = np.nonzero(cut)[0]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

def topic_cuts(content_wids, window=120, smooth=5):
    """TextTiling: cosine dissimilarity between adjacent windows of content words. Returns (positions, scores)
    where positions are content-word indices and scores are 'topic surprise' (1 - cosine); boundaries = peaks."""
    n = len(content_wids)
    gaps = np.arange(window, n - window, max(1, window // 6))
    sims = np.empty(len(gaps))
    for i, g in enumerate(gaps):
        l = content_wids[g - window:g]; r = content_wids[g:g + window]
        lc = np.bincount(l); rc = np.bincount(r)
        m = max(len(lc), len(rc)); lc = np.pad(lc, (0, m - len(lc))); rc = np.pad(rc, (0, m - len(rc)))
        denom = (np.linalg.norm(lc) * np.linalg.norm(rc)) or 1.0
        sims[i] = float(lc @ rc) / denom
    surprise = 1 - sims
    if smooth > 1:
        k = np.ones(smooth) / smooth; surprise = np.convolve(surprise, k, "same")
    # boundaries = local maxima of surprise above mean+0.5σ (TextTiling depth-style threshold)
    thr = surprise.mean() + 0.5 * surprise.std()
    peaks = [gaps[i] for i in range(1, len(gaps) - 1)
             if surprise[i] >= thr and surprise[i] >= surprise[i - 1] and surprise[i] >= surprise[i + 1]]
    return np.array(peaks), gaps, surprise

def f1_boundaries(pred, gold, tol):
    """Boundary-detection F1 with a tolerance window (a pred within `tol` of a gold counts as a hit)."""
    pred = np.sort(np.asarray(pred)); gold = np.sort(np.asarray(gold))
    if len(pred) == 0 or len(gold) == 0: return 0.0, 0.0, 0.0
    hit_p = sum(np.min(np.abs(gold - p)) <= tol for p in pred)
    hit_g = sum(np.min(np.abs(pred - g)) <= tol for g in gold)
    prec = hit_p / len(pred); rec = hit_g / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

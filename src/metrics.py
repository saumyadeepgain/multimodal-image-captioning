"""Dependency-free CIDEr-D (Vedantam et al., 2015), as in the coco-caption toolkit.

CIDEr is the primary metric in modern captioning papers: it weights n-grams by
TF-IDF, so matching a rare, informative phrase ("rock climber") counts more
than matching a generic one ("a man"). Before this module, CIDEr was only
available when ``pycocoevalcap`` (which also pulls in Java for METEOR) was
installed; the README nevertheless listed CIDEr numbers.

The implementation mirrors ``pycocoevalcap.cider`` (CIDEr-D): n = 1..4, document
frequencies computed over the reference set being evaluated, clipped
term-frequency vectors, a Gaussian length penalty (sigma = 6) and a x10 scale.
``tests/test_metrics.py`` checks it against pycocoevalcap when available.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence


def _ngrams(words: Sequence[str], n_max: int = 4) -> Counter:
    c = Counter()
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            c[tuple(words[i : i + n])] += 1
    return c


def _vec(counts: Counter, df: dict[tuple, float], log_n_docs: float, n_max: int):
    vec = [defaultdict(float) for _ in range(n_max)]
    norm = [0.0] * n_max
    length = 0
    for ng, tf in counts.items():
        n = len(ng) - 1
        vec[n][ng] = float(tf) * (log_n_docs - math.log(max(1.0, df.get(ng, 0.0))))
        norm[n] += vec[n][ng] ** 2
        if n == 1:
            length += tf
    return vec, [math.sqrt(x) for x in norm], length


def _sim(vh, vr, nh, nr, lh, lr, n_max, sigma):
    delta = float(lh - lr)
    out = [0.0] * n_max
    for n in range(n_max):
        for ng, val in vh[n].items():
            out[n] += min(val, vr[n].get(ng, 0.0)) * vr[n].get(ng, 0.0)  # clipped (CIDEr-D)
        if nh[n] != 0 and nr[n] != 0:
            out[n] /= nh[n] * nr[n]
        out[n] *= math.e ** (-(delta**2) / (2 * sigma**2))
    return out


def cider_d(
    hypotheses: list[list[str]],
    references: list[list[list[str]]],
    n_max: int = 4,
    sigma: float = 6.0,
) -> tuple[float, list[float]]:
    """Corpus CIDEr-D. Returns ``(mean score, per-image scores)``."""
    ref_counts = [[_ngrams(r, n_max) for r in refs] for refs in references]
    df: dict[tuple, float] = defaultdict(float)
    for refs in ref_counts:
        for ng in {ng for rc in refs for ng in rc}:
            df[ng] += 1
    log_n = math.log(float(len(references)))
    scores = []
    for hyp, refs in zip(hypotheses, ref_counts):
        vh, nh, lh = _vec(_ngrams(hyp, n_max), df, log_n, n_max)
        acc = [0.0] * n_max
        for rc in refs:
            vr, nr, lr = _vec(rc, df, log_n, n_max)
            acc = [a + b for a, b in zip(acc, _sim(vh, vr, nh, nr, lh, lr, n_max, sigma))]
        score = sum(acc) / n_max / max(len(refs), 1) * 10.0
        scores.append(score)
    return (sum(scores) / max(len(scores), 1)), scores


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(
    hypotheses: list[list[str]], references: list[list[list[str]]], beta: float = 1.2
) -> tuple[float, list[float]]:
    """ROUGE-L F-score as in coco-caption: max precision / recall over refs."""
    scores = []
    for hyp, refs in zip(hypotheses, references):
        precs, recs = [], []
        for r in refs:
            lcs = _lcs(r, hyp)
            precs.append(lcs / len(hyp) if hyp else 0.0)
            recs.append(lcs / len(r) if r else 0.0)
        p, r = max(precs, default=0.0), max(recs, default=0.0)
        scores.append(((1 + beta**2) * p * r) / (r + beta**2 * p) if p and r else 0.0)
    return sum(scores) / max(len(scores), 1), scores

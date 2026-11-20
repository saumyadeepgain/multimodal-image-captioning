import random

import pytest

from src.metrics import cider_d

WORDS = "a dog man runs on the grass in red ball water boy jumps".split()


def _corpus(seed=0, n=30):
    rng = random.Random(seed)

    def sent():
        return [rng.choice(WORDS) for _ in range(rng.randint(3, 10))]

    refs = [[sent() for _ in range(5)] for _ in range(n)]
    hyps = [sent() if i % 3 else list(refs[i][0]) for i in range(n)]
    return hyps, refs


def test_cider_perfect_match_scores_higher():
    hyps, refs = _corpus()
    exact = [r[0] for r in refs]
    assert cider_d(exact, refs)[0] > cider_d(hyps, refs)[0] > 0


def test_cider_matches_pycocoevalcap():
    Cider = pytest.importorskip("pycocoevalcap.cider.cider").Cider
    hyps, refs = _corpus(1)
    ours, per = cider_d(hyps, refs)
    gts = {i: [" ".join(r) for r in rs] for i, rs in enumerate(refs)}
    res = {i: [" ".join(h)] for i, h in enumerate(hyps)}
    theirs, their_per = Cider().compute_score(gts, res)
    assert ours == pytest.approx(theirs, rel=1e-6)
    assert per == pytest.approx(list(their_per), rel=1e-6)


def test_rouge_matches_pycocoevalcap():
    Rouge = pytest.importorskip("pycocoevalcap.rouge.rouge").Rouge
    from src.metrics import rouge_l

    hyps, refs = _corpus(2)
    gts = {i: [" ".join(r) for r in rs] for i, rs in enumerate(refs)}
    res = {i: [" ".join(h)] for i, h in enumerate(hyps)}
    assert rouge_l(hyps, refs)[0] == pytest.approx(Rouge().compute_score(gts, res)[0], rel=1e-6)

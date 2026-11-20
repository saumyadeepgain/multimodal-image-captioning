"""The classic sanity check: a model that cannot memorise one tiny batch has a
bug in the model, the loss, the optimizer or the training loop.

Both tests drive the *real* training loop (``src.train.train_epoch``: AMP
hooks, gradient clipping, scheduler, accumulation code path) with a loader
that yields the same batch over and over.
"""

import math

import torch
import torch.nn as nn

from src.dataset import Vocabulary
from src.inference import greedy_decode
from src.model import BACKBONE_DIM, ImageCaptioningModel
from src.train import train_epoch, warmup_cosine

CAPTIONS = [
    "a dog runs on the grass",
    "a man rides a red bike",
    "two children play in the water",
    "a black cat sleeps on a sofa",
]


def _batch():
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(CAPTIONS)
    encoded = [torch.tensor(vocab.encode(c)) for c in CAPTIONS]
    caps = nn.utils.rnn.pad_sequence(encoded, batch_first=True, padding_value=vocab.pad_idx)
    return vocab, caps


def _loss(model, x, caps, vocab, cached):
    model.eval()
    with torch.no_grad():
        logits = model(captions=caps, backbone_features=x) if cached else model(x, caps)
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1), ignore_index=vocab.pad_idx
        ).item()


def _overfit(cached: bool, steps: int):
    torch.manual_seed(0)
    vocab, caps = _batch()
    model = ImageCaptioningModel(
        64, len(vocab), num_layers=2, nhead=4, pretrained=False, dropout=0.0, pad_idx=vocab.pad_idx
    )
    if cached:
        x = torch.randn(len(CAPTIONS), 49, BACKBONE_DIM)
    else:
        x = torch.randn(len(CAPTIONS), 3, 64, 64)
    initial = _loss(model, x, caps, vocab, cached)
    optimizer = torch.optim.AdamW(model.param_groups(3e-3, 0.0), lr=3e-3)
    scheduler = warmup_cosine(optimizer, warmup_steps=5, total_steps=steps)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    stats = train_epoch(
        model,
        [(x, caps)] * steps,
        criterion,
        optimizer,
        scheduler,
        torch.device("cpu"),
        cached=cached,
    )
    final = _loss(model, x, caps, vocab, cached)
    return model, vocab, x, caps, initial, final, stats


def test_overfit_single_batch_cached_features():
    model, vocab, x, caps, initial, final, stats = _overfit(cached=True, steps=50)
    assert stats["steps"] == 50
    assert math.isfinite(stats["loss"]) and math.isfinite(stats["grad_norm"])
    assert initial > 2.0, initial  # starts near log(|V|)
    assert final < 0.05 * initial, (initial, final)  # loss drops by > 95 %
    # Memorised: greedy decoding reproduces every training caption exactly.
    seqs, _ = greedy_decode(model, None, vocab, max_length=12, backbone_features=x)
    assert [" ".join(vocab.decode(s)) for s in seqs.tolist()] == CAPTIONS


def test_overfit_single_batch_end_to_end_from_pixels():
    """Same check through the frozen ResNet-50 (projection + decoder learn)."""
    _, _, _, _, initial, final, _ = _overfit(cached=False, steps=30)
    assert final < 0.2 * initial, (initial, final)

"""Self-contained checkpoints: weights + model config + vocabulary.

The first version saved only ``state_dict()``. The demo app then had to guess
``vocab_size`` (it hard-coded 10000, so loading always failed) and had an empty
vocabulary, so it could never turn ids back into words.
"""

from __future__ import annotations

import torch

from .dataset import Vocabulary
from .model import ImageCaptioningModel


def save_checkpoint(path: str, model: ImageCaptioningModel, vocab: Vocabulary, **meta) -> None:
    torch.save({"model_state": model.state_dict(), "model_config": model.config,
                "vocab": vocab.to_dict(), **meta}, path)


def load_checkpoint(path: str, device="cpu", pretrained_backbone: bool = False):
    """Returns ``(model, vocab, meta)``. The backbone weights come from the
    checkpoint, so ImageNet weights are not downloaded again by default."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state" not in ckpt:
        raise ValueError(f"{path} is a bare state_dict from the old training script; "
                         "re-train with `python -m src.train` to get a full checkpoint.")
    vocab = Vocabulary.from_dict(ckpt["vocab"])
    model = ImageCaptioningModel(**ckpt["model_config"], pretrained=pretrained_backbone)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "model_config", "vocab")}
    return model, vocab, meta

"""Attention heat-map overlays.

Fixes vs. the first version:

* The old function placed subplot ``l`` for word index ``l`` but skipped
  ``<start>`` *after* computing the grid, so the grid had one empty cell and
  ``<end>``/``<pad>`` were plotted as if they were words.
* The image was shown at its original resolution while the model saw a
  224x224 *resized* image, so the heat-map was mis-aligned on non-square photos.
  The overlay now uses the same resize as the model input.
* A grey colormap with alpha 0.6 hid the photo; a perceptual colormap with a
  colour bar makes sharp vs. diffuse attention readable.
* ``attention_entropy`` quantifies how focused each word's attention is.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

SKIP = {"<start>", "<end>", "<pad>"}


def _to_grid(attn) -> np.ndarray:
    a = attn.detach().float().cpu().numpy() if isinstance(attn, torch.Tensor) else np.asarray(attn)
    g = int(round(math.sqrt(a.shape[-1])))
    return a.reshape(g, g)


def attention_entropy(attn) -> float:
    """Shannon entropy (nats) of one attention map; low = focused."""
    p = np.clip(_to_grid(attn).ravel(), 1e-12, None)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def plot_attention(image: Image.Image, result_caption: Sequence[str], attention_plot: List,
                   image_size: int = 224, cols: int = 4, cmap: str = "inferno"):
    """
    Args:
        image: the original PIL image.
        result_caption: words as returned by ``generate_caption_greedy``
            (``<start> w1 ... wn <end>``).
        attention_plot: one (49,) map per generated word, i.e.
            ``attention_plot[i]`` belongs to ``result_caption[i + 1]``.
    """
    img = np.asarray(image.convert("RGB").resize((image_size, image_size)))
    pairs = [(w, attention_plot[i - 1]) for i, w in enumerate(result_caption)
             if i > 0 and w not in SKIP and i - 1 < len(attention_plot)]
    if not pairs:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.axis("off")
        return fig

    rows = math.ceil(len(pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (word, attn) in zip(axes.ravel(), pairs):
        grid = _to_grid(attn)
        ax.imshow(img)
        ax.imshow(grid, cmap=cmap, alpha=0.55, extent=(0, image_size, image_size, 0),
                  interpolation="bicubic")
        ax.set_title(f"{word}  (H={attention_entropy(attn):.2f})", fontsize=12)
    fig.tight_layout()
    return fig

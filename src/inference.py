"""Caption generation: batched greedy decoding and vectorised beam search.

Old version vs. new:

* Greedy decoding handled one image at a time. ``greedy_decode`` decodes a
  whole batch in parallel and stops as soon as every sequence has emitted
  ``<end>`` -- this is what makes evaluating the test set fast.
* Beam search ran the decoder once **per beam** in a Python loop, used a
  length normalisation of ``score / len`` that counted ``<start>``, and could
  never return attention maps. The new version runs all beams as one batch,
  uses the GNMT length penalty, and returns the attention of the best beam.
* The image encoder runs exactly once per image; ``max_length`` is clamped to
  the decoder's positional-encoding size so long generations cannot crash.
* Both decoders use the decoder's KV-cache (one token per step instead of
  re-running the whole prefix), and beam search is batched across images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # imported for type hints only (avoids import cycles)
    from .dataset import Vocabulary
    from .model import ImageCaptioningModel

AttentionMaps = List[torch.Tensor]


def _clamp_len(model: "ImageCaptioningModel", max_length: int) -> int:
    return min(max_length, model.decoder.max_seq_length - 1)


def _memory(model: "ImageCaptioningModel", images: Optional[torch.Tensor],
            backbone_features: Optional[torch.Tensor], device: Any) -> torch.Tensor:
    if backbone_features is not None:
        return model.encode(backbone_features=backbone_features.to(device))
    return model.encode(images.to(device))


@torch.no_grad()
def greedy_decode(model: "ImageCaptioningModel", images: Optional[torch.Tensor], vocab: "Vocabulary",
                  max_length: int = 30, device: Any = "cpu", return_attention: bool = False,
                  backbone_features: Optional[torch.Tensor] = None
                  ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Decode a batch. Returns ``(ids (B, L) incl. <start>, attentions or None)``.

    ``attentions`` has shape (B, L-1, 49): one spatial map per generated token.
    Uses the decoder KV-cache, so each step processes a single token.
    """
    model.eval()
    memory = _memory(model, images, backbone_features, device)
    B, dev = memory.size(0), memory.device
    max_length = _clamp_len(model, max_length)
    caches = model.decoder.init_cache(memory)
    seqs = torch.full((B, 1), vocab.start_idx, dtype=torch.long, device=dev)
    finished = torch.zeros(B, dtype=torch.bool, device=dev)
    attn_steps = []
    for t in range(max_length):
        out = model.decoder.step(seqs[:, -1], t, caches, return_attention=return_attention)
        logits, attn = out if return_attention else (out, None)
        nxt = logits.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, vocab.pad_idx), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        if attn is not None:
            attn_steps.append(attn)
        finished |= nxt.eq(vocab.end_idx)
        if finished.all():
            break
    attns = torch.stack(attn_steps, 1) if attn_steps else None
    return seqs, attns


@torch.no_grad()
def beam_search_batch(model: "ImageCaptioningModel", images: Optional[torch.Tensor],
                      vocab: "Vocabulary", beam_size: int = 3, max_length: int = 30,
                      length_penalty: float = 0.7, device: Any = "cpu",
                      backbone_features: Optional[torch.Tensor] = None) -> Tuple[List[List[int]], List[float]]:
    """Beam search for a BATCH of images: all B x k hypotheses run as one batch
    and share a KV-cache that is re-ordered whenever beams are re-ranked."""
    model.eval()
    memory = _memory(model, images, backbone_features, device)
    B, k, dev = memory.size(0), beam_size, memory.device
    V = model.decoder.fc_out.out_features
    max_length = _clamp_len(model, max_length)
    memory = memory.repeat_interleave(k, 0)                         # (B*k, S, E)
    caches = model.decoder.init_cache(memory)

    seqs = torch.full((B * k, 1), vocab.start_idx, dtype=torch.long, device=dev)
    scores = torch.full((B, k), float("-inf"), device=dev)
    scores[:, 0] = 0.0          # all beams start identical; keep one alive at t=0
    scores = scores.view(-1)
    finished = torch.zeros(B * k, dtype=torch.bool, device=dev)
    lengths = torch.zeros(B * k, device=dev)
    pad_row = torch.full((V,), float("-inf"), device=dev)
    pad_row[vocab.pad_idx] = 0.0
    offsets = (torch.arange(B, device=dev) * k)[:, None]

    for t in range(max_length):
        logp = F.log_softmax(model.decoder.step(seqs[:, -1], t, caches).float(), dim=-1)
        logp = torch.where(finished[:, None], pad_row[None], logp)  # finished: only <pad>, cost 0
        cand = (scores[:, None] + logp).view(B, k * V)
        top, idx = cand.topk(k, dim=-1)                              # (B, k)
        src = (idx // V + offsets).view(-1)                          # flat source beam
        tok = (idx % V).view(-1)
        seqs = torch.cat([seqs[src], tok[:, None]], 1)
        model.decoder.reorder_cache(caches, src)
        was_finished = finished[src]
        lengths = lengths[src] + (~was_finished).float()
        finished = was_finished | tok.eq(vocab.end_idx)
        scores = top.view(-1)
        if finished.all():
            break

    lp = ((5.0 + lengths) / 6.0) ** length_penalty                  # GNMT length penalty
    norm = (scores / lp).view(B, k)
    best = norm.argmax(-1) + offsets[:, 0]
    return [seqs[i].tolist() for i in best.tolist()], scores[best].tolist()


@torch.no_grad()
def beam_search(model: "ImageCaptioningModel", image: Optional[torch.Tensor], vocab: "Vocabulary",
                beam_size: int = 3, max_length: int = 30, length_penalty: float = 0.7,
                device: Any = "cpu", backbone_features: Optional[torch.Tensor] = None
                ) -> Tuple[List[int], float]:
    """Beam search for ONE image."""
    ids, scores = beam_search_batch(model, image[:1] if image is not None else None, vocab,
                                    beam_size, max_length, length_penalty, device,
                                    backbone_features[:1] if backbone_features is not None else None)
    return ids[0], scores[0]


def ids_to_words(ids: List[int], vocab: "Vocabulary", keep_special: bool = False) -> List[str]:
    """Ids -> words. ``keep_special`` keeps <start>/<end> (needed for attention plots)."""
    if keep_special:
        words = []
        for i in ids:
            w = vocab.itos.get(int(i), "<unk>")
            if w == "<pad>":
                continue
            words.append(w)
            if w == "<end>":
                break
        return words
    return vocab.decode(ids)


# --------------------------------------------------------------------------- #
# Backwards-compatible single-image helpers (used by app.py and the notebook)
# --------------------------------------------------------------------------- #
def generate_caption_greedy(model: "ImageCaptioningModel", image: torch.Tensor, vocab: "Vocabulary",
                            max_length: int = 30, device: Any = "cpu"
                            ) -> Tuple[List[str], AttentionMaps]:
    """Returns ``(words incl. <start>/<end>, [attention (49,) per generated word])``."""
    seqs, attn = greedy_decode(model, image, vocab, max_length, device, return_attention=True)
    words = ids_to_words(seqs[0].tolist(), vocab, keep_special=True)
    n_generated = len(words) - 1
    return words, [attn[0, t].cpu() for t in range(n_generated)]


def generate_caption_beam_search(model: "ImageCaptioningModel", image: torch.Tensor,
                                 vocab: "Vocabulary", max_length: int = 30, beam_size: int = 3,
                                 device: Any = "cpu", return_attention: bool = False):
    """Beam search for one image; returns words (and attention maps if requested)."""
    ids, _ = beam_search(model, image, vocab, beam_size, max_length, device=device)
    words = ids_to_words(ids, vocab, keep_special=True)
    if not return_attention:
        return words
    # One extra forward pass over the chosen sequence gives its attention maps.
    with torch.no_grad():
        memory = model.encode(image.to(device))
        tokens = torch.tensor([ids[: len(words)]], device=memory.device)
        _, attn = model.decoder(memory, tokens, return_attention=True)
    # attn[t] = attention while predicting token t+1
    return words, [attn[0, t].cpu() for t in range(len(words) - 1)]

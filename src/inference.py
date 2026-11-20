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
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _clamp_len(model, max_length: int) -> int:
    return min(max_length, model.decoder.max_seq_length - 1)


@torch.no_grad()
def greedy_decode(model, images: Optional[torch.Tensor], vocab, max_length: int = 30,
                  device="cpu", return_attention: bool = False, backbone_features=None):
    """Decode a batch. Returns ``(ids (B, L) incl. <start>, attentions or None)``.

    ``attentions`` has shape (B, L-1, 49): one spatial map per generated token.
    """
    model.eval()
    if backbone_features is not None:
        memory = model.encode(backbone_features=backbone_features.to(device))
    else:
        memory = model.encode(images.to(device))
    B = memory.size(0)
    max_length = _clamp_len(model, max_length)
    seqs = torch.full((B, 1), vocab.start_idx, dtype=torch.long, device=memory.device)
    finished = torch.zeros(B, dtype=torch.bool, device=memory.device)
    attn_steps = []
    for _ in range(max_length):
        out = model.decoder(memory, seqs, return_attention=return_attention)
        logits, attn = out if return_attention else (out, None)
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, vocab.pad_idx), nxt)
        seqs = torch.cat([seqs, nxt[:, None]], dim=1)
        if attn is not None:
            attn_steps.append(attn[:, -1])
        finished |= nxt.eq(vocab.end_idx)
        if finished.all():
            break
    attns = torch.stack(attn_steps, 1) if attn_steps else None
    return seqs, attns


@torch.no_grad()
def beam_search(model, image: torch.Tensor, vocab, beam_size: int = 3, max_length: int = 30,
                length_penalty: float = 0.7, device="cpu", backbone_features=None
                ) -> Tuple[List[int], float]:
    """Beam search for ONE image (all beams evaluated in a single batch)."""
    model.eval()
    if backbone_features is not None:
        memory = model.encode(backbone_features=backbone_features.to(device))
    else:
        memory = model.encode(image.to(device))
    memory = memory[:1].expand(beam_size, -1, -1)
    dev = memory.device
    max_length = _clamp_len(model, max_length)
    V = model.decoder.fc_out.out_features

    seqs = torch.full((beam_size, 1), vocab.start_idx, dtype=torch.long, device=dev)
    scores = torch.full((beam_size,), float("-inf"), device=dev)
    scores[0] = 0.0  # all beams start identical; keep only one alive at t=0
    finished = torch.zeros(beam_size, dtype=torch.bool, device=dev)
    lengths = torch.zeros(beam_size, device=dev)

    for _ in range(max_length):
        logp = F.log_softmax(model.decoder(memory, seqs)[:, -1].float(), dim=-1)
        # A finished beam may only be extended by <pad> at zero cost.
        pad_row = torch.full((V,), float("-inf"), device=dev)
        pad_row[vocab.pad_idx] = 0.0
        logp = torch.where(finished[:, None], pad_row[None], logp)
        cand = (scores[:, None] + logp).view(-1)
        top, idx = cand.topk(beam_size)
        beam_idx, tok = idx // V, idx % V
        seqs = torch.cat([seqs[beam_idx], tok[:, None]], 1)
        was_finished = finished[beam_idx]
        lengths = lengths[beam_idx] + (~was_finished).float()
        finished = was_finished | tok.eq(vocab.end_idx)
        scores = top
        if finished.all():
            break

    lp = ((5.0 + lengths) / 6.0) ** length_penalty  # GNMT length penalty
    best = int((scores / lp).argmax())
    return seqs[best].tolist(), float(scores[best])


def ids_to_words(ids, vocab, keep_special: bool = False) -> List[str]:
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
def generate_caption_greedy(model, image, vocab, max_length=30, device="cpu"):
    """Returns ``(words incl. <start>/<end>, [attention (49,) per generated word])``."""
    seqs, attn = greedy_decode(model, image, vocab, max_length, device, return_attention=True)
    words = ids_to_words(seqs[0].tolist(), vocab, keep_special=True)
    n_generated = len(words) - 1
    return words, [attn[0, t].cpu() for t in range(n_generated)]


def generate_caption_beam_search(model, image, vocab, max_length=30, beam_size=3, device="cpu",
                                 return_attention: bool = False):
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

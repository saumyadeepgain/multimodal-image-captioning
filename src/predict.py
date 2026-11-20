"""Caption one or more images from the command line.

    python -m src.predict img1.jpg img2.jpg --beam 3 --attention-out attn.png
    python src/predict.py photos/*.jpg --json > captions.json

Captions are written to stdout (``path<TAB>caption`` or ``--json``) so the
output can be piped; log messages go to stderr. Unreadable images are reported
and skipped instead of aborting the whole batch.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):  # executed as `python src/predict.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")  # register the package for the relative imports below

import argparse
import json
from typing import Any, Dict, List, Optional

import torch

from .dataset import Vocabulary, get_transforms, open_image
from .inference import beam_search_batch, greedy_decode
from .model import ImageCaptioningModel
from .utils import get_device, get_logger, load_checkpoint


def caption_images(model: ImageCaptioningModel, vocab: Vocabulary, paths: List[str], beam: int = 3,
                   max_length: int = 30, device: Any = "cpu") -> List[str]:
    """Caption image files in one batch (greedy if ``beam <= 1``).

    Raises ``OSError`` / ``ValueError`` if any file cannot be read; see
    :func:`caption_files` for the tolerant version used by the CLI."""
    tf = get_transforms(train=False)
    x = torch.stack([tf(open_image(p)) for p in paths])
    if beam <= 1:
        seqs, _ = greedy_decode(model, x, vocab, max_length, device)
        ids = seqs.tolist()
    else:
        ids, _ = beam_search_batch(model, x, vocab, beam, max_length, device=device)
    return [" ".join(vocab.decode(s)) for s in ids]


def caption_files(model: ImageCaptioningModel, vocab: Vocabulary, paths: List[str],
                  beam: int = 3, max_length: int = 30, device: Any = "cpu",
                  batch_size: int = 32) -> Dict[str, Optional[str]]:
    """Caption many files in batches; unreadable files map to ``None``."""
    log = get_logger()
    readable: List[str] = []
    results: Dict[str, Optional[str]] = {}
    for p in paths:
        try:
            open_image(p)
            readable.append(p)
        except (OSError, ValueError, SyntaxError) as exc:
            log.error(f"Skipping {p}: {type(exc).__name__}: {exc}")
            results[p] = None
    for s in range(0, len(readable), batch_size):
        chunk = readable[s:s + batch_size]
        for p, c in zip(chunk, caption_images(model, vocab, chunk, beam, max_length, device)):
            results[p] = c
    return {p: results[p] for p in paths}


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Caption images with a trained checkpoint")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--beam", "--beam_size", type=int, default=3)
    ap.add_argument("--max-length", "--max_length", type=int, default=30)
    ap.add_argument("--batch-size", "--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    ap.add_argument("--attention-out", help="save an attention figure for the FIRST image")
    ap.add_argument("--json", action="store_true", help="write JSON instead of text")
    args = ap.parse_args(argv)

    log = get_logger(stream=sys.stderr)
    try:
        device = get_device(args.device)
        model, vocab, _ = load_checkpoint(args.checkpoint, device)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)
    caps = caption_files(model, vocab, args.images, args.beam, args.max_length, device,
                         args.batch_size)
    if args.json:
        sys.stdout.write(json.dumps(caps, indent=2) + "\n")
    else:
        for p, c in caps.items():
            sys.stdout.write(f"{p}\t{c if c is not None else '<unreadable>'}\n")

    if args.attention_out:
        from .attention_viz import plot_attention
        from .inference import generate_caption_beam_search, generate_caption_greedy
        first = args.images[0]
        if caps[first] is None:
            log.error(f"Cannot draw attention: {first} is unreadable.")
            sys.exit(1)
        img = open_image(first)
        x = get_transforms(train=False)(img).unsqueeze(0)
        if args.beam > 1:
            words, attn = generate_caption_beam_search(model, x, vocab, args.max_length, args.beam,
                                                       device, return_attention=True)
        else:
            words, attn = generate_caption_greedy(model, x, vocab, args.max_length, device)
        plot_attention(img, words, attn).savefig(args.attention_out, dpi=120, bbox_inches="tight")
        log.info(f"attention figure -> {args.attention_out}")
    if any(c is None for c in caps.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()

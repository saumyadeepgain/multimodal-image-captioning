"""Standalone inference: caption images or whole folders with a trained checkpoint.

    python -m src.predict photo.jpg                                  # one image
    python -m src.predict photos/ --recursive --format csv --output captions.csv
    python -m src.predict photos/ --save-attention viz/              # + attention figures
    python src/predict.py a.jpg b.png --beam 5 --device cpu --format json

Pipeline per image: load (EXIF-rotated, RGB) -> resize to the training image size
-> ImageNet normalisation -> model in eval mode under ``torch.inference_mode`` ->
greedy (``--beam 1``) or batched beam search -> caption.

Results go to stdout (or ``--output``) as text, JSON, JSON Lines or CSV; log
messages go to stderr, so output can be piped. Unreadable files are reported
and skipped; the exit code is 0 if every image was captioned, 2 if some were
skipped and 1 on fatal errors (missing checkpoint, no images found).
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):  # executed as `python src/predict.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")  # register the package for the relative imports below

import argparse
import csv
import io
import json
import time
from typing import Any, Optional

import torch
from PIL import Image, ImageOps

from .dataset import Vocabulary, get_transforms, open_image
from .inference import beam_search_batch, greedy_decode
from .model import ImageCaptioningModel
from .utils import get_device, get_logger, load_checkpoint

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")
FORMATS = ("text", "json", "jsonl", "csv")


# --------------------------------------------------------------------------- inputs
def collect_images(inputs: list[str], recursive: bool = False) -> list[str]:
    """Expand files and folders into a sorted, de-duplicated list of image paths.

    Files given explicitly are always kept (even without an image extension, so
    a broken or oddly named file is reported rather than silently ignored);
    files found inside folders are filtered by extension. Raises
    ``FileNotFoundError`` for a path that does not exist."""
    explicit: list[str] = []
    discovered: list[str] = []
    for item in inputs:
        if os.path.isfile(item):
            explicit.append(item)
        elif os.path.isdir(item):
            if recursive:
                for root, _, files in os.walk(item):
                    discovered += [os.path.join(root, f) for f in files]
            else:
                discovered += [
                    os.path.join(item, f)
                    for f in os.listdir(item)
                    if os.path.isfile(os.path.join(item, f))
                ]
        else:
            raise FileNotFoundError(f"Input not found: {item}")
    discovered = [p for p in discovered if p.lower().endswith(IMAGE_EXTENSIONS)]
    return list(dict.fromkeys(explicit + sorted(discovered)))


def load_for_model(path: str) -> Image.Image:
    """Open an image the way a phone/camera shows it (EXIF orientation applied), as RGB."""
    return ImageOps.exif_transpose(open_image(path)).convert("RGB")


# ------------------------------------------------------------------------ inference
@torch.inference_mode()
def caption_images(
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    paths: list[str],
    beam: int = 3,
    max_length: int = 30,
    device: Any = "cpu",
    image_size: int = 224,
    length_penalty: float = 0.7,
) -> list[str]:
    """Caption image files in one batch (greedy if ``beam <= 1``).

    Raises ``OSError`` / ``ValueError`` if any file cannot be read; see
    :func:`caption_files` for the tolerant, batched version used by the CLI."""
    model.eval()
    tf = get_transforms(train=False, image_size=image_size)
    x = torch.stack([tf(load_for_model(p)) for p in paths])
    if beam <= 1:
        seqs, _ = greedy_decode(model, x, vocab, max_length, device)
        ids = seqs.tolist()
    else:
        ids, _ = beam_search_batch(model, x, vocab, beam, max_length, length_penalty, device=device)
    return [" ".join(vocab.decode(s)) for s in ids]


def caption_files(
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    paths: list[str],
    beam: int = 3,
    max_length: int = 30,
    device: Any = "cpu",
    batch_size: int = 32,
    image_size: int = 224,
    length_penalty: float = 0.7,
) -> dict[str, Optional[str]]:
    """Caption many files in batches; unreadable files map to ``None``."""
    log = get_logger()
    readable: list[str] = []
    results: dict[str, Optional[str]] = {}
    for p in paths:
        try:
            load_for_model(p)
            readable.append(p)
        except (OSError, ValueError, SyntaxError) as exc:
            log.error(f"Skipping {p}: {type(exc).__name__}: {exc}")
            results[p] = None
    for s in range(0, len(readable), batch_size):
        chunk = readable[s : s + batch_size]
        captions = caption_images(
            model, vocab, chunk, beam, max_length, device, image_size, length_penalty
        )
        results.update(zip(chunk, captions))
    return {p: results[p] for p in paths}


def save_attention_figure(
    model: ImageCaptioningModel,
    vocab: Vocabulary,
    path: str,
    out_file: str,
    beam: int = 3,
    max_length: int = 30,
    device: Any = "cpu",
    image_size: int = 224,
) -> str:
    """Write a per-word attention figure (caption as title) for one image; returns the caption."""
    import matplotlib.pyplot as plt

    from .attention_viz import plot_attention
    from .inference import generate_caption_beam_search, generate_caption_greedy

    img = load_for_model(path)
    x = get_transforms(train=False, image_size=image_size)(img).unsqueeze(0)
    with torch.inference_mode():
        if beam > 1:
            words, attn = generate_caption_beam_search(
                model, x, vocab, max_length, beam, device, return_attention=True
            )
        else:
            words, attn = generate_caption_greedy(model, x, vocab, max_length, device)
    caption = " ".join(w for w in words if w not in ("<start>", "<end>", "<pad>"))
    fig = plot_attention(img, words, attn, image_size=image_size)
    fig.suptitle(caption, fontsize=14, y=1.02)
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    fig.savefig(out_file, dpi=120, bbox_inches="tight")
    plt.close(fig)  # avoid accumulating figures when processing folders
    return caption


# --------------------------------------------------------------------------- output
def format_results(results: dict[str, Optional[str]], fmt: str) -> str:
    """Render ``{path: caption or None}`` as ``text`` / ``json`` / ``jsonl`` / ``csv``.

    ``json_map`` is the legacy ``--json`` shape: ``{path: caption or null}``."""
    if fmt == "json_map":
        return json.dumps(results, indent=2) + "\n"
    rows = [
        {"image": p, "caption": c, "status": "ok" if c is not None else "unreadable"}
        for p, c in results.items()
    ]
    if fmt == "json":
        return json.dumps(rows, indent=2) + "\n"
    if fmt == "jsonl":
        return "".join(json.dumps(r) + "\n" for r in rows)
    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["image", "caption", "status"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()
    if fmt != "text":
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    return "".join(f"{r['image']}\t{r['caption'] or '<unreadable>'}\n" for r in rows)


# ------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Caption images (files and/or folders) with a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="image files and/or folders")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--recursive", "-r", action="store_true", help="search folders recursively")
    ap.add_argument(
        "--beam",
        "--beam_size",
        type=int,
        default=None,
        help="beam size (1 = greedy); default: eval.beam_size stored in the checkpoint",
    )
    ap.add_argument("--max-length", "--max_length", type=int, default=None)
    ap.add_argument("--batch-size", "--batch_size", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    ap.add_argument("--format", choices=FORMATS, default="text", help="output format")
    ap.add_argument("--output", "-o", default=None, help="write results here instead of stdout")
    ap.add_argument(
        "--save-attention",
        metavar="DIR",
        default=None,
        help="save <image>_attention.png for every captioned image into DIR",
    )
    ap.add_argument(
        "--attention-out",
        metavar="PNG",
        default=None,
        help="save the attention figure of the FIRST image to this file",
    )
    ap.add_argument(
        "--json", action="store_true", help="legacy: print {path: caption} JSON to stdout"
    )
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    log = get_logger(stream=sys.stderr)
    fmt = "json_map" if args.json else args.format
    try:
        paths = collect_images(args.inputs, args.recursive)
        if not paths:
            raise FileNotFoundError(
                f"No images ({', '.join(IMAGE_EXTENSIONS)}) found in {args.inputs}."
            )
        device = get_device(args.device)
        model, vocab, meta = load_checkpoint(args.checkpoint, device)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)
    model.eval()
    cfg = meta.get("config", {})
    eval_cfg, data_cfg = cfg.get("eval", {}), cfg.get("data", {})
    beam = args.beam if args.beam is not None else eval_cfg.get("beam_size", 3)
    max_len = args.max_length or eval_cfg.get("max_length", 30)
    penalty = eval_cfg.get("length_penalty", 0.7)
    image_size = data_cfg.get("image_size", 224)
    log.info(f"Captioning {len(paths)} image(s) on {device} | beam {beam} | {args.checkpoint}")

    start = time.perf_counter()
    results = caption_files(
        model, vocab, paths, beam, max_len, device, args.batch_size, image_size, penalty
    )
    elapsed = time.perf_counter() - start
    ok = [p for p, c in results.items() if c is not None]
    per_image = 1000 * elapsed / max(len(ok), 1)
    log.info(f"Captioned {len(ok)}/{len(paths)} in {elapsed:.2f}s ({per_image:.0f} ms/image)")

    text = format_results(results, fmt)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        log.info(f"Results -> {args.output}")
    else:
        sys.stdout.write(text)

    figures: list[tuple[str, str]] = []
    if args.save_attention:
        for p in ok:
            stem = os.path.splitext(os.path.basename(p))[0]
            figures.append((p, os.path.join(args.save_attention, f"{stem}_attention.png")))
    if args.attention_out:
        if results[paths[0]] is None:
            log.error(f"Cannot draw attention: {paths[0]} is unreadable.")
            sys.exit(1)
        figures.append((paths[0], args.attention_out))
    for src_path, out_file in figures:
        save_attention_figure(model, vocab, src_path, out_file, beam, max_len, device, image_size)
    if figures:
        log.info(f"Saved {len(figures)} attention figure(s)")
    if len(ok) < len(paths):
        sys.exit(2)


if __name__ == "__main__":
    main()

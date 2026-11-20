"""Corpus-level evaluation: BLEU-1..4, CIDEr-D and ROUGE-L (built in, see
``src/metrics.py``) plus optional METEOR (``--meteor``; needs pycocoevalcap + Java).

Standard captioning results are **corpus** scores against **all five**
references of each image (the original notebook scored single sentences
against one reference, and the README quoted mock numbers).

Usage::

    python -m src.evaluate --checkpoint checkpoints/best_model.pth --split test --beam 3
    python src/evaluate.py --checkpoint checkpoints/best_model.pth --device cpu

The split is rebuilt from the config stored in the checkpoint, so evaluation
always uses exactly the images the model was *not* trained on. Decoding
batches that hit CUDA out-of-memory are split in halves automatically.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):  # executed as `python src/evaluate.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")  # register the package for the relative imports below

import argparse
import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import tokenize
from .inference import beam_search_batch, greedy_decode
from .metrics import cider_d, rouge_l

if TYPE_CHECKING:
    from .dataset import ImageDataset, Vocabulary
    from .model import ImageCaptioningModel

logger = logging.getLogger("captioning")


def bleu_scores(hypotheses: list[list[str]], references: list[list[list[str]]]) -> dict[str, float]:
    """Corpus BLEU-1..4 (NLTK, smoothing method 1)."""
    smooth = SmoothingFunction().method1
    weights = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0), 3: (1 / 3,) * 3 + (0,), 4: (0.25,) * 4}
    return {
        f"BLEU-{n}": float(
            corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth)
        )
        for n, w in weights.items()
    }


def coco_scores(
    hypotheses: list[list[str]], references: list[list[list[str]]], meteor: bool = False
) -> dict[str, float]:
    """CIDEr-D and ROUGE-L (dependency-free), METEOR only on request because it
    starts a Java process and is slow."""
    out = {
        "CIDEr": cider_d(hypotheses, references)[0],
        "ROUGE-L": rouge_l(hypotheses, references)[0],
    }
    if meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor

            gts = {i: [" ".join(r) for r in refs] for i, refs in enumerate(references)}
            res = {i: [" ".join(h)] for i, h in enumerate(hypotheses)}
            out["METEOR"], _ = Meteor().compute_score(gts, res)
        except Exception as exc:  # missing package or Java
            logger.warning(f"METEOR skipped: {exc}")
    return {k: float(v) for k, v in out.items()}


def _oom_safe(fn: Callable[[torch.Tensor], list[list[int]]], x: torch.Tensor) -> list[list[int]]:
    """Run ``fn`` on a batch; on CUDA OOM, free memory and recurse on halves."""
    from .utils import free_device_memory, is_oom_error

    try:
        return fn(x)
    except Exception as exc:
        if not is_oom_error(exc) or x.size(0) == 1:
            raise
        free_device_memory()
        half = x.size(0) // 2
        logger.warning(f"Out of memory while decoding {x.size(0)} images; splitting the batch.")
        return _oom_safe(fn, x[:half]) + _oom_safe(fn, x[half:])


@torch.no_grad()
def evaluate_captions(
    model: ImageCaptioningModel,
    image_dataset: ImageDataset,
    vocab: Vocabulary,
    device: Any = "cpu",
    beam_size: int = 1,
    max_length: int = 30,
    batch_size: int = 64,
    num_workers: int = 2,
    limit: int = 0,
    feature_bank: Optional[tuple[torch.Tensor, dict[str, int]]] = None,
    meteor: bool = False,
    length_penalty: float = 0.7,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Caption the first ``limit`` (or all) images of an ``ImageDataset`` and return
    ``(metrics, sample predictions)``. ``feature_bank`` = cached backbone features
    ``(features [N, S, 2048], {image_name: row})``."""
    model.eval()
    n = len(image_dataset) if not limit else min(limit, len(image_dataset))
    if n == 0:
        raise ValueError("Cannot evaluate an empty split (0 images).")
    subset = torch.utils.data.Subset(image_dataset, range(n))
    hyps: list[list[str]] = [[] for _ in range(n)]
    start = time.perf_counter()

    def decode_images(imgs: torch.Tensor) -> list[list[int]]:
        if beam_size <= 1:
            seqs, _ = greedy_decode(model, imgs, vocab, max_length, device)
            return seqs.tolist()
        return beam_search_batch(
            model, imgs, vocab, beam_size, max_length, length_penalty, device=device
        )[0]

    def decode_features(feats: torch.Tensor) -> list[list[int]]:
        if beam_size <= 1:
            seqs, _ = greedy_decode(model, None, vocab, max_length, device, backbone_features=feats)
            return seqs.tolist()
        return beam_search_batch(
            model,
            None,
            vocab,
            beam_size,
            max_length,
            length_penalty,
            device=device,
            backbone_features=feats,
        )[0]

    if feature_bank is not None:
        feats, index = feature_bank
        rows = torch.tensor([index[image_dataset.images[i]] for i in range(n)])
        for s in range(0, n, batch_size):
            for j, seq in enumerate(_oom_safe(decode_features, feats[rows[s : s + batch_size]])):
                hyps[s + j] = vocab.decode(seq)
    else:
        loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers)
        for imgs, idx in tqdm(loader, desc=f"Decoding (beam={beam_size})", leave=False):
            for i, seq in zip(idx.tolist(), _oom_safe(decode_images, imgs)):
                hyps[i] = vocab.decode(seq)
    elapsed = time.perf_counter() - start
    refs = [[tokenize(c) for c in image_dataset.references[i]] for i in range(n)]
    metrics = bleu_scores(hyps, refs)
    metrics.update(coco_scores(hyps, refs, meteor))
    metrics["ms_per_image"] = 1000 * elapsed / n
    samples = [
        {
            "image": image_dataset.images[i],
            "prediction": " ".join(hyps[i]),
            "references": image_dataset.references[i],
        }
        for i in range(min(n, 20))
    ]
    return metrics, samples


def main(argv: Optional[list[str]] = None) -> None:
    """CLI: evaluate a checkpoint on its own val/test split."""
    from .config import Config
    from .dataset import build_datasets_from_config
    from .utils import get_device, get_logger, load_checkpoint, timestamped_log_path

    ap = argparse.ArgumentParser(description="Evaluate a captioning checkpoint")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument(
        "--img-dir",
        "--img_dir",
        default=None,
        help="override the image dir stored in the checkpoint",
    )
    ap.add_argument(
        "--captions",
        "--captions_file",
        default=None,
        help="override the captions file stored in the checkpoint",
    )
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument(
        "--beam",
        "--beam_size",
        type=int,
        default=None,
        help="default: eval.beam_size from the checkpoint's config",
    )
    ap.add_argument("--max-length", "--max_length", type=int, default=None)
    ap.add_argument("--batch-size", "--batch_size", type=int, default=None)
    ap.add_argument("--meteor", action="store_true", help="also compute METEOR (needs Java)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    ap.add_argument(
        "--num-workers",
        "--num_workers",
        type=int,
        default=None,
        help="DataLoader workers (default: data.num_workers from the checkpoint's config)",
    )
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    ap.add_argument("--out", default="checkpoints/eval_results.json")
    args = ap.parse_args(argv)

    log = get_logger(log_file=timestamped_log_path(os.path.dirname(args.out) or ".", "eval"))
    try:
        device = get_device(args.device)
        model, vocab, meta = load_checkpoint(args.checkpoint, device)
        # Rebuild EXACTLY the split used in training from the config saved in the checkpoint.
        cfg = Config.from_dict(meta.get("config", {}))
        if args.img_dir:
            cfg.data.img_dir = args.img_dir
        if args.captions:
            cfg.data.captions_file = args.captions
        cfg.data.verify_images = False  # evaluation keeps every image of the split
        beam = args.beam if args.beam is not None else cfg.eval.beam_size
        max_len = args.max_length or cfg.eval.max_length
        datasets, _ = build_datasets_from_config(cfg.data, cfg.seed, augment=False)
        log.info(
            f"Evaluating {args.checkpoint} on {args.split} "
            f"({len(datasets[f'{args.split}_images'])} images) | beam {beam} | {device}"
        )
        metrics, samples = evaluate_captions(
            model,
            datasets[f"{args.split}_images"],
            vocab,
            device,
            beam_size=beam,
            max_length=max_len,
            batch_size=args.batch_size or cfg.eval.batch_size,
            limit=args.limit,
            num_workers=cfg.data.num_workers if args.num_workers is None else args.num_workers,
            meteor=args.meteor or cfg.eval.meteor,
            length_penalty=cfg.eval.length_penalty,
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)
    for k, v in metrics.items():
        log.info(f"  {k:>12}: {v:.4f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "split": args.split,
                "beam": beam,
                "checkpoint": args.checkpoint,
                "metrics": metrics,
                "samples": samples,
            },
            f,
            indent=2,
        )
    log.info(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

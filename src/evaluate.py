"""Corpus-level evaluation: BLEU-1..4, CIDEr-D and ROUGE-L (built in, see
``src/metrics.py``) plus optional METEOR (``--meteor``; needs pycocoevalcap + Java).

Why this file exists: the first version only had a *sentence*-level BLEU loop
inside the notebook, compared each prediction against a single reference, and
every number in the README/notebook was mock data. Standard captioning
results are **corpus** BLEU against **all five** references of each image.

Usage::

    python -m src.evaluate --checkpoint checkpoints/best_model.pth \
        --img-dir data/archive/Images --captions data/archive/captions.txt --beam 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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


def bleu_scores(hypotheses: List[List[str]], references: List[List[List[str]]]) -> Dict[str, float]:
    smooth = SmoothingFunction().method1
    weights = {1: (1, 0, 0, 0), 2: (0.5, 0.5, 0, 0), 3: (1 / 3,) * 3 + (0,), 4: (0.25,) * 4}
    return {f"BLEU-{n}": corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth)
            for n, w in weights.items()}


def coco_scores(hypotheses: List[List[str]], references: List[List[List[str]]],
                meteor: bool = False) -> Dict[str, float]:
    """CIDEr-D and ROUGE-L (dependency-free), METEOR only on request because it
    starts a Java process and is slow."""
    out = {"CIDEr": cider_d(hypotheses, references)[0],
           "ROUGE-L": rouge_l(hypotheses, references)[0]}
    if meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            gts = {i: [" ".join(r) for r in refs] for i, refs in enumerate(references)}
            res = {i: [" ".join(h)] for i, h in enumerate(hypotheses)}
            out["METEOR"], _ = Meteor().compute_score(gts, res)
        except Exception as e:  # missing package or Java
            print(f"METEOR skipped: {e}")
    return {k: float(v) for k, v in out.items()}


@torch.no_grad()
def evaluate_captions(model: "ImageCaptioningModel", image_dataset: "ImageDataset",
                      vocab: "Vocabulary", device: Any = "cpu", beam_size: int = 1,
                      max_length: int = 30, batch_size: int = 64, num_workers: int = 2,
                      limit: int = 0,
                      feature_bank: Optional[Tuple[torch.Tensor, Dict[str, int]]] = None,
                      meteor: bool = False,
                      length_penalty: float = 0.7) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Caption the first ``limit`` (or all) images of an ``ImageDataset`` and return
    ``(metrics, sample predictions)``. ``feature_bank`` = cached backbone features."""
    model.eval()
    n = len(image_dataset) if not limit else min(limit, len(image_dataset))
    subset = torch.utils.data.Subset(image_dataset, range(n))
    hyps: List[List[str]] = [None] * n  # type: ignore
    start = time.perf_counter()

    def decode(imgs: Optional[torch.Tensor] = None,
               feats: Optional[torch.Tensor] = None) -> List[List[int]]:
        if beam_size <= 1:
            seqs, _ = greedy_decode(model, imgs, vocab, max_length, device, backbone_features=feats)
            return seqs.tolist()
        ids, _ = beam_search_batch(model, imgs, vocab, beam_size, max_length, length_penalty,
                                   device=device, backbone_features=feats)
        return ids

    if feature_bank is not None:
        # Cached backbone features (N, 49, 2048) + {image_name: row}
        feats, index = feature_bank
        rows = torch.tensor([index[image_dataset.images[i]] for i in range(n)])
        for s in range(0, n, batch_size):
            for j, seq in enumerate(decode(feats=feats[rows[s:s + batch_size]])):
                hyps[s + j] = vocab.decode(seq)
    else:
        loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers)
        for imgs, idx in tqdm(loader, desc=f"Decoding (beam={beam_size})", leave=False):
            for i, seq in zip(idx.tolist(), decode(imgs=imgs)):
                hyps[i] = vocab.decode(seq)
    elapsed = time.perf_counter() - start
    refs = [[tokenize(c) for c in image_dataset.references[i]] for i in range(n)]
    metrics = bleu_scores(hyps, refs)
    metrics.update(coco_scores(hyps, refs, meteor))
    metrics["ms_per_image"] = 1000 * elapsed / max(n, 1)
    samples = [{"image": image_dataset.images[i], "prediction": " ".join(hyps[i]),
                "references": image_dataset.references[i]} for i in range(min(n, 20))]
    return metrics, samples


def main() -> None:
    from .config import Config
    from .dataset import build_datasets_from_config
    from .utils import get_device, load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--img-dir", default=None, help="override the image dir stored in the checkpoint")
    ap.add_argument("--captions", default=None, help="override the captions file in the checkpoint")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--beam", type=int, default=None, help="default: eval.beam_size from config")
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--meteor", action="store_true", help="also compute METEOR (needs Java)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    ap.add_argument("--out", default="checkpoints/eval_results.json")
    args = ap.parse_args()

    device = get_device()
    model, vocab, meta = load_checkpoint(args.checkpoint, device)
    # Rebuild EXACTLY the split used in training from the config saved in the checkpoint.
    cfg = Config.from_dict(meta.get("config", {}))
    if args.img_dir:
        cfg.data.img_dir = args.img_dir
    if args.captions:
        cfg.data.captions_file = args.captions
    beam = args.beam if args.beam is not None else cfg.eval.beam_size
    max_len = args.max_length or cfg.eval.max_length
    datasets, _ = build_datasets_from_config(cfg.data, cfg.seed, augment=False)
    metrics, samples = evaluate_captions(model, datasets[f"{args.split}_images"], vocab, device,
                                         beam_size=beam, max_length=max_len,
                                         batch_size=cfg.eval.batch_size, limit=args.limit,
                                         meteor=args.meteor or cfg.eval.meteor,
                                         length_penalty=cfg.eval.length_penalty)
    print(json.dumps(metrics, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "beam": beam, "checkpoint": args.checkpoint,
                   "metrics": metrics, "samples": samples}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

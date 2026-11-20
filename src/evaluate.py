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
import time
from typing import Dict, List

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import tokenize
from .inference import beam_search_batch, greedy_decode
from .metrics import cider_d, rouge_l


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
def evaluate_captions(model, image_dataset, vocab, device="cpu", beam_size: int = 1,
                      max_length: int = 30, batch_size: int = 64, num_workers: int = 2,
                      limit: int = 0, feature_bank=None, meteor: bool = False):
    """Caption every image in an ``ImageDataset`` and score against all refs."""
    model.eval()
    n = len(image_dataset) if not limit else min(limit, len(image_dataset))
    subset = torch.utils.data.Subset(image_dataset, range(n))
    hyps: List[List[str]] = [None] * n  # type: ignore
    start = time.perf_counter()
    def decode(imgs=None, feats=None):
        if beam_size <= 1:
            seqs, _ = greedy_decode(model, imgs, vocab, max_length, device, backbone_features=feats)
            return seqs.tolist()
        ids, _ = beam_search_batch(model, imgs, vocab, beam_size, max_length, device=device,
                                   backbone_features=feats)
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


def main():
    from .checkpoint import load_checkpoint
    from .dataset import build_datasets

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--img-dir", default="data/archive/Images")
    ap.add_argument("--captions", default="data/archive/captions.txt")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=30)
    ap.add_argument("--meteor", action="store_true", help="also compute METEOR (needs Java)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N images")
    ap.add_argument("--out", default="checkpoints/eval_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, meta = load_checkpoint(args.checkpoint, device)
    cfg = meta.get("data", {})
    datasets, _ = build_datasets(args.img_dir, args.captions,
                                 freq_threshold=cfg.get("freq_threshold", 5),
                                 seed=cfg.get("seed", 42))
    metrics, samples = evaluate_captions(model, datasets[f"{args.split}_images"], vocab, device,
                                         beam_size=args.beam, max_length=args.max_length,
                                         limit=args.limit, meteor=args.meteor)
    print(json.dumps(metrics, indent=2))
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "beam": args.beam, "metrics": metrics,
                   "samples": samples}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()

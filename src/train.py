"""Training script.

    python -m src.train --img-dir data/archive/Images --captions data/archive/captions.txt

What changed vs. the first version (see docs/IMPROVEMENTS.pdf):

* It is actually runnable: the old ``src/train.py`` defined functions but had
  no ``__main__`` block, so ``python src/train.py`` did nothing.
* **Feature caching** (``--cache-features``, default when the CNN is frozen):
  ResNet-50 runs once per *image* instead of once per *caption per epoch*,
  typically a >10x speed-up of an epoch on Flickr8k.
* **AdamW + linear warm-up + cosine decay** stepped per batch, instead of Adam
  with a coarse StepLR. No weight decay on biases / norms / embeddings.
* **Label smoothing** (0.1) in the cross-entropy -- a standard regulariser for
  caption decoders that are prone to over-confident, generic captions.
* **Mixed precision** (bf16/fp16 autocast) on GPU.
* **Model selection on validation BLEU-4**, not validation loss; loss and
  caption quality are only loosely correlated.
* **Early stopping**, seeding, full checkpoints (weights+config+vocab) and a
  ``history.json`` log so the notebook can plot *real* curves.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import save_checkpoint
from .dataset import CachedFeatureDataset, MyCollate, build_datasets, make_loader
from .evaluate import evaluate_captions
from .model import ImageCaptioningModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


@torch.no_grad()
def precompute_features(model, image_dataset, device, batch_size=64, num_workers=2):
    """Run the frozen backbone once per image -> fp16 tensor (N, 49, 2048)."""
    model.eval()
    loader = DataLoader(image_dataset, batch_size=batch_size, num_workers=num_workers)
    feats = []
    for imgs, _ in tqdm(loader, desc="Caching CNN features"):
        feats.append(model.encoder.backbone(imgs.to(device)).half().cpu())
    return torch.cat(feats), {name: i for i, name in enumerate(image_dataset.images)}


def _forward(model, batch, device, cached: bool, amp_dtype):
    x, captions = batch
    x = x.to(device, non_blocking=True)
    captions = captions.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        if cached:
            logits = model(captions=captions, backbone_features=x)
        else:
            logits = model(x, captions)
    return logits.float(), captions[:, 1:]


def train_epoch(model, loader, criterion, optimizer, scheduler, device, cached=False,
                amp_dtype=None, scaler=None, max_grad_norm=1.0):
    model.train()
    total, n_tok = 0.0, 0
    grad_norms = []
    for batch in tqdm(loader, desc="Training", leave=False):
        logits, targets = _forward(model, batch, device, cached, amp_dtype)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)))
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        tokens = int(targets.ne(criterion.ignore_index).sum())
        total += loss.item() * tokens
        n_tok += tokens
    return total / max(n_tok, 1), float(np.mean(grad_norms)) if grad_norms else 0.0


@torch.no_grad()
def validate_epoch(model, loader, device, pad_idx, cached=False, amp_dtype=None):
    """Token-averaged cross-entropy *without* label smoothing (=> perplexity)."""
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total, n_tok = 0.0, 0
    for batch in tqdm(loader, desc="Validating", leave=False):
        logits, targets = _forward(model, batch, device, cached, amp_dtype)
        total += ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
        n_tok += int(targets.ne(pad_idx).sum())
    return total / max(n_tok, 1)


def train_model(model, train_loader, val_loader, vocab, num_epochs=20, learning_rate=3e-4,
                device="cuda", weight_decay=0.01, label_smoothing=0.1, warmup_ratio=0.05,
                cached=False, val_images=None, out_dir="checkpoints", patience=5,
                amp=True, data_meta=None, bleu_limit=0, val_feature_bank=None):
    """Train and keep the checkpoint with the best validation BLEU-4
    (falls back to validation loss when ``val_images`` is not given)."""
    device = torch.device(device)
    model = model.to(device)
    os.makedirs(out_dir, exist_ok=True)
    pad_idx = vocab.pad_idx
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.param_groups(learning_rate, weight_decay), lr=learning_rate)
    total_steps = num_epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, int(warmup_ratio * total_steps), total_steps)

    amp_dtype, scaler = None, None
    if amp and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype, scaler = torch.float16, torch.amp.GradScaler("cuda")

    history = {k: [] for k in ("train_loss", "val_loss", "val_perplexity", "val_bleu4",
                               "lr", "grad_norm", "epoch_time_s")}
    best, bad_epochs = -float("inf"), 0
    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()
        train_loss, gnorm = train_epoch(model, train_loader, criterion, optimizer, scheduler,
                                        device, cached, amp_dtype, scaler)
        val_loss = validate_epoch(model, val_loader, device, pad_idx, cached, amp_dtype)
        bleu4 = None
        if val_images is not None:
            m, _ = evaluate_captions(model, val_images, vocab, device, beam_size=1, limit=bleu_limit,
                                     num_workers=0, feature_bank=val_feature_bank)
            bleu4 = m["BLEU-4"]
        dt = time.perf_counter() - t0
        for k, v in zip(history, (train_loss, val_loss, math.exp(min(val_loss, 20)), bleu4,
                                  scheduler.get_last_lr()[0], gnorm, dt)):
            history[k].append(v)
        msg = f"Epoch {epoch:02d}/{num_epochs} | train {train_loss:.3f} | val {val_loss:.3f} " \
              f"(ppl {math.exp(min(val_loss, 20)):.1f})"
        if bleu4 is not None:
            msg += f" | val BLEU-4 {bleu4:.4f}"
        print(msg + f" | {dt:.0f}s")

        score = bleu4 if bleu4 is not None else -val_loss
        if score > best:
            best, bad_epochs = score, 0
            save_checkpoint(os.path.join(out_dir, "best_model.pth"), model, vocab, epoch=epoch,
                            val_loss=val_loss, val_bleu4=bleu4, data=data_meta or {})
            print("  ✓ saved new best model")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement for {patience}).")
                break
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
    return history


def main():
    ap = argparse.ArgumentParser(description="Train the image-captioning model")
    ap.add_argument("--img-dir", default="data/archive/Images")
    ap.add_argument("--captions", default="data/archive/captions.txt")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--embed-size", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freq-threshold", type=int, default=5)
    ap.add_argument("--fine-tune-cnn", action="store_true", help="also train ResNet layer4")
    ap.add_argument("--no-cache", action="store_true", help="disable CNN feature caching")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="random-init ResNet (smoke tests only)")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bleu-limit", type=int, default=0, help="val images used for BLEU (0 = all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached = not args.no_cache and not args.fine_tune_cnn
    datasets, vocab = build_datasets(args.img_dir, args.captions, args.freq_threshold,
                                     seed=args.seed, augment=not cached)
    print(f"Device {device} | vocab {len(vocab):,} | train pairs {len(datasets['train']):,} "
          f"| val images {len(datasets['val_images']):,} | feature cache: {cached}")

    model = ImageCaptioningModel(args.embed_size, len(vocab), args.layers, args.heads,
                                 train_CNN=args.fine_tune_cnn, dropout=args.dropout,
                                 pad_idx=vocab.pad_idx,
                                 pretrained=not args.no_pretrained).to(device)
    pad = vocab.pad_idx
    if cached:
        feats_by_split = {split: precompute_features(model, datasets[f"{split}_images"], device,
                                                     num_workers=args.workers)
                          for split in ("train", "val")}
        loaders = []
        for split, shuffle in (("train", True), ("val", False)):
            feats, index = feats_by_split[split]
            ds = CachedFeatureDataset(feats, index, datasets[split])
            loaders.append(DataLoader(ds, args.batch_size, shuffle=shuffle,
                                      collate_fn=MyCollate(pad), pin_memory=device.type == "cuda"))
        train_loader, val_loader = loaders
    else:
        train_loader = make_loader(datasets["train"], args.batch_size, True, args.workers, pad)
        val_loader = make_loader(datasets["val"], args.batch_size, False, args.workers, pad)

    data_meta = {"freq_threshold": args.freq_threshold, "seed": args.seed}
    train_model(model, train_loader, val_loader, vocab, args.epochs, args.lr, device,
                args.weight_decay, args.label_smoothing, cached=cached,
                val_images=datasets["val_images"], out_dir=args.out_dir, patience=args.patience,
                amp=not args.no_amp, data_meta=data_meta, bleu_limit=args.bleu_limit,
                val_feature_bank=feats_by_split["val"] if cached else None)


if __name__ == "__main__":
    main()

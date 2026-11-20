"""Training entry point.

    python -m src.train                                    # defaults (configs/default.yaml values)
    python -m src.train --config configs/default.yaml --set train.epochs=30 train.lr=1e-4
    python -m src.train --set train.resume=checkpoints/last.pth   # continue an interrupted run

Pipeline: seed -> datasets (leak-free split, vocab from train only) -> model ->
optional CNN-feature cache -> AdamW + warm-up/cosine -> epochs of
[train (AMP, grad accumulation, clipping) -> val loss -> val BLEU-4/CIDEr ->
checkpoint best + last -> early stopping].

History of fixes (details in docs/IMPROVEMENTS.pdf):
* round 1: the original script had no entry point; added feature caching,
  AdamW + warm-up/cosine, label smoothing, AMP, BLEU-based model selection;
* round 3: config-driven, exact resume (optimizer / scheduler / scaler / RNG /
  shuffle order), gradient accumulation, no per-step GPU->CPU syncs, file logging.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, load_config
from .dataset import (CachedFeatureDataset, ImageDataset, Vocabulary, build_datasets_from_config,
                      make_loader)
from .evaluate import evaluate_captions
from .model import ImageCaptioningModel
from .utils import (count_parameters, get_device, get_logger, load_training_state,
                    save_checkpoint, save_training_state, set_seed)

FeatureBank = Tuple[torch.Tensor, Dict[str, int]]
HISTORY_KEYS = ("train_loss", "val_loss", "val_perplexity", "val_bleu4", "val_cider", "lr",
                "grad_norm", "epoch_time_s")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int,
                  min_ratio: float = 0.05) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to the base LR, then cosine decay to ``min_ratio`` x base LR."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def setup_amp(device: torch.device, enabled: bool
              ) -> Tuple[Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    """bf16 autocast when supported (no scaler needed), else fp16 + GradScaler.
    AMP is only used on CUDA."""
    if not enabled or device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


@torch.no_grad()
def precompute_features(model: ImageCaptioningModel, image_dataset: ImageDataset,
                        device: Any, batch_size: int = 64, num_workers: int = 2) -> FeatureBank:
    """Run the frozen backbone once per image -> fp16 tensor (N, 49, 2048) + name index."""
    model.eval()
    loader = DataLoader(image_dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=torch.device(device).type == "cuda")
    feats = []
    for imgs, _ in tqdm(loader, desc="Caching CNN features", leave=False):
        feats.append(model.encoder.backbone(imgs.to(device, non_blocking=True)).half().cpu())
    return torch.cat(feats), {name: i for i, name in enumerate(image_dataset.images)}


def _forward(model: ImageCaptioningModel, batch: Tuple[torch.Tensor, torch.Tensor],
             device: torch.device, cached: bool, amp_dtype: Optional[torch.dtype]
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced forward. Returns fp32 logits (B, T-1, V) and targets (B, T-1)."""
    x, captions = batch
    x = x.to(device, non_blocking=True)
    captions = captions.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype or torch.float32,
                        enabled=amp_dtype is not None):
        if cached:
            logits = model(captions=captions, backbone_features=x)
        else:
            logits = model(x, captions)
    # Raw logits go to CrossEntropyLoss (which applies log-softmax itself);
    # the loss is computed in fp32 for numerical stability under AMP.
    return logits.float(), captions[:, 1:]


def train_epoch(model: ImageCaptioningModel, loader: DataLoader, criterion: nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device,
                cached: bool = False, amp_dtype: Optional[torch.dtype] = None,
                scaler: Optional[torch.amp.GradScaler] = None, max_grad_norm: float = 1.0,
                grad_accum_steps: int = 1) -> Dict[str, float]:
    """One epoch. Returns token-averaged training loss and mean gradient norm.

    Metrics are accumulated as *detached device tensors* and read once at the
    end, avoiding a GPU->CPU synchronisation on every step.
    """
    model.train()
    n_batches = len(loader)
    loss_sum = torch.zeros((), device=device)
    tok_sum = torch.zeros((), device=device)
    gnorm_sum = torch.zeros((), device=device)
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        logits, targets = _forward(model, batch, device, cached, amp_dtype)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        # Scale by the size of *this* accumulation group (the last one may be short).
        group_start = (i // grad_accum_steps) * grad_accum_steps
        group_size = min(grad_accum_steps, n_batches - group_start)
        scaled = loss / group_size
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        tokens = targets.ne(criterion.ignore_index).sum()
        loss_sum += loss.detach() * tokens
        tok_sum += tokens

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == n_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)              # clip the *true* gradients
            gnorm_sum += nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).detach()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            n_steps += 1
    return {"loss": (loss_sum / tok_sum.clamp(min=1)).item(),
            "grad_norm": (gnorm_sum / max(n_steps, 1)).item()}


@torch.no_grad()
def validate_epoch(model: ImageCaptioningModel, loader: DataLoader, device: torch.device,
                   pad_idx: int, cached: bool = False,
                   amp_dtype: Optional[torch.dtype] = None) -> float:
    """Token-averaged cross-entropy *without* label smoothing (exp() = perplexity)."""
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total = torch.zeros((), device=device)
    n_tok = torch.zeros((), device=device)
    for batch in tqdm(loader, desc="Validating", leave=False):
        logits, targets = _forward(model, batch, device, cached, amp_dtype)
        total += ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        n_tok += targets.ne(pad_idx).sum()
    return (total / n_tok.clamp(min=1)).item()


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def fit(cfg: Config, model: ImageCaptioningModel, train_loader: DataLoader, val_loader: DataLoader,
        vocab: Vocabulary, device: Any, cached: bool = False,
        val_images: Optional[ImageDataset] = None, val_feature_bank: Optional[FeatureBank] = None,
        logger=None) -> Dict[str, List[Any]]:
    """Train according to ``cfg.train``; keeps ``best_model.pth`` (by
    ``select_metric``) and a resumable ``last.pth`` in ``cfg.train.out_dir``."""
    tc = cfg.train
    device = torch.device(device)
    logger = logger or get_logger()
    model = model.to(device)
    os.makedirs(tc.out_dir, exist_ok=True)
    cfg.save(os.path.join(tc.out_dir, "config.yaml"))

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=tc.label_smoothing)
    optimizer = torch.optim.AdamW(model.param_groups(tc.lr, tc.weight_decay, tc.encoder_lr_mult),
                                  lr=tc.lr)
    steps_per_epoch = math.ceil(len(train_loader) / tc.grad_accum_steps)
    total_steps = tc.epochs * steps_per_epoch
    scheduler = warmup_cosine(optimizer, int(tc.warmup_ratio * total_steps), total_steps)
    amp_dtype, scaler = setup_amp(device, tc.amp)

    history: Dict[str, List[Any]] = {k: [] for k in HISTORY_KEYS}
    start_epoch, best, bad_epochs = 1, -float("inf"), 0
    if tc.resume:
        meta = load_training_state(tc.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = int(meta["epoch"]) + 1
        history = meta.get("history", history)
        best, bad_epochs = float(meta["best_score"]), int(meta["bad_epochs"])
        logger.info(f"Resumed from {tc.resume} at epoch {start_epoch}")

    use_bleu = val_images is not None and tc.select_metric in ("BLEU-4", "CIDEr")
    last_epoch = tc.epochs
    if tc.stop_after_epochs:
        last_epoch = min(tc.epochs, start_epoch - 1 + tc.stop_after_epochs)
    for epoch in range(start_epoch, last_epoch + 1):
        t0 = time.perf_counter()
        # Reseed the shuffle generator per epoch: identical batch order whether a
        # run is uninterrupted or resumed.
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(cfg.seed + epoch)
        stats = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, cached,
                            amp_dtype, scaler, tc.max_grad_norm, tc.grad_accum_steps)
        val_loss = validate_epoch(model, val_loader, device, vocab.pad_idx, cached, amp_dtype)
        bleu4 = cider = None
        if use_bleu:
            m, _ = evaluate_captions(model, val_images, vocab, device, beam_size=1,
                                     max_length=cfg.eval.max_length, limit=tc.bleu_limit,
                                     num_workers=0, feature_bank=val_feature_bank)
            bleu4, cider = m["BLEU-4"], m["CIDEr"]
        dt = time.perf_counter() - t0
        ppl = math.exp(min(val_loss, 20))
        for k, v in zip(HISTORY_KEYS, (stats["loss"], val_loss, ppl, bleu4, cider,
                                       scheduler.get_last_lr()[0], stats["grad_norm"], dt)):
            history[k].append(v)
        msg = (f"Epoch {epoch:02d}/{tc.epochs} | train {stats['loss']:.3f} | "
               f"val {val_loss:.3f} (ppl {ppl:.1f})")
        if bleu4 is not None:
            msg += f" | BLEU-4 {bleu4:.4f} | CIDEr {cider:.3f}"
        logger.info(msg + f" | {dt:.0f}s")

        score = {"BLEU-4": bleu4, "CIDEr": cider}.get(tc.select_metric) if use_bleu else -val_loss
        meta = dict(epoch=epoch, val_loss=val_loss, val_bleu4=bleu4, val_cider=cider,
                    config=cfg.to_dict())
        if score > best:
            best, bad_epochs = score, 0
            save_checkpoint(os.path.join(tc.out_dir, "best_model.pth"), model, vocab, **meta)
            logger.info(f"  saved new best model ({tc.select_metric if use_bleu else 'val_loss'})")
        else:
            bad_epochs += 1
        save_training_state(os.path.join(tc.out_dir, "last.pth"), model, vocab, optimizer,
                            scheduler, scaler, history=history, best_score=best,
                            bad_epochs=bad_epochs, **meta)
        with open(os.path.join(tc.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        if bad_epochs >= tc.patience:
            logger.info(f"Early stopping after {epoch} epochs (no improvement for {tc.patience}).")
            break
    return history


def build_training_loaders(cfg: Config, model: ImageCaptioningModel, device: torch.device,
                           datasets: Dict[str, Any], vocab: Vocabulary
                           ) -> Tuple[DataLoader, DataLoader, bool, Optional[FeatureBank]]:
    """Cached-feature loaders when the CNN is frozen (and caching is on), else image loaders."""
    tc, pad = cfg.train, vocab.pad_idx
    cached = tc.cache_features and not cfg.model.train_CNN
    if not cached:
        return (make_loader(datasets["train"], tc.batch_size, True, cfg.data.num_workers, pad,
                            seed=cfg.seed),
                make_loader(datasets["val"], tc.batch_size, False, cfg.data.num_workers, pad),
                False, None)
    banks = {s: precompute_features(model, datasets[f"{s}_images"], device,
                                    num_workers=cfg.data.num_workers) for s in ("train", "val")}
    # In-memory tensors: worker processes would only add IPC overhead.
    train_loader = make_loader(CachedFeatureDataset(*banks["train"], datasets["train"]),
                               tc.batch_size, True, 0, pad, seed=cfg.seed)
    val_loader = make_loader(CachedFeatureDataset(*banks["val"], datasets["val"]),
                             tc.batch_size, False, 0, pad)
    return train_loader, val_loader, True, banks["val"]


def run(cfg: Config) -> Dict[str, List[Any]]:
    """Full pipeline from a config."""
    logger = get_logger(log_file=os.path.join(cfg.train.out_dir, "train.log"))
    set_seed(cfg.seed, cfg.deterministic)
    device = get_device()
    cached = cfg.train.cache_features and not cfg.model.train_CNN
    datasets, vocab = build_datasets_from_config(cfg.data, cfg.seed,
                                                 augment=cfg.train.augment and not cached)
    model = ImageCaptioningModel.from_config(cfg.model, len(vocab), vocab.pad_idx).to(device)
    total, trainable = count_parameters(model)
    logger.info(f"Device {device} | vocab {len(vocab):,} | train pairs {len(datasets['train']):,} "
                f"| val images {len(datasets['val_images']):,} | params {total / 1e6:.1f}M "
                f"({trainable / 1e6:.1f}M trainable) | feature cache {cached}")
    train_loader, val_loader, cached, bank = build_training_loaders(cfg, model, device,
                                                                    datasets, vocab)
    return fit(cfg, model, train_loader, val_loader, vocab, device, cached,
               val_images=datasets["val_images"], val_feature_bank=bank, logger=logger)


def train_model(model: ImageCaptioningModel, train_loader: DataLoader, val_loader: DataLoader,
                vocab: Vocabulary, num_epochs: int = 20, learning_rate: float = 3e-4,
                device: Any = "cuda", weight_decay: float = 0.01, label_smoothing: float = 0.1,
                warmup_ratio: float = 0.05, cached: bool = False,
                val_images: Optional[ImageDataset] = None, out_dir: str = "checkpoints",
                patience: int = 5, amp: bool = True, bleu_limit: int = 0,
                val_feature_bank: Optional[FeatureBank] = None, **_: Any) -> Dict[str, List[Any]]:
    """Backwards-compatible keyword API (used by the notebook) around :func:`fit`."""
    cfg = Config()
    tc = cfg.train
    tc.epochs, tc.lr, tc.weight_decay = num_epochs, learning_rate, weight_decay
    tc.label_smoothing, tc.warmup_ratio, tc.out_dir = label_smoothing, warmup_ratio, out_dir
    tc.patience, tc.amp, tc.bleu_limit = patience, amp, bleu_limit
    return fit(cfg, model, train_loader, val_loader, vocab, device, cached, val_images,
               val_feature_bank)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the image-captioning model")
    ap.add_argument("--config", default=None, help="YAML config (default: built-in defaults)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="overrides, e.g. train.lr=1e-4 model.train_CNN=true")
    args = ap.parse_args()
    run(load_config(args.config, args.set))


if __name__ == "__main__":
    main()

"""Training entry point.

    python -m src.train                                         # defaults
    python src/train.py --config configs/default.yaml --lr 1e-4 --batch_size 32
    python src/train.py --resume checkpoints/last.pth           # continue an interrupted run
    python src/train.py --backend tensorboard                   # or: --backend wandb

Every key in ``configs/default.yaml`` is a flag (``--train.lr`` or, when
unambiguous, ``--lr``); ``--set train.lr=1e-4`` is also accepted.

Pipeline: seed -> device (cuda > mps > cpu) -> datasets (leak-free split, vocab
from train only, broken images dropped) -> model -> optional CNN-feature cache
-> AdamW + linear warm-up / cosine annealing -> epochs of [train (AMP with
GradScaler, gradient accumulation, clipping, automatic micro-batching on CUDA
OOM) -> val loss -> val BLEU-4 / CIDEr -> best + last checkpoints -> early
stopping], with logs to the console, a timestamped file and optionally
TensorBoard / W&B.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):  # executed as `python src/train.py`: make `src` importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")  # register the package for the relative imports below

import argparse
import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, add_config_arguments, config_from_args, load_config
from .dataset import (CachedFeatureDataset, ImageDataset, Vocabulary, build_datasets_from_config,
                      make_loader)
from .evaluate import evaluate_captions
from .model import ImageCaptioningModel
from .tracking import ExperimentTracker
from .utils import (count_parameters, free_device_memory, get_device, get_logger, is_oom_error,
                    load_training_state, save_checkpoint, save_training_state, set_seed,
                    timestamp, timestamped_log_path)

FeatureBank = Tuple[torch.Tensor, Dict[str, int]]
HISTORY_KEYS = ("train_loss", "val_loss", "val_perplexity", "val_bleu4", "val_cider", "lr",
                "grad_norm", "epoch_time_s")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int,
                  min_ratio: float = 0.05) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up to the base LR, then cosine annealing to ``min_ratio`` x base LR.

    Stepped once per *optimizer* step (not per batch, not per epoch)."""
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def setup_amp(device: torch.device, enabled: bool
              ) -> Tuple[Optional[torch.dtype], Optional[torch.amp.GradScaler]]:
    """Mixed precision on CUDA: bf16 autocast when supported (no scaler needed,
    same exponent range as fp32), otherwise fp16 autocast + ``GradScaler``.
    Disabled on CPU / MPS."""
    if not enabled or device.type != "cuda":
        return None, None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    return torch.float16, torch.amp.GradScaler("cuda")


@torch.no_grad()
def precompute_features(model: ImageCaptioningModel, image_dataset: ImageDataset,
                        device: Any, batch_size: int = 64, num_workers: int = 2) -> FeatureBank:
    """Run the frozen backbone once per image -> fp16 tensor (N, S, 2048) + name index.

    On CUDA OOM a batch is split in halves recursively."""
    model.eval()
    device = torch.device(device)
    loader = DataLoader(image_dataset, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=device.type == "cuda")

    def run(imgs: torch.Tensor) -> torch.Tensor:
        try:
            return model.encoder.backbone(imgs.to(device, non_blocking=True)).half().cpu()
        except Exception as exc:
            if not is_oom_error(exc) or imgs.size(0) == 1:
                raise
            free_device_memory()
            half = imgs.size(0) // 2
            return torch.cat([run(imgs[:half]), run(imgs[half:])])

    feats = [run(imgs) for imgs, _ in tqdm(loader, desc="Caching CNN features", leave=False)]
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


def _backward_batch(model: ImageCaptioningModel, batch: Tuple[torch.Tensor, torch.Tensor],
                    criterion: nn.CrossEntropyLoss, device: torch.device, cached: bool,
                    amp_dtype: Optional[torch.dtype], scaler: Optional[torch.amp.GradScaler],
                    scale: float, n_chunks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward + backward of one batch, optionally as ``n_chunks`` micro-batches.

    Each chunk's mean loss is weighted by its share of the batch's target
    tokens, so the accumulated gradient equals the full-batch gradient.
    Returns ``(sum of token losses, number of tokens)`` as detached tensors."""
    x, captions = batch
    total_tokens = captions[:, 1:].ne(criterion.ignore_index).sum().clamp(min=1)
    loss_sum = torch.zeros((), device=device)
    tok_sum = torch.zeros((), device=device)
    for xs, cs in zip(x.tensor_split(n_chunks), captions.tensor_split(n_chunks)):
        if cs.size(0) == 0:
            continue
        logits, targets = _forward(model, (xs, cs), device, cached, amp_dtype)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        tokens = targets.ne(criterion.ignore_index).sum()
        weighted = loss * (tokens / total_tokens.to(device)) * scale
        if scaler is not None:
            scaler.scale(weighted).backward()
        else:
            weighted.backward()
        loss_sum += loss.detach() * tokens
        tok_sum += tokens
    return loss_sum, tok_sum


def train_epoch(model: ImageCaptioningModel, loader: DataLoader, criterion: nn.CrossEntropyLoss,
                optimizer: torch.optim.Optimizer, scheduler: Any, device: torch.device,
                cached: bool = False, amp_dtype: Optional[torch.dtype] = None,
                scaler: Optional[torch.amp.GradScaler] = None, max_grad_norm: float = 1.0,
                grad_accum_steps: int = 1, max_oom_splits: int = 8,
                oom_state: Optional[Dict[str, int]] = None,
                tracker: Optional[ExperimentTracker] = None, global_step: int = 0,
                log_every: int = 50) -> Dict[str, float]:
    """One epoch with gradient accumulation, clipping and OOM recovery.

    Returns ``{"loss", "grad_norm", "steps"}``: token-averaged training loss,
    mean pre-clipping gradient norm and the number of optimizer steps.
    Metrics stay on the device and are read once at the end (no per-step
    GPU->CPU sync), except every ``log_every`` steps when a tracker is active.

    CUDA OOM: gradients of the current accumulation group are discarded, the
    cache is freed and the batch is retried as 2x more micro-batches (up to
    ``max_oom_splits``); the split factor is remembered in ``oom_state`` for
    the rest of training.
    """
    model.train()
    log = get_logger()
    oom_state = oom_state if oom_state is not None else {"chunks": 1}
    n_batches = len(loader)
    loss_sum = torch.zeros((), device=device)
    tok_sum = torch.zeros((), device=device)
    gnorm_sum = torch.zeros((), device=device)
    n_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        # Scale by the size of *this* accumulation group (the last one may be short).
        group_start = (i // grad_accum_steps) * grad_accum_steps
        group_size = min(grad_accum_steps, n_batches - group_start)
        while True:
            try:
                b_loss, b_tok = _backward_batch(model, batch, criterion, device, cached,
                                                amp_dtype, scaler, 1.0 / group_size,
                                                oom_state["chunks"])
                break
            except Exception as exc:
                if not is_oom_error(exc) or oom_state["chunks"] * 2 > max_oom_splits \
                        or batch[1].size(0) < oom_state["chunks"] * 2:
                    raise
                optimizer.zero_grad(set_to_none=True)
                free_device_memory()
                oom_state["chunks"] *= 2
                lost = i - group_start
                log.warning(f"Out of memory at batch {i}: retrying as {oom_state['chunks']} "
                            f"micro-batches" + (f" ({lost} earlier batch(es) of this "
                                                f"accumulation group discarded)" if lost else ""))
        loss_sum += b_loss
        tok_sum += b_tok

        if (i + 1) % grad_accum_steps == 0 or (i + 1) == n_batches:
            if scaler is not None:
                scaler.unscale_(optimizer)              # clip the *true* gradients
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).detach()
            gnorm_sum += gnorm
            if scaler is not None:
                scaler.step(optimizer)                  # skipped automatically on inf/NaN
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            n_steps += 1
            step = global_step + n_steps
            if tracker is not None and tracker.enabled and step % log_every == 0:
                tracker.log_metrics({"train/step_loss": (b_loss / b_tok.clamp(min=1)).item(),
                                     "train/lr": scheduler.get_last_lr()[0],
                                     "train/grad_norm": gnorm.item()}, step)
    return {"loss": (loss_sum / tok_sum.clamp(min=1)).item(),
            "grad_norm": (gnorm_sum / max(n_steps, 1)).item(),
            "steps": float(n_steps)}


@torch.no_grad()
def validate_epoch(model: ImageCaptioningModel, loader: DataLoader, device: torch.device,
                   pad_idx: int, cached: bool = False,
                   amp_dtype: Optional[torch.dtype] = None) -> float:
    """Token-averaged cross-entropy *without* label smoothing (exp() = perplexity).

    On CUDA OOM a batch is evaluated in halves recursively."""
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")
    total = torch.zeros((), device=device)
    n_tok = torch.zeros((), device=device)

    def run(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            logits, targets = _forward(model, batch, device, cached, amp_dtype)
            return (ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)),
                    targets.ne(pad_idx).sum())
        except Exception as exc:
            if not is_oom_error(exc) or batch[1].size(0) == 1:
                raise
            free_device_memory()
            half = batch[1].size(0) // 2
            a = run((batch[0][:half], batch[1][:half]))
            b = run((batch[0][half:], batch[1][half:]))
            return a[0] + b[0], a[1] + b[1]

    for batch in tqdm(loader, desc="Validating", leave=False):
        loss, tokens = run(batch)
        total += loss
        n_tok += tokens
    return (total / n_tok.clamp(min=1)).item()


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def fit(cfg: Config, model: ImageCaptioningModel, train_loader: DataLoader, val_loader: DataLoader,
        vocab: Vocabulary, device: Any, cached: bool = False,
        val_images: Optional[ImageDataset] = None, val_feature_bank: Optional[FeatureBank] = None,
        logger=None, tracker: Optional[ExperimentTracker] = None) -> Dict[str, List[Any]]:
    """Train according to ``cfg.train``; keeps ``best_model.pth`` (by
    ``select_metric``) and a resumable ``last.pth`` in ``cfg.train.out_dir``.

    Resuming (``cfg.train.resume``) restores weights, optimizer, scheduler, AMP
    scaler, RNG states, history, best metric, early-stopping counter, global
    step and the exact epoch."""
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
    scheduler = warmup_cosine(optimizer, int(tc.warmup_ratio * total_steps), total_steps,
                              tc.min_lr_ratio)
    amp_dtype, scaler = setup_amp(device, tc.amp)
    if amp_dtype is not None:
        logger.info(f"Mixed precision: {amp_dtype}" + (" + GradScaler" if scaler else ""))

    history: Dict[str, List[Any]] = {k: [] for k in HISTORY_KEYS}
    start_epoch, best, bad_epochs, global_step = 1, -float("inf"), 0, 0
    if tc.resume:
        meta = load_training_state(tc.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = int(meta["epoch"]) + 1
        history = meta.get("history", history)
        best, bad_epochs = float(meta["best_score"]), int(meta["bad_epochs"])
        global_step = int(meta.get("global_step", (start_epoch - 1) * steps_per_epoch))
        logger.info(f"Resumed from {tc.resume}: epoch {start_epoch}, best {tc.select_metric} "
                    f"{best:.4f}, {bad_epochs} epoch(s) without improvement")
        if start_epoch > tc.epochs:
            logger.info("Checkpoint already reached train.epochs; nothing to do.")

    use_bleu = val_images is not None and tc.select_metric in ("BLEU-4", "CIDEr")
    metric_name = tc.select_metric if use_bleu else "val_loss"
    last_epoch = tc.epochs
    if tc.stop_after_epochs:
        last_epoch = min(tc.epochs, start_epoch - 1 + tc.stop_after_epochs)
    oom_state = {"chunks": 1}
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, last_epoch + 1):
            t0 = time.perf_counter()
            # Reseed the shuffle generator per epoch: identical batch order whether a
            # run is uninterrupted or resumed.
            if getattr(train_loader, "generator", None) is not None:
                train_loader.generator.manual_seed(cfg.seed + epoch)
            stats = train_epoch(model, train_loader, criterion, optimizer, scheduler, device,
                                cached, amp_dtype, scaler, tc.max_grad_norm, tc.grad_accum_steps,
                                tc.max_oom_splits, oom_state, tracker, global_step,
                                cfg.tracking.log_every_n_steps)
            global_step += int(stats["steps"])
            val_loss = validate_epoch(model, val_loader, device, vocab.pad_idx, cached, amp_dtype)
            bleu4 = cider = None
            samples: List[Dict[str, Any]] = []
            if use_bleu:
                m, samples = evaluate_captions(model, val_images, vocab, device, beam_size=1,
                                               max_length=cfg.eval.max_length,
                                               limit=tc.bleu_limit, num_workers=0,
                                               feature_bank=val_feature_bank)
                bleu4, cider = m["BLEU-4"], m["CIDEr"]
            dt = time.perf_counter() - t0
            ppl = math.exp(min(val_loss, 20))
            lr = scheduler.get_last_lr()[0]
            for k, v in zip(HISTORY_KEYS, (stats["loss"], val_loss, ppl, bleu4, cider, lr,
                                           stats["grad_norm"], dt)):
                history[k].append(v)
            msg = (f"Epoch {epoch:02d}/{tc.epochs} | train {stats['loss']:.3f} | "
                   f"val {val_loss:.3f} (ppl {ppl:.1f})")
            if bleu4 is not None:
                msg += f" | BLEU-4 {bleu4:.4f} | CIDEr {cider:.3f}"
            logger.info(msg + f" | lr {lr:.2e} | {dt:.0f}s")
            if tracker is not None:
                tracker.log_metrics({"epoch": epoch, "train/loss": stats["loss"],
                                     "train/grad_norm": stats["grad_norm"], "val/loss": val_loss,
                                     "val/perplexity": ppl, "val/BLEU-4": bleu4,
                                     "val/CIDEr": cider, "lr": lr, "time/epoch_s": dt},
                                    global_step)
                if samples:
                    tracker.log_text("val/samples", "\n".join(
                        f"{s['image']}: {s['prediction']}" for s in samples[:5]), global_step)

            score = {"BLEU-4": bleu4, "CIDEr": cider}.get(tc.select_metric) if use_bleu \
                else -val_loss
            meta = dict(epoch=epoch, val_loss=val_loss, val_bleu4=bleu4, val_cider=cider,
                        select_metric=metric_name, config=cfg.to_dict())
            if score > best + tc.min_delta:
                best, bad_epochs = score, 0
                save_checkpoint(os.path.join(tc.out_dir, "best_model.pth"), model, vocab, **meta)
                logger.info(f"  saved new best model ({metric_name} {abs(score):.4f})")
            else:
                bad_epochs += 1
            save_training_state(os.path.join(tc.out_dir, "last.pth"), model, vocab, optimizer,
                                scheduler, scaler, history=history, best_score=best,
                                bad_epochs=bad_epochs, global_step=global_step, **meta)
            with open(os.path.join(tc.out_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            if bad_epochs >= tc.patience:
                logger.info(f"Early stopping after {epoch} epochs (no {metric_name} improvement "
                            f"> {tc.min_delta} for {tc.patience} epochs).")
                break
    except KeyboardInterrupt:
        logger.warning(f"Interrupted during epoch {epoch + 1}. Resume from the last completed "
                       f"epoch with: --resume {os.path.join(tc.out_dir, 'last.pth')}")
        raise
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
    """Full pipeline from a config: logs to console + ``<out_dir>/logs/train_<time>.log``."""
    # The run name is written into <out_dir>/config.yaml, so resuming with that
    # config continues the same TensorBoard / W&B run instead of starting a new one.
    cfg.tracking.run_name = cfg.tracking.run_name or timestamp()
    run_name = cfg.tracking.run_name
    logger = get_logger(log_file=timestamped_log_path(cfg.train.out_dir, "train"))
    set_seed(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
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
    tracker = ExperimentTracker(cfg.tracking, cfg.to_dict(), cfg.train.out_dir, logger, run_name)
    try:
        return fit(cfg, model, train_loader, val_loader, vocab, device, cached,
                   val_images=datasets["val_images"], val_feature_bank=bank, logger=logger,
                   tracker=tracker)
    finally:
        tracker.finish()


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
    return fit(cfg, model, train_loader, val_loader, vocab, get_device(str(device)), cached,
               val_images, val_feature_bank)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point; exits with a readable message on configuration / data errors."""
    parser = add_config_arguments(argparse.ArgumentParser(
        description="Train the image-captioning model. Every config key is a flag, e.g. "
                    "--lr 1e-4 --batch_size 32 --resume checkpoints/last.pth"))
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    try:
        run(cfg)
    except (FileNotFoundError, ValueError) as exc:
        get_logger().error(str(exc))
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()


__all__ = ["warmup_cosine", "setup_amp", "precompute_features", "train_epoch", "validate_epoch",
           "fit", "build_training_loaders", "run", "train_model", "main", "load_config"]

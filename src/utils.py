"""Reproducibility, device, logging and checkpoint utilities.

Checkpoints come in two flavours:

* ``best_model.pth`` -- *inference* checkpoint: weights, model config,
  vocabulary, resolved experiment config and metrics. Everything ``app.py``,
  ``src.evaluate`` and ``src.predict`` need, nothing more.
* ``last.pth`` -- *training-state* checkpoint written every epoch: the above
  plus optimizer, LR scheduler, AMP scaler, epoch, history, early-stopping
  state and all RNG states, so ``--set train.resume=checkpoints/last.pth``
  continues a run exactly where it stopped.

Both are plain dicts of tensors / numbers / strings / lists and are loaded with
``torch.load(weights_only=True)``, i.e. without executing pickled code (the
previous loader used ``weights_only=False``, which can run arbitrary code from
a malicious checkpoint file).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .dataset import Vocabulary, _seed_worker
from .model import ImageCaptioningModel


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs).

    ``deterministic=True`` additionally forces deterministic cuDNN / PyTorch
    kernels (bit-exact reruns on the same hardware, at some speed cost).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: distinct, reproducible NumPy / ``random``
    seeds per worker (``make_loader`` already installs it)."""
    _seed_worker(worker_id)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling (makes batch order reproducible)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_rng_state() -> Dict[str, Any]:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# --------------------------------------------------------------------------- #
# Device and logging
# --------------------------------------------------------------------------- #
def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(name: str = "captioning", log_file: Optional[str] = None) -> logging.Logger:
    """Console (+ optional file) logger with timestamps. Idempotent per handler:
    calling it again with a new ``log_file`` adds that file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file:
        path = os.path.abspath(log_file)
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == path
                   for h in logger.handlers):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = logging.FileHandler(path)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Returns ``(total, trainable)`` parameter counts (shared tensors once)."""
    seen, total, trainable = set(), 0, 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return total, trainable


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: ImageCaptioningModel, vocab: Vocabulary,
                    **meta: Any) -> None:
    """Inference checkpoint: weights + model config + vocabulary + metadata.
    Written atomically (temp file + rename) so a crash never leaves a corrupt file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model_state": model.state_dict(), "model_config": dict(model.config),
                "vocab": vocab.to_dict(), **meta}, tmp)
    os.replace(tmp, path)


def save_training_state(path: str, model: ImageCaptioningModel, vocab: Vocabulary,
                        optimizer: torch.optim.Optimizer, scheduler: Any,
                        scaler: Optional[torch.amp.GradScaler], **meta: Any) -> None:
    """Resumable checkpoint: inference checkpoint + optimizer/scheduler/scaler/RNG."""
    save_checkpoint(path, model, vocab,
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict(),
                    scaler_state=scaler.state_dict() if scaler is not None else None,
                    rng_state=get_rng_state(), **meta)


def _load(path: str, device: Any) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model_state" not in ckpt:
        raise ValueError(f"{path} is a bare state_dict from the original training script; "
                         "re-train with `python -m src.train` to get a full checkpoint.")
    return ckpt


def load_checkpoint(path: str, device: Any = "cpu", pretrained_backbone: bool = False
                    ) -> Tuple[ImageCaptioningModel, Vocabulary, Dict[str, Any]]:
    """Load an inference-ready model.

    Returns ``(model in eval mode, vocab, meta)``. Backbone weights come from
    the checkpoint, so ImageNet weights are not downloaded again by default.
    """
    ckpt = _load(path, device)
    vocab = Vocabulary.from_dict(ckpt["vocab"])
    model = ImageCaptioningModel(**ckpt["model_config"], pretrained=pretrained_backbone)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    skip = {"model_state", "model_config", "vocab", "optimizer_state", "scheduler_state",
            "scaler_state", "rng_state"}
    return model, vocab, {k: v for k, v in ckpt.items() if k not in skip}


def load_training_state(path: str, model: ImageCaptioningModel,
                        optimizer: torch.optim.Optimizer, scheduler: Any,
                        scaler: Optional[torch.amp.GradScaler], device: Any = "cpu"
                        ) -> Dict[str, Any]:
    """Restore model/optimizer/scheduler/scaler/RNG in place; returns the metadata
    (epoch, history, best score, ...)."""
    ckpt = _load(path, device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    if ckpt.get("rng_state"):
        set_rng_state(ckpt["rng_state"])
    return {k: v for k, v in ckpt.items()
            if k not in ("model_state", "optimizer_state", "scheduler_state", "scaler_state",
                         "rng_state")}

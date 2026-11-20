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
from datetime import datetime
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
    """Python / NumPy / torch / CUDA RNG states in a ``weights_only``-safe form."""
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    """Inverse of :func:`get_rng_state`."""
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
    """Pick the compute device.

    ``prefer`` = ``None`` / ``"auto"``: CUDA if available, else Apple MPS, else CPU.
    An explicit request that cannot be honoured (e.g. ``"cuda"`` on a machine
    without a GPU, or ``"cuda:3"`` with two GPUs) logs a warning and falls back
    to the automatic choice instead of crashing later inside a kernel.
    """
    log = logging.getLogger("captioning")
    if prefer and prefer != "auto":
        want = torch.device(prefer)
        if want.type == "cuda":
            if torch.cuda.is_available() and (want.index is None
                                              or want.index < torch.cuda.device_count()):
                return want
            log.warning(f"device '{prefer}' requested but not available; falling back.")
        elif want.type == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return want
            log.warning("device 'mps' requested but not available; falling back.")
        else:
            return want
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_oom_error(exc: BaseException) -> bool:
    """True for CUDA (and MPS) out-of-memory errors."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def free_device_memory() -> None:
    """Release cached allocator blocks after an OOM so a retry can succeed."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and getattr(torch.backends, "mps", None) and \
            torch.backends.mps.is_available():
        mps.empty_cache()


def timestamp() -> str:
    """``YYYYmmdd-HHMMSS`` for run names and log files."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def timestamped_log_path(out_dir: str, prefix: str = "train") -> str:
    """``<out_dir>/logs/<prefix>_<timestamp>.log`` (one file per run)."""
    return os.path.join(out_dir, "logs", f"{prefix}_{timestamp()}.log")


def get_logger(name: str = "captioning", log_file: Optional[str] = None,
               stream: Any = None) -> logging.Logger:
    """Console (+ optional file) logger with timestamps. Idempotent per handler:
    calling it again with a new ``log_file`` adds that file. ``stream`` defaults
    to stdout (CLIs whose stdout is data, like ``src.predict``, pass stderr)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        sh = logging.StreamHandler(stream or sys.stdout)
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
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise ValueError(f"Could not read checkpoint {path} (corrupted or not a checkpoint "
                         f"written by this project): {exc}") from exc
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
    if "optimizer_state" not in ckpt:
        raise ValueError(f"{path} is an inference checkpoint (e.g. best_model.pth) without "
                         "optimizer state; resume from last.pth instead.")
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise ValueError(f"{path} does not match the current model: did model.* settings or "
                         f"data.freq_threshold (vocabulary size) change since it was saved? "
                         f"{str(exc).splitlines()[0]}") from exc
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    if scaler is not None and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    if ckpt.get("rng_state"):
        set_rng_state(ckpt["rng_state"])
    return {k: v for k, v in ckpt.items()
            if k not in ("model_state", "optimizer_state", "scheduler_state", "scaler_state",
                         "rng_state")}

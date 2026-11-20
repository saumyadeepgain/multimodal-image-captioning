"""Centralised, typed configuration.

All hyper-parameters live in dataclasses with defaults, can be loaded from a
YAML file (``configs/default.yaml``) and overridden from the command line::

    python -m src.train --config configs/default.yaml --set train.lr=1e-4 model.num_layers=4

The resolved config is stored inside every checkpoint, so a run can always be
reproduced from its checkpoint alone.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    """Dataset location, split and vocabulary settings."""
    img_dir: str = "data/archive/Images"
    captions_file: str = "data/archive/captions.txt"
    val_frac: float = 0.1
    test_frac: float = 0.1
    freq_threshold: int = 5          # min train-caption count for a word to enter the vocab
    max_len: int = 40                # max caption tokens incl. <start>/<end>
    image_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Architecture settings (must match between training and inference)."""
    embed_size: int = 512
    num_layers: int = 3
    nhead: int = 8
    dropout: float = 0.1
    max_seq_length: int = 64
    tie_weights: bool = False
    train_CNN: bool = False          # fine-tune ResNet layer4
    pretrained: bool = True          # ImageNet weights for the backbone


@dataclass
class TrainConfig:
    """Optimisation settings."""
    epochs: int = 20
    batch_size: int = 64
    grad_accum_steps: int = 1        # effective batch = batch_size * grad_accum_steps
    lr: float = 3e-4
    encoder_lr_mult: float = 0.1     # LR multiplier for fine-tuned CNN layers
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    amp: bool = True                 # bf16/fp16 autocast on CUDA
    cache_features: bool = True      # only used when the CNN is frozen
    augment: bool = True             # only used when features are NOT cached
    patience: int = 5                # early stopping (epochs without improvement)
    select_metric: str = "BLEU-4"    # "BLEU-4", "CIDEr" or "val_loss"
    bleu_limit: int = 0              # val images decoded per epoch for selection (0 = all)
    out_dir: str = "checkpoints"
    resume: Optional[str] = None     # path to last.pth to resume from
    stop_after_epochs: Optional[int] = None  # end this process after N epochs (time-limited
                                             # jobs); continue later with train.resume


@dataclass
class EvalConfig:
    """Decoding / evaluation settings."""
    beam_size: int = 3
    max_length: int = 30
    length_penalty: float = 0.7
    batch_size: int = 64
    meteor: bool = False


@dataclass
class Config:
    """Root configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    deterministic: bool = False      # cuDNN deterministic algorithms (slower)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """Build a Config from a (possibly partial) nested dict; unknown keys raise."""
        cfg = cls()
        for key, value in (d or {}).items():
            _set(cfg, key, value)
        return cfg

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _set(obj: Any, key: str, value: Any) -> None:
    names = {f.name for f in fields(obj)}
    if key not in names:
        raise KeyError(f"unknown config key '{key}' for {type(obj).__name__}; "
                       f"valid keys: {sorted(names)}")
    current = getattr(obj, key)
    if dataclasses.is_dataclass(current):
        if not isinstance(value, dict):
            raise TypeError(f"'{key}' must be a mapping")
        for k, v in value.items():
            _set(current, k, v)
    else:
        setattr(obj, key, value)


def _parse_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply ``section.key=value`` overrides, e.g. ``train.lr=1e-4``."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        target: Any = cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        _set(target, parts[-1], _parse_value(raw))
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    """Defaults <- YAML file (if given) <- command-line overrides."""
    cfg = Config()
    if path:
        with open(path) as f:
            cfg = Config.from_dict(yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides or [])

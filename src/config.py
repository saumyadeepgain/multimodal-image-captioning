"""Centralised, typed, validated configuration.

All hyper-parameters live in dataclasses with defaults. Precedence (lowest to
highest): dataclass defaults < YAML file < command-line flags < ``--set``.

Every leaf key is also a command-line flag, both fully qualified and, where
unambiguous, by its short name::

    python src/train.py --config configs/default.yaml --lr 1e-4 --batch_size 32
    python src/train.py --train.lr 1e-4 --model.num_layers 4 --amp false
    python src/train.py --set train.lr=1e-4 model.num_layers=4      # equivalent

Short names that exist in several sections resolve by section priority
train > model > data > eval > tracking (e.g. ``--batch_size`` = ``train.batch_size``;
use ``--eval.batch_size`` for the other one).

Values are type-checked against the field annotation (``--lr abc`` is an
error, ``--epochs 2.5`` is an error, ``--lr 1`` becomes ``1.0``) and the whole
config is validated (e.g. ``embed_size`` divisible by ``nhead``). The resolved
config is stored inside every checkpoint, so a run can be reproduced from its
checkpoint alone.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import typing
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

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
    verify_images: bool = True       # drop missing / unreadable images before training


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
    warmup_ratio: float = 0.05       # linear warm-up, then cosine annealing
    min_lr_ratio: float = 0.05       # cosine floor as a fraction of the base LR
    max_grad_norm: float = 1.0
    amp: bool = True                 # bf16 autocast, or fp16 + GradScaler (CUDA only)
    cache_features: bool = True      # only used when the CNN is frozen
    augment: bool = True             # only used when features are NOT cached
    patience: int = 5                # early stopping (epochs without improvement)
    min_delta: float = 0.0           # minimum improvement that resets patience
    select_metric: str = "BLEU-4"    # "BLEU-4", "CIDEr" or "val_loss"
    bleu_limit: int = 0              # val images decoded per epoch for selection (0 = all)
    max_oom_splits: int = 8          # on CUDA OOM, split a batch into up to N micro-batches
    out_dir: str = "checkpoints"
    resume: Optional[str] = None     # path to last.pth to resume from
    stop_after_epochs: Optional[int] = None  # end this process after N epochs (time-limited
                                             # jobs); continue later with --resume


@dataclass
class EvalConfig:
    """Decoding / evaluation settings."""
    beam_size: int = 3
    max_length: int = 30
    length_penalty: float = 0.7
    batch_size: int = 64
    meteor: bool = False


@dataclass
class TrackingConfig:
    """Optional experiment tracking (logs always go to console + a timestamped file)."""
    backend: str = "none"            # "none", "tensorboard" or "wandb"
    project: str = "image-captioning"
    run_name: Optional[str] = None   # default: timestamp
    log_every_n_steps: int = 50      # step-level train loss / LR / grad-norm
    wandb_mode: str = "online"       # "online", "offline" or "disabled"


@dataclass
class Config:
    """Root configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seed: int = 42
    deterministic: bool = False      # cuDNN deterministic algorithms (slower)
    device: str = "auto"             # "auto" (cuda > mps > cpu), "cuda", "cuda:1", "mps", "cpu"

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

    def validate(self) -> "Config":
        """Raise ``ValueError`` listing every inconsistent setting."""
        errors: List[str] = []
        d, m, t, e, tr = self.data, self.model, self.train, self.eval, self.tracking
        if not (0 <= d.val_frac < 1 and 0 <= d.test_frac < 1 and d.val_frac + d.test_frac < 1):
            errors.append("data.val_frac + data.test_frac must be in [0, 1)")
        if d.max_len < 3:
            errors.append("data.max_len must be >= 3 (<start>, a word, <end>)")
        if d.image_size < 32:
            errors.append("data.image_size must be >= 32 (ResNet downsamples by 32)")
        if d.num_workers < 0:
            errors.append("data.num_workers must be >= 0")
        if m.embed_size % m.nhead != 0:
            errors.append(f"model.embed_size ({m.embed_size}) must be divisible by "
                          f"model.nhead ({m.nhead})")
        if not 0 <= m.dropout < 1:
            errors.append("model.dropout must be in [0, 1)")
        if m.max_seq_length < max(d.max_len, e.max_length + 1):
            errors.append("model.max_seq_length must be >= data.max_len and > eval.max_length")
        for name in ("epochs", "batch_size", "grad_accum_steps", "patience", "max_oom_splits"):
            if getattr(t, name) < 1:
                errors.append(f"train.{name} must be >= 1")
        if t.lr <= 0:
            errors.append("train.lr must be > 0")
        if not 0 <= t.warmup_ratio < 1 or not 0 <= t.min_lr_ratio <= 1:
            errors.append("train.warmup_ratio must be in [0, 1) and train.min_lr_ratio in [0, 1]")
        if not 0 <= t.label_smoothing < 1:
            errors.append("train.label_smoothing must be in [0, 1)")
        if t.select_metric not in ("BLEU-4", "CIDEr", "val_loss"):
            errors.append("train.select_metric must be 'BLEU-4', 'CIDEr' or 'val_loss'")
        if t.stop_after_epochs is not None and t.stop_after_epochs < 1:
            errors.append("train.stop_after_epochs must be >= 1 or null")
        if e.beam_size < 1 or e.max_length < 1 or e.batch_size < 1:
            errors.append("eval.beam_size, eval.max_length and eval.batch_size must be >= 1")
        if tr.backend not in ("none", "tensorboard", "wandb"):
            errors.append("tracking.backend must be 'none', 'tensorboard' or 'wandb'")
        if tr.wandb_mode not in ("online", "offline", "disabled"):
            errors.append("tracking.wandb_mode must be 'online', 'offline' or 'disabled'")
        if tr.log_every_n_steps < 1:
            errors.append("tracking.log_every_n_steps must be >= 1")
        if not (self.device == "auto" or self.device == "cpu" or self.device == "mps"
                or self.device.startswith("cuda")):
            errors.append("device must be 'auto', 'cpu', 'mps', 'cuda' or 'cuda:N'")
        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return self


# --------------------------------------------------------------------------- #
# Setting values with type checking
# --------------------------------------------------------------------------- #
def _field_types(obj: Any) -> Dict[str, Any]:
    return typing.get_type_hints(type(obj))


def _coerce(value: Any, annotation: Any, key: str) -> Any:
    """Check / convert ``value`` against a field annotation."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union and type(None) in args:          # Optional[X]
        if value is None or (isinstance(value, str) and value.lower() in ("none", "null")):
            return None
        annotation = next(a for a in args if a is not type(None))
    if isinstance(value, str) and annotation is not str:
        value = _parse_value(value)                             # CLI text -> typed value
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no"):
            return value.lower() in ("true", "1", "yes")
        raise TypeError(f"'{key}' expects true/false, got {value!r}")
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"'{key}' expects an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' expects a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            return str(value)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            return value[1:-1]                                  # --set k='a b' -> a b
        return value
    return value


def _set(obj: Any, key: str, value: Any, prefix: str = "") -> None:
    names = {f.name for f in fields(obj)}
    full = f"{prefix}{key}"
    if key not in names:
        raise KeyError(f"unknown config key '{full}'; valid keys here: {sorted(names)}")
    current = getattr(obj, key)
    if dataclasses.is_dataclass(current):
        if not isinstance(value, dict):
            raise TypeError(f"'{full}' must be a mapping")
        for k, v in value.items():
            _set(current, k, v, prefix=f"{full}.")
    else:
        setattr(obj, key, _coerce(value, _field_types(obj)[key], full))


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


def set_by_path(cfg: Config, dotted: str, value: Any) -> None:
    """``set_by_path(cfg, "train.lr", 1e-4)`` with type checking."""
    parts = dotted.split(".")
    target: Any = cfg
    for p in parts[:-1]:
        if not hasattr(target, p) or not dataclasses.is_dataclass(getattr(target, p)):
            raise KeyError(f"unknown config section '{p}' in '{dotted}'")
        target = getattr(target, p)
    _set(target, parts[-1], value, prefix=".".join(parts[:-1]) + ("." if len(parts) > 1 else ""))


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """Apply ``section.key=value`` overrides, e.g. ``train.lr=1e-4``."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        set_by_path(cfg, dotted, raw)          # parsed against the field type
    return cfg


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None,
                validate: bool = True) -> Config:
    """Defaults <- YAML file (if given) <- ``section.key=value`` overrides."""
    cfg = Config()
    if path:
        try:
            with open(path) as f:
                cfg = Config.from_dict(yaml.safe_load(f) or {})
        except FileNotFoundError:
            raise FileNotFoundError(f"config file not found: {path}") from None
    cfg = apply_overrides(cfg, overrides or [])
    return cfg.validate() if validate else cfg


# --------------------------------------------------------------------------- #
# Command-line flags for every key
# --------------------------------------------------------------------------- #
_SECTION_PRIORITY = ("train", "model", "data", "eval", "tracking")


def config_keys() -> List[Tuple[str, Any]]:
    """All leaf keys as ``(dotted_path, default)``."""
    out: List[Tuple[str, Any]] = []

    def walk(obj: Any, prefix: str) -> None:
        for f in fields(obj):
            value = getattr(obj, f.name)
            if dataclasses.is_dataclass(value):
                walk(value, f"{prefix}{f.name}.")
            else:
                out.append((f"{prefix}{f.name}", value))
    walk(Config(), "")
    return out


def short_aliases() -> Dict[str, str]:
    """Short flag name -> dotted key, resolving clashes by section priority."""
    by_name: Dict[str, List[str]] = {}
    for dotted, _ in config_keys():
        by_name.setdefault(dotted.split(".")[-1], []).append(dotted)
    aliases = {}
    for name, paths in by_name.items():
        rank = lambda p: (_SECTION_PRIORITY.index(p.split(".")[0])  # noqa: E731
                          if "." in p else -1)
        aliases[name] = sorted(paths, key=rank)[0]
    return aliases


def add_config_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--config``, ``--set`` and one flag per config key to ``parser``."""
    parser.add_argument("--config", default=None, help="YAML config file")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="overrides applied last, e.g. train.lr=1e-4")
    group = parser.add_argument_group("config keys (any key in configs/default.yaml)")
    aliases = short_aliases()
    alias_for = {dotted: name for name, dotted in aliases.items()}
    taken = {"--config", "--set", "-h", "--help"}
    for dotted, default in config_keys():
        flags = [f"--{dotted}"]
        name = alias_for.get(dotted)
        if name and name != dotted:
            flags.append(f"--{name}")
        flags += [f.replace("_", "-") for f in flags if "_" in f]
        flags = [f for f in dict.fromkeys(flags) if f not in taken]
        taken.update(flags)
        kwargs: Dict[str, Any] = dict(dest=f"cfg__{dotted}", default=argparse.SUPPRESS,
                                      help=f"(default: {default})", metavar="V")
        if isinstance(default, bool):
            kwargs.update(nargs="?", const="true")          # "--amp" or "--amp false"
        group.add_argument(*flags, **kwargs)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Defaults <- ``--config`` YAML <- per-key flags <- ``--set``; then validate."""
    cfg = load_config(getattr(args, "config", None), validate=False)
    for dest, raw in vars(args).items():
        if dest.startswith("cfg__"):
            set_by_path(cfg, dest[len("cfg__"):], raw)
    cfg = apply_overrides(cfg, getattr(args, "set", None) or [])
    return cfg.validate()

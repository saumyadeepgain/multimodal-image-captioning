"""Optional experiment tracking: TensorBoard or Weights & Biases.

Selected with ``tracking.backend`` in ``configs/default.yaml`` (``none`` by
default). Both packages are optional: if the selected backend is not
installed, or fails to start (e.g. W&B without network / login), a warning is
logged and training continues with a no-op tracker -- tracking must never be
the reason a run dies.

    tracker = ExperimentTracker(cfg.tracking, cfg.to_dict(), out_dir, logger)
    tracker.log_metrics({"train/loss": 2.3, "lr": 3e-4}, step=100)
    tracker.finish()
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from .config import TrackingConfig


class ExperimentTracker:
    """Thin facade over TensorBoard ``SummaryWriter`` / ``wandb``."""

    def __init__(
        self,
        tracking_cfg: TrackingConfig,
        run_config: dict[str, Any],
        out_dir: str,
        logger: Optional[logging.Logger] = None,
        run_name: Optional[str] = None,
    ):
        self.cfg = tracking_cfg
        self.logger = logger or logging.getLogger("captioning")
        self.backend = "none"
        self._writer: Any = None
        self._wandb: Any = None
        name = tracking_cfg.run_name or run_name
        if tracking_cfg.backend == "tensorboard":
            self._start_tensorboard(os.path.join(out_dir, "tensorboard", name or "run"))
        elif tracking_cfg.backend == "wandb":
            self._start_wandb(run_config, out_dir, name)

    # ------------------------------------------------------------------ setup
    def _start_tensorboard(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self.logger.warning(
                "tracking.backend=tensorboard but the 'tensorboard' package is "
                "not installed (pip install tensorboard); tracking disabled."
            )
            return
        self._writer = SummaryWriter(log_dir=log_dir)
        self.backend = "tensorboard"
        self.logger.info(
            f"TensorBoard logging to {log_dir}  (tensorboard --logdir {os.path.dirname(log_dir)})"
        )

    def _start_wandb(self, run_config: dict[str, Any], out_dir: str, name: Optional[str]) -> None:
        try:
            import wandb
        except ImportError:
            self.logger.warning(
                "tracking.backend=wandb but the 'wandb' package is not installed "
                "(pip install wandb); tracking disabled."
            )
            return
        try:
            # A stable id derived from the run name lets a resumed training run
            # continue the same W&B run (resume="allow").
            run_id = re.sub(r"[^A-Za-z0-9_-]", "-", name) if name else None
            self._wandb = wandb.init(
                project=self.cfg.project,
                name=name,
                id=run_id,
                resume="allow",
                config=run_config,
                dir=out_dir,
                mode=self.cfg.wandb_mode,
                reinit=True,
            )
            self.backend = "wandb"
            self.logger.info(f"W&B run '{self._wandb.name}' ({self.cfg.wandb_mode})")
        except Exception as exc:  # network / auth problems must not stop training
            self.logger.warning(f"Could not start W&B ({exc}); tracking disabled.")
            self._wandb = None

    # ---------------------------------------------------------------- logging
    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def log_metrics(self, metrics: dict[str, Optional[float]], step: int) -> None:
        """Log scalar metrics; ``None`` values are skipped."""
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        if not clean or not self.enabled:
            return
        try:
            if self._writer is not None:
                for k, v in clean.items():
                    self._writer.add_scalar(k, v, step)
            elif self._wandb is not None:
                self._wandb.log(clean, step=step)
        except Exception as exc:
            self.logger.warning(f"Tracking failed ({exc}); disabling tracker.")
            self.backend = "none"

    def log_text(self, key: str, text: str, step: int) -> None:
        """Log free text (e.g. sample captions)."""
        if not self.enabled:
            return
        try:
            if self._writer is not None:
                self._writer.add_text(key, text, step)
            elif self._wandb is not None:
                self._wandb.log({key: text}, step=step)
        except Exception as exc:
            self.logger.warning(f"Tracking failed ({exc}); disabling tracker.")
            self.backend = "none"

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None
        self.backend = "none"

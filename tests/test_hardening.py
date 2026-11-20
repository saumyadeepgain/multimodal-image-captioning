"""Tests for CLI flags, config validation, tracking, OOM recovery and data edge cases."""
import argparse
import logging
import os
import subprocess
import sys

import pytest
import torch

from src.config import add_config_arguments, config_from_args, load_config
from src.dataset import build_datasets, verify_image_pairs, load_captions
from src.model import ImageCaptioningModel
from src.tracking import ExperimentTracker
from src.train import train_epoch, warmup_cosine
from src.utils import get_device, is_oom_error, load_training_state

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse(argv):
    return config_from_args(add_config_arguments(argparse.ArgumentParser()).parse_args(argv))


# ----------------------------------------------------------------- CLI / config
def test_flat_and_dotted_cli_flags():
    cfg = _parse(["--lr", "1e-4", "--batch_size", "32", "--eval.batch_size", "8",
                  "--amp", "false", "--tie-weights", "--resume", "ck/last.pth",
                  "--run_name", "001", "--set", "train.epochs=3"])
    assert cfg.train.lr == 1e-4 and cfg.train.batch_size == 32 and cfg.eval.batch_size == 8
    assert cfg.train.amp is False and cfg.model.tie_weights is True
    assert cfg.train.resume == "ck/last.pth" and cfg.tracking.run_name == "001"
    assert cfg.train.epochs == 3


def test_cli_precedence_yaml_then_flags_then_set(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.01\n  epochs: 7\n")
    cfg = _parse(["--config", str(p), "--lr", "0.02", "--set", "train.lr=0.03"])
    assert cfg.train.lr == 0.03 and cfg.train.epochs == 7


@pytest.mark.parametrize("argv, message", [
    (["--lr", "abc"], "expects a number"),
    (["--epochs", "2.5"], "expects an integer"),
    (["--embed_size", "100"], "divisible"),
    (["--backend", "mlflow"], "tracking.backend"),
    (["--select_metric", "loss"], "select_metric"),
])
def test_invalid_values_are_rejected(argv, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _parse(argv)


def test_unknown_yaml_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  learning_rate: 0.1\n")
    with pytest.raises(KeyError, match="learning_rate"):
        load_config(str(p))


def test_script_mode_help_runs():
    """`python src/train.py --help` works without -m (relative-import bootstrap)."""
    for script in ("src/train.py", "src/evaluate.py", "src/predict.py"):
        out = subprocess.run([sys.executable, script, "--help"], cwd=REPO, capture_output=True,
                             text=True, timeout=120)
        assert out.returncode == 0, out.stderr
    assert "--batch_size" in subprocess.run([sys.executable, "src/train.py", "--help"], cwd=REPO,
                                            capture_output=True, text=True).stdout


# ---------------------------------------------------------------------- device
def test_device_fallback(caplog):
    assert get_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with caplog.at_level(logging.WARNING, logger="captioning"):
            assert get_device("cuda").type in ("cpu", "mps")
        assert "not available" in caplog.text
    assert get_device("auto").type in ("cuda", "mps", "cpu")


# -------------------------------------------------------------------- tracking
def test_tracker_noop_and_tensorboard(tmp_path):
    cfg = load_config()
    noop = ExperimentTracker(cfg.tracking, cfg.to_dict(), str(tmp_path))
    assert not noop.enabled
    noop.log_metrics({"a": 1.0}, 1)
    noop.finish()

    pytest.importorskip("tensorboard")
    cfg.tracking.backend = "tensorboard"
    tb = ExperimentTracker(cfg.tracking, cfg.to_dict(), str(tmp_path), run_name="r1")
    assert tb.enabled
    tb.log_metrics({"train/loss": 1.5, "skip": None}, step=3)
    tb.log_text("samples", "a dog", step=3)
    tb.finish()
    assert any(f.startswith("events.out") for f in os.listdir(tmp_path / "tensorboard" / "r1"))


def test_tracker_wandb_offline(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = load_config(None, ["tracking.backend=wandb", "tracking.wandb_mode=offline"])
    tr = ExperimentTracker(cfg.tracking, cfg.to_dict(), str(tmp_path), run_name="t")
    tr.log_metrics({"val/loss": 2.0}, step=1)
    tr.finish()
    assert (tmp_path / "wandb").exists()


# ----------------------------------------------------------------- OOM recovery
def test_oom_microbatching_matches_full_batch():
    """A model that 'OOMs' on batches > 4 trains to the same weights."""
    V = 20
    feats = torch.randn(8, 49, 2048)
    caps = torch.randint(4, V, (8, 7))
    caps[:, 0] = 1
    caps[:3, 5:] = 0                                   # unequal lengths: token weighting matters

    def one_epoch(oom_above):
        torch.manual_seed(0)
        m = ImageCaptioningModel(32, V, 1, 4, pretrained=False, dropout=0.0)
        if oom_above:
            real = m.forward

            def flaky(*a, **k):
                if k["captions"].size(0) > oom_above:
                    raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
                return real(*a, **k)
            m.forward = flaky
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        state = {"chunks": 1}
        stats = train_epoch(m, [(feats, caps)], torch.nn.CrossEntropyLoss(ignore_index=0), opt,
                            warmup_cosine(opt, 0, 10), torch.device("cpu"), cached=True,
                            oom_state=state)
        return m.state_dict(), stats, state

    full, s_full, _ = one_epoch(None)
    split, s_split, state = one_epoch(4)
    assert state["chunks"] == 2
    assert s_split["loss"] == pytest.approx(s_full["loss"], rel=1e-5)
    for k in full:
        assert torch.allclose(full[k], split[k], atol=1e-6), k


def test_oom_beyond_limit_reraises():
    m = ImageCaptioningModel(32, 20, 1, 4, pretrained=False)

    def always(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")
    m.forward = always
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    batch = (torch.randn(4, 49, 2048), torch.randint(1, 20, (4, 6)))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        train_epoch(m, [batch], torch.nn.CrossEntropyLoss(ignore_index=0), opt,
                    warmup_cosine(opt, 0, 1), torch.device("cpu"), cached=True, max_oom_splits=2)
    assert is_oom_error(RuntimeError("CUDA error: out of memory"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


# -------------------------------------------------------------- data edge cases
def test_missing_paths_raise_clear_errors(tiny_flickr, tmp_path):
    img_dir, cap_file = tiny_flickr
    with pytest.raises(FileNotFoundError, match="Image folder not found"):
        build_datasets(str(tmp_path / "nope"), cap_file)
    with pytest.raises(FileNotFoundError, match="Captions file not found"):
        build_datasets(img_dir, str(tmp_path / "nope.txt"))
    empty = tmp_path / "empty.txt"
    empty.write_text("image,caption\n")
    with pytest.raises(ValueError, match="No \\(image, caption\\) pairs"):
        load_captions(str(empty))


def test_corrupt_and_missing_images_are_dropped(tiny_flickr, caplog):
    img_dir, cap_file = tiny_flickr
    with open(os.path.join(img_dir, "img_0.jpg"), "wb") as f:
        f.write(b"not a jpeg")                         # corrupted file
    os.remove(os.path.join(img_dir, "img_1.jpg"))      # missing file
    with caplog.at_level(logging.WARNING, logger="captioning"):
        pairs = verify_image_pairs(img_dir, load_captions(cap_file))
    assert {i for i, _ in pairs} == {f"img_{k}.jpg" for k in range(2, 20)}
    assert "Dropping 2 of 20 images" in caplog.text


def test_runtime_corruption_falls_back_to_next_sample(tiny_flickr, caplog):
    img_dir, cap_file = tiny_flickr
    ds, _ = build_datasets(img_dir, cap_file, freq_threshold=1, val_frac=0.2, test_frac=0.2,
                           image_size=64)
    train = ds["train"]
    bad = train.imgs[0]
    with open(os.path.join(img_dir, bad), "wb") as f:
        f.write(b"\xff\xd8\xff garbage")                # truncated JPEG header
    with caplog.at_level(logging.WARNING, logger="captioning"):
        img, cap = train[0]
    assert img.shape == (3, 64, 64)
    j = next(k for k, name in enumerate(train.imgs) if name != bad)
    assert torch.equal(cap, train.encoded[j])          # caption belongs to the replacement
    assert "Unreadable image" in caplog.text
    train.strict = True
    with pytest.raises(OSError):
        train[0]


def test_all_images_missing_is_an_error(tiny_flickr):
    img_dir, cap_file = tiny_flickr
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
    with pytest.raises(FileNotFoundError, match="None of the 20 images"):
        build_datasets(img_dir, cap_file, verify_images=True)


# ------------------------------------------------------------- resume errors
def test_resume_error_messages(tmp_path):
    from src.dataset import Vocabulary
    from src.utils import save_checkpoint
    v = Vocabulary(1)
    v.build_vocabulary(["a b c"])
    m = ImageCaptioningModel(32, len(v), 1, 4, pretrained=False)
    opt = torch.optim.AdamW(m.parameters())
    sched = warmup_cosine(opt, 0, 1)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_training_state(str(tmp_path / "missing.pth"), m, opt, sched, None)
    save_checkpoint(str(tmp_path / "best.pth"), m, v)
    with pytest.raises(ValueError, match="resume from last.pth"):
        load_training_state(str(tmp_path / "best.pth"), m, opt, sched, None)
    (tmp_path / "junk.pth").write_bytes(b"junk")
    with pytest.raises(ValueError, match="corrupted"):
        load_training_state(str(tmp_path / "junk.pth"), m, opt, sched, None)

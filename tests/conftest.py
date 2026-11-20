"""Shared fixtures. Everything runs on CPU with synthetic data and a
randomly initialised backbone, so the suite needs no dataset, GPU or network."""

import os
import random
import sys

import pytest
from PIL import Image

os.environ.setdefault("MPLBACKEND", "Agg")  # headless matplotlib in CI
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORDS = ["a", "dog", "runs", "on", "the", "grass", "man", "rides", "bike", "street"]


def _make_flickr(root):
    """20 images x 5 captions in Kaggle ``captions.txt`` format (captions contain commas)."""
    rng = random.Random(0)
    img_dir = root / "Images"
    img_dir.mkdir()
    lines = ["image,caption"]
    for i in range(20):
        name = f"img_{i}.jpg"
        Image.new("RGB", (64 + i, 48), color=(i * 10, 100, 200)).save(img_dir / name)
        for _ in range(5):
            cap = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 9)))
            lines.append(f"{name},{cap}, and more .")
    cap_file = root / "captions.txt"
    cap_file.write_text("\n".join(lines))
    return str(img_dir), str(cap_file)


@pytest.fixture
def tiny_flickr(tmp_path):
    """Fresh copy per test (tests may corrupt or delete files)."""
    return _make_flickr(tmp_path)


@pytest.fixture(scope="session")
def trained_checkpoint(tmp_path_factory):
    """A real ``best_model.pth`` from a 2-epoch run on synthetic data (shared, read-only)."""
    from src.config import load_config
    from src.train import run

    root = tmp_path_factory.mktemp("trained")
    img_dir, cap_file = _make_flickr(root)
    out_dir = str(root / "ckpt")
    cfg = load_config(
        None,
        [
            f"data.img_dir={img_dir}",
            f"data.captions_file={cap_file}",
            "data.freq_threshold=1",
            "data.image_size=64",
            "data.num_workers=0",
            "data.val_frac=0.2",
            "data.test_frac=0.2",
            "model.embed_size=32",
            "model.num_layers=1",
            "model.nhead=4",
            "model.pretrained=false",
            "train.epochs=2",
            "train.amp=false",
            f"train.out_dir={out_dir}",
        ],
    )
    run(cfg)
    return {
        "checkpoint": os.path.join(out_dir, "best_model.pth"),
        "out_dir": out_dir,
        "img_dir": img_dir,
        "captions": cap_file,
    }

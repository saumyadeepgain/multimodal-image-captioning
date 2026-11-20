import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORDS = ["a", "dog", "runs", "on", "the", "grass", "man", "rides", "bike", "street"]


@pytest.fixture
def tiny_flickr(tmp_path):
    """20 images x 5 captions in Kaggle captions.txt format (with commas)."""
    import random
    rng = random.Random(0)
    img_dir = tmp_path / "Images"
    img_dir.mkdir()
    lines = ["image,caption"]
    for i in range(20):
        name = f"img_{i}.jpg"
        Image.new("RGB", (64 + i, 48), color=(i * 10, 100, 200)).save(img_dir / name)
        for _ in range(5):
            cap = " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 9)))
            lines.append(f"{name},{cap}, and more .")
    cap_file = tmp_path / "captions.txt"
    cap_file.write_text("\n".join(lines))
    return str(img_dir), str(cap_file)

"""Dataset tests: transforms, normalisation ranges, encoding, collate and loader shapes.

Uses the ``tiny_flickr`` fixture from ``conftest.py`` (20 synthetic images x 5
captions in Kaggle ``captions.txt`` format).
"""

import pytest
import torch
from PIL import Image

from src.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CachedFeatureDataset,
    MyCollate,
    Vocabulary,
    build_datasets,
    get_transforms,
    make_loader,
    tokenize,
)

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
LOW, HIGH = (0 - MEAN) / STD, (1 - MEAN) / STD  # normalised range per channel


# ----------------------------------------------------------------- transforms
@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("size", [64, 224])
def test_transform_output_shape_dtype_and_range(train, size):
    img = Image.new("RGB", (300, 200), color=(10, 120, 250))  # non-square input
    x = get_transforms(train=train, image_size=size)(img)
    assert x.shape == (3, size, size) and x.dtype == torch.float32
    assert torch.isfinite(x).all()
    assert (x >= LOW - 1e-5).all() and (x <= HIGH + 1e-5).all()


def test_normalisation_values_are_exact_for_black_and_white():
    tf = get_transforms(train=False, image_size=32)
    black = tf(Image.new("RGB", (32, 32), (0, 0, 0)))
    white = tf(Image.new("RGB", (32, 32), (255, 255, 255)))
    assert torch.allclose(black, LOW.expand(3, 32, 32), atol=1e-6)
    assert torch.allclose(white, HIGH.expand(3, 32, 32), atol=1e-6)


def test_eval_transform_is_deterministic_and_train_transform_is_random():
    pixels = torch.randint(0, 255, (96, 128, 3), dtype=torch.uint8).numpy()
    img = Image.fromarray(pixels)
    ev = get_transforms(train=False, image_size=64)
    assert torch.equal(ev(img), ev(img))
    tr = get_transforms(train=True, image_size=64)
    torch.manual_seed(0)
    a = tr(img)
    b = tr(img)
    assert not torch.equal(a, b)  # random crop + colour jitter


def test_grayscale_and_rgba_inputs_become_three_channels():
    tf = get_transforms(train=False, image_size=64)
    for mode in ("L", "RGBA"):
        assert tf(Image.new(mode, (50, 40)).convert("RGB")).shape == (3, 64, 64)


# ------------------------------------------------------- tokenisation / vocabulary
def test_tokenize_and_vocab_threshold():
    assert tokenize("A Dog's ball, 2 cats!") == ["a", "dog's", "ball", "2", "cats"]
    v = Vocabulary(freq_threshold=2)
    v.build_vocabulary(["a dog", "a cat", "a dog runs"])
    assert "a" in v.stoi and "dog" in v.stoi
    assert "cat" not in v.stoi and "runs" not in v.stoi  # below threshold
    assert [v.itos[i] for i in range(4)] == ["<pad>", "<start>", "<end>", "<unk>"]


def test_vocab_is_built_from_train_split_only(tiny_flickr, tmp_path):
    img_dir, cap_file = tiny_flickr
    with open(cap_file, "a") as f:  # word only in one image
        f.write("\nimg_19.jpg,zebra zebra zebra zebra zebra zebra")
    ds, vocab = build_datasets(img_dir, cap_file, freq_threshold=1, val_frac=0.2, test_frac=0.2)
    zebra_split = next(k for k in ("train", "val", "test") if "img_19.jpg" in ds[k].imgs)
    assert ("zebra" in vocab.stoi) == (zebra_split == "train")


# ----------------------------------------------------------- dataset items
def test_caption_dataset_items(tiny_flickr):
    img_dir, cap_file = tiny_flickr
    ds, vocab = build_datasets(img_dir, cap_file, freq_threshold=1, max_len=6, image_size=64)
    img, cap = ds["train"][0]
    assert img.shape == (3, 64, 64)
    assert cap.dtype == torch.long and cap.dim() == 1
    assert cap[0] == vocab.start_idx and cap[-1] == vocab.end_idx
    assert all(len(c) <= 6 for c in ds["train"].encoded)  # truncated to max_len


def test_image_dataset_groups_all_references(tiny_flickr):
    img_dir, cap_file = tiny_flickr
    ds, _ = build_datasets(img_dir, cap_file, freq_threshold=1, image_size=64)
    images = ds["val_images"]
    assert len(images) == len(set(ds["val"].imgs))
    assert all(len(r) == 5 for r in images.references)
    img, idx = images[0]
    assert img.shape == (3, 64, 64) and idx == 0


# --------------------------------------------------------------- collate / loader
def test_collate_pads_and_stacks():
    pad = 0
    batch = [
        (torch.randn(3, 32, 32), torch.tensor([1, 5, 2])),
        (torch.randn(3, 32, 32), torch.tensor([1, 5, 6, 7, 2])),
        (torch.randn(3, 32, 32), torch.tensor([1, 2])),
    ]
    imgs, caps = MyCollate(pad)(batch)
    assert imgs.shape == (3, 3, 32, 32)
    assert caps.shape == (3, 5) and caps.dtype == torch.long
    assert caps.tolist() == [[1, 5, 2, 0, 0], [1, 5, 6, 7, 2], [1, 2, 0, 0, 0]]
    assert torch.equal(imgs[1], batch[1][0])


def test_collate_cached_features():
    feats = [(torch.randn(49, 2048).half(), torch.tensor([1, 4, 2])) for _ in range(4)]
    x, caps = MyCollate(0)(feats)
    assert x.shape == (4, 49, 2048) and x.dtype == torch.float16 and caps.shape == (4, 3)


def test_loader_batch_shapes_and_reproducible_shuffle(tiny_flickr):
    img_dir, cap_file = tiny_flickr
    ds, vocab = build_datasets(img_dir, cap_file, freq_threshold=1, image_size=64)
    loader = make_loader(
        ds["train"], batch_size=8, shuffle=True, num_workers=0, pad_idx=vocab.pad_idx, seed=123
    )
    imgs, caps = next(iter(loader))
    assert imgs.shape == (8, 3, 64, 64)
    assert caps.shape[0] == 8 and (caps[:, 0] == vocab.start_idx).all()
    # Every row: <start> ... <end> followed only by padding.
    for row in caps.tolist():
        end = row.index(vocab.end_idx)
        assert all(t == vocab.pad_idx for t in row[end + 1 :])

    def order():
        return [
            c.tolist() for _, c in make_loader(ds["train"], 8, True, 0, vocab.pad_idx, seed=123)
        ]

    assert order() == order()


def test_cached_feature_dataset_aligns_rows_with_images(tiny_flickr):
    img_dir, cap_file = tiny_flickr
    ds, _ = build_datasets(img_dir, cap_file, freq_threshold=1, image_size=64)
    names = ds["train_images"].images
    feats = torch.arange(len(names)).float().view(-1, 1, 1).expand(-1, 49, 2048).contiguous()
    cached = CachedFeatureDataset(feats, {n: i for i, n in enumerate(names)}, ds["train"])
    for i in (0, 7, len(cached) - 1):
        f, cap = cached[i]
        assert f.shape == (49, 2048)
        assert names[int(f[0, 0])] == ds["train"].imgs[i]  # right image for this caption
        assert torch.equal(cap, ds["train"].encoded[i])

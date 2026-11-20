"""Data pipeline for Flickr8k image captioning.

Design notes (see docs/IMPROVEMENTS.pdf for the full discussion):

* The captions file is read and split **once**. The previous version re-read the
  CSV and re-ran the split for every split object, and parsed it with
  ``pd.read_csv`` which breaks on captions that contain commas.
* The split is done **by image**, so the five captions of one image never leak
  across train / val / test.
* Captions are tokenised and converted to ids **once** at construction time
  instead of on every ``__getitem__`` call (the old code ran NLTK's tokenizer
  ~40k times per epoch).
* The tokenizer is a small regex instead of NLTK ``word_tokenize``: it needs no
  ``punkt`` / ``punkt_tab`` download (newer NLTK versions renamed the resource,
  which broke the old code), is deterministic, and is ~20x faster.
* ``Vocabulary`` can be serialised to / from a plain dict, so a checkpoint can
  carry its own vocabulary. Without that the demo app could not decode ids.
* Edge cases: a missing image folder / captions file raises a clear error;
  with ``verify_images`` every image is checked once up front and missing or
  unreadable files are dropped (with a logged summary) instead of silently
  becoming black training images; a file that becomes unreadable mid-run is
  replaced by a neighbouring sample and logged once.
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = (PAD, START, END, UNK)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_IMAGE_ERRORS = (OSError, UnidentifiedImageError, ValueError, SyntaxError)
_FALLBACK_TRIES = 10
logger = logging.getLogger("captioning")
_warned_files: set = set()


def _warn_once(path: str, exc: BaseException) -> None:
    if path not in _warned_files:
        _warned_files.add(path)
        logger.warning(f"Unreadable image {path}: {type(exc).__name__}: {exc}")


def open_image(path: str) -> Image.Image:
    """Open an image as RGB, forcing a full decode so truncated/corrupt files
    fail here (PIL is lazy) rather than deep inside a transform."""
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def check_image(path: str) -> Optional[str]:
    """``None`` if the image is readable, else a short reason."""
    if not os.path.isfile(path):
        return "missing"
    try:
        with Image.open(path) as im:
            im.verify()
        return None
    except _IMAGE_ERRORS as exc:
        return f"{type(exc).__name__}: {exc}"


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer. Punctuation is dropped (it carries no
    information in Flickr8k captions and only inflates the vocabulary)."""
    return _TOKEN_RE.findall(str(text).lower())


class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.itos: dict[int, str] = dict(enumerate(SPECIAL_TOKENS))
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}

    # Kept for backwards compatibility with code that called Vocabulary.tokenize
    tokenize = staticmethod(tokenize)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD]

    @property
    def start_idx(self) -> int:
        return self.stoi[START]

    @property
    def end_idx(self) -> int:
        return self.stoi[END]

    def build_vocabulary(self, sentence_list: Iterable[str]) -> None:
        counts = Counter(tok for s in sentence_list for tok in tokenize(s))
        # Sort by (-frequency, word) so ids are deterministic across runs.
        for word, freq in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            if freq >= self.freq_threshold and word not in self.stoi:
                idx = len(self.itos)
                self.stoi[word] = idx
                self.itos[idx] = word

    def numericalize(self, text: str) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(tok, unk) for tok in tokenize(text)]

    def encode(self, text: str, max_len: Optional[int] = None) -> list[int]:
        """``<start> w1 ... wn <end>``, truncated so the total is <= max_len."""
        ids = self.numericalize(text)
        if max_len is not None:
            ids = ids[: max_len - 2]
        return [self.start_idx] + ids + [self.end_idx]

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> list[str]:
        words = []
        for i in ids:
            i = int(i)
            if i == self.end_idx and strip_special:
                break
            w = self.itos.get(i, UNK)
            if strip_special and w in (PAD, START):
                continue
            words.append(w)
        return words

    def to_dict(self) -> dict:
        return {
            "freq_threshold": self.freq_threshold,
            "itos": [self.itos[i] for i in range(len(self.itos))],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Vocabulary:
        v = cls(d.get("freq_threshold", 5))
        v.itos = dict(enumerate(d["itos"]))
        v.stoi = {w: i for i, w in v.itos.items()}
        return v


# --------------------------------------------------------------------------- #
# Reading and splitting
# --------------------------------------------------------------------------- #
def load_captions(captions_file: str) -> list[tuple[str, str]]:
    """Return ``[(image_name, caption), ...]``.

    Handles both common Flickr8k layouts:
      * Kaggle ``captions.txt``: header ``image,caption`` then ``img.jpg,caption``
      * Original ``Flickr8k.token.txt``: ``img.jpg#0<TAB>caption``
    Only the first separator is used, so commas inside captions are safe.
    """
    if not os.path.isfile(captions_file):
        raise FileNotFoundError(
            f"Captions file not found: {captions_file}. Download Flickr8k "
            "(https://www.kaggle.com/datasets/adityajn105/flickr8k) and set data.captions_file."
        )
    pairs: list[tuple[str, str]] = []
    with open(captions_file, encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                img, cap = line.split("\t", 1)
                img = img.split("#", 1)[0]
            elif "," in line:
                img, cap = line.split(",", 1)
            else:
                continue
            if line_no == 0 and img.strip().lower() == "image":
                continue  # header row
            if img.strip() and cap.strip():
                pairs.append((img.strip(), cap.strip()))
    if not pairs:
        raise ValueError(
            f"No (image, caption) pairs could be parsed from {captions_file}; "
            "expected 'image,caption' lines or 'img.jpg#0<TAB>caption'."
        )
    return pairs


def verify_image_pairs(
    img_dir: str, pairs: list[tuple[str, str]], num_threads: int = 16
) -> list[tuple[str, str]]:
    """Drop pairs whose image is missing or unreadable; log a summary.

    Raises ``FileNotFoundError`` if *no* image is usable (almost always a wrong
    ``data.img_dir``)."""
    images = sorted({img for img, _ in pairs})
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        reasons = dict(
            zip(images, pool.map(lambda i: check_image(os.path.join(img_dir, i)), images))
        )
    bad = {img: r for img, r in reasons.items() if r is not None}
    if len(bad) == len(images):
        raise FileNotFoundError(
            f"None of the {len(images)} images referenced in the captions "
            f"file were found/readable in {img_dir}."
        )
    if bad:
        examples = ", ".join(f"{k} ({v})" for k, v in list(bad.items())[:5])
        logger.warning(
            f"Dropping {len(bad)} of {len(images)} images that are missing or "
            f"unreadable, e.g. {examples}"
        )
    return [(img, cap) for img, cap in pairs if img not in bad]


def split_by_image(
    pairs: list[tuple[str, str]], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42
):
    """Split caption pairs into train/val/test with no image in two splits."""
    images = sorted({img for img, _ in pairs})
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = int(round(len(images) * val_frac))
    n_test = int(round(len(images) * test_frac))
    val_set = set(images[:n_val])
    test_set = set(images[n_val : n_val + n_test])
    splits = {"train": [], "val": [], "test": []}
    for img, cap in pairs:
        key = "val" if img in val_set else "test" if img in test_set else "train"
        splits[key].append((img, cap))
    return splits


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def get_transforms(train: bool, image_size: int = 224) -> T.Compose:
    """ImageNet-normalised transforms. Normalisation uses the fixed ImageNet
    statistics the backbone was trained with -- nothing is fitted on data, so
    there is no preprocessing leakage between splits."""
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        # Mild augmentation: no horizontal flip, because captions such as
        # "on the left" would become wrong.
        return T.Compose(
            [
                T.Resize((int(image_size * 1.14), int(image_size * 1.14))),
                T.RandomCrop(image_size),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class CaptionDataset(Dataset):
    """One item per (image, caption) pair. Captions are pre-encoded."""

    def __init__(
        self,
        img_dir: str,
        pairs: list[tuple[str, str]],
        vocab: Vocabulary,
        transform=None,
        max_len: int = 40,
        strict: bool = False,
    ):
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.strict = strict
        self.imgs = [img for img, _ in pairs]
        self.captions = [cap for _, cap in pairs]
        self.encoded = [
            torch.tensor(vocab.encode(c, max_len), dtype=torch.long) for c in self.captions
        ]

    def __len__(self) -> int:
        return len(self.imgs)

    def load_image(self, img_id: str) -> Image.Image:
        """Open one image; raises on missing / corrupt files."""
        return open_image(os.path.join(self.img_dir, img_id))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(image [3, H, W], caption ids [T])``.

        If the image cannot be read, the warning is logged once per file and the
        next sample is used instead (``strict=True`` raises instead). A
        replacement keeps its own caption, so image and text always match."""
        for attempt in range(_FALLBACK_TRIES):
            i = (index + attempt) % len(self.imgs)
            try:
                img = self.load_image(self.imgs[i])
            except _IMAGE_ERRORS as exc:
                if self.strict:
                    raise
                _warn_once(os.path.join(self.img_dir, self.imgs[i]), exc)
                continue
            if self.transform is not None:
                img = self.transform(img)
            return img, self.encoded[i]
        raise RuntimeError(
            f"{_FALLBACK_TRIES} consecutive unreadable images starting at "
            f"index {index} in {self.img_dir}"
        )


class ImageDataset(Dataset):
    """One item per *unique* image, with all its reference captions.

    Used for evaluation (BLEU must compare against all 5 references) and for
    pre-computing CNN features once per image instead of once per caption.
    """

    def __init__(
        self, img_dir: str, pairs: list[tuple[str, str]], transform=None, strict: bool = False
    ):
        refs = defaultdict(list)
        for img, cap in pairs:
            refs[img].append(cap)
        self.img_dir = img_dir
        self.transform = transform
        self.strict = strict
        self.images = sorted(refs)
        self.references = [refs[i] for i in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """``(image [3, H, W], index)``. An unreadable image cannot be swapped
        for another one here (its references would no longer match), so it is
        replaced by a black image and logged once; ``strict=True`` raises."""
        path = os.path.join(self.img_dir, self.images[index])
        try:
            img = open_image(path)
        except _IMAGE_ERRORS as exc:
            if self.strict:
                raise
            _warn_once(path, exc)
            img = Image.new("RGB", (224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img, index


class CachedFeatureDataset(Dataset):
    """(backbone_features, caption) pairs using a pre-computed feature bank.

    With a frozen encoder the ResNet output for an image never changes, so
    running ResNet-50 on every caption of every epoch is wasted work
    (5 captions x N epochs). This dataset looks features up instead.
    """

    def __init__(
        self, features: torch.Tensor, image_index: dict[str, int], caption_ds: CaptionDataset
    ):
        self.features = features
        self.rows = [image_index[i] for i in caption_ds.imgs]
        self.encoded = caption_ds.encoded

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[self.rows[index]], self.encoded[index]


class MyCollate:
    """Stack images/features and right-pad captions with ``pad_idx``."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        imgs = torch.stack([item[0] for item in batch])
        targets = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=self.pad_idx
        )
        return imgs, targets


# Backwards compatible alias
Flickr8kDataset = CaptionDataset


def build_datasets(
    img_dir: str,
    captions_file: str,
    freq_threshold: int = 5,
    max_len: int = 40,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    augment: bool = True,
    image_size: int = 224,
    verify_images: bool = False,
):
    """Read, (optionally) verify, split, build vocab (train only), create datasets.

    Returns ``(datasets, vocab)`` with keys ``train``/``val``/``test``
    (:class:`CaptionDataset`) and ``train_images``/``val_images``/``test_images``
    (:class:`ImageDataset`)."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f"Image folder not found: {img_dir}. Download Flickr8k "
            "(https://www.kaggle.com/datasets/adityajn105/flickr8k) and set data.img_dir."
        )
    pairs = load_captions(captions_file)
    if verify_images:
        pairs = verify_image_pairs(img_dir, pairs)
    splits = split_by_image(pairs, val_frac, test_frac, seed)
    if not splits["train"]:
        raise ValueError(
            "The training split is empty; lower data.val_frac / data.test_frac "
            "or check the dataset."
        )
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(c for _, c in splits["train"])  # train only: no leakage

    train_tf = get_transforms(train=augment, image_size=image_size)
    eval_tf = get_transforms(train=False, image_size=image_size)
    datasets = {
        "train": CaptionDataset(img_dir, splits["train"], vocab, train_tf, max_len),
        "val": CaptionDataset(img_dir, splits["val"], vocab, eval_tf, max_len),
        "test": CaptionDataset(img_dir, splits["test"], vocab, eval_tf, max_len),
        "train_images": ImageDataset(img_dir, splits["train"], eval_tf),
        "val_images": ImageDataset(img_dir, splits["val"], eval_tf),
        "test_images": ImageDataset(img_dir, splits["test"], eval_tf),
    }
    return datasets, vocab


def _seed_worker(worker_id: int) -> None:
    """Reproducible per-worker NumPy / ``random`` seeds (torch seeds itself)."""
    import numpy as np

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)  # noqa: NPY002 - seeds the global RNG used by libraries
    random.seed(worker_seed)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pad_idx: int,
    collate: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """DataLoader with padding collate, pinned memory on CUDA, persistent workers,
    seeded workers and (if ``seed`` is given) a reproducible shuffle order."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=MyCollate(pad_idx) if collate else None,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def build_datasets_from_config(
    data_cfg: Any, seed: int, augment: bool
) -> tuple[dict[str, Dataset], Vocabulary]:
    """``build_datasets`` driven by a ``config.DataConfig``."""
    return build_datasets(
        data_cfg.img_dir,
        data_cfg.captions_file,
        freq_threshold=data_cfg.freq_threshold,
        max_len=data_cfg.max_len,
        val_frac=data_cfg.val_frac,
        test_frac=data_cfg.test_frac,
        seed=seed,
        augment=augment,
        image_size=data_cfg.image_size,
        verify_images=getattr(data_cfg, "verify_images", False),
    )


def get_data_loaders(
    img_dir: str,
    captions_file: str,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 2,
    **kwargs,
):
    """Backwards compatible helper: returns train/val/test loaders and vocab.

    If ``transform`` is given it is used for every split (old behaviour);
    otherwise training gets light augmentation and val/test a plain resize.
    """
    datasets, vocab = build_datasets(img_dir, captions_file, **kwargs)
    if transform is not None:
        for k in ("train", "val", "test"):
            datasets[k].transform = transform
    pad = vocab.pad_idx
    return (
        make_loader(datasets["train"], batch_size, True, num_workers, pad),
        make_loader(datasets["val"], batch_size, False, num_workers, pad),
        make_loader(datasets["test"], batch_size, False, num_workers, pad),
        vocab,
    )

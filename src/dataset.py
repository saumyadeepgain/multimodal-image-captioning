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
"""

from __future__ import annotations

import os
import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = (PAD, START, END, UNK)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text: str) -> List[str]:
    """Lower-case word tokenizer. Punctuation is dropped (it carries no
    information in Flickr8k captions and only inflates the vocabulary)."""
    return _TOKEN_RE.findall(str(text).lower())


class Vocabulary:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        self.itos: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}

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

    def numericalize(self, text: str) -> List[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(tok, unk) for tok in tokenize(text)]

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        """``<start> w1 ... wn <end>``, truncated so the total is <= max_len."""
        ids = self.numericalize(text)
        if max_len is not None:
            ids = ids[: max_len - 2]
        return [self.start_idx] + ids + [self.end_idx]

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> List[str]:
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
        return {"freq_threshold": self.freq_threshold,
                "itos": [self.itos[i] for i in range(len(self.itos))]}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocabulary":
        v = cls(d.get("freq_threshold", 5))
        v.itos = dict(enumerate(d["itos"]))
        v.stoi = {w: i for i, w in v.itos.items()}
        return v


# --------------------------------------------------------------------------- #
# Reading and splitting
# --------------------------------------------------------------------------- #
def load_captions(captions_file: str) -> List[Tuple[str, str]]:
    """Return ``[(image_name, caption), ...]``.

    Handles both common Flickr8k layouts:
      * Kaggle ``captions.txt``: header ``image,caption`` then ``img.jpg,caption``
      * Original ``Flickr8k.token.txt``: ``img.jpg#0<TAB>caption``
    Only the first separator is used, so commas inside captions are safe.
    """
    pairs: List[Tuple[str, str]] = []
    with open(captions_file, encoding="utf-8") as f:
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
            pairs.append((img.strip(), cap.strip()))
    return pairs


def split_by_image(pairs: List[Tuple[str, str]], val_frac: float = 0.1,
                   test_frac: float = 0.1, seed: int = 42):
    """Split caption pairs into train/val/test with no image in two splits."""
    images = sorted({img for img, _ in pairs})
    rng = random.Random(seed)
    rng.shuffle(images)
    n_val = int(round(len(images) * val_frac))
    n_test = int(round(len(images) * test_frac))
    val_set = set(images[:n_val])
    test_set = set(images[n_val:n_val + n_test])
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
        return T.Compose([
            T.Resize((int(image_size * 1.14), int(image_size * 1.14))),
            T.RandomCrop(image_size),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), normalize])


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class CaptionDataset(Dataset):
    """One item per (image, caption) pair. Captions are pre-encoded."""

    def __init__(self, img_dir: str, pairs: List[Tuple[str, str]], vocab: Vocabulary,
                 transform=None, max_len: int = 40, strict: bool = False):
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.strict = strict
        self.imgs = [img for img, _ in pairs]
        self.captions = [cap for _, cap in pairs]
        self.encoded = [torch.tensor(vocab.encode(c, max_len), dtype=torch.long)
                        for c in self.captions]

    def __len__(self) -> int:
        return len(self.imgs)

    def load_image(self, img_id: str) -> Image.Image:
        path = os.path.join(self.img_dir, img_id)
        if not os.path.exists(path):
            if self.strict:
                raise FileNotFoundError(path)
            return Image.new("RGB", (224, 224))
        return Image.open(path).convert("RGB")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.load_image(self.imgs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.encoded[index]


class ImageDataset(Dataset):
    """One item per *unique* image, with all its reference captions.

    Used for evaluation (BLEU must compare against all 5 references) and for
    pre-computing CNN features once per image instead of once per caption.
    """

    def __init__(self, img_dir: str, pairs: List[Tuple[str, str]], transform=None,
                 strict: bool = False):
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = os.path.join(self.img_dir, self.images[index])
        if os.path.exists(path):
            img = Image.open(path).convert("RGB")
        elif self.strict:
            raise FileNotFoundError(path)
        else:
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

    def __init__(self, features: torch.Tensor, image_index: Dict[str, int],
                 caption_ds: CaptionDataset):
        self.features = features
        self.rows = [image_index[i] for i in caption_ds.imgs]
        self.encoded = caption_ds.encoded

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[self.rows[index]], self.encoded[index]


class MyCollate:
    """Stack images/features and right-pad captions with ``pad_idx``."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        imgs = torch.stack([item[0] for item in batch])
        targets = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=self.pad_idx)
        return imgs, targets


# Backwards compatible alias
Flickr8kDataset = CaptionDataset


def build_datasets(img_dir: str, captions_file: str, freq_threshold: int = 5,
                   max_len: int = 40, val_frac: float = 0.1, test_frac: float = 0.1,
                   seed: int = 42, augment: bool = True, image_size: int = 224):
    """Read, split, build vocab (train only) and create datasets."""
    if not os.path.exists(captions_file):
        raise FileNotFoundError(
            f"{captions_file} not found. Download Flickr8k (see README).")
    splits = split_by_image(load_captions(captions_file), val_frac, test_frac, seed)
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
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                pad_idx: int, collate: bool = True, seed: Optional[int] = None) -> DataLoader:
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


def build_datasets_from_config(data_cfg: Any, seed: int, augment: bool
                               ) -> Tuple[Dict[str, Dataset], Vocabulary]:
    """``build_datasets`` driven by a ``config.DataConfig``."""
    return build_datasets(data_cfg.img_dir, data_cfg.captions_file,
                          freq_threshold=data_cfg.freq_threshold, max_len=data_cfg.max_len,
                          val_frac=data_cfg.val_frac, test_frac=data_cfg.test_frac, seed=seed,
                          augment=augment, image_size=data_cfg.image_size)


def get_data_loaders(img_dir: str, captions_file: str, transform=None, batch_size: int = 32,
                     num_workers: int = 2, **kwargs):
    """Backwards compatible helper: returns train/val/test loaders and vocab.

    If ``transform`` is given it is used for every split (old behaviour);
    otherwise training gets light augmentation and val/test a plain resize.
    """
    datasets, vocab = build_datasets(img_dir, captions_file, **kwargs)
    if transform is not None:
        for k in ("train", "val", "test"):
            datasets[k].transform = transform
    pad = vocab.pad_idx
    return (make_loader(datasets["train"], batch_size, True, num_workers, pad),
            make_loader(datasets["val"], batch_size, False, num_workers, pad),
            make_loader(datasets["test"], batch_size, False, num_workers, pad),
            vocab)

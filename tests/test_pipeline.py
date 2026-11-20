import torch

from src.dataset import Vocabulary, build_datasets, load_captions, split_by_image, tokenize
from src.inference import (
    beam_search,
    generate_caption_beam_search,
    generate_caption_greedy,
    greedy_decode,
)
from src.model import ImageCaptioningModel


def tiny_model(vocab_size, **kw):
    torch.manual_seed(0)
    return ImageCaptioningModel(32, vocab_size, num_layers=1, nhead=4, pretrained=False, **kw)


def test_tokenize_and_vocab_roundtrip():
    assert tokenize("A dog, running!  Isn't it?") == ["a", "dog", "running", "isn't", "it"]
    v = Vocabulary(freq_threshold=1)
    v.build_vocabulary(["a dog runs", "a cat sits"])
    ids = v.encode("a dog flies", max_len=10)
    assert ids[0] == v.start_idx and ids[-1] == v.end_idx
    assert v.decode(ids) == ["a", "dog", "<unk>"]
    assert Vocabulary.from_dict(v.to_dict()).stoi == v.stoi


def test_caption_parsing_keeps_commas(tiny_flickr):
    _, cap_file = tiny_flickr
    pairs = load_captions(cap_file)
    assert len(pairs) == 100
    assert pairs[0][1].endswith(", and more .")


def test_split_has_no_image_leakage(tiny_flickr):
    _, cap_file = tiny_flickr
    s = split_by_image(load_captions(cap_file), 0.2, 0.2, seed=1)
    imgs = {k: {i for i, _ in v} for k, v in s.items()}
    assert not (imgs["train"] & imgs["val"]) and not (imgs["train"] & imgs["test"])
    assert not (imgs["val"] & imgs["test"])
    assert sum(len(v) for v in s.values()) == 100


def test_frozen_encoder_keeps_batchnorm_stats():
    m = tiny_model(20)
    bn = m.encoder.resnet[1]
    before = bn.running_mean.clone()
    m.train()
    m(torch.randn(2, 3, 64, 64), torch.randint(4, 20, (2, 6)))
    assert torch.equal(before, bn.running_mean), "frozen BN statistics must not drift"
    assert not any(p.requires_grad for p in m.encoder.resnet.parameters())


def test_fine_tune_only_layer4():
    m = tiny_model(20, train_CNN=True)
    trainable = {n.split(".")[0] for n, p in m.encoder.resnet.named_parameters() if p.requires_grad}
    assert trainable == {"7"}


def test_padding_does_not_change_logits():
    m = tiny_model(20).eval()
    img = torch.randn(1, 3, 64, 64)
    cap = torch.tensor([[1, 5, 6, 7, 2]])
    padded = torch.tensor([[1, 5, 6, 7, 2, 0, 0]])
    a = m(img, cap)
    b = m(img, padded)[:, : a.size(1)]
    assert torch.allclose(a, b, atol=1e-5)


def test_greedy_batch_matches_single_and_beam1(tiny_flickr):
    m = tiny_model(20).eval()
    v = Vocabulary(1)
    v.build_vocabulary([" ".join(f"w{i}" for i in range(16))])
    imgs = torch.randn(3, 3, 64, 64)
    batch, attn = greedy_decode(m, imgs, v, max_length=8, return_attention=True)
    assert attn.shape[:2] == (3, batch.size(1) - 1) and attn.shape[2] == 4  # 64px input -> 2x2 grid
    for i in range(3):
        single, _ = greedy_decode(m, imgs[i : i + 1], v, max_length=8)
        n = single.size(1)
        assert torch.equal(single[0], batch[i, :n])
        ids, _ = beam_search(m, imgs[i : i + 1], v, beam_size=1, max_length=8)
        assert ids[:n] == single[0].tolist()[: len(ids)]


def test_beam_and_greedy_wrappers_return_attention():
    m = tiny_model(20).eval()
    v = Vocabulary(1)
    v.build_vocabulary([" ".join(f"w{i}" for i in range(16))])
    img = torch.randn(1, 3, 64, 64)
    words, attn = generate_caption_greedy(m, img, v, max_length=100)  # clamped, no crash
    assert words[0] == "<start>" and len(attn) == len(words) - 1
    words, attn = generate_caption_beam_search(
        m, img, v, beam_size=3, max_length=10, return_attention=True
    )
    assert len(attn) == len(words) - 1


def test_end_to_end_train_eval_checkpoint(tiny_flickr, tmp_path):
    from torch.utils.data import DataLoader

    from src.dataset import CachedFeatureDataset, MyCollate
    from src.evaluate import evaluate_captions
    from src.train import precompute_features, train_model
    from src.utils import load_checkpoint

    img_dir, cap_file = tiny_flickr
    ds, vocab = build_datasets(
        img_dir, cap_file, freq_threshold=1, val_frac=0.2, test_frac=0.2, image_size=64
    )
    model = tiny_model(len(vocab), pad_idx=vocab.pad_idx)
    feats = {
        s: precompute_features(model, ds[f"{s}_images"], "cpu", num_workers=0)
        for s in ("train", "val")
    }
    loaders = [
        DataLoader(
            CachedFeatureDataset(*feats[s], ds[s]),
            16,
            shuffle=s == "train",
            collate_fn=MyCollate(vocab.pad_idx),
        )
        for s in ("train", "val")
    ]
    hist = train_model(
        model,
        *loaders,
        vocab,
        num_epochs=3,
        learning_rate=3e-3,
        device="cpu",
        cached=True,
        val_images=ds["val_images"],
        out_dir=str(tmp_path / "ck"),
        val_feature_bank=feats["val"],
    )
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    m2, v2, meta = load_checkpoint(str(tmp_path / "ck" / "best_model.pth"))
    assert len(v2) == len(vocab) and "epoch" in meta
    metrics, samples = evaluate_captions(m2, ds["test_images"], v2, num_workers=0)
    assert 0.0 <= metrics["BLEU-4"] <= 1.0 and samples


def test_kv_cache_matches_full_forward():
    m = tiny_model(30).eval()
    img, cap = torch.randn(2, 3, 64, 64), torch.randint(4, 30, (2, 9))
    with torch.no_grad():
        mem = m.encode(img)
        full, full_attn = m.decoder(mem, cap, return_attention=True)
        caches = m.decoder.init_cache(mem)
        steps = [m.decoder.step(cap[:, t], t, caches, return_attention=True) for t in range(9)]
    assert torch.allclose(torch.stack([s[0] for s in steps], 1), full, atol=1e-4)
    assert torch.allclose(torch.stack([s[1] for s in steps], 1), full_attn, atol=1e-5)


def test_batched_beam_matches_per_image_beam():
    from src.inference import beam_search_batch

    m = tiny_model(20).eval()
    v = Vocabulary(1)
    v.build_vocabulary([" ".join(f"w{i}" for i in range(16))])
    imgs = torch.randn(3, 3, 64, 64)
    batch_ids, _ = beam_search_batch(m, imgs, v, beam_size=4, max_length=10)
    for i in range(3):
        single, _ = beam_search(m, imgs[i : i + 1], v, beam_size=4, max_length=10)
        n = len(single)
        assert batch_ids[i][:n] == single


def test_initial_loss_is_near_uniform():
    """Guards against the tied-weight init bug: an untrained model should be
    close to a uniform guess, i.e. loss ~ log(vocab_size)."""
    import math

    V = 500
    m = ImageCaptioningModel(256, V, num_layers=2, nhead=4, pretrained=False).eval()
    cap = torch.randint(4, V, (4, 12))
    logits = m(torch.randn(4, 3, 64, 64), cap)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), cap[:, 1:].reshape(-1))
    assert loss.item() < 1.5 * math.log(V)

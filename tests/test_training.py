"""Tests for config, reproducibility, resume and gradient accumulation."""
import copy

import pytest
import torch

from src.config import Config, load_config
from src.dataset import CachedFeatureDataset, build_datasets, make_loader
from src.model import ImageCaptioningModel
from src.train import fit, precompute_features, run, train_epoch, warmup_cosine


def test_config_yaml_and_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("train:\n  lr: 0.001\nmodel:\n  num_layers: 2\n")
    cfg = load_config(str(p), ["train.lr=5e-4", "model.train_CNN=true", "train.resume=none"])
    assert cfg.train.lr == 5e-4 and cfg.model.num_layers == 2 and cfg.model.train_CNN is True
    assert cfg.train.resume is None and cfg.train.epochs == Config().train.epochs
    assert Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(KeyError):
        load_config(None, ["train.not_a_key=1"])


def test_fine_tune_keeps_frozen_stage_batchnorm_fixed():
    torch.manual_seed(0)
    m = ImageCaptioningModel(32, 20, 1, 4, train_CNN=True, pretrained=False)
    m.train()
    frozen_bn, tuned_bn = m.encoder.resnet[1], m.encoder.resnet[7][0].bn1
    assert not frozen_bn.training and tuned_bn.training
    before_frozen = frozen_bn.running_mean.clone()
    before_tuned = tuned_bn.running_mean.clone()
    m(torch.randn(2, 3, 64, 64), torch.randint(4, 20, (2, 6)))
    assert torch.equal(before_frozen, frozen_bn.running_mean)
    assert not torch.equal(before_tuned, tuned_bn.running_mean)


def _setup(tiny_flickr, seed=0, batch_size=8):
    img_dir, cap_file = tiny_flickr
    ds, vocab = build_datasets(img_dir, cap_file, freq_threshold=1, val_frac=0.2,
                               test_frac=0.2, image_size=64, seed=seed)
    torch.manual_seed(seed)
    model = ImageCaptioningModel(32, len(vocab), 1, 4, pretrained=False, dropout=0.0,
                                 pad_idx=vocab.pad_idx)
    banks = {s: precompute_features(model, ds[f"{s}_images"], "cpu", num_workers=0)
             for s in ("train", "val")}
    train = make_loader(CachedFeatureDataset(*banks["train"], ds["train"]), batch_size, True, 0,
                        vocab.pad_idx, seed=seed)
    val = make_loader(CachedFeatureDataset(*banks["val"], ds["val"]), batch_size, False, 0,
                      vocab.pad_idx)
    return model, vocab, train, val


def _cfg(out_dir, epochs):
    cfg = Config()
    cfg.train.epochs, cfg.train.out_dir, cfg.train.lr = epochs, str(out_dir), 3e-3
    cfg.train.select_metric, cfg.train.patience, cfg.train.amp = "val_loss", 100, False
    return cfg


def test_resume_is_bit_exact(tiny_flickr, tmp_path):
    model, vocab, train, val = _setup(tiny_flickr)
    init = copy.deepcopy(model.state_dict())
    full = fit(_cfg(tmp_path / "full", 3), model, train, val, vocab, "cpu", cached=True)
    final_full = copy.deepcopy(model.state_dict())

    model.load_state_dict(init)
    interrupted = _cfg(tmp_path / "part", 3)
    interrupted.train.stop_after_epochs = 2          # simulate a job killed after epoch 2
    fit(interrupted, model, train, val, vocab, "cpu", cached=True)
    cfg = _cfg(tmp_path / "part", 3)
    cfg.train.resume = str(tmp_path / "part" / "last.pth")
    resumed = fit(cfg, model, train, val, vocab, "cpu", cached=True)

    assert resumed["train_loss"] == full["train_loss"]
    for k in final_full:
        assert torch.equal(final_full[k], model.state_dict()[k]), k


def test_grad_accumulation_matches_large_batch(tiny_flickr):
    """2 x 4 accumulated == 1 x 8 when captions have equal token counts."""
    torch.manual_seed(0)
    V = 20
    feats = torch.randn(8, 49, 2048)
    caps = torch.randint(4, V, (8, 7))
    caps[:, 0] = 1

    def one_step(batch_size, accum):
        torch.manual_seed(0)
        m = ImageCaptioningModel(32, V, 1, 4, pretrained=False, dropout=0.0)
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        sched = warmup_cosine(opt, 0, 10)
        loader = [(feats[i:i + batch_size], caps[i:i + batch_size]) for i in range(0, 8, batch_size)]
        crit = torch.nn.CrossEntropyLoss(ignore_index=0)
        train_epoch(m, loader, crit, opt, sched, torch.device("cpu"), cached=True,
                    grad_accum_steps=accum)
        return m.state_dict()

    a, b = one_step(8, 1), one_step(4, 2)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_run_from_config_writes_artifacts(tiny_flickr, tmp_path):
    img_dir, cap_file = tiny_flickr
    cfg = load_config(None, [f"data.img_dir={img_dir!r}", f"data.captions_file={cap_file!r}",
                             "data.freq_threshold=1", "data.image_size=64", "data.num_workers=0",
                             "data.val_frac=0.2", "data.test_frac=0.2",
                             "model.embed_size=32", "model.num_layers=1", "model.nhead=4",
                             "model.pretrained=false", "train.epochs=2", "train.amp=false",
                             f"train.out_dir={str(tmp_path / 'run')!r}"])
    hist = run(cfg)
    assert len(hist["train_loss"]) == 2 and hist["val_cider"][0] is not None
    for f in ("best_model.pth", "last.pth", "history.json", "config.yaml"):
        assert (tmp_path / "run" / f).exists(), f
    logs = list((tmp_path / "run" / "logs").glob("train_*.log"))
    assert len(logs) == 1 and "Epoch 02/2" in logs[0].read_text()

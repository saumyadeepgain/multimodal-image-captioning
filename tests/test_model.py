"""Model tests on synthetic tensors: construction, shapes, finiteness, gradient flow.

All models use ``pretrained=False`` so no ImageNet weights are downloaded.
Shapes: B = batch, T = caption length, S = image regions, E = embed size, V = vocab.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.config import ModelConfig
from src.model import BACKBONE_DIM, ImageCaptioningModel

B, T, V, E = 3, 9, 50, 64
PAD = 0


def make_model(**overrides) -> ImageCaptioningModel:
    torch.manual_seed(0)
    kwargs = {
        "embed_size": E,
        "vocab_size": V,
        "num_layers": 2,
        "nhead": 4,
        "pretrained": False,
        "pad_idx": PAD,
    }
    kwargs.update(overrides)
    return ImageCaptioningModel(**kwargs)


def dummy_captions(batch: int = B, length: int = T) -> torch.Tensor:
    """``<start> w ... w <end> <pad>...`` with different lengths per row."""
    caps = torch.randint(4, V, (batch, length))
    caps[:, 0] = 1  # <start>
    for i in range(batch):
        end = length - 1 - i  # rows end at different places
        caps[i, end] = 2  # <end>
        caps[i, end + 1 :] = PAD
    return caps


def grid(image_size: int) -> int:
    return math.ceil(image_size / 32) ** 2  # ResNet-50 downsamples by 32


# --------------------------------------------------------------------------- init
def test_initialisation_and_config():
    m = make_model()
    assert m.config["vocab_size"] == V and m.config["embed_size"] == E
    assert len(m.decoder.layers) == 2
    assert m.decoder.fc_out.out_features == V
    # Frozen backbone: no trainable ResNet parameters, everything else trainable.
    assert not any(p.requires_grad for p in m.encoder.resnet.parameters())
    assert all(
        p.requires_grad for n, p in m.named_parameters() if not n.startswith("encoder.resnet")
    )
    # Padding row of the embedding starts at zero.
    assert torch.count_nonzero(m.decoder.embed.weight[PAD]) == 0


def test_from_config_matches_constructor():
    cfg = ModelConfig(embed_size=E, num_layers=2, nhead=4, pretrained=False)
    m = ImageCaptioningModel.from_config(cfg, V, PAD)
    assert m.config["num_layers"] == 2 and m.config["vocab_size"] == V


def test_fine_tune_unfreezes_only_layer4():
    m = make_model(train_CNN=True)
    stages = {n.split(".")[0] for n, p in m.encoder.resnet.named_parameters() if p.requires_grad}
    assert stages == {"7"}


# ------------------------------------------------------------------------- shapes
@pytest.mark.parametrize("image_size", [64, 224])
def test_forward_shapes_from_images(image_size):
    m = make_model().eval()
    images = torch.randn(B, 3, image_size, image_size)
    caps = dummy_captions()
    S = grid(image_size)
    with torch.no_grad():
        memory = m.encode(images)
        logits, attn = m(images, caps, return_attention=True)
    assert memory.shape == (B, S, E)
    assert logits.shape == (B, T - 1, V)  # teacher forcing drops last token
    assert attn.shape == (B, T - 1, S)
    assert torch.allclose(attn.sum(-1), torch.ones(B, T - 1), atol=1e-5)  # softmax rows


def test_forward_shapes_from_cached_features():
    m = make_model().eval()
    feats = torch.randn(B, 49, BACKBONE_DIM)
    with torch.no_grad():
        backbone = m.encoder.backbone(torch.randn(B, 3, 224, 224))
        logits = m(captions=dummy_captions(), backbone_features=feats)
        fp16 = m(captions=dummy_captions(), backbone_features=feats.half())
    assert backbone.shape == (B, 49, BACKBONE_DIM)
    assert logits.shape == fp16.shape == (B, T - 1, V)


def test_decoder_step_shapes():
    m = make_model().eval()
    memory = m.encode(torch.randn(B, 3, 64, 64))
    caches = m.decoder.init_cache(memory)
    logits, attn = m.decoder.step(torch.full((B,), 1), 0, caches, return_attention=True)
    assert logits.shape == (B, V) and attn.shape == (B, memory.size(1))
    assert caches[0]["k"].shape == (B, 4, 1, E // 4)  # (B, heads, t, head_dim)


def test_too_long_sequence_raises():
    m = make_model(max_seq_length=8).eval()
    with pytest.raises(ValueError, match="exceeds max_len"):
        m(torch.randn(1, 3, 64, 64), dummy_captions(1, 12))


# ------------------------------------------------------ loss, finiteness, gradients
@pytest.mark.parametrize(
    "overrides", [{}, {"tie_weights": True}, {"num_layers": 1}, {"train_CNN": True}]
)
def test_forward_backward_finite_and_gradients_flow(overrides):
    m = make_model(**overrides).train()
    images = torch.randn(B, 3, 64, 64)
    caps = dummy_captions()
    logits = m(images, caps)
    loss = F.cross_entropy(logits.reshape(-1, V), caps[:, 1:].reshape(-1), ignore_index=PAD)
    assert torch.isfinite(loss), loss
    assert loss.item() < 2 * math.log(V)  # sane initial loss
    loss.backward()
    trainable = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    assert trainable
    missing = [n for n, p in trainable if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    non_finite = [n for n, p in trainable if not torch.isfinite(p.grad).all()]
    assert not non_finite, f"NaN/Inf gradients in: {non_finite}"
    zero = [n for n, p in trainable if torch.count_nonzero(p.grad) == 0]
    assert not zero, f"all-zero gradients in: {zero}"
    frozen = [n for n, p in m.named_parameters() if not p.requires_grad and p.grad is not None]
    assert not frozen, f"frozen parameters received gradients: {frozen}"


def test_padding_positions_get_no_embedding_gradient():
    m = make_model().train()
    caps = dummy_captions()
    logits = m(torch.randn(B, 3, 64, 64), caps)
    F.cross_entropy(logits.reshape(-1, V), caps[:, 1:].reshape(-1), ignore_index=PAD).backward()
    assert torch.count_nonzero(m.decoder.embed.weight.grad[PAD]) == 0


def test_eval_is_deterministic_and_train_uses_dropout():
    m = make_model(dropout=0.3)
    images, caps = torch.randn(B, 3, 64, 64), dummy_captions()
    m.eval()
    with torch.no_grad():
        assert torch.equal(m(images, caps), m(images, caps))
    m.train()
    with torch.no_grad():
        assert not torch.equal(m(images, caps), m(images, caps))


def test_bf16_autocast_forward_is_finite():
    m = make_model().eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = m(torch.randn(B, 3, 64, 64), dummy_captions())
    assert torch.isfinite(logits.float()).all()


def test_param_groups_cover_every_trainable_parameter_once():
    m = make_model(train_CNN=True, tie_weights=True)
    groups = m.param_groups(lr=1e-3, weight_decay=0.01, encoder_lr_mult=0.1)
    ids = [id(p) for g in groups for p in g["params"]]
    expected = {id(p) for p in m.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == expected
    cnn_group = [g for g in groups if g["lr"] == pytest.approx(1e-4)]
    assert cnn_group and all(g["weight_decay"] == 0.01 for g in cnn_group)

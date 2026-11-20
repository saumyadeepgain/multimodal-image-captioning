"""ResNet-50 encoder + Transformer decoder with exposed cross-attention.

Changes compared with the first version (details in docs/IMPROVEMENTS.pdf):

* **Frozen BatchNorm really frozen.** ``requires_grad=False`` does not stop
  BatchNorm from updating its running mean/var in ``model.train()``; the old
  encoder silently drifted away from its ImageNet statistics while "frozen".
  The backbone is now kept in eval mode unless it is being fine-tuned.
* **No autograd graph for a frozen backbone** (``torch.no_grad``), which cuts
  activation memory and backward time.
* **Spatial position embeddings** for the 7x7 grid. Cross-attention is
  permutation invariant, so without them the decoder only knows *what* is in
  each region, not *where* the region is.
* **Pre-LayerNorm** decoder layers + GELU: more stable to train without long
  warm-up than the post-LN layout.
* **Key padding mask** on decoder self-attention, and a cached boolean
  causal mask (the old code rebuilt a float mask on every forward).
* **Attention weights only when asked for** (``need_weights``), letting
  PyTorch use the fused scaled-dot-product kernel during training.
* **Weight tying** between the token embedding and output projection
  (fewer parameters, usually better perplexity on small datasets).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models

BACKBONE_DIM = 2048


class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int, train_CNN: bool = False, pretrained: bool = True,
                 dropout: float = 0.1, grid_size: int = 7):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Drop avgpool + fc to keep the 7x7 spatial map.
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.train_CNN = train_CNN
        for name, p in self.resnet.named_parameters():
            # Only the last stage ("7.*" == layer4) is ever fine-tuned.
            p.requires_grad = bool(train_CNN and name.startswith("7."))

        self.project = nn.Sequential(
            nn.Linear(BACKBONE_DIM, embed_size),
            nn.LayerNorm(embed_size),
            nn.Dropout(dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, embed_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_CNN:
            self.resnet.eval()  # keep BatchNorm statistics frozen
        return self

    def backbone(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, 49, 2048) raw ResNet features."""
        if self.train_CNN and self.training:
            feats = self.resnet(images)
        else:
            with torch.no_grad():
                feats = self.resnet(images)
        return feats.flatten(2).transpose(1, 2)

    def project_features(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, 49, 2048) backbone features -> (B, 49, embed_size) memory."""
        x = self.project(feats.float())
        return x + self.pos_embed[:, : x.size(1)]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.project_features(self.backbone(images))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)]


class CustomTransformerDecoderLayer(nn.Module):
    """Pre-LN decoder layer that can return cross-attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None,
                need_weights: bool = False):
        x = self.norm1(tgt)
        sa, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                               key_padding_mask=tgt_key_padding_mask, need_weights=False)
        tgt = tgt + self.dropout1(sa)

        x = self.norm2(tgt)
        ca, weights = self.multihead_attn(x, memory, memory, need_weights=need_weights,
                                          average_attn_weights=True)
        tgt = tgt + self.dropout2(ca)

        tgt = tgt + self.dropout3(self.ff(self.norm3(tgt)))
        return tgt, weights


class DecoderTransformer(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int, num_layers: int = 3, nhead: int = 8,
                 max_seq_length: int = 64, dropout: float = 0.1, dim_feedforward: Optional[int] = None,
                 pad_idx: int = 0, tie_weights: bool = True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        self.emb_dropout = nn.Dropout(dropout)
        ff = dim_feedforward or 4 * embed_size
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(embed_size, nhead, ff, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        if tie_weights:
            self.fc_out.weight = self.embed.weight
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.pad_idx = pad_idx
        causal = torch.triu(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool), 1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Boolean mask, True = position may NOT be attended to."""
        return self.causal_mask[:sz, :sz]

    def forward(self, features, captions, return_attention: bool = False,
                padding_mask: Optional[torch.Tensor] = None):
        # features: (B, 49, E) memory; captions: (B, T) token ids
        T_ = captions.size(1)
        x = self.embed(captions) * math.sqrt(self.embed_size)
        x = self.emb_dropout(self.pos_encoder(x))
        mask = self.generate_square_subsequent_mask(T_)
        if padding_mask is None:
            padding_mask = captions.eq(self.pad_idx)
        attn = None
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            x, w = layer(x, features, tgt_mask=mask, tgt_key_padding_mask=padding_mask,
                         need_weights=return_attention and last)
            if w is not None:
                attn = w
        out = self.fc_out(self.final_norm(x))
        return (out, attn) if return_attention else out


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size: int, vocab_size: int, num_layers: int = 3, nhead: int = 8,
                 train_CNN: bool = False, pretrained: bool = True, dropout: float = 0.1,
                 max_seq_length: int = 64, pad_idx: int = 0, tie_weights: bool = True):
        super().__init__()
        self.config = dict(embed_size=embed_size, vocab_size=vocab_size, num_layers=num_layers,
                           nhead=nhead, train_CNN=train_CNN, dropout=dropout,
                           max_seq_length=max_seq_length, pad_idx=pad_idx,
                           tie_weights=tie_weights)
        self.encoder = EncoderCNN(embed_size, train_CNN, pretrained=pretrained, dropout=dropout)
        self.decoder = DecoderTransformer(embed_size, vocab_size, num_layers, nhead,
                                          max_seq_length=max_seq_length, dropout=dropout,
                                          pad_idx=pad_idx, tie_weights=tie_weights)

    def encode(self, images=None, backbone_features=None):
        if backbone_features is not None:
            return self.encoder.project_features(backbone_features)
        return self.encoder(images)

    def forward(self, images=None, captions=None, return_attention: bool = False,
                backbone_features=None):
        """Teacher forcing: decoder input is ``captions[:, :-1]`` and the
        logits are aligned with targets ``captions[:, 1:]``."""
        memory = self.encode(images, backbone_features)
        return self.decoder(memory, captions[:, :-1], return_attention)

    def param_groups(self, lr: float, weight_decay: float, encoder_lr_mult: float = 0.1):
        """AdamW groups: no decay on biases/norms/embeddings, lower LR for the CNN."""
        decay, no_decay, cnn = [], [], []
        seen = set()
        for name, p in self.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if name.startswith("encoder.resnet."):
                cnn.append(p)
            elif p.ndim < 2 or "embed" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "lr": lr, "weight_decay": weight_decay},
                  {"params": no_decay, "lr": lr, "weight_decay": 0.0}]
        if cnn:
            groups.append({"params": cnn, "lr": lr * encoder_lr_mult, "weight_decay": weight_decay})
        return groups

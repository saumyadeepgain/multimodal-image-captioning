"""ResNet-50 encoder + Transformer decoder with exposed cross-attention.

Changes compared with the first version (details in docs/IMPROVEMENTS.pdf):

* **Frozen BatchNorm really frozen.** ``requires_grad=False`` does not stop
  BatchNorm from updating its running mean/var in ``model.train()``; the old
  encoder silently drifted away from its ImageNet statistics while "frozen".
  The backbone is now kept in eval mode unless it is being fine-tuned.
* **No autograd graph for a frozen backbone** (``torch.no_grad``), which cuts
  activation memory and backward time.
* **Spatial position embeddings** for the 7x7 grid. Cross-attention is
  permutation invariant, so in principle the decoder cannot tell regions apart
  by location. NOTE: in our synthetic ablation (docs/ablation_results.json) this
  made no measurable difference -- ResNet features already leak position via
  zero padding -- but it costs only 49*d parameters, so it is kept.
* **Pre-LayerNorm** decoder layers + GELU: more stable to train without long
  warm-up than the post-LN layout.
* **Key padding mask** on decoder self-attention, and a cached boolean
  causal mask (the old code rebuilt a float mask on every forward).
* **Attention weights only when asked for** (``need_weights``), letting
  PyTorch use the fused scaled-dot-product kernel during training.
* **Scaled embedding init** N(0, d^-0.5). Optional weight tying
  (``tie_weights=True``, with T5-style d^-0.5 output scaling) is available but
  OFF by default: in a controlled ablation (docs/ablation_results.json) the
  untied output layer reached a clearly lower validation loss.
* **KV-cache for generation** (``init_cache`` / ``step``): each new token
  attends to cached keys/values instead of re-running the whole prefix, so
  generating T tokens costs O(T) decoder passes of length 1 instead of
  O(T^2) work; image keys/values are projected once per caption.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

BACKBONE_DIM = 2048


class EncoderCNN(nn.Module):
    def __init__(
        self,
        embed_size: int,
        train_CNN: bool = False,
        pretrained: bool = True,
        dropout: float = 0.1,
        grid_size: int = 7,
    ):
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

    def train(self, mode: bool = True) -> EncoderCNN:
        """Frozen ResNet stages always stay in eval mode, so their BatchNorm
        running statistics never change. When fine-tuning, only layer4 (child
        "7") is switched to train mode -- the earlier version put the *whole*
        ResNet in train mode, so layers 1-3 drifted although they were frozen."""
        super().train(mode)
        if mode:
            for name, stage in self.resnet.named_children():
                if not (self.train_CNN and name == "7"):
                    stage.eval()
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

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Full-sequence pass (training / scoring). Returns (output, cross-attn or None)."""
        x = self.norm1(tgt)
        sa, _ = self.self_attn(
            x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False
        )
        tgt = tgt + self.dropout1(sa)

        x = self.norm2(tgt)
        ca, weights = self.multihead_attn(
            x, memory, memory, need_weights=need_weights, average_attn_weights=True
        )
        tgt = tgt + self.dropout2(ca)

        tgt = tgt + self.dropout3(self.ff(self.norm3(tgt)))
        return tgt, weights

    # ----------------------------------------------------------- KV-cache path
    @staticmethod
    def _heads(x: torch.Tensor, nhead: int) -> torch.Tensor:
        B, T, E = x.shape
        return x.view(B, T, nhead, E // nhead).transpose(1, 2)  # (B, H, T, Dh)

    def project_memory(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-attention keys/values for the image, computed once."""
        mha = self.multihead_attn
        E = memory.size(-1)
        w, b = mha.in_proj_weight, mha.in_proj_bias
        k = F.linear(memory, w[E : 2 * E], b[E : 2 * E])
        v = F.linear(memory, w[2 * E :], b[2 * E :])
        return self._heads(k, mha.num_heads), self._heads(v, mha.num_heads)

    def step(
        self, x: torch.Tensor, cache: dict[str, Optional[torch.Tensor]], need_weights: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """One token (B, 1, E) using/extending ``cache`` (eval mode only)."""
        sa_mha, ca_mha = self.self_attn, self.multihead_attn
        H = sa_mha.num_heads
        h = self.norm1(x)
        q, k, v = F.linear(h, sa_mha.in_proj_weight, sa_mha.in_proj_bias).chunk(3, dim=-1)
        q, k, v = self._heads(q, H), self._heads(k, H), self._heads(v, H)
        cache["k"] = k if cache.get("k") is None else torch.cat([cache["k"], k], 2)
        cache["v"] = v if cache.get("v") is None else torch.cat([cache["v"], v], 2)
        a = F.scaled_dot_product_attention(q, cache["k"], cache["v"])  # past only: no mask
        x = x + sa_mha.out_proj(a.transpose(1, 2).flatten(2))

        h = self.norm2(x)
        E = h.size(-1)
        q = self._heads(F.linear(h, ca_mha.in_proj_weight[:E], ca_mha.in_proj_bias[:E]), H)
        weights = None
        if need_weights:
            scores = (q @ cache["mk"].transpose(-1, -2)) / math.sqrt(q.size(-1))
            p = scores.softmax(-1)
            a = p @ cache["mv"]
            weights = p.mean(1)  # (B, 1, S) averaged over heads, like nn.MultiheadAttention
        else:
            a = F.scaled_dot_product_attention(q, cache["mk"], cache["mv"])
        x = x + ca_mha.out_proj(a.transpose(1, 2).flatten(2))
        x = x + self.ff(self.norm3(x))
        return x, weights


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        num_layers: int = 3,
        nhead: int = 8,
        max_seq_length: int = 64,
        dropout: float = 0.1,
        dim_feedforward: Optional[int] = None,
        pad_idx: int = 0,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embed_size, max_seq_length)
        self.emb_dropout = nn.Dropout(dropout)
        ff = dim_feedforward or 4 * embed_size
        self.layers = nn.ModuleList(
            [
                CustomTransformerDecoderLayer(embed_size, nhead, ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        if tie_weights:
            self.fc_out.weight = self.embed.weight
        # Scaled init: with tied weights the default N(0, 1) embedding makes the
        # initial logits have std ~sqrt(d) (loss far above log|V|). N(0, d^-0.5)
        # gives unit-scale inputs after the sqrt(d) multiplier and sane logits.
        nn.init.normal_(self.embed.weight, mean=0.0, std=embed_size**-0.5)
        with torch.no_grad():
            self.embed.weight[pad_idx].zero_()
        nn.init.zeros_(self.fc_out.bias)
        # As in T5: with tied weights, rescale the decoder output by d^-0.5,
        # otherwise each position's own input embedding (still in the residual
        # stream) yields a logit of ~sqrt(d) for "repeat the current word".
        self.out_scale = embed_size**-0.5 if tie_weights else 1.0
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.pad_idx = pad_idx
        causal = torch.triu(torch.ones(max_seq_length, max_seq_length, dtype=torch.bool), 1)
        self.register_buffer("causal_mask", causal, persistent=False)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Boolean mask, True = position may NOT be attended to."""
        return self.causal_mask[:sz, :sz]

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
        return_attention: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Teacher-forced decoding. features (B, S, E), captions (B, T) -> logits (B, T, V)
        [, last-layer head-averaged cross-attention (B, T, S)]."""
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
            x, w = layer(
                x,
                features,
                tgt_mask=mask,
                tgt_key_padding_mask=padding_mask,
                need_weights=return_attention and last,
            )
            if w is not None:
                attn = w
        out = self.fc_out(self.final_norm(x) * self.out_scale)
        return (out, attn) if return_attention else out

    def init_cache(self, memory: torch.Tensor) -> list[dict[str, Optional[torch.Tensor]]]:
        """Per-layer cache with pre-projected image keys/values."""
        caches = []
        for layer in self.layers:
            mk, mv = layer.project_memory(memory)
            caches.append({"mk": mk, "mv": mv, "k": None, "v": None})
        return caches

    @staticmethod
    def reorder_cache(
        caches: list[dict[str, Optional[torch.Tensor]]], index: torch.Tensor
    ) -> list[dict[str, Optional[torch.Tensor]]]:
        """Select batch rows (used by beam search when beams are re-ranked)."""
        for c in caches:
            for key in ("mk", "mv", "k", "v"):
                if c[key] is not None:
                    c[key] = c[key].index_select(0, index)
        return caches

    def step(
        self,
        tokens: torch.Tensor,
        position: int,
        caches: list[dict[str, Optional[torch.Tensor]]],
        return_attention: bool = False,
    ):
        """Logits for the next token given the last tokens (B,) at ``position``."""
        if position >= self.max_seq_length:
            raise ValueError(f"position {position} exceeds max_len {self.max_seq_length}")
        x = self.embed(tokens[:, None]) * math.sqrt(self.embed_size)
        x = x + self.pos_encoder.pe[:, position : position + 1]
        attn = None
        for i, (layer, cache) in enumerate(zip(self.layers, caches)):
            last = i == len(self.layers) - 1
            x, w = layer.step(x, cache, need_weights=return_attention and last)
            if w is not None:
                attn = w[:, 0]
        logits = self.fc_out(self.final_norm(x) * self.out_scale)[:, 0]
        return (logits, attn) if return_attention else logits


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,
        embed_size: int,
        vocab_size: int,
        num_layers: int = 3,
        nhead: int = 8,
        train_CNN: bool = False,
        pretrained: bool = True,
        dropout: float = 0.1,
        max_seq_length: int = 64,
        pad_idx: int = 0,
        tie_weights: bool = False,
    ):
        super().__init__()
        self.config = dict(
            embed_size=embed_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
            nhead=nhead,
            train_CNN=train_CNN,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pad_idx=pad_idx,
            tie_weights=tie_weights,
        )
        self.encoder = EncoderCNN(embed_size, train_CNN, pretrained=pretrained, dropout=dropout)
        self.decoder = DecoderTransformer(
            embed_size,
            vocab_size,
            num_layers,
            nhead,
            max_seq_length=max_seq_length,
            dropout=dropout,
            pad_idx=pad_idx,
            tie_weights=tie_weights,
        )

    @classmethod
    def from_config(cls, model_cfg, vocab_size: int, pad_idx: int) -> ImageCaptioningModel:
        """Build from a ``config.ModelConfig``."""
        return cls(
            embed_size=model_cfg.embed_size,
            vocab_size=vocab_size,
            num_layers=model_cfg.num_layers,
            nhead=model_cfg.nhead,
            train_CNN=model_cfg.train_CNN,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            max_seq_length=model_cfg.max_seq_length,
            pad_idx=pad_idx,
            tie_weights=model_cfg.tie_weights,
        )

    def encode(
        self,
        images: Optional[torch.Tensor] = None,
        backbone_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Image memory (B, 49, E) from raw images or cached backbone features."""
        if backbone_features is not None:
            return self.encoder.project_features(backbone_features)
        return self.encoder(images)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        captions: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        backbone_features: Optional[torch.Tensor] = None,
    ):
        """Teacher forcing: decoder input is ``captions[:, :-1]`` and the
        logits are aligned with targets ``captions[:, 1:]``."""
        memory = self.encode(images, backbone_features)
        return self.decoder(memory, captions[:, :-1], return_attention)

    def param_groups(
        self, lr: float, weight_decay: float, encoder_lr_mult: float = 0.1
    ) -> list[dict[str, Any]]:
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
        groups = [
            {"params": decay, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
        if cnn:
            groups.append({"params": cnn, "lr": lr * encoder_lr_mult, "weight_decay": weight_decay})
        return groups

# 🖼️ Multimodal Image Captioning — CNN Encoder + Transformer Decoder

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/ResNet--50-CNN%20Encoder-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Transformer-Decoder-informational?style=for-the-badge"/>
</p>

<p align="center">
  <b>Image captioning with a ResNet-50 encoder and a Transformer decoder with cross-attention,</b><br>
  plus per-word attention heat-maps that show where the model "looks" for each word.
</p>

---

## 📌 Project overview

| | |
|---|---|
| **Task** | Image captioning (vision → language), Flickr8k: 8,091 images × 5 captions |
| **Framework** | PyTorch ≥ 2.1 |
| **Encoder** | ResNet-50 (ImageNet), frozen by default (BatchNorm kept in eval mode); optional layer4 fine-tuning |
| **Decoder** | Pre-LN Transformer: causal self-attention + cross-attention over the 7×7 = 49 image regions |
| **Decoding** | KV-cached greedy and batched beam search (GNMT length penalty), both return attention maps |
| **Metrics** | Corpus BLEU-1…4, CIDEr-D, ROUGE-L against all 5 references (built in, verified against `pycocoevalcap`); optional METEOR |
| **Engineering** | YAML/dataclass config, CNN-feature cache, AMP, gradient accumulation, exact resume, 20 tests + CI |

> Reference: Xu et al., *Show, Attend and Tell* (2015) — [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

📄 **[`docs/IMPROVEMENTS.pdf`](docs/IMPROVEMENTS.pdf)** is a step-by-step report of every
process in the pipeline: what the original code did, what was wrong, what replaced it and why,
with measured benchmarks and ablations.

---

## 🏗️ Architecture

```
Image 3×224×224
   │  ResNet-50 (frozen, BatchNorm in eval mode; optional layer4 fine-tune)
   ▼
2048×7×7  ──flatten──►  49×2048  ──Linear+LayerNorm──►  49×d  + learned 2-D position embedding
                                                           │  (memory: keys / values)
Caption ids ─► Embedding×√d + sinusoidal PE                │
   │                                                       │
   ▼          ┌───────────── N × Pre-LN decoder layer ─────┼────────┐
              │ LN → masked self-attention (causal + pad)  │        │
              │ LN → cross-attention  Q=text  K,V=image ◄──┘        │
              │ LN → FFN (GELU)                                     │
              └─────────────────────────────────────────────────────┘
   ▼
LayerNorm → Linear → next-token logits (raw logits → CrossEntropyLoss)
```

---

## 🗂️ Folder structure

```
.
├── configs/
│   └── default.yaml          # every hyper-parameter, with defaults
├── src/
│   ├── config.py             # typed dataclass config, YAML loading, CLI overrides
│   ├── dataset.py            # caption parsing, leak-free split, Vocabulary, datasets, loaders
│   ├── model.py              # EncoderCNN, Pre-LN Transformer decoder (+ KV-cache)
│   ├── train.py              # training loop: AMP, grad accumulation, early stopping, resume
│   ├── evaluate.py           # corpus evaluation CLI (BLEU, CIDEr-D, ROUGE-L, METEOR)
│   ├── inference.py          # KV-cached greedy and batched beam search
│   ├── metrics.py            # dependency-free CIDEr-D and ROUGE-L
│   ├── predict.py            # caption image files from the command line
│   ├── attention_viz.py      # attention overlays + attention entropy
│   └── utils.py              # seeding, device, logging, checkpoint save/load
├── tests/                    # pytest: pipeline, metrics, training/resume (CPU, synthetic data)
├── notebooks/
│   └── image_captioning.ipynb   # EDA, diagnostics, evaluation, attention plots
├── docs/
│   ├── IMPROVEMENTS.pdf      # step-by-step report (built by build_report.py)
│   ├── benchmark_*.py/json   # reproducible speed benchmarks
│   └── ablation.py/…json     # controlled ablation study
├── app.py                    # Gradio demo
├── pyproject.toml            # package metadata, extras, console scripts, pytest config
├── requirements.txt / requirements-dev.txt
└── .github/workflows/tests.yml
```

Generated at run time (git-ignored): `data/`, `checkpoints/` (`best_model.pth`, `last.pth`,
`config.yaml`, `history.json`, `train.log`, `eval_results.json`).

---

## ⚙️ Setup & installation

```bash
git clone https://github.com/saumyadeepgain/multimodal-image-captioning.git
cd multimodal-image-captioning
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt                    # or: pip install -e ".[demo,dev]"
```

Download **Flickr8k** from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract to:

```
data/archive/Images/        ← 8,091 .jpg files
data/archive/captions.txt   ← "image,caption" (the original Flickr8k.token.txt also works)
```

---

## 🚀 Training

```bash
python -m src.train                                   # built-in defaults = configs/default.yaml
python -m src.train --config configs/default.yaml --set train.epochs=30 train.lr=1e-4
```

Every hyper-parameter lives in `configs/default.yaml` (typed in `src/config.py`); override any
of them with `--set section.key=value`. Useful ones:

| Override | Effect |
|---|---|
| `model.train_CNN=true` | fine-tune ResNet layer4 (disables feature caching, enables augmentation) |
| `train.grad_accum_steps=4` | effective batch = `batch_size × 4` with the memory of one batch |
| `train.select_metric=CIDEr` | choose the best checkpoint by val CIDEr (`BLEU-4`, `CIDEr` or `val_loss`) |
| `train.bleu_limit=500` | decode only 500 val images per epoch for model selection (faster) |
| `train.stop_after_epochs=5` | stop this process after 5 epochs (time-limited jobs) |
| `train.resume=checkpoints/last.pth` | continue exactly where a run stopped |
| `deterministic=true` | deterministic cuDNN kernels for bit-exact reruns |

With the default frozen encoder, ResNet-50 features are computed **once per image** and cached
(fp16, ≈1.2 GB RAM for the train split), so every epoch only runs the decoder.

**Outputs** in `train.out_dir` (default `checkpoints/`):
`best_model.pth` (weights + model config + vocabulary + full experiment config),
`last.pth` (adds optimizer, scheduler, AMP scaler and RNG states for exact resume),
`config.yaml`, `history.json` (loss, perplexity, BLEU-4, CIDEr, LR, grad-norm, time per epoch)
and `train.log`.

**Resuming** is bit-exact: an interrupted and resumed run produces the same weights as an
uninterrupted one (tested in `tests/test_training.py`).

```bash
python -m src.train --config checkpoints/config.yaml --set train.resume=checkpoints/last.pth
```

---

## 📏 Evaluation & inference

```bash
python -m src.evaluate --split test --beam 3          # -> checkpoints/eval_results.json
python -m src.predict photo1.jpg photo2.jpg --beam 3 --attention-out attention.png
python app.py --checkpoint checkpoints/best_model.pth  # Gradio demo
```

`src.evaluate` rebuilds exactly the training-time split from the config stored in the
checkpoint, decodes the split in batches and reports corpus BLEU-1…4, CIDEr-D, ROUGE-L and
latency per image (`--meteor` adds METEOR; needs `pycocoevalcap` and Java).

The notebook `notebooks/image_captioning.ipynb` covers EDA, architecture, training diagnostics,
evaluation and attention plots, and automatically uses `history.json` / the trained checkpoint.

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

20 tests run on CPU in ~11 s with a random-init backbone and synthetic data: no split leakage,
caption parsing, frozen-BatchNorm behaviour (frozen and fine-tuned), padding invariance,
KV-cache = full forward, batched = per-image beam search, CIDEr-D/ROUGE-L = `pycocoevalcap`,
initial loss ≈ log V, gradient accumulation = large batch, bit-exact resume, and an end-to-end
run from a config. GitHub Actions runs them on every push.

---

## 📊 Expected results & measured metrics

**Caption quality:** run `python -m src.train` then `python -m src.evaluate` — the numbers
are written to `checkpoints/eval_results.json`. No trained checkpoint is committed, so this README
does not quote BLEU / CIDEr numbers. *(An earlier version of this README listed a results table
that came from placeholder "mock" data in the notebook; it was removed because it was not measured.)*
For orientation, published Flickr8k results for soft-attention models are around
BLEU-1 ≈ 0.67 and BLEU-4 ≈ 0.19–0.21 (Xu et al., 2015); a frozen ResNet-50 + 3-layer
Transformer decoder trained with the default config should land in that range. Our split is a
seeded random 80/10/10 split, not the official Karpathy split, so numbers are close to but not
strictly comparable with papers.

**Engineering speed-ups (measured, old code vs. new code)** — CPU, 4 threads, batch 16,
d=256, 3 layers, vocab 3k, 20-token captions. Script: see `docs/IMPROVEMENTS.pdf`.

| Operation | Old | New | Speed-up |
|---|---|---|---|
| Training step, raw images | 833 ms | 673 ms | 1.2× |
| Training step, cached CNN features (new) | — | 77 ms | **10.8×** |
| Greedy decoding, 16 images | 1,744 ms | 722 ms | 2.4× |
| Beam search k=3, 1 image | 222 ms | 89 ms | 2.5× |
| Beam search k=5, 1 image | 343 ms | 85 ms | 4.0× |
| Tokenise 40k captions | crashes on NLTK ≥ 3.9 (`punkt_tab`) | 98 ms | — |

KV-cache (d=512, 3 layers, 16 images × 30 tokens, identical output): full-prefix greedy
**1,029 ms → 208 ms (4.9×)**; batched beam search k=3 for all 16 images: 373 ms.
Scripts: `docs/benchmark_old_vs_new.py`, `docs/benchmark_kv_cache.py`.

**Ablations (synthetic task, 3 seeds, `docs/ablation.py`)** — validation cross-entropy, lower is better:

| Variant | Initial loss (log V = 2.94) | Val loss | Token acc. |
|---|---|---|---|
| Untied output layer (**default**) | 3.06 | **0.256** | 0.878 |
| Tied weights, no output scaling | 8.03 | 0.358 | 0.847 |
| Tied weights + T5 output scaling | 2.99 | 0.554 | 0.794 |
| Untied, no spatial position embedding | 3.06 | 0.257 | 0.878 |
| Untied, label smoothing 0.1 | 3.07 | 0.330* | 0.877 |

\* label smoothing raises plain cross-entropy by design; token accuracy is unchanged. The
position embedding showed no measurable effect on this task — reported as is.

---

## 🔍 Key design decisions

| Decision | Rationale |
|---|---|
| Frozen ResNet-50 + feature cache | ImageNet features are strong; caching makes each epoch ~10× cheaper |
| BatchNorm kept in eval mode when frozen | `requires_grad=False` alone still lets BN running stats drift |
| 49 region vectors + 2-D position embedding | Spatial attention; attention alone is order-blind |
| Pre-LN decoder, GELU, scaled embedding init | Stable training; initial loss ≈ log V |
| Untied output layer | Beat weight tying in a controlled ablation (see Results) |
| KV-cache decoding | O(T) generation instead of O(T²), identical output |
| AdamW + warm-up + cosine, label smoothing 0.1 | Standard Transformer recipe; less over-confident captions |
| Select checkpoint on val BLEU-4 | Loss and caption quality are only loosely correlated |
| Split by image, vocab from train only | No leakage between splits |
| One config object, saved in every checkpoint | Any result can be reproduced from its checkpoint |
| `last.pth` with optimizer/scheduler/RNG state | Exact resume after interruption |

---

## 🔮 Future work

- [ ] Self-critical sequence training (optimise CIDEr directly)
- [ ] CLIP / ViT encoder; MS-COCO scale training
- [ ] Sub-word (BPE) vocabulary

---

## 📚 References

1. Xu et al., *Show, Attend and Tell* (2015) — [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)
2. He et al., *Deep Residual Learning for Image Recognition* (2016)
3. Vaswani et al., *Attention Is All You Need* (2017)
4. Xiong et al., *On Layer Normalization in the Transformer Architecture* (2020) — Pre-LN
5. Szegedy et al., *Rethinking the Inception Architecture* (2016) — label smoothing
6. Flickr8k — [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

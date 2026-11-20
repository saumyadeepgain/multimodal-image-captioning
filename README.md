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

## 📌 Overview

- **ResNet-50** (ImageNet weights) → 7×7×2048 spatial feature map → 49 region vectors
- **Learned spatial position embeddings** so the decoder knows *where* each region is
- **Pre-LN Transformer decoder** (causal self-attention + cross-attention over the 49 regions)
- **KV-cached decoding**: batched greedy and batched beam search, both returning attention maps
- **Corpus BLEU-1…4, CIDEr-D, ROUGE-L** against all 5 references (built in, verified against
  `pycocoevalcap`); optional METEOR
- Dataset: **Flickr8k** (8,091 images × 5 captions), split **by image** 80 / 10 / 10

> Reference: Xu et al., *Show, Attend and Tell* (2015) — [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

📄 **A step-by-step write-up of every process, what changed and why, is in
[`docs/IMPROVEMENTS.pdf`](docs/IMPROVEMENTS.pdf).**

---

## 🏗️ Architecture

```
Image 3×224×224
   │  ResNet-50 (frozen, BatchNorm kept in eval mode; optional layer4 fine-tune)
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
LayerNorm → Linear → next-token logits
```

---

## 🚀 Getting started

```bash
git clone https://github.com/saumyadeepgain/multimodal-image-captioning.git
cd multimodal-image-captioning
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download **Flickr8k** from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract to:

```
data/archive/Images/        ← 8,091 .jpg files
data/archive/captions.txt   ← "image,caption" (the original Flickr8k.token.txt also works)
```

### Train

```bash
python -m src.train --epochs 20
```

With the default frozen encoder, ResNet-50 features are computed **once per image** and cached
(fp16, ≈1.2 GB RAM for the train split), so every epoch only runs the decoder.
Useful flags: `--fine-tune-cnn` (train ResNet layer4, disables caching and enables augmentation),
`--embed-size`, `--layers`, `--heads`, `--batch-size`, `--lr`, `--label-smoothing`, `--patience`.

Outputs: `checkpoints/best_model.pth` (weights + config + vocabulary, selected on **val BLEU-4**)
and `checkpoints/history.json` (loss, perplexity, BLEU-4, LR, grad-norm, epoch time).

### Evaluate

```bash
python -m src.evaluate --split test --beam 3
```

Writes `checkpoints/eval_results.json` with corpus BLEU-1…4, CIDEr-D, ROUGE-L, latency per
image and sample predictions. Add `--meteor` for METEOR (needs `pycocoevalcap` and Java).

### Caption your own images

```bash
python -m src.predict photo1.jpg photo2.jpg --beam 3 --attention-out attention.png
```

### Demo app

```bash
python app.py --checkpoint checkpoints/best_model.pth
```

Gradio UI: upload an image, choose greedy or beam search, see the caption and per-word attention.

### Notebook

`notebooks/image_captioning.ipynb` — EDA, architecture, training diagnostics, evaluation and
attention plots. It automatically uses `history.json` / the trained checkpoint when present.

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The 15 tests run on CPU in a few seconds with a random-init backbone and a synthetic dataset, and are
run by GitHub Actions on every push.

---

## 📊 Results

**Caption quality:** run `python -m src.train` then `python -m src.evaluate` — the numbers
are written to `checkpoints/eval_results.json`. No trained checkpoint is committed, so this README
does not quote BLEU / CIDEr numbers. *(An earlier version of this README listed a results table
that came from placeholder "mock" data in the notebook; it was removed because it was not measured.)*
For orientation, published Flickr8k results for soft-attention models are around
BLEU-1 ≈ 0.67 and BLEU-4 ≈ 0.19–0.21 (Xu et al., 2015).

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

## 🗂️ Project structure

```
.
├── src/
│   ├── dataset.py        # caption parsing, leak-free split, Vocabulary, datasets, transforms
│   ├── model.py          # EncoderCNN, Pre-LN Transformer decoder, ImageCaptioningModel
│   ├── train.py          # CLI training: feature cache, AdamW, warm-up+cosine, AMP, early stop
│   ├── inference.py      # KV-cached batched greedy and beam search
│   ├── evaluate.py       # corpus evaluation CLI
│   ├── metrics.py        # dependency-free CIDEr-D and ROUGE-L
│   ├── predict.py        # caption image files from the command line
│   ├── checkpoint.py     # self-contained checkpoints (weights + config + vocab)
│   └── attention_viz.py  # attention overlays + attention entropy
├── tests/                # pytest suite (CPU, synthetic data)
├── notebooks/image_captioning.ipynb
├── docs/IMPROVEMENTS.pdf # step-by-step report of the processes and changes
├── app.py                # Gradio demo
├── requirements.txt / requirements-dev.txt
└── .github/workflows/tests.yml
```

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

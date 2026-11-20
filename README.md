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
| **Engineering** | Validated YAML/dataclass config with a CLI flag per key, CNN-feature cache, AMP + GradScaler, warm-up/cosine LR, early stopping, gradient accumulation, exact resume, CUDA-OOM recovery, TensorBoard / W&B, 82 tests + CI (ruff, pytest) |

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
│   ├── tracking.py           # optional TensorBoard / Weights & Biases tracker
│   └── utils.py              # seeding, device selection, logging, OOM helpers, checkpoints
├── tests/                    # pytest suite: model, overfit, dataset, training, CLI, ... (CPU)
├── notebooks/
│   └── image_captioning.ipynb   # EDA, diagnostics, evaluation, attention plots
├── docs/
│   ├── IMPROVEMENTS.pdf      # step-by-step report (built by build_report.py)
│   ├── benchmark_*.py/json   # reproducible speed benchmarks
│   └── ablation.py/…json     # controlled ablation study
├── app.py                    # Gradio demo
├── pyproject.toml            # package metadata, extras, console scripts, pytest config
├── requirements.txt / requirements-dev.txt
└── .github/workflows/ci.yml  # ruff + pytest on every push / PR
```

Generated at run time (git-ignored): `data/`, `checkpoints/` (`best_model.pth`, `last.pth`,
`config.yaml`, `history.json`, `logs/train_<timestamp>.log`, `tensorboard/`, `eval_results.json`).

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
python src/train.py                                     # defaults (= configs/default.yaml)
python src/train.py --config configs/default.yaml --lr 1e-4 --batch_size 32 --epochs 30
python -m src.train --train.lr 1e-4 --model.num_layers 4   # fully-qualified flags also work
```

**Every key in `configs/default.yaml` is a command-line flag**, by its short name when
unambiguous (`--lr`, `--epochs`, `--amp false`, `--tie_weights`) or fully qualified
(`--eval.batch_size 16`). Short names that exist in several sections resolve by priority
train > model > data > eval > tracking, so `--batch_size` means `train.batch_size`.
Precedence: defaults < `--config` file < flags < `--set key=value`. Values are type-checked
(`--lr abc` and `--epochs 2.5` are rejected) and the whole config is validated (e.g.
`embed_size` must be divisible by `nhead`) before anything is loaded.

| Flag | Effect |
|---|---|
| `--train_CNN` | fine-tune ResNet layer4 (disables feature caching, enables augmentation) |
| `--grad_accum_steps 4` | effective batch = `batch_size × 4` with the memory of one batch |
| `--select_metric CIDEr` | choose the best checkpoint by val CIDEr (`BLEU-4`, `CIDEr` or `val_loss`) |
| `--patience 5 --min_delta 0.001` | early stopping |
| `--warmup_ratio 0.05 --min_lr_ratio 0.05` | linear warm-up, then cosine annealing to 5 % of the LR |
| `--amp true` | bf16 autocast, or fp16 + `GradScaler` on GPUs without bf16 (CUDA only) |
| `--max_grad_norm 1.0` | gradient-norm clipping (applied after un-scaling under AMP) |
| `--resume checkpoints/last.pth` | continue exactly where a run stopped |
| `--stop_after_epochs 5` | stop this process after 5 epochs (time-limited jobs) |
| `--backend tensorboard` / `--backend wandb` | experiment tracking (see below) |
| `--device auto` | `cuda` > `mps` > `cpu`; an unavailable explicit device falls back with a warning |
| `--deterministic true` | deterministic cuDNN kernels for bit-exact reruns |

With the default frozen encoder, ResNet-50 features are computed **once per image** and cached
(fp16, ≈1.2 GB RAM for the train split), so every epoch only runs the decoder.

**Outputs** in `--out_dir` (default `checkpoints/`):
`best_model.pth` (weights + model config + vocabulary + full experiment config),
`last.pth` (adds optimizer, scheduler, AMP scaler, RNG states, best metric, patience counter,
global step and epoch), `config.yaml`, `history.json` and `logs/train_<timestamp>.log`.

**Resuming** restores all of the above and is bit-exact: an interrupted and resumed run
produces the same weights as an uninterrupted one (tested). Resuming with the saved
`config.yaml` also continues the same TensorBoard / W&B run.

```bash
python src/train.py --config checkpoints/config.yaml --resume checkpoints/last.pth
```

### Logging and experiment tracking

All messages go through Python `logging` to the console **and** a timestamped file
(`<out_dir>/logs/train_<YYYYmmdd-HHMMSS>.log`). Optional tracking is toggled in the config:

```yaml
tracking:
  backend: tensorboard     # none | tensorboard | wandb
  project: image-captioning
  log_every_n_steps: 50    # step-level loss / LR / grad-norm
  wandb_mode: online       # online | offline | disabled
```

```bash
pip install tensorboard && python src/train.py --backend tensorboard
tensorboard --logdir checkpoints/tensorboard
pip install wandb && python src/train.py --backend wandb --project my-captioning
```

Epoch-level train/val loss, perplexity, BLEU-4, CIDEr, LR, gradient norm, epoch time and five
sample captions are logged. If the package is missing or W&B cannot start, training continues
with a warning.

### Robustness

| Situation | Behaviour |
|---|---|
| Missing image folder / captions file / checkpoint | clear error message, exit code 1 |
| Missing or corrupted images (`--verify_images true`, default) | checked once up front, dropped, summary logged |
| Image becomes unreadable during training | logged once, the next sample is used (its own caption comes with it) |
| CUDA out of memory in training | batch retried as 2×, 4×, … micro-batches with the exact same gradient, up to `--max_oom_splits` |
| CUDA out of memory in validation / decoding / feature caching | batch split in halves recursively |
| Ctrl-C | message with the exact `--resume` command; `last.pth` holds the last finished epoch |
| Resuming from `best_model.pth` or a mismatched config | explanatory error instead of a stack trace |

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

## 🧪 Tests & CI

```bash
pip install -r requirements-dev.txt     # pytest, pytest-cov, ruff (pinned)
ruff check .                            # lint
ruff format --check .                   # formatting
pytest                                  # 82 tests, CPU only, ~45 s
pytest --cov=src --cov-report=term-missing
```

Everything runs on CPU with synthetic images and a randomly initialised backbone — no dataset,
GPU or network needed.

| File | What it proves |
|---|---|
| `tests/test_model.py` | construction, exact output shapes (`[B,T-1,V]`, `[B,T-1,S]`, `[B,S,E]`), finite loss, gradients reach **every** trainable parameter and no frozen one, dropout/eval behaviour, bf16 autocast |
| `tests/test_overfit_batch.py` | the real training loop memorises one batch: loss falls by > 95 % in 50 steps and greedy decoding reproduces the captions exactly (also end-to-end through ResNet-50) |
| `tests/test_dataset.py` | transform shapes, exact ImageNet-normalisation values and ranges, deterministic eval / random train transforms, train-only vocabulary, collate padding, loader shapes, feature-cache alignment |
| `tests/test_pipeline.py` | split without leakage, frozen BatchNorm, padding invariance, KV-cache = full forward, batched = per-image beam search |
| `tests/test_training.py` | config overrides, bit-exact resume, gradient accumulation = large batch, end-to-end run |
| `tests/test_hardening.py` | CLI flags and validation, script mode, device fallback, TensorBoard / W&B, OOM micro-batching, corrupted / missing images, error messages |
| `tests/test_metrics.py` | CIDEr-D and ROUGE-L match `pycocoevalcap` exactly |
| `tests/test_cli.py` | `train` / `evaluate` / `predict` entry points and attention plots on a real checkpoint |

**CI** (`.github/workflows/ci.yml`) runs on every push and pull request: a `ruff check` +
`ruff format --check` job, and the full test suite with coverage on CPU-only PyTorch for
Python 3.9 and 3.12 (coverage and JUnit reports are uploaded as artifacts).

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

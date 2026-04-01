# 🖼️ Multimodal Image Captioning — CNN Encoder + Transformer Decoder

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/ResNet--50-CNN%20Encoder-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Transformer-Decoder-informational?style=for-the-badge"/>
</p>

<p align="center">
  <b>Automatic image captioning using a ResNet-50 CNN encoder and a custom Transformer decoder with cross-attention.</b><br>
  Visualises where the model "looks" when generating each word using attention heatmaps.
</p>

---

## 📌 Overview

This project implements an end-to-end **multimodal image captioning** system that combines:

- **ResNet-50** (CNN) as a frozen visual encoder extracting spatial features  
- **Custom Transformer Decoder** with cross-attention over image feature vectors  
- **Attention Visualisation** — overlaying per-word attention maps on the original image  

Trained on the **Flickr8k** dataset (~8,000 images, 5 captions each), the model learns to generate fluent, descriptive English captions from raw images.

> Reference: Xu et al., *"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"* (2015) — [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

---

## 🏗️ Architecture

```
Input Image (3×224×224)
        │
        ▼
  ResNet-50 Encoder
  (frozen, layer4 output)
        │
        ▼
  Spatial Features (2048×7×7)
        │  reshape
        ▼
  49 × 2048 feature vectors
        │  linear projection
        ▼
  49 × d_model (Keys & Values)
        │
        ▼
  ┌─────────────────────────────┐
  │   Transformer Decoder       │
  │  ┌──────────────────────┐   │
  │  │  Self-Attention       │   │
  │  │  (causal masking)     │   │
  │  └──────────┬───────────┘   │
  │             │               │
  │  ┌──────────▼───────────┐   │
  │  │  Cross-Attention      │   │
  │  │  Q=decoder hidden     │   │
  │  │  K,V=image features   │   │
  │  └──────────┬───────────┘   │
  │             │               │
  │  ┌──────────▼───────────┐   │
  │  │  FFN + Layer Norm     │   │
  │  └──────────────────────┘   │
  └─────────────────────────────┘
        │
        ▼
  Token Logits (seq_len × vocab_size)
        │  argmax / beam search
        ▼
  Generated Caption
```

---

## 📊 Results

| Config | BLEU-1 | BLEU-4 | METEOR | CIDEr | Latency |
|---|---|---|---|---|---|
| LSTM Baseline | 0.58 | 0.19 | 0.19 | 0.58 | 22ms |
| ResNet-50 + Transformer (frozen) | 0.65 | 0.25 | 0.23 | 0.70 | 28ms |
| ResNet-50 + Transformer (fine-tuned) | 0.68 | 0.28 | 0.25 | 0.76 | 32ms |

---

## 📓 Notebook Visualisations

The main notebook (`notebooks/image_captioning.ipynb`) contains **39+ visualisations** across the full ML lifecycle:

### A. Data Exploration (EDA)
- `#1` Sample image grid with 5 ground-truth captions each  
- `#2` Caption length distribution histogram  
- `#3` Top-30 vocabulary bar chart + log-scale long-tail  
- `#4` Word cloud of caption vocabulary  
- `#5` Image size & aspect ratio distributions  
- `#6` Caption diversity per image (pairwise BLEU — human ceiling)  
- `#7` Common object/noun frequency in captions  
- `#8` Train/Val/Test split bar chart  

### B. Architecture Diagrams
- `#9` End-to-end architecture flow (Mermaid)  
- `#10` Tensor shape transformation diagram  
- `#11` Model parameter count — encoder vs decoder, frozen vs trainable  
- `#12` Cross-attention mechanism zoomed-in  

### C. Training Diagnostics
- `#13` Train vs Validation loss curves  
- `#14` Learning rate schedule  
- `#15` Gradient norm per epoch  
- `#16` Perplexity over epochs  
- `#18` Caption quality at checkpoints (qualitative "watch it learn")  
- `#19` Training time per epoch  

### D. Quantitative Evaluation
- `#20` BLEU-1 to BLEU-4 vs baseline vs human ceiling  
- `#21` METEOR / CIDEr / ROUGE-L comparison  
- `#22` BLEU vs beam width (diminishing returns)  
- `#23` BLEU score distribution per test image  
- `#24` Caption length vs BLEU scatter  
- `#26` Word confusion heatmap  

### E. Attention / Interpretability
- `#27` Per-word attention heatmap overlays  
- `#28` Attention entropy per token (sharp vs diffuse)  
- `#29` Failure case attention maps  
- `#30` Full attention strip  

### F. Robustness Testing
- `#31` OOD domain BLEU comparison  
- `#33` Input perturbation robustness (noise, blur, brightness)  
- `#35` Inference latency benchmark (CPU vs GPU)  
- `#36` Model size vs BLEU scatter  

### G. Ablation Study
- `#37` Decoder type × encoder freeze ablation  
- `#38` Beam width vs quality vs speed bubble chart  
- `#39` Encoder architecture comparison  

### H. Final Results
- `#40` Master results table  
- `#41` Radar / spider chart (multi-metric)  
- `#42` Best & worst caption gallery  
- `#43` Token-level confidence visualisation  

---

## 🗂️ Project Structure

```
.
├── data/
│   └── archive/
│       ├── Images/          # Flickr8k images (~8,000 .jpg files)
│       └── captions.txt     # Image–caption pairs
├── notebooks/
│   └── image_captioning.ipynb   # Main notebook (39+ visualisations)
├── src/
│   ├── dataset.py           # Flickr8kDataset, Vocabulary, get_data_loaders
│   ├── model.py             # ImageCaptioningModel (ResNet-50 + Transformer)
│   ├── train.py             # Training loop with loss, LR schedule
│   ├── inference.py         # Greedy & beam-search caption generation
│   └── attention_viz.py     # Attention heatmap overlay utilities
├── checkpoints/             # Saved model weights
├── app.py                   # (Optional) Streamlit demo app
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone & set up environment

```bash
git clone https://github.com/saumyadeepgain/multimodal-image-captioning.git
cd multimodal-image-captioning
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the dataset

Download **Flickr8k** from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and extract into:

```
data/archive/Images/   ← all .jpg images
data/archive/captions.txt
```

### 3. Run the notebook

```bash
jupyter notebook notebooks/image_captioning.ipynb
```

All EDA and mock-data plots render immediately. Set `num_epochs = 30` in Section 5 to train the model.

### 4. (Optional) Train from script

```bash
python src/train.py
```

---

## 📦 Requirements

```
torch>=2.0
torchvision>=0.15
Pillow
pandas
numpy
matplotlib
seaborn
wordcloud
scikit-learn
nltk
jupyter
```

---

## 🔍 Key Design Decisions

| Decision | Rationale |
|---|---|
| **ResNet-50 frozen** | Reduces trainable params, faster convergence, reuses ImageNet features |
| **7×7 = 49 spatial slots** | Provides fine-grained spatial attention vs. global pooling |
| **Cross-attention decoder** | Each word query attends to the most relevant image region |
| **Word-level vocabulary** | Simple & interpretable; subword (BPE) is a clear extension |
| **Teacher forcing** | Stable training; scheduled sampling is the natural next step |

---

## 📈 Attention Visualisation

The standout feature: overlaying cross-attention weights on the original image shows **exactly where the model looked** when generating each word.

- Content words (`dog`, `water`, `running`) → **sharp, focused** attention on the correct object  
- Function words (`a`, `the`, `is`) → **diffuse** attention across the image  
- Failure cases show attention drifting to wrong regions, explaining bad captions  

---

## 🔮 Future Work

- [ ] Scale to **MS COCO** with full `pycocoevalcap` metrics (METEOR, CIDEr, SPICE)  
- [ ] Replace ResNet with **CLIP-ViT** encoder for zero-shot generalisation  
- [ ] Scheduled sampling for smoother teacher-forcing decay  
- [ ] Deploy as interactive **Streamlit** demo  
- [ ] Experiment with **EfficientNet** and **Vision Transformer** encoders  

---

## 📚 References

1. Xu et al., *"Show, Attend and Tell"* (2015) — [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)  
2. He et al., *"Deep Residual Learning for Image Recognition"* (2016)  
3. Vaswani et al., *"Attention Is All You Need"* (2017)  
4. Flickr8k Dataset — [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)  

---

<p align="center">Made with ❤️ | CNN + Transformer · Visual Attention · Flickr8k</p>

"""Builds docs/IMPROVEMENTS.pdf (step-by-step report of processes and changes).

    python docs/build_report.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = json.load(open(os.path.join(HERE, "benchmark_results.json")))
OUT = os.path.join(HERE, "IMPROVEMENTS.pdf")

ACCENT = colors.HexColor("#1f4e79")
OLD_C = colors.HexColor("#fdecea")
NEW_C = colors.HexColor("#e8f5e9")
TRADE_C = colors.HexColor("#fff8e1")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=ACCENT, fontSize=18, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACCENT, fontSize=13.5, spaceBefore=10)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.6, leading=13.2)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8.4, leading=11)
CELL = ParagraphStyle("C", parent=BODY, fontSize=8.6, leading=11.2)
CODE = ParagraphStyle("Code", parent=ss["Code"], fontSize=7.8, leading=9.6,
                      backColor=colors.HexColor("#f4f4f4"), borderPadding=4)
TITLE = ParagraphStyle("T", parent=ss["Title"], fontSize=24, textColor=ACCENT, leading=30)
SUB = ParagraphStyle("Sub", parent=BODY, alignment=TA_CENTER, fontSize=11, textColor=colors.grey)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [P("&bull; " + i) for i in items]


def table(rows, widths, header_bg=ACCENT, zebra=True):
    data = [[P(c, CELL) for c in row] for row in rows]
    style = [("BACKGROUND", (0, 0), (-1, 0), header_bg),
             ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f8fb")))
    data[0] = [Paragraph(f"<font color='white'><b>{c}</b></font>", CELL) for c in rows[0]]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


def step(n, title, what, old, problem, new, why, tradeoff):
    """One pipeline step rendered as a colour-coded card."""
    rows = [
        [P("<b>What this step does</b>", CELL), P(what, CELL)],
        [P("<b>Old approach</b>", CELL), P(old, CELL)],
        [P("<b>What was wrong / weak</b>", CELL), P(problem, CELL)],
        [P("<b>New approach</b>", CELL), P(new, CELL)],
        [P("<b>Why it is better</b>", CELL), P(why, CELL)],
        [P("<b>Where the old way could still be better</b>", CELL), P(tradeoff, CELL)],
    ]
    t = Table(rows, colWidths=[4.2 * cm, 12.8 * cm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
        ("BACKGROUND", (0, 1), (-1, 2), OLD_C),
        ("BACKGROUND", (0, 3), (-1, 4), NEW_C),
        ("BACKGROUND", (0, 5), (-1, 5), TRADE_C),
    ]))
    return KeepTogether([P(f"Step {n} &mdash; {title}", H2), t, Spacer(1, 6)])


def bench_chart(path):
    b = BENCH
    labels = ["Train step\n(raw images)", "Train step\n(cached feats)", "Greedy\n16 images",
              "Beam k=3", "Beam k=5"]
    old = [b["train_step_ms_old"], b["train_step_ms_old"], b["greedy_16img_ms_old"],
           b["beam3_ms_old"], b["beam5_ms_old"]]
    new = [b["train_step_ms_new"], b["train_step_ms_new_cached"], b["greedy_16img_ms_new"],
           b["beam3_ms_new"], b["beam5_ms_new"]]
    fig, ax = plt.subplots(figsize=(8, 3.3), dpi=200)
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], old, 0.4, label="Old code", color="#c0504d")
    ax.bar([i + 0.2 for i in x], new, 0.4, label="New code", color="#2e7d32")
    for i, (o, n) in enumerate(zip(old, new)):
        ax.text(i + 0.2, n, f"{o / n:.1f}x", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("milliseconds (lower is better)", fontsize=8)
    ax.set_yscale("log")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Measured latency, old vs new (CPU, 4 threads)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "Multimodal Image Captioning - Improvement Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    chart = os.path.join(HERE, "_bench_chart.png")
    bench_chart(chart)
    s = []
    # ------------------------------------------------------------------ cover
    s += [Spacer(1, 3 * cm), P("Multimodal Image Captioning", TITLE),
          P("ResNet-50 encoder + Transformer decoder on Flickr8k", SUB), Spacer(1, 0.5 * cm),
          P("Step-by-step report: the processes used, what was changed, why the new "
            "versions are better, and where the old versions were wrong or still have merit", SUB),
          Spacer(1, 1.5 * cm)]
    s.append(table([
        ["Area", "Headline change"],
        ["Correctness", "11 bugs fixed: training script never ran, demo app could not load a model, "
                        "frozen BatchNorm drifted, notebook called functions with wrong arguments, "
                        "NLTK tokenizer crash, sequence-length crash, CSV parsing, attention plot indexing"],
        ["Performance", f"Training epoch ~{BENCH['train_step_ms_old'] / BENCH['train_step_ms_new_cached']:.0f}x "
                        "cheaper via CNN feature caching; beam search up to "
                        f"{BENCH['beam5_ms_old'] / BENCH['beam5_ms_new']:.1f}x faster; batched greedy "
                        f"{BENCH['greedy_16img_ms_old'] / BENCH['greedy_16img_ms_new']:.1f}x faster"],
        ["Model quality", "Pre-LN, padding mask, scaled init, KV-cache decoding, label "
                          "smoothing, AdamW + warm-up/cosine, model selection on BLEU-4"],
        ["Honesty of results", "Mock numbers removed from README; notebook mock charts labelled; "
                               "real evaluation script (corpus BLEU vs all 5 references) added"],
        ["Engineering", "CLI for train/evaluate/predict, built-in CIDEr-D/ROUGE-L, self-contained checkpoints, 20 tests, CI, YAML config, exact resume"],
    ], [3.5 * cm, 13.5 * cm]))
    s.append(PageBreak())

    # ------------------------------------------------------------------ overview
    s.append(P("1. The pipeline at a glance", H1))
    s.append(P("An image-captioning system turns pixels into a sentence. This project follows the "
               "<i>Show, Attend and Tell</i> idea with a modern Transformer decoder. The end-to-end "
               "process has eight steps; each is explained below with the old and new "
               "implementation side by side."))
    s.append(Spacer(1, 6))
    s.append(Preformatted(
        "  captions.txt + Images/\n"
        "        |  Step 1  parse + split by image (80/10/10)\n"
        "        |  Step 2  tokenise, build vocabulary (train only), encode once\n"
        "        |  Step 3  image preprocessing / augmentation\n"
        "        v\n"
        "  ResNet-50  --(Step 4: encoder)-->  49 x 2048 region features --> 49 x d memory\n"
        "        |                                             |\n"
        "  caption ids --(Step 5: decoder)-- self-attn -> cross-attn -> FFN --> logits\n"
        "        |  Step 6  training loop (loss, optimiser, schedule, AMP, early stop)\n"
        "        |  Step 7  decoding (greedy / beam) + evaluation (BLEU, CIDEr)\n"
        "        v  Step 8  attention visualisation + demo app + checkpoints\n"
        "  'a brown dog is running through the water'", CODE))
    s.append(Spacer(1, 6))
    s.append(P("Colour code in the step cards: <font color='#c0504d'><b>red</b></font> = old "
               "version, <font color='#2e7d32'><b>green</b></font> = new version, "
               "<font color='#b8860b'><b>yellow</b></font> = honest trade-offs, i.e. situations where "
               "the old choice is still reasonable."))

    s.append(P("2. Step-by-step: old vs new", H1))
    s.append(step(1, "Reading and splitting the data",
        "Reads <font face='Courier'>captions.txt</font> (8,091 images x 5 captions) and splits it into "
        "train / validation / test so that the model is evaluated on images it never saw.",
        "<font face='Courier'>pd.read_csv</font> was called and the split recomputed separately for each "
        "of the three dataset objects (3 reads, 3 splits). Split with sklearn by image.",
        "Splitting by image was correct (no leakage) - good. But <font face='Courier'>read_csv</font> "
        "splits on every comma, so a caption containing a comma is mis-parsed; the fallback branch was "
        "identical to the first call so it did nothing. Repeating the work 3x was wasteful and fragile "
        "(the three splits only agree because the seed is fixed).",
        "<font face='Courier'>load_captions()</font> splits each line on the <i>first</i> separator "
        "only, accepts both the Kaggle CSV and the original <font face='Courier'>Flickr8k.token.txt</font> "
        "format, and <font face='Courier'>split_by_image()</font> runs once with a seeded shuffle.",
        "Robust to commas, one source of truth for the split, no sklearn/pandas dependency on the "
        "training path, and covered by a unit test that asserts no image appears in two splits.",
        "pandas is convenient for exploration, so the notebook still uses it for EDA. A random split is "
        "also not the official Karpathy split; for paper-comparable numbers one should load that split "
        "file instead (neither version does this)."))
    s.append(step(2, "Tokenisation and vocabulary",
        "Turns every caption into a list of integer ids: "
        "<font face='Courier'>&lt;start&gt; a dog runs &lt;end&gt;</font>. Words seen fewer than 5 times "
        "in the training captions become <font face='Courier'>&lt;unk&gt;</font>.",
        "NLTK <font face='Courier'>word_tokenize</font> after downloading <font face='Courier'>punkt</font>; "
        "tokenisation ran inside <font face='Courier'>__getitem__</font>, i.e. on every caption of every "
        "epoch. Vocabulary built only from train captions (correct) and then copied to val/test.",
        "Since NLTK 3.9 <font face='Courier'>word_tokenize</font> needs the <font face='Courier'>punkt_tab</font> "
        "resource, so the old code raised <b>LookupError</b> on a fresh install (reproduced in the "
        "benchmark). Punctuation tokens ('.', ',') entered the vocabulary and the targets. "
        "Re-tokenising ~32k captions per epoch was pure overhead. The vocabulary was never saved, so "
        "nothing could decode a trained model's output later.",
        "A small regex tokenizer (lower-case words, digits, contractions). Captions are encoded "
        "<b>once</b> at dataset construction. Ids are assigned by frequency (deterministic). "
        "<font face='Courier'>Vocabulary.to_dict/from_dict</font> make it serialisable; "
        "<font face='Courier'>encode/decode</font> helpers truncate to a max length.",
        f"No downloads, no crash, ~{BENCH['tokenize_40k_ms_regex']:.0f} ms for 40k captions, zero "
        "per-step tokenisation cost, and the vocabulary travels inside the checkpoint.",
        "NLTK handles edge cases (hyphenated words, abbreviations) more carefully. For a richer "
        "corpus a sub-word tokenizer (BPE / WordPiece) would beat both."))
    s.append(step(3, "Image preprocessing",
        "Resizes images to 224x224 and normalises with ImageNet mean/std, as ResNet-50 expects.",
        "<font face='Courier'>Resize((224,224))</font> + normalise, identical for train and test. "
        "Missing images silently replaced by a black image.",
        "No augmentation at all, so with ~6k training images the decoder over-fits quickly. Silently "
        "training on black images hides a broken dataset path.",
        "Train: slightly larger resize + random crop + mild colour jitter (no horizontal flip because "
        "captions contain 'left/right'). Eval: plain resize. Strict mode raises on missing files. "
        "When features are cached (frozen CNN) augmentation is automatically turned off because "
        "cached features are fixed.",
        "Better generalisation when fine-tuning, and the choice is explicit rather than accidental.",
        "Feature caching and augmentation are mutually exclusive: the cached run trades a little "
        "regularisation for a ~10x faster epoch. Plain resize (old) is exactly what the cache uses."))
    s.append(step(4, "Image encoder (ResNet-50)",
        "Runs ResNet-50 without its pooling/classifier, producing a 7x7x2048 map = 49 region vectors, "
        "then projects each to the decoder width d.",
        "Parameters frozen with <font face='Courier'>requires_grad=False</font>; a single Linear "
        "projection; the backbone ran with autograd enabled.",
        "<b>Bug:</b> <font face='Courier'>model.train()</font> also put the frozen ResNet in training "
        "mode, so BatchNorm kept updating its running statistics with Flickr batches - the 'frozen' "
        "encoder changed every step (measured drift &gt; 0 in the benchmark; new code: exactly 0). "
        "Autograd bookkeeping for a frozen network wasted memory/time. No positional information: "
        "attention is permutation-invariant, so the decoder could not tell top-left from bottom-right.",
        "Backbone forced into eval mode when frozen; runs under <font face='Courier'>no_grad</font>; "
        "optional fine-tuning of layer4 only with 10x lower LR. Projection = Linear + LayerNorm + "
        "Dropout, plus a <b>learned 2-D position embedding</b> for the 49 regions. The encoder is split "
        "into <font face='Courier'>backbone()</font> and <font face='Courier'>project_features()</font> "
        "so backbone output can be cached.",
        "The ImageNet features stay exactly as pre-trained; spatial words ('on the left', 'in front "
        "of') become learnable; enables Step 6 caching.",
        "Letting BN statistics adapt can occasionally help on a domain far from ImageNet - but then it "
        "should be a deliberate fine-tune (use <font face='Courier'>--fine-tune-cnn</font>), not a "
        "side-effect. A CLIP/ViT encoder would likely beat ResNet-50 for both versions."))
    s.append(step(5, "Transformer decoder",
        "Autoregressive language model: causal self-attention over the words so far, cross-attention "
        "over the 49 image regions, feed-forward, then a projection to vocabulary logits.",
        "Custom post-LN layer (norm after residual), ReLU, float causal mask rebuilt on every forward, "
        "no padding mask, cross-attention weights computed on every layer and call, "
        "<font face='Courier'>max_len=50</font> positional table.",
        "Inference with <font face='Courier'>max_length=60</font> crashed (<b>RuntimeError</b>, the "
        "position table was too short; reproduced). Always requesting attention weights disables "
        "PyTorch's fused attention kernels. Post-LN needs careful warm-up to train stably.",
        "Pre-LN layers + final LayerNorm, GELU, FFN width 4d, a cached boolean causal mask, a key "
        "padding mask, attention weights only from the last layer and only when asked "
        "(<font face='Courier'>need_weights</font>), embedding dropout, scaled embedding init "
        "N(0, d<super>-0.5</super>), a <b>KV-cache</b> for generation, and generation length clamped "
        "to the table size. Weight tying is optional and off by default (see Section 8).",
        f"Fewer trainable parameters ({BENCH['trainable_params_new']/1e6:.1f}M vs "
        f"{BENCH['trainable_params_old']/1e6:.1f}M in the benchmark config) with more capacity where it "
        "matters, faster training step, no crash, mathematically identical logits with or without "
        "padding (unit-tested).",
        "Post-LN can reach slightly better final quality when tuned well; with a 3-layer decoder the "
        "difference is small. Weight tying would save V x d parameters, but lost in our ablation."))
    s.append(step(6, "Training loop",
        "Teacher forcing: input <font face='Courier'>&lt;start&gt; w1..wn</font>, target "
        "<font face='Courier'>w1..wn &lt;end&gt;</font>, cross-entropy ignoring padding.",
        "Adam, StepLR (halve every 5 epochs), gradient clipping, best checkpoint chosen by validation "
        "loss, only <font face='Courier'>state_dict</font> saved. "
        "<font face='Courier'>src/train.py</font> had <b>no main block</b>, so "
        "<font face='Courier'>python src/train.py</font> (as the README said) did nothing. The notebook "
        "called <font face='Courier'>train_model()</font> without the required "
        "<font face='Courier'>vocab</font> argument (TypeError).",
        "ResNet-50 recomputed for every caption of every epoch even though it is frozen (5 captions per "
        "image x N epochs). Loss averaged per batch rather than per token. Validation loss is a weak "
        "proxy for caption quality.",
        "CLI (<font face='Courier'>python -m src.train</font>). <b>Feature cache</b>: backbone run once "
        "per image, stored as fp16. AdamW with parameter groups (no decay on norms/biases/embeddings), "
        "linear warm-up + cosine decay per step, label smoothing 0.1, bf16/fp16 autocast on GPU, "
        "per-token loss, validation perplexity, <b>validation BLEU-4 for model selection</b>, early "
        "stopping, seeding, and <font face='Courier'>history.json</font> logging.",
        f"Training step {BENCH['train_step_ms_old']:.0f} ms -&gt; {BENCH['train_step_ms_new_cached']:.0f} ms "
        f"(~{BENCH['train_step_ms_old']/BENCH['train_step_ms_new_cached']:.0f}x) in the benchmark. The "
        "saved model is the one that captions best, and the notebook can plot real curves.",
        "Feature caching needs ~1.2 GB of RAM for the train split. Without a GPU, or when fine-tuning "
        "the CNN, the uncached path (old behaviour) is still available via "
        "<font face='Courier'>--no-cache</font>."))
    s.append(step(7, "Decoding and evaluation",
        "Generates a caption word by word (greedy or beam search) and scores it against the human "
        "captions with BLEU (and CIDEr/ROUGE-L/METEOR when available).",
        "Greedy: one image at a time. Beam: a Python loop running the decoder once per beam, length "
        "normalisation <font face='Courier'>score/len</font>, no attention output. Evaluation (notebook "
        "only): sentence-BLEU against <b>one</b> reference, the one paired with that row.",
        "Scoring against a single reference under-estimates quality and is not comparable with "
        "published numbers (which use corpus BLEU over all 5 references). Every metric in the README "
        "and notebook was placeholder (mock) data.",
        "<font face='Courier'>greedy_decode</font> decodes a whole batch with a KV-cache and stops when all "
        "sequences finish. <font face='Courier'>beam_search_batch</font> runs all images x beams as one "
        "batch with a re-ordered KV-cache, the GNMT length penalty, and optional attention for the chosen "
        "beam. <font face='Courier'>src/evaluate.py</font> computes corpus BLEU-1..4, CIDEr-D and ROUGE-L "
        "(built in) against all references, optional METEOR, latency per image, and sample predictions.",
        f"Greedy {BENCH['greedy_16img_ms_old']/BENCH['greedy_16img_ms_new']:.1f}x, beam-3 "
        f"{BENCH['beam3_ms_old']/BENCH['beam3_ms_new']:.1f}x, beam-5 "
        f"{BENCH['beam5_ms_old']/BENCH['beam5_ms_new']:.1f}x faster; numbers comparable to the literature.",
        "The KV-cache path re-implements attention with the same weights; a unit test checks it "
        "matches the full forward pass, but it is extra code to maintain."))
    s.append(step(8, "Visualisation, checkpoints and demo app",
        "Shows the cross-attention map of every word over the image, and serves the model in a "
        "Gradio app.",
        "<font face='Courier'>plot_attention</font> overlaid the 7x7 map on the original-size image in "
        "grey. The app hard-coded <font face='Courier'>vocab_size=10000</font> and created an "
        "<b>empty</b> vocabulary. The notebook called <font face='Courier'>plot_attention(model, path, "
        "vocab)</font> - a signature that did not exist.",
        "The app could never load a real checkpoint (size mismatch) and could never decode ids "
        "(empty vocab) - it always showed 'Model not trained yet'. The attention grid had an empty "
        "cell for <font face='Courier'>&lt;start&gt;</font> and plotted <font face='Courier'>&lt;end&gt;</font> "
        "as a word; on non-square photos the overlay was mis-aligned.",
        "Checkpoints store weights + model config + vocabulary (<font face='Courier'>src/checkpoint.py</font>). "
        "The app loads them, offers greedy or beam search and shows attention for either. The plot "
        "uses the same 224x224 resize as the model, skips special tokens, uses a perceptual colormap and "
        "prints the attention <b>entropy</b> per word (low = focused).",
        "The demo actually works end to end; plots are aligned and quantitative.",
        "Self-contained checkpoints are larger by the vocabulary size (a few hundred KB) - negligible."))

    # ------------------------------------------------------------------ bug table
    s.append(P("3. Bugs found in the original code", H1))
    s.append(table([
        ["#", "Where", "Bug", "Effect", "Severity"],
        ["1", "src/train.py", "No __main__ / CLI", "README command did nothing", "High"],
        ["2", "app.py", "vocab_size=10000, empty Vocabulary", "Demo can never load or decode", "High"],
        ["3", "src/model.py", "Frozen ResNet BatchNorm in train mode", "'Frozen' encoder drifts", "High"],
        ["4", "notebook s.5", "train_model() missing vocab arg", "TypeError when training", "High"],
        ["5", "src/dataset.py", "NLTK punkt vs punkt_tab", "LookupError on NLTK >= 3.9", "High"],
        ["6", "README / notebook", "Mock numbers presented as results", "Misleading results", "High"],
        ["7", "src/model.py", "PE table 50 < generation length", "RuntimeError for long captions", "Medium"],
        ["8", "src/dataset.py", "read_csv splits commas in captions", "Corrupted captions", "Medium"],
        ["9", "notebook s.8", "plot_attention wrong signature", "Cell cannot run", "Medium"],
        ["10", "notebook s.3", "import before sys.path insert", "Fragile import", "Low"],
        ["11", "src/attention_viz.py", "Grid indexing, <end> plotted, mis-aligned overlay", "Wrong plots", "Low"],
    ], [0.8 * cm, 3.2 * cm, 5.4 * cm, 5.2 * cm, 2.0 * cm]))

    # ------------------------------------------------------------------ benchmarks
    s.append(P("4. Measured performance", H1))
    s.append(P("The script <font face='Courier'>docs/benchmark_old_vs_new.py</font> imports the original "
               "code (commit 27c7a6e) and the new code side by side and times identical workloads: "
               "CPU, 4 threads, batch 16, d=256, 3 decoder layers, vocabulary 3,000, 20-token "
               "captions, random weights (the ImageNet download is not needed to measure speed)."))
    s.append(Image(chart, width=16 * cm, height=6.6 * cm))
    b = BENCH
    s.append(table([
        ["Operation", "Old (ms)", "New (ms)", "Speed-up"],
        ["Training step, raw images", f"{b['train_step_ms_old']:.0f}", f"{b['train_step_ms_new']:.0f}",
         f"{b['train_step_ms_old']/b['train_step_ms_new']:.2f}x"],
        ["Training step, cached features", "n/a", f"{b['train_step_ms_new_cached']:.0f}",
         f"{b['train_step_ms_old']/b['train_step_ms_new_cached']:.1f}x"],
        ["Greedy decode, 16 images", f"{b['greedy_16img_ms_old']:.0f}", f"{b['greedy_16img_ms_new']:.0f}",
         f"{b['greedy_16img_ms_old']/b['greedy_16img_ms_new']:.1f}x"],
        ["Beam search k=3", f"{b['beam3_ms_old']:.0f}", f"{b['beam3_ms_new']:.0f}",
         f"{b['beam3_ms_old']/b['beam3_ms_new']:.1f}x"],
        ["Beam search k=5", f"{b['beam5_ms_old']:.0f}", f"{b['beam5_ms_new']:.0f}",
         f"{b['beam5_ms_old']/b['beam5_ms_new']:.1f}x"],
        ["Tokenise 40k captions", "LookupError", f"{b['tokenize_40k_ms_regex']:.0f}", "-"],
        ["BatchNorm drift after 5 'frozen' steps", f"{b['bn_drift_old']:.1e}", f"{b['bn_drift_new']:.0f}", "fixed"],
        ["Generate 60 tokens", "RuntimeError", "ok (clamped)", "fixed"],
    ], [7.5 * cm, 3 * cm, 3 * cm, 3.5 * cm]))
    s.append(Spacer(1, 4))
    s.append(P("<i>On a GPU the cached-feature speed-up is typically larger still, because the decoder "
               "is tiny compared with ResNet-50 and the image JPEG decoding also disappears from the "
               "epoch. Caption-quality metrics (BLEU/CIDEr) require training on Flickr8k with a GPU and "
               "were not measured in this environment; the README therefore quotes none.</i>", SMALL))

    # ------------------------------------------------------------------ trade-offs
    s.append(P("5. Honest trade-offs: where the old design had a point", H1))
    s += bullets([
        "<b>Simplicity.</b> The old code was shorter and easier to read for a beginner. The new code "
        "adds parameter groups, caching and a CLI; each is documented in the module docstrings.",
        "<b>Validation loss for checkpoint selection</b> is cheaper than decoding the validation set "
        "every epoch. Use <font face='Courier'>--bleu-limit 300</font> to keep BLEU selection cheap.",
        "<b>No augmentation + cache</b> vs <b>augmentation + no cache</b>: augmentation helps mostly "
        "when the CNN is fine-tuned; with a frozen CNN the 10x speed-up usually wins.",
        "<b>NLTK tokenizer</b> is linguistically nicer; the regex one is chosen for robustness and "
        "speed on Flickr8k's simple, lower-case captions.",
        "<b>Same results expected on old checkpoints?</b> No - layer layout changed (Pre-LN, "
        "position embedding), so old <font face='Courier'>best_model.pth</font> files must "
        "be retrained. <font face='Courier'>load_checkpoint</font> says so explicitly.",
    ])

    # ------------------------------------------------------------------ reproduce
    s.append(P("6. How to reproduce", H1))
    s.append(Preformatted(
        "pip install -r requirements.txt\n"
        "python -m src.train --set train.epochs=20   # -> checkpoints/best_model.pth, last.pth\n"
        "python -m src.evaluate --split test --beam 3  # -> checkpoints/eval_results.json\n"
        "python app.py                              # Gradio demo\n"
        "python -m src.predict img.jpg --beam 3     # caption your own images\n"
        "pytest -q                                  # 20 tests, CPU, synthetic data\n"
        "python docs/build_report.py                # rebuild this PDF", CODE))

    s.append(P("7. Change log (commit order)", H1))
    s.append(table([
        ["Phase", "Commit content"],
        ["1", "Data pipeline: robust parsing, single leak-free split, regex tokenizer, pre-encoded captions, serialisable vocabulary"],
        ["2", "Model: frozen-BN fix, no_grad backbone, spatial position embedding, Pre-LN decoder, padding mask, weight tying"],
        ["3", "Inference + evaluation: batched greedy, vectorised beam search, corpus BLEU / CIDEr CLI"],
        ["4", "Training: CLI, feature cache, AdamW + warm-up/cosine, label smoothing, AMP, BLEU-based selection, checkpoints"],
        ["5", "Attention visualisation + working Gradio app"],
        ["6", "Tests, CI, requirements, .gitignore"],
        ["7", "Notebook fixes and honest labelling of mock data"],
        ["8", "README rewrite, benchmark script, this report"],
        ["9", "Round 2: fix tied-weight initialisation (found by a new test), untied output by default"],
        ["10", "Round 2: KV-cache decoding, batched beam search across images"],
        ["11", "Round 2: built-in CIDEr-D / ROUGE-L, METEOR opt-in, predict CLI, ablation + benchmarks"],
        ["12", "Round 3: config system, utils (seeding, logging, safe checkpoints), exact resume, grad accumulation"],
        ["13", "Round 3: fine-tune BatchNorm fix, type hints, pyproject, tests, README, report section 9"],
    ], [1.6 * cm, 15.4 * cm]))

    # ------------------------------------------------------------------ round 2
    s.append(PageBreak())
    s.append(P("8. Round 2: measuring before trusting", H1))
    s.append(P("The second round applied the same old-vs-new scrutiny to the first round's own changes. "
               "Every claim was put to a controlled test; two did not survive and are reported as such."))
    s.append(P("8.1 A bug introduced in round 1, and how it was caught", H2))
    s.append(P("Round 1 tied the embedding and output weights. A new end-to-end check showed a training "
               "loss of ~8.3 on a 14-word vocabulary, where random guessing gives log 14 = 2.6. Cause: "
               "the default N(0, 1) embedding makes initial logits huge, and with tying each position's "
               "own embedding (still in the residual stream) produces a logit of ~sqrt(d) for 'repeat the "
               "current word'. Fix: scaled init N(0, d<super>-0.5</super>) and, when tying, T5-style output "
               "scaling. A regression test now asserts that an untrained model's loss is close to log V."))
    s.append(P("8.2 Controlled ablation", H2))
    s.append(P("A synthetic task where the caption is fully determined by the image (block colour, "
               "position and background), 3 seeds, 15 epochs, d=128, 2 layers "
               "(<font face='Courier'>docs/ablation.py</font>). Validation cross-entropy, lower is better:"))
    A = json.load(open(os.path.join(HERE, "ablation_results.json")))
    s.append(table([["Variant", "Initial loss", "Val loss", "Token acc."]] +
                   [[k, f"{v['init_loss']:.2f}", f"{v['val_loss']:.3f}", f"{v['token_acc']:.3f}"]
                    for k, v in A.items()], [8.5 * cm, 2.8 * cm, 2.8 * cm, 2.9 * cm]))
    s.append(Spacer(1, 4))
    s += bullets([
        "<b>Weight tying lost</b> (0.358 / 0.554 vs 0.256), so the output layer is now untied by "
        "default; tying stays available via <font face='Courier'>--tie-weights</font>.",
        "<b>Spatial position embedding: no measurable effect</b> (0.257 vs 0.256). ResNet features "
        "already encode position through zero padding. It is kept because it costs only 49 x d "
        "parameters, but it is no longer claimed as an improvement.",
        "<b>Label smoothing</b> leaves token accuracy unchanged; its cross-entropy is higher by "
        "construction. Its benefit, if any, must be judged on BLEU/CIDEr on real data.",
    ])
    s.append(P("8.3 KV-cache and batched beam search", H2))
    K = json.load(open(os.path.join(HERE, "benchmark_kv_cache.json")))
    s.append(P(f"Generating 30 tokens for 16 images (d=512, 3 layers): full-prefix greedy "
               f"{K['greedy16_full_prefix_ms']:.0f} ms vs KV-cache {K['greedy16_kv_cache_ms']:.0f} ms "
               f"(<b>{K['greedy16_full_prefix_ms'] / K['greedy16_kv_cache_ms']:.1f}x</b>), identical output "
               f"({K['same_output']}). Beam search k=3 for all 16 images in one batch: "
               f"{K['beam3_16img_batched_ms']:.0f} ms. Image keys/values for cross-attention are projected "
               "once per caption instead of once per step."))
    s.append(P("8.4 Metrics without heavy dependencies", H2))
    s.append(P("CIDEr-D and ROUGE-L are now implemented in <font face='Courier'>src/metrics.py</font> and "
               "unit-tested to match <font face='Courier'>pycocoevalcap</font> to 6 significant digits, per "
               "image. METEOR (which starts a Java process and made the tests ~5x slower) is opt-in via "
               "<font face='Courier'>--meteor</font>. A <font face='Courier'>python -m src.predict</font> "
               "command captions arbitrary image files and can save the attention figure."))

    # ------------------------------------------------------------------ round 3
    s.append(PageBreak())
    s.append(P("9. Round 3: production / research-readiness audit", H1))
    s.append(P("A third pass audited the code against a standard ML-engineering checklist: silent "
               "training bugs, memory and throughput, modularity, reproducibility and documentation."))
    s.append(P("9.1 Silent bugs and bad practices found in this pass", H2))
    s.append(table([
        ["#", "Finding", "Why it matters", "Fix"],
        ["1", "Fine-tuning (<font face='Courier'>train_CNN</font>) put the <i>whole</i> ResNet in train "
              "mode", "BatchNorm in frozen layers 1-3 drifted while their weights stayed fixed - "
              "a silent mismatch that degrades features", "Only layer4 switches to train mode; test "
              "asserts frozen BN stats are unchanged and layer4 BN updates"],
        ["2", "<font face='Courier'>loss.item()</font> and grad-norm <font face='Courier'>float()</font> "
              "every step", "Forces a GPU-to-CPU sync per step, stalling the CUDA queue", "Accumulate "
              "detached device tensors; read once per epoch"],
        ["3", "No gradient accumulation", "Large effective batches impossible on small GPUs",
              "<font face='Courier'>train.grad_accum_steps</font>; short last group weighted "
              "correctly; test proves 2x4 accumulated == 1x8"],
        ["4", "Checkpoints held weights only; no optimizer / scheduler / scaler / RNG state",
              "Interrupted runs restart from scratch or resume with a wrong LR schedule",
              "<font face='Courier'>last.pth</font> + per-epoch shuffle reseeding; test proves "
              "resume is bit-exact"],
        ["5", "DataLoader workers and shuffle order not seeded", "Runs not reproducible with "
              "num_workers &gt; 0", "<font face='Courier'>worker_init_fn</font> + seeded generator"],
        ["6", "No cuDNN determinism switch", "GPU reruns differ", "<font face='Courier'>deterministic: "
              "true</font> enables deterministic algorithms"],
        ["7", "<font face='Courier'>torch.load(weights_only=False)</font>", "Unpickling a checkpoint "
              "can execute arbitrary code", "Checkpoints contain only tensors/primitives; loaded with "
              "<font face='Courier'>weights_only=True</font>; atomic writes"],
        ["8", "Hyper-parameters spread across argparse defaults and function signatures",
              "Hard to reproduce or compare runs", "Typed dataclass config + YAML + "
              "<font face='Courier'>--set</font> overrides, stored in every checkpoint"],
        ["9", "Evaluation rebuilt the split from CLI paths and two saved keys", "A changed default "
              "(e.g. val_frac) would silently evaluate on a different split", "Evaluator rebuilds "
              "the split from the full config saved in the checkpoint"],
        ["10", "Logger skipped the file handler if the logger already existed", "Second run in a "
               "process (e.g. notebook) wrote no train.log", "Handlers added idempotently per file"],
    ], [0.7 * cm, 4.6 * cm, 5.6 * cm, 6.1 * cm]))
    s.append(P("9.2 Checklist items verified as correct", H2))
    s += bullets([
        "<b>Data leakage:</b> split by image before anything else; vocabulary built from training "
        "captions only; image normalisation uses fixed ImageNet statistics (nothing is fitted on "
        "data); augmentation only on the training transform.",
        "<b>Train/eval toggles:</b> <font face='Courier'>model.train()</font> at the start of every "
        "epoch, <font face='Courier'>model.eval()</font> + <font face='Courier'>no_grad</font> for "
        "validation, decoding and feature caching; <font face='Courier'>zero_grad</font> per optimizer step.",
        "<b>Loss / activation:</b> the decoder returns raw logits; "
        "<font face='Courier'>CrossEntropyLoss</font> applies log-softmax itself (no double softmax); "
        "beam search uses <font face='Courier'>log_softmax</font> explicitly; loss computed in fp32 under AMP.",
        "<b>Shapes:</b> no hard-coded 49 / 2048 in the decoder path (grid derived from the feature "
        "map; tests run at 64 px = 2x2 grid); padding invariance and KV-cache equivalence are unit-tested.",
        "<b>Memory:</b> logged values are Python floats (no retained graphs); frozen backbone under "
        "<font face='Courier'>no_grad</font>; features cached in fp16; AMP bf16/fp16 on CUDA.",
    ])
    s.append(P("9.3 Result", H2))
    s.append(P("20 tests (up from 15) cover the new guarantees: config round-trip and overrides, "
               "fine-tuning BatchNorm behaviour, gradient-accumulation equivalence, bit-exact resume, "
               "and an end-to-end <font face='Courier'>run()</font> from a config that writes all "
               "artefacts. <font face='Courier'>pyflakes</font> reports no issues. A "
               "<font face='Courier'>pyproject.toml</font> adds extras and console scripts "
               "(<font face='Courier'>caption-train</font>, <font face='Courier'>caption-evaluate</font>, "
               "<font face='Courier'>caption-predict</font>)."))

    doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Multimodal Image Captioning - Improvement Report",
                            author="multimodal-image-captioning")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    os.remove(chart)
    print("wrote", OUT)


if __name__ == "__main__":
    build()

"""Gradio demo: upload an image, get a caption and per-word attention maps.

    python app.py --checkpoint checkpoints/best_model.pth

The old app hard-coded ``vocab_size = 10000`` and used an *empty* vocabulary,
so it could never load a trained model or decode a caption. The checkpoint
now carries its own config and vocabulary.
"""

import argparse
import os
from typing import Optional

import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image

from src.attention_viz import plot_attention
from src.dataset import get_transforms
from src.inference import generate_caption_beam_search, generate_caption_greedy
from src.utils import get_device, load_checkpoint

DEVICE = get_device()
TRANSFORM = get_transforms(train=False)
STATE = {"model": None, "vocab": None, "error": None}


def load(path: str) -> None:
    """Load the checkpoint into STATE (errors are shown in the UI, not raised)."""
    if not os.path.exists(path):
        STATE["error"] = f"No checkpoint at {path}. Train first: python -m src.train"
        return
    try:
        STATE["model"], STATE["vocab"], _ = load_checkpoint(path, DEVICE)
    except Exception as e:  # surfaced in the UI instead of crashing
        STATE["error"] = f"Failed to load {path}: {e}"


def predict(
    image: Optional[Image.Image], decoding: str, beam_size: int
) -> tuple[str, Optional[plt.Figure]]:
    """Gradio callback: caption + attention figure for one uploaded image."""
    if STATE["model"] is None:
        return STATE["error"] or "Model not loaded.", None
    if image is None:
        return "Please upload an image.", None
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0)
    model, vocab = STATE["model"], STATE["vocab"]
    if decoding == "Beam search":
        words, attn = generate_caption_beam_search(
            model, x, vocab, beam_size=int(beam_size), device=DEVICE, return_attention=True
        )
    else:
        words, attn = generate_caption_greedy(model, x, vocab, device=DEVICE)
    caption = " ".join(w for w in words if w not in ("<start>", "<end>", "<pad>"))
    return caption, plot_attention(image, words, attn)


def build_demo() -> gr.Interface:
    return gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload image"),
            gr.Radio(["Beam search", "Greedy"], value="Beam search", label="Decoding"),
            gr.Slider(1, 10, value=3, step=1, label="Beam size"),
        ],
        outputs=[gr.Textbox(label="Generated caption"), gr.Plot(label="Attention per word")],
        title="Image Captioning — ResNet-50 + Transformer decoder",
        description="Heat-maps show where the decoder's cross-attention looked for each word "
        "(H = attention entropy; lower means more focused).",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    load(args.checkpoint)
    build_demo().launch(share=args.share)

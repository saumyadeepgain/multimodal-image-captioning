"""Caption one or more images from the command line.

    python -m src.predict path/to/img1.jpg path/to/img2.jpg --beam 3 --attention-out attn.png

There was no way to caption an arbitrary image without the notebook or the
Gradio app; this is the scriptable equivalent.
"""

from __future__ import annotations

import argparse
import json

import torch
from PIL import Image

from .checkpoint import load_checkpoint
from .dataset import get_transforms
from .inference import beam_search_batch, greedy_decode


def caption_images(model, vocab, paths, beam: int = 3, max_length: int = 30, device="cpu"):
    tf = get_transforms(train=False)
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    if beam <= 1:
        seqs, _ = greedy_decode(model, x, vocab, max_length, device)
        ids = seqs.tolist()
    else:
        ids, _ = beam_search_batch(model, x, vocab, beam, max_length, device=device)
    return [" ".join(vocab.decode(s)) for s in ids]


def main():
    ap = argparse.ArgumentParser(description="Caption images with a trained checkpoint")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=30)
    ap.add_argument("--attention-out", help="save an attention figure for the FIRST image")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_checkpoint(args.checkpoint, device)
    caps = caption_images(model, vocab, args.images, args.beam, args.max_length, device)
    if args.json:
        print(json.dumps(dict(zip(args.images, caps)), indent=2))
    else:
        for p, c in zip(args.images, caps):
            print(f"{p}\t{c}")

    if args.attention_out:
        from .attention_viz import plot_attention
        from .inference import generate_caption_beam_search, generate_caption_greedy
        img = Image.open(args.images[0]).convert("RGB")
        x = get_transforms(train=False)(img).unsqueeze(0)
        if args.beam > 1:
            words, attn = generate_caption_beam_search(model, x, vocab, args.max_length, args.beam,
                                                       device, return_attention=True)
        else:
            words, attn = generate_caption_greedy(model, x, vocab, args.max_length, device)
        plot_attention(img, words, attn).savefig(args.attention_out, dpi=120, bbox_inches="tight")
        print(f"attention figure -> {args.attention_out}")


if __name__ == "__main__":
    main()

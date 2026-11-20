# Controlled ablations (output layer, spatial position embedding, label smoothing) on a synthetic, learnable captioning task
# (caption depends on block colour/position). 3 seeds, 15 epochs, d=128, 2 layers.
# Usage: python docs/ablation.py . docs/ablation_results.json
import json
import math
import random
import sys

import torch

sys.path.insert(0, sys.argv[1])
from src.dataset import Vocabulary
from src.model import ImageCaptioningModel

torch.set_num_threads(4)
cols = {
    "red": (220, 30, 30),
    "green": (30, 200, 40),
    "blue": (30, 40, 220),
    "yellow": (230, 220, 30),
    "white": (240, 240, 240),
}
shapes = ["left", "right", "top", "bottom"]
rng = random.Random(0)


def make(n):
    X, C = [], []
    for _ in range(n):
        c1, c2 = rng.sample(list(cols), 2)
        pos = rng.choice(shapes)
        img = torch.zeros(3, 64, 64)
        img[:] = torch.tensor(cols[c2])[:, None, None] / 255
        sl = {
            "left": (slice(None), slice(0, 32)),
            "right": (slice(None), slice(32, 64)),
            "top": (slice(0, 32), slice(None)),
            "bottom": (slice(None), slice(None)),
        }[pos]
        if pos == "bottom":
            sl = (slice(32, 64), slice(None))
        img[:, sl[0], sl[1]] = torch.tensor(cols[c1])[:, None, None] / 255
        X.append(img)
        C.append(f"a {c1} block on the {pos} of a {c2} image")
    return torch.stack(X), C


Xtr, Ctr = make(600)
Xva, Cva = make(200)
v = Vocabulary(1)
v.build_vocabulary(Ctr)


def enc(captions):
    return torch.tensor([v.encode(c) for c in captions])


Ytr, Yva = enc(Ctr), enc(Cva)


def run(tie, scale, seed=0, epochs=15, pos=True, ls=0.0):
    torch.manual_seed(seed)
    m = ImageCaptioningModel(128, len(v), 2, 4, pretrained=False, tie_weights=tie)
    if not scale:
        m.decoder.out_scale = 1.0
    if not pos:
        m.encoder.pos_embed.data.zero_()
        m.encoder.pos_embed.requires_grad_(False)
    with torch.no_grad():
        ftr, fva = m.encoder.backbone(Xtr), m.encoder.backbone(Xva)
    opt = torch.optim.AdamW(m.param_groups(1e-3, 0.01), lr=1e-3)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=ls)
    ce_eval = torch.nn.CrossEntropyLoss()
    first = None
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(600)
        for i in range(0, 600, 32):
            b = perm[i : i + 32]
            out = m(captions=Ytr[b], backbone_features=ftr[b])
            loss = ce(out.reshape(-1, len(v)), Ytr[b][:, 1:].reshape(-1))
            if first is None:
                first = loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        out = m(captions=Yva, backbone_features=fva)
        vl = ce_eval(out.reshape(-1, len(v)), Yva[:, 1:].reshape(-1)).item()
        acc = (out.argmax(-1) == Yva[:, 1:]).float().mean().item()
    return first, vl, acc


res = {}
configs = [
    ("untied (default)", dict(tie=False, scale=False)),
    ("tied, no out-scale", dict(tie=True, scale=False)),
    ("tied + out-scale", dict(tie=True, scale=True)),
    ("untied, NO spatial pos-embedding", dict(tie=False, scale=False, pos=False)),
    ("untied, label smoothing 0.1", dict(tie=False, scale=False, ls=0.1)),
]
for name, kw in configs:
    r = [run(seed=s, **kw) for s in range(3)]
    res[name] = {
        k: sum(x[i] for x in r) / 3 for i, k in enumerate(["init_loss", "val_loss", "token_acc"])
    }
    print(name, {k: round(x, 4) for k, x in res[name].items()}, flush=True)
print("log|V| =", round(math.log(len(v)), 3))
json.dump(res, open(sys.argv[2], "w"), indent=1)

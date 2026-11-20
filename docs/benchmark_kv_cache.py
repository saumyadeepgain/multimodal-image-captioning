"""Decoding speed: KV-cache (src/inference.py) vs re-running the full prefix.
Usage: python docs/benchmark_kv_cache.py . docs/benchmark_kv_cache.json"""

import json
import sys
import time

import torch

sys.path.insert(0, sys.argv[1])
from src.dataset import Vocabulary
from src.inference import beam_search_batch, greedy_decode
from src.model import ImageCaptioningModel

torch.set_num_threads(4)
torch.manual_seed(0)
V, E, L, B = 3000, 512, 30, 16
m = ImageCaptioningModel(E, V, 3, 8, pretrained=False).eval()
with torch.no_grad():
    m.decoder.fc_out.bias[2] = -1e4  # never emit <end>: always decode L tokens
vocab = Vocabulary(1)
vocab.build_vocabulary([" ".join(f"w{i}" for i in range(V - 4))])
feats = torch.randn(B, 49, 2048)


@torch.no_grad()
def greedy_full_prefix(feats):
    mem = m.encode(backbone_features=feats)
    seqs = torch.full((feats.size(0), 1), vocab.start_idx)
    for _ in range(L):
        nxt = m.decoder(mem, seqs)[:, -1].argmax(-1)
        seqs = torch.cat([seqs, nxt[:, None]], 1)
    return seqs


def t(fn, n=3):
    fn()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - s) / n * 1000


a = greedy_full_prefix(feats)
b, _ = greedy_decode(m, None, vocab, L, backbone_features=feats)
res = {
    "same_output": bool(torch.equal(a, b)),
    "greedy16_full_prefix_ms": t(lambda: greedy_full_prefix(feats)),
    "greedy16_kv_cache_ms": t(lambda: greedy_decode(m, None, vocab, L, backbone_features=feats)),
    "beam3_16img_batched_ms": t(
        lambda: beam_search_batch(m, None, vocab, 3, L, backbone_features=feats), 2
    ),
}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2], "w"), indent=1)

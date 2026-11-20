# Benchmark: original code (git 27c7a6e) vs. improved code, CPU only.
# Usage: git archive 27c7a6e src | tar -x -C OLD_DIR/old ; python docs/benchmark_old_vs_new.py OLD_DIR $PWD
# (expects the old src/ under OLD_DIR/old/src; writes OLD_DIR/bench.json)
import sys, time, json, importlib, types
import torch, torchvision.models as tvm
torch.set_num_threads(4); torch.manual_seed(0)
S=sys.argv[1]; REPO=sys.argv[2]
_orig=tvm.resnet50
tvm.resnet50=lambda weights=None, **k: _orig(weights=None, **k)   # no download

def load_pkg(root, name):
    sys.path.insert(0, root)
    mods={}
    for m in ["model","inference"]:
        spec=importlib.util.spec_from_file_location(f"{name}.{m}", f"{root}/src/{m}.py",
              submodule_search_locations=None)
        mod=importlib.util.module_from_spec(spec); sys.modules[f"{name}.{m}"]=mod
        mod.__package__=name; spec.loader.exec_module(mod); mods[m]=mod
    sys.path.pop(0); return mods
pkg=types.ModuleType("newsrc"); pkg.__path__=[REPO+"/src"]; sys.modules["newsrc"]=pkg
pkg2=types.ModuleType("oldsrc"); pkg2.__path__=[S+"/old/src"]; sys.modules["oldsrc"]=pkg2
import importlib.util
new=load_pkg(REPO,"newsrc"); old=load_pkg(S+"/old","oldsrc")
sys.path.insert(0,REPO); from src.dataset import Vocabulary
V=3000; E=256; B=16; T=16
res={}
def timeit(fn, n=5, warm=1):
    for _ in range(warm): fn()
    t=time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t)/n*1000

imgs=torch.randn(B,3,224,224); caps=torch.randint(4,V,(B,T)); caps[:,0]=1
# --- training step (images through frozen ResNet)
for tag,m in [("old",old["model"].ImageCaptioningModel(E,V,3,8)),
              ("new",new["model"].ImageCaptioningModel(E,V,3,8,pretrained=False))]:
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],1e-4)
    crit=torch.nn.CrossEntropyLoss(ignore_index=0)
    def step():
        m.train(); out=m(imgs,caps); loss=crit(out.reshape(-1,V),caps[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    bn=m.encoder.resnet[1]; before=bn.running_mean.clone()
    res[f"train_step_ms_{tag}"]=timeit(step,4)
    res[f"bn_drift_{tag}"]=float((bn.running_mean-before).abs().max())
    res[f"trainable_params_{tag}"]=sum(p.numel() for p in m.parameters() if p.requires_grad)
    if tag=="new":
        feats=m.encoder.backbone(imgs).half()
        def step_cached():
            m.train(); out=m(captions=caps,backbone_features=feats)
            loss=crit(out.reshape(-1,V),caps[:,1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step()
        res["train_step_ms_new_cached"]=timeit(step_cached,10)
        newm=m
    else: oldm=m

# --- decoding: vocab with random model; force full length by disabling <end>
vocab=Vocabulary(1); vocab.build_vocabulary([" ".join(f"w{i}" for i in range(V-4))])
for m in (oldm,newm):
    with torch.no_grad(): m.decoder.fc_out.bias[vocab.end_idx]=-1e4
L=20; N=16
res["greedy_16img_ms_old"]=timeit(lambda:[old["inference"].generate_caption_greedy(oldm,imgs[i:i+1],vocab,L,"cpu") for i in range(N)],2)
res["greedy_16img_ms_new"]=timeit(lambda:new["inference"].greedy_decode(newm,imgs[:N],vocab,L,"cpu"),2)
for k in (3,5):
    res[f"beam{k}_ms_old"]=timeit(lambda:old["inference"].generate_caption_beam_search(oldm,imgs[:1],vocab,L,k,"cpu"),2)
    res[f"beam{k}_ms_new"]=timeit(lambda:new["inference"].beam_search(newm,imgs[:1],vocab,k,L,device="cpu"),2)
# --- old crash: long generation beyond PE length
try:
    old["inference"].generate_caption_greedy(oldm,imgs[:1],vocab,60,"cpu"); res["old_len60"]="ok"
except Exception as e: res["old_len60"]=type(e).__name__
# --- tokenizer speed
caps_txt=["A black dog is running through the green grass , chasing a red ball ."]*40000
from src.dataset import tokenize
res["tokenize_40k_ms_regex"]=timeit(lambda:[tokenize(c) for c in caps_txt],1,0)
try:
    import nltk; nltk.word_tokenize("a test")
    res["tokenize_40k_ms_nltk"]=timeit(lambda:[nltk.word_tokenize(c.lower()) for c in caps_txt],1,0)
except Exception as e: res["tokenize_40k_ms_nltk"]=f"fails: {type(e).__name__}"
print(json.dumps(res,indent=1)); json.dump(res,open(S+"/bench.json","w"),indent=1)

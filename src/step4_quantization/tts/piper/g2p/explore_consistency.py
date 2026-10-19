# checks that espeak(vi) phonemes of a written syllable do not depend on context (result: 0 conflicts / 2070 syllables)
import json, re, time, collections, sys
sys.stdout.reconfigure(encoding="utf-8")
from piper import PiperVoice
V="../outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx"
v=PiperVoice.load(V,config_path=V+".json")
cfg=json.load(open(V+".json",encoding="utf-8")); m=cfg["phoneme_id_map"]
PUNCT=set(".,!?;:")
sents=[s.strip() for s in open("data/fleurs_vi.txt",encoding="utf-8") if s.strip()]
t0=time.time(); ph=v.phonemize("Xin chào"); print("phonemize ms",(time.time()-t0)*1000)
LET=re.compile(r"^[^\W\d_]+[.,!?;:]*$")
tab=collections.defaultdict(collections.Counter); used=0; skipped=0; t0=time.time()
for s in sents[:1500]:
    toks=s.split()
    if not all(LET.match(t) for t in toks): skipped+=1; continue
    p=v.phonemize(s)
    if len(p)!=1: skipped+=1; continue
    words=("".join(p[0])).split(" ")
    if len(words)!=len(toks): skipped+=1; continue
    used+=1
    for t,w in zip(toks,words):
        core=t.rstrip(".,!?;:"); 
        tab[core.lower()][w.rstrip(".,!?;:")]+=1
print("used",used,"skipped",skipped,"time",time.time()-t0,"unique syl",len(tab))
conf={k:c for k,c in tab.items() if len(c)>1}
print("syllables with >1 phoneme string:",len(conf))
for k,c in list(conf.items())[:15]: print(" ",k,dict(c))

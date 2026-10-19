"""Build the syllable->phoneme lookup tables (espeak-ng via piper) and measure how often the tensor G2P matches espeak on FLEURS-vi test.
Run from piper_npu/g2p after fetch_data.py."""
# -*- coding: utf-8 -*-
import json, re, sys, time, unicodedata, collections, pickle, os
os.makedirs('build', exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, torch
sys.path.insert(0, ".")
import g2p_graph as G
from piper import PiperVoice
V="../outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx"
voice=PiperVoice.load(V,config_path=V+".json"); idm=json.load(open(V+".json",encoding="utf-8"))["phoneme_id_map"]
NFC=G.NFC
def phon_ids(word):
    p=voice.phonemize(word)
    if len(p)!=1: return None
    try: return [idm[c][0] for c in p[0]]
    except KeyError: return None
letters,classes,pats,pcls=G.build_patterns()
# vocabulary: phobert syllables + fleurs train/dev words (test split held out)
syl=[l.strip() for l in open("data/syllables_phobert.txt",encoding="utf-8") if l.strip()]
tsv=lambda sp:[l.rstrip("\n").split("\t")[2] for l in open(__import__("huggingface_hub").hf_hub_download("google/fleurs",f"data/vi_vn/{sp}.tsv",repo_type="dataset"),encoding="utf-8") if l.count("\t")>3]
train=tsv("train")+tsv("dev"); test=tsv("test")
def words(sents): return {NFC(w.lower()) for s in sents for w in re.findall(r"[^\W\d_]+",s)}
vocab=set(syl)|{w for w in words(train) if w.isalpha()}

# ---- combinatorial completion: onsets x rimes learned from the phobert syllable list + tone-placement variants
ONS=["ngh","ng","nh","ph","th","tr","ch","kh","gh","gi","qu","b","c","d","đ","g","h","k","l","m","n","p","q","r","s","t","v","x"]
def split_onset(w):
    for o in ONS:
        if w.startswith(o) and len(w)>len(o): return o,w[len(o):]
    return "",w
rimes=collections.Counter(); onsets=set([""])
for w in syl:
    o,r=split_onset(w)
    if any(ch in "aăâeêioôơuưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ" for ch in r[:1]): rimes[r]+=1; onsets.add(o)
FRONT=set("eêiyéèẻẽẹếềểễệíìỉĩịýỳỷỹỵ")
def legal(o,r):
    f=r[0] in FRONT
    if o=="k" and not f: return False
    if o in ("c","q") and (f and o=="c"): return False
    if o=="gh" and not f: return False
    if o=="g" and f: return False
    if o=="ngh" and not f: return False
    if o=="ng" and f: return False
    if o=="qu" and r[0] in "uú": return False
    if o=="q": return False
    return True
def tone_of(r):
    d=unicodedata.normalize("NFD",r)
    for ch in d:
        if ch in "̀̉̃": return "other"
        if ch in "̣́": return "sn"
    return "none"
def coda_ok(r):
    stop=re.search(r"(c|ch|p|t)$",r) is not None
    return (not stop) or tone_of(r)=="sn"
good_rimes={r for r,c in rimes.items() if c>=3 and coda_ok(r)}
gen_words={o+r for o in onsets for r in good_rimes if legal(o,r)}
TM="̣̀́̉̃"
def old_style(w):
    d=unicodedata.normalize("NFD",w); out={w}
    for pair in ("oa","oe","uy"):
        # tone on 2nd vowel -> tone on 1st vowel
        m=re.search(pair[0]+"([̣̀́̉̃]?)"+pair[1]+"([̣̀́̉̃])",d)
        if m and not m.group(1):
            out.add(NFC(d[:m.start()]+pair[0]+m.group(2)+pair[1]+d[m.end():]))
    return out
for w in list(gen_words)+list(vocab):
    gen_words|=old_style(NFC(w))
print("generated candidates",len(gen_words))
vocab=set(vocab)|gen_words
# tone-placement variants (hòa/hoà, thủy/thuỷ ...)
def variants(w):
    out={w}
    d=unicodedata.normalize("NFD",w)
    for a,b in [("oa","oa"),("oe","oe"),("uy","uy")]:
        pass
    return out
KPMAX=14; tab={}; t0=time.time()
for w in sorted(vocab):
    if len(w)>G.S: continue
    ids=phon_ids(w)
    if ids and len(ids)<=KPMAX: tab[w]=ids
print("table words",len(tab),"build s",round(time.time()-t0,1),"max phonemes/token",max(len(v) for v in tab.values()))
for pch in G.PUNCT_CLASSES: tab[pch]=[idm[pch][0]]
tab[" "]=[3]
keys,phs,Cp=G.build_tables(list(tab),lambda t:tab[t],classes,letters,idm)
print("entries",len(keys),"C",len(classes))
pickle.dump((letters,classes,pats,pcls,keys,phs,tab),open("build/g2p_tables.pkl","wb"))
model=G.VietG2P(letters,classes,pats,pcls,keys,phs).eval()
print("params(buffers) MB fp32:",round(sum(b.numel() for b in model.buffers())*4/1e6,1))
def run(text):
    b=NFC(text).encode("utf-8")[:G.LB]; a=np.zeros((1,G.LB),np.int32); a[0,:len(b)]=list(b)
    with torch.no_grad(): x,xl,oov=model(torch.from_numpy(a))
    return x[0,:int(xl[0])].tolist(), int(oov[0])
SUPPORTED=re.compile(r"^[^\W\d_\s\.,!?;:\"()\-'/“”‘’«»\[\]–—]*$")
def supported(s): return all(re.match(r"[^\W\d_]",c) or c.isspace() or c in '.,!?;:"()-\'/“”‘’«»[]–—' for c in s)
stats=collections.Counter(); bad=[]
for s in test:
    s=s.strip()
    if not supported(s): stats["unsupported_chars(digits/symbols)"]+=1; continue
    ph=voice.phonemize(s)
    if len(ph)!=1: stats["multi_sentence(skip)"]+=1; continue
    ref=voice.phonemes_to_ids(ph[0])
    if len(ref)>512: stats['ref>512ids(out of model spec)']+=1; continue
    got,oov=run(s)
    if got==ref: stats["EXACT"]+=1
    else:
        stats["mismatch_oov>0" if oov>0 else "mismatch_oov=0"]+=1; bad.append((s,oov,ref,got))
print(dict(stats))
for s,oov,ref,got in bad[:6]:
    print("-",s[:90],"| oov",oov,"| len ref/got",len(ref),len(got))
pickle.dump(bad,open("build/bad.pkl","wb"))

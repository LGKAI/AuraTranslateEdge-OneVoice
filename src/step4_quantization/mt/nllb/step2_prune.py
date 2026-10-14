import sys, json, re, torch, numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from nllb_static import NLLBStatic
NAME="facebook/nllb-200-distilled-600M"; tok=AutoTokenizer.from_pretrained(NAME)
LANG={"vi_vn":"vie_Latn","en_us":"eng_Latn","ko_kr":"kor_Hang","cmn_hans_cn":"zho_Hans"}
keep=set(tok.all_special_ids)|{tok.convert_tokens_to_ids(l) for l in LANG.values()}; texts={}
for fl,nl in LANG.items():
    rows=[]
    for sp in ["train","dev","test"]:
        p=hf_hub_download("google/fleurs",f"data/{fl}/{sp}.tsv",repo_type="dataset")
        rows.append([l.rstrip("\n").split("\t")[2] for l in open(p,encoding="utf-8") if l.count("\t")>3])
    texts[nl]=rows
    tok.src_lang=nl
    for s in rows[0]+rows[1]: keep.update(tok(s)["input_ids"])
keep=sorted(keep); print("kept tokens:",len(keep),"of",len(tok))
json.dump({"keep":keep},open("work/keep_ids.json","w"))
# coverage on held-out test split
tot=miss=0
for nl,rows in texts.items():
    tok.src_lang=nl
    for s in rows[2]:
        ids=tok(s)["input_ids"]; tot+=len(ids); miss+=sum(1 for i in ids if i not in set(keep))
print(f"test-split token coverage: {100*(1-miss/tot):.3f}% ({miss} unseen of {tot})")
json.dump({k:v[2][:60] for k,v in texts.items()},open("work/fleurs_test.json","w",encoding="utf-8"),ensure_ascii=False)

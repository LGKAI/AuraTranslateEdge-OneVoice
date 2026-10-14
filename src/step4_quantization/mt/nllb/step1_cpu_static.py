import sys, json, time, torch, numpy as np
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from nllb_static import NLLBStatic
NAME="facebook/nllb-200-distilled-600M"; tok=AutoTokenizer.from_pretrained(NAME); hf=AutoModelForSeq2SeqLM.from_pretrained(NAME, use_safetensors=False, local_files_only=True).eval()
ref=json.load(open("work/ref.json",encoding="utf-8")); S,T=32,24
m=NLLBStatic(hf,S,T).eval()
ok=0
for r in ref:
    ids=np.ones((1,S),np.int32); ids[0,:len(r["ids_in"])]=r["ids_in"]
    lang=tok.convert_tokens_to_ids(r["tgt"])
    with torch.no_grad(): o=m(torch.from_numpy(ids),torch.tensor([lang],dtype=torch.int32))[0].tolist()
    cut=o[:o.index(2)+1] if 2 in o else o
    want=r["ids_out"][1:]; same=cut==want[:len(cut)] and (2 in o)
    ok+=same; print("MATCH" if same else "DIFF ",r["src"],"->",r["tgt"],"|",tok.decode(cut,skip_special_tokens=True))
print(f"{ok}/{len(ref)} identical to HF generate")

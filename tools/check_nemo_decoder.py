from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
p=Path(__file__).resolve().parents[1]/'models/source/nemo-en-tn/hf';torch.set_num_threads(4);t=AutoTokenizer.from_pretrained(str(p));m=AutoModelForSeq2SeqLM.from_pretrained(str(p),dtype=torch.bfloat16).eval().to('mps')
for s in ['0 The discount is <extra_id_0> 12.5 % <extra_id_1> .','0 The discount is <extra_id_0> 12.5 percent <extra_id_1> .','0 The discount is <extra_id_0> 15% <extra_id_1> .','0 The discount is <extra_id_0> 15 percent <extra_id_1> .']:
 x=t(s,return_tensors='pt').to('mps')
 with torch.inference_mode():y=m.generate(**x,max_new_tokens=80)
 print(s,'=>',t.decode(y[0],skip_special_tokens=True),flush=True)

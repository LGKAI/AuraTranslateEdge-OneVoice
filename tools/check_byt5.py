from pathlib import Path
import torch,json,time
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
r=Path(__file__).resolve().parents[1];p=r/'models/source/byt5-vi-normalization';torch.set_num_threads(4)
tok=AutoTokenizer.from_pretrained(str(p));model=AutoModelForSeq2SeqLM.from_pretrained(str(p),dtype=torch.bfloat16).eval().to('mps')
cases=['Cuộc họp lúc 3:30 chiều ngày 15/11/2026.','Giá vé 50.000 đồng, giảm 15%. Nhiệt độ 37,5°C.','Xin gọi số 0912 345 678. Chuyến bay VN123, mã ZIP 67297.','Phòng 330 có giá 330 đồng. Cuộc họp lúc 14h45.']
out=[]
for t in cases:
 start=time.perf_counter();x=tok('Normalize: '+t,return_tensors='pt').to('mps')
 with torch.inference_mode():y=model.generate(**x,max_new_tokens=2048,num_beams=2)
 text=tok.decode(y[0],skip_special_tokens=True);d={'input':t,'output':text,'seconds':time.perf_counter()-start};out.append(d);print(json.dumps(d,ensure_ascii=False),flush=True)
(r/'results/byt5_smoke.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))

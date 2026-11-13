from pathlib import Path
import torch,numpy as np,json,onnxruntime as ort
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
p=root/'models/source/nllb-hf'
t=AutoTokenizer.from_pretrained(str(p),src_lang='eng_Latn');m=AutoModelForSeq2SeqLM.from_pretrained(str(p)).eval()
a=t('The ticket costs fifteen dollars.',return_tensors='pt');seq=a['input_ids'].numpy().astype(np.int32);x=np.ones((1,64),np.int32);x[0,:seq.shape[1]]=seq
so=ort.SessionOptions();so.intra_op_num_threads=4;so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
enc=ort.InferenceSession(str(root/'build/nllb/nllb_enc.onnx'),so);dec=ort.InferenceSession(str(root/'build/nllb/nllb_decB.onnx'),so)
ck,cv,bias=enc.run(None,{'src_ids':x});keep=np.load(root/'s2s_edge/assets/nllb_keep.npy');sk=np.zeros((12,4,16,96,64),np.float32);sv=sk.copy();out=[]
# Compare forced prefix steps with HF, tolerating FP16 embedding rounding.
prefix=[2,256006]
for i,tok in enumerate(prefix):
 oh=np.zeros(96,np.float32);oh[i]=1;mask=np.where(np.arange(96)<=i,0,-1e4).astype(np.float32)
 lg,sk,sv=dec.run(None,{'tok':np.full(4,tok,np.int32),'oh':oh,'mask':mask,'sk':sk,'sv':sv,'ck':ck,'cv':cv,'bias':bias})
 with torch.no_grad():ref=m(input_ids=torch.tensor(x.astype('int64')),attention_mask=torch.tensor((x!=1).astype('int64')),decoder_input_ids=torch.tensor([prefix[:i+1]])).logits[0,-1].numpy()[keep]
 delta=lg[0]-ref;out.append({'step':i,'mean_abs_logit_error':float(abs(delta).mean()),'max_abs_logit_error':float(abs(delta).max()),'argmax_equal':bool(lg[0].argmax()==ref.argmax())})
print(json.dumps(out,indent=2));(root/'results/mt_numerics.json').write_text(json.dumps(out,indent=2))

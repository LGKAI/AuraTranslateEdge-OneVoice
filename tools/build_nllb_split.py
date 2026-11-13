"""Re-export missing split/beam wrappers using branch nllb's NLLBStatic math
and contest_VNG's exact tensor contracts. Same pretrained weights and keep IDs.
"""
from pathlib import Path
import sys,gc,json,time
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F,onnxruntime as ort
from transformers import AutoModelForSeq2SeqLM
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src/step4_quantization/mt/nllb'))
from nllb_static import NLLBStatic
S,T,H,D,L=64,96,16,64,12
torch.set_num_threads(4)
class Encoder(nn.Module):
 def __init__(self,base):super().__init__();self.b=base
 def forward(self,src_ids):
  e,bias=self.b.encode(src_ids)
  k=torch.stack([self.b._split(x.encoder_attn.k_proj(e),S) for x in self.b.dec_layers])
  v=torch.stack([self.b._split(x.encoder_attn.v_proj(e),S) for x in self.b.dec_layers])
  return k,v,bias
class Decoder(nn.Module):
 def __init__(self,base,B):super().__init__();self.b=base;self.B=B
 def split(self,x):return x.reshape(self.B,1,H,D).permute(0,2,1,3)
 def merge(self,x):return x.permute(0,2,1,3).reshape(self.B,1,H*D)
 def attn(self,q,k,v,bias=None):
  s=(q*(D**-.5))@k.transpose(-1,-2)
  return torch.softmax(s+(bias if bias is not None else 0),-1)@v
 def forward(self,tok,oh,mask,sk,sv,ck,cv,bias):
  b,B=self.b,self.B
  ti=b.fwd[tok.long()].long();pos=oh@b.pos_dec[:T]
  x=(b.emb[ti].float()*b.scale+pos).reshape(B,1,H*D)
  if B==1:sk=sk.unsqueeze(1);sv=sv.unsqueeze(1)
  ks=[];vs=[];write=oh.reshape(1,1,T,1)
  for li,layer in enumerate(b.dec_layers):
   y=layer.self_attn_layer_norm(x);a=layer.self_attn
   k=sk[li]*(1-write)+self.split(a.k_proj(y))*write
   v=sv[li]*(1-write)+self.split(a.v_proj(y))*write
   x=x+a.out_proj(self.merge(self.attn(self.split(a.q_proj(y)),k,v,mask.reshape(1,1,1,T))))
   y=layer.encoder_attn_layer_norm(x);a=layer.encoder_attn
   x=x+a.out_proj(self.merge(self.attn(self.split(a.q_proj(y)),ck[li].unsqueeze(0),cv[li].unsqueeze(0),bias.unsqueeze(0))))
   y=layer.final_layer_norm(x);x=x+layer.fc2(F.relu(layer.fc1(y)));ks.append(k);vs.append(v)
  logits=F.linear(b.dec_ln(x).reshape(B,H*D).to(b.emb.dtype),b.emb).float()
  k=torch.stack(ks);v=torch.stack(vs)
  if B==1:return b.inv[logits.argmax(-1)].int(),k.squeeze(1),v.squeeze(1)
  return logits,k,v
print('LOAD_HF',flush=True)
hf=AutoModelForSeq2SeqLM.from_pretrained(str(root/'models/source/nllb-hf')).eval()
keep=np.load(root/'s2s_edge/assets/nllb_keep.npy').tolist();base=NLLBStatic(hf,S=S,T=T,keep_ids=keep,half_emb=True).eval();del hf;gc.collect()
out=root/'build/nllb';out.mkdir(exist_ok=True)
a=torch.ones(1,S,dtype=torch.int32);a[0,:3]=torch.tensor([256047,35378,2],dtype=torch.int32)
e=Encoder(base).eval()
with torch.no_grad():ck,cv,bias=e(a)
print('EXPORT_ENCODER',flush=True)
torch.onnx.export(e,(a,),str(out/'nllb_enc.onnx'),input_names=['src_ids'],output_names=['ck','cv','bias'],opset_version=17,do_constant_folding=True,dynamo=False)
print('ENCODER_READY',flush=True)
for B in [1,4]:
 dec=Decoder(base,B).eval();sk=torch.zeros((L,H,T,D) if B==1 else (L,B,H,T,D));oh=torch.zeros(T);oh[0]=1;mask=torch.full((T,),-1e4);mask[0]=0
 args=(torch.full((B,),2,dtype=torch.int32),oh,mask,sk,sk.clone(),ck,cv,bias)
 name='nllb_dec.onnx' if B==1 else 'nllb_decB.onnx';print('EXPORT_DECODER',B,flush=True)
 torch.onnx.export(dec,args,str(out/name),input_names=['tok','oh','mask','sk','sv','ck','cv','bias'],output_names=['nxt' if B==1 else 'logits','sk_out','sv_out'],opset_version=17,do_constant_folding=True,dynamo=False)
 print('DECODER_READY',B,(out/name).stat().st_size,flush=True)
print('NLLB_SPLIT_READY',len(keep),flush=True)

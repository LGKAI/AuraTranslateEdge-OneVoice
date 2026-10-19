import sys, unicodedata, numpy as np, torch, onnxruntime as ort
sys.path.insert(0,"MeloTTS"); sys.path.insert(0,"."); sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForMaskedLM
from melo_front import load_tables, LB, N, L
from melo_full import BertStatic
from melo_graphA import GraphA
name="hfl/chinese-roberta-wwm-ext-large"; tok=AutoTokenizer.from_pretrained(name); mlm=AutoModelForMaskedLM.from_pretrained(name).eval()
G=GraphA(load_tables(),BertStatic(mlm,L),tok).eval()
tx=["然而事实并非如此","手写原稿未能保留下来","州长办公室表示，伤者中有十九人是警察","植物在自然环境中状态最好"]
def enc(t): b=np.zeros((1,LB),np.int32); by=unicodedata.normalize("NFC",t).encode(); b[0,:len(by)]=list(by); return b
B=np.concatenate([enc(t) for t in tx]); np.savez("work/graphA_inputs.npz", text_bytes=B)
torch.onnx.export(G,(torch.from_numpy(B[:1]),),"work/graphA.onnx",input_names=["text_bytes"],output_names=["output_0","output_1","output_2","output_3","output_4"],opset_version=17,do_constant_folding=True,dynamo=False)
import os; print("onnx MB", os.path.getsize("work/graphA.onnx")/1e6)
s=ort.InferenceSession("work/graphA.onnx",providers=["CPUExecutionProvider"])
for i in range(len(tx)):
    r=s.run(None,{"text_bytes":B[i:i+1]}); 
    with torch.no_grad(): t=G(torch.from_numpy(B[i:i+1]))
    print(tx[i], "ort-vs-torch maxdiff bert", float(np.abs(r[4]-t[4].numpy()).max()), "x equal", np.array_equal(r[0],t[0].numpy()))

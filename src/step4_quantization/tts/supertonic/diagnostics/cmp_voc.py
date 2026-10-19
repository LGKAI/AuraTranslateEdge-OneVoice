import os
import sys, json, re, numpy as np, onnxruntime as ort, jiwer
from scipy.signal import resample_poly
sys.stdout.reconfigure(encoding="utf-8")
SENT=json.load(open("../data/sentences.json",encoding="utf-8")); d=np.load("hw_full_dbg.npz"); cpu=["CPUExecutionProvider"]
voc=ort.InferenceSession("static_N96_L150/vocoder_static.onnx",providers=cpu)
asr=ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"),providers=cpu)
LID={"en":4,"ko":12}
nw=lambda s: re.sub(r"\s+"," ",re.sub(r"[^\w\s]","",s.lower().strip())).strip(); nc=lambda s: re.sub(r"\s+","",re.sub(r"[^\w\s]","",s.lower()))
def at(w,lang):
    w=resample_poly(w.astype(np.float32),16000,44100).astype(np.float32); n=min(len(w),464000); b=np.zeros(464000,np.float32); b[:n]=w[:n]
    o=asr.run(None,{"wav":b.reshape(1,-1),"wav_len":np.array([n],np.int32),"language":np.array([LID[lang]],np.int32),"textnorm":np.array([14],np.int32)})
    return bytes(int(x)&0xFF for x in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00",b"").decode("utf-8",errors="replace").strip()
res={"en":[],"ko":[]}
for i,(lang,text) in enumerate(SENT):
    L=int(round(float(d["output_1"][i][0]))); x5=d["output_4"][i].astype(np.float32)           # NPU final latent
    w=voc.run(None,{"latent":x5})[0].reshape(-1)[:L*3072]                                    # vocoder fp32 CPU
    h=at(w,lang); res[lang].append((jiwer.wer(nw(text),nw(h)) if nw(h) else 1, jiwer.cer(nc(text),nc(h)) if nc(h) else 1))
    print(i,lang,"->",h)
for l in res: print("NPU latent + CPU fp32 vocoder",l,"WER %.1f%% CER %.1f%%"%(np.mean([a for a,b in res[l]])*100,np.mean([b for a,b in res[l]])*100))

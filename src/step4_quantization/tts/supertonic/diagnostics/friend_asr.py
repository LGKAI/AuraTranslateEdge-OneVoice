import os
import sys, re, numpy as np, onnxruntime as ort, soundfile as sf, jiwer
from scipy.signal import resample_poly
sys.stdout.reconfigure(encoding="utf-8")
asr=ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"),providers=["CPUExecutionProvider"])
LID={"en":4,"ko":12}
nw=lambda s: re.sub(r"\s+"," ",re.sub(r"[^\w\s]","",s.lower().strip())).strip(); nc=lambda s: re.sub(r"\s+","",re.sub(r"[^\w\s]","",s.lower()))
EN="The OneVoice AI Challenge runs on Qualcomm Dragonwing NPU, delivering ultra low latency and high quality speech synthesis."
KO="퀄컴 드래곤윙 NPU는 엣지 디바이스에서 실시간으로 음성을 합성하며, 뛰어난 전력 효율과 뛰어난 음질을 제공합니다."
for f,lang,ref in [("live_dragonwing_fresh_english_complete","en",EN),("live_dragonwing_fresh_korean_complete","ko",KO)]:
    w,sr=sf.read(f"friend_wav/{f}.wav",dtype="float32"); w=w if w.ndim==1 else w[:,0]
    # ASR window is 29 s max; clip files are ~12 s
    x=resample_poly(w,16000,sr).astype(np.float32); n=min(len(x),464000); b=np.zeros(464000,np.float32); b[:n]=x[:n]
    o=asr.run(None,{"wav":b.reshape(1,-1),"wav_len":np.array([n],np.int32),"language":np.array([LID[lang]],np.int32),"textnorm":np.array([14],np.int32)})
    h=bytes(int(v)&0xFF for v in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00",b"").decode("utf-8",errors="replace").strip()
    print(f"{f} ({len(w)/sr:.1f}s, sr {sr})\n  ref: {ref}\n  hyp: {h}\n  WER {jiwer.wer(nw(ref),nw(h)):.2f} CER {jiwer.cer(nc(ref),nc(h)):.2f}")

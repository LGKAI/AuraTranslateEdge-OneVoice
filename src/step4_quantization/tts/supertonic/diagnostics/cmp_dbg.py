import sys, json, numpy as np, onnxruntime as ort
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS
N=96; tts=TTS(); m=tts.model; st=tts.get_voice_style(voice_name="M1")
SENT=json.load(open("../data/sentences.json",encoding="utf-8")); d=np.load("hw_full_dbg.npz")
sess=ort.InferenceSession("supertonic_full_N96_L150_s5_dbg.onnx",providers=["CPUExecutionProvider"])
cos=lambda a,b: float(np.dot(a.ravel().astype(np.float64),b.ravel().astype(np.float64))/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
for i in [0,1,6]:
    lang,text=SENT[i]; v=[ord(c) for c in m.text_processor._preprocess_text(text,lang)]; cp=np.zeros((1,N),np.int32); cp[0,:len(v)]=v
    wav,lat,temb,x1,x5,dur=sess.run(None,{"cp":cp,"style_dp":st.dp,"style_ttl":st.ttl}); n=len(v); L=int(lat[0])
    h=lambda k:d[k][i]
    print(f"[{i}] {lang} n={n} lat {L}/{int(round(float(h('output_1')[0])))} | dur_raw cpu {float(dur[0]):.4f} npu {float(h('output_5')[0]):.4f}")
    print(f"     text_emb cos {cos(temb[:,:,:n],h('output_2')[:,:,:n]):.5f} (cpu max {np.abs(temb).max():.2f} npu max {np.abs(h('output_2')).max():.2f})")
    print(f"     latent after step1 cos {cos(x1[:,:,:L],h('output_3')[:,:,:L]):.5f} | after step5 cos {cos(x5[:,:,:L],h('output_4')[:,:,:L]):.5f}")
    print(f"     wav cos {cos(wav[0,:L*3072],h('output_0')[0,:L*3072]):.5f}")

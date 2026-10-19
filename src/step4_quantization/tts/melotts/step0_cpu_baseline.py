# -*- coding: utf-8 -*-
"""CPU baseline: MeloTTS-Chinese -> wav, then SenseVoice fp32 ASR (zh) -> CER.  Also prints model structure."""
import sys, os, json, time, re
sys.path.insert(0, "MeloTTS"); sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, onnxruntime as ort, jiwer, torch
from scipy.signal import resample_poly
from melo.api import TTS
SENT = ["今天天气真好，我们一起去公园散步吧。", "请把客厅的灯打开。", "我想预订今晚两个人的餐桌。", "机器翻译可以帮助人们跨越语言沟通。",
        "火车站怎么走最快？", "这家餐厅的菜非常好吃。", "明天早上我有一个重要的会议。", "请问附近有没有药店？"]
t0 = time.time(); tts = TTS(language="ZH", device="cpu"); print("load s", round(time.time() - t0, 1)); m = tts.model
print("params M:", round(sum(p.numel() for p in m.parameters()) / 1e6, 1), "| sr", tts.hps.data.sampling_rate, "| spk", dict(tts.hps.data.spk2id))
for n, c in m.named_children(): print(f"  {n:12s} {type(c).__name__:28s} {sum(p.numel() for p in c.parameters())/1e6:6.1f} M")
SV = os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx")
asr = ort.InferenceSession(SV, providers=["CPUExecutionProvider"]); os.makedirs("work/wav_cpu", exist_ok=True); rows = []
nc = lambda s: re.sub(r"[^\w]", "", s)
for i, t in enumerate(SENT):
    t1 = time.time(); w = tts.tts_to_file(t, tts.hps.data.spk2id["ZH"], f"work/wav_cpu/{i}.wav", speed=1.0, quiet=True) if False else None
    audio = tts.tts_to_file(t, tts.hps.data.spk2id["ZH"], None, speed=1.0, quiet=True); sr = tts.hps.data.sampling_rate
    sf.write(f"work/wav_cpu/{i}.wav", audio, sr)
    x = resample_poly(audio.astype(np.float32), 16000, sr).astype(np.float32); n = min(len(x), 464000); b = np.zeros(464000, np.float32); b[:n] = x[:n]
    o = asr.run(None, {"wav": b.reshape(1, -1), "wav_len": np.array([n], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    h = bytes(int(v) & 0xFF for v in np.asarray(o[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
    cer = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; rows.append({"text": t, "hyp": h, "cer": cer, "sec": len(audio) / sr, "synth_s": round(time.time() - t1, 2)})
    print(f"[{i}] {len(audio)/sr:.2f}s synth {time.time()-t1:.1f}s CER {cer:.2f} | {t} => {h}", flush=True)
print("mean CER %.3f" % np.mean([r["cer"] for r in rows])); json.dump(rows, open("work/cpu_baseline.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

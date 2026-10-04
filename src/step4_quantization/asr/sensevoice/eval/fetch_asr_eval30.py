# -*- coding: utf-8 -*-
"""Lay bo test MO RONG (30 cau/ngon ngu, KHONG trung 5 cau cu test[:5]) tu google/fleurs
(cung nguon voi data/asr). Luu vao data/asr_eval30/ + manifest_eval30.json (khong ghi de manifest cu).
Chi lay cau <=29s (gioi han static shape cua model) va co transcript."""
import io, os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT = os.path.join(ROOT, "data", "asr_eval30")
SR = 16000
MAX_S = 29.0
N_PER_LANG = 30
START = 5   # bo qua 5 cau dau (da dung o bo test cu)
CONF = {"en": "en_us", "zh": "cmn_hans_cn", "ko": "ko_kr"}

manifest = []
for lang, conf in CONF.items():
    print(f"[{lang}] loading google/fleurs {conf} test[{START}:{START+80}] ...", flush=True)
    # streaming: chi doc split test (tranh tai ca train/dev ~GB moi ngon ngu)
    ds = load_dataset("google/fleurs", conf, split="test", streaming=True, trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(decode=False)).skip(START).take(80)
    got = 0
    for i, ex in enumerate(ds):
        if got >= N_PER_LANG:
            break
        a = ex["audio"]
        raw = a.get("bytes")
        wav, sr = sf.read(io.BytesIO(raw) if raw is not None else a["path"], dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        text = (ex.get("transcription") or ex.get("raw_transcription") or "").strip()
        dur = len(wav) / SR
        if not text or dur > MAX_S or dur < 1.0:
            continue
        d = os.path.join(OUT, lang)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{lang}_e{got}.wav")
        sf.write(p, wav, SR, subtype="PCM_16")
        manifest.append({"lang": lang, "path": os.path.relpath(p, ROOT).replace("\\", "/"),
                         "transcript": text, "duration_s": round(dur, 3)})
        got += 1
    print(f"[{lang}] saved {got} clips", flush=True)

with open(os.path.join(ROOT, "data", "manifest_eval30.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("TOTAL", len(manifest), flush=True)

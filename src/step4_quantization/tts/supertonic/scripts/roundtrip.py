# -*- coding: utf-8 -*-
"""Step 1 quality gate: TTS (original dynamic vs static padded chain, both CPU fp32) -> SenseVoice fp32 ASR -> WER/CER vs input text.
usage: python roundtrip.py N L"""
import sys, os, json, re, time
import numpy as np, onnxruntime as ort, jiwer
from scipy.signal import resample_poly
sys.stdout.reconfigure(encoding="utf-8")
from supertonic import TTS

N, L = int(sys.argv[1]), int(sys.argv[2])
HERE = (os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); D = os.path.join(HERE, f"static_N{N}_L{L}")
SV = os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx")
tts = TTS(); m = tts.model; style = tts.get_voice_style(voice_name="M1")
cpu = ["CPUExecutionProvider"]
S = {k: ort.InferenceSession(os.path.join(D, f"{k}_static.onnx"), providers=cpu) for k in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]}
asr = ort.InferenceSession(SV, providers=cpu)
SENT = [("en", "The quick brown fox jumps over the lazy dog near the river bank."),
        ("en", "Please turn on the lights in the living room."),
        ("en", "Machine translation helps people communicate across languages."),
        ("en", "I would like to book a table for two people tonight."),
        ("en", "The weather today is sunny with a light breeze."),
        ("en", "Please call me back as soon as you finish the meeting."),
        ("ko", "안녕하세요, 오늘 날씨가 정말 좋네요."),
        ("ko", "저는 내일 아침에 회의가 있어서 일찍 일어나야 합니다."),
        ("ko", "이 음식은 정말 맛있어서 또 먹고 싶어요."),
        ("ko", "기차역까지 가는 가장 빠른 길을 알려 주세요."),
        ("ko", "인공지능 기술은 우리 생활을 빠르게 바꾸고 있습니다."),
        ("ko", "주말에는 가족과 함께 공원에서 산책을 합니다.")]
LID = {"zh": 3, "en": 4, "ko": 12}


def static_chain(text, lang, seed):
    ids, mask = m.text_processor([text], lang); n = ids.shape[1]
    assert n <= N, f"text too long {n}>{N}"
    ids_p = np.zeros((1, N), np.int64); ids_p[:, :n] = ids
    mask_p = np.zeros((1, 1, N), np.float32); mask_p[:, :, :n] = 1
    dur = S["duration_predictor"].run(None, {"text_ids": ids_p, "style_dp": style.dp, "text_mask": mask_p})[0] / 1.05
    emb = S["text_encoder"].run(None, {"text_ids": ids_p, "style_ttl": style.ttl, "text_mask": mask_p})[0]
    lat = int(np.ceil(int(dur[0] * 44100) / 3072))
    assert lat <= L, f"latent {lat}>{L}"
    lm = np.zeros((1, 1, L), np.float32); lm[:, :, :lat] = 1
    x = np.random.RandomState(seed).randn(1, 144, L).astype(np.float32) * lm
    for s in range(5):
        x = S["vector_estimator"].run(None, {"noisy_latent": x, "text_emb": emb, "style_ttl": style.ttl, "text_mask": mask_p, "latent_mask": lm,
                                             "current_step": np.array([s], np.float32), "total_step": np.array([5], np.float32)})[0]
    w = S["vocoder"].run(None, {"latent": x})[0].reshape(-1)
    return w[:lat * 3072]


def asr_text(wav44, lang):
    w = resample_poly(wav44.astype(np.float32), 16000, 44100).astype(np.float32)
    n = min(len(w), 464000); buf = np.zeros(464000, np.float32); buf[:n] = w[:n]
    o = asr.run(None, {"wav": buf.reshape(1, -1), "wav_len": np.array([n], np.int32), "language": np.array([LID[lang]], np.int32), "textnorm": np.array([14], np.int32)})
    a = np.asarray(o[0]).astype(np.int64).reshape(-1)
    return bytes(int(b) & 0xFF for b in a).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


nw = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()
nc = lambda s: re.sub(r"\s+", "", re.sub(r"[^\w\s]", "", s.lower()))
res = {"orig": [], "static": []}
os.makedirs(os.path.join(HERE, "wav_cpu"), exist_ok=True)
import soundfile as sf
for i, (lang, text) in enumerate(SENT):
    np.random.seed(i)
    wo, dur = m([text], style, total_step=5, speed=1.05, lang=lang)
    wo = wo.reshape(-1)[: int(dur[0] * 44100)]
    ws = static_chain(text, lang, i)
    sf.write(os.path.join(HERE, "wav_cpu", f"{i:02d}_{lang}_orig.wav"), wo, 44100); sf.write(os.path.join(HERE, "wav_cpu", f"{i:02d}_{lang}_static.wav"), ws, 44100)
    for tag, w in (("orig", wo), ("static", ws)):
        h = asr_text(w, lang)
        res[tag].append({"lang": lang, "ref": text, "hyp": h, "wer": jiwer.wer(nw(text), nw(h)) if nw(h) else 1.0, "cer": jiwer.cer(nc(text), nc(h)) if nc(h) else 1.0, "sec": len(w) / 44100})
    print(f"[{i}] {lang} dur orig {res['orig'][-1]['sec']:.2f}s static {res['static'][-1]['sec']:.2f}s | orig WER {res['orig'][-1]['wer']:.2f} | static WER {res['static'][-1]['wer']:.2f}\n     static hyp: {res['static'][-1]['hyp']}", flush=True)
for tag in res:
    for lang in ("en", "ko"):
        xs = [r for r in res[tag] if r["lang"] == lang]
        print(f"{tag:7s} {lang}: mean WER {np.mean([r['wer'] for r in xs])*100:.1f}%  CER {np.mean([r['cer'] for r in xs])*100:.1f}%  (n={len(xs)})")
json.dump(res, open(os.path.join(HERE, f"roundtrip_cpu_N{N}_L{L}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

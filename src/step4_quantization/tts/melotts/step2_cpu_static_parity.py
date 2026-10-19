# -*- coding: utf-8 -*-
"""Static graph vs original infer() on CPU (sdp_ratio=0, same constant noise). Then ASR round trip of the static audio."""
import sys, os, re, json, numpy as np, torch, onnxruntime as ort, jiwer, soundfile as sf
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
from melo.api import TTS
from melo import utils
from melo_static import MeloStatic
N, Y = 96, 384
tts = TTS(language="ZH", device="cpu"); hps = tts.hps
S = MeloStatic(tts.model, N, Y, noise_scale=0.6).eval()
SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
nc = lambda s: re.sub(r"[^\w]", "", s)
cos = lambda a, b: float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
cers = []; os.makedirs("work/wav_static_cpu", exist_ok=True)
for i, t in enumerate(SENT):
    bert, ja, ph, tone, lang = utils.get_text_for_tts_infer(t, "ZH", hps, "cpu", tts.symbol_to_id); n = len(ph)
    pad = lambda v: torch.cat([v, torch.zeros(N - n, dtype=v.dtype)])[None]
    bp = torch.zeros(1, 1024, N); bp[0, :, :n] = bert
    with torch.no_grad():
        o, yl = S(pad(ph).int(), pad(tone).int(), pad(lang).int(), bp, torch.tensor([n], dtype=torch.int32))
        # reference: original infer with the SAME constant noise injected through noise_scale=0 comparison is not possible; compare with noise 0 on both
        S.noise_scale = 0.0; o0, _ = S(pad(ph).int(), pad(tone).int(), pad(lang).int(), bp, torch.tensor([n], dtype=torch.int32)); S.noise_scale = 0.6
        ref = tts.model.infer(ph[None], torch.LongTensor([n]), torch.LongTensor([1]), tone[None], lang[None], bert[None], ja[None], sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.8, length_scale=1.0)[0][0, 0].numpy()
    L = int(yl[0]) * 512; a = o0[0, :L].numpy(); b = ref[:min(L, len(ref))]
    c = cos(a[:len(b)], b)
    x = resample_poly(o[0, :L].numpy().astype(np.float32), 16000, 44100).astype(np.float32); m_ = min(len(x), 464000); buf = np.zeros(464000, np.float32); buf[:m_] = x[:m_]
    out = asr.run(None, {"wav": buf.reshape(1, -1), "wav_len": np.array([m_], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    h = bytes(int(v) & 0xFF for v in np.asarray(out[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
    cer = jiwer.cer(nc(t), nc(h)) if nc(h) else 1.0; cers.append(cer); sf.write(f"work/wav_static_cpu/{i}.wav", o[0, :L].numpy(), 44100)
    print(f"[{i}] N={n} frames {int(yl[0])} (ref {len(ref)//512}) | noise-0 audio cos vs original {c:.5f} | ASR CER {cer:.2f} | {h}", flush=True)
print("static CPU mean CER %.3f" % np.mean(cers))

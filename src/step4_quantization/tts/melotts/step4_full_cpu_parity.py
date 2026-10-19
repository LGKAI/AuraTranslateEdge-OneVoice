# -*- coding: utf-8 -*-
"""CPU: BERT static + expansion + VITS static (windowed) vs the host-BERT pipeline; ASR CER of the full static pipeline."""
import sys, os, re, json, numpy as np, torch, onnxruntime as ort, jiwer
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
from transformers import AutoTokenizer, AutoModelForMaskedLM
from melo.api import TTS
from melo import utils
from melo_static import MeloStatic
from melo_full import BertStatic, MeloFull, host_front
N, Y, L = 96, 384, 32
tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
name = "hfl/chinese-roberta-wwm-ext-large"; tok = AutoTokenizer.from_pretrained(name); mlm = AutoModelForMaskedLM.from_pretrained(name).eval()
full = MeloFull(BertStatic(mlm, L), MeloStatic(tts.model, N, Y, group=5)).eval()
SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
nc = lambda s: re.sub(r"[^\w]", "", s); cers = []
cos = lambda a, b: float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
for i, t in enumerate(SENT):
    bi, bm, E, x, tone, lang, xl = host_front(tts, tok, t, N, L)
    with torch.no_grad():
        o, yl = full(torch.from_numpy(bi), torch.from_numpy(bm), torch.from_numpy(E), torch.from_numpy(x), torch.from_numpy(tone), torch.from_numpy(lang), torch.from_numpy(xl))
        bert_ref, ja, ph, tn, lg = utils.get_text_for_tts_infer(t, "ZH", tts.hps, "cpu", tts.symbol_to_id); n = int(xl[0])
        h = full.b(torch.from_numpy(bi), torch.from_numpy(bm)); bp = torch.matmul(h[0].t(), torch.from_numpy(E))[:, :n]
    print(f"[{i}] BERT features (static NPU-style vs host HF): cos {cos(bp.numpy().ravel(), bert_ref.numpy().ravel()):.6f}", end=" | ")
    Ln = int(yl[0]) * 512; w = resample_poly(o[0, :Ln].numpy().astype(np.float32), 16000, 44100).astype(np.float32); m_ = min(len(w), 464000); buf = np.zeros(464000, np.float32); buf[:m_] = w[:m_]
    out = asr.run(None, {"wav": buf.reshape(1, -1), "wav_len": np.array([m_], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    hyp = bytes(int(v) & 0xFF for v in np.asarray(out[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
    cer = jiwer.cer(nc(t), nc(hyp)) if nc(hyp) else 1.0; cers.append(cer); print(f"CER {cer:.2f} | {hyp}", flush=True)
print("full static pipeline (CPU) mean CER %.3f" % np.mean(cers))

# -*- coding: utf-8 -*-
"""ASR CER: VITS+BERT static pipeline fed by (a) host MeloTTS front vs (b) static-table NPU front, on FLEURS-test clauses."""
import sys, os, re, json, unicodedata, numpy as np, torch, onnxruntime as ort, jiwer
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from scipy.signal import resample_poly
from transformers import AutoTokenizer, AutoModelForMaskedLM
from melo.api import TTS
from huggingface_hub import hf_hub_download
from melo_static import MeloStatic
from melo_full import BertStatic, MeloFull, host_front
from melo_front import MeloFront, load_tables, LB, N, L
Y = 384; tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
name = "hfl/chinese-roberta-wwm-ext-large"; tok = AutoTokenizer.from_pretrained(name); mlm = AutoModelForMaskedLM.from_pretrained(name).eval()
full = MeloFull(BertStatic(mlm, L), MeloStatic(tts.model, N, Y, group=5)).eval(); T = load_tables(); F = MeloFront(T).eval()
asr = ort.InferenceSession(os.environ.get("SV_ONNX", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice\outputs\sensevoice-e2e-onnx\model_sv_combined_single_inline.onnx"), providers=["CPUExecutionProvider"])
nc = lambda s: re.sub(r"[^\w]", "", s)
def hear(o, yl):
    Ln = int(yl[0]) * 512; w = resample_poly(o[0, :Ln].numpy().astype(np.float32), 16000, 44100).astype(np.float32); m_ = min(len(w), 464000); buf = np.zeros(464000, np.float32); buf[:m_] = w[:m_]
    r = asr.run(None, {"wav": buf.reshape(1, -1), "wav_len": np.array([m_], np.int32), "language": np.array([3], np.int32), "textnorm": np.array([14], np.int32)})
    return bytes(int(v) & 0xFF for v in np.asarray(r[0]).astype(np.int64).reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
sents = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))] if os.environ.get("SAME40") else []
for l in open(hf_hub_download("google/fleurs", "data/cmn_hans_cn/test.tsv", repo_type="dataset"), encoding="utf-8"):
    f = l.rstrip("\n").split("\t")
    if len(f) > 3:
        for c in re.split(r"[。！？；]", f[2]):
            c = c.strip()
            if 6 <= len(c) <= 20 and re.fullmatch(r"[一-鿿，、]+", c) and c not in sents: sents.append(c)
sents = list(dict.fromkeys(sents))[:int(os.environ.get("NS", 40))]
A, B = [], []
for i, t in enumerate(sents):
    bi, bm, E, x, tone, lang, xl = host_front(tts, tok, t, N, L)
    b = np.zeros((1, LB), np.int32); by = unicodedata.normalize("NFC", t).encode("utf-8"); b[0, :len(by)] = list(by)
    with torch.no_grad():
        gx, gt, gl, gxl, gE, *_ = F(torch.from_numpy(b))
        oa, ya = full(*[torch.from_numpy(v) for v in (bi, bm, E, x, tone, lang, xl)])
        ob, yb = full(torch.from_numpy(bi), torch.from_numpy(bm), gE, gx, gt, gl, gxl)
    ha, hb = hear(oa, ya), hear(ob, yb); ca = jiwer.cer(nc(t), nc(ha)) if nc(ha) else 1.0; cb = jiwer.cer(nc(t), nc(hb)) if nc(hb) else 1.0
    A.append(ca); B.append(cb); print(f"[{i}] host {ca:.2f} static {cb:.2f} | {t} | {hb}", flush=True)
print(f"MEAN CER host-front {np.mean(A):.3f} | static-table front {np.mean(B):.3f} | n={len(A)}")

# -*- coding: utf-8 -*-
"""Dung chung cho eval30: nap manifest, decode byte_stream, cham diem (WER/CER)."""
import os, re, json
import numpy as np, soundfile as sf, jiwer

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
MANIFEST = os.path.join(ROOT, "data", "manifest_eval30.json")
MAX_WAV_SAMPLES = 464000
LID = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14


def load_items():
    return json.load(open(MANIFEST, encoding="utf-8"))


def load_wav_padded(item):
    wav, sr = sf.read(os.path.join(ROOT, item["path"].replace("/", os.sep)), dtype="float32")
    assert sr == 16000
    n = min(len(wav), MAX_WAV_SAMPLES)
    out = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)
    out[:n] = wav[:n]
    return out.reshape(1, MAX_WAV_SAMPLES), np.array([n], dtype=np.int32)


def decode_bytes(arr):
    a = np.asarray(arr).astype(np.int64).reshape(-1)
    return bytes(int(b) & 0xFF for b in a).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


def _norm_words(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()


def _norm_chars(s):
    return re.sub(r"\s+", "", re.sub(r"[^\w\s]", "", s.lower()))


def score(lang, ref, hyp):
    """Tra ve dict: wer (en/ko), cer (zh/ko). Dung chuan hoa chung cho ref va hyp."""
    r_c, h_c = _norm_chars(ref), _norm_chars(hyp)
    cer = jiwer.cer(r_c, h_c) if r_c else 1.0
    r_w, h_w = _norm_words(ref), _norm_words(hyp)
    wer = jiwer.wer(r_w, h_w) if (r_w and h_w) else 1.0
    return {"wer": wer, "cer": cer}


def summarize(items, hyps):
    """Gom theo ngon ngu: metric chinh (en=WER, zh=CER, ko=WER va CER)."""
    per = []
    for it, h in zip(items, hyps):
        s = score(it["lang"], it["transcript"], h)
        per.append({"lang": it["lang"], "audio": os.path.basename(it["path"]), "dur": it["duration_s"],
                    "reference": it["transcript"], "hypothesis": h, **s})
    summ = {}
    for L in ("en", "zh", "ko"):
        xs = [p for p in per if p["lang"] == L]
        if xs:
            summ[L] = {"n": len(xs), "wer": float(np.mean([p["wer"] for p in xs]) * 100),
                       "cer": float(np.mean([p["cer"] for p in xs]) * 100)}
    return per, summ


def print_summary(title, summ):
    print(f"\n=== {title} ===", flush=True)
    print(f"  en: n={summ['en']['n']}  WER={summ['en']['wer']:.2f}%", flush=True)
    print(f"  zh: n={summ['zh']['n']}  CER={summ['zh']['cer']:.2f}%", flush=True)
    print(f"  ko: n={summ['ko']['n']}  WER={summ['ko']['wer']:.2f}%  CER={summ['ko']['cer']:.2f}%", flush=True)

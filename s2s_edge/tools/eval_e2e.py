# -*- coding: utf-8 -*-
"""End-to-end speech-to-speech evaluation (the judges' view): input audio -> pipeline -> OUTPUT AUDIO -> ASR of the output in the target language
 -> compared with the reference translation (FLORES-200 devtest). Reports ASR-chrF / ASR-BLEU, source-ASR error rate, Piper out-of-vocabulary counts.
  python tools/eval_e2e.py --models models.dev.json --backend cpu --eval ../eval --dirs vi-en,en-vi --n 12 --out logs/e2e.json
The output-side ASR is a different model family from the one that produced the transcript (Zipformer for vi output, SenseVoice for en/zh/ko output)."""
import argparse, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jiwer, sacrebleu
from scipy.signal import resample_poly
from math import gcd
from s2s_edge.registry import Registry
from s2s_edge.pipeline import Translator
from s2s_edge.audio import load_audio

ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--backend", default="cpu"); ap.add_argument("--eval", required=True)
ap.add_argument("--dirs", default="vi-en,vi-zh,vi-ko,en-vi,zh-vi,ko-vi"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--out", default="logs/e2e.json"); ap.add_argument("--glossary", default=None)
a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); reg = Registry(a.models, a.backend, a.glossary); res = {}
norm_words = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip(); norm_chars = lambda s: re.sub(r"[\W_]+", "", s.lower())
for d in a.dirs.split(","):
    src, tgt = d.split("-"); tr = Translator(reg, src, tgt); out_asr = reg.asr(tgt); man = json.load(open(os.path.join(a.eval, f"manifest_{src}.json"), encoding="utf-8"))[: a.n]
    hyps, refs, src_err, rows, t0 = [], [], [], [], time.time()
    for it in man:
        w = load_audio(os.path.join(a.eval, it["path"])); r = tr.translate(w)
        text_src = " ".join(x["asr"] for x in r); text_mt = " ".join(x["mt"] for x in r) if tgt != "zh" else "".join(x["mt"] for x in r)
        wav_out = np.concatenate([x["wav"] for x in r]) if r else np.zeros(1600, np.float32); sr = r[0]["sr"] if r else 16000
        g = gcd(16000, int(sr)); w16 = resample_poly(wav_out, 16000 // g, int(sr) // g).astype(np.float32)
        heard = " ".join(out_asr(w16[i: i + int(out_asr.max_s * 16000)]) for i in range(0, max(len(w16), 1), int(out_asr.max_s * 16000)))
        ref = it["refs"][tgt]; hyps.append(heard); refs.append(ref)
        ref_src = it["refs"][src]; src_err.append(jiwer.cer(norm_chars(ref_src), norm_chars(text_src)) if src in ("zh", "ko") else jiwer.wer(norm_words(ref_src), norm_words(text_src)))
        rows.append(dict(src_asr=text_src, mt=text_mt, heard=heard, ref=ref, oov=getattr(tr.tts, "oov", 0)))
        print(f"[{d}] mt: {text_mt[:70]} | heard: {heard[:70]}", flush=True)
    tok = {"zh": "zh", "ko": "ko-mecab"}.get(tgt, "13a")
    res[d] = dict(n=len(hyps), asr_chrF=round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2), asr_BLEU=round(sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok if tgt != "ko" else "13a").score, 2),
                  src_asr_err=round(float(np.mean(src_err)), 3), piper_oov_total=getattr(tr.tts, "oov", None), seconds=round(time.time() - t0), rows=rows)
    print(d, {k: v for k, v in res[d].items() if k != "rows"}, flush=True)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

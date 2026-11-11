# -*- coding: utf-8 -*-
"""Evaluate the pipeline on the organisers' sample set (Drive folder "Vie-Eng / Vie-Kor / Vie-Man dataset", 100-180 clips per pair).
For every clip: source audio -> pipeline -> output audio, then
  * src_err : ASR error of the transcript vs the source text (WER for vi/en, CER for zh/ko)
  * mt_gold : translation of the GOLD source text (isolates MT from ASR errors)
  * mt_e2e  : translation actually produced from the ASR transcript
  * heard   : ASR of the OUTPUT audio in the target language (judges' view: ASR-chrF / ASR-BLEU)
  * timing  : per-stage wall time (on the device with --backend npu this is the number to report)
References are normalised the way the EV set is (lowercase, no punctuation, numbers as words) before scoring.
Rows whose reference does not match the source (extreme length ratio, runaway repetition) are flagged; scores are given on all rows and on the clean subset.
  python tools/eval_btc.py --data /path/onevoice_btc_data --models models/models.json --backend npu --dirs vi-en,en-vi --out logs/btc.json"""
import argparse, csv, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jiwer, sacrebleu
from math import gcd
from scipy.signal import resample_poly
from s2s_edge.registry import Registry
from s2s_edge.pipeline import Translator
from s2s_edge.audio import load_audio
from s2s_edge.textnorm import normalize


def load_audio_any(path):
    """The Korean clips are MP3 data in .wav files; libsndfile >= 1.1 reads MP3, otherwise fall back to ffmpeg."""
    try:
        return load_audio(path)
    except Exception:
        import subprocess
        raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def xlsx_rows(path):
    import openpyxl
    rows = list(openpyxl.load_workbook(path, read_only=True).active.values)
    return [r for r in rows[1:] if r and r[0]]


def manifest(root, d):
    """-> list of dict(id, wav, src_text, ref)"""
    src, tgt = d.split("-"); j = lambda *p: os.path.join(root, *p); out = []
    if d in ("en-vi", "vi-en"):
        sub, x = ("Eng-Vie", "OneVoice_E_to_V_samples_100.xlsx") if d == "en-vi" else ("Vie-Eng", "OneVoice_V_to_E_samples_100.xlsx")
        for r in xlsx_rows(j("Vie-Eng dataset", sub, x)):
            out.append(dict(id=r[0], wav=j("Vie-Eng dataset", sub, "wav", f"{r[0]}.wav"), src_text=str(r[1]), ref=str(r[2])))
    else:
        other = "ko" if "ko" in d else "zh"; folder, x = ("Vie-Kor dataset", "OneVoice_V_to_K_samples_180.xlsx") if other == "ko" else ("Vie-Man dataset", "OneVoice_V_to_M_samples_180.xlsx")
        for r in xlsx_rows(j(folder, x)):                      # id, vie_text, other_text, vie_voice_path, other_voice_path
            texts = {"vi": str(r[1]), other: str(r[2])}; wavs = {"vi": r[3], other: r[4]}
            out.append(dict(id=r[0], wav=j(folder, wavs[src]), src_text=texts[src], ref=texts[tgt]))
    return [m for m in out if os.path.exists(m["wav"])]


def norm_text(s, lang):
    s = normalize(s, lang) if lang in ("vi", "en", "zh", "ko") else s
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def err_rate(ref, hyp, lang):
    r, h = norm_text(ref, lang), norm_text(hyp, lang)
    if not r: return None
    if lang in ("zh", "ko"): return jiwer.cer(r.replace(" ", ""), h.replace(" ", "") or "_")
    return jiwer.wer(r, h or "_")


def suspicious(src_text, ref):
    a, b = len(re.sub(r"\W", "", src_text)), len(re.sub(r"\W", "", ref))
    toks = ref.split(); rep = len(toks) >= 9 and len(set(toks)) / len(toks) < 0.35
    return rep or a == 0 or b == 0 or not (0.2 <= b / a <= 5.0)


def scores(hyps, refs, lang):
    tok = {"zh": "zh"}.get(lang, "13a"); H = [norm_text(h, lang) for h in hyps]; R = [norm_text(r, lang) for r in refs]
    if lang == "ko": H, R = [" ".join(h.replace(" ", "")) for h in H], [" ".join(r.replace(" ", "")) for r in R]     # character level for Korean
    return dict(chrF=round(sacrebleu.corpus_chrf(H, [R]).score, 2), BLEU=round(sacrebleu.corpus_bleu(H, [R], tokenize=tok).score, 2))


def corpus_err(rows, lang):
    """Corpus-level WER / CER (total edits / total reference length): one mislabelled clip cannot dominate it the way a mean of per-clip rates does."""
    R = [norm_text(x["src_text"], lang) for x in rows]; H = [norm_text(x["asr"], lang) for x in rows]
    keep = [i for i, r in enumerate(R) if r]
    if not keep: return None
    R, H = [R[i] for i in keep], [H[i] or "_" for i in keep]
    if lang in ("zh", "ko"): return jiwer.cer([r.replace(" ", "") for r in R], [h.replace(" ", "") for h in H])
    return jiwer.wer(R, H)


def flagged(x):
    """Reference that does not fit the clip: translation pair mismatch (see suspicious) or a source label far shorter/longer than what is spoken."""
    return x["flag"] or (x["src_err"] is not None and x["src_err"] > 1.0)


def summarize(rows, src, tgt, heard=True):
    summ = dict(n=len(rows))
    for name, sub in (("all", rows), ("clean", [x for x in rows if not flagged(x)])):
        if not sub: continue
        refs = [x["ref"] for x in sub]; e = corpus_err(sub, src)
        s = dict(n=len(sub), src_err=round(e, 4) if e is not None else None, mt_e2e=scores([x["mt_e2e"] for x in sub], refs, tgt))
        if any(x["mt_gold"] for x in sub): s["mt_gold"] = scores([x["mt_gold"] for x in sub], refs, tgt)     # absent with --no_gold
        if heard and any(x["heard"] for x in sub): s["heard"] = scores([x["heard"] for x in sub], refs, tgt)
        tot = [x["total_s"] for x in sub if "total_s" in x]; aud = [x["audio_s"] for x in sub]
        if tot: s["latency"] = dict(mean_s=round(float(np.mean(tot)), 3), p50_s=round(float(np.median(tot)), 3), p90_s=round(float(np.percentile(tot, 90)), 3),
                                    rtf=round(float(np.sum(tot) / max(np.sum(aud), 1e-6)), 3), **{k: round(float(np.mean([x[k] for x in sub if k in x])), 3) for k in ("asr_s", "mt_s", "tts_s", "graph_s")})
        summ[name] = s
    return summ


ap = argparse.ArgumentParser(); ap.add_argument("--data", default=""); ap.add_argument("--models", default=""); ap.add_argument("--backend", default="npu")
ap.add_argument("--dirs", default="vi-en,en-vi,vi-ko,ko-vi,vi-zh,zh-vi"); ap.add_argument("--n", type=int, default=0); ap.add_argument("--out", default="logs/btc.json")
ap.add_argument("--glossary", default=None); ap.add_argument("--no_heard", action="store_true", help="skip ASR of the output audio (faster)")
ap.add_argument("--split_file", default="", help="JSON from tools/btc_split.py"); ap.add_argument("--split", default="", help="dev | holdout | quick (needs --split_file)")
ap.add_argument("--no_gold", action="store_true", help="skip translating the gold source text (saves one MT run per clip; mt_gold is then left out)")
ap.add_argument("--summarize", nargs="+", default=None, help="recompute summaries from saved result JSON files (no models needed) and print one table")
a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
if a.summarize:
    print("| direction | n (clean) | src ASR err | MT gold chrF/BLEU | MT from ASR chrF/BLEU | heard chrF/BLEU |"); print("|---|---|---|---|---|---|")
    for p in a.summarize:
        for d, v in json.load(open(p, encoding="utf-8")).items():
            src, tgt = d.split("-"); s = summarize(v["rows"], src, tgt); c = s.get("clean", s["all"]); f = lambda m: f"{c[m]['chrF']} / {c[m]['BLEU']}" if m in c else "-"
            print(f"| {d} | {s['n']} ({c['n']}) | {'CER' if src in ('zh', 'ko') else 'WER'} {100 * c['src_err']:.1f} % | {f('mt_gold')} | {f('mt_e2e')} | {f('heard')} |")
    sys.exit(0)
if not (a.data and a.models): ap.error("--data and --models are required unless --summarize is given")
reg = Registry(a.models, a.backend, a.glossary); res = {}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
for d in a.dirs.split(","):
    src, tgt = d.split("-"); tr = Translator(reg, src, tgt); out_asr = None if a.no_heard else reg.asr(tgt)
    man = manifest(a.data, d)
    if a.split:
        keep = set(json.load(open(a.split_file, encoding="utf-8"))["by_dir"][d][a.split]); man = [m for m in man if str(m["id"]) in keep]
    man = man[: a.n] if a.n else man; rows = []; t0 = time.time()
    for it in man:
        w = load_audio_any(it["wav"]); r = tr.translate(w)
        asr = " ".join(x["asr"] for x in r).strip(); j = "" if tgt == "zh" else " "; mt = j.join(x["mt"] for x in r).strip()
        mt_gold = tr.mt(it["src_text"], src, tgt) if it["src_text"].strip() and not a.no_gold else ""
        tm = {k: round(sum(x["timing"][k] for x in r), 4) for k in ("asr_s", "mt_s", "tts_s", "total_s", "graph_s")} if r else {}
        heard = ""
        if out_asr is not None and r:
            wav = np.concatenate([x["wav"] for x in r]); sr = int(r[0]["sr"]); g = gcd(16000, sr); w16 = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
            step = int(out_asr.max_s * 16000); heard = j.join(out_asr(w16[i: i + step]) for i in range(0, max(len(w16), 1), step)).strip()
        rows.append(dict(id=it["id"], audio_s=round(len(w) / 16000, 2), src_text=it["src_text"], ref=it["ref"], asr=asr, src_err=err_rate(it["src_text"], asr, src),
                         mt_gold=mt_gold, mt_e2e=mt, heard=heard, flag=suspicious(it["src_text"], it["ref"]), **tm))
        print(f"[{d}] {it['id']} err={rows[-1]['src_err'] if rows[-1]['src_err'] is None else round(rows[-1]['src_err'], 2)} | {mt[:60]} | {tm.get('total_s')}", flush=True)
    summ = dict(summarize(rows, src, tgt, out_asr is not None), seconds=round(time.time() - t0))
    res[d] = dict(summary=summ, rows=rows); print(d, json.dumps(summ, ensure_ascii=False), flush=True)
    with open(os.path.splitext(a.out)[0] + f"_{d}.csv", "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"]); wr.writeheader(); wr.writerows(rows)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

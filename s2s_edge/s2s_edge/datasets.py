# -*- coding: utf-8 -*-
"""Readers and scorers for the two organiser data sets (shared by the web UI; the CLI scripts tools/eval_btc.py / eval_adv.py use the same layout).
  btc : "Vie-Eng / Vie-Kor / Vie-Man dataset" with xlsx sheets: source audio, source text, reference translation (100 / 180 clips per direction)
  adv : same folder names, *_ADV_* files: 25 sentences x 4 noise levels per direction, transcript of each clip only (NO reference translation)"""
import csv, glob, os, re

DIRS = ["vi-en", "en-vi", "vi-ko", "ko-vi", "vi-zh", "zh-vi"]
CONDS = ["clean", "factory_15db", "construction_10db", "heavy_5db"]
ADV_LAYOUT = {"en-vi": ("Vie-Eng dataset", "Eng-Vie/wav", "Eng-Vie", "csv"), "vi-en": ("Vie-Eng dataset", "Vie-Eng/wav", "Vie-Eng", "csv"),
              "vi-ko": ("Vie-Kor dataset", "Vie_voice", "Vie_text", "txt"), "ko-vi": ("Vie-Kor dataset", "Kor_voice", "Kor_text", "txt"),
              "vi-zh": ("Vie-Man dataset", "Vie_voice", "Vie_text", "txt"), "zh-vi": ("Vie-Man dataset", "Man_voice", "Man_text", "txt")}


def find_sets(data_root):
    """-> {"btc": folder, "adv": folder} for the folders present under data_root."""
    out = {}
    for name, cands in (("btc", ["onevoice_btc_data"]), ("adv", ["onevoice_btc_adv"])):
        for c in cands:
            p = os.path.join(data_root, c)
            if os.path.isdir(p): out[name] = p
    return out


def _xlsx_rows(path):
    import openpyxl
    rows = list(openpyxl.load_workbook(path, read_only=True).active.values)
    return [r for r in rows[1:] if r and r[0]]


def btc_items(root, d):
    src, tgt = d.split("-"); j = lambda *p: os.path.join(root, *p); out = []
    if d in ("en-vi", "vi-en"):
        sub, x = ("Eng-Vie", "OneVoice_E_to_V_samples_100.xlsx") if d == "en-vi" else ("Vie-Eng", "OneVoice_V_to_E_samples_100.xlsx")
        for r in _xlsx_rows(j("Vie-Eng dataset", sub, x)):
            out.append(dict(id=str(r[0]), wav=j("Vie-Eng dataset", sub, "wav", f"{r[0]}.wav"), text=str(r[1]), ref=str(r[2]), cond=""))
    else:
        other = "ko" if "ko" in d else "zh"
        folder, x = ("Vie-Kor dataset", "OneVoice_V_to_K_samples_180.xlsx") if other == "ko" else ("Vie-Man dataset", "OneVoice_V_to_M_samples_180.xlsx")
        for r in _xlsx_rows(j(folder, x)):                       # id, vie_text, other_text, vie_voice_path, other_voice_path
            texts = {"vi": str(r[1]), other: str(r[2])}; wavs = {"vi": r[3], other: r[4]}
            out.append(dict(id=str(r[0]), wav=j(folder, wavs[src]), text=texts[src], ref=texts[tgt], cond=""))
    return [m for m in out if os.path.exists(m["wav"])]


def adv_items(root, d):
    folder, wdir, tdir, kind = ADV_LAYOUT[d]; base = os.path.join(root, folder); texts = {}
    if kind == "csv":
        for p in glob.glob(os.path.join(base, tdir, "*.csv")):
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                texts[r["id"]] = next(v for k, v in r.items() if k.startswith("transcript"))
    else:
        for f in glob.glob(os.path.join(base, tdir, "*.txt")):
            texts[os.path.splitext(os.path.basename(f))[0]] = open(f, encoding="utf-8-sig").read().strip()
    out = []
    for w in sorted(glob.glob(os.path.join(base, wdir, "*.wav"))):
        cid = os.path.splitext(os.path.basename(w))[0]; m = re.match(r"(.+_ADV_\d+)_(.+)$", cid)
        if not m or m.group(2) not in CONDS: continue
        txt = texts.get(cid) or texts.get(f"{m.group(1)}_clean") or texts.get(m.group(1))
        if txt: out.append(dict(id=cid, wav=w, text=txt, ref="", cond=m.group(2)))
    return out


def items(set_name, root, d):
    return btc_items(root, d) if set_name == "btc" else adv_items(root, d)


# ---------------------------------------------------------------- scoring (same normalisation as tools/eval_btc.py)
def norm_text(s, lang):
    from .textnorm import normalize
    s = re.sub(r"[^\w\s]", " ", normalize(s or "", lang).lower()); return re.sub(r"\s+", " ", s).strip()


def asr_err(refs, hyps, lang):
    """Corpus WER (vi, en) / CER (zh, ko): total edits / total reference length."""
    import jiwer
    R = [norm_text(r, lang) for r in refs]; H = [norm_text(h, lang) or "_" for h in hyps]; keep = [i for i, r in enumerate(R) if r]
    if not keep: return None
    R, H = [R[i] for i in keep], [H[i] for i in keep]
    if lang in ("zh", "ko"): return jiwer.cer([r.replace(" ", "") for r in R], [h.replace(" ", "") for h in H])
    return jiwer.wer(R, H)


def _prep(h, r, lang):
    H, R = [norm_text(x, lang) for x in h], [norm_text(x, lang) for x in r]
    if lang == "ko": H, R = [" ".join(x.replace(" ", "")) for x in H], [" ".join(x.replace(" ", "")) for x in R]
    return H, R


def chrf(hyps, refs, lang):
    import sacrebleu
    H, R = _prep(hyps, refs, lang); return sacrebleu.corpus_chrf(H, [R]).score


def sent_chrf(hyp, ref, lang):
    import sacrebleu
    H, R = _prep([hyp], [ref], lang); return sacrebleu.sentence_chrf(H[0], [R[0]]).score

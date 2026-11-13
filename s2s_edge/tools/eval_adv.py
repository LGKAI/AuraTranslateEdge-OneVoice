# -*- coding: utf-8 -*-
"""Noise-robustness test on the "ADV" set (Drive: Vie-Eng / Vie-Kor / Vie-Man dataset, *_ADV_*): 25 sentences per direction, each recorded
4 times: clean, factory 15 dB, construction 10 dB, heavy 5 dB.
This set has NO translation references (in Vie-Kor / Vie-Man the Korean / Chinese text and the Vietnamese text with the same id are
different sentences: each is only the transcript of its own audio). So it measures what noise does to the pipeline:
  * asr_err     : WER (vi, en) / CER (ko, zh) of the transcript vs the gold transcript
  * consistency : chrF of the translation produced from noisy audio vs the translation of the GOLD transcript (pseudo-reference).
                  100 = noise changed nothing; it does NOT say whether the translation itself is correct (use tools/eval_btc.py for that)
  * latency     : per-stage time (on the box with --backend npu)
  python tools/eval_adv.py --data /root/onevoice/onevoice_btc_adv --models models_qcs8550/models.json --backend npu --out logs/adv.json [--dirs vi-ko,ko-vi] [--n 0]
  python tools/eval_adv.py --summarize logs/adv.json"""
import argparse, csv, glob, json, os, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONDS = ["clean", "factory_15db", "construction_10db", "heavy_5db"]
# direction -> (folder, wav subfolder, text subfolder or CSV, id prefix)
LAYOUT = {"en-vi": ("Vie-Eng dataset", "Eng-Vie/wav", "Eng-Vie/OneVoice_E_to_V_advanced.csv", "EV"),
          "vi-en": ("Vie-Eng dataset", "Vie-Eng/wav", "Vie-Eng/OneVoice_V_to_E_advanced.csv", "VE"),
          "vi-ko": ("Vie-Kor dataset", "Vie_voice", "Vie_text", "VK"), "ko-vi": ("Vie-Kor dataset", "Kor_voice", "Kor_text", "VK"),
          "vi-zh": ("Vie-Man dataset", "Vie_voice", "Vie_text", "VM"), "zh-vi": ("Vie-Man dataset", "Man_voice", "Man_text", "VM")}


def manifest(root, d):
    folder, wdir, tsrc, pre = LAYOUT[d]; base = os.path.join(root, folder); out = []
    texts = {}
    if tsrc.endswith(".csv"):
        p = os.path.join(base, tsrc)
        if not os.path.exists(p): p = (glob.glob(os.path.join(base, os.path.dirname(tsrc), "*.csv")) or [p])[0]
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            texts[r["id"]] = next(v for k, v in r.items() if k.startswith("transcript"))
    else:
        for f in glob.glob(os.path.join(base, tsrc, "*.txt")): texts[os.path.splitext(os.path.basename(f))[0]] = open(f, encoding="utf-8-sig").read().strip()
    for w in sorted(glob.glob(os.path.join(base, wdir, "*.wav"))):
        cid = os.path.splitext(os.path.basename(w))[0]; m = re.match(r"(.+_ADV_\d+)_(.+)$", cid)
        if not m or m.group(2) not in CONDS: continue
        sid, cond = m.group(1), m.group(2)
        txt = texts.get(cid) or texts.get(f"{sid}_clean") or texts.get(sid)
        if txt: out.append(dict(id=cid, sid=sid, cond=cond, wav=w, text=txt))
    return out


def norm_text(s, lang):
    from s2s_edge.textnorm import normalize
    s = re.sub(r"[^\w\s]", " ", normalize(s, lang).lower()); return re.sub(r"\s+", " ", s).strip()


def asr_err(refs, hyps, lang):
    import jiwer
    R = [norm_text(r, lang) for r in refs]; H = [norm_text(h, lang) or "_" for h in hyps]
    if lang in ("zh", "ko"): return jiwer.cer([r.replace(" ", "") for r in R], [h.replace(" ", "") for h in H])
    return jiwer.wer(R, H)


def chrf(hyps, refs, lang):
    import sacrebleu
    H = [norm_text(h, lang) for h in hyps]; R = [norm_text(r, lang) for r in refs]
    if lang == "ko": H, R = [" ".join(h.replace(" ", "")) for h in H], [" ".join(r.replace(" ", "")) for r in R]
    return sacrebleu.corpus_chrf(H, [R]).score


def summarize(res):
    table = {}
    for d, rows in res.items():
        src, tgt = d.split("-"); table[d] = {}
        for c in CONDS:
            sub = [x for x in rows if x["cond"] == c]
            if not sub: continue
            table[d][c] = dict(n=len(sub), asr_err=round(100 * asr_err([x["text"] for x in sub], [x["asr"] for x in sub], src), 1),
                               consistency=round(chrf([x["mt"] for x in sub], [x["mt_gold"] for x in sub], tgt), 1),
                               total_s=round(float(np.mean([x["total_s"] for x in sub])), 3))
    return table


def print_table(t):
    print("| direction | metric | " + " | ".join(CONDS) + " |"); print("|---|---|" + "---|" * len(CONDS))
    for d, v in t.items():
        src = d.split("-")[0]
        print(f"| {d} | ASR {'CER' if src in ('zh', 'ko') else 'WER'} % | " + " | ".join(str(v.get(c, {}).get("asr_err", "-")) for c in CONDS) + " |")
        print(f"| {d} | consistency chrF | " + " | ".join(str(v.get(c, {}).get("consistency", "-")) for c in CONDS) + " |")
        print(f"| {d} | total s / clip | " + " | ".join(str(v.get(c, {}).get("total_s", "-")) for c in CONDS) + " |")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=""); ap.add_argument("--models", default=""); ap.add_argument("--backend", default="npu")
    ap.add_argument("--dirs", default="vi-en,en-vi,vi-ko,ko-vi,vi-zh,zh-vi"); ap.add_argument("--n", type=int, default=0, help="sentences per direction (0 = all 25)")
    ap.add_argument("--out", default="logs/adv.json"); ap.add_argument("--summarize", default="")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
    if a.summarize: print_table(summarize(json.load(open(a.summarize, encoding="utf-8")))); return
    if not (a.data and a.models): ap.error("--data and --models are required")
    from s2s_edge.registry import Registry
    from s2s_edge.pipeline import Translator
    from s2s_edge.audio import load_audio
    reg = Registry(a.models, a.backend); res = json.load(open(a.out, encoding="utf-8")) if os.path.exists(a.out) else {}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for d in a.dirs.split(","):
        if d in res and len(res[d]) and not a.n: print(d, "already done (delete", a.out, "to redo)"); continue
        src, tgt = d.split("-"); tr = Translator(reg, src, tgt); man = manifest(a.data, d)
        if a.n: keep = sorted({m["sid"] for m in man})[: a.n]; man = [m for m in man if m["sid"] in keep]
        print(f"=== {d}: {len(man)} clips", flush=True); gold = {}; rows = []
        for m in man:
            if m["sid"] not in gold: gold[m["sid"]] = tr.e.translate_text(m["text"], src, tgt)
            r = tr.translate(load_audio(m["wav"])); j = "" if tgt == "zh" else " "
            tm = {k: round(sum(x["timing"][k] for x in r), 4) for k in ("asr_s", "mt_s", "tts_s", "total_s", "graph_s")} if r else dict(asr_s=0, mt_s=0, tts_s=0, total_s=0, graph_s=0)
            rows.append(dict(m, asr=" ".join(x["asr"] for x in r).strip(), mt=j.join(x["mt"] for x in r).strip(), mt_gold=gold[m["sid"]], **tm))
            print(f"[{d}] {m['id']}: {rows[-1]['asr'][:50]} | {rows[-1]['mt'][:50]} | {tm['total_s']:.2f} s", flush=True)
        res[d] = rows; json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    t = summarize(res); json.dump(t, open(os.path.splitext(a.out)[0] + "_summary.json", "w"), indent=1); print_table(t)


if __name__ == "__main__":
    main()

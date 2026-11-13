# -*- coding: utf-8 -*-
"""Web UI served BY THE BOX: inference runs on the box (NPU); the laptop browser only displays it.
  box   : bash tools/box_serve.sh                (= python -m s2s_edge serve --data_root /root/onevoice)
  laptop: python tools/laptop_web.py             (= adb forward tcp:8770 tcp:8770 + opens http://127.0.0.1:8770)
Bound to 127.0.0.1 only: reachable through the USB cable with adb forward, no network.

Per clip: ASR (text, time, error vs the gold transcript) -> MT (text, time, chrF vs the reference when the set has one) -> TTS (audio, time)
-> "heard" = ASR of the TTS audio in the target language (checks that the speech says what MT wrote: tts_err = WER/CER(heard, MT);
heard_chrf = chrF(heard, reference), the judges' ASR-chrF view). Clips with a likely problem get issue tags.
Batches run in a thread ON THE BOX (closing the browser does not stop them); results: logs/webui/<run>.jsonl + logs/webui/<run>/*.wav,
and GET /api/export?run=.. packs everything (CSV, summary, issues, offline HTML report, output audio) into one zip.

API: GET /api/info | /api/items?set&dir | /api/audio?set&dir&id | /api/out?run&file | /api/summary?run | /api/rows?run&since | /api/batch/status
     /api/export?run | /health ;  POST /api/batch/start {sets, dirs, limit, cond} | /api/batch/stop | /api/run_item {set, dir, id, run}
     | /api/translate?pair=vi,ko or ?src&tgt (WAV body) | /api/text {text, src, tgt} | /translate (tools/laptop_ptt.py)"""
import base64, csv, html, io, json, os, platform, re, threading, time, wave, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import gcd
from urllib.parse import parse_qs, urlparse
import numpy as np
from . import datasets as ds
from .audio import load_audio, save_wav, concat_results
from .registry import DIRECTIONS

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui.html")
ISSUE_TEXT = {"asr_empty": "ASR không nghe ra gì", "asr_bad": "ASR sai nhiều (> 30 %)", "mt_empty": "MT ra rỗng", "mt_repeat": "MT lặp từ",
              "mt_length": "độ dài bản dịch bất thường", "mt_low": "chrF MT thấp (< 20)", "tts_empty": "TTS không ra âm thanh",
              "tts_bad": "TTS đọc khác chữ MT (ASR nghe lại sai > 30 %)", "slow": "chậm (lâu hơn độ dài câu nói, RTF > 1)", "error": "lỗi chạy",
              "mt_fallback": "MT LÙI VỀ 600M (3.3B không chạy được)",
              "asr_cleaned": "bộ lọc đã cắt rác ASR/MT (xem cột asr_raw, asr_cleaned)"}


def _wav_bytes(x, sr):
    b = io.BytesIO(); save_wav(b, x, sr); return b.getvalue()


def _to16k(x, sr):
    if sr == 16000: return np.asarray(x, np.float32)
    from scipy.signal import resample_poly
    g = gcd(16000, int(sr)); return resample_poly(x, 16000 // g, int(sr) // g).astype(np.float32)


def _read_wav(body):
    with wave.open(io.BytesIO(body)) as w:
        sr, ch, sw, raw = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    if sw != 2: raise ValueError("WAV must be 16-bit PCM")
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    if ch > 1: x = x.reshape(-1, ch).mean(1)
    return _to16k(x, sr)


def _units(s, lang):
    n = len(re.sub(r"[\W\d_]", "", s or "")); return n * (2.5 if lang == "zh" else 1.6 if lang == "ko" else 1.0)


def _repeats(s):
    t = (s or "").split()
    if len(t) < 8: t = list((s or "").replace(" ", ""))                     # zh / ko: characters
    if len(t) < 8: return False
    grams = [tuple(t[i:i + 3]) for i in range(len(t) - 2)]; return len(set(grams)) / len(grams) < 0.5


def issues_of(r):
    if r.get("error"): return ["error"]
    out = []; src, tgt = r["dir"].split("-")
    if not r["asr"]: out.append("asr_empty")
    elif r.get("asr_err") is not None and r["asr_err"] > 30: out.append("asr_bad")
    if r["asr"] and not r["mt"]: out.append("mt_empty")
    if r["mt"] and _repeats(r["mt"]): out.append("mt_repeat")
    if r["mt"] and r["asr"]:
        ratio = _units(r["mt"], tgt) / max(_units(r["asr"], src), 1)
        if not 0.35 <= ratio <= 2.8: out.append("mt_length")
    if r.get("chrf") is not None and r["chrf"] < 20: out.append("mt_low")
    if r["mt"] and not r.get("out_audio_s"): out.append("tts_empty")
    if r.get("tts_err") is not None and r["tts_err"] > 30: out.append("tts_bad")
    if r.get("audio_s") and r["total_s"] > r["audio_s"]: out.append("slow")
    if [x for x in r.get("asr_cleaned", []) if x != "mt_split"]: out.append("asr_cleaned")                 # mt_split (sentence-by-sentence MT) is normal, not an issue
    if r.get("mt_wanted") and r.get("mt_model") and r["mt_wanted"] != r["mt_model"]: out.append("mt_fallback")
    return out


def _device_info(reg, models_path):
    rd = lambda p: open(p).read().strip() if os.path.exists(p) else ""
    try:
        import qai_appbuilder; qv = getattr(qai_appbuilder, "__version__", "installed")
    except Exception as e: qv = f"not importable ({type(e).__name__})"
    osn = rd("/etc/os-release").split("PRETTY_NAME=")[-1].split("\n")[0].strip('"') if os.path.exists("/etc/os-release") else platform.platform()
    return dict(host=platform.node(), machine=platform.machine(), os=osn, soc_id=rd("/sys/devices/soc0/soc_id"), soc_machine=rd("/sys/devices/soc0/machine"),
                backend=reg.backend, models=os.path.abspath(models_path), perf=os.environ.get("S2S_PERF", "BURST (default)"), qai_appbuilder=qv,
                fastrpc=sorted(x for x in os.listdir("/dev") if "rpc" in x or "dsp" in x) if os.path.isdir("/dev") else [])


class WebApp:
    def __init__(self, reg, models_path, data_root, default, out_dir="logs/webui"):
        self.reg, self.models_path, self.data_root, self.default = reg, models_path, data_root, default
        self.sets = ds.find_sets(data_root) if data_root else {}; self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.lock = threading.Lock(); self._tr = {}; self._conv = {}; self._items = {}; self._asr = {}; self.runs = {}
        self.info = _device_info(reg, models_path); self.batch = dict(run=None, running=False, done=0, total=0, current="", started=0.0, stop=False, error="")
        for p in sorted(os.listdir(out_dir)):                                # runs of earlier sessions stay available for export
            if p.endswith(".jsonl") and p != "mic.jsonl":
                try:
                    rows = {}
                    for l in open(os.path.join(out_dir, p), encoding="utf-8"):
                        if l.strip(): r = json.loads(l); rows[(r["set"], r["dir"], r["id"])] = r          # last line of a re-run wins
                    self.runs[p[:-6]] = list(rows.values())
                except Exception: pass

    # ---- pipeline objects (graphs are cached by the registry: switching direction does not reload them)
    def translator(self, src, tgt):
        from .pipeline import Translator
        if (src, tgt) not in self._tr: self._tr[(src, tgt)] = Translator(self.reg, src, tgt)
        return self._tr[(src, tgt)]

    def conversation(self, pair):
        from .conversation import Conversation
        if pair not in self._conv: self._conv[pair] = Conversation(self.reg, pair=pair)
        return self._conv[pair]

    def asr(self, lang):
        if lang not in self._asr: self._asr[lang] = self.reg.asr(lang)
        return self._asr[lang]

    def items(self, set_name, d):
        k = (set_name, d)
        if k not in self._items: self._items[k] = {m["id"]: m for m in ds.items(set_name, self.sets[set_name], d)}
        return self._items[k]

    # ---- core: one utterance, then "heard"
    def _run(self, tr, wav):
        t0 = time.perf_counter(); res = tr.translate(wav); wall = time.perf_counter() - t0
        segs = [dict(src=r.get("src", getattr(tr, "src", None)), tgt=r.get("tgt", getattr(tr, "tgt", None)), asr=r["asr"], asr_raw=r.get("asr_raw", r["asr"]), asr_clean=list(r.get("asr_clean", [])), mt=r["mt"],
                     timing={k: round(float(v), 4) for k, v in r["timing"].items()}) for r in res]
        tm = {k: round(sum(s["timing"].get(k, 0.0) for s in segs), 4) for k in ("asr_s", "mt_s", "tts_s", "total_s", "graph_s")}; tm["wall_s"] = round(wall, 4)
        out, osr = concat_results(res) if res else (np.zeros(0, np.float32), 16000)
        return segs, tm, out, osr

    def heard(self, out, osr, lang):
        """ASR of the output speech in the target language (not part of the pipeline latency)."""
        if not len(out): return "", 0.0
        t0 = time.perf_counter(); a = self.asr(lang); w = _to16k(out, osr); step = int(a.max_s * 16000); j = "" if lang == "zh" else " "
        txt = j.join(a(w[i: i + step]) for i in range(0, len(w), step)).strip(); return txt, round(time.perf_counter() - t0, 4)

    def run_item(self, set_name, d, cid, run, with_heard=True):
        m = self.items(set_name, d)[cid]; src, tgt = d.split("-"); wav = load_audio(m["wav"])
        row = dict(run=run, set=set_name, dir=d, id=cid, cond=m["cond"], text=m["text"], ref=m["ref"], audio_s=round(len(wav) / 16000, 2))
        try:
            with self.lock:                                                   # one job at a time on the NPU
                segs, tm, out, osr = self._run(self.translator(src, tgt), wav)
                j = "" if tgt == "zh" else " "; mt = j.join(s["mt"] for s in segs).strip()
                hd, hs = self.heard(out, osr, tgt) if with_heard else ("", 0.0)
        except Exception as e:
            import traceback; traceback.print_exc()
            row.update(asr="", mt="", heard="", error=repr(e)[:300], asr_s=0, mt_s=0, tts_s=0, total_s=0, graph_s=0, wall_s=0, heard_s=0, out_audio_s=0, n_segments=0,
                       asr_err=None, chrf=None, tts_err=None, heard_chrf=None, file="")
            row["issues"] = issues_of(row); self._store(run, row); return row
        asr = " ".join(s["asr"] for s in segs).strip(); tr_ = self.translator(src, tgt); mo = getattr(tr_, "mt", None)
        if hasattr(mo, "wanted"): row.update(mt_model=mo.last, mt_wanted=mo.wanted(src, tgt))                      # which MT really produced this clip
        raws = []
        for sg in segs:
            if sg.get("asr_raw") and sg["asr_raw"] not in raws: raws.append(sg["asr_raw"])                  # stream mode: every sentence of a segment carries the same raw text
        row.update(asr_raw=" ".join(raws).strip(), asr_cleaned=sorted({x for sg in segs for x in sg.get("asr_clean", [])}))
        row.update(asr=asr, mt=mt, heard=hd, out_audio_s=round(len(out) / osr, 2) if len(out) else 0.0, n_segments=len(segs), heard_s=hs, segments=segs, **tm)
        e = ds.asr_err([m["text"]], [asr], src); row["asr_err"] = None if e is None else round(100 * e, 1)
        row["chrf"] = round(ds.sent_chrf(mt, m["ref"], tgt), 1) if m["ref"] and mt else None
        te = ds.asr_err([mt], [hd], tgt) if mt and with_heard else None; row["tts_err"] = None if te is None else round(100 * te, 1)
        row["heard_chrf"] = round(ds.sent_chrf(hd, m["ref"], tgt), 1) if m["ref"] and hd else None
        fn = f"{set_name}_{d}_{cid}.wav"; od = os.path.join(self.out_dir, run); os.makedirs(od, exist_ok=True)
        if len(out): open(os.path.join(od, fn), "wb").write(_wav_bytes(out, osr))
        row["file"] = fn if len(out) else ""; row["issues"] = issues_of(row); self._store(run, row)
        return row

    def _store(self, run, row):
        rows = self.runs.setdefault(run, [])
        rows[:] = [r for r in rows if not (r["set"] == row["set"] and r["dir"] == row["dir"] and r["id"] == row["id"])] + [row]   # a re-run replaces the old row
        with open(os.path.join(self.out_dir, f"{run}.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---- batches in a thread on the box
    def start_batch(self, sets, dirs, limit=0, cond="all"):
        if self.batch["running"]: raise RuntimeError("a batch is already running: " + str(self.batch["run"]))
        base_id = lambda cid: re.sub(r"_[a-z]+(_\d+db)?$", "", cid); plan = []
        for s in sets:
            if s not in self.sets: continue
            for d in dirs:
                it = list(self.items(s, d).values())
                if s == "adv" and cond != "all": it = [x for x in it if x["cond"] == cond]
                if limit:
                    if s == "adv" and cond == "all":
                        keep = []
                        for x in it:
                            if base_id(x["id"]) not in keep: keep.append(base_id(x["id"]))
                        keep = set(keep[:limit]); it = [x for x in it if base_id(x["id"]) in keep]
                    else: it = it[:limit]
                plan += [(s, d, x["id"]) for x in it]
        run = "_".join(sets) + "_" + time.strftime("%Y%m%d_%H%M%S")
        self.batch.update(run=run, running=True, done=0, total=len(plan), current="", started=time.time(), stop=False, error="")

        def work():
            try:
                for i, (s, d, cid) in enumerate(plan):
                    if self.batch["stop"]: break
                    self.batch["current"] = f"{s} {d} {cid}"; r = self.run_item(s, d, cid, run); self.batch["done"] = i + 1
                    print(f"  [{i + 1}/{len(plan)} {s} {d}] {cid}: ASR {r['asr_s']:.2f}s | MT {r['mt_s']:.2f}s | TTS {r['tts_s']:.2f}s | total {r['total_s']:.2f}s"
                          + (f" | {','.join(r['issues'])}" if r["issues"] else ""), flush=True)
            except Exception as e:
                self.batch["error"] = repr(e)[:300]
            finally:
                self.batch["running"] = False; self.batch["current"] = ""
        threading.Thread(target=work, daemon=True).start(); return dict(run=run, total=len(plan))

    def summary(self, run):
        rows = [r for r in self.runs.get(run, []) if not r.get("error")]; groups = {}
        for r in rows:
            groups.setdefault((r["set"], r["dir"], ""), []).append(r)
            if r["cond"]: groups.setdefault((r["set"], r["dir"], r["cond"]), []).append(r)
        out = []
        for (s, d, c), g in sorted(groups.items(), key=lambda kv: (kv[0][0], ds.DIRS.index(kv[0][1]), ds.CONDS.index(kv[0][2]) if kv[0][2] else -1)):
            src, tgt = d.split("-"); tot = [x["total_s"] for x in g]; e = ds.asr_err([x["text"] for x in g], [x["asr"] for x in g], src)
            gm = [x for x in g if x["mt"]]; te = ds.asr_err([x["mt"] for x in gm], [x["heard"] for x in gm], tgt) if gm and any(x["heard"] for x in gm) else None
            has_ref = all(x["ref"] for x in g)
            out.append(dict(set=s, dir=d, cond=c or "all", n=len(g), asr_metric="CER" if src in ("zh", "ko") else "WER", asr_err=None if e is None else round(100 * e, 2),
                            chrf=round(ds.chrf([x["mt"] for x in g], [x["ref"] for x in g], tgt), 2) if has_ref else None,
                            tts_metric="CER" if tgt in ("zh", "ko") else "WER", tts_err=None if te is None else round(100 * te, 2),
                            heard_chrf=round(ds.chrf([x["heard"] for x in g], [x["ref"] for x in g], tgt), 2) if has_ref and any(x["heard"] for x in g) else None,
                            asr_s=round(float(np.mean([x["asr_s"] for x in g])), 3), mt_s=round(float(np.mean([x["mt_s"] for x in g])), 3),
                            tts_s=round(float(np.mean([x["tts_s"] for x in g])), 3), total_s=round(float(np.mean(tot)), 3),
                            p50_s=round(float(np.median(tot)), 3), p90_s=round(float(np.percentile(tot, 90)), 3),
                            rtf=round(float(np.sum(tot) / max(sum(x["audio_s"] for x in g), 1e-6)), 3), n_issues=sum(1 for x in g if x["issues"])))
        return out

    def mt_status(self):
        mo = getattr(self.reg, "_mt_obj", None)
        return mo.status() if hasattr(mo, "status") else dict(route=None, big_in_use=False, degraded=False, errors=[], small=getattr(mo, "active", None))

    def issue_counts(self, run):
        c = {}
        for r in self.runs.get(run, []):
            for i in r.get("issues", []): c[i] = c.get(i, 0) + 1
        return [dict(code=k, text=ISSUE_TEXT.get(k, k), n=v) for k, v in sorted(c.items(), key=lambda kv: -kv[1])]

    # ---- one file with everything
    def export(self, run):
        rows = self.runs.get(run, []); zp = os.path.join(self.out_dir, f"{run}.zip"); od = os.path.join(self.out_dir, run)
        cols = ["set", "dir", "id", "cond", "audio_s", "text", "asr_raw", "asr", "asr_cleaned", "asr_err", "asr_s", "ref", "mt", "chrf", "mt_s", "heard", "tts_err", "heard_chrf", "tts_s",
                "out_audio_s", "total_s", "graph_s", "heard_s", "mt_model", "mt_wanted", "issues", "file", "error"]
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            b = io.StringIO(); w = csv.DictWriter(b, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(dict(r, issues=" ".join(r.get("issues", [])), asr_cleaned=" ".join(r.get("asr_cleaned", []))))
            z.writestr("results.csv", "﻿" + b.getvalue())
            sm = self.summary(run); z.writestr("summary.json", json.dumps(dict(run=run, device=self.info, summary=sm, issues=self.issue_counts(run)), ensure_ascii=False, indent=1))
            if sm:
                b = io.StringIO(); w = csv.DictWriter(b, fieldnames=list(sm[0].keys())); w.writeheader(); w.writerows(sm); z.writestr("summary.csv", "﻿" + b.getvalue())
            b = io.StringIO(); w = csv.writer(b); w.writerow(["set", "dir", "id", "cond", "segment", "of", "src", "tgt", "asr", "mt", "asr_s", "mt_s", "tts_s", "total_s", "graph_s", "audio_out_s"])
            for r in rows:
                for k, sg in enumerate(r.get("segments") or []):
                    t = sg["timing"]; w.writerow([r["set"], r["dir"], r["id"], r["cond"], k + 1, len(r["segments"]), sg["src"], sg["tgt"], sg["asr"], sg["mt"],
                                                  t.get("asr_s"), t.get("mt_s"), t.get("tts_s"), t.get("total_s"), t.get("graph_s"), t.get("audio_out_s")])
            z.writestr("segments.csv", "﻿" + b.getvalue())
            z.writestr("report.html", self.report_html(run, rows, sm))
            for r in rows:                                                    # translated audio (what the box produced) and source audio (16 kHz wav, played in report.html)
                if r.get("file") and os.path.exists(os.path.join(od, r["file"])): z.write(os.path.join(od, r["file"]), "out/" + r["file"], compress_type=zipfile.ZIP_STORED)
                try:
                    m = self.items(r["set"], r["dir"])[r["id"]]; z.writestr(zipfile.ZipInfo(f"in/{r['set']}_{r['dir']}_{r['id']}.wav", (2026, 1, 1, 0, 0, 0)), _wav_bytes(load_audio(m["wav"]), 16000), compress_type=zipfile.ZIP_STORED)
                except Exception as e: print(f"  export: no source audio for {r['id']}: {e!r}", flush=True)
            md = os.path.join(self.out_dir, "mic")                            # microphone / file / typed tests of this session
            if os.path.isdir(md):
                for fn in sorted(os.listdir(md)): z.write(os.path.join(md, fn), "mic/" + fn, compress_type=zipfile.ZIP_STORED)
        return zp

    def report_html(self, run, rows, sm):
        e = lambda s: html.escape(str(s if s is not None else "–")); f = lambda x: "–" if x is None else f"{x:.2f}"
        srow = "".join(f"<tr><td>{e(x['set'])}</td><td>{e(x['dir'])}</td><td>{e(x['cond'])}</td><td>{x['n']}</td><td>{e(x['asr_metric'])} {e(x['asr_err'])}</td><td>{e(x['chrf'])}</td>"
                       f"<td>{e(x['tts_metric'])} {e(x['tts_err'])}</td><td>{e(x['heard_chrf'])}</td><td>{f(x['asr_s'])}</td><td>{f(x['mt_s'])}</td><td>{f(x['tts_s'])}</td><td><b>{f(x['total_s'])}</b></td>"
                       f"<td>{f(x['p50_s'])} / {f(x['p90_s'])}</td><td>{x['n_issues']}</td></tr>" for x in sm)
        irow = "".join(f"<li>{e(x['text'])}: <b>{x['n']}</b></li>" for x in self.issue_counts(run))
        crow = []
        for r in rows:
            aud = f'<audio controls preload="none" src="out/{e(r["file"])}"></audio>' if r.get("file") else ""
            ain = f'<audio controls preload="none" src="in/{e(r["set"])}_{e(r["dir"])}_{e(r["id"])}.wav"></audio>'
            ref = f"<br><small>chuẩn: {e(r['ref'])}</small>" if r.get("ref") else ""
            iss = e(", ".join(ISSUE_TEXT.get(i, i) for i in r.get("issues", [])) or r.get("error") or "")
            crow.append(f"<tr class='{'bad' if r.get('issues') else ''}'><td>{e(r['dir'])}<br><small>{e(r['id'])} {e(r['cond'])}</small></td>"
                        f"<td>{ain}<br><small>gốc: {e(r['text'])}</small><br>{e(r.get('asr'))}<br><small>{f(r.get('asr_s'))} s · lỗi {e(r.get('asr_err'))} %</small></td>"
                        f"<td>{e(r.get('mt'))}<br><small>{f(r.get('mt_s'))} s · chrF {e(r.get('chrf'))}</small>{ref}</td>"
                        f"<td>{aud}<br><small>nghe lại: {e(r.get('heard'))}<br>{f(r.get('tts_s'))} s · lỗi TTS {e(r.get('tts_err'))} %</small></td>"
                        f"<td><b>{f(r.get('total_s'))}</b></td><td><small>{iss}</small></td></tr>")
        d = self.info
        return (f"<!doctype html><meta charset=utf-8><title>OneVoice {e(run)}</title><style>body{{font:14px system-ui;margin:20px}}table{{border-collapse:collapse;width:100%}}"
                f"td,th{{border-bottom:1px solid #ddd;padding:5px;vertical-align:top;text-align:left}}small{{color:#666}}tr.bad{{background:#fff4f2}}audio{{height:28px;width:200px}}</style>"
                f"<h2>OneVoice: kết quả chạy trên chip ({e(run)})</h2>"
                f"<p>Máy {e(d['host'])} · {e(d['machine'])} · SoC id {e(d['soc_id'])} · {e(d['os'])} · backend <b>{e(d['backend'])}</b> · model {e(d['models'])} · perf {e(d['perf'])} · {len(rows)} clip</p>"
                f"<h3>Tổng hợp</h3><table><tr><th>Bộ</th><th>Chiều</th><th>Mức ồn</th><th>Clip</th><th>Lỗi ASR %</th><th>chrF MT</th><th>Lỗi TTS % (ASR nghe lại so với chữ MT)</th>"
                f"<th>chrF nghe lại</th><th>ASR s</th><th>MT s</th><th>TTS s</th><th>Tổng s</th><th>p50/p90</th><th>Clip có vấn đề</th></tr>{srow}</table>"
                f"<h3>Vấn đề</h3><ul>{irow or '<li>không có</li>'}</ul>"
                f"<h3>Từng clip</h3><table><tr><th>Clip</th><th>Âm thanh gốc và ASR</th><th>MT</th><th>Âm thanh dịch (TTS) và ASR nghe lại</th><th>Tổng s</th><th>Vấn đề</th></tr>{''.join(crow)}</table>")

    # ---- microphone / typed text
    def translate_wav(self, body, q):
        wav = _read_wav(body)
        tr = self.translator(q["src"], q["tgt"]) if q.get("src") and q.get("tgt") else self.conversation(tuple((q.get("pair") or self.default).split(",")))
        with self.lock:
            segs, tm, out, osr = self._run(tr, wav)
            tgt = segs[0]["tgt"] if segs else None; hd, hs = self.heard(out, osr, tgt) if tgt else ("", 0.0)
        stamp = time.strftime("%Y%m%d_%H%M%S"); od = os.path.join(self.out_dir, "mic"); os.makedirs(od, exist_ok=True)
        open(os.path.join(od, f"{stamp}_in.wav"), "wb").write(body); wb = _wav_bytes(out, osr) if len(out) else b""
        if wb: open(os.path.join(od, f"{stamp}_out.wav"), "wb").write(wb)
        with open(os.path.join(od, "mic.jsonl"), "a", encoding="utf-8") as f: f.write(json.dumps(dict(stamp=stamp, segments=segs, heard=hd, **tm), ensure_ascii=False) + "\n")
        return dict(segments=segs, timing=tm, heard=hd, heard_s=hs, box_s=tm["wall_s"], audio_s=round(len(wav) / 16000, 2), wav_b64=base64.b64encode(wb).decode())

    def translate_stream(self, body, q):
        """Sentence-pipelined version of translate_wav (what `live` mode does): yields one dict per sentence as soon as its audio is ready, so the
        browser can start playing sentence 1 while the box translates and synthesises sentence 2. Last item: {"done": True, ...totals}."""
        wav = _read_wav(body)
        tr = self.translator(q["src"], q["tgt"]) if q.get("src") and q.get("tgt") else self.conversation(tuple((q.get("pair") or self.default).split(",")))
        t0 = time.perf_counter(); n = 0; first = None; tgt = None; outs = []; osr = 16000
        with self.lock:
            for r in tr.translate_stream(wav):
                if not r["mt"]: continue
                n += 1; tgt = r["tgt"]; wb = _wav_bytes(r["wav"], r["sr"]) if len(r["wav"]) else b""; outs.append(r); osr = r["sr"]
                if first is None and wb: first = round(time.perf_counter() - t0, 3)
                yield dict(i=n, src=r["src"], tgt=r["tgt"], asr=r["asr"], mt=r["mt"], timing={k: round(float(v), 4) for k, v in r["timing"].items()},
                           at_s=round(time.perf_counter() - t0, 3), wav_b64=base64.b64encode(wb).decode())
            total = round(time.perf_counter() - t0, 3); hd = ""
            if outs and tgt:
                allw, asr_sr = concat_results(outs); hd, _ = self.heard(allw, asr_sr, tgt)
        yield dict(done=True, n=n, first_audio_s=first, total_s=total, audio_s=round(len(wav) / 16000, 2), heard=hd)

    def text(self, a):
        """Typed sentence -> MT -> TTS -> heard (tests MT and TTS formats such as times, dates, numbers, names, without the ASR).
        src == tgt: TTS only (type exactly what the TTS should read)."""
        src, tgt, txt = a["src"], a["tgt"], a["text"].strip()
        tr = self.translator(src, tgt) if src != tgt else self.translator("vi" if tgt != "vi" else "en", tgt)
        with self.lock:
            t0 = time.perf_counter(); mt = txt if src == tgt else tr.e.translate_text(txt, src, tgt)      # same MT path as the speech pipeline
            t1 = time.perf_counter(); out, osr = tr.tts.synthesize(mt) if mt else (np.zeros(0, np.float32), 16000); t2 = time.perf_counter()
            hd, hs = self.heard(out, osr, tgt)
        te = ds.asr_err([mt], [hd], tgt) if mt else None
        return dict(src=src, tgt=tgt, text=txt, mt=mt, heard=hd, mt_s=round(t1 - t0, 3), tts_s=round(t2 - t1, 3), heard_s=hs, tts_err=None if te is None else round(100 * te, 1),
                    out_audio_s=round(len(out) / osr, 2) if len(out) else 0, wav_b64=base64.b64encode(_wav_bytes(out, osr) if len(out) else b"").decode())


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a): pass

        def _send(self, code, body, ctype="application/json; charset=utf-8"):
            if not isinstance(body, (bytes, bytearray)): body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def _file(self, path, ctype, name=None):
            if not os.path.exists(path): return self._send(404, {"error": "no such file"})
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(os.path.getsize(path)))
            if name: self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            with open(path, "rb") as f:
                for blk in iter(lambda: f.read(1 << 20), b""): self.wfile.write(blk)

        def _body(self): return self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))

        def do_GET(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"): return self._send(200, open(HTML, "rb").read(), "text/html; charset=utf-8")
                if u.path == "/health": return self._send(200, {"ok": True, "dirs": [list(d) for d in DIRECTIONS]})
                if u.path == "/api/info":
                    counts = {s: {d: len(app.items(s, d)) for d in ds.DIRS} for s in app.sets}
                    return self._send(200, dict(device=app.info, sets=list(app.sets), counts=counts, default_pair=app.default, data_root=app.data_root,
                                                runs=sorted(app.runs, reverse=True), batch=app.batch, mt=app.mt_status()))
                if u.path == "/api/items": return self._send(200, [dict(id=m["id"], text=m["text"], ref=m["ref"], cond=m["cond"]) for m in app.items(q["set"], q["dir"]).values()])
                if u.path == "/api/audio":
                    m = app.items(q["set"], q["dir"])[q["id"]]; return self._send(200, _wav_bytes(load_audio(m["wav"]), 16000), "audio/wav")
                if u.path == "/api/out": return self._file(os.path.join(app.out_dir, os.path.basename(q["run"]), os.path.basename(q["file"])), "audio/wav")
                if u.path == "/api/batch/status": return self._send(200, app.batch)
                if u.path == "/api/rows":
                    rows = app.runs.get(q.get("run", ""), []); return self._send(200, dict(n=len(rows), rows=rows[int(q.get("since", 0)):]))
                if u.path == "/api/summary": return self._send(200, dict(summary=app.summary(q.get("run", "")), issues=app.issue_counts(q.get("run", ""))))
                if u.path == "/api/export":
                    run = os.path.basename(q.get("run", "")); return self._file(app.export(run), "application/zip", f"onevoice_{run}.zip")
                return self._send(404, {"error": "not found"})
            except Exception as e:
                import traceback; traceback.print_exc(); return self._send(500, {"error": repr(e)})

        def do_POST(self):
            u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/batch/start":
                    a = json.loads(self._body() or b"{}")
                    return self._send(200, app.start_batch(a.get("sets", ["btc", "adv"]), a.get("dirs", ds.DIRS), int(a.get("limit", 0)), a.get("cond", "all")))
                if u.path == "/api/batch/stop": app.batch["stop"] = True; return self._send(200, app.batch)
                if u.path == "/api/run_item":
                    a = json.loads(self._body() or b"{}"); return self._send(200, app.run_item(a["set"], a["dir"], a["id"], a.get("run") or "hardcases"))
                if u.path == "/api/text": return self._send(200, app.text(json.loads(self._body() or b"{}")))
                if u.path == "/api/stream":
                    body = self._body(); self.send_response(200); self.send_header("Content-Type", "application/x-ndjson; charset=utf-8"); self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close"); self.end_headers(); self.close_connection = True
                    try:
                        for item in app.translate_stream(body, q):
                            self.wfile.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")); self.wfile.flush()
                            if not item.get("done"): print(f"  stream [{item['i']}] {item['src']}->{item['tgt']}: {item['asr']}  ->  {item['mt']}  (+{item['at_s']:.2f}s)", flush=True)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        self.wfile.write((json.dumps({"error": repr(e)}, ensure_ascii=False) + "\n").encode("utf-8")); self.wfile.flush()
                    return
                if u.path in ("/api/translate", "/translate"):
                    r = app.translate_wav(self._body(), q)
                    for s in r["segments"]: print(f"  mic {s['src']}->{s['tgt']}: {s['asr']}  ->  {s['mt']}  ({s['timing']['total_s']:.2f}s)", flush=True)
                    return self._send(200, r)
                return self._send(404, {"error": "not found"})
            except Exception as e:
                import traceback; traceback.print_exc(); return self._send(500, {"error": repr(e)})
    return H


def serve(reg, models_path, data_root, default_pair, port, warm_dirs=None):
    app = WebApp(reg, models_path, data_root, default_pair)
    if warm_dirs:
        print("warming up (loading models)...", flush=True)
        try: reg.warmup(warm_dirs)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"!! warmup failed: {e!r} (the page still starts; errors will show per clip)", flush=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    st = app.mt_status()
    print(("!!! MT DEGRADED: " + str(st["errors"])) if st.get("degraded") else (f"MT routing OK: {st['route']}" if st.get("route") else f"MT: {st.get('small')}"), flush=True)
    print(f"web UI + API on 127.0.0.1:{port} | data sets: {app.sets or 'none (give --data_root)'} | backend {reg.backend}\n"
          f"laptop: python tools/laptop_web.py   (or: adb forward tcp:{port} tcp:{port} and open http://127.0.0.1:{port})", flush=True)
    srv.serve_forever()

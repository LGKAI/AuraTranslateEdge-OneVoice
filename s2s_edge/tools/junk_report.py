# -*- coding: utf-8 -*-
"""Find ASR / MT junk in real results, so the lists in s2s_edge/asr_clean.py grow from evidence, not guesses.
Run it on the laptop after a chip run (results.csv inside the web zip, or outputOnevoice/all_results.csv, or the ADV json of tools/eval_adv.py):
  python tools/junk_report.py <results.csv | adv.json> [...more files] [--out junk_report.md]
It reports
  1. what the filter REMOVED (rows tagged asr_cleaned): read these to check precision (was every cut really junk?)
  2. candidates it did NOT remove: short pieces at the start / end of the ASR text (a lone sentence of <= 2 words / <= 2 CJK chars, or a
     1-3 letter token followed by "." / ":") that are absent from the gold transcript -> possible new fillers / labels (recall)
  3. MT outputs that still contain symbols, a leading punctuation mark or a label."""
import argparse, collections, csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.asr_clean import clean_asr, clean_mt

SENT = re.compile(r"(?<=[.。!?！？])\s*")
CJK = re.compile(r"[㐀-鿿가-힣]")
norm = lambda s: re.sub(r"[\W_]+", "", s).lower()


def load(p):
    if p.endswith(".json"):
        d = json.load(open(p, encoding="utf-8")); rows = []
        for k, v in d.items():
            for r in (v if isinstance(v, list) else []):
                rows.append(dict(id=r.get("id"), dir=k, text=r.get("text", ""), asr=r.get("asr", ""), asr_raw=r.get("asr_raw", r.get("asr", "")), mt=r.get("mt", ""), src_file=os.path.basename(p)))
        return rows
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    for r in rows: r.setdefault("asr_raw", r.get("asr", "")); r["asr_raw"] = r["asr_raw"] or r.get("asr", ""); r["src_file"] = os.path.basename(p)
    return rows


def small(piece):
    p = piece.strip(); w = norm(p)
    if not w: return False
    return (len(CJK.findall(p)) <= 2 and len(w) <= 2) if CJK.search(p) else len(p.split()) <= 2 and len(w) <= 6


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", default="junk_report.md"); a = ap.parse_args()
    rows = [r for p in a.files for r in load(p)]; removed = []; cand = collections.defaultdict(list); mtj = []
    for r in rows:
        src, tgt = (r.get("dir") or "vi-en").split("-")[:2]; raw = (r.get("asr_raw") or "").strip(); gold = norm(r.get("text", ""))
        clean, tags = clean_asr(raw, src)
        if tags: removed.append((r, raw, clean, tags))
        parts = [x for x in SENT.split(clean) if x.strip()]
        edges = ([parts[0]] if parts else []) + ([parts[-1]] if len(parts) > 1 else [])
        for e in edges:
            if len(parts) > 1 and small(e) and norm(e) and norm(e) not in gold: cand[(src, e.strip())].append(r["id"])
        m = re.match(r"^\s*([^\W\d_]{1,3})\s*[.:：]\s+\S", clean)
        if m and norm(m.group(1)) not in gold[:12]: cand[(src, m.group(1) + ".")].append(r["id"])
        mt = r.get("mt") or ""
        if mt:
            c2, t2 = clean_mt(mt, clean, tgt)
            if t2: mtj.append((r, mt, c2, t2))
            elif re.search(r"[^\w\s.,!?;:'\"()\-–—%/。，、！？；：…“”‘’《》（）·$€₫+=&@#*°]", mt): mtj.append((r, mt, mt, ["other_symbol?"]))
    L = [f"# Junk report ({len(rows)} rows from {', '.join(os.path.basename(f) for f in a.files)})", "",
         f"## 1. Removed by the filter: {len(removed)} rows (CHECK each one was junk)", "", "| id | dir | tags | raw ASR | after filter |", "|---|---|---|---|---|"]
    L += [f"| {r['id']} | {r.get('dir')} | {' '.join(t)} | {raw[:70]} | {c[:70]} |" for r, raw, c, t in removed]
    L += ["", "## 2. Candidates NOT removed (short edge pieces absent from the gold transcript), most frequent first", "", "| lang | piece | n | ids |", "|---|---|---|---|"]
    L += [f"| {k[0]} | `{k[1]}` | {len(v)} | {', '.join(map(str, v[:6]))} |" for k, v in sorted(cand.items(), key=lambda kv: -len(kv[1]))]
    L += ["", f"## 3. MT outputs with junk: {len(mtj)}", "", "| id | dir | tags | MT | after guard |", "|---|---|---|---|---|"]
    L += [f"| {r['id']} | {r.get('dir')} | {' '.join(t)} | {mt[:70]} | {c[:70]} |" for r, mt, c, t in mtj]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"rows {len(rows)} | removed {len(removed)} | candidates {len(cand)} ({sum(map(len, cand.values()))} rows) | MT junk {len(mtj)} -> {a.out}")


if __name__ == "__main__": main()

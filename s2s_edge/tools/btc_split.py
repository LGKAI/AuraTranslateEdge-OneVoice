# -*- coding: utf-8 -*-
"""Fixed dev / holdout / quick split of the organisers' sample set, so method choices are made on `dev` and confirmed ONCE on `holdout`.
The split is by row id (sha1 of the id), so vi-ko and ko-vi (same rows, same speakers' sentences) fall on the same side and the holdout never leaks.
  quick   : 30 rows per direction taken from dev (a fast check, about 2 min per direction on the NPU)
  dev     : 60 % of the rows (method tuning)
  holdout : 40 % of the rows (final confirmation only; do not look at it while tuning)
  python tools/btc_split.py --data /path/onevoice_btc_data --out eval/btc_split.json
Then: python tools/eval_btc.py ... --split_file eval/btc_split.json --split quick"""
import argparse, hashlib, json, os

FILES = {   # direction -> (folder, sub-folder or "", xlsx)
    "vi-en": ("Vie-Eng dataset", "Vie-Eng", "OneVoice_V_to_E_samples_100.xlsx"),
    "en-vi": ("Vie-Eng dataset", "Eng-Vie", "OneVoice_E_to_V_samples_100.xlsx"),
    "vi-ko": ("Vie-Kor dataset", "", "OneVoice_V_to_K_samples_180.xlsx"), "ko-vi": ("Vie-Kor dataset", "", "OneVoice_V_to_K_samples_180.xlsx"),
    "vi-zh": ("Vie-Man dataset", "", "OneVoice_V_to_M_samples_180.xlsx"), "zh-vi": ("Vie-Man dataset", "", "OneVoice_V_to_M_samples_180.xlsx"),
}


def ids_of(path):
    import openpyxl
    rows = list(openpyxl.load_workbook(path, read_only=True).active.values)
    return [str(r[0]) for r in rows[1:] if r and r[0]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="eval/btc_split.json")
    ap.add_argument("--holdout", type=float, default=0.4); ap.add_argument("--quick", type=int, default=30); a = ap.parse_args()
    key = lambda i: hashlib.sha1(i.encode("utf-8")).hexdigest()
    res = {"note": "dev = tune, holdout = confirm once, quick = first rows of dev by hash", "by_dir": {}}
    for d, (folder, sub, x) in FILES.items():
        ids = sorted(ids_of(os.path.join(a.data, folder, sub, x)), key=key)
        cut = int(round(len(ids) * (1 - a.holdout))); dev, hold = ids[:cut], ids[cut:]
        res["by_dir"][d] = {"dev": dev, "holdout": hold, "quick": dev[: a.quick]}
        print(f"{d}: {len(ids)} rows -> dev {len(dev)}, holdout {len(hold)}, quick {min(a.quick, len(dev))}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("saved", a.out)


if __name__ == "__main__": main()

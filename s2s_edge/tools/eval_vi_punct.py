# -*- coding: utf-8 -*-
"""Accuracy of textnorm.vi_punct on the Vietnamese sentences of the organisers' sample set: strip punctuation and case, restore, compare the final mark.
  python tools/eval_vi_punct.py --data /path/onevoice_btc_data"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openpyxl
from s2s_edge.textnorm import vi_punct

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--show", type=int, default=20); a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8")
files = ["Vie-Eng dataset/Vie-Eng/OneVoice_V_to_E_samples_100.xlsx", "Vie-Kor dataset/OneVoice_V_to_K_samples_180.xlsx", "Vie-Man dataset/OneVoice_V_to_M_samples_180.xlsx"]
sents = [str(r[1]).strip() for f in files for r in list(openpyxl.load_workbook(os.path.join(a.data, f), read_only=True).active.values)[1:] if r and r[0]]
sents = [s for s in sents if s and s[-1] in ".?!"]; cm = {(g, p): 0 for g in "?." for p in "?."}; bad = []
for s in sents:
    gold = "?" if s[-1] == "?" else "."; pred = vi_punct(re.sub(r"[^\w\s]", "", s).lower())[-1]; cm[(gold, pred)] += 1
    if gold != pred: bad.append((gold, s))
ok = cm[("?", "?")] + cm[(".", ".")]
print(f"accuracy {ok}/{len(sents)} = {ok / len(sents):.3f}   questions found {cm[('?', '?')]}/{cm[('?', '?')] + cm[('?', '.')]}   statements kept {cm[('.', '.')]}/{cm[('.', '.')] + cm[('.', '?')]}")
for g, s in bad[: a.show]: print(" ", g, s)

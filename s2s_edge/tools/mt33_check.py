# -*- coding: utf-8 -*-
"""ON THE BOX (user system): NLLB-200 3.3B (w8a16, 4 graphs) on the NPU vs the laptop float run of the same graphs and vs the chip's 600M output,
on eval/mt33_testset.json (20 chip-ASR sentences per direction). Prints chrF per direction, identical-output rate, MT time per sentence, memory.
  runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/mt33_check.py 2>&1 | grep -a "^##" | tee /tmp/mt33_check.txt'"""
import json, os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry
from s2s_edge import datasets as ds
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reg = Registry(os.path.join(ROOT, "models_qcs8550", "models_nllb33.json"), "npu"); t0 = time.time(); mt = reg.mt()
st = mt.status() if hasattr(mt, "status") else {}
print(f"## MT built in {time.time() - t0:.1f} s; route {st.get('route')}; 3.3B in use: {st.get('big_in_use')}; degraded: {st.get('degraded')} {st.get('errors', '')}", flush=True)
if st.get("degraded") or not st.get("big_in_use"):
    print("## " + "!" * 100); print("## FAIL: NLLB-3.3B IS NOT IN USE: the numbers below would be 600M, NOT the target. Read the error above and fix it first."); print("## " + "!" * 100); sys.exit(2)
data = json.load(open(os.path.join(ROOT, "eval", "mt33_testset.json"), encoding="utf-8")); res = {}
for d, rows in data.items():
    s, t = d.split("-"); hyp, tm = [], []
    for r in rows:
        a = time.perf_counter(); hyp.append(mt(r["src"], s, t)); assert mt.last == mt.wanted(s, t), f"{d}: produced by {mt.last}, wanted {mt.wanted(s, t)}"; tm.append(time.perf_counter() - a)
    R = [r["ref"] for r in rows]; same = sum(h.strip() == r["laptop_bf16_split"].strip() for h, r in zip(hyp, rows))
    print(f"## {d}: chrF NPU-3.3B {ds.chrf(hyp, R, t):.2f} | laptop float 3.3B {ds.chrf([r['laptop_bf16_split'] for r in rows], R, t):.2f} | chip 600M {ds.chrf([r['chip600m'] for r in rows], R, t):.2f}"
          f" | identical to laptop {same}/{len(rows)} | MT {sum(tm) / len(tm):.2f} s/sentence (max {max(tm):.2f})", flush=True)
    res[d] = [dict(r, npu33=h, t=round(x, 3)) for r, h, x in zip(rows, hyp, tm)]
json.dump(res, open(os.path.join(ROOT, "logs", "mt33_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"## peak RSS of this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)

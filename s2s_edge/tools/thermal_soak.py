# -*- coding: utf-8 -*-
"""Does the box slow down when it gets hot? Runs one graph back-to-back for N seconds on the golden inputs and logs latency + temperatures.
The KIMQ8550 datasheet lists thermal as "TBD" (typical 11.9 W, max 18 W on Ubuntu), and a full BTC run keeps the NPU busy for about an hour,
so a drift between the first and the last minute is the number to look at (slower at the end = throttling; plan cooling / pauses).
  python tools/thermal_soak.py --models models_qcs8550/models.json --graph mt_decB --seconds 600
Writes logs/soak_<graph>_<time>.csv (t, latency_ms, max temperature) and prints p50 of the first and last minute."""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry

GOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "golden")


def temps():
    out = {}
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try: out[open(os.path.join(z, "type")).read().strip()] = int(open(os.path.join(z, "temp")).read().strip()) / 1000.0
        except Exception: pass
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--backend", default="npu"); ap.add_argument("--graph", default="mt_decB")
    ap.add_argument("--seconds", type=int, default=600); a = ap.parse_args()
    reg = Registry(a.models, a.backend); g = reg.graph(a.graph); z = np.load(reg.gold_path(a.graph)); ins = {n[3:]: z[n] for n in z.files if n.startswith("in_")}
    os.makedirs("logs", exist_ok=True); path = f"logs/soak_{a.graph}_{time.strftime('%Y%m%d_%H%M%S')}.csv"; rows = []; t0 = time.time(); last_t = -1e9; tmax, hot = float("nan"), ""
    with open(path, "w") as f:
        f.write("t_s,latency_ms,max_temp_c,hottest_zone\n")
        while time.time() - t0 < a.seconds:
            s = time.perf_counter(); g(**ins); ms = (time.perf_counter() - s) * 1000; t = time.time() - t0
            if t - last_t >= 5:
                T = temps(); last_t = t
                if T: hot, tmax = max(T.items(), key=lambda kv: kv[1])
            rows.append((t, ms)); f.write(f"{t:.2f},{ms:.2f},{tmax},{hot}\n")
    t, ms = np.array([r[0] for r in rows]), np.array([r[1] for r in rows]); first, last = ms[t < 60], ms[t > t[-1] - 60]
    print(f"{a.graph}: {len(ms)} runs in {t[-1]:.0f} s; p50 first minute {np.median(first):.2f} ms, last minute {np.median(last):.2f} ms "
          f"(x{np.median(last) / np.median(first):.2f}); max temperature seen {tmax} C ({hot}); log {path}")


if __name__ == "__main__": main()

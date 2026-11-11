# -*- coding: utf-8 -*-
"""Profile the context binaries produced by relink_device.py on the hosted device: proves each one loads on that SoC and gives the chip time per call.
  python tools/profile_links.py --links models_qcs8550/link_jobs.json --device "QCS8550 (Proxy)" --out models_qcs8550/profile.json"""
import argparse, json, sys
import qai_hub as hub

ap = argparse.ArgumentParser(); ap.add_argument("--links", required=True); ap.add_argument("--device", default="QCS8550 (Proxy)"); ap.add_argument("--out", required=True)
a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); dev = hub.Device(a.device); links = json.load(open(a.links)); jobs = {}
for n, jid in links.items():
    jobs[n] = hub.submit_profile_job(model=hub.get_job(jid).get_target_model(), device=dev, name=f"s2s_profile_{n}"); print(n, "profile job", jobs[n].job_id, flush=True)
res = {}
for n, j in jobs.items():
    st = j.wait()
    if not st.success: res[n] = dict(job=j.job_id, ok=False, message=st.message); print("FAILED", n, st.message, flush=True); continue
    s = j.download_profile()["execution_summary"]
    res[n] = dict(job=j.job_id, ok=True, ms=round(s["estimated_inference_time"] / 1000, 2), peak_mb=round(s.get("estimated_inference_peak_memory", 0) / 2**20, 1))
    print("ok", n, res[n], flush=True)
json.dump(res, open(a.out, "w"), indent=1)

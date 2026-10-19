# -*- coding: utf-8 -*-
"""Probe: link a compiled DLC into a QNN context binary (prepared offline) and profile -> does first-load time drop vs the DLC (432 s for VITS g5)?"""
import sys, json, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
src = sys.argv[1]; key = sys.argv[2]; dev = hub.Device("Dragonwing IQ-9075 EVK")
tm = hub.get_job(src).get_target_model()
lj = hub.submit_link_job(models=[tm], device=dev, name=f"link_{key}"); print(key, "link", lj.job_id, flush=True); lj.wait(); st = lj.get_status(); print(key, "link", st.code, (st.message or "")[:300], flush=True)
if str(st.code) != "SUCCESS": sys.exit(1)
cb = lj.get_target_model(); pj = hub.submit_profile_job(model=cb, device=dev, name=f"link_{key}_profile"); print(key, "profile", pj.job_id, flush=True); pj.wait()
st = pj.get_status(); print(key, "profile", st.code, (st.message or "")[:300], flush=True)
p = pj.download_profile(); es = p["execution_summary"]
print(key, "PROFILE", f"{es['estimated_inference_time']/1000:.1f} ms | first load {es.get('first_load_time',0)/1e6:.1f}s | warm load {es.get('warm_load_time',0)/1e6:.1f}s | peak {es.get('estimated_inference_peak_memory',0)/1e6:.0f} MB", flush=True)

import sys, json, os
sys.path.insert(0, "src/step4_npu")
import qai_hub as hub
from pathlib import Path
import deploy_piper_components as d
out = Path("outputs/piper_vi_npu"); jobs = {}
dev = hub.Device(d.DEVICE_NAME)
for comp in d.COMPONENTS:
    j = hub.submit_compile_job(model=str(out/"components"/d.MODEL_FILES[comp]), device=dev,
        options=d.COMPILE_OPTIONS[comp], name=f"piper_vi_{comp}_iq9075_test")
    jobs[comp] = j.job_id; print(comp, j.job_id, j.url, flush=True)
json.dump(jobs, open("compile_jobs.json","w"), indent=1)

# -*- coding: utf-8 -*-
"""Usage: python submit_compile.py <key> <onnx>   e.g.  python submit_compile.py fullg8 build/piper_vi_full_g8.onnx
Submits the compile job for the Dragonwing IQ-9075 EVK and records its id in build/jobs.json."""
import sys, os, json
import qai_hub as hub
k, f = sys.argv[1], sys.argv[2]
os.makedirs("build", exist_ok=True)
J = json.load(open("build/jobs.json")) if os.path.exists("build/jobs.json") else {}
j = hub.submit_compile_job(model=f, device=hub.Device("Dragonwing IQ-9075 EVK"),
                           options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io", name=f"piper_vi_{k}")
J[k] = j.job_id
json.dump(J, open("build/jobs.json", "w"), indent=1)
print(k, j.job_id, j.url)

"""Utility: download detailed compiler logs for a given AI Hub job id."""
import sys
import os

import qai_hub as hub

job_id = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", f"joblogs_{job_id}")
job = hub.get_job(job_id)
job.download_job_logs(out_dir)
print(f"logs downloaded to {out_dir}")
for root, dirs, files in os.walk(out_dir):
    for fn in files:
        print(os.path.join(root, fn))

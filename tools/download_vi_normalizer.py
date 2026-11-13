from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
import json
r=Path(__file__).resolve().parents[1];repo='nmcuong/ByT5-Vi-Normalization';info=HfApi().model_info(repo);out=r/'models/source/byt5-vi-normalization'
print('REVISION',info.sha,flush=True)
snapshot_download(repo,revision=info.sha,local_dir=str(out),allow_patterns=['*.json','*.safetensors','README.md','LICENSE'],max_workers=3)
(out/'provenance.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'use':'task-finetuned Vietnamese written-to-spoken normalization'},indent=2));print('READY',flush=True)

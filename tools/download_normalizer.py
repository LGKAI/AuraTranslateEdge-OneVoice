from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
import json,sys
r=Path(__file__).resolve().parents[1];repo=sys.argv[1] if len(sys.argv)>1 else 'Qwen/Qwen3-1.7B';info=HfApi().model_info(repo);out=r/'models/source'/('qwen3-4b-instruct-mlx4' if '4bit' in repo else 'qwen3-1.7b')
print('REVISION',info.sha,flush=True)
snapshot_download(repo,revision=info.sha,local_dir=str(out),allow_patterns=['*.json','*.jinja','*.safetensors','*.txt','*.model','LICENSE','README.md'],max_workers=3)
(out/'provenance.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'use':'local learned text normalization; no task-specific fine-tuning'},indent=2));print('READY',flush=True)

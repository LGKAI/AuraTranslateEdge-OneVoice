from pathlib import Path
import tarfile,torch,json,hashlib
from transformers import AutoConfig,AutoModelForSeq2SeqLM,AutoTokenizer
r=Path(__file__).resolve().parents[1];p=r/'models/source/nemo-en-tn'
config=AutoConfig.from_pretrained(str(p/'t5-base-files'));m=AutoModelForSeq2SeqLM.from_config(config)
with tarfile.open(p/'decoder.nemo') as f:state=torch.load(f.extractfile('./model_weights.ckpt'),map_location='cpu',weights_only=True)
m.load_state_dict({k.removeprefix('model.'):v for k,v in state.items()},strict=True)
out=p/'hf';m.save_pretrained(str(out));AutoTokenizer.from_pretrained(str(p/'t5-base-files')).save_pretrained(str(out))
(out/'provenance.json').write_text(json.dumps({'source':'https://api.ngc.nvidia.com/v2/models/nvidia/nemo/neural_text_normalization_t5/versions/1.5.0/files/neural_text_normalization_t5_decoder.nemo','version':'1.5.0','archive_sha256':hashlib.sha256((p/'decoder.nemo').read_bytes()).hexdigest(),'weight_mapping':'remove model. prefix; strict load; no finetuning','covering_grammar':False},indent=2));print('READY')

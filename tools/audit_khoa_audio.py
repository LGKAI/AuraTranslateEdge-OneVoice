"""Independent ASR cross-check; never uses references as prompts."""
from pathlib import Path
import json,time,math
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
import mlx_whisper
r=Path(__file__).resolve().parents[1];source=r.parent/'Khoa';out=r/'results/khoa_audit';out.mkdir(exist_ok=True,parents=True);rows=[]
for p in sorted(source.glob('*.json')):
 d=json.loads(p.read_text());row={'id':p.stem,'metadata':d}
 for k,lang in [('in','en'),('out','vi')]:
  w,sr=sf.read(p.with_name(p.stem+'_'+k+'.wav'),dtype='float32')
  if w.ndim>1:w=w.mean(axis=1)
  if sr!=16000:
   g=math.gcd(sr,16000);w=resample_poly(w,16000//g,sr//g).astype(np.float32)
  start=time.perf_counter();rec=mlx_whisper.transcribe(w,path_or_hf_repo='/private/tmp/onevoice-audit-whisper',language=lang,temperature=0.0,condition_on_previous_text=False,verbose=None)
  row['independent_'+k]={'text':rec['text'].strip(),'seconds':time.perf_counter()-start,'segments':[{z:v for z,v in seg.items() if z in ('start','end','text','no_speech_prob','avg_logprob')} for seg in rec['segments']]}
  print(p.stem,k,repr(rec['text'].strip()),flush=True)
 rows.append(row);(out/'independent_asr.partial.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(out/'independent_asr.json').write_text(json.dumps({'model':'mlx-community/whisper-small-mlx','local_model':'/private/tmp/onevoice-audit-whisper','method':'forced English for input / Vietnamese for output; no initial prompts or references; model output is cross-check evidence, not human gold transcription','clips':rows},ensure_ascii=False,indent=2));print('AUDIT_COMPLETE',len(rows),flush=True)

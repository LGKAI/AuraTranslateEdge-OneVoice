"""Audit existing en-vi recordings without modifying or translating them again."""
from pathlib import Path
import json,time,math,hashlib,csv,collections
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
import mlx_whisper
r=Path(__file__).resolve().parents[1];source=r.parent/'en-vi/BTC';out=r/'results/en_vi_audit';out.mkdir(exist_ok=True,parents=True)
old=json.loads((r/'results/khoa_audit/independent_asr.json').read_text())['clips'];old={z['id']:z for z in old};rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(source.glob('*.json')):
 d=json.loads(p.read_text());row={'id':p.stem,'metadata':d,'files':{}}
 for k,lang in [('in','en'),('out','vi')]:
  f=p.with_name(p.stem+'_'+k+'.wav')
  if not f.exists():row['files'][k]={'exists':False};print(p.stem,k,'MISSING',flush=True);continue
  w,sr=sf.read(f,dtype='float32');row['files'][k]={'exists':True,'sha256':sha(f),'duration':len(w)/sr,'rate':sr,'finite':bool(np.isfinite(w).all()),'peak':float(np.max(abs(w))),'rms':float(np.sqrt(np.mean(w*w)))}
  previous=r.parent/'Khoa'/f.name
  if p.stem in old and previous.exists() and sha(previous)==row['files'][k]['sha256']:
   row['independent_'+k]=old[p.stem]['independent_'+k];row['files'][k]['reused_identical_audio_check']=True
  else:
   if w.ndim>1:w=w.mean(axis=1)
   if sr!=16000:
    g=math.gcd(sr,16000);w=resample_poly(w,16000//g,sr//g).astype(np.float32)
   start=time.perf_counter();rec=mlx_whisper.transcribe(w,path_or_hf_repo='/private/tmp/onevoice-audit-whisper',language=lang,temperature=0.0,condition_on_previous_text=False,verbose=None)
   row['independent_'+k]={'text':rec['text'].strip(),'seconds':time.perf_counter()-start}
  print(p.stem,k,repr(row['independent_'+k]['text']),flush=True)
 rows.append(row);(out/'audit.partial.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(out/'audit.json').write_text(json.dumps({'scope':'50 English→Vietnamese cases; existing files only, no rerunning translation','model':'mlx-community/whisper-small-mlx','method':'independent ASR on actual input and output; no reference prompts; reuse only when audio hashes match Khoa recordings','limitation':'ASR cross-check is not human listening gold; especially tone and names may be misrecognized','cases':rows},ensure_ascii=False,indent=2));print('AUDIT_COMPLETE',len(rows),flush=True)

"""Current contest_VNG front-end rerun. Frozen model/data, isolated output.
Quality metric is text-conformance, not audio MOS/WER. Public CPU acoustic
models are substitutes because fused ONNX/QNN files are absent.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import sys,json,re,time,random,hashlib,subprocess,platform,statistics,csv,resource,gc
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
import numpy as np,pycrfsuite,psutil,soundfile as sf
import benchmark as lib
from cases import CASES
from audio_demo import SAMPLES
ROOT=lib.ROOT.parents[1]
SYSTEM=ROOT.parent/'AuraTranslateEdge_branches/contest_VNG'
OUT=ROOT/'output/tts-format-rerun-2026-11-11';OUT.mkdir(parents=True,exist_ok=True)
(OUT/'audio').mkdir(exist_ok=True)
sys.path.insert(0,str(SYSTEM/'s2s_edge'))
from s2s_edge.identifiers import prepare_for_tts
from s2s_edge.textnorm import normalize as baseline
from s2s_edge.names import VietnameseNames,latin_to_vi
from s2s_edge.tts import split_text,TtsEnKo
ASSETS=SYSTEM/'s2s_edge/assets'
VN=VietnameseNames(str(ASSETS/'cmudict_min.json'),(ASSETS/'piper_vi_words.txt').read_text().splitlines())
METHODS=['B0','M1','M2','M3'];engines={};mlm=None;crf=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def before_norm(text,lang):
 return prepare_for_tts(text,lang,lambda s,l:s,(lambda s:latin_to_vi(s,VN)) if lang=='vi' else None)

def normalize(text,lang,method,trace=False):
 if method=='B0':return baseline(text,lang),[]
 engine=engines[lang];parts=[];traces=[]
 # Sentence-local scoring; all held-out microtests are one sentence.
 for sent in re.split(r'(?<=[.!?])\s+',text):
  ms,spans=engine.prepare(sent)
  if method=='M1':choices=[0]*len(spans)
  elif method=='M2':
   labels=crf.tag([lib.features(ms,i,lang) for i in range(len(ms))])
   choices=[next((j for j,(k,v) in enumerate(cs) if k==labels[i]),0) for i,m,cs in spans]
  else:
   choices=[]
   for i,m,cs in spans:
    unique=list(dict.fromkeys(v for k,v in cs));scores=[mlm.score(sent,m.start(),m.end(),v) for v in unique]
    best=unique[int(np.argmax(scores))];choices.append(next(j for j,(k,v) in enumerate(cs) if v==best))
  parts.append(engine.render(sent,spans,choices))
  if trace:traces.extend([dict(source=m.group(),selected=cs[ch],candidates=cs) for (_,m,cs),ch in zip(spans,choices)])
 return ' '.join(parts),traces

def summary(rows,field):
 out={}
 for method in METHODS:
  rr=[r for r in rows];lat=[x for r in rr for x in r[field][method]['ms']]
  out[method]=dict(passed=sum(r[field][method]['pass'] for r in rr),total=len(rr),p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95)))
  for key in ['lang','level','kind']:
   out[method]['by_'+key]={v:dict(passed=sum(r[field][method]['pass'] for r in rr if r[key]==v),total=sum(r[key]==v for r in rr)) for v in sorted({r[key] for r in rr})}
 return out

def main():
 global engines,mlm,crf
 start=time.perf_counter();proc=psutil.Process();env={'started_local':datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),'system_root':str(SYSTEM),'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'mlm_threads':4,'repeats':5,'acoustic_backend':'public CPU, not fused ONNX / QNN','crf_training_sentences':2304,'crf_training':'synthetic, seed 521, frozen checkpoint; no fitting on held-out labels','mlm_revision':'45c032ab32cc946ad88a166f7cb282f58c753c2e'}
 env['branches']={p.name:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in sorted(SYSTEM.parent.iterdir()) if (p/'.git').exists()}
 env['source_sha256']={str(p.relative_to(SYSTEM)):sha(p) for p in [SYSTEM/'s2s_edge/s2s_edge/textnorm.py',SYSTEM/'s2s_edge/s2s_edge/tts.py',SYSTEM/'s2s_edge/s2s_edge/names.py',SYSTEM/'s2s_edge/s2s_edge/identifiers.py']}
 env['test_cases_sha256']=sha(lib.ROOT/'test_cases.json');env['crf_sha256']=sha(lib.ASSETS/'tagger.crfsuite');env['nemo_commit']=subprocess.check_output(['git','-C','/private/tmp/tts-nemo-source','rev-parse','HEAD'],text=True).strip()
 env['fused_tts_files_present']=False
 env['identifier_policy']='v1: typed phone/code spans spelled before foreign names and all three normalizers'
 env['production_tts_dirty']=True
 env['rss_start_mb']=proc.memory_info().rss/1e6
 crf=pycrfsuite.Tagger();crf.open(str(lib.ASSETS/'tagger.crfsuite'))
 engines={lang:lib.Engine(lang) for lang in ['vi','en']}
 env['rss_wfst_crf_mb']=proc.memory_info().rss/1e6
 mlm=lib.MLM();env['rss_after_mlm_load_mb']=proc.memory_info().rss/1e6;env['mlm_parameters']=mlm.parameters
 print('LOADED',env['started_local'],flush=True)
 rows=[]
 for ci,c in enumerate(CASES):
  row={**c,'before_normalization':before_norm(c['text'],c['lang']),'current_order':{},'raw_format':{}}
  # Untimed warm-up, then randomized fixed-seed interleaving for five repeats.
  vals={m:[] for m in METHODS}
  for rep in range(6):
   order=METHODS.copy();random.Random(521+ci*7+rep).shuffle(order)
   for method in order:
    t=time.perf_counter();source=before_norm(c['text'],c['lang']);output,_=normalize(source,c['lang'],method);dt=(time.perf_counter()-t)*1000
    if rep:vals[method].append(dt)
    row['current_order'][method]=dict(text=output,**{'pass':lib.check(c,output)},ms=vals[method])
  # Single diagnostic pass without prior name respelling; latency not compared.
  for method in METHODS:
   t=time.perf_counter();raw=prepare_for_tts(c['text'],c['lang'],lambda s,l:s);output,_=normalize(raw,c['lang'],method)
   row['raw_format'][method]=dict(text=output,**{'pass':lib.check(c,output)},ms=[(time.perf_counter()-t)*1000])
  rows.append(row)
  (OUT/'regression.partial.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
  if ci%12==11:print('REGRESSION',ci+1,flush=True)
 env['rss_after_regression_mb']=proc.memory_info().rss/1e6
 demos=[]
 for lang,text in SAMPLES.items():
  source=before_norm(text,lang)
  for method in METHODS:
   durations=[]
   for rep in range(4):
    t=time.perf_counter();output,trace=normalize(source,lang,method,trace=rep==3);dt=(time.perf_counter()-t)*1000
    if rep:durations.append(dt)
   demos.append(dict(lang=lang,method=method,input=text,before_normalization=source,text_for_tts=output,normalization_ms=durations,normalization_p50_ms=statistics.median(durations),trace=trace))
   print('DEMO_TEXT',lang,method,round(statistics.median(durations),2),flush=True)
 env['artifact_bytes']={p.name:p.stat().st_size for p in lib.ASSETS.iterdir() if p.is_file()}
 env['far_bytes']={p.name:p.stat().st_size for p in lib.CACHE.glob('*.far')}
 modelroot=Path('/private/tmp/tts-hf/hub/models--distilbert--distilbert-base-multilingual-cased/snapshots')/env['mlm_revision']
 env['mlm_bytes']={p.name:p.stat().st_size for p in modelroot.iterdir() if p.is_file()}
 env['piper_model_bytes']=Path('/private/tmp/tts-piper-model/vi_VN-vais1000-medium.onnx').stat().st_size
 env['supertonic_model_bytes']={p.name:p.stat().st_size for p in Path('/Users/khoa/.cache/supertonic3/onnx').iterdir() if p.suffix=='.onnx'}
 result=dict(environment=env,summary=summary(rows,'current_order'),raw_format_summary=summary(rows,'raw_format'),cases=rows,demos=demos)
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 # Acoustic rendering keeps the current splitter thresholds: VI 26 words, EN 82 chars.
 # Public runtime is used instead of missing fused graph; no NPU latency claim.
 del mlm,engines;gc.collect()
 from piper.voice import PiperVoice
 from supertonic import TTS
 piper=PiperVoice.load('/private/tmp/tts-piper-model/vi_VN-vais1000-medium.onnx')
 supertonic=TTS(model_dir='/Users/khoa/.cache/supertonic3',auto_download=False,intra_op_num_threads=4,inter_op_num_threads=1)
 style=supertonic.get_voice_style('M1');front=TtsEnKo(None,'en',str(ASSETS))
 for d in demos:
  lang=d['lang'];np.random.seed(521);t=time.perf_counter();parts=[];graph_inputs=[]
  chunks=split_text(d['text_for_tts'],26 if lang=='vi' else 82,'word' if lang=='vi' else 'char')
  for i,piece in enumerate(chunks):
   if lang=='vi':
    wav=np.concatenate([c.audio_float_array for c in piper.synthesize(piece)]);sr=piper.config.sample_rate;graph_inputs.append(piece)
   else:
    # Exactly record current graph front-end including supported-character filtering.
    ready=front._clean(front.p._preprocess_text(piece,'en'))[:front.N];graph_inputs.append(ready)
    content=re.sub(r'</?en>','',ready).strip()
    wav,_=supertonic.synthesize(content,voice_style=style,lang='en',total_steps=8,speed=1.,max_chunk_length=240)
    wav=np.asarray(wav).reshape(-1);sr=supertonic.sample_rate
   parts.append(wav)
   if i<len(chunks)-1:parts.append(np.zeros(int(sr*.18),np.float32))
  wav=np.concatenate(parts);file=OUT/'audio'/f'{lang}_{d["method"]}.wav';sf.write(file,wav,sr,subtype='PCM_16')
  d.update(wav_file=str(file),sample_rate=sr,audio_seconds=len(wav)/sr,synthesis_seconds=time.perf_counter()-t,chunks=len(chunks),graph_frontend_inputs=graph_inputs,peak=float(np.max(np.abs(wav))),rms=float(np.sqrt(np.mean(wav**2))),sha256=sha(file))
  assert np.isfinite(wav).all() and d['audio_seconds']>5 and d['rms']>.001
  print('AUDIO',lang,d['method'],round(d['audio_seconds'],2),round(d['synthesis_seconds'],2),flush=True)
  (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 env['elapsed_seconds']=time.perf_counter()-start;env['rss_final_mb']=proc.memory_info().rss/1e6
 # macOS reports ru_maxrss in bytes. Shared process peak cannot be assigned to one model.
 env['process_peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
 env['finished_local']=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 with (OUT/'cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['id','lang','level','kind','input','expected','method','output','pass','p50_ms'])
  for c in rows:
   for method in METHODS:
    z=c['current_order'][method];w.writerow([c['id'],c['lang'],c['level'],c['kind'],c['text'],' | '.join(c['expected']),method,z['text'],z['pass'],statistics.median(z['ms'])])
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=(OUT/'requirements-lock.txt').open('w'))
 print('COMPLETE',json.dumps(result['summary']),flush=True)
if __name__=='__main__':main()

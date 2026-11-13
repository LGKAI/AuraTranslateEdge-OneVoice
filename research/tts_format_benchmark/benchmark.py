"""Reproducible research prototypes, isolated from production pipeline.
M1 deterministic candidate priority; M2 CRF chooses semantic class;
M3 pretrained multilingual MLM ranks WFST candidates (no fine-tuning).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import re,json,time,random,statistics,platform,sys,importlib.util,hashlib,subprocess,logging
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import pynini
from pynini.lib import rewrite,pynutil
from sklearn_crfsuite import CRF
import joblib,psutil
from nemo_text_processing.text_normalization.normalize import Normalizer
from cases import CASES
ROOT=Path(__file__).resolve().parent
ASSETS=ROOT/'artifacts'; ASSETS.mkdir(exist_ok=True)
CACHE=Path(os.environ.get('TTS_FST_CACHE','/private/tmp/tts-fst-cache'))
# Lexical segmentation only. Expansions are produced by FSTs, never this regex.
TOK=re.compile(r'https?://[^\s]+(?<![.,])|[\w.+-]+@[\w.-]+\.[A-Za-z]+|[+]?\d+(?:[ \u00a0]\d{3,4}){2,3}|[A-Za-z]+-?\d+(?:[.-]\d+)*|[+-]?(?:[$€£]\s*)?\d+(?:[.,:/h–-]\d+)*(?:st|nd|rd|th|e-\d+)?(?:\s*(?:AM|PM|am|pm|%|đồng|VND|USD|km|kg|m²|°C))?|[^\W\d_]+(?:[-\x27][^\W\d_]+)*|[^\s]',re.UNICODE)
def tokens(text):return list(TOK.finditer(text))
def shape(s):return re.sub(r'\d','D',s)
def features(ms,i,lang):
 s=ms[i].group(); f={'bias':1.,'lang':lang,'shape':shape(s),'numeric':any(c.isdigit() for c in s),'upper':s.isupper(),'slash':'/' in s,'colon':':' in s,'zero':s.startswith('0')}
 if not f['numeric']: f['word']=s.lower()
 for d in [-4,-3,-2,-1,1,2,3,4]:
  j=i+d
  if 0<=j<len(ms):
   w=ms[j].group().lower(); f[f'w{d}']=shape(w) if any(c.isdigit() for c in w) else w
 return f

def train_crf():
 # Separate training templates; numeric shapes/values may overlap. No CASES labels used.
 contexts={
 'vi':{
 'cardinal':['Cửa hàng có {} sản phẩm','Đếm được {} chiếc ghế','Kho còn {} hộp','Tổng số là {} món'],
 'decimal':['Giá trị thập phân {} được ghi lại','Hệ số thập phân {} là kết quả','Số thực {} cần tính','Kết quả phép đo là {}'],
 'date':['Cuộc họp vào ngày {} nhé','Sinh nhật là ngày {} hàng năm','Ngày giao hàng là {}','Lịch hẹn vào {} tuần tới'],
 'fraction':['Dùng {} cái bánh','Chia {} phần bột','Lấy {} lượng đường','Phân số {} nhỏ hơn một'],
 'telephone':['Hãy gọi số {} ngay','Điện thoại liên hệ {} nhé','Số khẩn cấp {} đang hoạt động','Liên lạc qua số {}'],
 'code':['Mã số bí mật {} được lưu','Mã bưu chính là {}','Mật khẩu truy cập {}','Mã chuyến bay {}'],
 'time':['Đồng hồ chỉ {} rồi','Buổi họp lúc {} nhé','Khởi hành lúc {} sáng','Hẹn gặp lúc {}'],
 'money':['Chi phí là {} thôi','Giá bán {} mỗi món','Thanh toán {} ngay','Số tiền là {}'],
 'measure':['Độ dài là {} nhé','Khoảng cách {} nữa','Cân nặng {} thôi','Đo được {}'],
 'percent':['Tăng {} so với trước','Giảm giá {} hôm nay','Lãi suất {} mỗi năm','Tỷ lệ là {}'],
 'range':['Trong khoảng {} ngày','Từ {} người tham dự','Chờ khoảng {} giờ','Có từ {} hộp'],
 'ordinal':['Lần thứ {} diễn ra','Đứng thứ {} trong lớp','Vị trí thứ {}','Hạng thứ {}'],
 },
 'en':{
 'cardinal':['We counted {} boxes','There were {} chairs','A total of {} items arrived','The store has {} products'],
 'decimal':['The decimal value {} was recorded','The decimal coefficient is {}','The real number {} was calculated','A measurement of {} was recorded'],
 'date':['The meeting is on {} tomorrow','The birthday is on {} every year','The delivery date is {}','An appointment on {} is planned'],
 'fraction':['Use {} of a pie','Take {} of the sugar','A fraction of {} is needed','Divide {} of the cake'],
 'telephone':['Please call number {} now','The emergency number is {}','Telephone contact {} is available','You can phone {} today'],
 'code':['The secret PIN is {}','The postal ZIP code is {}','The password is {}','The flight code is {}'],
 'time':['The clock shows {} now','We leave at {} today','Meet us at {} tomorrow','Departure time is {}'],
 'money':['The cost is {} today','Please pay {} now','A price of {} was listed','The amount is {}'],
 'measure':['A distance of {} remains','The length is {} exactly','A weight of {} was measured','The measurement is {}'],
 'percent':['An increase of {} was observed','Save {} today','The annual rate is {}','The proportion is {}'],
 'range':['Between {} people arrived','Wait about {} days','A range of {} is allowed','Use {} boxes'],
 'ordinal':['This is the {} attempt','The {} runner arrived','My {} visit ended','Choose the {} item'],
 }}
 r=random.Random(521); X=[];Y=[];train=[]
 for lang,bykind in contexts.items():
  for kind,templates in bykind.items():
   for template in templates:
    for _ in range(24):
     a=r.randint(6,28);b=r.randint(2,9);v=r.randint(230,889)
     vals={'cardinal':str(v),'decimal':f'{a}{"," if lang=="vi" else "."}{b}', 'date':f'{a}/{b}', 'fraction':f'{b}/{a}', 'telephone':str(v)+str(r.randint(100,999)), 'code':str(r.randint(10000,99999)), 'time':f'{r.randint(1,22)}:{r.randint(10,59)}', 'money':f'{v} đồng' if lang=='vi' else f'${v}', 'measure':f'{a} km', 'percent':f'{a}%', 'range':f'{b}-{a}', 'ordinal':str(a) if lang=='vi' else f'{a}th'}
     target=vals[kind]; left,right=template.split('{}');text=left+target+right;ms=tokens(text)
     yy=[kind if m.start()<len(left)+len(target) and m.end()>len(left) else 'plain' for m in ms]
     X.append([features(ms,i,lang) for i in range(len(ms))]);Y.append(yy);train.append(dict(lang=lang,text=text,target=target,label=kind))
 model=CRF(algorithm='lbfgs',c1=.05,c2=.05,max_iterations=120,all_possible_transitions=True,model_filename=str(ASSETS/'tagger.crfsuite'),keep_tempfiles=True)
 t=time.perf_counter();model.fit(X,Y); elapsed=time.perf_counter()-t
 (ROOT/'train_cases.json').write_text(json.dumps(train,ensure_ascii=False,indent=2))
 return model,{'sentences':len(train),'seconds':elapsed,'token_accuracy_train':sum(a==b for yy,pp in zip(Y,model.predict(X)) for a,b in zip(yy,pp))/sum(map(len,Y))}

class Engine:
 def __init__(self,lang):
  self.lang=lang; self.n=Normalizer(input_case='cased',lang=lang,cache_dir=str(CACHE),post_process=True)
  pre=f'nemo_text_processing.text_normalization.{lang}'
  from importlib import import_module
  C=lambda name,side='taggers':getattr(import_module(pre+'.'+side+'.'+name),name.title().replace('_','')+'Fst')
  cardinal=C('cardinal')(deterministic=True)
  self.graphs={}
  self.graphs['cardinal']=cardinal.fst
  ordinal=C('ordinal')(cardinal=cardinal,deterministic=True)
  self.graphs['ordinal']=ordinal.fst
  self.graphs['decimal']=C('decimal')(cardinal=cardinal,deterministic=True).fst
  if lang=='en':
   self.graphs['date']=C('date')(cardinal=cardinal,deterministic=True).fst
   self.graphs['fraction']=C('fraction')(cardinal=cardinal,deterministic=True).fst
  else:
   self.graphs['date']=C('date')(cardinal=cardinal,deterministic=True).fst
   self.graphs['fraction']=C('fraction')(cardinal=cardinal,deterministic=True).fst
  self.graphs['time']=C('time')(cardinal=cardinal,deterministic=True).fst
  # Generic reusable unbounded digit spelling / alphanumeric spelling FSTs.
  ds='không một hai ba bốn năm sáu bảy tám chín'.split() if lang=='vi' else 'zero one two three four five six seven eight nine'.split()
  letters='a bê xê đê e ép giê hát i giây ca e-lờ em en o pê quy e-rờ ét tê u vê đúp-vê ích i-dài dét'.split() if lang=='vi' else list('abcdefghijklmnopqrstuvwxyz')
  self.digit=pynini.string_map([(str(i),w+' ') for i,w in enumerate(ds)])
  letter=pynini.string_map([(c,w+' ') for c,w in zip('abcdefghijklmnopqrstuvwxyz',letters)]+[(c.upper(),w+' ') for c,w in zip('abcdefghijklmnopqrstuvwxyz',letters)])
  sep=pynini.union(*[pynutil.delete(s) for s in [' ','\u00a0','-','(',')']])
  plus=pynini.cross('+','cộng ' if lang=='vi' else 'plus ')
  self.spell_digits=(plus.ques+(self.digit|sep).plus).optimize()
  self.spell_code=(self.digit|letter|sep).plus.optimize()
  for k,g in self.graphs.items():g.write(str(ASSETS/f'{lang}_{k}.fst'))
  self.spell_digits.write(str(ASSETS/f'{lang}_digits.fst'));self.spell_code.write(str(ASSETS/f'{lang}_code.fst'))
 def verbalize(self,tag):
  self.n.parser('tokens { '+tag+' }'); ts=self.n.parser.parse()
  for perm in self.n.generate_permutations(ts):
   lattice=self.n.find_verbalizer(pynini.escape(perm))
   if lattice.num_states():return Normalizer.select_verbalizer(lattice).replace('\u00a0',' ')
  return None
 def candidates(self,s):
  out=[]
  try: default=self.n.normalize(s);tag=re.search(r'tokens\s*\{\s*(\w+)\s*\{',Normalizer.select_tag(self.n.find_tags(pynini.escape(s))))
  except Exception: default=s;tag=None
  out.append((tag.group(1) if tag else 'plain',default))
  for k,g in self.graphs.items():
   probes=[s]+(['ngày '+s] if self.lang=='vi' and k=='date' else [])
   for probe in probes:
    try:
     tagged=rewrite.top_rewrites(pynini.escape(probe),g,2)
     for z in tagged:
      v=self.verbalize(z)
      if v:out.append((k,v))
    except (rewrite.Error,pynini.FstOpError):pass
  for kind,g in [('telephone',self.spell_digits),('code',self.spell_code)]:
   try: out.append((kind,rewrite.one_top_rewrite(pynini.escape(s),g).strip()))
   except (rewrite.Error,pynini.FstOpError):pass
  # Always include unchanged candidate. Model may preserve words but cannot validate calendars.
  out.append(('plain',s))
  uniq=[]
  for k,v in out:
   v=re.sub(r'\s+',' ',v).strip()
   if (k,v) not in uniq:uniq.append((k,v))
  # Paper-inspired pruning: prefer fully verbalized candidates over residual digits.
  normalized=[(k,v) for k,v in uniq if not any(ch.isdigit() for ch in v)]
  return normalized or uniq
 def prepare(self,text):
  ms=tokens(text);spans=[]
  for i,m in enumerate(ms):
   if any(c.isdigit() for c in m.group()) or '@' in m.group():spans.append((i,m,self.candidates(m.group())))
  return ms,spans
 def render(self,text,spans,choices):
  out=text
  for (_,m,cs),choice in reversed(list(zip(spans,choices))):
   v=cs[choice][1]
   # Avoid duplicated lexical boundary prefix introduced by a date verbalizer.
   if self.lang=='vi' and text[:m.start()].rstrip().lower().endswith('ngày') and v.startswith('ngày '):v=v[5:]
   out=out[:m.start()]+v+out[m.end():]
  return out

class MLM:
 def __init__(self):
  import torch
  from transformers import AutoTokenizer,AutoModelForMaskedLM
  self.torch=torch;torch.set_num_threads(4)
  self.name='distilbert/distilbert-base-multilingual-cased'
  self.tok=AutoTokenizer.from_pretrained(self.name,local_files_only=True,revision='45c032ab32cc946ad88a166f7cb282f58c753c2e')
  self.model=AutoModelForMaskedLM.from_pretrained(self.name,local_files_only=True,revision='45c032ab32cc946ad88a166f7cb282f58c753c2e').eval()
  self.parameters=sum(x.numel() for x in self.model.parameters())
 def score(self,text,a,b,v):
  # Pseudo-log-likelihood of only replacement tokens; full sentence is context.
  full=text[:a]+v+text[b:];end=a+len(v)
  enc=self.tok(full,return_offsets_mapping=True,return_tensors='pt',truncation=True,max_length=128)
  offsets=enc.pop('offset_mapping')[0].tolist()
  positions=[i for i,(x,y) in enumerate(offsets) if y>a and x<end and y>x]
  if not positions:return -1e9
  ids=enc['input_ids'].repeat(len(positions),1);mask=enc['attention_mask'].repeat(len(positions),1)
  truth=ids[range(len(positions)),positions].clone();ids[range(len(positions)),positions]=self.tok.mask_token_id
  with self.torch.inference_mode():
   logits=self.model(input_ids=ids,attention_mask=mask).logits
   lp=logits[range(len(positions)),positions].log_softmax(-1)
   return lp[range(len(positions)),truth].mean().item()

# Canonicalization only ignores punctuation, case and declared regional variants.
# Digits remain digits, so unresolved input fails against written-word gold.
def canon(s,lang):
 s=s.lower().replace('ngàn','nghìn').replace('linh','lẻ')
 if lang=='vi':s=s.replace('mươi tư','mươi bốn')
 if lang=='en':s=re.sub(r'\band\b',' ',s)
 s=re.sub(r'[^\w\s]',' ',s);return ' '.join(s.split())
def check(c,out):return any(canon(out,c['lang'])==canon(g,c['lang']) for g in c['expected'])

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=3);p.add_argument('--no-mlm',action='store_true');args=p.parse_args()
 start=time.perf_counter();proc=psutil.Process();initial=proc.memory_info().rss
 spec=importlib.util.spec_from_file_location('old',str(ROOT.parents[1]/'s2s_edge/s2s_edge/textnorm.py'));old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 print('TRAIN_CRF',flush=True);crf,training=train_crf();print(training,flush=True)
 engines={};init={}
 for lang in ['vi','en']:
  t=time.perf_counter();engines[lang]=Engine(lang);init[lang]=time.perf_counter()-t
 print('GRAMMARS_READY',init,flush=True)
 before_lm=proc.memory_info().rss;mlm=None
 if not args.no_mlm:
  t=time.perf_counter();mlm=MLM();init['mlm']=time.perf_counter()-t
 after_lm=proc.memory_info().rss
 results=[]
 for ci,c in enumerate(CASES):
  text=c['text'];lang=c['lang'];e=engines[lang];times=defaultdict(list);outputs={};meta={}
  # First iteration is recorded for correctness but excluded from warm latency.
  for rep in range(args.repeats+1):
   t=time.perf_counter();outputs['B0']=old.normalize(text,lang);dt=time.perf_counter()-t
   if rep:times['B0'].append(dt*1000)
   # Each method includes fresh segmentation/candidate generation, no output cache.
   for method in ['M1','M2']+(['M3'] if mlm else []):
    t=time.perf_counter();ms,spans=e.prepare(text)
    if method=='M1':choices=[0]*len(spans)
    elif method=='M2':
     labels=crf.predict_single([features(ms,i,lang) for i in range(len(ms))]);choices=[]
     for i,m,cs in spans:choices.append(next((j for j,(k,v) in enumerate(cs) if k==labels[i]),0))
    else:
     choices=[]
     for i,m,cs in spans:
      unique=list(dict.fromkeys(v for k,v in cs));scores=[mlm.score(text,m.start(),m.end(),v) for v in unique]
      best=unique[int(np.argmax(scores))];choices.append(next(j for j,(k,v) in enumerate(cs) if v==best))
    outputs[method]=e.render(text,spans,choices);dt=time.perf_counter()-t
    if rep:times[method].append(dt*1000)
    if method=='M1':meta={'spans':[dict(source=m.group(),options=cs) for i,m,cs in spans]}
  row={**c,'outputs':{k:{'text':v,'pass':check(c,v),'latencies_ms':times[k]} for k,v in outputs.items()},**meta}
  results.append(row)
  (ROOT/'results.partial.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
  print(ci+1,c['id'],{k:(o['pass'],round(statistics.median(o['latencies_ms']),2)) for k,o in row['outputs'].items()},flush=True)
 summary={}
 for method in results[0]['outputs']:
  lat=[t for c in results for t in c['outputs'][method]['latencies_ms']]
  summary[method]=dict(passed=sum(c['outputs'][method]['pass'] for c in results),total=len(results),p50_ms=float(np.percentile(lat,50)),p95_ms=float(np.percentile(lat,95)),by_lang={},by_level={},by_kind={})
  for field,key in [('lang','by_lang'),('level','by_level'),('kind','by_kind')]:
   for value in sorted({c[field] for c in results}):
    rows=[c for c in results if c[field]==value];summary[method][key][value]={'passed':sum(c['outputs'][method]['pass'] for c in rows),'total':len(rows)}
 # Size artifacts != total runtime. Include dependency/runtime footprints separately.
 packages=Path(sys.prefix)/'lib/python3.11/site-packages'
 def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.is_dir() else p.stat().st_size
 env=dict(python=sys.version,platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),threads_mlm=4,repo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),nemo_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/private/tmp/tts-nemo-source').decode().strip(),initial_rss_mb=initial/1e6,rss_before_lm_mb=before_lm/1e6,rss_after_lm_mb=after_lm/1e6,final_rss_mb=proc.memory_info().rss/1e6,init_seconds=init,repeats=args.repeats,elapsed_seconds=time.perf_counter()-start,training=training,artifact_bytes={f.name:f.stat().st_size for f in ASSETS.iterdir() if f.is_file()},nemo_far_bytes={f.name:f.stat().st_size for f in CACHE.glob('*.far')},package_bytes={n:size(packages/n) for n in ['pynini','pywrapfst','sklearn','sklearn_crfsuite','torch','transformers'] if (packages/n).exists()},environment_bytes=size(Path(sys.prefix)),mlm_parameters=mlm.parameters if mlm else None)
 if mlm:
  hf=Path(os.environ.get('HF_HOME','/private/tmp/tts-hf'));env['mlm_files']={str(f.relative_to(hf)):f.stat().st_size for f in hf.rglob('*') if f.is_file() and f.name in ['model.safetensors','pytorch_model.bin','config.json','tokenizer.json','vocab.txt']}
 report=dict(environment=env,summary=summary,cases=results)
 (ROOT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('COMPLETE',summary,flush=True)
 subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open(ROOT/'requirements-lock.txt','w'))
if __name__=='__main__':main()

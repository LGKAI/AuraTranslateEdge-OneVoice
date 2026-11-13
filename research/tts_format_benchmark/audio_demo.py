"""Two dense held-out paragraphs; three normalization methods; fixed TTS voices.
Uses public CPU Piper vais1000 and locally cached official Supertonic 3, not fused QNN graphs.
"""
import os,json,re,time,sys,hashlib,wave
from pathlib import Path
import numpy as np
import soundfile as sf
import pycrfsuite
from benchmark import Engine,MLM,features,tokens,ROOT
OUT=ROOT.parents[1]/'output/audio/tts-format-demo';OUT.mkdir(parents=True,exist_ok=True)
SAMPLES={
'vi':'Cuộc họp lúc 3:30 chiều; chuyến bay VN123 khởi hành lúc 14h45 ngày 15/11/2026. Gọi 0912 345 678; mã bưu chính 67297. Patricia Johnson muốn check-out phòng 204. Sinh năm 1990; tiền phòng 50.000 đồng, giảm 15%. Nhiệt độ 37,5°C; hệ số 0,125; quãng đường 10 km. Ăn 1/4 chiếc bánh; hẹn ngày 1/4. Chờ 3-5 ngày; ngoài trời -5°C. Email a1@example.com; phiên bản 2.10.3.',
'en':'The meeting is at 3:30 PM on November 15, 2026; flight VN123 leaves at 14:45. Call 555-0123; ZIP code 67297. Patricia Johnson will check-out of room 204. Born in 1990, she pays $15.50 with a 15% discount. The temperature is 37.5°C; coefficient 0.125; distance 10 km. Eat 1/4 of the cake; meet on 1/4. Wait 3-5 days at -5°C. This is the 21st visit; email a1@example.com; version 2.10.3.'}
NAMES={'M1':'WFST deterministic','M2':'CRF + WFST','M3':'WFST + multilingual DistilBERT MLM'}

def main():
 tagger=pycrfsuite.Tagger();tagger.open(str(ROOT/'artifacts/tagger.crfsuite'))
 engines={l:Engine(l) for l in SAMPLES};mlm=MLM()
 # Import just the existing name handling, without importing the production registry.
 sys.path.insert(0,str(ROOT.parents[1]/'s2s_edge'))
 from s2s_edge.names import VietnameseNames,latin_to_vi
 assets=ROOT.parents[1]/'s2s_edge/assets'
 vn=VietnameseNames(str(assets/'cmudict_min.json'),(assets/'piper_vi_words.txt').read_text().splitlines())
 rows=[]
 for lang,text in SAMPLES.items():
  (OUT/f'{lang}_input.txt').write_text(text,encoding='utf-8');e=engines[lang]
  for method in NAMES:
   t=time.perf_counter();parts=[];trace=[]
   # Sentence-local context prevents 128-token truncation of the long paragraph.
   for sent in re.split(r'(?<=[.!?])\s+',text):
    ms,spans=e.prepare(sent);choices=[]
    if method=='M1':choices=[0]*len(spans)
    elif method=='M2':
     labels=tagger.tag([features(ms,i,lang) for i in range(len(ms))])
     choices=[next((j for j,(k,v) in enumerate(cs) if k==labels[i]),0) for i,m,cs in spans]
    else:
     for i,m,cs in spans:
      unique=list(dict.fromkeys(v for k,v in cs));scores=[mlm.score(sent,m.start(),m.end(),v) for v in unique]
      best=unique[int(np.argmax(scores))];choices.append(next(j for j,(k,v) in enumerate(cs) if v==best))
    parts.append(e.render(sent,spans,choices))
    trace.extend([{'source':m.group(),'chosen':cs[ch],'candidates':cs} for (_,m,cs),ch in zip(spans,choices)])
   normalized=' '.join(parts);t_norm=time.perf_counter()-t
   # Same legacy fallback for all three Vietnamese outputs; no manual correction.
   actual=latin_to_vi(normalized,vn).lower() if lang=='vi' else normalized
   row=dict(lang=lang,method=method,method_name=NAMES[method],input=text,normalized=normalized,text_for_tts=actual,normalization_seconds=t_norm,trace=trace)
   rows.append(row);(OUT/f'{lang}_{method}_normalized.txt').write_text(normalized+'\n\nTEXT FOR TTS:\n'+actual,encoding='utf-8')
   print('NORMALIZED',lang,method,round(t_norm,3),normalized,flush=True)
 (OUT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 # Release MLM before loading acoustic models.
 del mlm,engines
 import gc;gc.collect()
 from piper.voice import PiperVoice
 from supertonic import TTS
 print('LOADING_PIPER',flush=True)
 piper=PiperVoice.load('/private/tmp/tts-piper-model/vi_VN-vais1000-medium.onnx')
 print('LOADING_SUPERTONIC',flush=True)
 supertonic=TTS(model_dir='/Users/khoa/.cache/supertonic3',auto_download=False,intra_op_num_threads=4,inter_op_num_threads=1)
 style=supertonic.get_voice_style('M1')
 for row in rows:
  lang=row['lang'];method=row['method'];t=time.perf_counter();np.random.seed(521)
  if lang=='vi':
   pieces=[]
   for sent in re.split(r'(?<=[.!?])\s+',row['text_for_tts']):
    for chunk in piper.synthesize(sent):pieces.append(chunk.audio_float_array)
    pieces.append(np.zeros(int(.25*piper.config.sample_rate),dtype=np.float32))
   wav=np.concatenate(pieces);sr=piper.config.sample_rate
  else:
   wav,dur=supertonic.synthesize(row['text_for_tts'],voice_style=style,lang='en',total_steps=8,speed=1.,max_chunk_length=240)
   wav=np.asarray(wav).reshape(-1);sr=supertonic.sample_rate
  path=OUT/f'{lang}_{method}.wav';sf.write(path,wav,sr,subtype='PCM_16')
  row.update(wav_file=str(path),sample_rate=sr,audio_seconds=len(wav)/sr,synthesis_seconds=time.perf_counter()-t,rms=float(np.sqrt(np.mean(wav**2))),peak=float(np.max(np.abs(wav))),tts_engine='Piper vi_VN-vais1000-medium CPU' if lang=='vi' else 'Supertonic 3 M1 CPU, 8 steps, speed 1.0',seed=521)
  assert np.isfinite(wav).all() and row['audio_seconds']>5 and row['rms']>.001
  print('AUDIO_READY',lang,method,row['audio_seconds'],row['synthesis_seconds'],flush=True)
  (OUT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 (OUT/'README.md').write_text('# Two-case audio comparison\n\nThree research prototypes, not paper reproductions. Same voice and settings for each language. No manual correction of normalized output. English Supertonic 3 public CPU, Vietnamese Piper vais1000 public CPU, not the repository fused NPU graphs. The CRF was trained on 2,304 synthetic template sentences; DistilBERT is pretrained and not fine-tuned. Vietnamese names use the repository legacy transliteration fallback after normalization. Normalization uses sentence-local context. See manifest.json for all intermediate texts and parameters.\n\nCase VI includes hours, date, phone, flight code, postal code, English names, hyphenated word, room, year, money, percent, decimals, units, fraction/date ambiguity, range, negative number, email, version. Case EN adds ordinal 21st and English money/date conventions. This is a listening demo, not a statistically representative accuracy evaluation.\n',encoding='utf-8')
if __name__=='__main__':main()

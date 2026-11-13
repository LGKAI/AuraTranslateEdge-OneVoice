from pathlib import Path
import json,hashlib,subprocess,difflib,shutil
import numpy as np,soundfile as sf
r=Path(__file__).resolve().parents[1];out=r/'results/neural_fix';final=out/'final'
cases=json.loads((r/'results/reference_inputs/cases.json').read_text())
lines=['S2S sau bổ sung chuẩn hóa học máy - 12/11/2026','ASR/MT/TTS: graph CPU tích hợp từ các nhánh; chuẩn hóa: model cục bộ trên Apple Metal.','Input speech là fixture tổng hợp; không truyền văn bản tham chiếu vào ASR/MT.',''];summary=[]
for c in cases:
 key=c['src']+'_to_'+c['tgt'];d=json.loads((final/(key+'.json')).read_text());old=json.loads((r/'results/rerun_2026-11-12'/(key+'.json')).read_text())
 assert [z['asr_raw'] for z in d['segments']]==[z['asr_raw'] for z in old['segments']], 'ASR differs from baseline'
 assert d['neural_normalization'] and d['segments'] and not d['fallbacks']
 w,s=sf.read(final/'audio'/(key+'.wav'));assert len(w)>0 and np.isfinite(w).all() and abs(w).max()>1e-4
 assert all(not any(t.isdecimal() for t in z['tts_text']) for z in d['segments'])
 for t in d['normalization_trace']:
  if t['stage']=='asr_segmentation':assert t['content_preserved']
 source_wav=r/c['input_wav']; target_wav=final/'audio'/Path(c['input_wav']).name
 if source_wav.resolve()!=target_wav.resolve():shutil.copy2(source_wav,target_wav)
 lines.extend([key,'INPUT THAM CHIẾU: '+c['written'],'LỜI DÙNG ĐỂ TẠO INPUT AUDIO: '+c['spoken'],'INPUT WAV: '+d['input']])
 for z in d['segments']:
  lines.extend(['ASR THÔ: '+z['asr_raw'],'SAU CHUẨN HÓA NGUỒN: '+z['asr'],'BẢN DỊCH: '+z['mt'],'TEXT THỰC TẾ VÀO TTS: '+z['tts_text'],'THỜI GIAN: '+json.dumps(z['timing'])])
 lines.extend(['OUTPUT WAV: '+d['output'],''])
 summary.append({'direction':key,'same_raw_asr_as_baseline':True,'source_audio_seconds':sum(z['timing']['audio_in_s'] for z in d['segments']),'output_seconds':len(w)/s,'processing_seconds':sum(z['timing']['total_s'] for z in d['segments']),'wall_including_load_seconds':d['wall_including_load_s'],'peak':float(abs(w).max()),'rms':float(np.sqrt(np.mean(w*w))),'no_residual_digits_in_tts_text':True})
(final/'transcript.txt').write_text('\n'.join(lines));(final/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
manifest={};patch=[]
for p in sorted((r/'s2s_edge/s2s_edge').glob('*.py')):
 manifest[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest();source='s2s_edge/s2s_edge/'+p.name
 q=subprocess.run(['git','-C',str(r),'show','abe8e30f33228fa3189025a935113c219b8f4ede:'+source],capture_output=True);old=q.stdout.decode() if q.returncode==0 else ''
 if old!=p.read_text():patch.extend(difflib.unified_diff(old.splitlines(True),p.read_text().splitlines(True),fromfile='a/'+source,tofile='b/'+source))
(out/'implementation.patch').write_text(''.join(patch));(out/'code_sha256.json').write_text(json.dumps(manifest,indent=2))
models={}
for key,p in [('qwen_segmentation',r/'models/source/qwen3-4b-instruct-mlx4/model.safetensors'),('vi_decoder',r/'models/source/byt5-vi-normalization/model.safetensors'),('en_decoder',r/'models/source/nemo-en-tn/hf/model.safetensors'),('identifier_tagger',r/'models/normalization/identifier.crfsuite')]:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 models[key]={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
(out/'normalization_models.json').write_text(json.dumps(models,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2));print('Normalization weights bytes:',sum(d['bytes'] for d in models.values()))

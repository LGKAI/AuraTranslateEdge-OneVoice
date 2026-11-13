"""Real HTTP audio -> actual models integration check, no mocked transcripts."""
import json,time,urllib.request,urllib.error
from pathlib import Path
r=Path(__file__).resolve().parents[1];base='http://localhost:8766';jobs=[]
for src,name in [('en','en_to_vi_special_input.wav'),('vi','vi_to_en_special_input.wav')]:
 p=r/'results/neural_fix/final/audio'/name
 req=urllib.request.Request(base+'/api/jobs?src='+src,data=p.read_bytes(),headers={'Content-Type':'audio/wav'},method='POST')
 d=json.load(urllib.request.urlopen(req));jobs.append(d['id']);print('SUBMITTED',src,d['id'],flush=True)
for key in jobs:
 deadline=time.monotonic()+240
 while time.monotonic()<deadline:
  d=json.load(urllib.request.urlopen(base+'/api/jobs/'+key))
  if d['status'] in ('done','error'):break
  time.sleep(1)
 assert d['status']=='done',d
 audio=urllib.request.urlopen(base+d['output_url']).read();assert audio.startswith(b'RIFF') and len(audio)>1000
 print('DONE',d['src'],d['tgt'],len(audio),d['asr'],d['translation'],flush=True)
try:
 urllib.request.urlopen(urllib.request.Request(base+'/api/jobs?src=en',data=b'not audio',method='POST'))
 raise AssertionError('Invalid audio accepted')
except urllib.error.HTTPError as e:assert e.code==400
(r/'results/mic_web/api_check.json').write_text(json.dumps({'jobs':jobs,'both_directions':'real audio ASR/MT/TTS completed','invalid_audio_rejected':True},indent=2))
print('API_CHECK_PASSED',flush=True)

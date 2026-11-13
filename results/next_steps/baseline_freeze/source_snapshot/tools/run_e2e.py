import sys,json,time,pathlib,os,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'s2s_edge'))
from s2s_edge.registry import Registry
from s2s_edge.pipeline import Translator
from s2s_edge.audio import load_audio,save_wav,concat_results
src,tgt=sys.argv[1:3]
inputs=ROOT/'results/neural_fix/final/audio'
p=pathlib.Path(sys.argv[3]) if len(sys.argv)>3 else inputs/('vi_to_en_special_input.wav' if src=='vi' else 'en_to_vi_special_input.wav')
out=pathlib.Path(os.environ.get('S2S_RESULT_DIR',str(ROOT/'results/e2e')))
(out/'audio').mkdir(parents=True,exist_ok=True)
if os.environ.get('S2S_NEURAL_TN','0')=='1':
 audit=out/f'{src}_to_{tgt}_normalization.jsonl';audit.write_text('')
 os.environ.setdefault('S2S_TN_AUDIT',str(audit))
start=time.perf_counter()
r=Registry(str(ROOT/'models/models.cpu.json'),'cpu')
t=Translator(r,src,tgt)
raw_asr=[]
class RecordAsr:
 def __init__(self,model):self.model=model
 def __getattr__(self,key):return getattr(self.model,key)
 def __call__(self,wav):
  text=self.model(wav);raw_asr.append(text);return text
t.e._asr[src]=RecordAsr(t.asr)
print('LOADED',src,tgt,time.perf_counter()-start,flush=True)
results=[]
from s2s_edge.audio import segment,SR
w=load_audio(str(p))
for a,b in segment(w,max_s=min(t.asr.max_s,12.0)):
 z=t.translate_segment(w[a:b]); z['span']=[a/SR,b/SR];z['asr_raw']=raw_asr[-1]
 print('SEGMENT',z['asr'],'=>',z['mt'],z['timing'],flush=True)
 results.append(z)
audio,sr=concat_results(results); path=out/'audio'/f'{src}_to_{tgt}.wav';save_wav(str(path),audio,sr)
data={'src':src,'tgt':tgt,'input':str(p),'output':str(path),'backend':'ONNX Runtime CPU','wall_including_load_s':time.perf_counter()-start,'decoder':getattr(t.mt,'active',None),'fallbacks':getattr(t.mt,'failures',[]),'segments':[{k:v for k,v in z.items() if k!='wav'} for z in results]}
data['neural_normalization']=os.environ.get('S2S_NEURAL_TN','0')=='1'
if data['neural_normalization']:
 from s2s_edge.neural_textnorm import get
 data['normalization_model']=str(get().path)
 data['normalization_device']=get().device
 data['normalization_trace']=get().records
(out/f'{src}_to_{tgt}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print('DONE',path,flush=True)

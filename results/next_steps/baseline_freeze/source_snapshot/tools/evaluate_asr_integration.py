"""Actual audio -> ASR -> MT -> hybrid TTS before/after selective ASR integration."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'s2s_edge'))
os.environ['S2S_NEURAL_TN']='0'
os.environ['S2S_TTS_FORMAT']='hybrid'
from s2s_edge.registry import Registry
from s2s_edge.conversation import Engine
from s2s_edge.audio import load_audio,save_wav,segment,concat_results,SR
import numpy as np
mode=sys.argv[1];assert mode in ('before','after')
out=ROOT/'results/asr_integration'/mode;out.mkdir(parents=True,exist_ok=True)
reg=Registry(str(ROOT/'models/models.cpu.json'),'cpu');engine=Engine(reg);rows=[]
for src,tgt,filename in [('vi',t,'vi_to_en_special_input.wav') for t in ('en','ko','zh')]+[('en','vi','en_to_vi_special_input.wav'),('ko','vi','vi_to_ko.wav'),('zh','vi','vi_to_zh.wav')]:
    path=ROOT/'results/e2e/audio'/filename;wav=load_audio(str(path));start=time.perf_counter();results=[]
    try:
        for a,b in segment(wav,max_s=min(engine.asr(src).max_s,12.0)):
            z=engine.translate_segment(wav[a:b],src,tgt);z['span']=[a/SR,b/SR];results.append(z)
        audio,sr=concat_results(results);assert len(audio) and np.isfinite(audio).all()
        target=out/(src+'_to_'+tgt+'.wav');save_wav(str(target),audio,sr)
        row={'src':src,'tgt':tgt,'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output':str(target),'output_seconds':len(audio)/sr,'seconds':time.perf_counter()-start,'status':'ok','segments':[{k:v for k,v in z.items() if k!='wav'} for z in results]}
    except Exception as exc:
        import traceback;traceback.print_exc();row={'src':src,'tgt':tgt,'status':'error','error':str(exc)}
    rows.append(row);(out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(mode,src,tgt,row['status'],round(row.get('seconds',0),2),flush=True)
print('COMPLETE',mode,flush=True)

import numpy as np,soundfile as sf,json
from pathlib import Path
import mlx_whisper
r=Path(__file__).resolve().parents[1];w,sr=sf.read(r.parent/'en-vi/BTC/EV-2748_in.wav',dtype='float32');gain=.5/float(np.max(np.abs(w)))
x=mlx_whisper.transcribe(w*gain,path_or_hf_repo='/private/tmp/onevoice-audit-whisper',language='en',temperature=0.0,condition_on_previous_text=False,verbose=None)
d={'id':'EV-2748','original_peak':float(np.max(abs(w))),'original_rms':float(np.sqrt(np.mean(w*w))),'zero_fraction':float(np.mean(w==0)),'gain':gain,'independent_asr_after_peak_normalization':x['text'],'note':'Gain only, source file unchanged. Independent ASR is not human gold.'}
(r/'results/en_vi_audit/quiet_input.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print(d)

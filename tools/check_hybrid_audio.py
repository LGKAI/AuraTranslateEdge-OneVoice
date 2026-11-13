"""Independent local ASR cross-check of newly synthesized audio; not human MOS."""
import json
import math
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import mlx_whisper
ROOT = Path(__file__).resolve().parents[1]
files = [(lang, ROOT/'results/multilingual_tts_format/hybrid'/(lang+'.wav')) for lang in ('vi','en','ko','zh')]
files += [('vi', ROOT/'results/en_vi_format_replay/hybrid'/(key+'_out.wav')) for key in ('EV-3775','EV-5301','EV-0358')]
rows = []
for lang, p in files:
    w, sr = sf.read(p, dtype='float32')
    if w.ndim > 1: w = w.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000); w = resample_poly(w, 16000//g, sr//g).astype(np.float32)
    start = time.perf_counter()
    result = mlx_whisper.transcribe(w, path_or_hf_repo='/private/tmp/onevoice-audit-whisper', language=lang,
                                   temperature=0.0, condition_on_previous_text=False, verbose=None)
    row = {'lang':lang, 'file':str(p), 'independent_asr':result['text'].strip(), 'seconds':time.perf_counter()-start}
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
(ROOT/'results/hybrid_format/audio_crosscheck.json').write_text(json.dumps({'method':'Whisper small local; forced language; no reference prompts; ASR hypotheses are not human ground truth','cases':rows},ensure_ascii=False,indent=2))

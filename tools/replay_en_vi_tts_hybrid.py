"""Replay all existing MT texts into actual TTS, retaining original recordings."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'s2s_edge'))
os.environ['S2S_TTS_FORMAT'] = 'hybrid'
from s2s_edge.registry import Registry
from s2s_edge.audio import save_wav
out = ROOT/'results/hybrid_format/en_vi_tts'
out.mkdir(parents=True, exist_ok=True)
reg = Registry(str(ROOT/'models/models.cpu.json'), 'cpu')
model = reg.tts('vi')
rows = []
for p in sorted((ROOT.parent/'en-vi/BTC').glob('*.json')):
    d = json.loads(p.read_text()); text = d['mt']; start = time.perf_counter()
    row = {'id':p.stem, 'mt_input':text, 'scope':'actual TTS replay of saved MT; not re-running ASR/MT'}
    if not text.strip():
        row['status'] = 'skipped_empty_mt'
    else:
        try:
            wav, sr = model.synthesize(text)
            assert len(wav) and np.isfinite(wav).all()
            target = out/(p.stem+'.wav'); save_wav(str(target), wav, sr)
            row.update(status='ok', output=str(target), tts_text=model.tts_input_text,
                       trace=model.format_trace, output_seconds=len(wav)/sr,
                       seconds=time.perf_counter()-start)
        except Exception as exc:
            row.update(status='error', error=str(exc))
    rows.append(row)
    (out/(p.stem+'.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2))
    print(row['id'], row['status'], flush=True)
(out/'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))

"""Real EN->VI S2S replay; records actual TTS text, never supplies a transcript."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 's2s_edge'))
from s2s_edge.registry import Registry
from s2s_edge.pipeline import Translator
from s2s_edge.audio import load_audio, save_wav, concat_results, segment, SR

mode = sys.argv[1]
assert mode in ('default', 'neural', 'hybrid')
os.environ['S2S_NEURAL_TN'] = '1' if mode in ('neural', 'hybrid') else '0'
os.environ['S2S_TTS_FORMAT'] = {'default':'legacy', 'neural':'neural', 'hybrid':'hybrid'}[mode]
out = ROOT / 'results/en_vi_format_replay' / mode
out.mkdir(parents=True, exist_ok=True)
os.environ['S2S_TN_AUDIT'] = str(out / 'normalization.jsonl')
reg = Registry(str(ROOT / 'models/models.cpu.json'), 'cpu')
start = time.perf_counter()
t = Translator(reg, 'en', 'vi')
print('LOADED', mode, round(time.perf_counter()-start, 2), flush=True)
rows = []
for key in ['EV-3775', 'EV-5301', 'EV-0358']:
    path = ROOT.parent / 'en-vi/BTC' / (key + '_in.wav')
    w = load_audio(str(path))
    begin = time.perf_counter()
    results = []
    try:
        for a, b in segment(w, max_s=min(t.asr.max_s, 12.0)):
            z = t.translate_segment(w[a:b])
            z['span'] = [a/SR, b/SR]
            results.append(z)
        audio, sr = concat_results(results)
        target = out / (key + '_out.wav')
        save_wav(str(target), audio, sr)
        row = dict(id=key, mode=mode, input=str(path),
                   input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   output=str(target), sample_rate=sr, input_seconds=len(w)/SR,
                   output_seconds=len(audio)/sr, elapsed_seconds=time.perf_counter()-begin,
                   decoder=getattr(t.mt, 'active', None),
                   fallbacks=getattr(t.mt, 'failures', []),
                   segments=[{k:v for k,v in z.items() if k!='wav'} for z in results],
                   format_trace=getattr(t.tts, 'format_trace', {}),
                   input_kind='original audio; no transcript supplied',
                   backend='ONNX Runtime CPU; neural mode also uses Apple Metal')
        print('DONE', key, mode, json.dumps(row['segments'], ensure_ascii=False), flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        row = dict(id=key, mode=mode, error=str(exc), elapsed_seconds=time.perf_counter()-begin)
    (out / (key + '.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2))
    rows.append(row)
(out / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print('COMPLETE', mode, flush=True)

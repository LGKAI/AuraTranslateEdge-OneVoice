"""Generate actual TTS audio and capture normalization/chunks for format probes."""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 's2s_edge'))
from s2s_edge.registry import Registry
from s2s_edge.audio import save_wav
from s2s_edge import tts as module

mode = sys.argv[1]
os.environ['S2S_NEURAL_TN'] = '1' if mode == 'neural' else '0'
os.environ['S2S_TTS_FORMAT'] = {'default':'legacy', 'neural':'neural', 'hybrid':'hybrid'}[mode]
out = ROOT / 'results/multilingual_tts_format' / mode
out.mkdir(parents=True, exist_ok=True)
os.environ['S2S_TN_AUDIT'] = str(out / 'normalization.jsonl')
cases = {
    'en': 'The meeting is at 3:30 PM on 21/12/2027. The price is $18.25. The temperature is 36.8 degrees Celsius. My ZIP code is 04782. My phone number is 0908246135. The flight is VN247.',
    'ko': '회의는 2027년 12월 21일 3:30에 있습니다. 가격은 18.25달러입니다. 온도는 36.8도입니다. 우편번호는 04782입니다. 전화번호는 0908246135입니다. 항공편은 VN247입니다.',
    'zh': '会议在2027年12月21日3:30开始。价格是18.25美元。温度是36.8摄氏度。邮政编码是04782。电话号码是0908246135。航班号是VN247。',
}
if mode == 'hybrid':
    cases['vi'] = 'Cuộc họp lúc 3:30 chiều ngày 21/12/2027. Giá là $18.25. Nhiệt độ 36,8°C. Mã ZIP 04782. Điện thoại 0908246135. Chuyến bay VN247. Số nhà 734.'
reg = Registry(str(ROOT / 'models/models.cpu.json'), 'cpu')
rows = []
for lang, text in cases.items():
    if mode == 'neural' and lang != 'en':
        continue  # Korean/Chinese have no learned normalization branch.
    model = reg.tts(lang)
    normalized = []
    chunks = []
    original_normalize = module.normalize
    original_synth = model._synth
    def record_normalize(s, language):
        y = original_normalize(s, language)
        normalized.append(y)
        return y
    def record_synth(piece):
        chunks.append(piece)
        return original_synth(piece)
    module.normalize = record_normalize
    model._synth = record_synth
    start = time.perf_counter()
    try:
        audio, sr = model.synthesize(text)
        assert len(audio) and np.isfinite(audio).all()
        target = out / (lang + '.wav')
        save_wav(str(target), audio, sr)
        row = dict(lang=lang, mode=mode, input=text,
                   normalized_text=getattr(model, 'tts_input_text', None) or normalized[-1],
                   synth_chunks=chunks, output=str(target), sample_rate=sr,
                   output_seconds=len(audio)/sr, elapsed_seconds=time.perf_counter()-start,
                   format_trace=getattr(model, 'format_trace', {}),
                   finite=True, scope='actual TTS only; authored format probes, not S2S or a human listening score')
    except Exception as exc:
        import traceback
        row = dict(lang=lang, mode=mode, input=text, error=str(exc),
                   traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-start,
                   scope='actual TTS invocation failed; no audio generated')
    finally:
        module.normalize = original_normalize
        model._synth = original_synth
    rows.append(row)
    (out / (lang + '.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2))
    print(json.dumps(row, ensure_ascii=False), flush=True)
(out / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))

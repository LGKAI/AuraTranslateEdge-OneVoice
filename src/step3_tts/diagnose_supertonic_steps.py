"""Step 5 debug -- find the minimum Supertonic total_steps that eliminates
the Korean syllable-repetition artifact found in pipeline_s2s.py's CSV
output (steps=5 default: "chamsok sokha" duplicated syllables). Tests
5/6/7/8 on the same sentence that showed the bug, scores via round-trip
SenseVoice CER against the MT reference text.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_device, rtf, normalize_text_for_cer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KO_TEXT = "사람들이 너무 많아서 세인트 피터스퀘어에서 장례식에 참석하는 것은 불가능했습니다."
OUT_DIR = os.path.join(ROOT, "outputs", "diagnose_ko_steps")
STEPS_TO_TEST = [5, 6, 7, 8, 10]


def main():
    import jiwer
    from supertonic import TTS
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    import soundfile as sf
    import numpy as np

    os.makedirs(OUT_DIR, exist_ok=True)
    tts = TTS(model_dir=os.path.expanduser("~/.cache/supertonic3"), auto_download=False)
    style = tts.get_voice_style(voice_name="M1")

    device = get_device()
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    asr = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                     vad_kwargs={"max_single_segment_time": 30000}, device=device_str, disable_update=True)

    print(f"{'steps':6s} {'synth_s':8s} {'CER':7s}  transcript")
    for steps in STEPS_TO_TEST:
        t0 = time.perf_counter()
        wav, audio_sec = tts.synthesize(text=KO_TEXT, lang="ko", voice_style=style,
                                        total_steps=steps, speed=1.0)
        elapsed = time.perf_counter() - t0
        wav_path = os.path.join(OUT_DIR, f"steps{steps}.wav")
        sf.write(wav_path, np.asarray(wav).squeeze(), 24000)

        result = asr.generate(input=wav_path, cache={}, language="ko", use_itn=True, batch_size_s=60)
        hyp = rich_transcription_postprocess(result[0]["text"]).strip()
        cer = jiwer.cer(normalize_text_for_cer(KO_TEXT), normalize_text_for_cer(hyp))
        print(f"{steps:<6d} {elapsed:<8.2f} {cer:<7.4f} {hyp}")


if __name__ == "__main__":
    main()

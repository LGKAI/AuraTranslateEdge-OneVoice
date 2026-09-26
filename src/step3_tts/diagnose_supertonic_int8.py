"""Step 4 diagnostic -- bisect WHICH of Supertonic's 4 ONNX submodels breaks
when dynamically quantized to int8. Builds a "mixed" model dir per config
(some submodels fp32 original, some int8 quantized), synthesizes 1 en + 1 ko
sentence per config, and does a quick round-trip transcription via SenseVoice
to get a fast pass/fail signal (not full 5-sentence WER, just enough to tell
broken from working).
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_device

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG_DIR = os.path.expanduser("~/.cache/supertonic3")
INT8_DIR = os.path.join(ROOT, "outputs", "supertonic-int8")
WORK_BASE = os.path.join(ROOT, "outputs", "supertonic-diagnose")

SUBMODELS = ["text_encoder", "vector_estimator", "vocoder", "duration_predictor"]
TEST_TEXT = {"en": "The service is frequently used by shipping.", "ko": "중동의 따뜻한 기후에서는 집이 중요하지 않았습니다."}

CONFIGS = [
    ("all_fp32_baseline", []),                 # sanity check: should be perfect
    ("all_int8_broken", SUBMODELS),             # already confirmed broken
    ("only_text_encoder_int8", ["text_encoder"]),
    ("only_vector_estimator_int8", ["vector_estimator"]),
    ("only_vocoder_int8", ["vocoder"]),
    ("only_duration_predictor_int8", ["duration_predictor"]),
]


def build_mixed_dir(name, int8_submodels):
    work_dir = os.path.join(WORK_BASE, name)
    onnx_dir = os.path.join(work_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    for sub in SUBMODELS:
        fname = f"{sub}.onnx"
        src = os.path.join(INT8_DIR, fname) if sub in int8_submodels else os.path.join(ORIG_DIR, "onnx", fname)
        shutil.copy2(src, os.path.join(onnx_dir, fname))
    for extra in ["tts.json", "unicode_indexer.json"]:
        shutil.copy2(os.path.join(ORIG_DIR, "onnx", extra), os.path.join(onnx_dir, extra))
    voice_styles_dst = os.path.join(work_dir, "voice_styles")
    if not os.path.exists(voice_styles_dst):
        shutil.copytree(os.path.join(ORIG_DIR, "voice_styles"), voice_styles_dst)
    for extra in ["config.json"]:
        src = os.path.join(ORIG_DIR, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(work_dir, extra))
    return work_dir


def load_sensevoice():
    from funasr import AutoModel
    device = get_device()
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    vad_kwargs = {"max_single_segment_time": 30000}
    return AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", vad_kwargs=vad_kwargs, device=device_str)


def transcribe(model, path, lang):
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    result = model.generate(input=path, cache={}, language=lang, use_itn=True, batch_size_s=60)
    return rich_transcription_postprocess(result[0]["text"]).strip()


def main():
    from supertonic import TTS
    import soundfile as sf
    import numpy as np

    sensevoice = load_sensevoice()
    os.makedirs(WORK_BASE, exist_ok=True)

    print(f"\n{'CONFIG':32s} {'LANG':4s} {'RESULT':s}")
    print("-" * 90)
    for name, int8_submodels in CONFIGS:
        work_dir = build_mixed_dir(name, int8_submodels)
        try:
            tts = TTS(model_dir=work_dir, auto_download=False)
            style = tts.get_voice_style(voice_name="M1")
            for lang, text in TEST_TEXT.items():
                wav, audio_sec = tts.synthesize(text=text, lang=lang, voice_style=style, total_steps=8, speed=1.0)
                wav = np.asarray(wav).squeeze()
                wav_path = os.path.join(WORK_BASE, f"{name}_{lang}.wav")
                sf.write(wav_path, wav, 24000)
                hyp = transcribe(sensevoice, wav_path, lang)
                verdict = "OK" if hyp.strip() else "BROKEN(empty)"
                print(f"{name:32s} {lang:4s} nonzero={np.count_nonzero(wav)}/{wav.size:6d}  "
                      f"asr='{hyp[:60]}'  [{verdict}]")
        except Exception as e:
            print(f"{name:32s} FAILED TO LOAD/RUN: {e}")


if __name__ == "__main__":
    main()

"""Step 5 optimization -- search for a fixed numpy random seed that reliably
avoids Supertonic's Korean repetition artifact across MULTIPLE different
Korean sentences, so the pipeline can seed-and-synthesize once instead of
retrying. Supertonic's flow-matching sampler draws its initial noise from
the global numpy RNG (core.py: `np.random.randn(...)`, no seed param in the
public API) -- fixing that RNG before each call makes the sampler
deterministic for a given (seed, text) pair.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import get_device, normalize_text_for_cer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "find_ko_seed")

KO_TEXTS = [
    "사람들이 너무 많아서 세인트 피터스퀘어에서 장례식에 참석하는 것은 불가능했습니다.",
    "다리 밑 수직 간격은 15미터이며 공사는 2011년 8월에 마무리되었으며 해당 다리의 통행금지는 2017년 3월까지이다",
    "염소 사육은 대략 일만 년 전에 이란의 자그로스산맥에서 시작한 것으로 보입니다",
    "홍콩의 스카이라인을 이루는 빌딩 행렬은 빅토리아 항구의 수면에 선명히 비치는 모습 때문에 반짝이는 막대그래프에 비유된다",
]
SEEDS = [0, 1, 7, 42, 123, 2024]


def main():
    import numpy as np
    import jiwer
    from supertonic import TTS
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    import soundfile as sf

    os.makedirs(OUT_DIR, exist_ok=True)
    tts = TTS(model_dir=os.path.expanduser("~/.cache/supertonic3"), auto_download=False)
    style = tts.get_voice_style(voice_name="M1")

    device = get_device()
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    asr = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                     vad_kwargs={"max_single_segment_time": 30000}, device=device_str, disable_update=True)

    print(f"{'seed':6s} {'text_idx':9s} {'CER':7s}")
    results = {}
    for seed in SEEDS:
        cers = []
        for ti, text in enumerate(KO_TEXTS):
            np.random.seed(seed)
            t0 = time.perf_counter()
            wav, audio_sec = tts.synthesize(text=text, lang="ko", voice_style=style, total_steps=5, speed=1.0)
            elapsed = time.perf_counter() - t0
            wav = np.asarray(wav).squeeze()
            path = os.path.join(OUT_DIR, f"seed{seed}_t{ti}.wav")
            sf.write(path, wav, 24000)

            result = asr.generate(input=path, cache={}, language="ko", use_itn=True, batch_size_s=60)
            hyp = rich_transcription_postprocess(result[0]["text"]).strip()
            cer = jiwer.cer(normalize_text_for_cer(text), normalize_text_for_cer(hyp))
            cers.append(cer)
            print(f"{seed:<6d} {ti:<9d} {cer:<7.4f} ({elapsed:.2f}s)")
        results[seed] = cers
        worst = max(cers)
        print(f"  seed={seed}: worst CER={worst:.4f}, all={[round(c,3) for c in cers]}\n")

    print("=== SUMMARY ===")
    for seed, cers in results.items():
        fails = sum(1 for c in cers if c > 0.05)
        print(f"seed={seed}: fails={fails}/{len(cers)}  max_cer={max(cers):.3f}  mean_cer={sum(cers)/len(cers):.3f}")


if __name__ == "__main__":
    main()

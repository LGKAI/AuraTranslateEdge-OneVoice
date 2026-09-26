"""Step 5 -- end-to-end speech-to-speech translation pipeline (ASR -> MT -> TTS).

Chains the per-module picks from step1-4 into one measurable pipeline:

  audio in (16kHz)
    -> ASR   Zipformer-30M int8 (vi, sherpa-onnx) / SenseVoice-Small (en/zh/ko)
    -> MT    NLLB-200-distilled-600M, CTranslate2 int8
    -> TTS   Piper (vi) / Supertonic int8-mixed (en, ko) / MeloTTS (zh)
  -> audio out

Runs in two device modes for comparison:

  --device cpu   : everything on CPU. SenseVoice runs fp32 PyTorch (funasr)
                   because it measured 10-20x faster than the int8 ONNX export
                   on plain CPU (the int8 ONNX is for the QNN/NPU deploy path).
  --device cuda  : SenseVoice + MeloTTS on CUDA (PyTorch), NLLB int8 on CUDA
                   via CTranslate2. Zipformer / Piper / Supertonic stay on CPU
                   (sherpa-onnx PyPI wheel is CPU-only on Windows; the local
                   ONNX Runtime is CPU-only) -- noted per-row in the CSV.

Metrics per test item: per-stage latency, end-to-end latency, input-ASR
WER/CER vs manifest reference, and round-trip ASR of the synthesized output
vs the MT text (intelligibility signal, same method as step3).

Usage:
    python pipeline_s2s.py --device cpu
    python pipeline_s2s.py --device cuda
"""
import os
import sys
import csv
import json
import time
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import (SR, load_wav, save_wav, rtf, normalize_text, normalize_text_for_cer)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASR_MANIFEST = os.path.join(ROOT, "data", "asr", "manifest.json")

ZIPFORMER_SNAP = os.path.join(ROOT, "third_party", "zipformer",
                              "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
SENSEVOICE_ONNX_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
NLLB_CT2_DIR = os.path.join(ROOT, "outputs", "nllb-ct2-int8")
SUPERTONIC_DIR = os.path.join(ROOT, "outputs", "supertonic-deploy")  # 3x int8 + fp32 vocoder
PIPER_VOICE = os.path.join(ROOT, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")

LANG_TO_FLORES = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
CER_LANGS = {"zh", "ko"}

# (input_lang, target_lang, [input file indexes from data/asr/<lang>/])
TEST_PLAN = [
    ("vi", "en", [0, 1]),
    ("vi", "zh", [2]),
    ("vi", "ko", [3]),
    ("en", "vi", [0, 1]),
    ("zh", "vi", [0]),
    ("ko", "vi", [0]),
]


# ---------------------------------------------------------------- ASR modules
class ZipformerVi:
    """Vietnamese ASR: Zipformer-30M-RNNT int8 ONNX via sherpa-onnx (CPU)."""

    def __init__(self):
        import sherpa_onnx
        snap = ZIPFORMER_SNAP
        snap = os.path.join(snap, os.listdir(snap)[0])
        self.device_used = "cpu (sherpa-onnx, int8)"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=os.path.join(snap, "tokens.generated.txt"),
            encoder=os.path.join(snap, "encoder-epoch-20-avg-10.int8.onnx"),
            decoder=os.path.join(snap, "decoder-epoch-20-avg-10.int8.onnx"),
            joiner=os.path.join(snap, "joiner-epoch-20-avg-10.int8.onnx"),
            num_threads=2, sample_rate=SR, feature_dim=80,
            decoding_method="greedy_search", provider="cpu",
        )

    def transcribe(self, wav):
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SR, wav)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


class SenseVoiceOnnx:
    """En/Zh/Ko ASR: SenseVoice-Small int8 ONNX via funasr_onnx (deploy config).

    NOT used by this pipeline's --device cpu/cuda comparison -- SenseVoiceFp32
    is faster and higher-quality on a plain dev machine (see its docstring).
    Kept here, tested and working (step4.md verified it end-to-end: en WER
    7.6%, zh/ko CER ~9.5-9.8%), as the reference implementation for the real
    QNN/NPU deploy path, where PyTorch isn't available and this int8 ONNX
    export is what actually ships."""

    def __init__(self):
        from funasr_onnx import SenseVoiceSmall
        from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess
        self._post = rich_transcription_postprocess
        self.device_used = "cpu (funasr_onnx, int8)"
        self.model = SenseVoiceSmall(model_dir=SENSEVOICE_ONNX_DIR, quantize=True, batch_size=1)

    def transcribe_path(self, path, lang):
        result = self.model(wav_content=path, language=lang, use_itn=True)
        return self._post(result[0]).strip()


class SenseVoiceFp32:
    """En/Zh/Ko ASR: SenseVoice-Small fp32 PyTorch via funasr.

    Measured on this dev machine (2026-08-09): fp32 PyTorch is ~10-20x FASTER
    than the int8 ONNX export on plain CPU (RTF 0.03-0.13 vs ~0.7) AND higher
    quality (step4.md: int8 hurts zh/ko CER). The int8 ONNX export only makes
    sense for the QNN/NPU deploy path, where PyTorch can't run."""

    def __init__(self, device="cpu"):
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        self._post = rich_transcription_postprocess
        self.device_used = f"{device} (funasr, fp32)"
        device_str = "cuda:0" if device == "cuda" else "cpu"
        try:
            self.model = AutoModel(model="FunAudioLLM/SenseVoiceSmall", vad_model="fsmn-vad",
                                   vad_kwargs={"max_single_segment_time": 30000},
                                   hub="hf", device=device_str, disable_update=True)
        except TypeError:
            self.model = AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                                   vad_kwargs={"max_single_segment_time": 30000},
                                   device=device_str, disable_update=True)

    def transcribe_path(self, path, lang):
        result = self.model.generate(input=path, cache={}, language=lang,
                                     use_itn=True, batch_size_s=60,
                                     merge_vad=True, merge_length_s=15)
        return self._post(result[0]["text"]).strip()


# ------------------------------------------------------------------ MT module
class NllbMt:
    def __init__(self, device):
        import ctranslate2
        from transformers import AutoTokenizer
        self.device_used = f"{device} (ctranslate2, int8)"
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translator = ctranslate2.Translator(NLLB_CT2_DIR, device=device)

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = LANG_TO_FLORES[src_lang]
        tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
        result = self.translator.translate_batch(
            [tokens], target_prefix=[[LANG_TO_FLORES[tgt_lang]]],
            beam_size=4, max_decoding_length=256)
        out_tokens = result[0].hypotheses[0][1:]  # drop target-lang prefix token
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True)


# ----------------------------------------------------------------- TTS modules
class PiperVi:
    def __init__(self):
        from piper.voice import PiperVoice
        self.device_used = "cpu (onnxruntime)"
        self.voice = PiperVoice.load(PIPER_VOICE)

    def synth(self, text, out_path):
        audio = b"".join(c.audio_int16_bytes for c in self.voice.synthesize(text))
        import numpy as np
        wav = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        save_wav(out_path, wav, 22050)
        return len(wav) / 22050.0


class SupertonicTTS:
    """Supertonic en/ko TTS.

    fp32 dir by default: measured faster than the int8-mixed dir on plain x86
    CPU (dynamic int8 convs don't accelerate under ONNX Runtime without VNNI/
    NPU; RTF ~1.2 fp32 vs ~3.1 int8-mixed here). int8-mixed (178MB vs 398MB)
    remains the edge-deploy artifact for the QNN path.
    total_steps=5: flow-matching steps 8->5 gives 1.5x speedup with no
    measurable round-trip ASR change on English test sentences. Korean is
    a different story: its flow-matching sampler is stochastic and produces
    a repeated-syllable artifact ("가능가능했습니다") on a real fraction of
    single-attempt runs -- measured 3/6 failures (round-trip CER>5%) at
    BOTH total_steps=5 and total_steps=8 (2026-08-09,
    outputs/ko_reliability.log), so raising the step count does NOT fix
    Korean's reliability; kept at 5 since it doesn't help anyway and 5 is
    faster. The actual fix is in pipeline_s2s.py's tts_synth(): a
    quality-gated retry loop for lang="ko" that re-synthesizes up to 5x
    and keeps the best round-trip-verified attempt."""

    def __init__(self, variant="fp32", total_steps=5):
        from supertonic import TTS
        model_dir = SUPERTONIC_DIR if variant == "int8mix" else os.path.expanduser("~/.cache/supertonic3")
        self.device_used = f"cpu (onnxruntime, {variant}, steps={total_steps})"
        self.total_steps = total_steps
        self.tts = TTS(model_dir=model_dir, auto_download=False)
        self.style = self.tts.get_voice_style(voice_name="M1")
        self.sr = getattr(self.tts, "sample_rate", 44100)

    def synth(self, text, lang, out_path):
        import numpy as np
        wav, audio_sec = self.tts.synthesize(text=text, lang=lang, voice_style=self.style,
                                             total_steps=self.total_steps, speed=1.0)
        save_wav(out_path, np.asarray(wav).squeeze(), self.sr)
        return float(audio_sec)


class MeloTtsZh:
    """MeloTTS zh TTS. BERT prosody features disabled: measured 5x faster on
    CPU (3.3s vs 17.5s on the test sentence), identical round-trip ASR
    transcript, and avoids the hidden 1.35GB bert-base-multilingual-uncased
    runtime download (not counted in step4.md's 199MB). Naturalness tradeoff
    flagged for human listening check before final demo."""

    def __init__(self, device, disable_bert=True):
        from melo.api import TTS
        self.device_used = f"{device} (pytorch, fp32, disable_bert={disable_bert})"
        self.model = TTS(language="ZH", device=device)
        if disable_bert:
            self.model.hps.data.disable_bert = True
        self.speaker_id = list(self.model.hps.data.spk2id.values())[0]
        self.sr = self.model.hps.data.sampling_rate

    def synth(self, text, out_path):
        import soundfile as sf
        self.model.tts_to_file(text, self.speaker_id, out_path, speed=1.0, quiet=True)
        wav, sr = sf.read(out_path)
        return len(wav) / sr


# ------------------------------------------------------------------- pipeline
def score(metric_lang, ref, hyp):
    import jiwer
    if metric_lang in CER_LANGS:
        return jiwer.cer(normalize_text_for_cer(ref), normalize_text_for_cer(hyp))
    return jiwer.wer(normalize_text(ref), normalize_text(hyp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = ap.parse_args()
    mode = args.device
    out_dir = os.path.join(ROOT, "outputs", "pipeline", mode)
    os.makedirs(out_dir, exist_ok=True)
    results_csv = os.path.join(ROOT, "outputs", f"pipeline_results_{mode}.csv")

    with open(ASR_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    by_lang = {}
    for it in manifest:
        by_lang.setdefault(it["lang"], []).append(it)

    print(f"[pipeline] === device mode: {mode} ===", flush=True)

    # ---- load models (with per-module device notes)
    t_load0 = time.perf_counter()
    print("[pipeline] loading Zipformer (vi ASR)...", flush=True)
    asr_vi = ZipformerVi()
    print("[pipeline] loading SenseVoice (en/zh/ko ASR)...", flush=True)
    asr_multi = SenseVoiceFp32(mode)
    print("[pipeline] loading NLLB-600M ct2-int8 (MT)...", flush=True)
    mt = NllbMt("cuda" if mode == "cuda" else "cpu")
    print("[pipeline] loading Piper (vi TTS)...", flush=True)
    tts_vi = PiperVi()
    print("[pipeline] loading Supertonic (en/ko TTS)...", flush=True)
    tts_multi = SupertonicTTS()
    print("[pipeline] loading MeloTTS (zh TTS)...", flush=True)
    tts_zh = MeloTtsZh("cuda" if mode == "cuda" else "cpu")
    print(f"[pipeline] all models loaded in {time.perf_counter() - t_load0:.1f}s", flush=True)

    def asr_transcribe(wav_path, lang):
        if lang == "vi":
            return asr_vi.transcribe(load_wav(wav_path))
        return asr_multi.transcribe_path(wav_path, lang)

    # Supertonic Korean has a real, empirically-measured reliability issue:
    # its flow-matching sampler is stochastic, and repeated syllables ("가능
    # 가능했습니다") show up in ~50% of single-attempt runs on some sentences
    # -- confirmed 3/6 failures at total_steps=5 AND 3/6 at total_steps=8
    # (2026-08-09, outputs/ko_reliability.log), so raising the step count
    # alone does NOT fix it. The reliable fix is a quality-gated retry:
    # synthesize, round-trip through the ASR we already have loaded, and
    # retry if the CER looks like a repetition artifact. With a per-attempt
    # failure rate of ~50%, 3 attempts brings failure probability to ~12%
    # and 5 attempts to ~3%.
    KO_RETRY_MAX_ATTEMPTS = 5
    KO_RETRY_CER_THRESHOLD = 0.05

    def tts_synth(text, lang, out_path):
        if lang == "vi":
            return tts_vi.synth(text, out_path)
        if lang == "zh":
            return tts_zh.synth(text, out_path)
        if lang != "ko":
            return tts_multi.synth(text, lang, out_path)

        best_cer, best_audio_sec, attempts = None, None, 0
        best_path = out_path + ".best"
        for attempt in range(1, KO_RETRY_MAX_ATTEMPTS + 1):
            attempts = attempt
            audio_sec = tts_multi.synth(text, lang, out_path)
            hyp = asr_transcribe(out_path, lang)
            cer = score(lang, text, hyp)
            if best_cer is None or cer < best_cer:
                best_cer, best_audio_sec = cer, audio_sec
                shutil.copyfile(out_path, best_path)
            if cer <= KO_RETRY_CER_THRESHOLD:
                break
        shutil.move(best_path, out_path)
        if attempts > 1:
            print(f"    [ko-retry] {attempts} attempt(s), best round-trip CER={best_cer:.3f}", flush=True)
        return best_audio_sec

    # ---- warmup (excluded from timings): one tiny pass per module
    print("[pipeline] warmup pass...", flush=True)
    warm_wav = os.path.join(ROOT, by_lang["vi"][0]["path"])
    warm_text = asr_vi.transcribe(load_wav(warm_wav))
    warm_en = mt.translate("Xin chào, đây là kiểm tra khởi động.", "vi", "en")
    tts_vi.synth("khởi động", os.path.join(out_dir, "_warmup_vi.wav"))
    tts_multi.synth("warm up", "en", os.path.join(out_dir, "_warmup_en.wav"))
    tts_zh.synth("预热", os.path.join(out_dir, "_warmup_zh.wav"))
    _ = warm_text, warm_en

    rows = []
    for src_lang, tgt_lang, idxs in TEST_PLAN:
        items = by_lang[src_lang]
        for idx in idxs:
            item = items[idx]
            in_path = os.path.join(ROOT, item["path"])
            wav = load_wav(in_path)
            in_audio_sec = len(wav) / SR
            tag = f"{src_lang}2{tgt_lang}_{idx}"

            t0 = time.perf_counter()
            hyp = asr_transcribe(in_path, src_lang)
            t_asr = time.perf_counter() - t0

            # Zipformer-6000h outputs ALL-CAPS Vietnamese; feeding that
            # verbatim into NLLB produces garbage translations (verified
            # 2026-08-09: caps -> "When a Struggle Curses the World...",
            # lowercased -> correct). Lowercase before MT; NLLB re-cases
            # the target language on output anyway.
            mt_in = hyp.lower() if src_lang == "vi" else hyp

            t0 = time.perf_counter()
            mt_text = mt.translate(mt_in, src_lang, tgt_lang)
            t_mt = time.perf_counter() - t0

            out_path = os.path.join(out_dir, f"{tag}.wav")
            t0 = time.perf_counter()
            out_audio_sec = tts_synth(mt_text, tgt_lang, out_path)
            t_tts = time.perf_counter() - t0

            e2e = t_asr + t_mt + t_tts

            # quality: (1) input ASR vs manifest reference
            in_score = score(src_lang, item["transcript"], hyp)
            # quality: (2) round-trip ASR of the synthesized output vs MT text
            rt_hyp = asr_transcribe(out_path, tgt_lang)
            rt_score = score(tgt_lang, mt_text, rt_hyp)

            row = {
                "direction": f"{src_lang}->{tgt_lang}",
                "in_file": item["path"], "in_audio_sec": round(in_audio_sec, 2),
                "asr_sec": round(t_asr, 3), "mt_sec": round(t_mt, 3),
                "tts_sec": round(t_tts, 3), "out_audio_sec": round(out_audio_sec, 2),
                "e2e_sec": round(e2e, 3),
                "asr_score": round(in_score, 4), "rt_score": round(rt_score, 4),
                "asr_hyp": hyp, "mt_text": mt_text, "rt_hyp": rt_hyp,
                "device_mode": mode,
            }
            rows.append(row)
            metric = "CER" if src_lang in CER_LANGS else "WER"
            rt_metric = "CER" if tgt_lang in CER_LANGS else "WER"
            print(f"[pipeline] {tag}: ASR {t_asr:.2f}s ({metric}={in_score:.3f}) -> "
                  f"MT {t_mt:.2f}s -> TTS {t_tts:.2f}s | e2e {e2e:.2f}s "
                  f"(in {in_audio_sec:.1f}s -> out {out_audio_sec:.1f}s) | "
                  f"roundtrip {rt_metric}={rt_score:.3f}", flush=True)
            print(f"    hyp: {hyp[:90]}", flush=True)
            print(f"    mt : {mt_text[:90]}", flush=True)
            print(f"    rt : {rt_hyp[:90]}", flush=True)

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"\n[pipeline] === SUMMARY ({mode}, n={n}) ===")
    print(f"  mean ASR stage : {sum(r['asr_sec'] for r in rows) / n:.3f}s")
    print(f"  mean MT  stage : {sum(r['mt_sec'] for r in rows) / n:.3f}s")
    print(f"  mean TTS stage : {sum(r['tts_sec'] for r in rows) / n:.3f}s")
    print(f"  mean end-to-end: {sum(r['e2e_sec'] for r in rows) / n:.3f}s")
    print(f"  mean input-ASR score : {sum(r['asr_score'] for r in rows) / n:.4f}")
    print(f"  mean round-trip score: {sum(r['rt_score'] for r in rows) / n:.4f}")
    print(f"[pipeline] wrote {results_csv}")


if __name__ == "__main__":
    main()

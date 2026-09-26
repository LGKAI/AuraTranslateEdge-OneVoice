"""Step 3 quality eval -- round-trip intelligibility check for TTS output.
Feeds each synthesized WAV (from test_tts_supertonic.py / test_tts_melotts.py
/ test_tts_piper.py) back through the SAME ASR models already chosen and
validated in Step 1 (Zipformer for vi, SenseVoice for en/zh/ko), then scores
WER (vi/en) or CER (zh/ko) against the original source text. This can't
replace actually listening to the audio (see outputs/tts_*/*.wav), but it's
a standard automatic proxy for TTS intelligibility: low error rate means the
synthesized speech is clear enough for a real ASR model to transcribe back
correctly.
"""
import os
import sys
import csv
import glob

import jiwer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import SR, get_device, load_wav, normalize_text, normalize_text_for_cer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_CSV = os.path.join(ROOT, "outputs", "tts_quality_results.csv")

ZIPFORMER_REPO = "hynt/Zipformer-30M-RNNT-6000h"
ZIPFORMER_CACHE_DIR = os.path.join(ROOT, "third_party", "zipformer")
CER_LANGS = {"zh", "ko"}


def load_zipformer(device):
    import sherpa_onnx
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(repo_id=ZIPFORMER_REPO, cache_dir=ZIPFORMER_CACHE_DIR)

    def pick(pattern):
        all_matches = sorted(glob.glob(os.path.join(local_dir, pattern)))
        non_int8 = [f for f in all_matches if ".int8." not in f]
        chosen = non_int8 or all_matches
        return chosen[0] if chosen else None

    encoder, decoder, joiner = pick("encoder*.onnx"), pick("decoder*.onnx"), pick("joiner*.onnx")
    tokens = pick("tokens.txt") or pick("*tokens*.txt")
    if not tokens:
        bpe_model = pick("bpe.model") or pick("*.model")
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)
        tokens = os.path.join(local_dir, "tokens.generated.txt")
        with open(tokens, "w", encoding="utf-8") as f:
            for i in range(sp.get_piece_size()):
                f.write(f"{sp.id_to_piece(i)} {i}\n")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens, encoder=encoder, decoder=decoder, joiner=joiner,
        num_threads=2, sample_rate=SR, feature_dim=80, decoding_method="greedy_search",
        provider="cuda" if device.type == "cuda" else "cpu")


def transcribe_zipformer(recognizer, wav):
    stream = recognizer.create_stream()
    stream.accept_waveform(SR, wav)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def load_sensevoice(device_str):
    from funasr import AutoModel
    vad_kwargs = {"max_single_segment_time": 30000}
    return AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                      vad_kwargs=vad_kwargs, device=device_str)


def transcribe_sensevoice(model, path, lang):
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    result = model.generate(input=path, cache={}, language=lang, use_itn=True,
                             batch_size_s=60, merge_vad=True, merge_length_s=15)
    return rich_transcription_postprocess(result[0]["text"]).strip()


def find_tts_csvs():
    pattern = os.path.join(ROOT, "outputs", "tts_*_results.csv")
    return [p for p in glob.glob(pattern) if "quality" not in os.path.basename(p)]


def main():
    device = get_device()
    device_str = "cuda:0" if device.type == "cuda" else "cpu"
    print(f"[test_tts_eval_quality] device = {device_str}")

    csvs = find_tts_csvs()
    if not csvs:
        print("[test_tts_eval_quality] No outputs/tts_*_results.csv found -- "
              "run test_tts_supertonic.py / test_tts_melotts.py / test_tts_piper.py first.")
        return

    all_rows = []
    for csv_path in csvs:
        engine = os.path.basename(csv_path).replace("tts_", "").replace("_results.csv", "")
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["engine"] = engine
                all_rows.append(row)
    print(f"[test_tts_eval_quality] {len(all_rows)} synthesized clips found across {len(csvs)} engine(s)")

    need_zipformer = any(r["lang"] == "vi" for r in all_rows)
    need_sensevoice = any(r["lang"] in ("en", "zh", "ko") for r in all_rows)

    zipformer = load_zipformer(device) if need_zipformer else None
    sensevoice = load_sensevoice(device_str) if need_sensevoice else None

    out_rows = []
    for row in all_rows:
        lang, wav_path, ref = row["lang"], row["wav_path"], row["text"]
        wav = load_wav(wav_path)
        if lang == "vi":
            hyp = transcribe_zipformer(zipformer, wav)
        else:
            hyp = transcribe_sensevoice(sensevoice, wav_path, lang)

        if lang in CER_LANGS:
            metric, score = "cer", jiwer.cer(normalize_text_for_cer(ref), normalize_text_for_cer(hyp))
        else:
            metric, score = "wer", jiwer.wer(normalize_text(ref), normalize_text(hyp))

        out_rows.append({"engine": row["engine"], "lang": lang, "idx": row["idx"],
                          "rtf": row["rtf"], "metric": metric, "score": round(score, 4),
                          "ref_text": ref, "roundtrip_asr_text": hyp})
        print(f"[test_tts_eval_quality] {row['engine']:12s} {lang} [{row['idx']}]  "
              f"{metric.upper()}={score:.3f}  RTF={row['rtf']}  asr='{hyp[:50]}'")

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"[test_tts_eval_quality] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()

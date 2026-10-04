"""One-request Zipformer service boundary.

The service owns the recognizer for its whole lifetime.  A request performs
one local ``decode_stream`` call; decoder/joiner orchestration stays inside
sherpa-onnx instead of crossing an AI Hub API boundary for every token.

The current sherpa-onnx Python transducer binding supports CPU/CUDA, not a
custom QNN transducer provider.  On Qualcomm, keep this same request boundary
but replace the backend with a native QNN loop (or a future sherpa QNN
transducer build); do not call AI Hub per token.
"""
import argparse
import json
import os
import time

import soundfile as sf
import sherpa_onnx


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "third_party_zipformer_real_full")


def find(name):
    matches = [os.path.join(MODEL_DIR, name)]
    for p in matches:
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p
    raise FileNotFoundError(p)


def tokens_path():
    path = os.path.join(MODEL_DIR, "tokens.generated.txt")
    if not os.path.exists(path):
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=find("bpe.model"))
        with open(path, "w", encoding="utf-8") as f:
            for i in range(sp.get_piece_size()):
                f.write(f"{sp.id_to_piece(i)} {i}\n")
    return path


def build_recognizer(provider="cpu"):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=tokens_path(),
        encoder=find("encoder-epoch-20-avg-10.onnx"),
        decoder=find("decoder-epoch-20-avg-10.onnx"),
        joiner=find("joiner-epoch-20-avg-10.onnx"),
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=provider,
    )


def transcribe(recognizer, wav_path):
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"expected 16 kHz audio, got {sr}")
    started = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, wav)
    recognizer.decode_stream(stream)
    elapsed = time.perf_counter() - started
    return {
        "audio": wav_path,
        "seconds": len(wav) / sr,
        "latency_seconds": elapsed,
        "rtf": elapsed / (len(wav) / sr),
        "text": stream.result.text.strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    recognizer = build_recognizer(args.provider)
    print(json.dumps(transcribe(recognizer, args.audio), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

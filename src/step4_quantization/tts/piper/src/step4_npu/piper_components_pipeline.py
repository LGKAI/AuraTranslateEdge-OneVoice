#!/usr/bin/env python3
"""Step 4 NPU -- Piper (vi) split-model synthesis (host glue + fp32 reference).

Implements the official Qualcomm PiperTTS orchestration for the 4 NPU
components, running components either via PyTorch (fp32 reference) or via
ONNX Runtime (fp32 reference / local sanity) -- later the SAME inputs feed the
compiled QNN binaries on IQ-9075.

All neural computation lives in the 4 components (encoder/sdp/flow/decoder);
this module only does non-neural glue: espeak phonemization, padding, the
argmax alignment (generate_path) and the sliding-window vocoder assembly.

Usage:
    python piper_components_pipeline.py --onnx_dir outputs/piper_vi_npu/components \
        --config outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx.json \
        --text 'Xin chào' --backend torch|ort
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MAX_SEQ_LEN = 512
ENCODER_HIDDEN_DIM = 192
UPSAMPLED_MAX_SEQ_LEN = 1536
DEC_SEQ_OVERLAP = 12
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_LEN = MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP
UPSAMPLE_FACTOR = 256
SAMPLE_RATE = 22050
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_SCALE_W = 0.8


def phonemize_vi(text: str, id_map: dict | None = None) -> list[int]:
    """Real Piper Vietnamese phonemizer using espeak-ng and phoneme_id_map."""
    try:
        from piper import PiperVoice
        onnx_candidates = [
            Path("outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx"),
            Path("vi_VN-vais1000-medium.onnx"),
        ]
        cfg_candidates = [
            Path("outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx.json"),
            Path("vi_VN-vais1000-medium.onnx.json"),
        ]
        m_path = next((p for p in onnx_candidates if p.exists()), None)
        c_path = next((p for p in cfg_candidates if p.exists()), None)
        if m_path and c_path:
            voice = PiperVoice.load(str(m_path), config_path=str(c_path))
            phonemes = voice.phonemize(text)
            if phonemes and len(phonemes) > 0:
                return voice.phonemes_to_ids(phonemes[0])
    except Exception as e:
        logger.warning("PiperVoice phonemizer error (%s), checking pre-tokenized dataset...", e)

    # Fallback: check golden pre-tokenized dataset
    data_path = Path("outputs/piper_vi_npu/piper_vi_npu_data.npz")
    if data_path.exists():
        d = np.load(data_path, allow_pickle=True)
        if "test_texts" in d:
            test_texts = [str(t) for t in d["test_texts"]]
            for idx, t in enumerate(test_texts):
                if text.strip() == t.strip():
                    n = int(d["test_lengths"][idx])
                    return list(d["test_input"][idx][:n])
        if "calib_texts" in d:
            calib_texts = [str(t) for t in d["calib_texts"]]
            for idx, t in enumerate(calib_texts):
                if text.strip() == t.strip():
                    n = int(d["input_lengths"][idx])
                    return list(d["input"][idx][:n])

    raise RuntimeError(f"Cannot phonemize text '{text[:40]}': Piper espeak phonemizer unavailable.")


def prepare_input(phoneme_ids: list[int], max_seq_len: int = MAX_SEQ_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Prepare static padded int32 input for encoder."""
    arr = np.zeros((1, max_seq_len), dtype=np.int32)
    l = min(len(phoneme_ids), max_seq_len)
    arr[0, :l] = phoneme_ids[:l]
    return arr, np.array([l], dtype=np.int32)


def prepare_input_npu_direct(upstream_byte_tensor: np.ndarray, upstream_length: np.ndarray):
    return upstream_byte_tensor.astype(np.int32), upstream_length.astype(np.int32)


def generate_path_np(duration, mask):
    """generate_path (monotonic alignment) in numpy.

    Mirrors the torch reference exactly:
      cum = cumsum(duration) -> [b,1,t_x]; x = arange(t_y)
      path[b*t_x, t_y] = (x < cum)
      pad t_x axis by 1 at front, diff along t_x -> per-phoneme boundaries
      -> [b,1,t_y,t_x] * mask
    """
    b, _, t_y, t_x = mask.shape
    cum = np.cumsum(duration, axis=-1).reshape(b * t_x)
    x = np.arange(t_y, dtype=cum.dtype)
    path = (x[None, :] < cum[:, None]).astype(mask.dtype)  # [b*t_x, t_y]
    path = path.reshape(b, t_x, t_y)
    path_padded = np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    path = path - path_padded
    return path[:, None, :, :].transpose(0, 1, 3, 2) * mask


class ComponentRunner:
    """Run the NPU components via torch or onnxruntime."""

    def __init__(self, onnx_dir: Path, backend: str = "ort"):
        self.onnx_dir = Path(onnx_dir)
        self.backend = backend
        if backend == "ort":
            import onnxruntime as ort
            enc_file = "byte_text_encoder.onnx" if (self.onnx_dir / "byte_text_encoder.onnx").exists() else "piper_vi_encoder.onnx"
            self.sess = {
                "byte_text_encoder": ort.InferenceSession(str(self.onnx_dir / enc_file),
                                                          providers=["CPUExecutionProvider"]),
                "encoder": ort.InferenceSession(str(self.onnx_dir / "piper_vi_encoder.onnx"),
                                                providers=["CPUExecutionProvider"]),
                "sdp": ort.InferenceSession(str(self.onnx_dir / "piper_vi_sdp.onnx"),
                                            providers=["CPUExecutionProvider"]),
                "monotonic_aligner": ort.InferenceSession(str(self.onnx_dir / "monotonic_aligner.onnx"),
                                                          providers=["CPUExecutionProvider"]) if (self.onnx_dir / "monotonic_aligner.onnx").exists() else None,
                "flow": ort.InferenceSession(str(self.onnx_dir / "piper_vi_flow.onnx"),
                                             providers=["CPUExecutionProvider"]),
                "decoder": ort.InferenceSession(str(self.onnx_dir / "piper_vi_decoder.onnx"),
                                                providers=["CPUExecutionProvider"]),
                "overlap_add": ort.InferenceSession(str(self.onnx_dir / "overlap_add.onnx"),
                                                    providers=["CPUExecutionProvider"]) if (self.onnx_dir / "overlap_add.onnx").exists() else None,
                "audio_resampler": ort.InferenceSession(str(self.onnx_dir / "audio_resampler.onnx"),
                                                        providers=["CPUExecutionProvider"]) if (self.onnx_dir / "audio_resampler.onnx").exists() else None,
            }
        else:
            import torch
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from export_piper_components import (
                Decoder, Encoder, Flow, SDP, build_model_from_onnx,
            )
            from byte_text_pipeline import ByteLevelTextEncoder
            from alignment_pipeline import MonotonicAligner
            from overlap_add_pipeline import VectorizedOverlapAdd
            from resampler_pipeline import build_sinc_resample_matrix
            self.torch = torch
            gen = build_model_from_onnx(
                str(self.onnx_dir.parent / "vi_VN-vais1000-medium.onnx"),
                str(self.onnx_dir.parent / "vi_VN-vais1000-medium.onnx.json"))
            self.byte_enc = ByteLevelTextEncoder()
            self.enc = Encoder(gen)
            self.sdp = SDP(gen)
            self.align = MonotonicAligner()
            self.flow = Flow(gen)
            self.dec = Decoder(gen)
            self.ola = VectorizedOverlapAdd()
            self.byte_enc.eval(); self.enc.eval(); self.sdp.eval(); self.align.eval(); self.flow.eval(); self.dec.eval(); self.ola.eval()

    def _run(self, comp, feeds):
        if self.backend == "ort":
            sess = self.sess.get(comp)
            if sess is None:
                raise RuntimeError(f"Session for {comp} is not available in {self.onnx_dir}")
            return sess.run(None, feeds)
        import torch
        tfeeds = {k: torch.from_numpy(v) for k, v in feeds.items()}
        with torch.no_grad():
            model = getattr(self, {"byte_text_encoder": "byte_enc", "encoder": "enc",
                                   "sdp": "sdp", "monotonic_aligner": "align",
                                   "flow": "flow", "decoder": "dec", "overlap_add": "ola"}[comp])
            out = model(*tfeeds.values())
            if not isinstance(out, tuple):
                out = (out,)
            return [o.numpy() for o in out]

    def byte_encoder(self, byte_indices, byte_lengths):
        outs = self._run("byte_text_encoder", {"byte_indices": byte_indices, "byte_lengths": byte_lengths})
        return outs[0], outs[1], outs[2], outs[3]

    def encoder(self, x, x_lengths):
        outs = self._run("encoder", {"x": x, "x_lengths": x_lengths})
        return outs[0], outs[1], outs[2], outs[3]

    def sdp(self, x_encoded, x_mask, length_scale, noise_scale_w):
        outs = self._run("sdp", {"x_encoded": x_encoded, "x_mask": x_mask,
                                 "length_scale": length_scale,
                                 "noise_scale_w": noise_scale_w})
        return outs[0], outs[1]

    def monotonic_aligner(self, w_ceil, x_mask, y_lengths):
        w_in = np.asarray(w_ceil).reshape(1, 1, MAX_SEQ_LEN).astype(np.float32)
        xm_in = np.asarray(x_mask).reshape(1, 1, MAX_SEQ_LEN).astype(np.float32)
        yl_in = np.asarray(y_lengths).flatten()[:1].astype(np.int32)
        if self.backend == "ort" and self.sess.get("monotonic_aligner") is not None:
            outs = self._run("monotonic_aligner", {"w_ceil": w_in, "x_mask": xm_in, "y_lengths": yl_in})
            return outs[0], outs[1]
        # Torch fallback / direct vector execution
        import torch
        from alignment_pipeline import MonotonicAligner
        aligner = getattr(self, "align", None) or MonotonicAligner()
        aligner.eval()
        with torch.no_grad():
            a_t, ym_t = aligner(torch.from_numpy(w_in), torch.from_numpy(xm_in), torch.from_numpy(yl_in))
            return a_t.numpy(), ym_t.numpy()

    def flow(self, m_p, logs_p, y_mask, attn_squeezed, noise_scale):
        outs = self._run("flow", {"m_p": m_p, "logs_p": logs_p, "y_mask": y_mask,
                                  "attn_squeezed": attn_squeezed,
                                  "noise_scale": noise_scale})
        return outs[0]

    def decoder(self, z_buf):
        outs = self._run("decoder", {"z": z_buf})
        return outs[0]

    def overlap_add(self, curr_chunk, prev_tail, is_first):
        outs = self._run("overlap_add", {"curr_chunk": curr_chunk, "prev_tail": prev_tail, "is_first": is_first})
        return outs[0], outs[1]


def synthesize(comp: ComponentRunner, phoneme_ids: list[int],
               noise_scale: float = DEFAULT_NOISE_SCALE,
               length_scale: float = DEFAULT_LENGTH_SCALE,
               noise_scale_w: float = DEFAULT_NOISE_SCALE_W) -> tuple:
    """Full pipeline; returns (audio_np, y_lengths, z, attn_squeezed, w_ceil)."""
    x, x_lengths = prepare_input(phoneme_ids)

    x_encoded, m_p, logs_p, x_mask = comp.encoder(x, x_lengths)

    y_lengths, w_ceil = comp.sdp(
        x_encoded, x_mask,
        np.array([length_scale], dtype=np.float32),
        np.array([noise_scale_w], dtype=np.float32),
    )

    yl_int = int(y_lengths[0])
    attn_squeezed, y_mask = comp.monotonic_aligner(w_ceil, x_mask, np.array([yl_int], dtype=np.int32))

    z = comp.flow(m_p, logs_p, y_mask, attn_squeezed,
                  np.array([noise_scale], dtype=np.float32))

    # sliding-window decode
    dec_win_len = MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP
    z_buf = np.zeros((1, ENCODER_HIDDEN_DIM, dec_win_len), dtype=np.float32)
    first_len = min(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP, yl_int)
    z_buf[:, :, :first_len] = z[:, :, :first_len]
    audio_chunk = comp.decoder(z_buf)
    audio = audio_chunk.squeeze()[:MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR]
    total = MAX_DEC_SEQ_LEN
    while total < yl_int:
        start_f = total - DEC_SEQ_OVERLAP
        end_f = total + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP
        actual_end = min(end_f, yl_int, z.shape[2])
        z_buf = np.zeros((1, ENCODER_HIDDEN_DIM, dec_win_len), dtype=np.float32)
        if start_f < z.shape[2] and actual_end > start_f:
            valid_span = actual_end - start_f
            z_buf[:, :, :valid_span] = z[:, :, start_f:actual_end]
        audio_chunk = comp.decoder(z_buf)
        chunk_valid_frames = min(MAX_DEC_SEQ_LEN, max(0, yl_int - total))
        valid_audio = audio_chunk.squeeze()[DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR:
                                            (DEC_SEQ_OVERLAP + chunk_valid_frames) * UPSAMPLE_FACTOR]
        audio = np.concatenate([audio, valid_audio])
        total += MAX_DEC_SEQ_LEN

    audio = audio[:yl_int * UPSAMPLE_FACTOR]
    return audio, yl_int, z, attn_squeezed, w_ceil


def main():
    parser = argparse.ArgumentParser(description="Piper (vi) split-model synthesis")
    parser.add_argument("--onnx_dir", type=Path,
                        default=Path("outputs/piper_vi_npu/components"))
    parser.add_argument("--config", type=Path,
                        default=Path("outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx.json"))
    parser.add_argument("--text", default="Xin chào thế giới")
    parser.add_argument("--backend", default="ort", choices=["ort", "torch"])
    parser.add_argument("--out_wav", type=Path, default=Path("outputs/piper_vi_npu/test_synth.wav"))
    args = parser.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]

    comp = ComponentRunner(args.onnx_dir, args.backend)
    phoneme_ids = phonemize_vi(args.text, id_map)
    logger.info("phonemes: %d", len(phoneme_ids))

    audio, y_lengths, *_ = synthesize(comp, phoneme_ids)
    logger.info("audio: %d samples (%.2fs), y_lengths=%d",
                len(audio), len(audio) / SAMPLE_RATE, y_lengths)

    args.out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(args.out_wav), audio.astype(np.float32), SAMPLE_RATE)
    except ImportError:
        import wave
        audio_clipped = np.clip(audio, -1.0, 1.0)
        int16_data = (audio_clipped * 32767.0).astype(np.int16)
        with wave.open(str(args.out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(int16_data.tobytes())
    logger.info("saved %s", args.out_wav)


if __name__ == "__main__":
    main()

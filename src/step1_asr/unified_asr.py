"""Unified ASR Pipeline for OneVoice.
Automatically routes audio to Zipformer-30M (for Vietnamese) 
or SenseVoice-Small (for English/Chinese/Korean).
Supports optional GTCRN denoising before transcription.
"""
import os
import sys
import tempfile
import torch

# Ensure src/ is in the python path to import from other modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from common import SR, get_device, load_wav, save_wav
except ImportError:
    from src.common import SR, get_device, load_wav, save_wav

# Import model loading functions from existing scripts
try:
    from step1_asr.test_asr_zipformer import find_model_files
    from step1_asr.test_asr_multi import MODEL_ID_HF, MODEL_ID_MODELSCOPE, run_asr as run_sensevoice
except ImportError:
    try:
        from test_asr_zipformer import find_model_files
        from test_asr_multi import MODEL_ID_HF, MODEL_ID_MODELSCOPE, run_asr as run_sensevoice
    except ImportError:
        from src.step1_asr.test_asr_zipformer import find_model_files
        from src.step1_asr.test_asr_multi import MODEL_ID_HF, MODEL_ID_MODELSCOPE, run_asr as run_sensevoice

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class UnifiedASRPipeline:
    def __init__(self, use_denoiser=False):
        self.device = get_device()
        self.device_str = "cuda:0" if self.device.type == "cuda" else "cpu"
        self.use_denoiser = use_denoiser
        
        print(f"[UnifiedASRPipeline] Initializing on {self.device_str}... (Denoiser: {self.use_denoiser})")
        
        if self.use_denoiser:
            sys.path.insert(0, os.path.join(ROOT, "third_party", "gtcrn"))
            from gtcrn import GTCRN
            self.denoiser = GTCRN().eval().to(self.device)
            ckpt_path = os.path.join(ROOT, "third_party", "gtcrn", "checkpoints", "model_trained_on_dns3.tar")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.denoiser.load_state_dict(ckpt["model"])
            print("[UnifiedASRPipeline] GTCRN Denoiser loaded.")
        
        # Load Zipformer (Vietnamese)
        import sherpa_onnx
        encoder, decoder, joiner, tokens = find_model_files()
        self.zipformer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=2,
            sample_rate=SR,
            feature_dim=80,
            decoding_method="greedy_search",
            provider="cuda" if self.device.type == "cuda" else "cpu",
        )
        print("[UnifiedASRPipeline] Zipformer-30M (Vi) loaded.")
        
        # Load SenseVoice (En/Zh/Ko)
        from funasr import AutoModel
        try:
            self.sensevoice = AutoModel(model=MODEL_ID_HF, hub="hf", device=self.device_str)
            print("[UnifiedASRPipeline] SenseVoice-Small (En/Zh/Ko) loaded via HuggingFace.")
        except TypeError:
            self.sensevoice = AutoModel(model=MODEL_ID_MODELSCOPE, device=self.device_str)
            print("[UnifiedASRPipeline] SenseVoice-Small (En/Zh/Ko) loaded via ModelScope.")

    def _denoise_audio(self, wav_np):
        n_fft, hop, win = 512, 256, 512
        window = torch.hann_window(win).pow(0.5).to(self.device)
        wav = torch.from_numpy(wav_np).to(self.device)
        spec_complex = torch.stft(wav, n_fft, hop, win, window, return_complex=True)
        spec = torch.view_as_real(spec_complex)
        with torch.no_grad():
            out_spec = self.denoiser(spec[None])[0]
        out_complex = torch.view_as_complex(out_spec.contiguous())
        enh = torch.istft(out_complex, n_fft, hop, win, window)
        return enh.detach().cpu().numpy()

    def transcribe(self, path: str, lang: str) -> str:
        """Transcribes the audio file located at `path` using the model appropriate for `lang`."""
        if self.use_denoiser:
            wav = load_wav(path)
            enhanced = self._denoise_audio(wav)
            # save to a temp file for SenseVoice
            temp_path = os.path.join(tempfile.gettempdir(), f"temp_denoised_{os.path.basename(path)}")
            save_wav(temp_path, enhanced)
            path_to_transcribe = temp_path
            wav_to_transcribe = enhanced
        else:
            path_to_transcribe = path
            wav_to_transcribe = load_wav(path)

        res = ""
        try:
            if lang == "vi":
                stream = self.zipformer.create_stream()
                stream.accept_waveform(SR, wav_to_transcribe)
                self.zipformer.decode_stream(stream)
                res = stream.result.text.strip()
            elif lang in ["en", "zh", "ko"]:
                res = run_sensevoice(self.sensevoice, path_to_transcribe, lang)
            else:
                raise ValueError(f"Unsupported language requested: {lang}")
        finally:
            if self.use_denoiser and os.path.exists(path_to_transcribe):
                os.remove(path_to_transcribe)
                
        return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified ASR Transcription CLI")
    parser.add_argument("audio_path", nargs="?", default="data/asr/vi/vi_0.wav", help="Path to audio file")
    parser.add_argument("--lang", default="vi", choices=["vi", "en", "zh", "ko"], help="Language of audio (vi/en/zh/ko)")
    parser.add_argument("--denoise", action="store_true", help="Enable GTCRN denoiser")
    args = parser.parse_args()

    target_audio = os.path.join(ROOT, args.audio_path) if not os.path.isabs(args.audio_path) else args.audio_path
    if not os.path.exists(target_audio):
        print(f"[ERROR] Audio file not found: {target_audio}")
    else:
        print(f"Transcribing '{target_audio}' with lang='{args.lang}' (denoiser={args.denoise})...")
        pipeline = UnifiedASRPipeline(use_denoiser=args.denoise)
        result_text = pipeline.transcribe(target_audio, args.lang)
        print("=" * 60)
        print(f"[TRANSCRIPTION ({args.lang.upper()})]: {result_text}")
        print("=" * 60)

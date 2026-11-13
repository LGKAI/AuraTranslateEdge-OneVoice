"""S2S engine adapter for the web demo.

The web server only talks to `get_engine()` and the three methods below.
To plug in real models, implement a class with the same interface (for example
by wrapping ZipformerVi / SenseVoiceFp32 / NllbMt / PiperVi / SupertonicTTS /
MeloTtsZh from src/step5_pipeline/pipeline_s2s.py) and return it from
`get_engine()`.

Interface
---------
    name: str                                  shown in the UI header
    asr(wav: bytes, lang: str) -> str          wav = 16 kHz mono PCM16 WAV file bytes
    mt(text: str, src: str, tgt: str) -> str
    tts(text: str, lang: str) -> bytes         returns a complete WAV file (any sample rate)

Language codes: "vi", "en", "zh", "ko". Only the directions in PAIRS are served.
"""
import io
import math
import os
import random
import time
import wave

LANGS = ("vi", "en", "zh", "ko")
# Supported directions: Vietnamese <-> each foreign language.
PAIRS = (("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"), ("vi", "ko"), ("ko", "vi"))


class MockEngine:
    """Placeholder engine: canned sentences + a synthetic voice-like tone.

    Sleeps a little at each stage so the UI's progress states are visible.
    """

    name = "Mock engine"

    _SAMPLES = [
        {
            "vi": "Xin chào, hôm nay dây chuyền số ba hoạt động bình thường.",
            "en": "Hello, production line three is running normally today.",
            "zh": "你好，今天三号生产线运行正常。",
            "ko": "안녕하세요, 오늘 3번 생산 라인은 정상적으로 가동되고 있습니다.",
        },
        {
            "vi": "Vui lòng đeo mũ bảo hộ trước khi vào khu vực này.",
            "en": "Please wear a safety helmet before entering this area.",
            "zh": "进入该区域前请佩戴安全帽。",
            "ko": "이 구역에 들어가기 전에 안전모를 착용해 주세요.",
        },
        {
            "vi": "Lô hàng này cần được giao trước ba giờ chiều.",
            "en": "This shipment needs to be delivered before three in the afternoon.",
            "zh": "这批货需要在下午三点前送达。",
            "ko": "이 화물은 오후 3시 전에 배송되어야 합니다.",
        },
        {
            "vi": "Máy này đang quá nhiệt, hãy tắt nó ngay.",
            "en": "This machine is overheating, turn it off right away.",
            "zh": "这台机器过热了，请立即关掉。",
            "ko": "이 기계가 과열되고 있으니 즉시 꺼 주세요.",
        },
    ]

    def __init__(self):
        self._i = -1
        self._current = self._SAMPLES[0]

    def asr(self, wav, lang):
        time.sleep(random.uniform(0.35, 0.6))
        self._i = (self._i + 1) % len(self._SAMPLES)
        self._current = self._SAMPLES[self._i]
        return self._current[lang]

    def mt(self, text, src, tgt):
        time.sleep(random.uniform(0.25, 0.45))
        for s in self._SAMPLES:
            if s.get(src) == text:
                return s[tgt]
        return f"[{tgt}] {text}"

    def tts(self, text, lang):
        time.sleep(random.uniform(0.3, 0.5))
        return _babble_wav(text)


def _babble_wav(text, sr=16000):
    """Soft syllable-like tone bursts, length proportional to text."""
    n_syll = max(4, min(28, len(text) // 3))
    rng = random.Random(len(text))
    frames = bytearray()
    for _ in range(n_syll):
        dur = rng.uniform(0.11, 0.19)
        f0 = rng.uniform(170, 240)
        n = int(dur * sr)
        for i in range(n):
            t = i / sr
            env = math.sin(math.pi * i / n) ** 2
            f = f0 * (1 + 0.06 * math.sin(2 * math.pi * 5 * t))
            s = (math.sin(2 * math.pi * f * t) + 0.35 * math.sin(4 * math.pi * f * t)
                 + 0.15 * math.sin(6 * math.pi * f * t))
            v = int(max(-1, min(1, 0.22 * env * s)) * 32767)
            frames += v.to_bytes(2, "little", signed=True)
        frames += b"\x00\x00" * int(rng.uniform(0.03, 0.08) * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _find_s2s_root():
    """Folder that contains the s2s_edge Python package: $S2S_PKG, the installed package folder (web_demo/ copied next to s2s_edge/),
    or the repo layout (repo/s2s_edge/s2s_edge)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.environ.get("S2S_PKG"), os.path.dirname(here), os.path.join(os.path.dirname(here), "s2s_edge")):
        if c and os.path.isfile(os.path.join(c, "s2s_edge", "__init__.py")):
            return os.path.abspath(c)
    return None


def _wav_to_f32(data):
    """WAV bytes (any rate, mono/stereo, PCM16/PCM32/float) -> float32 mono at 16 kHz."""
    import numpy as np
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes(); raw = w.readframes(n)
    if sw == 2: x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    elif sw == 4: x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1: x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    else: raise ValueError(f"unsupported WAV sample width {sw}")
    if ch > 1: x = x.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 16000); x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def _f32_to_wav(x, sr):
    import numpy as np
    pcm = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sr)); w.writeframes(pcm)
    return buf.getvalue()


class S2SEngine:
    """The real pipeline (s2s_edge): NPU on the AIBOX 8550 (backend npu), ONNX on a laptop (backend cpu, needs the ONNX exports).
    Same code path as the CLI / box web UI: VAD segments -> ASR -> junk filter (clean_asr) -> Vietnamese punctuation -> MT (3.3B / 600M route,
    sentence by sentence, output guard, names, Korean politeness) -> TTS."""

    def __init__(self, models=None, backend=None):
        import sys
        root = _find_s2s_root()
        if root is None: raise RuntimeError("s2s_edge package not found: set S2S_PKG=<folder that contains s2s_edge/>")
        if root not in sys.path: sys.path.insert(0, root)
        from s2s_edge.registry import Registry
        from s2s_edge.conversation import Engine
        self.backend = backend or os.environ.get("S2S_BACKEND", "npu")
        models = models or os.environ.get("S2S_MODELS") or _default_models(root)
        self.reg = Registry(models, self.backend); self.e = Engine(self.reg); self.root, self.models = root, models
        self.name = f"s2s_edge · {self.backend.upper()}"
        if os.environ.get("S2S_WARMUP", "1") != "0":                     # load every graph once so the first request is not 1-2 min
            self.reg.warmup([p for p in PAIRS], log=lambda m: print("[warmup]", m, flush=True))
        st = self.e.mt.status() if hasattr(self.e.mt, "status") else {}
        if st: self.name += " · MT " + ("3.3B+600M" if st.get("big_in_use") else "600M") + (" (DEGRADED)" if st.get("degraded") else "")
        self.info = dict(root=root, models=os.path.abspath(models), backend=self.backend, mt=st)

    def asr(self, wav, lang):
        from s2s_edge.audio import segment_with_breaks
        from s2s_edge.asr_clean import clean_asr
        x = _wav_to_f32(wav); a = self.e.asr(lang); out = []
        for (s, t), brk in segment_with_breaks(x, max_s=getattr(a, "max_s", None), vad_profile=self.reg.vad_profile):
            seg = self.e.prep(x[s:t]); text, raw, we = self.e._asr_stage(seg, lang)
            text, _ = clean_asr(text, lang)
            if text: text, _ = self.e._punct_stage(text, lang, raw, we, brk, min_chars=1)
            if text: out.append(text)
        return ("" if lang == "zh" else " ").join(out)

    def mt(self, text, src, tgt):
        return self.e.translate_text(text, src, tgt) if text.strip() else ""

    def tts(self, text, lang):
        if not text.strip(): return _f32_to_wav([], 16000)
        wav, sr = self.e.tts(lang).synthesize(text)
        return _f32_to_wav(wav, sr)


class LocalTorchEngine:
    """Host-machine local AI engine for Windows/laptop demos when AIBOX NPU binaries or full ONNX bundles are absent.

    Combines the integrated s2s_edge text/audio stages (normalize_level, clean_asr, vi_punct/restore_vi, split_sentences,
    clean_mt, ensure_polite_ko, tts_format) with local PyTorch/ONNX models (Zipformer-30M, SenseVoiceSmall, NLLB-200-600M,
    Piper, Supertonic).
    """

    def __init__(self):
        import sys
        root = _find_s2s_root()
        if root and root not in sys.path:
            sys.path.insert(0, root)
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._zipformer = None
        self._sv = None
        self._sv_post = None
        self._nllb_tok = None
        self._nllb_model = None
        self._piper = {}
        self._supertonic = None
        self._st_style = None
        dev = "CPU"
        try:
            import torch
            if torch.cuda.is_available():
                dev = "CUDA"
        except Exception:
            pass
        import importlib.util
        melo_src = os.path.join(self.repo_root, "src", "step4_quantization", "tts", "melotts", "MeloTTS")
        melo_onnx = self._find_melo_onnx()
        melo_ok = (importlib.util.find_spec("melo") is not None) or os.path.isdir(melo_src) or (melo_onnx is not None)
        self.name = f"s2s_edge · LOCAL ({dev})"
        self.info = dict(
            mode="local",
            repo_root=self.repo_root,
            device=dev.lower(),
            asr="Zipformer / SenseVoice",
            mt="NLLB-200-600M",
            tts="Piper (VI) · MeloTTS (ZH) · Supertonic (EN/KO)",
            melo_installed=melo_ok,
        )

    def _find_melo_onnx(self):
        for rel in (
            os.path.join(self.repo_root, "build", "melo", "melo_one.onnx"),
            os.path.join(os.path.dirname(self.repo_root), "build", "melo", "melo_one.onnx"),
            os.path.join(self.repo_root, "models", "melo_one.onnx"),
        ):
            if os.path.isfile(rel):
                return rel
        return None

    def _get_zipformer(self):
        if self._zipformer is None:
            import sherpa_onnx
            snap_base = os.path.join(self.repo_root, "third_party", "zipformer",
                                     "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
            cands = [os.path.join(snap_base, d) for d in os.listdir(snap_base)] if os.path.isdir(snap_base) else []
            sdir = next((c for c in cands if os.path.isfile(os.path.join(c, "encoder-epoch-20-avg-10.int8.onnx"))
                         or os.path.isfile(os.path.join(c, "encoder-epoch-20-avg-10.onnx"))), "")
            if not sdir:
                raise RuntimeError("Zipformer checkpoint not found under third_party/zipformer")
            use_int8 = os.path.isfile(os.path.join(sdir, "encoder-epoch-20-avg-10.int8.onnx"))
            sfx = ".int8.onnx" if use_int8 else ".onnx"
            tok = os.path.join(sdir, "tokens.txt")
            if not os.path.isfile(tok):
                tok = os.path.join(sdir, "tokens.generated.txt")
            self._zipformer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=os.path.join(sdir, f"encoder-epoch-20-avg-10{sfx}"),
                decoder=os.path.join(sdir, f"decoder-epoch-20-avg-10{sfx}"),
                joiner=os.path.join(sdir, f"joiner-epoch-20-avg-10{sfx}"),
                tokens=tok, num_threads=4, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
            )
        return self._zipformer

    def _get_sensevoice(self):
        if self._sv is None:
            print("[local] Loading SenseVoiceSmall (first request)...", flush=True)
            import torch
            from funasr import AutoModel
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                self._sv = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device=dev, disable_update=True)
            except TypeError:
                self._sv = AutoModel(model="iic/SenseVoiceSmall", device=dev, disable_update=True)
            self._sv_post = rich_transcription_postprocess
        return self._sv, self._sv_post

    def _get_nllb(self):
        if self._nllb_model is None:
            print("[local] Loading NLLB-200-600M (first request, may take ~30s on Windows)...", flush=True)
            import torch
            import transformers.modeling_utils
            import transformers.utils.import_utils
            for mod in (transformers.modeling_utils, transformers.utils.import_utils):
                if hasattr(mod, "check_torch_load_is_safe"):
                    mod.check_torch_load_is_safe = lambda: None
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                self._nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", local_files_only=True)
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", local_files_only=True).to(dev)
            except Exception:
                self._nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
                self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(dev)
        return self._nllb_model, self._nllb_tok

    def _get_piper(self, lang):
        if lang not in self._piper:
            from piper import PiperVoice
            local_vi = os.path.join(self.repo_root, "src", "step3_tts", "vi_VN-vais1000-medium.onnx")
            if lang == "vi" and os.path.isfile(local_vi):
                cfg = local_vi + ".json"
                self._piper[lang] = PiperVoice.load(local_vi, config_path=cfg if os.path.isfile(cfg) else None)
            else:
                from huggingface_hub import hf_hub_download
                voices = {
                    "vi": ("vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx", "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"),
                    "en": ("en/en_US/amy/medium/en_US-amy-medium.onnx", "en/en_US/amy/medium/en_US-amy-medium.onnx.json"),
                    "zh": ("zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx", "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"),
                }
                sub_onnx, sub_json = voices.get(lang, voices["en"])
                try:
                    p_onnx = hf_hub_download("rhasspy/piper-voices", sub_onnx, local_files_only=True)
                    p_json = hf_hub_download("rhasspy/piper-voices", sub_json, local_files_only=True)
                except Exception:
                    p_onnx = hf_hub_download("rhasspy/piper-voices", sub_onnx)
                    p_json = hf_hub_download("rhasspy/piper-voices", sub_json)
                self._piper[lang] = PiperVoice.load(p_onnx, config_path=p_json)
        return self._piper[lang]

    def _get_supertonic(self):
        if self._supertonic is None:
            from supertonic import TTS
            cache_dir = os.path.expanduser("~/.cache/supertonic3")
            deploy_dir = os.path.join(self.repo_root, "outputs", "supertonic-deploy")
            model_dir = deploy_dir if os.path.isdir(deploy_dir) else (cache_dir if os.path.isdir(cache_dir) else None)
            self._supertonic = TTS(model_dir=model_dir, auto_download=model_dir is None) if model_dir else TTS(auto_download=True)
            self._st_style = self._supertonic.get_voice_style(voice_name="M1")
        return self._supertonic, self._st_style

    def asr(self, wav, lang):
        from s2s_edge.asr_clean import clean_asr
        from s2s_edge.audio import normalize_level, segment
        from s2s_edge.conversation import restore_vi
        x = normalize_level(_wav_to_f32(wav))
        spans = segment(x, vad_profile="hybrid") or [(0, len(x))]
        out = []
        if lang == "vi":
            rec = self._get_zipformer()
            for s, t in spans:
                st = rec.create_stream()
                st.accept_waveform(16000, x[s:t])
                rec.decode_stream(st)
                raw = st.result.text.strip().lower()
                txt, _ = clean_asr(raw, "vi")
                if txt:
                    out.append(restore_vi(txt))
            return " ".join(out)
        sv, post = self._get_sensevoice()
        for s, t in spans:
            res = sv.generate(input=x[s:t], cache={}, language=lang, use_itn=True)
            raw = post(res[0]["text"]).strip()
            txt, _ = clean_asr(raw, lang)
            if txt:
                out.append(txt)
        return ("" if lang == "zh" else " ").join(out)

    def mt(self, text, src, tgt):
        if not text.strip():
            return ""
        from s2s_edge.asr_clean import clean_mt
        from s2s_edge.ko_polite import polite
        from s2s_edge.mt import split_sentences
        codes = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
        model, tok = self._get_nllb()
        tok.src_lang = codes[src]
        tgt_id = tok.convert_tokens_to_ids(codes[tgt])
        pieces = []
        for unit in split_sentences(text, src):
            inputs = tok(unit, return_tensors="pt").to(model.device)
            out_ids = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=128)
            raw = tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
            cleaned, _ = clean_mt(raw, unit, tgt)
            if cleaned:
                if tgt == "ko":
                    cleaned = polite(cleaned)
                pieces.append(cleaned)
        return ("" if tgt == "zh" else " ").join(pieces)

    def _get_melo_zh(self):
        if getattr(self, "_melo_zh", None) is None:
            import sys
            melo_repo = os.path.join(self.repo_root, "src", "step4_quantization", "tts", "melotts", "MeloTTS")
            if os.path.isdir(melo_repo) and melo_repo not in sys.path:
                sys.path.insert(0, melo_repo)
            try:
                import torch
                from melo.api import TTS
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                m = TTS(language="ZH", device=dev)
                if hasattr(m.hps, "data"):
                    m.hps.data.disable_bert = True
                spk = list(m.hps.data.spk2id.values())[0]
                self._melo_zh = ("torch", m, spk, m.hps.data.sampling_rate)
            except Exception as torch_exc:
                onnx_path = self._find_melo_onnx()
                if onnx_path:
                    from s2s_edge.backends import make_runner
                    from s2s_edge.tts import TtsZh
                    self._melo_zh = ("onnx", TtsZh(make_runner("tts_zh", {"onnx": onnx_path}, "cpu")))
                else:
                    raise RuntimeError(f"MeloTTS ZH chưa sẵn sàng ({torch_exc})") from torch_exc
        return self._melo_zh

    def tts(self, text, lang):
        import numpy as np
        if not text.strip():
            return _f32_to_wav([], 16000)
        try:
            from s2s_edge.tts_format import prepare as prepare_tts
            text, _ = prepare_tts(text, lang)
        except Exception:
            pass
        if lang == "zh":
            melo = self._get_melo_zh()
            if melo[0] == "torch":
                _, m, spk, sr = melo
                wav = m.tts_to_file(text, spk, output_path=None, speed=1.0, quiet=True)
                return _f32_to_wav(np.asarray(wav, np.float32).squeeze(), sr)
            _, tts_zh = melo
            wav, sr = tts_zh.synthesize(text)
            return _f32_to_wav(wav, sr)
        if lang in ("en", "ko"):
            try:
                st, style = self._get_supertonic()
                wav, _ = st.synthesize(text=text, lang=lang, voice_style=style, total_steps=5, speed=1.0)
                return _f32_to_wav(np.asarray(wav).squeeze(), getattr(st, "sample_rate", 44100))
            except Exception as exc:
                if lang == "ko":
                    raise RuntimeError(f"Supertonic KO TTS failed: {exc}") from exc
        voice = self._get_piper(lang)
        raw_bytes = b"".join(c.audio_int16_bytes for c in voice.synthesize(text))
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return _f32_to_wav(audio, voice.config.sample_rate)


def _default_models(root):
    """The model set tools/session0.sh validated (logs/models_ok.txt), else models_qcs8550/models.json (3.3B route + 600M)."""
    ok = os.path.join(root, "logs", "models_ok.txt")
    if os.path.isfile(ok):
        p = open(ok, encoding="utf-8").read().strip()
        if p: return p if os.path.isabs(p) else os.path.join(root, p)
    return os.path.join(root, "models_qcs8550", "models.json")


def get_engine():
    """S2S_ENGINE=real -> S2SEngine (AIBOX NPU or CPU ONNX), local -> LocalTorchEngine (Windows/dev host models),
    mock -> MockEngine (UI testing without models)."""
    import sys
    env_want = os.environ.get("S2S_ENGINE")
    if env_want:
        want = env_want.lower()
    elif sys.platform == "win32" and not os.environ.get("S2S_MODELS") and not os.environ.get("S2S_BACKEND"):
        print("[web_demo] Windows host without S2S_MODELS/S2S_BACKEND -> using LocalTorchEngine "
              "(set S2S_ENGINE=mock for UI test or S2S_ENGINE=real + S2S_BACKEND=cpu for ONNX)", flush=True)
        want = "local"
    else:
        want = "real" if _find_s2s_root() else "mock"
    if want == "mock":
        return MockEngine()
    if want == "local":
        return LocalTorchEngine()
    return S2SEngine()

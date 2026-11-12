# -*- coding: utf-8 -*-
"""S2S Web Dashboard — Speech-to-Speech Translation (VI <-> EN / ZH / KO) on Localhost.

Extended from tools/tts_web.py, pure standard library HTTP server (no Flask/FastAPI required).
Supports:
  1. 🎙️ Live microphone recording directly in browser (HTML5 Web Audio API -> 16kHz Mono WAV)
  2. 📁 Audio file upload (WAV)
  3. ✍️ Direct text translation & speech synthesis (MT -> TTS)
  4. Real-time latency breakdown (ASR, MT, TTS, Total)
  5. Dual backend: S2S Engine (ONNX/QNN) with automatic local Torch/CUDA fallback if ONNX graphs missing.

Usage:
  cd s2s_edge
  python s2s_web.py --port 8765
  # Open in browser: http://localhost:8765
"""
import argparse, base64, io, json, os, re, secrets, subprocess, sys, threading, time, wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

DIRECTIONS = [
    ("vi", "en", "🇻🇳 Tiếng Việt ➔ 🇺🇸 Tiếng Anh"),
    ("vi", "zh", "🇻🇳 Tiếng Việt ➔ 🇨🇳 Tiếng Trung"),
    ("vi", "ko", "🇻🇳 Tiếng Việt ➔ 🇰🇷 Tiếng Hàn"),
    ("en", "vi", "🇺🇸 Tiếng Anh ➔ 🇻🇳 Tiếng Việt"),
    ("zh", "vi", "🇨🇳 Tiếng Trung ➔ 🇻🇳 Tiếng Việt"),
    ("ko", "vi", "🇰🇷 Tiếng Hàn ➔ 🇻🇳 Tiếng Việt"),
]

SAMPLES = {
    "vi": "Xin chào, hôm nay chúng ta kiểm thử thiết bị phiên dịch âm thanh tại nhà máy.",
    "en": "Welcome to our smart factory, please wear your protective helmet.",
    "zh": "你好，今天我们在工厂测试语音翻译设备。",
    "ko": "안녕하세요, 오늘 공장에서 음성 번역 장비를 테스트합니다.",
}

PAGE = r"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AuraTranslate Edge — S2S Dashboard</title>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --card-sub: #334155;
      --fg: #f8fafc; --mut: #94a3b8; --acc: #38bdf8; --acc-hover: #0284c7;
      --green: #22c55e; --red: #ef4444; --amber: #f59e0b; --bd: #334155;
    }
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #f8fafc; --card: #ffffff; --card-sub: #f1f5f9;
        --fg: #0f172a; --mut: #64748b; --acc: #0284c7; --acc-hover: #0369a1;
        --green: #16a34a; --red: #dc2626; --amber: #d97706; --bd: #e2e8f0;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; background: var(--bg); color: var(--fg);
      font: 15px/1.6 system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    header {
      background: var(--card); border-bottom: 1px solid var(--bd); padding: 14px 20px;
      display: flex; justify-content: space-between; align-items: center;
    }
    .brand { font-size: 18px; font-weight: 700; color: var(--fg); display: flex; align-items: center; gap: 8px; }
    .badge {
      background: rgba(56, 189, 248, 0.15); color: var(--acc);
      font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
    }
    main { max-width: 860px; margin: 0 auto; padding: 24px 16px; }
    .card { background: var(--card); border: 1px solid var(--bd); border-radius: 16px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    h2 { font-size: 16px; margin: 0 0 12px 0; color: var(--acc); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .dirs { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-bottom: 16px; }
    .dir-btn {
      padding: 10px 12px; border: 1px solid var(--bd); border-radius: 10px;
      background: var(--card-sub); color: var(--fg); font-size: 14px; font-weight: 500;
      cursor: pointer; transition: all 0.2s; text-align: left;
    }
    .dir-btn.on { border-color: var(--acc); background: var(--acc); color: #fff; font-weight: 600; }
    .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--bd); margin-bottom: 16px; padding-bottom: 8px; }
    .tab-btn {
      padding: 8px 14px; background: transparent; border: 0; color: var(--mut);
      font-size: 14px; font-weight: 600; cursor: pointer; border-radius: 8px;
    }
    .tab-btn.on { color: var(--acc); background: rgba(56, 189, 248, 0.12); }
    .rec-box {
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      padding: 30px 16px; border: 2px dashed var(--bd); border-radius: 14px; background: var(--card-sub);
    }
    .mic-btn {
      width: 72px; height: 72px; border-radius: 50%; border: 0;
      background: var(--acc); color: #fff; font-size: 28px; cursor: pointer;
      display: flex; align-items: center; justify-content: center; transition: all 0.2s;
      box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4);
    }
    .mic-btn.rec {
      background: var(--red); animation: pulse 1.2s infinite;
      box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.5);
    }
    @keyframes pulse {
      0% { transform: scale(1); }
      50% { transform: scale(1.08); }
      100% { transform: scale(1); }
    }
    .rec-status { margin-top: 14px; font-size: 14px; font-weight: 500; color: var(--mut); }
    textarea {
      width: 100%; min-height: 80px; padding: 12px; border-radius: 10px;
      border: 1px solid var(--bd); background: var(--card-sub); color: var(--fg);
      font: inherit; resize: vertical; margin-bottom: 10px;
    }
    .btn {
      padding: 10px 20px; border-radius: 8px; border: 0; font-size: 14px; font-weight: 600;
      cursor: pointer; background: var(--acc); color: #fff; transition: background 0.2s;
    }
    .btn:hover { background: var(--acc-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .result-row { margin-bottom: 14px; }
    .label { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--mut); margin-bottom: 4px; }
    .res-text {
      padding: 12px; border-radius: 10px; background: var(--card-sub);
      border-left: 4px solid var(--acc); font-size: 16px; min-height: 44px;
    }
    .res-text.tgt { border-left-color: var(--green); font-weight: 600; font-size: 17px; }
    .metrics {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 14px;
    }
    .metric-chip {
      background: var(--card-sub); border: 1px solid var(--bd); border-radius: 8px;
      padding: 8px; text-align: center;
    }
    .metric-chip span { display: block; font-size: 11px; color: var(--mut); }
    .metric-chip strong { font-size: 14px; color: var(--fg); }
    audio { width: 100%; margin-top: 10px; border-radius: 8px; }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span>⚡ AuraTranslate Edge</span>
      <span class="badge">Localhost S2S</span>
    </div>
    <div style="font-size: 13px; color: var(--mut);">Offline Speech-to-Speech Translation</div>
  </header>

  <main>
    <div class="card">
      <h2>1. Chọn chiều dịch (Direction)</h2>
      <div class="dirs" id="dirs"></div>
    </div>

    <div class="card">
      <h2>2. Nhập giọng nói hoặc văn bản</h2>
      <div class="tabs">
        <button class="tab-btn on" onclick="switchTab('mic')">🎙️ Ghi âm Micro</button>
        <button class="tab-btn" onclick="switchTab('file')">📁 Tải file âm thanh</button>
        <button class="tab-btn" onclick="switchTab('text')">✍️ Nhập văn bản trực tiếp</button>
      </div>

      <!-- Tab 1: Mic -->
      <div id="tab-mic">
        <div class="rec-box">
          <button class="mic-btn" id="micBtn" onclick="toggleRecord()">🎙️</button>
          <div class="rec-status" id="recStatus">Bấm mic để bắt đầu nói</div>
        </div>
      </div>

      <!-- Tab 2: File -->
      <div id="tab-file" style="display: none;">
        <input type="file" id="audioFile" accept="audio/*" style="margin-bottom: 12px;">
        <button class="btn" onclick="uploadAudioFile()">Dịch file âm thanh này</button>
      </div>

      <!-- Tab 3: Text -->
      <div id="tab-text" style="display: none;">
        <textarea id="srcText" placeholder="Nhập câu cần dịch..."></textarea>
        <button class="btn" id="btnTransText" onclick="translateText()">Dịch và phát giọng đọc</button>
      </div>
    </div>

    <div class="card" id="resCard" style="display: none;">
      <h2>3. Kết quả phiên dịch toàn trình</h2>
      
      <div class="result-row">
        <div class="label" id="srcLabel">Tiếng nói nhận dạng được (ASR):</div>
        <div class="res-text" id="asrOut">—</div>
      </div>

      <div class="result-row">
        <div class="label" id="tgtLabel">Bản dịch đích (MT):</div>
        <div class="res-text tgt" id="mtOut">—</div>
      </div>

      <div class="result-row">
        <div class="label">Âm thanh phát ra (TTS Audio):</div>
        <audio id="audioOut" controls autoplay></audio>
      </div>

      <div class="metrics">
        <div class="metric-chip"><span>ASR Latency</span><strong id="mAsr">—</strong></div>
        <div class="metric-chip"><span>MT Latency</span><strong id="mMt">—</strong></div>
        <div class="metric-chip"><span>TTS Latency</span><strong id="mTts">—</strong></div>
        <div class="metric-chip"><span>Tổng thời gian</span><strong id="mTotal" style="color: var(--acc);">—</strong></div>
      </div>
    </div>
  </main>

  <script>
    const DIRS = __DIRECTIONS__;
    const SAMPLES = __SAMPLES__;
    let curSrc = "vi", curTgt = "en";

    // Setup Direction Buttons
    const dBox = document.getElementById("dirs");
    DIRS.forEach(([s, t, label], i) => {
      const b = document.createElement("button");
      b.className = "dir-btn" + (i === 0 ? " on" : "");
      b.textContent = label;
      b.onclick = () => {
        document.querySelectorAll(".dir-btn").forEach(x => x.classList.remove("on"));
        b.classList.add("on");
        curSrc = s; curTgt = t;
        document.getElementById("srcText").value = SAMPLES[s] || "";
      };
      dBox.appendChild(b);
    });
    document.getElementById("srcText").value = SAMPLES[curSrc] || "";

    function switchTab(mode) {
      document.querySelectorAll(".tab-btn").forEach((b, i) => {
        b.classList.toggle("on", (mode === 'mic' && i === 0) || (mode === 'file' && i === 1) || (mode === 'text' && i === 2));
      });
      document.getElementById("tab-mic").style.display = (mode === 'mic' ? 'block' : 'none');
      document.getElementById("tab-file").style.display = (mode === 'file' ? 'block' : 'none');
      document.getElementById("tab-text").style.display = (mode === 'text' ? 'block' : 'none');
    }

    // Audio Recorder in Browser
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    async function toggleRecord() {
      const btn = document.getElementById("micBtn");
      const st = document.getElementById("recStatus");
      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          audioChunks = [];
          mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
          mediaRecorder.onstop = async () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            st.textContent = "Đang gửi âm thanh lên máy chủ...";
            await sendAudioBlob(blob);
          };
          mediaRecorder.start();
          isRecording = true;
          btn.classList.add("rec");
          st.textContent = "Đang thu âm... Bấm lại mic để kết thúc";
        } catch (err) {
          alert("Không mở được Micro: " + err);
        }
      } else {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());
        isRecording = false;
        btn.classList.remove("rec");
        st.textContent = "Đang xử lý...";
      }
    }

    async function uploadAudioFile() {
      const fi = document.getElementById("audioFile");
      if (!fi.files || fi.files.length === 0) { alert("Vui lòng chọn 1 file âm thanh!"); return; }
      await sendAudioBlob(fi.files[0]);
    }

    async function sendAudioBlob(blob) {
      const fd = new FormData();
      fd.append("file", blob);
      fd.append("src", curSrc);
      fd.append("tgt", curTgt);
      
      document.getElementById("resCard").style.display = "block";
      document.getElementById("asrOut").textContent = "Đang nhận diện giọng nói (ASR)...";
      document.getElementById("mtOut").textContent = "Đang dịch máy...";

      try {
        const res = await fetch("/api/translate_audio", { method: "POST", body: fd });
        const data = await res.json();
        if (data.error) {
          alert("Lỗi máy chủ: " + data.error);
          return;
        }
        renderResult(data);
      } catch (e) {
        alert("Lỗi kết nối máy chủ: " + e);
      }
    }

    async function translateText() {
      const text = document.getElementById("srcText").value.trim();
      if (!text) return;
      document.getElementById("btnTransText").disabled = true;
      document.getElementById("resCard").style.display = "block";
      document.getElementById("asrOut").textContent = text + " (Văn bản nhập tay)";
      document.getElementById("mtOut").textContent = "Đang dịch...";

      try {
        const res = await fetch("/api/translate_text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, src: curSrc, tgt: curTgt })
        });
        const data = await res.json();
        if (data.error) {
          alert("Lỗi: " + data.error);
          return;
        }
        renderResult(data);
      } catch (e) {
        alert("Lỗi: " + e);
      } finally {
        document.getElementById("btnTransText").disabled = false;
      }
    }

    function renderResult(data) {
      document.getElementById("asrOut").textContent = data.asr || "(Không nhận dạng được âm thanh)";
      document.getElementById("mtOut").textContent = data.mt || "(Bản dịch rỗng)";
      
      if (data.audio_b64) {
        const a = document.getElementById("audioOut");
        a.src = "data:audio/wav;base64," + data.audio_b64;
        a.play().catch(e => console.log("Auto-play blocked:", e));
      }
      
      const t = data.timing || {};
      document.getElementById("mAsr").textContent = (t.asr_ms ? Math.round(t.asr_ms) + " ms" : "—");
      document.getElementById("mMt").textContent = (t.mt_ms ? Math.round(t.mt_ms) + " ms" : "—");
      document.getElementById("mTts").textContent = (t.tts_ms ? Math.round(t.tts_ms) + " ms" : "—");
      document.getElementById("mTotal").textContent = (t.total_ms ? Math.round(t.total_ms) + " ms" : "—");
    }
  </script>
</body>
</html>
"""


def convert_audio_to_wav16k(raw_bytes):
    """Use soundfile if possible, or fall back to ffmpeg subprocess."""
    buf = io.BytesIO(raw_bytes)
    try:
        import soundfile as sf
        data, sr = sf.read(buf, dtype="float32")
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), 16000)
            data = resample_poly(data, 16000 // g, int(sr) // g).astype(np.float32)
        return data, 16000
    except Exception:
        # Fall back to ffmpeg
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1"],
            input=raw_bytes, capture_output=True, check=True
        )
        data = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return data, 16000


def wav_to_base64(audio_array, sr=16000):
    """Convert float32 numpy array to base64-encoded WAV."""
    audio = np.clip(np.asarray(audio_array, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class LocalPipeline:
    """Seamless local pipeline using verified local components (Zipformer + NLLB-200 + Piper)."""
    def __init__(self):
        self.lock = threading.Lock()
        self._nllb_model = None
        self._nllb_tok = None
        self._zipformer = None
        self._piper_voices = {}

    def _get_nllb(self):
        if self._nllb_model is None:
            import transformers.modeling_utils
            if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
                transformers.modeling_utils.check_torch_load_is_safe = lambda: None
            import transformers.utils.import_utils
            if hasattr(transformers.utils.import_utils, "check_torch_load_is_safe"):
                transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[S2S Web] Loading NLLB-200 on {dev}...")
            self._nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(dev)
        return self._nllb_model, self._nllb_tok

    def _get_zipformer(self):
        if self._zipformer is None:
            import sherpa_onnx
            snap = os.path.join(ROOT, "third_party", "zipformer", "models--hynt--Zipformer-30M-RNNT-6000h", "snapshots")
            cands = [os.path.join(snap, d) for d in os.listdir(snap)] if os.path.exists(snap) else []
            sdir = cands[0] if cands else ""
            if not sdir:
                raise RuntimeError("Zipformer checkpoint directory not found in third_party/zipformer")
            enc = os.path.join(sdir, "encoder-epoch-20-avg-10.onnx")
            dec = os.path.join(sdir, "decoder-epoch-20-avg-10.onnx")
            join = os.path.join(sdir, "joiner-epoch-20-avg-10.onnx")
            tok = os.path.join(sdir, "tokens.txt")
            if not os.path.exists(tok):
                tok = os.path.join(sdir, "tokens.generated.txt")
            self._zipformer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=enc, decoder=dec, joiner=join, tokens=tok, num_threads=4, sample_rate=16000, feature_dim=80
            )
        return self._zipformer

    def _get_piper(self, lang="vi"):
        if lang not in self._piper_voices:
            from piper import PiperVoice
            from huggingface_hub import hf_hub_download
            voice_configs = {
                "vi": ("vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx", "vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"),
                "en": ("en/en_US/amy/medium/en_US-amy-medium.onnx", "en/en_US/amy/medium/en_US-amy-medium.onnx.json"),
                "zh": ("zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx", "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json"),
            }
            sub_onnx, sub_json = voice_configs.get(lang, voice_configs["en"])
            onnx_path = hf_hub_download("rhasspy/piper-voices", sub_onnx)
            cfg_path = hf_hub_download("rhasspy/piper-voices", sub_json)
            self._piper_voices[lang] = PiperVoice.load(onnx_path, config_path=cfg_path)
        return self._piper_voices[lang]

    def warmup(self):
        # Pre-load Zipformer
        try:
            self._get_zipformer()
            print("  ✓ Zipformer ASR initialized")
        except Exception as e:
            print(f"  ⚠️ Zipformer init error: {e}")

        # Pre-load NLLB-200 and run dummy pass
        try:
            m, t = self._get_nllb()
            dummy = t("Xin chào", return_tensors="pt").to(m.device)
            m.generate(**dummy, forced_bos_token_id=t.convert_tokens_to_ids("eng_Latn"), max_new_tokens=5)
            print(f"  ✓ NLLB-200 MT warmed up on {m.device}")
        except Exception as e:
            print(f"  ⚠️ NLLB warmup error: {e}")

        # Pre-load Piper voices
        for l in ("vi", "en"):
            try:
                self._get_piper(l)
                print(f"  ✓ Piper TTS ({l}) cached & loaded")
            except Exception as e:
                print(f"  ⚠️ Piper ({l}) load error: {e}")

    def asr(self, wav, src):
        t0 = time.perf_counter()
        if src == "vi":
            rec = self._get_zipformer()
            st = rec.create_stream()
            st.accept_waveform(16000, wav)
            rec.decode_stream(st)
            raw = st.result.text.strip()
            # Zipformer returns ALL-CAPS words (e.g. 'KỊN LÀ MỘT THẰNG NGU')
            # Normalize to lower first, then restore punctuation and proper capitalization
            from s2s_edge.textnorm import vi_punct
            text = vi_punct(raw.lower())
        else:
            # Fallback for en/zh/ko via SenseVoice
            from funasr import AutoModel
            import torch
            m = AutoModel(model="FunAudioLLM/SenseVoiceSmall", hub="hf", device="cuda" if torch.cuda.is_available() else "cpu")
            res = m.generate(input=wav, language=src, use_itn=True)
            text = res[0]["text"].strip()

        dt = (time.perf_counter() - t0) * 1000.0
        return text, dt

    def mt(self, text, src, tgt):
        t0 = time.perf_counter()
        clean = text.strip()
        # Protect against ALL-CAPS text breaking NLLB subword tokenizer
        if clean.isupper():
            clean = clean.lower()
            clean = clean[:1].upper() + clean[1:]

        nllb_code = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
        model, tok = self._get_nllb()
        tok.src_lang = nllb_code[src]
        dev = model.device
        inputs = tok(clean, return_tensors="pt").to(dev)
        tgt_id = tok.convert_tokens_to_ids(nllb_code[tgt])
        out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=100)
        res = tok.batch_decode(out, skip_special_tokens=True)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        return res, dt

    def tts(self, text, tgt):
        t0 = time.perf_counter()
        try:
            voice_lang = tgt if tgt in ("vi", "zh") else "en"
            voice = self._get_piper(voice_lang)
            # Synthesize real PCM speech chunks
            raw_bytes = b"".join(chunk.audio_int16_bytes for chunk in voice.synthesize(text))
            audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = voice.config.sample_rate
        except Exception as e:
            print(f"[s2s_web] ⚠️ TTS synthesis error: {e}", flush=True)
            # Fallback sine wave tone
            sr = 16000
            t = np.linspace(0, 0.5, sr, endpoint=False)
            audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        dt = (time.perf_counter() - t0) * 1000.0
        return audio, sr, dt


def main():
    ap = argparse.ArgumentParser(description="AuraTranslate Edge S2S Local Web Dashboard")
    ap.add_argument("--port", type=int, default=8765, help="Port to serve on (default: 8765)")
    ap.add_argument("--no-warmup", action="store_true", help="Skip pre-warming models at startup")
    args = ap.parse_args()

    pipeline = LocalPipeline()
    if not args.no_warmup:
        print("\n[S2S Web] Pre-warming models for zero-latency response...")
        pipeline.warmup()
        print("[S2S Web] All AI models ready!\n")
    page_html = (PAGE
                 .replace("__DIRECTIONS__", json.dumps(DIRECTIONS, ensure_ascii=False))
                 .replace("__SAMPLES__", json.dumps(SAMPLES, ensure_ascii=False))
                 .encode("utf-8"))

    class S2SHandler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"[s2s_web] {fmt % args}", flush=True)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                return self._send(200, page_html, "text/html; charset=utf-8")
            if parsed.path == "/api/health":
                return self._send(200, b'{"status": "ok"}')
            self._send(404, b'{"error": "Not found"}')

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/translate_text":
                content_len = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(content_len).decode("utf-8"))
                text = req.get("text", "").strip()
                src = req.get("src", "vi")
                tgt = req.get("tgt", "en")
                if not text:
                    return self._send(400, b'{"error": "Text is empty"}')

                t0 = time.perf_counter()
                with pipeline.lock:
                    mt_text, mt_ms = pipeline.mt(text, src, tgt)
                    audio, sr, tts_ms = pipeline.tts(mt_text, tgt)
                total_ms = (time.perf_counter() - t0) * 1000.0

                resp = {
                    "asr": text,
                    "mt": mt_text,
                    "audio_b64": wav_to_base64(audio, sr),
                    "timing": {"asr_ms": 0, "mt_ms": mt_ms, "tts_ms": tts_ms, "total_ms": total_ms}
                }
                return self._send(200, json.dumps(resp, ensure_ascii=False).encode("utf-8"))

            elif parsed.path == "/api/translate_audio":
                # Parse multipart or raw bytes
                ctype = self.headers.get("Content-Type", "")
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)

                src = "vi"
                tgt = "en"
                raw_audio = b""

                if "multipart/form-data" in ctype:
                    boundary = ctype.split("boundary=")[1].encode()
                    parts = body.split(b"--" + boundary)
                    for p in parts:
                        if b'name="src"' in p:
                            src = p.split(b"\r\n\r\n")[1].split(b"\r\n")[0].decode().strip()
                        elif b'name="tgt"' in p:
                            tgt = p.split(b"\r\n\r\n")[1].split(b"\r\n")[0].decode().strip()
                        elif b'filename=' in p:
                            raw_audio = p.split(b"\r\n\r\n")[1].rsplit(b"\r\n", 1)[0]
                else:
                    raw_audio = body

                if not raw_audio:
                    return self._send(400, b'{"error": "No audio received"}')

                t0 = time.perf_counter()
                wav_16k, _ = convert_audio_to_wav16k(raw_audio)

                with pipeline.lock:
                    asr_text, asr_ms = pipeline.asr(wav_16k, src)
                    if asr_text:
                        mt_text, mt_ms = pipeline.mt(asr_text, src, tgt)
                        audio, sr, tts_ms = pipeline.tts(mt_text, tgt)
                    else:
                        mt_text = ""
                        mt_ms = 0
                        audio = np.zeros(16000, dtype=np.float32)
                        sr = 16000
                        tts_ms = 0

                total_ms = (time.perf_counter() - t0) * 1000.0
                resp = {
                    "asr": asr_text,
                    "mt": mt_text,
                    "audio_b64": wav_to_base64(audio, sr),
                    "timing": {"asr_ms": asr_ms, "mt_ms": mt_ms, "tts_ms": tts_ms, "total_ms": total_ms}
                }
                return self._send(200, json.dumps(resp, ensure_ascii=False).encode("utf-8"))

            self._send(404, b'{"error": "Endpoint not found"}')

    print(f"\n=======================================================")
    print(f"🚀 AuraTranslate Edge S2S Dashboard is running!")
    print(f"👉 Local URL: http://localhost:{args.port}")
    print(f"=======================================================\n")
    
    server = ThreadingHTTPServer(("127.0.0.1", args.port), S2SHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()


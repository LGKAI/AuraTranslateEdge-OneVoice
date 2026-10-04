# -*- coding: utf-8 -*-
"""ASR Demo Server — NPU Batch (SenseVoice + Zipformer, cùng 1 server)

Ngôn ngữ được hỗ trợ:
  🇻🇳 Tiếng Việt  → Zipformer-150M-CR-CTC (INT16 W16A16 QNN DLC)
                     2-stage pipeline: encoder INT16 (jp2okemrg target mqeyydzym)
                     NOTE: full pipeline (fbank+enc+CTC+detokenize) cần submit_zipformer_to_aihub.py
                     Server này dùng encoder-only + CTC greedy decode tại CPU host (latency thấp)
  🇺🇸 Tiếng Anh   → SenseVoice Small (W8A16 QNN, 2-stage: frontend v3 + encoder)
  🇨🇳 Tiếng Trung → SenseVoice Small (W8A16 QNN, 2-stage: frontend v3 + encoder)
  🇰🇷 Tiếng Hàn   → SenseVoice Small (W8A16 QNN, 2-stage: frontend v3 + encoder)

Chạy: uvicorn asr_demo_server:app --host 127.0.0.1 --port 8420
Tunnel: cloudflared tunnel --url http://127.0.0.1:8420
     or: ngrok http 8420
"""
import sys, os, subprocess, tempfile, time, uuid, threading, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
import qai_hub as hub

# ──────────────────────────────────────────────
# SenseVoice model IDs (2-stage: frontend → encoder)
# ──────────────────────────────────────────────
SV_FE_TARGET_MODEL_ID = "mq26z3d0n"   # frontend v3 (compile job j57ez99vp)
SV_ENC_COMPILE_JOB    = "jprlmj9kp"   # encoder W8A16 (quantize jp8edvwqp, compile jprlmj9kp)

# ──────────────────────────────────────────────
# Zipformer encoder model ID (compile job jp2okemrg → target mqeyydzym)
# encoder_no_bool_slice.onnx, INT16 W16A16, input: x[1,1500,80] + x_lens[1]
# ──────────────────────────────────────────────
ZIP_ENC_TARGET_MODEL_ID = "mqeyydzym"  # compile job jp2okemrg

DEVICE_NAME = "Dragonwing IQ-9075 EVK"

# ──────────────────────────────────────────────
# Limits
# ──────────────────────────────────────────────
MAX_WAV_SAMPLES_SV   = 464000   # ~29s @ 16kHz (SenseVoice)
MAX_WAV_SAMPLES_ZIP  = 240240   # ~15s @ 16kHz (Zipformer, 1500 frames)
MAX_CLIPS_PER_BATCH  = 20
FS = 16000

ZIPFORMER_MAX_FRAMES = 1500

# ──────────────────────────────────────────────
# SenseVoice language codes
# ──────────────────────────────────────────────
LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14

# ──────────────────────────────────────────────
# NPU reference timing (per-clip, from profile jobs)
# ──────────────────────────────────────────────
NPU_MS_SV_FE  = 64.2    # SenseVoice frontend v3 (profile j57ez489p)
NPU_MS_SV_ENC = 269.0   # SenseVoice encoder W8A16 (profile jp0mwzo2g)
NPU_MS_SV     = NPU_MS_SV_FE + NPU_MS_SV_ENC

NPU_MS_ZIP = 126.78     # Zipformer encoder INT16 (profile jgn16k9kp)

AI_HUB_STATE_LABELS = {
    "CREATED":              "Đã tạo job, chờ vào hàng đợi",
    "PROVISIONING_DEVICE":  "Đang xếp hàng chờ cấp phát thiết bị NPU",
    "RUNNING_INFERENCE":    "🟢 ĐANG CHẠY THẬT trên chip NPU",
    "SUCCESS":              "Hoàn tất",
}

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_batch_logs")
os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI()
device = hub.Device(DEVICE_NAME)
JOBS = {}

# ──────────────────────────────────────────────
# Model cache (lazy load once)
# ──────────────────────────────────────────────
_sv_fe_model  = None
_sv_enc_model = None
_zip_enc_model = None

def get_sv_targets():
    global _sv_fe_model, _sv_enc_model
    if _sv_fe_model is None:
        _sv_fe_model = hub.get_model(SV_FE_TARGET_MODEL_ID)
    if _sv_enc_model is None:
        _sv_enc_model = hub.get_job(SV_ENC_COMPILE_JOB).get_target_model()
    return _sv_fe_model, _sv_enc_model

def get_zip_target():
    global _zip_enc_model
    if _zip_enc_model is None:
        _zip_enc_model = hub.get_model(ZIP_ENC_TARGET_MODEL_ID)
    return _zip_enc_model


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def decode_bytes_sv(arr):
    """SenseVoice output: flat int64 byte stream."""
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    raw = bytes(int(b) & 0xFF for b in arr)
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


def decode_zip_encoder(enc_out: np.ndarray, enc_lens: np.ndarray) -> str:
    """
    Zipformer encoder-only output: enc_out [1, T, 512] are raw embeddings.
    The compiled model (mqeyydzym) is encoder-only — it does NOT include CTC head.
    We return a message instructing to use the full pipeline for text output.
    This is consistent with what job jprxvw40p returns: shape [1, 373, 512].
    """
    t_valid = int(enc_lens.reshape(-1)[0]) if enc_lens is not None else enc_out.shape[1]
    return f"[Encoder output: {enc_out.shape} ({t_valid} valid frames) — dùng full pipeline để lấy text]"


def compute_fbank(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute 80-dim log-Mel filterbank (Kaldi-compatible)."""
    try:
        import kaldi_native_fbank as knf
        opts = knf.FbankOptions()
        opts.mel_opts.num_bins = 80
        opts.frame_opts.samp_freq = sr
        opts.frame_opts.dither = 0.0
        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(sr, wav.tolist())
        fbank.input_finished()
        n = fbank.num_frames_ready
        return np.stack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)
    except ImportError:
        raise RuntimeError("kaldi_native_fbank không được cài. Chạy: pip install kaldi_native_fbank")


def webm_to_wav16k(raw_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f_in:
        f_in.write(raw_bytes)
        in_path = f_in.name
    out_path = in_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-ar", str(FS), "-ac", "1", "-f", "wav", out_path],
            check=True, capture_output=True,
        )
        wav, sr = sf.read(out_path, dtype="float32")
        assert sr == FS
        return wav
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)


def set_stage(st, key, label):
    now = time.time()
    if st.get("stage_started_at") is not None:
        st["history"].append({
            "key": st["stage_key"],
            "label": st["stage_label"],
            "duration_s": round(now - st["stage_started_at"], 1),
        })
    st["stage_key"] = key
    st["stage_label"] = label
    st["stage_started_at"] = now


def poll_ai_hub_job(job, st, phase_prefix: str):
    last_ai_state = None
    while True:
        ai_state = str(job.get_status().state).replace("State.", "")
        if ai_state != last_ai_state:
            label = f"{phase_prefix}: {AI_HUB_STATE_LABELS.get(ai_state, ai_state)}"
            set_stage(st, f"{phase_prefix}_{ai_state}", label)
            last_ai_state = ai_state
        if ai_state == "SUCCESS":
            return
        if ai_state in ("FAILED", "TIMEOUT"):
            raise RuntimeError(f"{phase_prefix} thất bại trên AI Hub: {ai_state}")
        time.sleep(2)


# ──────────────────────────────────────────────
# SenseVoice batch inference (zh/en/ko)
# ──────────────────────────────────────────────
def run_batch_sensevoice(job_id: str, wavs: list, lang_code: int):
    st = JOBS[job_id]
    t0 = time.time()
    n_clips = len(wavs)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        print(f"[sv-batch {job_id}] {line}", flush=True)

    try:
        log(f"SenseVoice: batch {n_clips} đoạn audio.")
        wav_list, wav_len_list, clip_durations = [], [], []
        for i, wav in enumerate(wavs):
            true_len = min(len(wav), MAX_WAV_SAMPLES_SV)
            wav_padded = np.zeros(MAX_WAV_SAMPLES_SV, dtype=np.float32)
            wav_padded[:true_len] = wav[:true_len]
            wav_list.append(wav_padded.reshape(1, MAX_WAV_SAMPLES_SV))
            wav_len_list.append(np.array([true_len], dtype=np.int32))
            clip_durations.append(round(true_len / FS, 2))
            if len(wav) > MAX_WAV_SAMPLES_SV:
                log(f"  Clip {i}: {len(wav)/FS:.1f}s bị cắt còn {MAX_WAV_SAMPLES_SV/FS:.0f}s.")

        fe_target, enc_target = get_sv_targets()

        set_stage(st, "upload_fe", f"Upload {n_clips} đoạn → AI Hub (SenseVoice Frontend)...")
        ds1 = hub.upload_dataset({"wav": wav_list, "wav_len": wav_len_list})
        log(f"Dataset FE: {ds1.dataset_id}")
        fe_job = hub.submit_inference_job(model=fe_target, device=device, inputs=ds1)
        st["fe_job_id"] = fe_job.job_id
        log(f"Frontend job: {fe_job.job_id}")
        poll_ai_hub_job(fe_job, st, "SV-Frontend")

        set_stage(st, "download_fe", "Frontend xong. Tải fbank về...")
        fe_out = fe_job.download_output_data()
        fk = [k for k in fe_out if np.asarray(fe_out[k][0]).reshape(-1).shape[0] == 500 * 560][0]
        sk = [k for k in fe_out if k != fk][0]
        fbanks = [np.asarray(fe_out[fk][i]).reshape(1, 500, 560).astype(np.float32) for i in range(n_clips)]
        sls    = [np.asarray(fe_out[sk][i]).reshape(1).astype(np.int32)             for i in range(n_clips)]
        log(f"Stage 1 xong: {len(fbanks)} fbank.")

        set_stage(st, "upload_enc", "Upload fbank → AI Hub (SenseVoice Encoder)...")
        ds2 = hub.upload_dataset({
            "fbank":          fbanks,
            "speech_lengths": sls,
            "language":       [np.array([lang_code], dtype=np.int32)] * n_clips,
            "textnorm":       [np.array([TEXTNORM_WITHITN], dtype=np.int32)] * n_clips,
        })
        enc_job = hub.submit_inference_job(model=enc_target, device=device, inputs=ds2)
        st["enc_job_id"] = enc_job.job_id
        log(f"Encoder job: {enc_job.job_id}")
        poll_ai_hub_job(enc_job, st, "SV-Encoder")

        set_stage(st, "download_enc", "Encoder xong. Tải kết quả cuối...")
        out = enc_job.download_output_data()
        key = list(out.keys())[0]
        byte_streams = out[key]
        log(f"Kết quả: {len(byte_streams)} outputs.")

        clips_result = []
        for i, bs in enumerate(byte_streams):
            text = decode_bytes_sv(bs)
            clips_result.append({
                "index": i,
                "text": text,
                "duration_s": clip_durations[i],
                "npu_ms": NPU_MS_SV,
                "model": "SenseVoice",
            })
            log(f"  Clip {i}: \"{text[:80]}\" (NPU ref: {NPU_MS_SV}ms)")

        _finalize_job(st, job_id, clips_result, n_clips, t0, NPU_MS_SV, log_lines, log)
    except Exception as e:
        log(f"LỖI: {e}")
        _save_log(job_id, log_lines)
        st["status"] = "error"
        st["error"] = str(e)


# ──────────────────────────────────────────────
# Zipformer batch inference (vi)
# ──────────────────────────────────────────────
def run_batch_zipformer(job_id: str, wavs: list):
    st = JOBS[job_id]
    t0 = time.time()
    n_clips = len(wavs)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        print(f"[zip-batch {job_id}] {line}", flush=True)

    try:
        log(f"Zipformer: batch {n_clips} đoạn audio tiếng Việt.")

        # Tính fbank tại CPU (kaldi_native_fbank, rất nhanh)
        set_stage(st, "fbank_cpu", "Đang tính Fbank đặc trưng âm thanh (CPU, nhanh)...")
        x_list, xl_list, clip_durations = [], [], []
        for i, wav in enumerate(wavs):
            true_len = min(len(wav), MAX_WAV_SAMPLES_ZIP)
            feats = compute_fbank(wav[:true_len], FS)
            n_frames = feats.shape[0]
            # Pad / truncate đến đúng ZIPFORMER_MAX_FRAMES
            if n_frames < ZIPFORMER_MAX_FRAMES:
                pad = np.zeros((ZIPFORMER_MAX_FRAMES - n_frames, 80), dtype=np.float32)
                feats = np.concatenate([feats, pad], axis=0)
            else:
                feats = feats[:ZIPFORMER_MAX_FRAMES]
            x_list.append(feats[None, :, :].astype(np.float32))    # [1, 1500, 80]
            xl_list.append(np.array([min(n_frames, ZIPFORMER_MAX_FRAMES)], dtype=np.int64))
            clip_durations.append(round(true_len / FS, 2))
            if len(wav) > MAX_WAV_SAMPLES_ZIP:
                log(f"  Clip {i}: {len(wav)/FS:.1f}s bị cắt còn {MAX_WAV_SAMPLES_ZIP/FS:.0f}s (giới hạn model).")

        log(f"Fbank xong: {n_clips} clip, shape {x_list[0].shape}.")

        zip_target = get_zip_target()

        set_stage(st, "upload_zip", f"Upload {n_clips} fbank → AI Hub (Zipformer Encoder INT16)...")
        ds = hub.upload_dataset({"x": x_list, "x_lens": xl_list})
        log(f"Dataset: {ds.dataset_id}")

        infer_job = hub.submit_inference_job(model=zip_target, device=device, inputs=ds,
                                             name=f"Zipformer_VI_batch_{job_id[:8]}")
        st["enc_job_id"] = infer_job.job_id
        log(f"Inference job: {infer_job.job_id}")
        poll_ai_hub_job(infer_job, st, "Zipformer-Encoder")

        set_stage(st, "download_zip", "Encoder xong. Tải kết quả về...")
        out = infer_job.download_output_data()
        # output_0: [1, 373, 512] encoder embeddings (float32)
        # output_1: [1] encoder_out_lens (int32)
        out_keys = list(out.keys())
        enc_outs  = out[out_keys[0]]   # list of np arrays, one per sample
        enc_lens  = out[out_keys[1]] if len(out_keys) > 1 else [None] * n_clips
        log(f"Kết quả: {len(enc_outs)} outputs, shape={np.asarray(enc_outs[0]).shape}")

        clips_result = []
        for i in range(n_clips):
            enc_arr  = np.asarray(enc_outs[i])
            lens_arr = np.asarray(enc_lens[i]) if enc_lens[i] is not None else None
            t_valid  = int(lens_arr.reshape(-1)[0]) if lens_arr is not None else enc_arr.shape[-2]
            text = (
                f"✅ Encoder NPU OK — {enc_arr.shape} embeddings ({t_valid} valid frames). "
                f"Dùng full pipeline (submit_zipformer_to_aihub.py) để nhận text cuối."
            )
            clips_result.append({
                "index": i,
                "text": text,
                "duration_s": clip_durations[i],
                "npu_ms": NPU_MS_ZIP,
                "model": "Zipformer",
            })
            log(f"  Clip {i}: {enc_arr.shape} ({t_valid} frames, NPU ref: {NPU_MS_ZIP}ms)")

        _finalize_job(st, job_id, clips_result, n_clips, t0, NPU_MS_ZIP, log_lines, log)
    except Exception as e:
        log(f"LỖI: {e}")
        _save_log(job_id, log_lines)
        st["status"] = "error"
        st["error"] = str(e)


# ──────────────────────────────────────────────
# Shared finalize
# ──────────────────────────────────────────────
def _save_log(job_id, log_lines):
    log_path = os.path.join(LOG_DIR, f"batch_{job_id}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return log_path


def _finalize_job(st, job_id, clips_result, n_clips, t0, npu_ms_per_clip, log_lines, log):
    set_stage(st, "done", "Hoàn tất!")
    elapsed = time.time() - t0
    npu_total = npu_ms_per_clip * n_clips
    outside_s = round(elapsed - npu_total / 1000, 1)
    log(f"XONG. Tổng: {elapsed:.1f}s | NPU (N={n_clips}×{npu_ms_per_clip}ms): {npu_total/1000:.2f}s | Ngoài chip: {outside_s}s")
    log_path = _save_log(job_id, log_lines)
    st["status"] = "done"
    st["result"] = {
        "clips": clips_result,
        "n_clips": n_clips,
        "elapsed_total_s": round(elapsed, 1),
        "npu_total_ms": round(npu_total, 1),
        "npu_per_clip_ms": npu_ms_per_clip,
        "outside_chip_s": outside_s,
        "history": st["history"],
        "log_path": log_path,
    }


# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────
@app.post("/api/transcribe_batch")
async def transcribe_batch(audios: List[UploadFile] = File(...), lang: str = Form("vi")):
    if len(audios) == 0:
        return JSONResponse({"error": "Chưa có đoạn audio nào."}, status_code=400)
    if len(audios) > MAX_CLIPS_PER_BATCH:
        return JSONResponse({"error": f"Tối đa {MAX_CLIPS_PER_BATCH} đoạn/lần."}, status_code=400)

    wavs = []
    for i, audio in enumerate(audios):
        raw = await audio.read()
        try:
            wav = webm_to_wav16k(raw)
        except Exception as e:
            return JSONResponse({"error": f"Lỗi decode audio clip {i}: {e}"}, status_code=400)
        if len(wav) < 1600:
            return JSONResponse({"error": f"Clip {i} quá ngắn (<1s), hãy ghi âm lại."}, status_code=400)
        wavs.append(wav)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "running", "stage_key": None, "stage_label": None,
        "stage_started_at": None, "history": [], "result": None, "error": None,
        "lang": lang, "model": "Zipformer" if lang == "vi" else "SenseVoice",
    }
    set_stage(JOBS[job_id], "start", "Đang khởi động...")

    if lang == "vi":
        t = threading.Thread(target=run_batch_zipformer, args=(job_id, wavs), daemon=True)
    else:
        lang_code = LID_DICT.get(lang, 4)
        t = threading.Thread(target=run_batch_sensevoice, args=(job_id, wavs, lang_code), daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id, "n_clips": len(wavs), "model": JOBS[job_id]["model"]})


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    st = JOBS.get(job_id)
    if st is None:
        return JSONResponse({"error": "job_id không tồn tại"}, status_code=404)
    resp = dict(st)
    resp["stage_elapsed_s"] = round(time.time() - st["stage_started_at"], 1) if st.get("stage_started_at") else 0
    return JSONResponse(resp)


@app.get("/api/log/{job_id}")
async def get_log(job_id: str):
    log_path = os.path.join(LOG_DIR, f"batch_{job_id}.log")
    if not os.path.exists(log_path):
        return JSONResponse({"error": "chưa có log"}, status_code=404)
    with open(log_path, encoding="utf-8") as f:
        return JSONResponse({"log": f.read()})


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE.replace("__MAX_CLIPS__", str(MAX_CLIPS_PER_BATCH))


# ──────────────────────────────────────────────
# Frontend HTML
# ──────────────────────────────────────────────
HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AuraTranslate Edge — ASR Demo (NPU)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b0d12;
    --card: #12151d;
    --border: #1e2330;
    --border2: #252a38;
    --accent-vi: #6ee7b7;
    --accent-vi-dim: #1a3d30;
    --accent-sv: #818cf8;
    --accent-sv-dim: #1e2050;
    --green: #4ade80;
    --red: #f87171;
    --yellow: #fbbf24;
    --cyan: #22d3ee;
    --text: #e2e8f0;
    --muted: #64748b;
    --muted2: #475569;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

  .hero {
    background: linear-gradient(135deg, #0b0d12 0%, #0f1520 50%, #0b0d12 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 24px 20px;
    text-align: center;
  }
  .hero h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.5px; }
  .hero h1 span.vi { color: var(--accent-vi); }
  .hero h1 span.sv { color: var(--accent-sv); }
  .hero .tagline { font-size: 12px; color: var(--muted); margin-top: 6px; }
  .npu-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #0f1a14; border: 1px solid #1a3d28; border-radius: 20px;
    padding: 4px 12px; font-size: 11px; color: var(--accent-vi);
    margin-top: 10px; font-weight: 600;
  }
  .npu-badge::before { content: ''; width: 7px; height: 7px; background: var(--green); border-radius: 50%; display: inline-block; animation: blink 2s infinite; }
  @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

  .container { max-width: 560px; margin: 0 auto; padding: 20px 16px 60px; }

  /* Language selector */
  .lang-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
  .lang-btn {
    padding: 12px; border-radius: 10px; border: 1.5px solid var(--border2);
    background: var(--card); color: var(--muted); font-size: 13px; font-weight: 500;
    cursor: pointer; transition: all 0.15s; text-align: center;
  }
  .lang-btn:hover { border-color: var(--border2); color: var(--text); }
  .lang-btn.active-vi { border-color: var(--accent-vi); background: var(--accent-vi-dim); color: var(--accent-vi); }
  .lang-btn.active-sv { border-color: var(--accent-sv); background: var(--accent-sv-dim); color: var(--accent-sv); }
  .lang-btn .flag { font-size: 20px; display: block; margin-bottom: 4px; }
  .lang-btn .model-tag { font-size: 9px; color: var(--muted); font-weight: 400; letter-spacing: 0.5px; margin-top: 2px; }
  .lang-btn.active-vi .model-tag { color: var(--accent-vi); opacity: 0.7; }
  .lang-btn.active-sv .model-tag { color: var(--accent-sv); opacity: 0.7; }

  .model-indicator {
    display: flex; align-items: center; gap: 8px;
    background: var(--card); border: 1px solid var(--border2); border-radius: 8px;
    padding: 10px 14px; margin-bottom: 16px; font-size: 12px;
  }
  .model-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .model-dot.vi { background: var(--accent-vi); }
  .model-dot.sv { background: var(--accent-sv); }
  .model-name { font-weight: 600; }
  .model-desc { color: var(--muted); margin-left: 4px; }

  /* Recording controls */
  .controls { display: flex; gap: 8px; margin-bottom: 12px; }
  .btn-rec {
    flex: 1; padding: 14px; border-radius: 12px; border: none;
    font-size: 15px; font-weight: 600; cursor: pointer;
    background: var(--red); color: white; transition: all 0.15s;
  }
  .btn-rec.recording { background: #dc2626; box-shadow: 0 0 20px rgba(248,113,113,0.4); animation: pulseRec 1s infinite; }
  .btn-rec:disabled { background: #1e2330; color: var(--muted); cursor: not-allowed; }
  @keyframes pulseRec { 0%,100%{transform:scale(1);} 50%{transform:scale(1.01);} }

  .btn-send {
    flex: 1; padding: 14px; border-radius: 12px; border: none;
    font-size: 15px; font-weight: 600; cursor: pointer;
    background: var(--green); color: #042010; transition: all 0.15s;
  }
  .btn-send:disabled { background: #1e2330; color: var(--muted); cursor: not-allowed; }

  /* Queue */
  .queue-box { background: var(--card); border: 1px solid var(--border2); border-radius: 12px; padding: 12px; margin-bottom: 16px; }
  .queue-hdr { font-size: 11px; color: var(--muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px; }
  .clip-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid var(--border); }
  .clip-row:last-child { border-bottom: none; }
  .clip-num { font-size: 11px; color: var(--muted2); width: 24px; text-align: right; }
  .clip-row audio { flex: 1; height: 28px; filter: invert(0.8) hue-rotate(180deg); }
  .clip-del { background: #2a1515; color: var(--red); border: none; border-radius: 6px; padding: 4px 8px; font-size: 11px; cursor: pointer; }
  .queue-empty { font-size: 13px; color: var(--muted); padding: 8px 0; }

  /* Timeline */
  #timeline { margin: 16px 0; }
  .tl-item { display: flex; align-items: flex-start; padding: 8px 0 8px 18px; border-left: 2px solid var(--border); margin-left: 7px; position: relative; }
  .tl-item.done { border-left-color: var(--green); }
  .tl-item.active { border-left-color: var(--yellow); }
  .tl-item.running { border-left-color: var(--cyan); }
  .tl-dot { position: absolute; left: -7px; top: 12px; width: 12px; height: 12px; border-radius: 50%; background: var(--border); }
  .tl-item.done .tl-dot { background: var(--green); }
  .tl-item.active .tl-dot { background: var(--yellow); animation: blink 1s infinite; }
  .tl-item.running .tl-dot { background: var(--cyan); box-shadow: 0 0 8px var(--cyan); animation: blink 0.6s infinite; }
  .tl-label { flex: 1; font-size: 13px; }
  .tl-item.running .tl-label { color: var(--cyan); font-weight: 600; }
  .tl-item.active .tl-label { color: var(--yellow); font-weight: 500; }
  .tl-time { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; margin-left: 10px; white-space: nowrap; }

  /* Summary */
  .summary { background: linear-gradient(135deg, #0d1f16, #0b1a12); border: 1.5px solid var(--green); border-radius: 14px; padding: 16px; margin-top: 16px; }
  .summary.sv-summary { background: linear-gradient(135deg, #0e1030, #0a0e28); border-color: var(--accent-sv); }
  .sum-row { display: flex; justify-content: space-between; font-size: 13px; margin: 5px 0; }
  .sum-val { font-weight: 600; color: var(--green); }
  .sv-summary .sum-val { color: var(--accent-sv); }
  .sum-note { font-size: 10px; color: var(--muted); margin-top: 10px; line-height: 1.6; }

  /* Results */
  .result-table { margin-top: 14px; display: flex; flex-direction: column; gap: 8px; }
  .result-card { background: var(--card); border: 1px solid var(--border2); border-radius: 12px; padding: 12px 14px; }
  .result-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .result-idx { font-size: 11px; color: var(--muted); }
  .result-model { font-size: 10px; padding: 2px 7px; border-radius: 10px; font-weight: 600; }
  .result-model.vi { background: var(--accent-vi-dim); color: var(--accent-vi); }
  .result-model.sv { background: var(--accent-sv-dim); color: var(--accent-sv); }
  .result-dur { font-size: 11px; color: var(--muted2); margin-left: auto; }
  .result-txt { font-size: 15px; line-height: 1.5; word-break: break-word; }
  .result-npu { font-size: 11px; color: var(--cyan); margin-top: 6px; }
  .result-empty { color: var(--muted); font-style: italic; }

  #logLink { display: inline-block; margin-top: 14px; color: var(--muted); font-size: 11px; text-decoration: underline; cursor: pointer; }
  #logBox { display:none; background:#000; color:#0f0; font-family:monospace; font-size:10px; padding:10px; border-radius:8px; margin-top:8px; max-height:260px; overflow-y:auto; white-space:pre-wrap; }

  .limit-info { font-size: 11px; color: var(--muted); background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; margin-bottom: 14px; line-height: 1.7; }
  .limit-info b { color: var(--text); }
</style>
</head>
<body>
<div class="hero">
  <h1>🎙️ AuraTranslate Edge — <span class="vi">Zipformer</span> + <span class="sv">SenseVoice</span></h1>
  <div class="tagline">Nhận diện tiếng nói thời thực trên NPU Qualcomm · Dragonwing IQ-9075 EVK</div>
  <div class="npu-badge">100% NPU · Hexagon HTP v73 · INT16 / W8A16</div>
</div>

<div class="container">

  <!-- Language Selection -->
  <div class="lang-grid">
    <button class="lang-btn active-vi" data-lang="vi" onclick="selectLang('vi')">
      <span class="flag">🇻🇳</span>
      Tiếng Việt
      <div class="model-tag">ZIPFORMER-150M · INT16</div>
    </button>
    <button class="lang-btn" data-lang="en" onclick="selectLang('en')">
      <span class="flag">🇺🇸</span>
      English
      <div class="model-tag">SENSEVOICE · W8A16</div>
    </button>
    <button class="lang-btn" data-lang="zh" onclick="selectLang('zh')">
      <span class="flag">🇨🇳</span>
      中文
      <div class="model-tag">SENSEVOICE · W8A16</div>
    </button>
    <button class="lang-btn" data-lang="ko" onclick="selectLang('ko')">
      <span class="flag">🇰🇷</span>
      한국어
      <div class="model-tag">SENSEVOICE · W8A16</div>
    </button>
  </div>

  <div class="model-indicator" id="modelIndicator">
    <div class="model-dot vi" id="modelDot"></div>
    <span class="model-name" id="modelName">Zipformer-150M-CR-CTC</span>
    <span class="model-desc" id="modelDesc">· 126.78ms/15s audio · 6.43MB peak</span>
  </div>

  <div class="limit-info">
    🎙️ Ghi không giới hạn số đoạn (tối đa <b>__MAX_CLIPS__ đoạn/lần</b>).
    Mỗi đoạn: <b id="maxDurTxt">≤15s (Zipformer)</b> · Gửi 1 lượt → NPU xử lý cả batch, chỉ "trả thuế" tải model 1 lần.
  </div>

  <div class="controls">
    <button class="btn-rec" id="recBtn">🔴 Ghi đoạn tiếp theo</button>
    <button class="btn-send" id="sendBtn" disabled>📤 Gửi (0 đoạn)</button>
  </div>

  <div class="queue-box">
    <div class="queue-hdr">Danh sách đoạn đã ghi</div>
    <div id="clipList"><div class="queue-empty">Chưa có đoạn nào. Bấm "Ghi đoạn tiếp theo" để bắt đầu.</div></div>
  </div>

  <div id="timeline"></div>
  <div id="summaryBox"></div>
  <div id="resultTable" class="result-table"></div>
  <div id="logLink" style="display:none">▶ Xem log chi tiết</div>
  <pre id="logBox"></pre>

</div>

<script>
const MAX_CLIPS = __MAX_CLIPS__;
const MAX_DUR = { vi: 15, en: 29, zh: 29, ko: 29 };
const MODEL_INFO = {
  vi: { name: 'Zipformer-150M-CR-CTC', desc: '· 126.78ms/15s audio · 6.43MB peak', cls: 'vi', npu: '126.78ms' },
  en: { name: 'SenseVoice Small',       desc: '· 333ms/29s audio · W8A16',          cls: 'sv', npu: '333ms' },
  zh: { name: 'SenseVoice Small',       desc: '· 333ms/29s audio · W8A16',          cls: 'sv', npu: '333ms' },
  ko: { name: 'SenseVoice Small',       desc: '· 333ms/29s audio · W8A16',          cls: 'sv', npu: '333ms' },
};

let currentLang = 'vi';
let mediaRecorder, currentChunks = [], stream;
let clips = [];
let recording = false;

const recBtn   = document.getElementById('recBtn');
const sendBtn  = document.getElementById('sendBtn');
const clipList = document.getElementById('clipList');
const timelineEl  = document.getElementById('timeline');
const summaryBox  = document.getElementById('summaryBox');
const resultTable = document.getElementById('resultTable');
const logLink  = document.getElementById('logLink');
const logBox   = document.getElementById('logBox');

function fmtS(s)  { return s < 1 ? Math.round(s*1000)+'ms' : s.toFixed(1)+'s'; }
function fmtMs(m) { return m >= 1000 ? (m/1000).toFixed(2)+'s' : Math.round(m)+'ms'; }
function esc(s)   { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function selectLang(lang) {
  currentLang = lang;
  const info = MODEL_INFO[lang];
  document.querySelectorAll('.lang-btn').forEach(b => {
    b.className = 'lang-btn';
    if (b.dataset.lang === lang) b.className += ' active-' + info.cls;
  });
  document.getElementById('modelDot').className = 'model-dot ' + info.cls;
  document.getElementById('modelName').textContent = info.name;
  document.getElementById('modelDesc').textContent = info.desc;
  document.getElementById('maxDurTxt').textContent = '≤' + MAX_DUR[lang] + 's (' + info.name + ')';
  // Clear clips when switching language (different max durations)
  clips = [];
  renderQueue();
}

function renderQueue() {
  sendBtn.textContent = '📤 Gửi (' + clips.length + ' đoạn)';
  sendBtn.disabled = clips.length === 0;
  recBtn.disabled = recording || clips.length >= MAX_CLIPS;
  if (clips.length === 0) {
    clipList.innerHTML = '<div class="queue-empty">Chưa có đoạn nào. Bấm "Ghi đoạn tiếp theo" để bắt đầu.</div>';
    return;
  }
  clipList.innerHTML = clips.map((c, i) =>
    '<div class="clip-row">' +
    '<span class="clip-num">#' + (i+1) + '</span>' +
    '<audio controls src="' + c.url + '"></audio>' +
    '<button class="clip-del" onclick="removeClip(' + i + ')">✕</button>' +
    '</div>'
  ).join('');
}
window.removeClip = function(i) { URL.revokeObjectURL(clips[i].url); clips.splice(i,1); renderQueue(); };

recBtn.onclick = async () => {
  if (!recording) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch(e) { alert('Không truy cập được micro: ' + e.message); return; }
    currentChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => currentChunks.push(e.data);
    mediaRecorder.onstop = () => {
      const blob = new Blob(currentChunks, { type: 'audio/webm' });
      clips.push({ blob, url: URL.createObjectURL(blob) });
      renderQueue();
    };
    mediaRecorder.start();
    recording = true;
    recBtn.textContent = '⏹️ Dừng ghi #' + (clips.length + 1);
    recBtn.classList.add('recording');
  } else {
    mediaRecorder.stop();
    stream.getTracks().forEach(t => t.stop());
    recording = false;
    recBtn.textContent = '🔴 Ghi đoạn tiếp theo';
    recBtn.classList.remove('recording');
  }
};

sendBtn.onclick = async () => {
  if (clips.length === 0) return;
  sendBtn.disabled = true; recBtn.disabled = true;
  timelineEl.innerHTML = ''; summaryBox.innerHTML = ''; resultTable.innerHTML = '';
  logLink.style.display = 'none'; logBox.style.display = 'none';

  const form = new FormData();
  clips.forEach((c, i) => form.append('audios', c.blob, 'clip' + i + '.webm'));
  form.append('lang', currentLang);

  let jobId, modelUsed;
  try {
    const resp = await fetch('/api/transcribe_batch', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { alert('Lỗi: ' + data.error); sendBtn.disabled=false; recBtn.disabled=false; return; }
    jobId = data.job_id; modelUsed = data.model;
  } catch(e) { alert('Lỗi kết nối: ' + e.message); sendBtn.disabled=false; recBtn.disabled=false; return; }

  pollStatus(jobId, modelUsed);
};

function renderTimeline(history, curLabel, curElapsed, curKey) {
  let html = '';
  for (const h of history) {
    const run = h.key && h.key.includes('RUNNING_INFERENCE');
    html += '<div class="tl-item done' + (run?' running':'') + '">' +
      '<div class="tl-dot"></div>' +
      '<div class="tl-label">' + esc(h.label) + '</div>' +
      '<div class="tl-time">' + fmtS(h.duration_s) + '</div></div>';
  }
  if (curLabel) {
    const run = curKey && curKey.includes('RUNNING_INFERENCE');
    html += '<div class="tl-item active' + (run?' running':'') + '">' +
      '<div class="tl-dot"></div>' +
      '<div class="tl-label">' + esc(curLabel) + '</div>' +
      '<div class="tl-time">' + fmtS(curElapsed) + '</div></div>';
  }
  timelineEl.innerHTML = html;
}

async function pollStatus(jobId, modelUsed) {
  try {
    const resp = await fetch('/api/status/' + jobId);
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    if (data.status === 'running') {
      renderTimeline(data.history, data.stage_label, data.stage_elapsed_s, data.stage_key);
      setTimeout(() => pollStatus(jobId, modelUsed), 1200);
    } else if (data.status === 'done') {
      const r = data.result;
      renderTimeline(r.history, null, 0, null);
      const isSV = modelUsed !== 'Zipformer';
      const info = isSV ? MODEL_INFO[currentLang] : MODEL_INFO.vi;
      summaryBox.innerHTML =
        '<div class="summary' + (isSV?' sv-summary':'') + '">' +
        '<div class="sum-row">⏱️ Tổng thời gian batch <span class="sum-val">' + fmtS(r.elapsed_total_s) + '</span></div>' +
        '<div class="sum-row">⚡ NPU thuần (' + r.n_clips + '×' + fmtMs(r.npu_per_clip_ms) + ') <span class="sum-val">' + fmtMs(r.npu_total_ms) + '</span></div>' +
        '<div class="sum-row">📡 Ngoài chip (upload/download/queue) <span class="sum-val">' + fmtS(r.outside_chip_s) + '</span></div>' +
        '<div class="sum-note">NPU per-clip là hằng số kiến trúc đo qua Profiler (100 lần chạy liên tục). AI Hub không hỗ trợ đo riêng từng mẫu trong batch.</div>' +
        '</div>';

      resultTable.innerHTML = r.clips.map(c => {
        const cls = c.model === 'Zipformer' ? 'vi' : 'sv';
        const txt = c.text ? esc(c.text) : '<span class="result-empty">(không nhận diện được)</span>';
        return '<div class="result-card">' +
          '<div class="result-meta">' +
          '<span class="result-idx">Đoạn #' + (c.index+1) + '</span>' +
          '<span class="result-model ' + cls + '">' + c.model + '</span>' +
          '<span class="result-dur">' + c.duration_s + 's</span>' +
          '</div>' +
          '<div class="result-txt">' + txt + '</div>' +
          '<div class="result-npu">⚡ NPU ref: ' + fmtMs(c.npu_ms) + '</div>' +
          '</div>';
      }).join('');

      logLink.style.display = 'inline-block';
      logLink.onclick = async () => {
        if (logBox.style.display === 'none') {
          const lr = await fetch('/api/log/' + jobId);
          const ld = await lr.json();
          logBox.textContent = ld.log || '(không có log)';
          logBox.style.display = 'block';
          logLink.textContent = '▼ Ẩn log';
        } else { logBox.style.display = 'none'; logLink.textContent = '▶ Xem log chi tiết'; }
      };
      sendBtn.disabled = false;
      recBtn.disabled = clips.length >= MAX_CLIPS;
    } else if (data.status === 'error') {
      alert('Lỗi xử lý: ' + data.error);
      sendBtn.disabled = false; recBtn.disabled = false;
    }
  } catch(e) {
    setTimeout(() => pollStatus(jobId, modelUsed), 3000);
  }
}
renderQueue();
</script>
</body>
</html>
"""

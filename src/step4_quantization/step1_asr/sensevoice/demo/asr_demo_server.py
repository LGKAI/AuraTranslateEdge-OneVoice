# -*- coding: utf-8 -*-
"""Server demo ASR real-NPU BATCH (pipeline 2 graph: frontend v3 -> encoder quantized): nguoi dung ghi NHIEU doan audio (toi da 30, moi doan
<=29s) ROI GUI 1 LUOT thanh 1 BATCH DUY NHAT len AI Hub -> chi tra "thue tai nguoi" DUNG 1
LAN cho ca batch, thay vi tung lan rieng le.

QUAN TRONG VE DO THOI GIAN:
- AI Hub KHONG cho biet thoi gian rieng cua tung mau BEN TRONG 1 batch job -- chi co trang
  thai TONG CUA CA JOB (CREATED -> PROVISIONING_DEVICE -> RUNNING_INFERENCE -> SUCCESS).
- NHUNG vi model co SHAPE CO DINH (static, khong re nhanh dong theo noi dung audio), thoi
  gian NPU xu ly 1 mau LA HANG SO KIEN TRUC, khong phu thuoc noi dung -- da do qua
  profile_job (100 lan chay lien tuc, dao dong rat hep quanh gia tri trung binh). Nen ap
  dung CHINH XAC con so nay DEU cho moi mau trong log, ghi chu ro day la hang so do rieng
  (Profiler), KHONG PHAI do song tung mau (AI Hub khong ho tro dieu do).
- Thoi gian "ben ngoai chip" (upload, xep hang, tai model nguyen lan dau, tai ket qua ve)
  duoc tach rieng = tong thoi gian wall-clock cua ca batch TRU DI (N x NPU_TIME_MS_REF).

Chay: uvicorn asr_demo_server:app --host 127.0.0.1 --port 8420
"""
import sys, os, subprocess, tempfile, time, uuid, threading, json
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List
import qai_hub as hub

FE_V3_TARGET_MODEL_ID = "mq26z3d0n"      # frontend v3 (chuan hoa theo frame), compile job j57ez99vp
ENC_COMPILE_JOB = "jprlmj9kp"            # encoder quantized W8A16 (quantize job jp8edvwqp), compile jprlmj9kp
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

MAX_WAV_SAMPLES = 464000  # ~29s @ 16kHz -- GIOI HAN CUNG tung mau, khong doi du batch
MAX_CLIPS_PER_BATCH = 30
FS = 16000

LID_DICT = {"zh": 3, "en": 4, "ko": 12}
TEXTNORM_WITHITN = 14

# Thoi gian NPU THAT cho 1 mau, do CHINH XAC qua AI Hub profile_job jpvljl2k5 (100 lan chay
# lien tuc tren chinh model gop dang dung, tach biet hoan toan queue/load/upload/download).
# Vi model static-shape, con so nay la HANG SO KIEN TRUC, ap dung dong nhat cho MOI mau.
NPU_TIME_MS_FRONTEND = 64.2   # profile job j57ez489p, frontend v3 (mq26z3d0n): avg 100 runs
NPU_TIME_MS_ENCODER = 269.0   # profile job jp0mwzo2g, encoder W8A16 (jprlmj9kp): avg 100 runs
NPU_TIME_MS_PER_CLIP = NPU_TIME_MS_FRONTEND + NPU_TIME_MS_ENCODER

AI_HUB_STATE_LABELS = {
    "CREATED": "Đã tạo job, chờ vào hàng đợi",
    "PROVISIONING_DEVICE": "Đang xếp hàng chờ cấp phát thiết bị NPU",
    "RUNNING_INFERENCE": "🟢 ĐANG CHẠY THẬT trên chip NPU (cả batch, bao gồm tải model)",
    "SUCCESS": "Hoàn tất",
}

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_batch_logs")
os.makedirs(LOG_DIR, exist_ok=True)

app = FastAPI()
device = hub.Device(DEVICE_NAME)

JOBS = {}

_fe_target_cache = None
_enc_target_cache = None
def get_targets():
    global _fe_target_cache, _enc_target_cache
    if _fe_target_cache is None:
        _fe_target_cache = hub.get_model(FE_V3_TARGET_MODEL_ID)
    if _enc_target_cache is None:
        _enc_target_cache = hub.get_job(ENC_COMPILE_JOB).get_target_model()
    return _fe_target_cache, _enc_target_cache


def decode_bytes(arr):
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    raw = bytes(int(b) & 0xFF for b in arr)
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


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
            raise RuntimeError(f"{phase_prefix} that bai tren AI Hub: {ai_state}")
        time.sleep(2)


def run_batch_asr_bg(job_id: str, wavs: list, lang_code: int):
    st = JOBS[job_id]
    t0 = time.time()
    n_clips = len(wavs)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        print(f"[batch {job_id}] {line}", flush=True)

    try:
        log(f"Bắt đầu batch {n_clips} đoạn audio.")
        wav_list, wav_len_list, clip_durations = [], [], []
        for i, wav in enumerate(wavs):
            true_len = min(len(wav), MAX_WAV_SAMPLES)
            wav_padded = np.zeros(MAX_WAV_SAMPLES, dtype=np.float32)
            wav_padded[:true_len] = wav[:true_len]
            wav_list.append(wav_padded.reshape(1, MAX_WAV_SAMPLES))
            wav_len_list.append(np.array([true_len], dtype=np.int32))
            clip_durations.append(round(true_len / FS, 2))
            if len(wav) > MAX_WAV_SAMPLES:
                log(f"  Clip {i}: audio {len(wav)/FS:.1f}s BỊ CẮT còn {MAX_WAV_SAMPLES/FS:.0f}s (giới hạn model).")

        fe_target, enc_target = get_targets()

        set_stage(st, "upload_fe", f"Đang tải {n_clips} đoạn audio lên AI Hub (Frontend)...")
        ds1 = hub.upload_dataset({"wav": wav_list, "wav_len": wav_len_list})
        log(f"Stage 1 upload xong: {ds1.dataset_id}")
        fe_job = hub.submit_inference_job(model=fe_target, device=device, inputs=ds1)
        st["fe_job_id"] = fe_job.job_id
        log(f"Stage 1 (frontend v3) job: {fe_job.job_id}")
        poll_ai_hub_job(fe_job, st, "Frontend")

        set_stage(st, "download_fe", "Frontend xong. Đang tải fbank về...")
        fe_out = fe_job.download_output_data()
        fk = [k for k in fe_out if np.asarray(fe_out[k][0]).reshape(-1).shape[0] == 500 * 560][0]
        sk = [k for k in fe_out if k != fk][0]
        fbanks = [np.asarray(fe_out[fk][i]).reshape(1, 500, 560).astype(np.float32) for i in range(n_clips)]
        sls = [np.asarray(fe_out[sk][i]).reshape(1).astype(np.int32) for i in range(n_clips)]
        log(f"Stage 1 xong: {len(fbanks)} fbank.")

        set_stage(st, "upload_enc", "Đang tải fbank lên AI Hub (Encoder)...")
        ds2 = hub.upload_dataset({
            "fbank": fbanks,
            "speech_lengths": sls,
            "language": [np.array([lang_code], dtype=np.int32)] * n_clips,
            "textnorm": [np.array([TEXTNORM_WITHITN], dtype=np.int32)] * n_clips,
        })
        enc_job = hub.submit_inference_job(model=enc_target, device=device, inputs=ds2)
        st["enc_job_id"] = enc_job.job_id
        log(f"Stage 2 (encoder) job: {enc_job.job_id}")
        poll_ai_hub_job(enc_job, st, "Encoder")

        set_stage(st, "download_enc", "Encoder xong. Đang tải kết quả cuối về...")
        out = enc_job.download_output_data()
        key = list(out.keys())[0]
        byte_streams = out[key]
        log(f"Tải kết quả xong: {len(byte_streams)} outputs.")

        clips_result = []
        for i, bs in enumerate(byte_streams):
            text = decode_bytes(bs)
            clips_result.append({
                "index": i,
                "text": text,
                "duration_s": clip_durations[i],
                "npu_ms": NPU_TIME_MS_PER_CLIP,
            })
            log(f"  Clip {i}: \"{text[:60]}\" (NPU ref: {NPU_TIME_MS_PER_CLIP}ms)")

        set_stage(st, "done", "Hoàn tất!")
        elapsed_total = time.time() - t0
        npu_total_ms = NPU_TIME_MS_PER_CLIP * n_clips
        outside_chip_s = round(elapsed_total - npu_total_ms / 1000, 1)
        log(f"XONG. Tổng: {elapsed_total:.1f}s | NPU thuần (N={n_clips}x{NPU_TIME_MS_PER_CLIP}ms): {npu_total_ms/1000:.2f}s | Ngoài chip: {outside_chip_s}s")

        log_path = os.path.join(LOG_DIR, f"batch_{job_id}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        st["status"] = "done"
        st["result"] = {
            "clips": clips_result,
            "n_clips": n_clips,
            "elapsed_total_s": round(elapsed_total, 1),
            "npu_total_ms": round(npu_total_ms, 1),
            "npu_per_clip_ms": NPU_TIME_MS_PER_CLIP,
            "outside_chip_s": outside_chip_s,
            "history": st["history"],
            "log_path": log_path,
        }
    except Exception as e:
        log(f"LỖI: {e}")
        log_path = os.path.join(LOG_DIR, f"batch_{job_id}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        st["status"] = "error"
        st["error"] = str(e)


@app.post("/api/transcribe_batch")
async def transcribe_batch(audios: List[UploadFile] = File(...), lang: str = Form("zh")):
    if len(audios) == 0:
        return JSONResponse({"error": "Chưa có đoạn audio nào."}, status_code=400)
    if len(audios) > MAX_CLIPS_PER_BATCH:
        return JSONResponse({"error": f"Tối đa {MAX_CLIPS_PER_BATCH} đoạn/lần."}, status_code=400)

    lang_code = LID_DICT.get(lang, 3)
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
    }
    set_stage(JOBS[job_id], "start", "Đang khởi động...")
    t = threading.Thread(target=run_batch_asr_bg, args=(job_id, wavs, lang_code), daemon=True)
    t.start()
    return JSONResponse({"job_id": job_id, "n_clips": len(wavs)})


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


HTML_PAGE = """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SenseVoice NPU ASR Batch Demo</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 560px; margin: 0 auto; padding: 24px 16px; background: #0f1115; color: #eee; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .sub { color: #999; font-size: 13px; margin-bottom: 20px; line-height: 1.5; }
  select { width: 100%; padding: 10px; font-size: 16px; margin-bottom: 16px; border-radius: 8px; background: #1c1f26; color: #fff; border: 1px solid #333; }
  button { padding: 14px; font-size: 16px; border-radius: 12px; border: none; font-weight: 600; cursor: pointer; }
  #recBtn { background: #e5484d; color: white; width: 100%; margin-bottom: 10px; }
  #recBtn.recording { background: #ff6b6b; animation: pulse 1s infinite; }
  #sendBtn { background: #4ade80; color: #0a1f0f; width: 100%; margin-bottom: 16px; }
  #sendBtn:disabled { background: #333; color: #777; cursor: not-allowed; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

  #queueBox { background: #1c1f26; border-radius: 12px; padding: 12px; margin-bottom: 16px; border: 1px solid #333; }
  #queueCount { font-size: 13px; color: #999; margin-bottom: 8px; }
  .clip-row { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid #2a2d35; }
  .clip-row:last-child { border-bottom: none; }
  .clip-num { font-size: 12px; color: #666; width: 22px; }
  .clip-row audio { flex: 1; height: 32px; }
  .clip-del { background: #3a1a1a; color: #e5484d; border: none; border-radius: 6px; padding: 4px 8px; font-size: 12px; cursor: pointer; }

  #timeline { margin-top: 16px; }
  .tl-item { display: flex; align-items: flex-start; padding: 8px 0; border-left: 2px solid #2a2d35; margin-left: 9px; padding-left: 16px; position: relative; }
  .tl-item.active { border-left-color: #4ade80; }
  .tl-dot { position: absolute; left: -7px; top: 10px; width: 12px; height: 12px; border-radius: 50%; background: #444; }
  .tl-item.done .tl-dot { background: #4ade80; }
  .tl-item.active .tl-dot { background: #fbbf24; animation: pulse 1s infinite; }
  .tl-item.running .tl-dot { background: #22d3ee; box-shadow: 0 0 8px #22d3ee; animation: pulse 0.6s infinite; }
  .tl-label { flex: 1; font-size: 14px; }
  .tl-time { font-size: 13px; color: #888; font-variant-numeric: tabular-nums; margin-left: 8px; white-space: nowrap; }
  .tl-item.running .tl-label { color: #22d3ee; font-weight: 700; }
  .tl-item.active .tl-label { color: #fbbf24; font-weight: 600; }

  .summary { background: linear-gradient(135deg, #1a3a2a, #163024); border: 2px solid #4ade80; border-radius: 14px; padding: 16px; margin-top: 16px; }
  .summary .row { display: flex; justify-content: space-between; font-size: 14px; margin: 4px 0; }
  .summary .row b { color: #4ade80; }
  .summary .note { font-size: 11px; color: #6b9e80; margin-top: 8px; line-height: 1.5; }

  .result-table { margin-top: 14px; }
  .result-row { background: #1c1f26; border: 1px solid #2a2d35; border-radius: 10px; padding: 10px 12px; margin-bottom: 8px; }
  .result-row .idx { font-size: 11px; color: #666; }
  .result-row .txt { font-size: 15px; margin: 4px 0; }
  .result-row .ms { font-size: 12px; color: #22d3ee; }

  #logLink { display: inline-block; margin-top: 12px; color: #999; font-size: 12px; text-decoration: underline; cursor: pointer; }
  #logBox { display:none; background: #000; color: #0f0; font-family: monospace; font-size: 11px; padding: 10px; border-radius: 8px; margin-top: 8px; max-height: 300px; overflow-y: auto; white-space: pre-wrap; }
</style>
</head>
<body>
  <h1>🎙️ SenseVoice ASR Batch — NPU thật</h1>
  <div class="sub">Ghi NHIỀU đoạn (tối đa __MAX_CLIPS__ đoạn, mỗi đoạn ≤29s) → gửi 1 LƯỢT DUY NHẤT → chỉ trả "thuế tải model" 1 lần cho cả batch thay vì từng câu.</div>

  <select id="langSelect">
    <option value="zh">🇨🇳 Tiếng Trung (zh)</option>
    <option value="en">🇺🇸 Tiếng Anh (en)</option>
    <option value="ko">🇰🇷 Tiếng Hàn (ko)</option>
  </select>

  <button id="recBtn">🔴 Ghi đoạn tiếp theo</button>
  <button id="sendBtn" disabled>📤 Gửi tất cả (0 đoạn)</button>

  <div id="queueBox">
    <div id="queueCount">Chưa có đoạn nào. Bấm "Ghi đoạn tiếp theo" để bắt đầu.</div>
    <div id="clipList"></div>
  </div>

  <div id="timeline"></div>
  <div id="summaryBox"></div>
  <div id="resultTable" class="result-table"></div>
  <div id="logLink" style="display:none">▶ Xem log chi tiết</div>
  <pre id="logBox"></pre>

<script>
let mediaRecorder, currentChunks = [], stream;
let clips = [];  // {blob, url}
let recording = false;
const MAX_CLIPS = __MAX_CLIPS__;

const recBtn = document.getElementById('recBtn');
const sendBtn = document.getElementById('sendBtn');
const queueCount = document.getElementById('queueCount');
const clipList = document.getElementById('clipList');
const timelineEl = document.getElementById('timeline');
const summaryBox = document.getElementById('summaryBox');
const resultTable = document.getElementById('resultTable');
const logLink = document.getElementById('logLink');
const logBox = document.getElementById('logBox');
const langSelect = document.getElementById('langSelect');

function fmtMs(ms) { return ms >= 1000 ? (ms/1000).toFixed(2)+'s' : Math.round(ms)+'ms'; }
function fmtS(s) { return s < 1 ? Math.round(s*1000)+'ms' : s.toFixed(1)+'s'; }

function renderQueue() {
  queueCount.textContent = clips.length === 0
    ? 'Chưa có đoạn nào. Bấm "Ghi đoạn tiếp theo" để bắt đầu.'
    : clips.length + ' / ' + MAX_CLIPS + ' đoạn đã ghi.';
  clipList.innerHTML = clips.map((c, i) =>
    '<div class="clip-row"><span class="clip-num">#' + (i+1) + '</span>' +
    '<audio controls src="' + c.url + '"></audio>' +
    '<button class="clip-del" onclick="removeClip(' + i + ')">Xoá</button></div>'
  ).join('');
  sendBtn.disabled = clips.length === 0;
  sendBtn.textContent = '📤 Gửi tất cả (' + clips.length + ' đoạn)';
  recBtn.disabled = clips.length >= MAX_CLIPS;
}
window.removeClip = function(i) { clips.splice(i, 1); renderQueue(); };

recBtn.onclick = async () => {
  if (!recording) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      alert('Không truy cập được micro: ' + e.message);
      return;
    }
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
    recBtn.textContent = '⏹️ Dừng ghi đoạn #' + (clips.length + 1);
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
  sendBtn.disabled = true;
  recBtn.disabled = true;
  timelineEl.innerHTML = '';
  summaryBox.innerHTML = '';
  resultTable.innerHTML = '';
  logLink.style.display = 'none';
  logBox.style.display = 'none';

  const form = new FormData();
  clips.forEach((c, i) => form.append('audios', c.blob, 'clip' + i + '.webm'));
  form.append('lang', langSelect.value);

  let jobId;
  try {
    const resp = await fetch('/api/transcribe_batch', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { alert('Lỗi: ' + data.error); sendBtn.disabled = false; recBtn.disabled = false; return; }
    jobId = data.job_id;
  } catch (e) {
    alert('Lỗi kết nối: ' + e.message);
    sendBtn.disabled = false; recBtn.disabled = false;
    return;
  }
  pollStatus(jobId);
};

function renderTimeline(history, currentLabel, currentElapsed, currentKey) {
  let html = '';
  for (const h of history) {
    const isRunning = h.key.includes('RUNNING_INFERENCE');
    html += '<div class="tl-item done ' + (isRunning ? 'running' : '') + '"><div class="tl-dot"></div>' +
      '<div class="tl-label">' + h.label + '</div><div class="tl-time">' + fmtS(h.duration_s) + '</div></div>';
  }
  if (currentLabel) {
    const isRunning = (currentKey || '').includes('RUNNING_INFERENCE');
    html += '<div class="tl-item active ' + (isRunning ? 'running' : '') + '"><div class="tl-dot"></div>' +
      '<div class="tl-label">' + currentLabel + '</div><div class="tl-time">' + fmtS(currentElapsed) + '</div></div>';
  }
  timelineEl.innerHTML = html;
}

async function pollStatus(jobId) {
  try {
    const resp = await fetch('/api/status/' + jobId);
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }
    if (data.status === 'running') {
      renderTimeline(data.history, data.stage_label, data.stage_elapsed_s, data.stage_key);
      setTimeout(() => pollStatus(jobId), 1000);
    } else if (data.status === 'done') {
      const r = data.result;
      renderTimeline(r.history, null, 0, null);
      summaryBox.innerHTML =
        '<div class="summary">' +
        '<div class="row">⏱️ Tổng thời gian batch<b>' + fmtS(r.elapsed_total_s) + '</b></div>' +
        '<div class="row">⚡ NPU thuần (' + r.n_clips + ' đoạn × ' + fmtMs(r.npu_per_clip_ms) + ')<b>' + fmtMs(r.npu_total_ms) + '</b></div>' +
        '<div class="row">📡 Ngoài chip (tải model + hàng đợi + upload/download)<b>' + fmtS(r.outside_chip_s) + '</b></div>' +
        '<div class="note">NPU per-clip là hằng số đo qua Profiler (model shape tĩnh → thời gian không đổi theo nội dung), không phải đo sống từng đoạn — AI Hub không hỗ trợ đo riêng từng mẫu trong 1 batch job.</div>' +
        '</div>';
      resultTable.innerHTML = r.clips.map(c =>
        '<div class="result-row"><div class="idx">Đoạn #' + (c.index+1) + ' · ' + c.duration_s + 's</div>' +
        '<div class="txt">' + (c.text || '(không nhận diện được / im lặng)') + '</div>' +
        '<div class="ms">⚡ NPU: ' + fmtMs(c.npu_ms) + '</div></div>'
      ).join('');
      logLink.style.display = 'inline-block';
      logLink.onclick = async () => {
        if (logBox.style.display === 'none') {
          const lr = await fetch('/api/log/' + jobId);
          const ld = await lr.json();
          logBox.textContent = ld.log || '(không có log)';
          logBox.style.display = 'block';
          logLink.textContent = '▼ Ẩn log';
        } else {
          logBox.style.display = 'none';
          logLink.textContent = '▶ Xem log chi tiết';
        }
      };
      sendBtn.disabled = false;
      recBtn.disabled = clips.length >= MAX_CLIPS;
    } else if (data.status === 'error') {
      alert('Lỗi xử lý: ' + data.error);
      sendBtn.disabled = false;
      recBtn.disabled = false;
    }
  } catch (e) {
    setTimeout(() => pollStatus(jobId), 3000);
  }
}

renderQueue();
</script>
</body>
</html>
"""

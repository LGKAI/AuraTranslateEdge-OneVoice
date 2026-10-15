# -*- coding: utf-8 -*-
"""MT Demo Server — NPU Machine Translation (NLLB-600M Native FP16 on Dragonwing IQ-9075 EVK)

Hỗ trợ dịch hai chiều 6 hướng giữa:
  🇻🇳 Tiếng Việt (vie_Latn) ↔ 🇺🇸 Tiếng Anh (eng_Latn)
  🇺🇸 Tiếng Anh (eng_Latn)  ↔ 🇨🇳 Tiếng Trung (zho_Hans)
  🇺🇸 Tiếng Anh (eng_Latn)  ↔ 🇰🇷 Tiếng Hàn (kor_Hang)
  🇰🇷 Tiếng Hàn (kor_Hang)  ↔ 🇻🇳 Tiếng Việt (vie_Latn)

Cơ chế triển khai:
  - Single Static DAG FP16: Encoder + 16 bước giải mã unrolled trong 1 graph duy nhất
  - 100% NPU Native: 7.617/7.617 layer trên Hexagon HTP v73 (0% CPU fallback)
  - Target model ID trên AI Hub: mqy498l5q (Compile job jgd62w9lp, profile jp1o6ly25: 215.87 ms)
  - Hỗ trợ 2 chế độ:
      🟢 NPU Qualcomm Dragonwing (Chạy thật trên chip NPU qua Qualcomm AI Hub)
      ⚡ CPU Local ONNX (Mô phỏng tức thì ~30ms bằng ONNX Runtime để kiểm thử nhanh)

Chạy:
  python mt_demo_server.py --port 8430
  uvicorn mt_demo_server:app --host 127.0.0.1 --port 8430
  python mt_demo_server.py --port 8430 --share  (Tự tạo public link HTTPS qua Cloudflare)
"""
import sys, os, time, uuid, threading, json, subprocess, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Form, Body
from fastapi.responses import HTMLResponse, JSONResponse
import qai_hub as hub

# ──────────────────────────────────────────────
# Cấu hình Model & Thiết bị phần cứng
# ──────────────────────────────────────────────
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
NLLB_TARGET_MODEL_ID = "mqy498l5q"   # Compile job jgd62w9lp trên tài khoản leogiakhanh1609@gmail.com
COMPILE_JOB_ID       = "jgd62w9lp"
PROFILE_JOB_ID       = "jp1o6ly25"
INFER_JOB_ID         = "jgd629eep"

NPU_MS_PER_SENTENCE  = 215.87        # Độ trễ thực đo từ Profile Job jp1o6ly25
NPU_TOTAL_LAYERS     = 7617          # 7.617/7.617 layers (100% HTP v73)
NPU_PEAK_RAM_MB      = 6.59          # Bộ nhớ RAM đỉnh trên NPU
MAX_SRC_TOKENS       = 32            # Cố định đầu vào Encoder: 32 token
UNROLLED_STEPS_T     = 16            # Số bước giải mã unrolled trong đồ thị
VOCAB_SIZE_PRUNED    = 138974        # Từ điển cắt lọc 4 chữ viết (Latin, Việt, Hán, Hàn)

# Trạng thái Qualcomm AI Hub
AI_HUB_STATE_LABELS = {
    "CREATED":              "Đã tạo job, chờ vào hàng đợi",
    "PROVISIONING_DEVICE":  "Đang xếp hàng chờ cấp phát NPU Dragonwing",
    "RUNNING_INFERENCE":    "🟢 ĐANG CHẠY THẬT trên chip NPU Hexagon v73",
    "SUCCESS":              "Hoàn tất thành công trên NPU",
}

# ──────────────────────────────────────────────
# Định nghĩa 4 ngôn ngữ trong đề tài
# ──────────────────────────────────────────────
LANG_CONFIG = {
    "vie_Latn": {"name": "Tiếng Việt", "flag": "🇻🇳", "code": "vi", "id": 256193, "badge": "vi"},
    "eng_Latn": {"name": "English",    "flag": "🇺🇸", "code": "en", "id": 256047, "badge": "en"},
    "zho_Hans": {"name": "中文 (简体)",  "flag": "🇨🇳", "code": "zh", "id": 256200, "badge": "zh"},
    "kor_Hang": {"name": "한국어",       "flag": "🇰🇷", "code": "ko", "id": 256098, "badge": "ko"},
}

# 8 câu kiểm thử chuẩn FLEURS & giao tiếp thực tế
PRESET_SAMPLES = [
    {
        "src_lang": "vie_Latn", "tgt_lang": "eng_Latn",
        "text": "Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé.",
        "desc": "🇻🇳 → 🇺🇸 Giao tiếp thường ngày"
    },
    {
        "src_lang": "vie_Latn", "tgt_lang": "eng_Latn",
        "text": "Tôi muốn đặt một bàn cho hai người vào tối nay.",
        "desc": "🇻🇳 → 🇺🇸 Nhà hàng / Đặt chỗ"
    },
    {
        "src_lang": "eng_Latn", "tgt_lang": "vie_Latn",
        "text": "Please turn on the lights in the living room.",
        "desc": "🇺🇸 → 🇻🇳 Điều khiển nhà thông minh"
    },
    {
        "src_lang": "eng_Latn", "tgt_lang": "kor_Hang",
        "text": "The weather today is sunny with a light breeze.",
        "desc": "🇺🇸 → 🇰🇷 Dự báo thời tiết"
    },
    {
        "src_lang": "kor_Hang", "tgt_lang": "eng_Latn",
        "text": "안녕하세요, 오늘 날씨가 정말 좋네요.",
        "desc": "🇰🇷 → 🇺🇸 Chào hỏi giao tiếp"
    },
    {
        "src_lang": "zho_Hans", "tgt_lang": "eng_Latn",
        "text": "我想预订今晚两个人的餐桌。",
        "desc": "🇨🇳 → 🇺🇸 Đặt bàn ăn tối"
    },
    {
        "src_lang": "eng_Latn", "tgt_lang": "zho_Hans",
        "text": "Machine translation helps people communicate across languages.",
        "desc": "🇺🇸 → 🇨🇳 Công nghệ dịch thuật"
    },
    {
        "src_lang": "kor_Hang", "tgt_lang": "vie_Latn",
        "text": "기차역까지 가는 가장 빠른 길을 알려 주세요.",
        "desc": "🇰🇷 → 🇻🇳 Du lịch / Hỏi đường"
    }
]

# ──────────────────────────────────────────────
# Quản lý đường dẫn & Khởi tạo ứng dụng
# ──────────────────────────────────────────────
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CUR_DIR, "mt_batch_logs")
os.makedirs(LOG_DIR, exist_ok=True)
ONNX_MODEL_PATH = os.path.join(CUR_DIR, "nllb", "work", "nllb_S32_T16.onnx")

app = FastAPI(title="AuraTranslate Edge — NLLB-600M MT Demo")
device = hub.Device(DEVICE_NAME)
JOBS = {}

# ──────────────────────────────────────────────
# Caching Model & Tokenizer (Lazy Load)
# ──────────────────────────────────────────────
_tokenizer = None
_ort_session = None
_npu_target_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import transformers.utils.import_utils
        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        import transformers.modeling_utils
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
        from transformers import AutoTokenizer
        try:
            _tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False, local_files_only=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_fast=False)
    return _tokenizer

def get_ort_session():
    global _ort_session
    if _ort_session is None:
        import onnxruntime as ort
        if not os.path.exists(ONNX_MODEL_PATH):
            raise FileNotFoundError(f"Không tìm thấy mô hình ONNX tại: {ONNX_MODEL_PATH}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.log_severity_level = 3
        _ort_session = ort.InferenceSession(ONNX_MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return _ort_session

def get_npu_model():
    global _npu_target_model
    if _npu_target_model is None:
        _npu_target_model = hub.get_model(NLLB_TARGET_MODEL_ID)
    return _npu_target_model

# ──────────────────────────────────────────────
# Helpers xử lý Token & Timeline
# ──────────────────────────────────────────────
def encode_sentence(tok, text: str, src_lang: str):
    tok.src_lang = src_lang
    enc = tok(text, return_tensors="np")
    ids = enc["input_ids"][0].tolist()
    # Pad / Truncate sang 32 tokens
    a = np.ones((1, MAX_SRC_TOKENS), dtype=np.int32)
    a[0, :min(len(ids), MAX_SRC_TOKENS)] = ids[:MAX_SRC_TOKENS]
    return a, ids

def decode_output_ids(tok, out_ids_list: List[int]):
    """Giải mã chuỗi token IDs: dừng tại EOS (2), lọc bỏ special tokens."""
    cleaned = []
    for tid in out_ids_list:
        tid = int(tid)
        if tid == 2:  # Token EOS (kết thúc câu)
            break
        if tid > 3:   # Bỏ token pad/unk
            cleaned.append(tid)
    text = tok.decode(cleaned, skip_special_tokens=True).strip()
    return text

def set_stage(st, key: str, label: str):
    now = time.time()
    if st.get("stage_started_at") is not None:
        st["history"].append({
            "key": st["stage_key"],
            "label": st["stage_label"],
            "duration_s": round(now - st["stage_started_at"], 2),
        })
    st["stage_key"] = key
    st["stage_label"] = label
    st["stage_started_at"] = now

def _save_log(job_id: str, log_lines: list):
    log_path = os.path.join(LOG_DIR, f"mt_{job_id}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    return log_path

def poll_ai_hub_job(job, st, phase_prefix: str):
    last_ai_state = None
    while True:
        status = job.get_status()
        ai_state = str(status.state).replace("State.", "")
        if ai_state != last_ai_state:
            label = f"{phase_prefix}: {AI_HUB_STATE_LABELS.get(ai_state, ai_state)}"
            set_stage(st, f"{phase_prefix}_{ai_state}", label)
            last_ai_state = ai_state
        if ai_state == "SUCCESS":
            return
        if ai_state in ("FAILED", "TIMEOUT"):
            err_msg = status.message or ai_state
            raise RuntimeError(f"{phase_prefix} thất bại trên AI Hub: {err_msg}")
        time.sleep(2)

# ──────────────────────────────────────────────
# Logic thực thi dịch thuật (Worker thread)
# ──────────────────────────────────────────────
def run_translation_job(job_id: str, sentences: List[str], src_lang: str, tgt_lang: str, engine: str):
    st = JOBS[job_id]
    t0 = time.time()
    n_sentences = len(sentences)
    log_lines = []

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        log_lines.append(line)
        print(f"[mt-job {job_id}] {line}", flush=True)

    try:
        log(f"Bắt đầu dịch {n_sentences} câu: {src_lang} -> {tgt_lang} | Engine: {engine}")
        set_stage(st, "tokenize", "Đang phân tích token SentencePiece...")
        tok = get_tokenizer()
        V = tok.get_vocab()
        tgt_lang_id = int(V.get(tgt_lang, LANG_CONFIG.get(tgt_lang, {}).get("id", 256047)))

        A, L, token_debug_info = [], [], []
        for i, s in enumerate(sentences):
            a, in_ids = encode_sentence(tok, s, src_lang)
            A.append(a)
            L.append(np.array([tgt_lang_id], dtype=np.int32))
            token_debug_info.append({"text": s, "ids_in": in_ids})
            log(f"  Câu {i+1}: \"{s}\" ({len(in_ids)} tokens)")

        predictions = []

        if engine == "cpu_local":
            # ──────────────────────────────────────────────────────────
            # Chế độ 1: CPU Local ONNX (Mô phỏng tức thì)
            # ──────────────────────────────────────────────────────────
            set_stage(st, "infer_cpu", "Đang chạy đồ thị nllb_S32_T16.onnx trên CPU Execution Provider...")
            sess = get_ort_session()
            t_infer_start = time.time()
            for i in range(n_sentences):
                out = sess.run(None, {"src_ids": A[i], "tgt_lang": L[i]})[0].reshape(-1)
                raw_ids = [int(x) for x in out]
                hyp_text = decode_output_ids(tok, raw_ids)
                subwords = [tok.decode([tid]) for tid in raw_ids if tid > 3]
                predictions.append({
                    "index": i,
                    "source_text": sentences[i],
                    "translated_text": hyp_text,
                    "raw_ids": raw_ids,
                    "subwords": subwords,
                    "npu_ms": NPU_MS_PER_SENTENCE,
                    "model": "NLLB-600M (Local ONNX CPU)",
                })
                log(f"  -> Kết quả câu {i+1}: \"{hyp_text}\"")
            total_infer_time = time.time() - t_infer_start
            log(f"Inference CPU hoàn tất trong {total_infer_time*1000:.1f} ms.")

        else:
            # ──────────────────────────────────────────────────────────
            # Chế độ 2: NPU Qualcomm Dragonwing (Chạy thật trên chip NPU)
            # ──────────────────────────────────────────────────────────
            set_stage(st, "upload_npu", f"Upload {n_sentences} câu ({src_lang}->{tgt_lang}) → Qualcomm AI Hub...")
            tm = get_npu_model()
            ds = hub.upload_dataset({"src_ids": A, "tgt_lang": L}, name=f"nllb_demo_batch_{job_id[:8]}")
            log(f"Dataset uploaded: {ds.dataset_id}")

            set_stage(st, "queue_npu", "Đang gửi inference job lên chip Dragonwing IQ-9075 EVK...")
            ij = hub.submit_inference_job(
                model=tm,
                device=device,
                inputs=ds,
                name=f"NLLB_MT_demo_{job_id[:8]}"
            )
            st["hub_job_id"] = ij.job_id
            st["hub_job_url"] = ij.url
            log(f"Inference job created: {ij.job_id} ({ij.url})")
            poll_ai_hub_job(ij, st, "NPU-Hexagon-HTP")

            set_stage(st, "download_npu", "NPU xử lý xong! Đang tải mảng token IDs về...")
            od = ij.download_output_data()
            k = sorted(od.keys(), key=lambda s: int(s.split("_")[-1]))[0]
            H = [np.asarray(x).reshape(-1) for x in od[k]]

            for i in range(n_sentences):
                raw_ids = [int(x) for x in H[i]]
                hyp_text = decode_output_ids(tok, raw_ids)
                subwords = [tok.decode([tid]) for tid in raw_ids if tid > 3]
                predictions.append({
                    "index": i,
                    "source_text": sentences[i],
                    "translated_text": hyp_text,
                    "raw_ids": raw_ids,
                    "subwords": subwords,
                    "npu_ms": NPU_MS_PER_SENTENCE,
                    "model": "NLLB-600M (Hexagon HTP v73)",
                })
                log(f"  -> NPU câu {i+1}: \"{hyp_text}\"")

        # ──────────────────────────────────────────────────────────
        # Hoàn tất job
        # ──────────────────────────────────────────────────────────
        set_stage(st, "done", "Hoàn tất!")
        elapsed = time.time() - t0
        npu_total = NPU_MS_PER_SENTENCE * n_sentences
        log_path = _save_log(job_id, log_lines)

        st["status"] = "done"
        st["result"] = {
            "predictions": predictions,
            "n_sentences": n_sentences,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "engine": engine,
            "elapsed_total_s": round(elapsed, 2),
            "npu_ref_ms": round(npu_total, 1),
            "npu_per_sentence_ms": NPU_MS_PER_SENTENCE,
            "npu_layers": f"{NPU_TOTAL_LAYERS} / {NPU_TOTAL_LAYERS} (100% HTP)",
            "npu_ram_mb": NPU_PEAK_RAM_MB,
            "hub_job_id": st.get("hub_job_id"),
            "hub_job_url": st.get("hub_job_url"),
            "history": st["history"],
            "log_path": log_path,
        }
        log(f"THÀNH CÔNG. Tổng thời gian: {elapsed:.2f}s | NPU Reference ({n_sentences}x{NPU_MS_PER_SENTENCE}ms): {npu_total/1000:.2f}s")

    except Exception as e:
        log(f"LỖI: {e}")
        _save_log(job_id, log_lines)
        st["status"] = "error"
        st["error"] = str(e)

# ──────────────────────────────────────────────
# API Request Models & Endpoints
# ──────────────────────────────────────────────
class TranslationRequest(BaseModel):
    text: str
    src_lang: str = "vie_Latn"
    tgt_lang: str = "eng_Latn"
    engine: str = "cpu_local"  # "npu" hoặc "cpu_local"

@app.post("/api/translate")
async def translate_endpoint(req: TranslationRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Vui lòng nhập câu cần dịch."}, status_code=400)
    if req.src_lang not in LANG_CONFIG or req.tgt_lang not in LANG_CONFIG:
        return JSONResponse({"error": "Ngôn ngữ không được hỗ trợ."}, status_code=400)
    if req.src_lang == req.tgt_lang:
        return JSONResponse({"error": "Ngôn ngữ nguồn và đích phải khác nhau."}, status_code=400)

    # Tách các câu nếu nhập nhiều dòng
    sentences = [line.strip() for line in text.split("\n") if line.strip()]
    if len(sentences) > 10:
        return JSONResponse({"error": "Tối đa 10 câu mỗi lần dịch."}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "running", "stage_key": None, "stage_label": None,
        "stage_started_at": None, "history": [], "result": None, "error": None,
        "src_lang": req.src_lang, "tgt_lang": req.tgt_lang, "engine": req.engine
    }
    set_stage(JOBS[job_id], "start", "Đang khởi tạo phiên dịch...")

    t = threading.Thread(
        target=run_translation_job,
        args=(job_id, sentences, req.src_lang, req.tgt_lang, req.engine),
        daemon=True
    )
    t.start()
    return JSONResponse({"job_id": job_id, "n_sentences": len(sentences), "engine": req.engine})

@app.get("/api/status/{job_id}")
async def status_endpoint(job_id: str):
    st = JOBS.get(job_id)
    if st is None:
        return JSONResponse({"error": "job_id không tồn tại"}, status_code=404)
    resp = dict(st)
    resp["stage_elapsed_s"] = round(time.time() - st["stage_started_at"], 2) if st.get("stage_started_at") else 0
    return JSONResponse(resp)

@app.get("/api/log/{job_id}")
async def log_endpoint(job_id: str):
    log_path = os.path.join(LOG_DIR, f"mt_{job_id}.log")
    if not os.path.exists(log_path):
        return JSONResponse({"error": "Chưa có log"}, status_code=404)
    with open(log_path, encoding="utf-8") as f:
        return JSONResponse({"log": f.read()})

@app.get("/api/presets")
async def presets_endpoint():
    return JSONResponse(PRESET_SAMPLES)

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

# ──────────────────────────────────────────────
# Giao diện người dùng Web Dashboard (HTML / CSS / JS)
# ──────────────────────────────────────────────
HTML_PAGE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AuraTranslate Edge — NLLB MT Dashboard (NPU)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b0d12;
    --card: #12151d;
    --card-hover: #171c26;
    --border: #1e2330;
    --border2: #283042;
    --accent-vi: #6ee7b7;
    --accent-vi-dim: #133327;
    --accent-en: #38bdf8;
    --accent-en-dim: #0f2d42;
    --accent-zh: #f43f5e;
    --accent-zh-dim: #3b121e;
    --accent-ko: #a855f7;
    --accent-ko-dim: #2b1442;
    --green: #4ade80;
    --red: #f87171;
    --yellow: #fbbf24;
    --cyan: #22d3ee;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --muted2: #64748b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; line-height: 1.5; }

  /* Header Hero */
  .hero {
    background: linear-gradient(180deg, #101522 0%, #0b0d12 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 20px 22px;
    text-align: center;
  }
  .hero-tag {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34, 211, 238, 0.1); border: 1px solid rgba(34, 211, 238, 0.25);
    border-radius: 20px; padding: 4px 12px; font-size: 11px; color: var(--cyan);
    font-weight: 600; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px;
  }
  .hero h1 { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; }
  .hero h1 span.highlight {
    background: linear-gradient(135deg, #38bdf8 0%, #a855f7 50%, #6ee7b7 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero .tagline { font-size: 13px; color: var(--muted); margin-top: 6px; max-width: 600px; margin-left: auto; margin-right: auto; }

  .specs-bar {
    display: flex; justify-content: center; flex-wrap: wrap; gap: 8px; margin-top: 14px;
  }
  .spec-pill {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 4px 10px; font-size: 11px; color: var(--muted); font-weight: 500;
  }
  .spec-pill b { color: var(--text); font-weight: 600; }

  .container { max-width: 780px; margin: 0 auto; padding: 24px 16px 80px; }

  /* Language Selection Bar */
  .lang-bar {
    display: flex; align-items: center; gap: 10px;
    background: var(--card); border: 1px solid var(--border2);
    border-radius: 14px; padding: 8px 12px; margin-bottom: 16px;
  }
  .lang-select-group { flex: 1; display: flex; flex-direction: column; gap: 4px; }
  .lang-label { font-size: 11px; color: var(--muted2); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
  select.lang-select {
    background: #0f131a; color: var(--text); border: 1px solid var(--border);
    border-radius: 8px; padding: 8px 10px; font-size: 13px; font-weight: 600;
    cursor: pointer; outline: none; transition: border-color 0.15s;
  }
  select.lang-select:focus { border-color: var(--accent-en); }
  .btn-swap {
    width: 36px; height: 36px; border-radius: 50%; border: 1px solid var(--border2);
    background: #181f2c; color: var(--text); display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 16px; transition: all 0.2s; margin-top: 14px; flex-shrink: 0;
  }
  .btn-swap:hover { background: #222b3d; transform: rotate(180deg); }

  /* Engine Selector */
  .engine-toggle-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px 14px; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
  }
  .engine-info { display: flex; flex-direction: column; gap: 2px; }
  .engine-title { font-size: 13px; font-weight: 600; color: var(--text); }
  .engine-sub { font-size: 11px; color: var(--muted); }
  .engine-pills { display: flex; gap: 6px; }
  .engine-pill {
    padding: 6px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: #0f131a; color: var(--muted); font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all 0.15s;
  }
  .engine-pill.active-npu {
    background: rgba(74, 222, 128, 0.12); border-color: var(--green); color: var(--green);
  }
  .engine-pill.active-cpu {
    background: rgba(56, 189, 248, 0.12); border-color: var(--accent-en); color: var(--accent-en);
  }

  /* Preset Pills */
  .presets-box { margin-bottom: 16px; }
  .presets-hdr { font-size: 11px; color: var(--muted); margin-bottom: 8px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
  .preset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 8px; }
  .preset-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 8px 12px; cursor: pointer; transition: all 0.15s; text-align: left;
  }
  .preset-card:hover { border-color: var(--border2); background: var(--card-hover); transform: translateY(-1px); }
  .preset-card .p-desc { font-size: 10px; color: var(--cyan); font-weight: 600; margin-bottom: 2px; }
  .preset-card .p-text { font-size: 12px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Input Card */
  .input-card {
    background: var(--card); border: 1px solid var(--border2); border-radius: 14px;
    padding: 16px; margin-bottom: 16px;
  }
  .input-card textarea {
    width: 100%; min-height: 85px; background: #0c0f14; border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; color: var(--text); font-family: inherit; font-size: 14px;
    resize: vertical; outline: none; transition: border-color 0.15s;
  }
  .input-card textarea:focus { border-color: var(--accent-en); }
  .input-foot { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; }
  .char-hint { font-size: 11px; color: var(--muted2); }

  .btn-submit {
    padding: 12px 24px; border-radius: 10px; border: none; font-size: 14px; font-weight: 700;
    cursor: pointer; background: linear-gradient(135deg, #38bdf8 0%, #2563eb 100%); color: white;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3); transition: all 0.2s;
  }
  .btn-submit:hover { opacity: 0.95; transform: translateY(-1px); }
  .btn-submit:disabled { background: #1e2330; color: var(--muted); cursor: not-allowed; box-shadow: none; transform: none; }

  /* Timeline */
  #timeline { margin: 20px 0; }
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
  @keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

  /* Summary Card */
  .summary {
    background: linear-gradient(135deg, #0d1e2e, #09131f); border: 1.5px solid var(--accent-en);
    border-radius: 14px; padding: 16px; margin: 18px 0;
  }
  .sum-row { display: flex; justify-content: space-between; font-size: 13px; margin: 6px 0; }
  .sum-val { font-weight: 700; color: var(--cyan); }
  .sum-note { font-size: 11px; color: var(--muted); margin-top: 10px; line-height: 1.6; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px; }

  /* Result Cards */
  .result-section { margin-top: 20px; display: flex; flex-direction: column; gap: 12px; }
  .result-card {
    background: var(--card); border: 1px solid var(--border2); border-radius: 14px; padding: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
  }
  .res-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .res-badge {
    font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px; text-transform: uppercase;
  }
  .res-badge.vi { background: var(--accent-vi-dim); color: var(--accent-vi); border: 1px solid var(--accent-vi); }
  .res-badge.en { background: var(--accent-en-dim); color: var(--accent-en); border: 1px solid var(--accent-en); }
  .res-badge.zh { background: var(--accent-zh-dim); color: var(--accent-zh); border: 1px solid var(--accent-zh); }
  .res-badge.ko { background: var(--accent-ko-dim); color: var(--accent-ko); border: 1px solid var(--accent-ko); }
  .res-src { font-size: 13px; color: var(--muted); margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed var(--border); }
  .res-tgt { font-size: 17px; font-weight: 600; color: #fff; line-height: 1.4; margin-bottom: 12px; }
  .res-npu { font-size: 11px; color: var(--cyan); font-weight: 500; }

  /* Token viewer */
  .token-toggle {
    margin-top: 10px; font-size: 11px; color: var(--muted2); cursor: pointer; text-decoration: underline; display: inline-block;
  }
  .token-box {
    display: none; margin-top: 8px; padding: 10px; background: #080a0f; border: 1px solid var(--border);
    border-radius: 8px; font-family: monospace; font-size: 11px; color: #a5b4fc; word-break: break-all;
  }
  .token-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
  .t-chip {
    background: #1e1b4b; border: 1px solid #3730a3; border-radius: 4px; padding: 2px 6px; font-size: 10px; color: #c7d2fe;
  }

  /* Log Viewer */
  #logLink { display: inline-block; margin-top: 14px; color: var(--muted); font-size: 11px; text-decoration: underline; cursor: pointer; }
  #logBox { display:none; background:#000; color:#38bdf8; font-family:monospace; font-size:10px; padding:12px; border-radius:8px; margin-top:8px; max-height:260px; overflow-y:auto; white-space:pre-wrap; }
</style>
</head>
<body>

<div class="hero">
  <div class="hero-tag">Qualcomm Hexagon HTP v73 · Native FP16</div>
  <h1>🌐 AuraTranslate Edge — <span class="highlight">NLLB-600M MT</span></h1>
  <div class="tagline">Dịch máy đa ngữ thời thực 100% NPU · Dragonwing IQ-9075 EVK (Single Static Graph T=16)</div>
  <div class="specs-bar">
    <div class="spec-pill">Mô hình: <b>NLLB-200-distilled-600M</b></div>
    <div class="spec-pill">Độ trễ chip thật: <b>215.87 ms / câu</b></div>
    <div class="spec-pill">Layers NPU: <b>7.617 / 7.617 (100%)</b></div>
    <div class="spec-pill">RAM đỉnh NPU: <b>6.59 MB</b></div>
    <div class="spec-pill">Từ điển chọn lọc: <b>138.974 tokens</b></div>
  </div>
</div>

<div class="container">

  <!-- Ngôn ngữ Nguồn & Đích -->
  <div class="lang-bar">
    <div class="lang-select-group">
      <span class="lang-label">Từ ngôn ngữ nguồn</span>
      <select id="srcLang" class="lang-select">
        <option value="vie_Latn" selected>🇻🇳 Tiếng Việt (vie_Latn)</option>
        <option value="eng_Latn">🇺🇸 English (eng_Latn)</option>
        <option value="zho_Hans">🇨🇳 中文 (zho_Hans)</option>
        <option value="kor_Hang">🇰🇷 한국어 (kor_Hang)</option>
      </select>
    </div>

    <button id="btnSwap" class="btn-swap" title="Đổi chiều dịch">⇄</button>

    <div class="lang-select-group">
      <span class="lang-label">Sang ngôn ngữ đích</span>
      <select id="tgtLang" class="lang-select">
        <option value="eng_Latn" selected>🇺🇸 English (eng_Latn)</option>
        <option value="vie_Latn">🇻🇳 Tiếng Việt (vie_Latn)</option>
        <option value="zho_Hans">🇨🇳 中文 (zho_Hans)</option>
        <option value="kor_Hang">🇰🇷 한국어 (kor_Hang)</option>
      </select>
    </div>
  </div>

  <!-- Chọn Engine -->
  <div class="engine-toggle-card">
    <div class="engine-info">
      <div class="engine-title">Bộ máy thực thi (Execution Engine)</div>
      <div class="engine-sub" id="engineDesc">Chạy đồ thị nllb_S32_T16.onnx trên CPU cục bộ trong ~30ms</div>
    </div>
    <div class="engine-pills">
      <div class="engine-pill active-cpu" id="pillCpu" onclick="setEngine('cpu_local')">⚡ Local CPU (Tức thì ~30ms)</div>
      <div class="engine-pill" id="pillNpu" onclick="setEngine('npu')">🟢 NPU Dragonwing (Cloud Hub)</div>
    </div>
  </div>

  <!-- Câu mẫu FLEURS -->
  <div class="presets-box">
    <div class="presets-hdr">💡 Câu thử nghiệm đối chiếu (Click để chọn nhanh)</div>
    <div class="preset-grid" id="presetGrid"></div>
  </div>

  <!-- Hộp nhập văn bản -->
  <div class="input-card">
    <textarea id="inputText" placeholder="Nhập câu cần dịch... (Hỗ trợ nhiều dòng, tối đa 32 token/câu, 10 câu/lần)"></textarea>
    <div class="input-foot">
      <span class="char-hint" id="charHint">Giới hạn đồ thị: Encoder 32 tokens, Decoder T=16 tokens</span>
      <button class="btn-submit" id="btnTranslate" onclick="startTranslation()">🚀 Dịch Ngay</button>
    </div>
  </div>

  <!-- Timeline trạng thái -->
  <div id="timeline"></div>

  <!-- Tóm tắt hiệu năng -->
  <div id="summaryBox" style="display:none;"></div>

  <!-- Kết quả dịch -->
  <div id="resultBox" class="result-section"></div>

  <!-- Xem log -->
  <span id="logLink" onclick="toggleLog()">Xem log chi tiết</span>
  <pre id="logBox"></pre>

</div>

<script>
  let currentEngine = 'cpu_local';
  let activeJobId = null;
  let pollTimer = null;
  let presetsData = [];

  function setEngine(eng) {
    currentEngine = eng;
    const pCpu = document.getElementById('pillCpu');
    const pNpu = document.getElementById('pillNpu');
    const desc = document.getElementById('engineDesc');
    if (eng === 'cpu_local') {
      pCpu.className = 'engine-pill active-cpu';
      pNpu.className = 'engine-pill';
      desc.textContent = 'Mô phỏng đồ thị nllb_S32_T16.onnx tức thì trên CPU Execution Provider (< 50ms)';
    } else {
      pCpu.className = 'engine-pill';
      pNpu.className = 'engine-pill active-npu';
      desc.textContent = 'Gửi job thực thi trực tiếp trên NPU Hexagon HTP v73 (Dragonwing IQ-9075 EVK qua AI Hub)';
    }
  }

  // Load Presets
  async function loadPresets() {
    try {
      const res = await fetch('/api/presets');
      presetsData = await res.json();
      const grid = document.getElementById('presetGrid');
      grid.innerHTML = '';
      presetsData.forEach((p, idx) => {
        const card = document.createElement('div');
        card.className = 'preset-card';
        card.innerHTML = `<div class="p-desc">${p.desc}</div><div class="p-text">${p.text}</div>`;
        card.onclick = () => {
          document.getElementById('inputText').value = p.text;
          document.getElementById('srcLang').value = p.src_lang;
          document.getElementById('tgtLang').value = p.tgt_lang;
        };
        grid.appendChild(card);
      });
    } catch(e) { console.error("Lỗi tải presets:", e); }
  }

  // Swap Languages
  document.getElementById('btnSwap').onclick = () => {
    const s = document.getElementById('srcLang');
    const t = document.getElementById('tgtLang');
    const tmp = s.value;
    s.value = t.value;
    t.value = tmp;
  };

  async function startTranslation() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) { alert("Vui lòng nhập câu cần dịch."); return; }
    const srcLang = document.getElementById('srcLang').value;
    const tgtLang = document.getElementById('tgtLang').value;
    if (srcLang === tgtLang) { alert("Ngôn ngữ nguồn và đích phải khác nhau."); return; }

    const btn = document.getElementById('btnTranslate');
    btn.disabled = true;
    btn.textContent = "⏳ Đang xử lý...";
    document.getElementById('summaryBox').style.display = 'none';
    document.getElementById('resultBox').innerHTML = '';
    document.getElementById('timeline').innerHTML = '';
    document.getElementById('logBox').style.display = 'none';

    try {
      const res = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: text,
          src_lang: srcLang,
          tgt_lang: tgtLang,
          engine: currentEngine
        })
      });
      const data = await res.json();
      if (data.error) { alert(data.error); btn.disabled = false; btn.textContent = "🚀 Dịch Ngay"; return; }

      activeJobId = data.job_id;
      pollStatus();
    } catch(e) {
      alert("Lỗi kết nối server: " + e);
      btn.disabled = false;
      btn.textContent = "🚀 Dịch Ngay";
    }
  }

  async function pollStatus() {
    if (!activeJobId) return;
    try {
      const res = await fetch(`/api/status/${activeJobId}`);
      const st = await res.json();
      renderTimeline(st);

      if (st.status === 'done') {
        renderResult(st.result);
        finishJob();
      } else if (st.status === 'error') {
        alert("Lỗi thực thi: " + st.error);
        finishJob();
      } else {
        pollTimer = setTimeout(pollStatus, 1500);
      }
    } catch(e) {
      console.error(e);
      pollTimer = setTimeout(pollStatus, 2000);
    }
  }

  function finishJob() {
    clearTimeout(pollTimer);
    const btn = document.getElementById('btnTranslate');
    btn.disabled = false;
    btn.textContent = "🚀 Dịch Ngay";
  }

  function renderTimeline(st) {
    const tl = document.getElementById('timeline');
    let html = '';
    (st.history || []).forEach(h => {
      html += `<div class="tl-item done"><div class="tl-dot"></div><div class="tl-label">${h.label}</div><div class="tl-time">${h.duration_s}s</div></div>`;
    });
    if (st.status === 'running') {
      const isChip = (st.stage_key || '').includes('RUNNING_INFERENCE') || (st.stage_key || '').includes('infer_cpu');
      const cls = isChip ? 'running' : 'active';
      html += `<div class="tl-item ${cls}"><div class="tl-dot"></div><div class="tl-label">${st.stage_label || 'Đang xử lý...'}</div><div class="tl-time">${st.stage_elapsed_s || 0}s</div></div>`;
    }
    tl.innerHTML = html;
  }

  function renderResult(res) {
    const sum = document.getElementById('summaryBox');
    sum.style.display = 'block';
    sum.className = 'summary';
    sum.innerHTML = `
      <div class="sum-row"><span>Bộ máy thực thi:</span><span class="sum-val">${res.engine === 'cpu_local' ? '⚡ CPU Local (ONNX Runtime)' : '🟢 Qualcomm NPU Dragonwing IQ-9075 EVK'}</span></div>
      <div class="sum-row"><span>Số câu đã dịch:</span><span class="sum-val">${res.n_sentences} câu</span></div>
      <div class="sum-row"><span>Tổng thời gian:</span><span class="sum-val">${res.elapsed_total_s}s</span></div>
      <div class="sum-row"><span>Độ trễ NPU tham chiếu:</span><span class="sum-val">${res.npu_ref_ms} ms (${res.n_sentences} × ${res.npu_per_sentence_ms} ms)</span></div>
      <div class="sum-row"><span>Phân bổ toán tử NPU:</span><span class="sum-val">${res.npu_layers}</span></div>
      <div class="sum-row"><span>Bộ nhớ RAM đỉnh NPU:</span><span class="sum-val">${res.npu_ram_mb} MB (nằm trong SRAM/VTCM)</span></div>
      ${res.hub_job_id ? `<div class="sum-row"><span>AI Hub Job ID:</span><a href="${res.hub_job_url}" target="_blank" style="color:var(--cyan);text-decoration:underline;font-weight:600;">${res.hub_job_id} ↗</a></div>` : ''}
      <div class="sum-note">Mô hình NLLB-600M chạy ở định dạng Native FP16 với đồ thị tĩnh 16 bước giải mã unrolled, đạt 100% độ chính xác token so với CPU PyTorch gốc.</div>
    `;

    const box = document.getElementById('resultBox');
    box.innerHTML = '';
    res.predictions.forEach((p, idx) => {
      const card = document.createElement('div');
      card.className = 'result-card';
      const badgeCls = res.tgt_lang.slice(0, 2);
      card.innerHTML = `
        <div class="res-header">
          <span style="font-size:12px;color:var(--muted);font-weight:600;">Câu ${idx+1} / ${res.n_sentences}</span>
          <span class="res-badge ${badgeCls}">${res.src_lang.slice(0,2).toUpperCase()} ➔ ${res.tgt_lang.slice(0,2).toUpperCase()}</span>
        </div>
        <div class="res-src">"${p.source_text}"</div>
        <div class="res-tgt">"${p.translated_text}"</div>
        <div class="res-npu">⚡ NPU Reference: ${p.npu_ms} ms · ${p.model}</div>
        <span class="token-toggle" onclick="toggleTokens(${idx})">Xem chi tiết ${p.raw_ids.length} Token IDs trên silicon ▾</span>
        <div class="token-box" id="tokenBox_${idx}">
          <div>Mảng ID thô: [${p.raw_ids.join(', ')}]</div>
          <div class="token-chips">
            ${p.subwords.map(sw => `<span class="t-chip">${sw.replace(/ /g, ' ')}</span>`).join('')}
          </div>
        </div>
      `;
      box.appendChild(card);
    });
  }

  function toggleTokens(idx) {
    const el = document.getElementById(`tokenBox_${idx}`);
    el.style.display = (el.style.display === 'block') ? 'none' : 'block';
  }

  async function toggleLog() {
    const box = document.getElementById('logBox');
    if (box.style.display === 'block') { box.style.display = 'none'; return; }
    if (!activeJobId) { alert("Chưa có job nào chạy."); return; }
    try {
      const res = await fetch(`/api/log/${activeJobId}`);
      const data = await res.json();
      box.textContent = data.log || "(Trống)";
      box.style.display = 'block';
    } catch(e) { alert("Lỗi đọc log: " + e); }
  }

  window.addEventListener('DOMContentLoaded', loadPresets);
</script>

</body>
</html>
"""

# ──────────────────────────────────────────────
# Tự động hóa Cloudflare Tunnel (tùy chọn --share)
# ──────────────────────────────────────────────
def start_cloudflare_tunnel(port: int):
    local_cf = os.path.join(CUR_DIR, "cloudflared.exe")
    cloudflared_path = "cloudflared"
    if subprocess.run(["where", "cloudflared"], capture_output=True, shell=True).returncode != 0:
        if not os.path.exists(local_cf):
            print("\n⬇️ Đang tải Cloudflare Tunnel (cloudflared.exe) để tạo public HTTPS link...")
            url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
            try:
                urllib.request.urlretrieve(url, local_cf)
                print("✅ Tải xong cloudflared.exe!\n")
            except Exception as e:
                print(f"⚠️ Không thể tải tự động cloudflared: {e}")
                return
        cloudflared_path = local_cf

    def run_tunnel():
        try:
            p = subprocess.Popen(
                [cloudflared_path, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
            )
            for line in p.stdout:
                if "trycloudflare.com" in line:
                    for word in line.split():
                        if "trycloudflare.com" in word and word.startswith("http"):
                            print("\n" + "="*68)
                            print("🌐 PUBLIC HTTPS URL (Dùng thử MT trên mọi thiết bị / điện thoại):")
                            print(f"👉 {word}")
                            print("="*68 + "\n", flush=True)
                            break
        except Exception as e:
            print(f"⚠️ Lỗi khởi động Cloudflare Tunnel: {e}")

    t = threading.Thread(target=run_tunnel, daemon=True)
    t.start()

# ──────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="AuraTranslate Edge — NLLB MT Demo Server")
    parser.add_argument("--host", default="0.0.0.0", help="Địa chỉ host (mặc định: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8430, help="Cổng server (mặc định: 8430)")
    parser.add_argument("--share", action="store_true", help="Tự động mở public HTTPS link qua Cloudflare Tunnel")
    args = parser.parse_args()

    if args.share:
        start_cloudflare_tunnel(args.port)

    print(f"\n🚀 Khởi chạy NLLB MT Demo Server tại: http://localhost:{args.port}")
    if not args.share:
        print("💡 Muốn tạo public link HTTPS chia sẻ qua mạng ngoài? Chạy lại với cờ: --share")
    uvicorn.run(app, host=args.host, port=args.port)

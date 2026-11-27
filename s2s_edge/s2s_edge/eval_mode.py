# -*- coding: utf-8 -*-
"""OVAC Phase 3 automated offline Eval Mode runner.

Follows the official Phase 3 Eval Mode Packaging Guide:
  - Reads <input_dir>/manifest.csv (columns: id,audio,direction)
  - Writes <output_dir>/{STARTED, audio/<id>.wav, result.csv, log.txt, DONE}
  - On a fresh run (no output/STARTED): deletes contents INSIDE output/ (never output/ itself),
    initializes output/audio, output/log.txt, output/result.csv (UTF-8 no BOM, LF), then creates STARTED.
  - On a resumed run (output/STARTED exists): preserves existing files, reads completed IDs from
    output/result.csv (enforcing manifest order), appends to output/log.txt, and resumes from the first missing ID.
  - Loads and warms up all required models ONCE using a single shared Engine in a persistent worker subprocess.
  - Enforces a strict 60-second per-sentence timeout; hung/crashed workers are terminated and respawned
    so the evaluation run never hangs.
  - Normalizes output WAV audio (PCM 16-bit, mono, >= 16 kHz, peak <= -1 dBFS, target ~ -23 LUFS).
  - Flushes + fsyncs result.csv after every sentence and writes DONE on completion.
"""
import csv
import datetime
import hashlib
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import wave
import numpy as np

DIR_MAP = {
    "vi2en": ("vi", "en"),
    "en2vi": ("en", "vi"),
    "vi2zh": ("vi", "zh"),
    "zh2vi": ("zh", "vi"),
    "vi2ko": ("vi", "ko"),
    "ko2vi": ("ko", "vi"),
}

CSV_HEADER = ["id", "status", "output_text", "message", "duration_ms"]
PEAK_CEILING_DBFS = -1.0
PEAK_CEILING = float(10.0 ** (PEAK_CEILING_DBFS / 20.0))  # ~0.8912509
TARGET_LUFS_DB = -23.0
MAX_GAIN_DB = 24.0
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_TEAM = "GiaSuDinhCao"


def _ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sanitize_cell(text, max_len=None):
    """Ensure a CSV cell has no CR, LF, or TAB characters and optional max length."""
    if text is None:
        return ""
    s = re.sub(r"[\r\n\t]+", " ", str(text)).strip()
    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..." if max_len > 3 else s[:max_len]
    return s


def _append_log(log_path, sentence_id, msg):
    line = f"[{_ts()}] [{sentence_id}] {sanitize_cell(msg)}\n"
    with open(log_path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line)
        f.flush()


def _write_csv_header(csv_path):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, lineterminator="\n")
        w.writerow(CSV_HEADER)
        f.flush()
        os.fsync(f.fileno())


def _append_csv_row(csv_path, row_id, status, output_text, message, duration_ms):
    row = [
        sanitize_cell(row_id),
        sanitize_cell(status),
        sanitize_cell(output_text),
        sanitize_cell(message, max_len=200),
        int(max(0, round(duration_ms))),
    ]
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, lineterminator="\n")
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _read_manifest(manifest_path):
    """Read input/manifest.csv -> list of (id, audio_rel, direction), tolerating header spaces & trailing blank lines."""
    rows = []
    seen = set()
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        raw_reader = csv.reader(f)
        raw_header = next(raw_reader, None)
        if not raw_header:
            raise ValueError("manifest.csv is empty or missing header")
        fields = [c.strip() for c in raw_header]
        for req in ("id", "audio", "direction"):
            if req not in fields:
                raise ValueError(f"manifest.csv missing required column '{req}' (found: {fields})")
        idx_id = fields.index("id")
        idx_aud = fields.index("audio")
        idx_dir = fields.index("direction")
        for line_no, raw_row in enumerate(raw_reader, start=2):
            if not raw_row or all(not c.strip() for c in raw_row):
                continue
            rid = raw_row[idx_id].strip() if idx_id < len(raw_row) else ""
            aud = raw_row[idx_aud].strip() if idx_aud < len(raw_row) else ""
            d = raw_row[idx_dir].strip() if idx_dir < len(raw_row) else ""
            if not rid and not aud and not d:
                continue
            if not rid:
                raise ValueError(f"manifest.csv row {line_no} has empty id")
            if rid in seen:
                raise ValueError(f"manifest.csv row {line_no} has duplicate id '{rid}'")
            seen.add(rid)
            rows.append((rid, aud, d))
    return rows


def _clear_dir_contents(dir_path):
    """Delete all contents INSIDE dir_path, never deleting dir_path itself."""
    for name in os.listdir(dir_path):
        p = os.path.join(dir_path, name)
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p)
        else:
            os.unlink(p)


def _load_existing_results(csv_path, manifest_ids_ordered):
    """Read already-completed contiguous prefix (or valid rows) from result.csv on resume.

    To strictly satisfy Rule 7 ('The number of data rows in result.csv must match the manifest,
    with id values in the same order'), only the contiguous prefix matching manifest order is kept
    before appending new rows, or if all completed rows are kept, any out-of-order/corrupt trailing
    rows are repaired so subsequent appends maintain exact manifest order.
    Returns (completed_dict, prior_duration_s).
    """
    valid_index = {rid: idx for idx, rid in enumerate(manifest_ids_ordered)}
    if not os.path.isfile(csv_path):
        _write_csv_header(csv_path)
        return {}, 0.0

    parsed = {}
    dirty = False
    file_order = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CSV_HEADER:
            dirty = True
        else:
            for row in reader:
                if len(row) != len(CSV_HEADER):
                    dirty = True
                    continue
                rid, st, out_t, msg, dur = row
                if rid not in valid_index or st not in ("ok", "empty", "error", "timeout", "unsupported"):
                    dirty = True
                    continue
                try:
                    dur_int = int(dur)
                except ValueError:
                    dirty = True
                    continue
                if rid in parsed:
                    dirty = True
                    continue
                clean_row = [sanitize_cell(rid), st, sanitize_cell(out_t), sanitize_cell(msg, max_len=200), str(max(0, dur_int))]
                if clean_row != row:
                    dirty = True
                parsed[rid] = (st, clean_row, max(0, dur_int))
                file_order.append(rid)

    # Keep only contiguous prefix from index 0 so appending remaining rows guarantees exact manifest order!
    # (Unless ALL manifest rows are already in parsed, in which case sort them into exact manifest order.)
    if len(parsed) == len(manifest_ids_ordered):
        ordered_ids = list(manifest_ids_ordered)
        if file_order != ordered_ids:
            dirty = True
    else:
        ordered_ids = []
        for rid in manifest_ids_ordered:
            if rid in parsed:
                ordered_ids.append(rid)
            else:
                break
        if len(ordered_ids) != len(parsed) or file_order != ordered_ids:
            dirty = True

    completed = {}
    kept_rows = []
    prior_dur_ms = 0
    for rid in ordered_ids:
        st, clean_row, dur_int = parsed[rid]
        completed[rid] = st
        kept_rows.append(clean_row)
        prior_dur_ms += dur_int

    if dirty:
        tmp = csv_path + ".repair.tmp"
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, doublequote=True, lineterminator="\n")
            w.writerow(CSV_HEADER)
            w.writerows(kept_rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, csv_path)
    return completed, prior_dur_ms / 1000.0


def _acquire_run_lock(output_dir):
    """Process-level lock outside output_dir so duplicate runs cannot corrupt output/."""
    key = hashlib.sha256(os.path.abspath(output_dir).encode("utf-8")).hexdigest()[:16]
    lock_path = os.path.join(tempfile.gettempdir(), f"ovac_eval_{key}.lock")
    fh = open(lock_path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise RuntimeError(f"Another eval_mode instance is already running for output directory: {output_dir}")
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh, lock_path


def _release_run_lock(fh, lock_path):
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        fh.close()
    except OSError:
        pass
    try:
        os.unlink(lock_path)
    except OSError:
        pass


def load_audio_any(path, sr=16000):
    """Read input WAV (or fallback via ffmpeg if encoded unusually) as 16 kHz mono float32."""
    from .audio import load_audio
    try:
        return load_audio(path, sr=sr)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise
        p = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True,
            check=True,
        )
        if not p.stdout:
            raise ValueError("ffmpeg decoded 0 bytes from input audio")
        return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def normalize_output_audio(wav, sr):
    """Normalize translated TTS audio: target ~ -23 LUFS (active speech level), peak strictly <= -1 dBFS, no clipping."""
    from .audio import active_rms
    x = np.asarray(wav, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros(0, dtype=np.float32), int(sr)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - float(np.mean(x))
    peak = float(np.max(np.abs(x)))
    if peak < 1e-6:
        return np.zeros(0, dtype=np.float32), int(sr)
    sr = int(sr)
    if sr < 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
        peak = float(np.max(np.abs(x)))
        if peak < 1e-6:
            return np.zeros(0, dtype=np.float32), sr
    rms = active_rms(x, sr=sr)
    target_rms = float(10.0 ** (TARGET_LUFS_DB / 20.0))
    max_gain = float(10.0 ** (MAX_GAIN_DB / 20.0))
    if np.isfinite(rms) and rms > 1e-7:
        gain = min(target_rms / rms, max_gain)
    else:
        gain = 1.0
    if peak * gain > PEAK_CEILING:
        gain = PEAK_CEILING / peak
    out = np.clip(x * gain, -PEAK_CEILING, PEAK_CEILING).astype(np.float32)
    return out, sr


def save_wav_atomic(path, wav, sr):
    """Write PCM 16-bit mono WAV atomically with fsync and verify non-empty audio."""
    tmp_path = path + ".tmp"
    pcm = np.round(np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
    if pcm.size == 0:
        raise ValueError("Refusing to write empty WAV")
    try:
        with open(tmp_path, "wb") as raw_f:
            with wave.open(raw_f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sr))
                wf.writeframes(pcm.tobytes())
            raw_f.flush()
            os.fsync(raw_f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    with wave.open(path, "rb") as wf:
        if wf.getnframes() <= 0 or wf.getframerate() < 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Written WAV failed format verification")


def _safe_remove(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _process_one_sentence(translator, audio_path, tgt, out_wav_path):
    """Execute one sentence: start timer before reading input WAV, stop after writing output WAV."""
    from .audio import concat_results
    t0 = time.perf_counter()
    try:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Input audio not found: {audio_path}")
        wav_in = load_audio_any(audio_path, sr=16000)
        if wav_in.size == 0:
            dur_ms = int(round((time.perf_counter() - t0) * 1000))
            _safe_remove(out_wav_path)
            return ("empty", "", "Input audio is empty", dur_ms, "")
        res = translator.translate(wav_in)
        # Fallback for ultra-short clips (< 0.3s) that contain speech energy but are shorter than VAD min_s=0.3s
        if not res and 0 < wav_in.size < 16000 and float(np.percentile(np.abs(wav_in), 95)) >= 3e-5:
            if hasattr(translator, "translate_segment"):
                seg_r = translator.translate_segment(wav_in)
                if seg_r and (seg_r.get("mt") or len(seg_r.get("wav", ())) > 0):
                    res = [seg_r]
        sep = "" if tgt == "zh" else " "
        mt_text = sanitize_cell(sep.join(r.get("mt", "") for r in res if r.get("mt")))
        wav_out, sr_out = concat_results(res)
        wav_norm, sr_norm = normalize_output_audio(wav_out, sr_out)
        if wav_norm.size == 0:
            dur_ms = int(round((time.perf_counter() - t0) * 1000))
            _safe_remove(out_wav_path)
            return ("empty", mt_text, "Pipeline produced no output speech", dur_ms, "")
        save_wav_atomic(out_wav_path, wav_norm, sr_norm)
        dur_ms = int(round((time.perf_counter() - t0) * 1000))
        return ("ok", mt_text, "", dur_ms, "")
    except Exception as e:
        dur_ms = int(round((time.perf_counter() - t0) * 1000))
        _safe_remove(out_wav_path)
        tb = traceback.format_exc()
        err_msg = sanitize_cell(f"{type(e).__name__}: {e}", max_len=200)
        return ("error", "", err_msg, dur_ms, tb)


class _SharedEngineTranslator:
    """Thin direction-bound wrapper sharing one Engine instance across all directions (saves host RAM)."""

    def __init__(self, registry, engine, src, tgt):
        from .registry import resolve_runtime_profile
        self.r = registry
        self.e = engine
        self.src = src
        self.tgt = tgt
        self.vad_profile = resolve_runtime_profile(vad_profile=getattr(registry, "vad_profile", None))["vad_profile"]
        self.asr = self.e.asr(src)
        self.mt = self.e.mt
        self.tts = self.e.tts(tgt)

    def translate_segment(self, wav, breaks=None):
        return self.e.translate_segment(wav, self.src, self.tgt, breaks=breaks)

    def translate(self, wav):
        from .audio import SR, SEG_MAX_S, LEGACY_PROFILES, segment_with_breaks
        cap = 12.0 if self.vad_profile in LEGACY_PROFILES else SEG_MAX_S
        max_seg = min(getattr(self.asr, "max_s", cap), cap)
        out = []
        for (a, b), brk in segment_with_breaks(wav, max_s=max_seg, vad_profile=self.vad_profile):
            r = self.translate_segment(wav[a:b], breaks=brk)
            r["span"] = (a / SR, b / SR)
            out.append(r)
        return out


def _worker_main(conn, cfg):
    """Persistent child process: loads all required models ONCE via a single shared Engine, then serves sentences."""
    try:
        from .registry import Registry
        from .conversation import Engine

        t_load = time.perf_counter()
        reg = Registry(
            cfg["models"],
            cfg["backend"],
            cfg.get("glossary"),
            ort_policy=cfg.get("ort_policy"),
            vad_profile=cfg.get("vad_profile"),
            source_mode=cfg.get("source_mode"),
            tts_format=cfg.get("tts_format"),
        )
        dirs = cfg["directions"]
        if dirs:
            reg.warmup(dirs, log=lambda m: print(f"[eval-warmup] {m}", flush=True))
        eng = Engine(reg, denoise=cfg.get("denoise", False))
        translators = {}
        for src, tgt in dirs:
            translators[(src, tgt)] = _SharedEngineTranslator(reg, eng, src, tgt)
        conn.send(("ready", round(time.perf_counter() - t_load, 2)))
    except Exception as e:
        conn.send(("init_error", f"{type(e).__name__}: {e}", traceback.format_exc()))
        return

    while True:
        try:
            task = conn.recv()
        except EOFError:
            break
        if task is None:
            break
        rid, audio_path, src, tgt, out_wav_path = task
        tr = translators.get((src, tgt))
        if tr is None:
            tr = _SharedEngineTranslator(reg, eng, src, tgt)
            translators[(src, tgt)] = tr
        result = _process_one_sentence(tr, audio_path, tgt, out_wav_path)
        conn.send(("result", rid, result))


class _WorkerPool:
    """Manages a single persistent inference worker with hard timeout kill + respawn."""

    def __init__(self, cfg, log_path, init_timeout_s=600.0):
        self.cfg = cfg
        self.log_path = log_path
        self.init_timeout_s = init_timeout_s
        self.proc = None
        self.conn = None

    def start(self):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child_conn, self.cfg), daemon=True)
        self.proc.start()
        child_conn.close()
        self.conn = parent_conn
        _append_log(self.log_path, "SYSTEM", f"Starting inference worker pid={self.proc.pid} (loading models once)")
        if not self.conn.poll(self.init_timeout_s):
            self.kill()
            raise RuntimeError(f"Worker failed to load models within {self.init_timeout_s:.0f}s")
        msg = self.conn.recv()
        if msg[0] == "init_error":
            self.kill()
            _append_log(self.log_path, "SYSTEM", f"Worker init error: {msg[1]} | {msg[2]}")
            raise RuntimeError(f"Worker model load failed: {msg[1]}")
        _append_log(self.log_path, "SYSTEM", f"Worker ready in {msg[1]}s (pid={self.proc.pid})")

    def run_sentence(self, rid, audio_path, src, tgt, out_wav_path, timeout_s):
        if self.proc is None or not self.proc.is_alive():
            self.start()
        t_start = time.perf_counter()
        self.conn.send((rid, audio_path, src, tgt, out_wav_path))
        if self.conn.poll(timeout_s):
            try:
                tag, got_id, result = self.conn.recv()
                if tag == "result" and got_id == rid:
                    return result
            except EOFError:
                pass
            dur_ms = int(round((time.perf_counter() - t_start) * 1000))
            self.kill()
            _safe_remove(out_wav_path)
            return ("error", "", "Inference worker terminated unexpectedly", dur_ms, "")
        dur_ms = int(round((time.perf_counter() - t_start) * 1000))
        self.kill()
        _safe_remove(out_wav_path)
        return ("timeout", "", f"Sentence processing exceeded {int(timeout_s)}s timeout", dur_ms, "")

    def kill(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass
            self.conn = None
        if self.proc is not None:
            try:
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=2.0)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join(timeout=2.0)
            except OSError:
                pass
            self.proc = None

    def close(self):
        if self.conn is not None and self.proc is not None and self.proc.is_alive():
            try:
                self.conn.send(None)
                self.proc.join(timeout=5.0)
            except OSError:
                pass
        self.kill()


def run_eval(
    input_dir,
    output_dir,
    models_path,
    backend="npu",
    team_name=DEFAULT_TEAM,
    timeout_s=DEFAULT_TIMEOUT_S,
    glossary=None,
    denoise=False,
    ort_policy=None,
    vad_profile=None,
    source_mode=None,
    tts_format=None,
    _worker_override=None,
):
    """Run OVAC Phase 3 eval mode on input_dir -> output_dir. Returns 0 on completion."""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    manifest_path = os.path.join(input_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.csv not found at {manifest_path}")

    os.makedirs(output_dir, exist_ok=True)
    lock_fh, lock_path = _acquire_run_lock(output_dir)
    t_run_start = time.perf_counter()
    worker = None

    try:
        rows = _read_manifest(manifest_path)
        manifest_ids = [rid for rid, _, _ in rows]

        started_path = os.path.join(output_dir, "STARTED")
        audio_out_dir = os.path.join(output_dir, "audio")
        csv_path = os.path.join(output_dir, "result.csv")
        log_path = os.path.join(output_dir, "log.txt")
        done_path = os.path.join(output_dir, "DONE")

        resuming = os.path.exists(started_path)
        prior_elapsed_s = 0.0
        if not resuming:
            _clear_dir_contents(output_dir)
            os.makedirs(audio_out_dir, exist_ok=True)
            with open(log_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"[{_ts()}] [SYSTEM] New eval run started: input={input_dir} output={output_dir} total={len(rows)}\n")
                f.flush()
            _write_csv_header(csv_path)
            with open(started_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"started={_ts()} pid={os.getpid()}\n")
                f.flush()
                os.fsync(f.fileno())
            completed = {}
        else:
            os.makedirs(audio_out_dir, exist_ok=True)
            completed, prior_elapsed_s = _load_existing_results(csv_path, manifest_ids)
            _append_log(log_path, "SYSTEM", f"Resuming eval run: {len(completed)}/{len(rows)} rows already completed")
            if len(completed) == len(rows) and os.path.isfile(done_path):
                _append_log(log_path, "SYSTEM", "All manifest rows already completed and DONE exists; preserving existing output")
                print(f"DONE (preserved): {open(done_path, encoding='utf-8').read().strip()}", flush=True)
                return 0

        pending = [(rid, aud, d) for rid, aud, d in rows if rid not in completed]
        needed_dirs = sorted({DIR_MAP[d] for _, _, d in pending if d in DIR_MAP})

        if pending and needed_dirs:
            if _worker_override is not None:
                worker = _worker_override
            else:
                cfg = dict(
                    models=os.path.abspath(models_path),
                    backend=backend,
                    directions=needed_dirs,
                    glossary=glossary,
                    denoise=denoise,
                    ort_policy=ort_policy,
                    vad_profile=vad_profile,
                    source_mode=source_mode,
                    tts_format=tts_format,
                )
                worker = _WorkerPool(cfg, log_path)
                worker.start()

        total = len(rows)
        done_count = len(completed)
        for rid, audio_rel, direction in rows:
            if rid in completed:
                continue
            done_count += 1
            out_wav_path = os.path.join(audio_out_dir, f"{rid}.wav")
            _safe_remove(out_wav_path)

            if direction not in DIR_MAP:
                status = "unsupported"
                out_text = ""
                message = f"Unsupported direction: {direction}"
                dur_ms = 0
                tb = ""
                _append_log(log_path, rid, f"unsupported direction={direction}")
            else:
                src, tgt = DIR_MAP[direction]
                audio_abs = os.path.normpath(os.path.join(input_dir, audio_rel))
                _append_log(log_path, rid, f"start dir={direction} audio={audio_rel}")
                status, out_text, message, dur_ms, tb = worker.run_sentence(
                    rid, audio_abs, src, tgt, out_wav_path, timeout_s
                )
                if status != "ok":
                    _safe_remove(out_wav_path)
                log_detail = f"done status={status} duration_ms={dur_ms}"
                if message:
                    log_detail += f" msg={message}"
                _append_log(log_path, rid, log_detail)
                if tb:
                    for tb_line in tb.strip().splitlines():
                        _append_log(log_path, rid, f"TRACE: {tb_line}")

            _append_csv_row(csv_path, rid, status, out_text, message, dur_ms)
            completed[rid] = status
            print(f"[{done_count}/{total}] {rid} ({direction}) -> {status} ({dur_ms} ms)", flush=True)

        ok_count = sum(1 for st in completed.values() if st == "ok")
        elapsed_s = int(round(prior_elapsed_s + (time.perf_counter() - t_run_start)))
        team_token = re.sub(r"\s+", "", str(team_name or DEFAULT_TEAM))
        done_line = f"team={team_token} total={total} ok={ok_count} elapsed_s={elapsed_s}\n"
        with open(done_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(done_line)
            f.flush()
            os.fsync(f.fileno())
        _append_log(log_path, "SYSTEM", f"Finished: {done_line.strip()}")
        print(f"DONE: {done_line.strip()}", flush=True)
        return 0
    finally:
        if worker is not None and _worker_override is None:
            worker.close()
        _release_run_lock(lock_fh, lock_path)

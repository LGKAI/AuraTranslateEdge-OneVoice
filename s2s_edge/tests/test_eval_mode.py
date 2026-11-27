# -*- coding: utf-8 -*-
"""Unit & integration tests for OVAC Phase 3 Eval Mode (s2s_edge/eval_mode.py).

Verifies every item in the official Phase 3 Eval Mode Packaging Guide checklist:
  - Fresh run clears contents inside output/ without deleting output/ itself
  - STARTED, audio/<id>.wav, result.csv, log.txt, DONE creation
  - CSV rules: UTF-8 without BOM, LF endings, no CR/LF/TAB in cells, message <= 200 chars, exact header & order
  - Tolerates header spaces and trailing blank lines in manifest.csv
  - Status handling: ok, empty, error (corrupt WAV), timeout (real subprocess watchdog kill + respawn), unsupported
  - Short clip (< 0.3s) speech fallback via translate_segment
  - Non-ok rows write NO WAV file
  - Output WAV format: PCM 16-bit, mono, >= 16 kHz, peak <= -1 dBFS, no clipping, quiet audio boosted
  - Resume from STARTED preserves prior results, enforces exact manifest row order, preserves already-DONE runs
  - Paths containing spaces
"""
import csv
import multiprocessing as mp
import os
import pathlib
import re
import sys
import tempfile
import time
import wave
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from s2s_edge.eval_mode import (
    CSV_HEADER,
    PEAK_CEILING,
    _WorkerPool,
    _process_one_sentence,
    normalize_output_audio,
    run_eval,
)


def _make_wav(path, sr=16000, sec=0.5, amp=0.3, freq=440.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = np.arange(int(sr * sec), dtype=np.float32) / sr
    x = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    pcm = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


class _FakeTranslator:
    def __init__(self, behavior):
        self.behavior = behavior

    def translate(self, wav):
        mode = self.behavior(wav)
        if mode == "vad_miss_short":
            return []
        return [self.translate_segment(wav)]

    def translate_segment(self, wav):
        mode = self.behavior(wav)
        if mode in ("ok_loud", "vad_miss_short"):
            t = np.arange(22050, dtype=np.float32) / 22050.0
            w = (1.8 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
            return {"mt": "Hello,\n\"world\"\tline2", "wav": w, "sr": 22050}
        if mode == "ok_quiet":
            t = np.arange(44100, dtype=np.float32) / 44100.0
            w = (0.005 * np.sin(2.0 * np.pi * 250.0 * t)).astype(np.float32)
            return {"mt": "Xin chao", "wav": w, "sr": 44100}
        if mode == "empty":
            return {"mt": "", "wav": np.zeros(0, np.float32), "sr": 22050}
        raise RuntimeError("Boom\nwith\ttab\nand " + "x" * 260)


class _InProcessFakeWorker:
    def __init__(self, behaviors):
        self.behaviors = behaviors

    def run_sentence(self, rid, audio_path, src, tgt, out_wav_path, timeout_s):
        b = self.behaviors.get(rid, "ok_quiet")
        tr = _FakeTranslator(lambda _wav: b)
        return _process_one_sentence(tr, audio_path, tgt, out_wav_path)


def test_normalize_output_audio_bounds_and_boost():
    sr = 22050
    t = np.arange(sr, dtype=np.float32) / sr
    hot = (2.0 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    out, out_sr = normalize_output_audio(hot, sr)
    assert out_sr == sr
    assert float(np.max(np.abs(out))) <= PEAK_CEILING + 1e-6

    quiet = (0.004 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    q_out, q_sr = normalize_output_audio(quiet, sr)
    assert q_sr == sr
    assert float(np.max(np.abs(q_out))) > 0.05
    assert float(np.max(np.abs(q_out))) <= PEAK_CEILING + 1e-6

    sil_out, _ = normalize_output_audio(np.zeros(sr, dtype=np.float32), sr)
    assert sil_out.size == 0


def test_eval_fresh_run_statuses_short_clip_csv_and_spaces_in_paths(tmp_path):
    in_dir = tmp_path / "input with spaces"
    out_dir = tmp_path / "output with spaces"
    out_dir.mkdir()
    stale = out_dir / "stale_file.txt"
    stale.write_text("old", encoding="utf-8")
    out_inode = os.stat(out_dir).st_ino

    # Ultra-short 0.2s clip (< 0.3s VAD min_s) -> rescued via translate_segment fallback
    _make_wav(str(in_dir / "audio" / "T0001.wav"), sec=0.2)
    _make_wav(str(in_dir / "audio" / "T0002.wav"))
    _make_wav(str(in_dir / "audio" / "T0003.wav"))
    (in_dir / "audio" / "T0004.wav").write_bytes(b"not a valid wav file")
    _make_wav(str(in_dir / "audio" / "T0005.wav"))

    # Include header spaces and trailing blank lines
    manifest = in_dir / "manifest.csv"
    manifest.write_text(
        "id, audio, direction\n"
        "T0001,audio/T0001.wav,vi2en\n"
        "T0002,audio/T0002.wav,en2vi\n"
        "T0003,audio/T0003.wav,vi2ko\n"
        "T0004,audio/T0004.wav,ko2vi\n"
        "T0005,audio/T0005.wav,fr2en\n\n   \n",
        encoding="utf-8",
    )

    worker = _InProcessFakeWorker({"T0001": "vad_miss_short", "T0002": "ok_quiet", "T0003": "empty"})
    rc = run_eval(str(in_dir), str(out_dir), models_path="dummy.json", team_name="GiaSuDinhCao", _worker_override=worker)
    assert rc == 0

    assert os.stat(out_dir).st_ino == out_inode
    assert not stale.exists()
    assert (out_dir / "STARTED").is_file()
    assert (out_dir / "log.txt").is_file()
    assert (out_dir / "DONE").is_file()

    raw_csv = (out_dir / "result.csv").read_bytes()
    assert not raw_csv.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw_csv

    rows = list(csv.reader(raw_csv.decode("utf-8").splitlines()))
    assert rows[0] == CSV_HEADER
    assert [r[0] for r in rows[1:]] == ["T0001", "T0002", "T0003", "T0004", "T0005"]
    assert [r[1] for r in rows[1:]] == ["ok", "ok", "empty", "error", "unsupported"]

    for r in rows[1:]:
        for cell in r:
            assert "\n" not in cell and "\r" not in cell and "\t" not in cell
        assert len(r[3]) <= 200

    for ok_id in ("T0001", "T0002"):
        wp = out_dir / "audio" / f"{ok_id}.wav"
        assert wp.is_file()
        with wave.open(str(wp), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() >= 16000
            assert wf.getnframes() > 0
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
            assert float(np.max(np.abs(pcm))) <= PEAK_CEILING + 1e-3

    for bad_id in ("T0003", "T0004", "T0005"):
        assert not (out_dir / "audio" / f"{bad_id}.wav").exists()

    done_txt = (out_dir / "DONE").read_text(encoding="utf-8")
    assert len(done_txt.splitlines()) == 1
    assert re.match(r"^team=GiaSuDinhCao total=5 ok=2 elapsed_s=\d+$", done_txt.strip())


def test_eval_resume_after_interruption_and_preserve_done(tmp_path):
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    for i in range(1, 4):
        _make_wav(str(in_dir / "audio" / f"T000{i}.wav"))
    (in_dir / "manifest.csv").write_text(
        "id,audio,direction\n"
        "T0001,audio/T0001.wav,vi2en\n"
        "T0002,audio/T0002.wav,en2vi\n"
        "T0003,audio/T0003.wav,vi2zh\n",
        encoding="utf-8",
    )

    (out_dir / "audio").mkdir(parents=True)
    (out_dir / "STARTED").write_text("started=previous\n", encoding="utf-8")
    (out_dir / "log.txt").write_text("[prev] [T0001] prior log\n", encoding="utf-8")
    _make_wav(str(out_dir / "audio" / "T0001.wav"), amp=0.2)
    orig_t1_bytes = (out_dir / "audio" / "T0001.wav").read_bytes()
    (out_dir / "result.csv").write_text(
        "id,status,output_text,message,duration_ms\n"
        "T0001,ok,Previous translation,,2000\n"
        "T0002,ok,Truncated line without duration",
        encoding="utf-8",
    )

    worker = _InProcessFakeWorker({"T0002": "ok_quiet", "T0003": "ok_loud"})
    rc = run_eval(str(in_dir), str(out_dir), models_path="dummy.json", _worker_override=worker)
    assert rc == 0

    assert (out_dir / "audio" / "T0001.wav").read_bytes() == orig_t1_bytes
    log_txt = (out_dir / "log.txt").read_text(encoding="utf-8")
    assert "[prev] [T0001] prior log" in log_txt
    assert "Resuming eval run: 1/3 rows already completed" in log_txt

    rows = list(csv.reader((out_dir / "result.csv").read_text(encoding="utf-8").splitlines()))
    assert [r[0] for r in rows[1:]] == ["T0001", "T0002", "T0003"]
    assert rows[1][2] == "Previous translation"
    assert [r[1] for r in rows[1:]] == ["ok", "ok", "ok"]
    done_before = (out_dir / "DONE").read_text(encoding="utf-8")
    assert re.match(r"^team=GiaSuDinhCao total=3 ok=3 elapsed_s=\d+$", done_before.strip())

    # Re-running on an already-completed output directory with STARTED + DONE preserves DONE untouched
    rc2 = run_eval(str(in_dir), str(out_dir), models_path="dummy.json", _worker_override=worker)
    assert rc2 == 0
    assert (out_dir / "DONE").read_text(encoding="utf-8") == done_before


def _sleep_worker_main(conn, _cfg):
    conn.send(("ready", 0.01))
    while True:
        try:
            task = conn.recv()
        except EOFError:
            break
        if task is None:
            break
        rid, audio_path, src, tgt, out_wav_path = task
        if rid == "SLOW":
            time.sleep(5.0)
            conn.send(("result", rid, ("ok", "late", "", 5000, "")))
        else:
            _make_wav(out_wav_path, sr=22050, sec=0.2, amp=0.2)
            conn.send(("result", rid, ("ok", "fast ok", "", 40, "")))


class _TimeoutTestWorkerPool(_WorkerPool):
    def start(self):
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self.proc = ctx.Process(target=_sleep_worker_main, args=(child_conn, self.cfg), daemon=True)
        self.proc.start()
        child_conn.close()
        self.conn = parent_conn
        assert self.conn.poll(10.0)
        assert self.conn.recv()[0] == "ready"


def test_eval_timeout_kills_and_respawns_worker(tmp_path):
    in_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    _make_wav(str(in_dir / "audio" / "SLOW.wav"))
    _make_wav(str(in_dir / "audio" / "FAST.wav"))
    (in_dir / "manifest.csv").write_text(
        "id,audio,direction\n"
        "SLOW,audio/SLOW.wav,vi2en\n"
        "FAST,audio/FAST.wav,en2vi\n",
        encoding="utf-8",
    )

    pool = _TimeoutTestWorkerPool({}, str(out_dir / "log.txt"))
    try:
        rc = run_eval(str(in_dir), str(out_dir), models_path="dummy.json", timeout_s=1.0, _worker_override=pool)
    finally:
        pool.close()

    assert rc == 0
    rows = list(csv.reader((out_dir / "result.csv").read_text(encoding="utf-8").splitlines()))
    assert [r[0] for r in rows[1:]] == ["SLOW", "FAST"]
    assert [r[1] for r in rows[1:]] == ["timeout", "ok"]
    assert not (out_dir / "audio" / "SLOW.wav").exists()
    assert (out_dir / "audio" / "FAST.wav").is_file()
    assert re.match(r"^team=GiaSuDinhCao total=2 ok=1 elapsed_s=\d+$", (out_dir / "DONE").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    test_normalize_output_audio_bounds_and_boost()
    print("PASS: test_normalize_output_audio_bounds_and_boost")
    test_eval_fresh_run_statuses_short_clip_csv_and_spaces_in_paths(pathlib.Path(tempfile.mkdtemp()))
    print("PASS: test_eval_fresh_run_statuses_short_clip_csv_and_spaces_in_paths")
    test_eval_resume_after_interruption_and_preserve_done(pathlib.Path(tempfile.mkdtemp()))
    print("PASS: test_eval_resume_after_interruption_and_preserve_done")
    test_eval_timeout_kills_and_respawns_worker(pathlib.Path(tempfile.mkdtemp()))
    print("PASS: test_eval_timeout_kills_and_respawns_worker")
    print("ALL EVAL MODE TESTS PASSED")

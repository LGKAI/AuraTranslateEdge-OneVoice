# -*- coding: utf-8 -*-
"""Push-to-talk on the LAPTOP, translation on the BOX (the AIBOX 8550 has no mic / speaker).
Laptop mic -> WAV -> box NPU (ASR -> MT -> TTS) -> WAV -> laptop speaker. Everything goes over the USB-C cable (adb), no network.

Mode "serve" (default, fast: models stay loaded on the box)
  box   :  cd /root/onevoice/s2s_edge_pkg && bash tools/box_serve.sh vi,ko          (keep this adb shell open)
  laptop:  python tools/laptop_ptt.py                                             (runs `adb forward tcp:8770 tcp:8770` itself)
Mode "file" (fallback, no server; loads the models for every utterance, so several seconds slower)
  laptop:  python tools/laptop_ptt.py --mode file --pair vi,ko
Other:   --wav some.wav    send an existing file instead of recording (also works without a microphone)
         --device N        input device index (python -m sounddevice lists them)
Needs on the laptop: pip install sounddevice soundfile numpy ; adb in PATH.
Timing printed: what the BOX measured per stage (the number to report) and the laptop<->box round trip (adb transfer, reported separately)."""
import argparse, base64, io, json, os, shutil, subprocess, sys, time, urllib.request
ADB = shutil.which("adb") or r"C:\platform-tools\adb.exe"
import numpy as np

PKG = "/root/onevoice/s2s_edge_pkg"


def adb(*a, check=True):
    return subprocess.run([ADB, *a], check=check, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def record(sr, device):
    import sounddevice as sd
    buf = []
    st = sd.InputStream(samplerate=sr, channels=1, dtype="int16", device=device, callback=lambda d, *_: buf.append(d.copy()))
    input("[Enter] bắt đầu nói / start speaking ... "); st.start()
    input("   đang ghi / recording ... [Enter] để dịch / to translate "); st.stop(); st.close()
    x = np.concatenate(buf)[:, 0] if buf else np.zeros(0, np.int16)
    if len(x) and np.abs(x).max() < 300: print("   (mic gần như im lặng: kiểm tra quyền Microphone / thiết bị mặc định)")
    return x


def wav_bytes(x, sr):
    import wave
    b = io.BytesIO()
    with wave.open(b, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.astype(np.int16).tobytes())
    return b.getvalue()


def play(data):
    import sounddevice as sd, soundfile as sf
    if not data: print("   (không có âm thanh đầu ra)"); return
    y, sr = sf.read(io.BytesIO(data)); sd.play(y, sr); sd.wait()


def show(segs, box_s, rt):
    for s in segs:
        t = s["timing"]
        print(f"   {s['src']}->{s['tgt']}: {s['asr']}\n      -> {s['mt']}\n      box: ASR {t['asr_s'] * 1000:.0f} | MT {t['mt_s'] * 1000:.0f} | TTS {t['tts_s'] * 1000:.0f} | total {t['total_s'] * 1000:.0f} ms")
    if not segs: print("   (xin nói lại / please repeat)")
    print(f"   box total {box_s * 1000:.0f} ms | laptop<->box round trip {rt * 1000:.0f} ms (includes USB transfer)")


def via_server(data, port):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/translate", data=data, headers={"Content-Type": "audio/wav"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read().decode("utf-8"))
    if "error" in r: raise RuntimeError(r["error"])
    return r["segments"], r["box_s"], base64.b64decode(r["wav_b64"])


def via_file(data, pair, models):
    open("ptt_in.wav", "wb").write(data); adb("push", "ptt_in.wav", "/tmp/ptt_in.wav")
    src = pair.split(",")
    cmd = (f"cd {PKG} && . .venv/bin/activate && . tools/env.sh && python -m s2s_edge file --pair {pair} --in /tmp/ptt_in.wav --out /tmp/ptt_out.wav"
           + (f" --models {models}" if models else " --models $(cat logs/models_ok.txt 2>/dev/null || echo models_qcs8550/models.json)") + " --backend npu")
    t0 = time.time(); out = adb("shell", cmd, check=False); box = time.time() - t0
    print(out.strip()); adb("pull", "/tmp/ptt_out.wav", "ptt_out.wav", check=False)
    data = open("ptt_out.wav", "rb").read() if os.path.exists("ptt_out.wav") else b""
    return None, box, data


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="serve", choices=["serve", "file"]); ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--pair", default="vi,ko", help="file mode only (serve mode uses the pair the box server was started with)")
    ap.add_argument("--models", default=""); ap.add_argument("--wav", default=""); ap.add_argument("--device", type=int, default=None); ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--no-play", action="store_true"); ap.add_argument("--save", default="ptt_logs", help="folder for in/out WAV + JSON of every turn")
    a = ap.parse_args(); sys.stdout.reconfigure(encoding="utf-8"); os.makedirs(a.save, exist_ok=True)
    if a.mode == "serve":
        try: adb("forward", f"tcp:{a.port}", f"tcp:{a.port}")
        except Exception as e: print(f"adb forward failed ({e!r}); trying 127.0.0.1:{a.port} anyway (server on this machine?)")
        try: print("box server:", urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=10).read().decode())
        except Exception as e: sys.exit(f"box server not reachable on port {a.port} ({e!r}). Start it on the box: bash tools/box_serve.sh {a.pair}")
    turn = 0
    while True:
        if a.wav:
            import soundfile as sf
            y, sr = sf.read(a.wav, dtype="int16"); y = y if y.ndim == 1 else y[:, 0]; data = wav_bytes(y, sr)
        else:
            x = record(a.sr, a.device)
            if len(x) < a.sr * 0.3: print("   (quá ngắn / too short)"); continue
            data = wav_bytes(x, a.sr)
        turn += 1; stamp = time.strftime("%H%M%S"); open(os.path.join(a.save, f"{stamp}_{turn:03d}_in.wav"), "wb").write(data)
        t0 = time.time()
        try:
            segs, box, out = via_server(data, a.port) if a.mode == "serve" else via_file(data, a.pair, a.models)
        except Exception as e:
            print("   lỗi / error:", e); continue
        rt = time.time() - t0
        if segs is not None: show(segs, box, rt)
        else: print(f"   box command {box * 1000:.0f} ms incl. model loading (file mode)")
        open(os.path.join(a.save, f"{stamp}_{turn:03d}_out.wav"), "wb").write(out)
        json.dump(dict(segments=segs, box_s=box, roundtrip_s=rt), open(os.path.join(a.save, f"{stamp}_{turn:03d}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if not a.no_play: play(out)
        if a.wav: break


if __name__ == "__main__":
    main()

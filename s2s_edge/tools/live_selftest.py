# -*- coding: utf-8 -*-
"""Exercise the real live-mode code path (`python -m s2s_edge live ...`, incl. warmup) without a microphone: the microphone is replaced by
WAV files and the speaker by files in logs/live_selftest/. Everything else (argument handling, warmup, translation loop, printing) is the code
the demo runs. Use it on the laptop (CPU) and on the box before plugging in a USB mic.
  python tools/live_selftest.py --wav samples/vi_0.wav -- --src vi --tgt en --models models_qcs8550/models.json --backend npu
  python tools/live_selftest.py --wav a.wav --wav b.wav -- --pair vi,ko --models ... --backend cpu
Arguments after `--` go to `s2s_edge live`. Exit code 0 when every utterance produced non-empty text and audio."""
import argparse, builtins, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import s2s_edge.__main__ as cli
from s2s_edge.audio import save_wav

ap = argparse.ArgumentParser(); ap.add_argument("--wav", action="append", required=True); ap.add_argument("--realtime", action="store_true", help="the fake speaker takes as long as the audio (shows the effect of pipelined playback)"); ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args(); rest = [x for x in a.rest if x != "--"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_btc_io import load_audio_any                                   # handles the MP3-in-.wav clips of the sample set
queue = [load_audio_any(p) for p in a.wav]; out_dir = os.path.join("logs", "live_selftest"); os.makedirs(out_dir, exist_ok=True)
played, heard_at = [], []


def fake_listen(*_, **__):
    if not queue: raise KeyboardInterrupt                             # ends the live loop like Ctrl+C
    heard_at.append(time.perf_counter()); return queue.pop(0)


def fake_play(x, sr):
    p = os.path.join(out_dir, f"out_{len(played)}.wav"); save_wav(p, x, sr); played.append((p, len(x) / sr, time.perf_counter()))
    if a.realtime: time.sleep(len(x) / sr)


cli.listen, cli.play = fake_listen, fake_play; builtins.input = lambda *_: ""
sys.argv = ["s2s_edge", "live", *rest]
t0 = time.perf_counter()
try: cli.main()
except KeyboardInterrupt: pass
print(f"\nselftest: {len(a.wav)} utterance(s) in, {len(played)} audio output(s): " + ", ".join(f"{os.path.basename(p)} {d:.1f} s" for p, d, _ in played))
ok = len(played) >= len(a.wav) and all(d > 0.2 for _, d, _ in played)
print("PASS" if ok else "FAIL", f"(total {time.perf_counter() - t0:.0f} s incl. warmup; outputs in {out_dir})")
if heard_at and played: print(f"from the end of input to the last audio finished: {played[-1][2] + (played[-1][1] if a.realtime else 0) - heard_at[0]:.1f} s; first audio started {played[0][2] - heard_at[0]:.1f} s after input")
sys.exit(0 if ok else 1)

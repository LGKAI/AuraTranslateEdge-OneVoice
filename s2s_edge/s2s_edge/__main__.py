# -*- coding: utf-8 -*-
"""Command line:
  python -m s2s_edge file  --src vi --tgt en --in in.wav --out out.wav   one file, per-stage timing
  python -m s2s_edge live  --pair vi,ko                                   microphone / loudspeaker plugged into the box
  python -m s2s_edge serve --data_root /root/onevoice                      web UI + API on 127.0.0.1:8770 (laptop: adb forward, open the page);
                                                                          models stay loaded, inference on the box. See s2s_edge/webui.py."""
import argparse, os, queue, sys, threading, time
import numpy as np
from .audio import load_audio, save_wav, listen, play, concat_results
from .registry import Registry, DIRECTIONS
from .pipeline import Translator


class Player:
    """Plays queued audio in a background thread, in order (sentence-pipelined live output). `play` is looked up at call time so tests can
    replace it (tools/live_selftest.py)."""

    def __init__(self):
        self.q = queue.Queue(); threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            wav, sr = self.q.get()
            try: globals()["play"](wav, sr)
            except Exception as e: print(f"  playback error: {e!r}", file=sys.stderr)
            finally: self.q.task_done()

    def put(self, wav, sr): self.q.put((wav, sr))

    def wait(self): self.q.join()


def fmt_t(t):
    return f"ASR {t['asr_s'] * 1000:.0f} | MT {t['mt_s'] * 1000:.0f} | TTS {t['tts_s'] * 1000:.0f} | total {t['total_s'] * 1000:.0f} ms (NPU graphs {t['graph_s'] * 1000:.0f} ms)"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="s2s_edge", description="Offline speech-to-speech translation (vi <-> en/zh/ko)")
    ap.add_argument("cmd", choices=["file", "live", "serve"])
    ap.add_argument("--src", choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--tgt", choices=["vi", "en", "zh", "ko"])
    ap.add_argument("--pair", help="bidirectional conversation between two languages, e.g. vi,ko (language of each utterance is detected)")
    ap.add_argument("--auto", action="store_true", help="Vietnamese + any of en/zh/ko, partner language detected per utterance")
    ap.add_argument("--partner", default="en", help="first partner language in --auto mode")
    ap.add_argument("--glossary", default=None, help="JSON {\"Name\": {\"vi\": ..., \"zh\": ..., \"ko\": ...}} overriding name transliteration")
    ap.add_argument("--models", default=os.environ.get("S2S_MODELS", "models/models.json"))
    ap.add_argument("--backend", default=os.environ.get("S2S_BACKEND", "npu"), choices=["npu", "cpu"])
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--no-play", action="store_true")
    ap.add_argument("--denoise", action="store_true", help="GTCRN speech enhancement (CPU) before ASR")
    ap.add_argument("--continuous", action="store_true", help="live mode without pressing Enter: always listening, translate and speak after each utterance")
    ap.add_argument("--no-warmup", action="store_true", help="live mode: skip loading + running every needed graph once before the first utterance")
    ap.add_argument("--no-stream", action="store_true", help="live mode: translate the whole utterance before playing (default: sentence-pipelined playback)")
    ap.add_argument("--vad-profile", default=None, choices=["hybrid", "legacy", "upstream"], help="VAD (default hybrid, or $S2S_VAD_PROFILE)")
    ap.add_argument("--ort-policy", default=None, choices=["auto", "basic", "disable"])
    ap.add_argument("--source-mode", default=None, choices=["0", "1"])
    ap.add_argument("--tts-format", default=None, choices=["hybrid", "legacy"])
    ap.add_argument("--port", type=int, default=8770, help="serve mode: TCP port, bound to 127.0.0.1 only")
    ap.add_argument("--data_root", default=os.environ.get("S2S_DATA", "/root/onevoice"), help="serve mode: folder with onevoice_btc_data/ and onevoice_btc_adv/ (data-set runs from the web UI)")
    a = ap.parse_args()
    t_load = time.perf_counter()
    reg = Registry(a.models, a.backend, a.glossary, ort_policy=a.ort_policy, vad_profile=a.vad_profile, source_mode=a.source_mode, tts_format=a.tts_format)
    if a.cmd == "serve":                                                     # web UI + API (s2s_edge/webui.py); all six directions available
        from .webui import serve
        return serve(reg, a.models, a.data_root, a.pair or "vi,ko", a.port, None if a.no_warmup else list(DIRECTIONS))
    if a.pair: dirs = [("vi", x) for x in a.pair.split(",") if x != "vi"] + [(x, "vi") for x in a.pair.split(",") if x != "vi"]
    elif a.auto: dirs = list(DIRECTIONS)
    else: dirs = [(a.src, a.tgt)]
    if a.pair or a.auto:
        from .conversation import Conversation
        pr = tuple(a.pair.split(",")) if a.pair else None
        if pr and "vi" not in pr: sys.exit("one language of the pair must be vi")
        tr = Conversation(reg, pair=pr, auto=a.auto, partner=a.partner, denoise=a.denoise); label = "auto" if a.auto else a.pair
    else:
        if not a.src or not a.tgt or (a.src, a.tgt) not in DIRECTIONS: sys.exit(f"give --src/--tgt (one side must be vi), or --pair / --auto. Supported: {DIRECTIONS}")
        tr = Translator(reg, a.src, a.tgt, denoise=a.denoise); label = f"{a.src}->{a.tgt}"
    if a.cmd == "file":
        wav_in = load_audio(a.inp); t_ready = time.perf_counter(); res = tr.translate(wav_in); t_done = time.perf_counter()
        for r in res:
            print(f"[{r['span'][0]:.1f}-{r['span'][1]:.1f}s] {r.get('src', a.src)}->{r.get('tgt', a.tgt)}: {r['asr']}\n   -> {r['mt']}\n   {fmt_t(r['timing'])}")
        print(f"audio {len(wav_in) / 16000:.1f} s | translation {(t_done - t_ready) * 1000:.0f} ms | model loading {(t_ready - t_load) * 1000:.0f} ms (paid once in live / serve mode)")
        wav, sr = concat_results(res); save_wav(a.out, wav, sr)
        print("saved", a.out)
    else:
        if not a.no_warmup:
            print("warming up (loading models)..."); reg.warmup([d for d in dirs if d in DIRECTIONS])
        player = Player()
        print(f"Live {label}. Enter = speak, Ctrl+C = quit." + ("" if a.no_stream else " (sentence-pipelined output)"))
        while True:
            if not a.continuous:
                input("[Enter] to speak > ")
            print("listening...")
            w = listen()
            if len(w) < 4800:
                if not a.continuous: print("(nothing heard)")
                continue
            if a.no_stream:                                                    # old behaviour: whole utterance, then play
                for r in tr.translate(w):
                    t = r["timing"]
                    print(f"  heard : {r['asr']}\n  output: {r['mt']}\n  {t['total_s'] * 1000:.0f} ms (ASR {t['asr_s'] * 1000:.0f} | MT {t['mt_s'] * 1000:.0f} | TTS {t['tts_s'] * 1000:.0f}), graphs {t['graph_s'] * 1000:.0f} ms")
                    if len(r["wav"]) and not a.no_play:
                        play(r["wav"], r["sr"])
                continue
            # pipelined: sentence k is played while sentence k+1 is translated / synthesised
            t_end = time.perf_counter(); first = None; n = 0
            for r in tr.translate_stream(w):
                t = r["timing"]; n += 1; i, cnt = r.get("part", (0, 1))
                print(f"  [{i + 1}/{cnt}] heard : {r['asr']}\n        output: {r['mt']}\n        ASR {t['asr_s'] * 1000:.0f} | MT {t['mt_s'] * 1000:.0f} | TTS {t['tts_s'] * 1000:.0f} ms")
                if len(r["wav"]) and not a.no_play:
                    if first is None: first = time.perf_counter() - t_end; print(f"        first audio {first * 1000:.0f} ms after the end of speech")
                    player.put(r["wav"], r["sr"])
            if n == 0: print("  (xin nói lại / please repeat)")              # nothing intelligible: say nothing instead of a meaningless sentence
            player.wait()                                                     # do not listen while the loudspeaker is still playing (the mic would hear it)


if __name__ == "__main__":
    main()

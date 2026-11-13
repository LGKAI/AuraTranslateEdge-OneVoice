# -*- coding: utf-8 -*-
"""Model-free checks of the host-side ASR code (window choice, output parsing, ITN repair, segmenter, level normalisation, CMVN, LFR).
  python tools/test_asr_host.py           (seconds, no ONNX / NPU needed; exits non-zero on the first failure)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from s2s_edge.asr import pick, windows, zip_outputs, fix_itn, load_cmvn, sv_features, _word_ends, LFR_M, LFR_N
from s2s_edge.audio import segment, normalize_level, active_rms, vad_mode, SR
from s2s_edge.conversation import restore_vi, _break_positions, _align_breaks
from s2s_edge.registry import ASSETS
from s2s_edge.textnorm import vi_punct

ok = 0


def check(cond, what):
    global ok
    if not cond: sys.exit(f"FAIL: {what}")
    ok += 1


# window choice
ws = windows([(1500, "g1500"), (300, "g300"), (800, "g800")], 1500)
check([w for w, _ in ws] == [300, 800, 1500], "windows sorted")
check(pick(ws, 1) == (300, "g300") and pick(ws, 300) == (300, "g300") and pick(ws, 301) == (800, "g800"), "smallest window that fits")
check(pick(ws, 5000) == (1500, "g1500"), "largest window when nothing fits")
check(windows("g", 1500) == [(1500, "g")], "a single graph is one window")

# Zipformer outputs recognised by size (multi-graph contexts may return flat arrays)
for T in (300, 1500):
    tp = ((T - 7) // 2 + 1) // 2
    out = [np.zeros(tp * 2000, np.float32), np.array([tp - 3]), np.ones(tp * 512, np.float32)]
    lg, el, eo = zip_outputs(out, T); check(lg.shape == (1, tp, 2000) and eo.shape == (tp, 512) and int(el[0]) == tp - 3, f"zip_outputs flat T={T}")
    lg, el, eo = zip_outputs([np.zeros((1, tp, 2000)), np.array([5])], T); check(eo is None, "CTC-only graph has no encoder_out")

# English time repair (SenseVoice ITN drops the "▁" and ":" tokens)
cases = {"on Friday, at530 AM.": "on Friday, at 5:30 AM.", "It is 1230 PM now": "It is 12:30 PM now", "from930am to1130 a.m.": "from 9:30 am to 11:30 a.m.",
         "Meet at 5:30 AM.": "Meet at 5:30 AM.", "about 1530 PM": "about 1530 PM", "I have 530 amps": "I have 530 amps", "the 10 AM bus": "the 10 AM bus",
         "Kat5 is a name": "Kat5 is a name", "at 12 PM.": "at 12 PM."}
for a, b in cases.items(): check(fix_itn(a, "en") == b, f"fix_itn {a!r} -> {fix_itn(a, 'en')!r}")
check(fix_itn("at530 AM", "zh") == "at530 AM", "fix_itn only for en/auto")

# CMVN parse and LFR geometry (FunASR: ceil(T/6) LFR frames of 7 stacked fbank frames)
cm = load_cmvn(os.path.join(ASSETS, "sensevoice_am.mvn")); check(cm.shape == (2, 560) and np.all(cm[1] > 0), "am.mvn -> (2, 560)")
for n in (400, 401, 16000, 16000 * 7 + 123):
    w = (np.random.default_rng(0).standard_normal(n) * 0.05).astype(np.float32); x = sv_features(w, cm); T = 1 + (n - 400) // 160
    check(x.shape == (-(-T // LFR_N), 80 * LFR_M), f"sv_features shape n={n}: {x.shape}")


# segmenter: two words with a weak coda and a short pause become one segment that keeps the coda; separate utterances stay separate
def tone(s, amp, f=220.0): t = np.arange(int(s * SR)) / SR; return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
rng = np.random.default_rng(1); noise = lambda s: (rng.standard_normal(int(s * SR)) * 3e-4).astype(np.float32)
utt = np.concatenate([noise(1.0), tone(0.6, 0.3), tone(0.25, 0.012, 3000.0), noise(0.6), tone(0.8, 0.3), tone(0.3, 0.01, 4000.0), noise(1.5)])
segs = segment(utt); coda_end = (1.0 + 0.6 + 0.25 + 0.6 + 0.8 + 0.3) * SR
check(len(segs) == 1, f"in-sentence pause joined: {segs}"); check(segs[0][1] >= coda_end, f"weak coda kept: ends {segs[0][1] / SR:.2f} s < {coda_end / SR:.2f} s")
two = np.concatenate([noise(1.0), tone(1.5, 0.3), noise(2.5), tone(1.5, 0.3), noise(1.0)])
check(len(segment(two)) == 2, f"utterances 2.5 s apart stay separate: {segment(two)}")
long = np.concatenate([noise(0.5)] + [np.concatenate([tone(2.0, 0.3), noise(0.3)]) for _ in range(9)] + [noise(0.5)])
check(all((b - a) / SR <= 12.0 + 1e-6 for a, b in segment(long, max_s=12.0)), "segments never longer than max_s")
segs = segment(two); check(all(segs[i][1] <= segs[i + 1][0] for i in range(len(segs) - 1)), "padded segments do not overlap")

# return_breaks + S2S_VAD_MODE: 2 sentences 0.4 s apart -> 1 seg + 1 break in hybrid, 2 segs in split
paused = np.concatenate([noise(0.5), tone(1.0, 0.3), noise(0.4), tone(1.0, 0.3), noise(0.5)])
old_mode = os.environ.pop("S2S_VAD_MODE", None)
try:
    check(vad_mode() == "hybrid", f"default vad_mode={vad_mode()}")
    segs, brks = segment(paused, return_breaks=True)
    check(len(segs) == 1 and len(brks) == 1 and len(brks[0]) == 1, f"0.4s pause -> 1 seg + 1 break: {segs}, {brks}")
    b_abs = segs[0][0] + brks[0][0]
    check(1.4 * SR <= b_abs <= 2.0 * SR, f"break sample inside 0.4s pause: {b_abs / SR:.2f} s")
    segs2, brks2 = segment(two, return_breaks=True)
    check(len(segs2) == 2 and brks2 == [[], []], f"separate utterances have empty breaks: {brks2}")
    os.environ["S2S_VAD_MODE"] = "split"
    check(vad_mode() == "split", f"split vad_mode={vad_mode()}")
    check(len(segment(paused)) == 2, f"split mode separates 0.4s pause: {segment(paused)}")
finally:
    if old_mode is None:
        os.environ.pop("S2S_VAD_MODE", None)
    else:
        os.environ["S2S_VAD_MODE"] = old_mode

# level normalisation: quiet speech is brought up, loud speech left alone, peak never clipped, silence untouched
q = tone(2.0, 0.003); y = normalize_level(q); check(abs(20 * np.log10(active_rms(y)) + 24) < 0.6, f"quiet -> -24 dBFS active RMS ({20 * np.log10(active_rms(y)):.1f})")
spiky = np.concatenate([tone(2.0, 0.003), np.array([0.5], np.float32)]); y = normalize_level(spiky)
check(np.abs(y).max() <= 1.0 and abs(20 * np.log10(active_rms(y[:-1])) + 24) < 1.0, "one click does not block the gain, no overflow")
lv = [20 * np.log10(active_rms(normalize_level(tone(1.0, 10 ** (d / 20) * np.sqrt(2))))) for d in np.arange(-60, -3, 0.5)]
check(all(b >= a - 0.05 for a, b in zip(lv, lv[1:])), "level map is monotonic")
loud = tone(1.0, 0.1); check(normalize_level(loud) is loud, "normal level unchanged")
check(np.all(normalize_level(np.zeros(1600, np.float32)) == 0), "digital silence unchanged")

# break-to-word mapping + restore_vi / vi_punct with breaks
class _FakeSp:
    def decode(self, ids): return "".join(ids).replace("\u2581", " ").strip()

we = _word_ends(_FakeSp(), ["\u2581tôi", "\u2581đi", "\u2581h", "ọc"], [2, 5, 8, 10])
check(we == [2, 5, 10], f"_word_ends: {we}")
check(_break_positions([int(0.28 * SR)], we) == [2], f"break positions: {_break_positions([int(0.28 * SR)], we)}")
check(_align_breaks(["tôi", "đi", "làm", "việc"], ["tôi", "đi", "làm_việc"], [2]) == [2], "align breaks across merged token")
old_vi_punct = os.environ.get("S2S_VI_PUNCT")
os.environ["S2S_VI_PUNCT"] = "1"
try:
    s1 = restore_vi("tôi rất hài lòng bao nhiêu tiền", breaks=[4])
    check(s1 == "Tôi rất hài lòng. Bao nhiêu tiền?", f"restore_vi break + question: {s1!r}")
    s2 = restore_vi("anh đi đâu thế tôi đi làm", breaks=[4])
    check(s2 == "Anh đi đâu thế? Tôi đi làm.", f"restore_vi question before break: {s2!r}")
    s3 = vi_punct("tôi đi làm, anh ở nhà", breaks=[3])
    check(s3 == "Tôi đi làm. Anh ở nhà.", f"vi_punct break strips comma: {s3!r}")
finally:
    if old_vi_punct is None:
        os.environ.pop("S2S_VI_PUNCT", None)
    else:
        os.environ["S2S_VI_PUNCT"] = old_vi_punct

print(f"ok: {ok} checks passed")

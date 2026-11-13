# -*- coding: utf-8 -*-
"""ON THE BOX (as the user that can open the NPU): step-by-step SenseVoice (asr_xx) on a few real clips, to find why ko / zh come out empty.
  runuser -u system -- bash -c 'cd /tmp/onevoice/s2s_edge_pkg && . .venv/bin/activate && . tools/env.sh 2>/dev/null && python tools/asr_debug.py 2>&1 | grep -a "^##" | tee /tmp/asr_debug.txt'
Prints, per clip: frontend outputs (shape, dtype, the length value), encoder output statistics and the decoded text, for several language ids / textnorm values."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry
from s2s_edge.audio import load_audio
from s2s_edge.asr import LID
DATA = os.environ.get("S2S_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "onevoice_btc_data"))
CLIPS = [("ko", "Vie-Kor dataset/Kor_voice/VK-0143.wav"), ("zh", "Vie-Man dataset/Man_voice/VM-0223.wav"), ("en", None)]
CFG = os.environ.get("S2S_MODELS", "models_qcs8550/models.json"); print("## models:", CFG); reg = Registry(CFG, "npu"); fe, enc = reg.graph("asr_xx_frontend"), reg.graph("asr_xx_encoder")
print("## in_names frontend:", fe.in_names, fe.in_types, "| encoder:", enc.in_names, enc.in_types)
for lang, rel in CLIPS:
    w = load_audio(os.path.join(DATA, rel)) if rel else load_audio(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "en_0.wav"))
    n = min(len(w), 464000); b = np.zeros((1, 464000), np.float32); b[0, :n] = w[:n]
    f = fe(wav=b, wav_len=np.array([n], np.int32)); fb = [np.asarray(x) for x in f]
    print(f"## clip {lang}: {n} samples ({n / 16000:.2f} s); frontend outputs:", [(x.shape, str(x.dtype), x.reshape(-1)[:2].tolist() if x.size < 10 else round(float(np.abs(x).mean()), 4)) for x in fb])
    big = [x for x in fb if x.size > 1000][0].reshape(1, 500, 560).astype(np.float32); sl = [x for x in fb if x.size == 1][0].reshape(1).astype(np.int32)
    for lid_name, lid in (("own", LID[lang]), ("en4", 4)):
        for tn in (14,):
            o = enc(fbank=big, speech_lengths=sl, language=np.array([lid], np.int32), textnorm=np.array([tn], np.int32)); a = np.asarray(o[0]).astype(np.int64).reshape(-1)
            txt = bytes(int(v) & 0xFF for v in a).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
            print(f"##   lid={lid_name}({lid}) textnorm={tn}: sl={int(sl[0])} out dtype={np.asarray(o[0]).dtype} nonzero={int((a != 0).sum())} first={a[:12].tolist()} -> {txt!r}")

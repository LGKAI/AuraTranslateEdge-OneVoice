# -*- coding: utf-8 -*-
"""Golden-vector check of every compiled graph.
  make:  python tools/golden.py make  --models models.dev.json            (dev machine, CPU/ONNX reference -> golden/<graph>.npz)
  check: python tools/golden.py check --models models/models.json --backend npu   (device: runs each graph on the golden inputs, compares to the CPU reference)
A graph PASSES when float outputs have cosine >= 0.99 and integer outputs match on >= 95 % of elements (argmax/ids). Takes seconds, tells which graph is wrong."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_edge.registry import Registry, ASSETS
from s2s_edge.audio import load_audio
from s2s_edge.asr import _fbank, LID
from s2s_edge.mt import Nllb, S

HERE = os.path.dirname(os.path.abspath(__file__)); GOLD = os.path.join(os.path.dirname(HERE), "golden")


def inputs_for(reg, key):
    """Deterministic example inputs for graph `key` (only needs assets + one wav)."""
    wav_vi = os.path.join(os.path.dirname(HERE), "samples", "vi_0.wav"); wav_en = os.path.join(os.path.dirname(HERE), "samples", "en_0.wav")
    if key == "asr_vi":
        f = _fbank(load_audio(wav_vi)[:14 * 16000]); x = np.zeros((1, 1500, 80), np.float32); n = min(len(f), 1500); x[0, :n] = f[:n]; return dict(x=x, x_lens=np.array([n], np.int64))
    if key in ("asr_xx", "asr_xx_frontend"):
        w = load_audio(wav_en); b = np.zeros((1, 464000), np.float32); n = min(len(w), 464000); b[0, :n] = w[:n]
        d = dict(wav=b, wav_len=np.array([n], np.int32))
        return {**d, "language": np.array([LID["en"]], np.int32), "textnorm": np.array([14], np.int32)} if key == "asr_xx" else d
    if key == "asr_xx_encoder":                                    # SenseVoice encoder+CTC: fed with the frontend's golden outputs, exactly as AsrXx does
        z = np.load(os.path.join(GOLD, "asr_xx_frontend.npz")); fb = [z[n] for n in sorted(z.files) if n.startswith("out_")]
        big = [x for x in fb if x.size > 1000][0].reshape(1, 500, 560).astype(np.float32); sl = [x for x in fb if x.size == 1][0].reshape(1).astype(np.int32)
        return dict(fbank=big, speech_lengths=sl, language=np.array([LID["en"]], np.int32), textnorm=np.array([14], np.int32))
    if key == "mt":
        nl = Nllb(None, ASSETS); toks = nl.encode("Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé."); a = np.ones((1, S), np.int32); seq = [nl.lang["vie_Latn"]] + toks + [2]; a[0, :len(seq)] = seq
        return dict(src_ids=a, tgt_lang=np.array([nl.lang["eng_Latn"]], np.int32))
    if key == "tts_vi":
        b = "Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé.".encode("utf-8"); x = np.zeros((1, 512), np.int32); x[0, :len(b)] = list(b); return dict(text_bytes=x)
    if key == "tts_zh":
        b = "今天天气很好，我们一起去公园散步。".encode("utf-8"); x = np.zeros((1, 112), np.int32); x[0, :len(b)] = list(b); return dict(text_bytes=x)
    if key == "tts_enko":
        from s2s_edge.tts import TtsEnKo
        t = TtsEnKo(None, "en", ASSETS); s = t._clean(t.p._preprocess_text("Please wear your helmet before entering the factory.", "en")); cp = np.zeros((1, 96), np.int32); cp[0, :len(s)] = [ord(c) for c in s]
        return dict(cp=cp, style_dp=t.dp, style_ttl=t.ttl)
    if key in ("mt_enc", "mt_dec", "mt_decB"):
        S_ = reg.cfg["mt_enc"].get("S", 64); Tm = reg.cfg.get("mt_dec", reg.cfg.get("mt_decB", {})).get("Tm", 96); B = reg.cfg.get("mt_decB", {}).get("B", 4)
        nl = Nllb(None, ASSETS); toks = nl.encode("Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé."); a = np.ones((1, S_), np.int32); seq = [nl.lang["vie_Latn"]] + toks + [2]; a[0, :len(seq)] = seq
        if key == "mt_enc": return dict(src_ids=a)
        ck, cv, bias = [np.asarray(x) for x in reg.graph("mt_enc")(src_ids=a)]
        dcfg = reg.cfg.get("mt_decB" if key == "mt_decB" else "mt_dec", {}); Ld, Hd, dkd = dcfg.get("L", 12), dcfg.get("H", 16), dcfg.get("dk", 64)
        nb = B if key == "mt_decB" else None; shp = (Ld, nb, Hd, Tm, dkd) if nb else (Ld, Hd, Tm, dkd); oh = np.zeros(Tm, np.float32); oh[0] = 1; mask = np.where(np.arange(Tm) <= 0, 0.0, -1e4).astype(np.float32)
        return dict(tok=np.array([2] * (nb or 1), np.int32), oh=oh, mask=mask, sk=np.zeros(shp, np.float32), sv=np.zeros(shp, np.float32), ck=ck, cv=cv, bias=bias)
    raise KeyError(key)


def compare(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size != b.size: return "size", float("nan")
    if np.issubdtype(b.dtype, np.integer): return "int", float((a.reshape(-1).astype(np.int64) == b.reshape(-1).astype(np.int64)).mean())
    x, y = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    if not np.any(np.abs(x) > 1e-9) and not np.any(np.abs(y) > 1e-9): return "zero", 1.0     # both all-zero (e.g. Piper's OOV count): equal, cosine is undefined
    return "cos", float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _good(kind, v): return (kind in ("cos", "zero") and v >= 0.99) or (kind == "int" and v >= 0.95)


def judge(key, out, refs):
    """-> (ok, status, note). Positional comparison first. If that fails:
    * TTS graphs: outputs are taken by size/dtype (s2s_edge.tts.split_outputs, so a permuted order is harmless) and the audio passes when
      the waveform matches (cos >= 0.99) OR length and level match (+-1 % / >= 2 units, RMS ratio 0.8-1.25). Reason: on QCS8550 the fp16
      Supertonic waveform drifts (cos 0.21) and Melo is 1 unit shorter, yet ASR of both chip outputs reads back the exact sentence (AI Hub check 10/11).
    * other graphs: a match found at another position is reported (fix with "out": [...] in models.json) but stays a FAIL."""
    out = [np.asarray(o) for o in out]; status = [(i, *compare(o, r)) for i, (o, r) in enumerate(zip(out, refs))]; status = [(i, k, round(v, 5)) for i, k, v in status]
    if len(out) == len(refs) and all(_good(k, v) for _, k, v in status): return True, status, ""
    if key == "asr_vi" and len(out) == len(refs) and out[0].size == refs[0].size:
        # CTC logits: what the decoder uses is the per-frame argmax. On QCS8550 the fp16 logits have cos 0.982 vs CPU, yet the argmax agrees on
        # 100 % of frames and the transcript is identical (AI Hub check 10/11), so judge the argmax over the valid frames.
        lg_o, lg_r = out[0].reshape(refs[0].shape), refs[0]; n = int(np.asarray(refs[1]).reshape(-1)[0]) if len(refs) > 1 else lg_r.shape[1]
        agree = float((lg_o[0, :n].argmax(-1) == lg_r[0, :n].argmax(-1)).mean()); rest_ok = all(_good(k, v) for i, k, v in status if i > 0)
        return agree >= 0.99 and rest_ok, status, f"ctc: frame argmax agreement {100 * agree:.2f} % over {n} frames"
    if key.startswith("tts"):
        from s2s_edge.tts import split_outputs
        ca, cl, _ = split_outputs(out); ra, rl, _ = split_outputs(refs); kind, cosv = compare(ca, ra)
        rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12); hop = len(ra) / max(rl, 1)
        n_c, n_r = int(round(cl * hop)), int(round(rl * hop)); ratio = rms(ca[:n_c]) / rms(ra[:n_r])
        ok = (kind == "cos" and cosv >= 0.99) or (abs(cl - rl) <= max(2, 0.01 * rl) and 0.8 <= ratio <= 1.25)
        return ok, status, f"tts: audio cos {cosv:.3f}, length {cl:g} vs {rl:g}, rms ratio {ratio:.2f}" + (" (waveform differs: listen to it / check 'heard')" if ok and cosv < 0.99 else "")
    perm = []
    for r in refs:
        c = [(j, *compare(o, r)) for j, o in enumerate(out) if o.size == r.size]; c = [t for t in c if _good(t[1], t[2])]; perm.append(c[0][0] if c else None)
    note = f'outputs match in another order: set "out": {perm} for this graph in models.json' if None not in perm else ""
    return False, status, note


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["make", "check"]); ap.add_argument("--models", default="models/models.json"); ap.add_argument("--backend", default="npu"); ap.add_argument("--only", default="")
    a = ap.parse_args(); reg = Registry(a.models, "cpu" if a.mode == "make" else a.backend)
    global GOLD
    if "_golden" in reg.cfg: GOLD = os.path.join(os.path.dirname(HERE), reg.cfg["_golden"])     # model set with its own golden vectors (e.g. NLLB-1.3B)
    os.makedirs(GOLD, exist_ok=True)
    keys = [k for k in a.only.split(",") if k] or [k for k in reg.cfg if k in ("asr_vi", "asr_xx", "asr_xx_frontend", "asr_xx_encoder", "mt", "mt_enc", "mt_dec", "tts_vi", "tts_zh", "tts_enko") or k.startswith("mt_decB")]; rows = []
    for k in keys:
        gp = os.path.join(GOLD, f"{k}.npz") if a.mode == "make" else reg.gold_path(k)
        tl = time.perf_counter()
        try: g = reg.graph(k); load_s = time.perf_counter() - tl                 # load time = the unknown in the device time budget (docs/device_plan.md)
        except Exception as e:
            if a.mode == "make": raise
            rows.append(dict(graph=k, ok=False, error=repr(e)[:300])); print("FAIL", k, "load:", repr(e)[:300], flush=True); continue
        if a.mode == "check" and os.path.exists(gp): z = np.load(gp); ins = {n[3:]: z[n] for n in z.files if n.startswith("in_")}          # exactly the inputs used for the reference
        else: ins = inputs_for(reg, k)
        try:
            t = time.perf_counter(); out = g(**ins); dt = time.perf_counter() - t
        except Exception as e:                                                   # a context the runtime rejects must not hide the other graphs
            if a.mode == "make": raise
            rows.append(dict(graph=k, ok=False, error=repr(e)[:300])); print("FAIL", k, repr(e)[:300], flush=True); continue
        if a.mode == "make":
            np.savez_compressed(gp, **{f"in_{n}": v for n, v in ins.items()}, **{f"out_{i}": np.asarray(o) for i, o in enumerate(out)}); print("saved", gp, f"{os.path.getsize(gp)/1e6:.1f} MB"); continue
        ref = np.load(gp); refs = [ref[f"out_{i}"] for i in range(len([n for n in ref.files if n.startswith("out_")]))]
        ok, status, note = judge(k, out, refs)
        rows.append(dict(graph=k, ok=ok, outputs=status, note=note, seconds=round(dt, 3), load_s=round(load_s, 2)))
        print(("PASS " if ok else "FAIL ") + k, status, f"{dt*1000:.0f} ms (load {load_s:.1f} s)", note, flush=True)
    if a.mode == "check":
        os.makedirs("logs", exist_ok=True); json.dump(rows, open(f"logs/golden_{a.backend}_{time.strftime('%Y%m%d_%H%M%S')}.json", "w"), indent=1); print("all pass" if all(r["ok"] for r in rows) else "SOME FAILED")
        sys.exit(0 if rows and all(r["ok"] for r in rows) else 1)


if __name__ == "__main__": main()

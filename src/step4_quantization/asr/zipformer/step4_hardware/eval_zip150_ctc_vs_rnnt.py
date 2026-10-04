# -*- coding: utf-8 -*-
"""ZipFormer-150M-CR-CTC-RNNT-6000h: do WER cua head CTC (KHONG train them) so voi RNN-T cua CHINH model do,
tren 5 cau FLEURS-vi sach + 5 cau nhieu 5dB + 5 cau nhieu 0dB (cung bo voi step1/step4).
Encoder ONNX cua tac gia xuat 512-chieu (sau encoder_proj) -> ta lay tensor 768-chieu TRUOC encoder_proj
(/Transpose_1_output_0) lam dau ra them, roi ap head CTC (ctc_output.1) lay tu pretrained.pt.
Chay trong tools/venv_nllb_export."""
import json, os, re, sys, time
import numpy as np, onnx, onnxruntime as ort, soundfile as sf, jiwer
import sentencepiece as spm
import kaldi_native_fbank as knf

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT)
D = "third_party_zipformer_150m_crctc"
HEAD = os.path.join(D, "ctc_head.npz")
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()
sp = spm.SentencePieceProcessor(); sp.load(os.path.join(D, "bpe.model"))

# ---- 1. head CTC tu pretrained.pt (mmap: khong nap 2.4GB vao RAM) ----
if not os.path.exists(HEAD):
    import torch, pathlib
    pathlib.PosixPath = pathlib.WindowsPath      # ckpt luu tu Linux: co pickle PosixPath trong params
    ck = torch.load(os.path.join(D, "pretrained.pt"), map_location="cpu", mmap=True, weights_only=False)
    sd = ck.get("model", ck)
    keys = [k for k in sd if "ctc" in k.lower()]
    print("keys chua 'ctc':", keys, "| tong key:", len(sd), "| cac khoa cap 1:", list(ck.keys())[:6])
    wk = [k for k in keys if k.endswith("weight")][0]; bk = [k for k in keys if k.endswith("bias")][0]
    W, b = sd[wk].float().numpy(), sd[bk].float().numpy()
    print("ctc head:", wk, W.shape, bk, b.shape)
    np.savez(HEAD, W=W, b=b)
hd = np.load(HEAD); W, b = hd["W"], hd["b"]           # W: (vocab, 768)
assert W.shape == (2000, 768), W.shape

# ---- 2. encoder ONNX + them dau ra tensor 768-chieu truoc encoder_proj ----
enc_path = os.path.join(D, "encoder-epoch-11-avg-2.onnx")
m = onnx.load(enc_path)
pre = "/Transpose_1_output_0"
m.graph.output.append(onnx.helper.make_tensor_value_info(pre, onnx.TensorProto.FLOAT, ["N", "T", 768]))
so = ort.SessionOptions(); so.log_severity_level = 3
enc = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(os.path.join(D, "decoder-epoch-11-avg-2.onnx"), so, providers=["CPUExecutionProvider"])
joi = ort.InferenceSession(os.path.join(D, "joiner-epoch-11-avg-2.onnx"), so, providers=["CPUExecutionProvider"])
print("enc outputs:", [o.name for o in enc.get_outputs()], "| dec in:", [i.name for i in dec.get_inputs()], "| joiner in:", [i.name for i in joi.get_inputs()])


def fbank(wav, snip=False, hf=-400.0):
    o = knf.FbankOptions(); o.mel_opts.num_bins = 80; o.frame_opts.samp_freq = 16000; o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = snip; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = hf
    f = knf.OnlineFbank(o); f.accept_waveform(16000, wav.tolist()); f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


def ctc_collapse(ids):
    out, prev = [], None
    for t in ids:
        if t != 0 and t != prev: out.append(int(t))
        prev = t
    return out


def rnnt_greedy(enc_out):                              # enc_out (T, 512); context_size = 2, toi da 1 ky hieu/frame
    hyp = [-1, -1]; d = dec.run(None, {dec.get_inputs()[0].name: np.array([hyp[-2:]], dtype=np.int64)})[0]
    for t in range(enc_out.shape[0]):
        lg = joi.run(None, {joi.get_inputs()[0].name: enc_out[t:t + 1], joi.get_inputs()[1].name: d})[0]
        k = int(lg.argmax())
        if k != 0:
            hyp.append(k); d = dec.run(None, {dec.get_inputs()[0].name: np.array([hyp[-2:]], dtype=np.int64)})[0]
    return [t for t in hyp[2:]]


rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for tag in ("clean", "snr5", "snr0"):
    for r in rows:
        f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
        if os.path.exists(f): items.append((tag, r["transcript"], sf.read(f, dtype="float32")[0]))

for cfg_name, snip, hf in [("sherpa-like (snip_edges=False, high_freq=-400)", False, -400.0),
                           ("knf mac dinh nhu eval cu (snip_edges=True, high_freq=0)", True, 0.0)]:
    res = {}; t_enc = 0.0; t_aud = 0.0
    for tag, ref, wav in items:
        x = fbank(wav, snip, hf)[None]; xl = np.array([x.shape[1]], dtype=np.int64)
        t0 = time.time(); e512, el, e768 = enc.run(None, {"x": x, "x_lens": xl}); t_enc += time.time() - t0; t_aud += len(wav) / 16000
        logits = e768[0] @ W.T + b                                          # (T', 2000)
        ctc_txt = sp.decode(ctc_collapse(logits.argmax(-1)))
        rnnt_txt = sp.decode(rnnt_greedy(e512[0]))
        d = res.setdefault(tag, {"ctc": [], "rnnt": []})
        d["ctc"].append(jiwer.wer(norm(ref), norm(ctc_txt))); d["rnnt"].append(jiwer.wer(norm(ref), norm(rnnt_txt)))
        if tag == "clean" and len(d["ctc"]) <= 5:
            print(f"  [{tag}] REF : {ref[:90]}\n         CTC : {ctc_txt[:90]}\n         RNNT: {rnnt_txt[:90]}")
    print(f"\n== {cfg_name} | ORT CPU RTF(encoder) = {t_enc/t_aud:.3f}")
    for tag, d in res.items():
        print(f"   {tag:6s} n={len(d['ctc'])}  CTC WER = {np.mean(d['ctc'])*100:5.2f}%   RNNT WER = {np.mean(d['rnnt'])*100:5.2f}%")
    print("   (30M RNN-T tham chieu: clean 6.22%)\n")

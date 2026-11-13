# -*- coding: utf-8 -*-
"""Zipformer-150M-CR-CTC-RNNT-6000h -> static ONNX graphs for SEVERAL window lengths ("buckets"), same surgery as build_zip150_ctc_static.py
(bool Slice wrapped in Cast, encoder_proj replaced by the CTC head of pretrained.pt, x / x_lens pinned), but cross-platform and parameterised:

  python build_zip150_buckets.py --model-dir third_party/zipformer/third_party_zipformer_150m_crctc --out-dir out/zip150_buckets \
         --frames 300,500,800,1200,1500 [--with-encoder-out] [--check-wavs data/asr/vi]

Why: the NPU graph is static, so the 15 s window (1500 fbank frames, 163 ms on QCS8550) is paid even for a 2 s utterance. s2s_edge picks the
smallest compiled window that holds the utterance (models.json keys "asr_vi" + "asr_vi@<frames>", see s2s_edge/registry.py). Padding is masked
by x_lens inside the encoder, so a shorter window gives the same transcript (checked here: argmax agreement + transcript equality against the
largest window on --check-wavs).
--with-encoder-out also keeps encoder_proj and exposes "encoder_out" (T', 512): the input of the RNN-T joiner, for transducer decoding on the host.
Inputs needed in --model-dir: encoder-epoch-11-avg-2.onnx (Hugging Face hynt/ZipFormer-150M-CR-CTC-RNNT-6000h) and the CTC head as ctc_head.npz
(W (2000,768), b (2000,)) or jit_script.pt / pretrained.pt to extract it from (needs torch)."""
import argparse, glob, os, re, sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_zipformer_for_qnn import find_bool_slices, wrap_bool_slice


def ctc_head(model_dir):
    p = os.path.join(model_dir, "ctc_head.npz")
    if not os.path.exists(p):
        import torch
        for name in ("jit_script.pt", "pretrained.pt"):
            src = os.path.join(model_dir, name)
            if not os.path.exists(src): continue
            if name == "jit_script.pt": sd = torch.jit.load(src, map_location="cpu").state_dict()
            else: ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False); sd = ck.get("model", ck)
            w = [k for k in sd if "ctc" in k.lower() and k.endswith("weight")][0]; b = [k for k in sd if "ctc" in k.lower() and k.endswith("bias")][0]
            np.savez(p, W=sd[w].float().numpy(), b=sd[b].float().numpy()); break
        else: sys.exit("no ctc_head.npz / jit_script.pt / pretrained.pt in --model-dir")
    z = np.load(p); return z["W"].astype(np.float32), z["b"].astype(np.float32)


def build(src_onnx, W, b, T, with_encoder_out=False):
    m = onnx.load(src_onnx)
    for n in find_bool_slices(m): wrap_bool_slice(m.graph, n)
    assert not find_bool_slices(m), "bool Slice left"
    byout = {o: n for n in m.graph.node for o in n.output}
    mm, ad = byout["/encoder_proj/MatMul_output_0"], byout["encoder_out"]
    assert mm.op_type == "MatMul" and ad.op_type == "Add"
    pre = mm.input[0]
    if not with_encoder_out:                                       # CTC only (as deployed): drop encoder_proj (768 -> 512, used by the joiner)
        inits = {mm.input[1], ad.input[1] if ad.input[0] == mm.output[0] else ad.input[0]}
        users = {i for n in m.graph.node if n not in (mm, ad) for i in n.input}
        assert not (inits & users), "encoder_proj weights used elsewhere"
        m.graph.node.remove(mm); m.graph.node.remove(ad)
        for i in [x for x in m.graph.initializer if x.name in inits]: m.graph.initializer.remove(i)
    m.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W.T), "ctc_head.W"))
    m.graph.initializer.append(numpy_helper.from_array(b, "ctc_head.b"))
    m.graph.node.append(helper.make_node("MatMul", [pre, "ctc_head.W"], ["ctc_head/mm"], name="ctc_head/MatMul"))
    m.graph.node.append(helper.make_node("Add", ["ctc_head/mm", "ctc_head.b"], ["ctc_logits"], name="ctc_head/Add"))
    del m.graph.output[:]
    m.graph.output.append(helper.make_tensor_value_info("ctc_logits", TensorProto.FLOAT, [1, "T", W.shape[0]]))
    m.graph.output.append(helper.make_tensor_value_info("encoder_out_lens", TensorProto.INT64, [1]))
    if with_encoder_out: m.graph.output.append(helper.make_tensor_value_info("encoder_out", TensorProto.FLOAT, [1, "T", 512]))
    for inp in m.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if inp.name == "x":
            for d, v in zip(dims, (1, T, 80)): d.ClearField("dim_param"); d.dim_value = v
        elif inp.name == "x_lens": dims[0].ClearField("dim_param"); dims[0].dim_value = 1
    return m


def fbank(w):
    import kaldi_native_fbank as knf
    o = knf.FbankOptions(); o.mel_opts.num_bins = 80; o.frame_opts.samp_freq = 16000; o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = -400.0
    f = knf.OnlineFbank(o); f.accept_waveform(16000, w.tolist()); f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)


def check(paths, wav_dir, bpe):
    """Every window vs the largest one on real clips that fit: frame argmax agreement and transcript equality."""
    import onnxruntime as ort, soundfile as sf, sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=bpe); so = ort.SessionOptions(); so.log_severity_level = 3
    sess = {T: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for T, p in paths.items()}; Tmax = max(sess)
    wavs = sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))[:40]
    def run(T, f):
        n = min(len(f), T); x = np.zeros((1, T, 80), np.float32); x[0, :n] = f[:n]
        lg, el = sess[T].run(None, {"x": x, "x_lens": np.array([n], np.int64)})[:2]; ids = lg[0, :int(el[0])].argmax(-1); prev = None; out = []
        for t in ids:
            if t != 0 and t != prev: out.append(int(t))
            prev = t
        return ids, sp.decode(out).lower()
    stats = {T: [0, 0, 0, 0] for T in sess if T != Tmax}                 # clips, same text, frames equal, frames
    for p in wavs:
        w, sr = sf.read(p, dtype="float32"); f = fbank(w if w.ndim == 1 else w.mean(1)); ref_ids, ref = run(Tmax, f)
        for T in stats:
            if len(f) > T: continue
            ids, hyp = run(T, f); s = stats[T]; s[0] += 1; s[1] += hyp == ref; s[2] += int((ids == ref_ids).sum()); s[3] += len(ref_ids)
    for T, (c, same, eq, tot) in sorted(stats.items()):
        print(f"  window {T:5d}: {c} clips fit, identical transcript {same}/{c}, frame argmax agreement {eq / max(tot, 1):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", default="300,500,800,1200,1500"); ap.add_argument("--encoder", default="encoder-epoch-11-avg-2.onnx")
    ap.add_argument("--with-encoder-out", action="store_true"); ap.add_argument("--check-wavs", default="")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    W, b = ctc_head(a.model_dir); paths = {}
    for T in sorted(int(x) for x in a.frames.split(",")):
        m = build(os.path.join(a.model_dir, a.encoder), W, b, T, a.with_encoder_out)
        dst = os.path.join(a.out_dir, f"zip150_{'ctc_rnnt' if a.with_encoder_out else 'ctc'}_static_{T}.onnx"); onnx.save(m, dst); paths[T] = dst
        print(f"saved {dst} ({os.path.getsize(dst) / 1e6:.0f} MB), window {T} frames = {T / 100:.1f} s", flush=True)
    if a.check_wavs: check(paths, a.check_wavs, os.path.join(a.model_dir, "bpe.model"))


if __name__ == "__main__":
    main()

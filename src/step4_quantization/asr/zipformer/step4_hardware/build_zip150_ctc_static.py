# -*- coding: utf-8 -*-
"""ZipFormer-150M-CR-CTC-RNNT-6000h -> encoder + head CTC (KHONG train, KHONG decoder/joiner) -> ONNX tinh 1500 frame.
Buoc: (1) va 3 Slice kieu bool bang Cast(bool->int32) (cach da dung cho 30M) -> (2) cat encoder_proj (256/768->512)
va gan MatMul+Add cua head CTC (ctc_output.1 tu pretrained.pt) -> dau ra ctc_logits (1,T',2000) -> (3) pin x=(1,1500,80),
x_lens=(1,) -> (4) kiem chung ORT: logits tinh (dem 0 toi 1500 frame) == logits dong, va WER CTC tren 15 mau.
Chay trong tools/venv_nllb_export."""
import json, os, re, sys, time
import numpy as np, onnx, onnxruntime as ort, soundfile as sf, jiwer
from onnx import helper, numpy_helper, TensorProto
import sentencepiece as spm
import kaldi_native_fbank as knf

ROOT = r"C:\Users\Admin\Downloads\OneVoice"; os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src", "step4_hardware"))
from prepare_zipformer_for_qnn import find_bool_slices, wrap_bool_slice
D = "third_party_zipformer_150m_crctc"; OUT = os.path.join(ROOT, "outputs", "zip150_ctc"); os.makedirs(OUT, exist_ok=True)
T = 1500
norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip(), flags=re.UNICODE)).strip()
sp = spm.SentencePieceProcessor(); sp.load(os.path.join(D, "bpe.model"))
hd = np.load(os.path.join(D, "ctc_head.npz")); W, b = hd["W"].astype(np.float32), hd["b"].astype(np.float32)   # (2000,768)

m = onnx.load(os.path.join(D, "encoder-epoch-11-avg-2.onnx"))
print("nodes", len(m.graph.node), flush=True)

# (1) va Slice bool
bs = find_bool_slices(m); print("bool Slice:", bs, flush=True)
for n in bs: wrap_bool_slice(m.graph, n)
assert not find_bool_slices(m), "con Slice bool"

# (2) cat encoder_proj, gan head CTC
byout = {o: n for n in m.graph.node for o in n.output}
mm, ad = byout["/encoder_proj/MatMul_output_0"], byout["encoder_out"]
assert mm.op_type == "MatMul" and ad.op_type == "Add"
pre = mm.input[0]; assert pre == "/Transpose_1_output_0", pre
proj_inits = {mm.input[1], "encoder_proj.bias"}
users = {i for n in m.graph.node if n not in (mm, ad) for i in n.input}
assert not (proj_inits & users), "trong so encoder_proj con dung cho noi khac"
m.graph.node.remove(mm); m.graph.node.remove(ad)
for i in [x for x in m.graph.initializer if x.name in proj_inits]: m.graph.initializer.remove(i)
m.graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(W.T), "ctc_head.W"))     # (768,2000)
m.graph.initializer.append(numpy_helper.from_array(b, "ctc_head.b"))
m.graph.node.append(helper.make_node("MatMul", [pre, "ctc_head.W"], ["ctc_head/mm"], name="ctc_head/MatMul"))
m.graph.node.append(helper.make_node("Add", ["ctc_head/mm", "ctc_head.b"], ["ctc_logits"], name="ctc_head/Add"))
del m.graph.output[:]
m.graph.output.append(helper.make_tensor_value_info("ctc_logits", TensorProto.FLOAT, ["N", "T", 2000]))
m.graph.output.append(helper.make_tensor_value_info("encoder_out_lens", TensorProto.INT64, ["N"]))

# (3) pin shape
for inp in m.graph.input:
    dims = inp.type.tensor_type.shape.dim
    if inp.name == "x":
        for d, v in zip(dims, (1, T, 80)): d.ClearField("dim_param"); d.dim_value = v
    elif inp.name == "x_lens":
        dims[0].ClearField("dim_param"); dims[0].dim_value = 1
dst = os.path.join(OUT, f"zip150_ctc_static_{T}.onnx"); onnx.save(m, dst)
print("saved", dst, f"{os.path.getsize(dst)/1e6:.0f}MB", "| nodes", len(m.graph.node), flush=True)

# (4) kiem chung
def fbank(wav):
    o = knf.FbankOptions(); o.mel_opts.num_bins = 80; o.frame_opts.samp_freq = 16000; o.frame_opts.dither = 0.0
    o.frame_opts.snip_edges = False; o.mel_opts.low_freq = 20.0; o.mel_opts.high_freq = -400.0
    f = knf.OnlineFbank(o); f.accept_waveform(16000, wav.tolist()); f.input_finished()
    return np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)

def collapse(ids):
    out, prev = [], None
    for t in ids:
        if t != 0 and t != prev: out.append(int(t))
        prev = t
    return sp.decode(out)

rows = [r for r in json.load(open("data/asr/manifest.json", encoding="utf-8")) if r["lang"] == "vi"]
items = []
for tag in ("clean", "snr5", "snr0"):
    for r in rows:
        f = r["path"] if tag == "clean" else r["path"].replace("data/asr/vi/", "data/asr_mixed/vi/").replace(".wav", f"_{tag}.wav")
        if os.path.exists(f): items.append((tag, r["transcript"], fbank(sf.read(f, dtype="float32")[0])))
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
res = {}
for tag, ref, feats in items:
    n = min(len(feats), T); x = np.zeros((1, T, 80), np.float32); x[0, :n] = feats[:n]
    lg, el = sess.run(None, {"x": x, "x_lens": np.array([n], np.int64)})
    real = int(el[0]); hyp = collapse(lg[0, :real].argmax(-1))
    res.setdefault(tag, []).append(jiwer.wer(norm(ref), norm(hyp)))
print("\nONNX TINH (dem 0 toi 1500 frame, x_lens that) — WER CTC:", {k: f"{np.mean(v)*100:.2f}% (n={len(v)})" for k, v in res.items()}, flush=True)
print("frames sau subsampling:", lg.shape, "| nhan: 1500 frame ~ 15 s")

import sys, os, json, collections, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from melo.api import TTS
from melo import utils
from melo_static import MeloStatic
N, Y = int(sys.argv[1]), int(sys.argv[2]); G = int(sys.argv[3]) if len(sys.argv) > 3 else 0
tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
for f in tts.model.flow.flows:
    if hasattr(f, "enc") and hasattr(f.enc, "remove_weight_norm"):
        try: f.enc.remove_weight_norm()
        except Exception: pass
S = MeloStatic(tts.model, N, Y, group=G).eval(); S_ref = MeloStatic(tts.model, N, Y, group=0).eval()
t = "今天天气真好，我们一起去公园散步吧。"; bert, ja, ph, tone, lang = utils.get_text_for_tts_infer(t, "ZH", tts.hps, "cpu", tts.symbol_to_id); n = len(ph)
pad = lambda v: torch.cat([v, torch.zeros(N - n, dtype=v.dtype)])[None].int(); bp = torch.zeros(1, 1024, N); bp[0, :, :n] = bert
args = (pad(ph), pad(tone), pad(lang), bp, torch.tensor([n], dtype=torch.int32))
with torch.no_grad(): want = S(*args)
path = f"work/melo_N{N}_Y{Y}_g{G}.onnx"
torch.onnx.export(S, args, path, input_names=["x", "tone", "lang", "bert", "x_len"], output_names=["audio", "y_len"], opset_version=17, do_constant_folding=True, dynamo=False)
m = onnx.load(path, load_external_data=False); print("size MB", os.path.getsize(path) // 10**6, "| nodes", len(m.graph.node), dict(collections.Counter(x.op_type for x in m.graph.node).most_common(12)))
bad = [x.op_type for x in m.graph.node if x.op_type in ("NonZero", "If", "Loop", "Range", "ScatterND", "GatherND")]; print("risky ops:", collections.Counter(bad))
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); got = s.run(None, {k: v.numpy() for k, v in zip(["x", "tone", "lang", "bert", "x_len"], args)})
with torch.no_grad(): full = S_ref(*args)
L = int(want[1][0]) * 512; a_ = want[0][0, :L].numpy().astype(np.float64); b_ = full[0][0, :L].numpy().astype(np.float64); print("windowed vs full-length decode: cos %.5f" % float(a_ @ b_ / np.linalg.norm(a_) / np.linalg.norm(b_)))
print("ORT vs torch: y_len", int(got[1][0]), int(want[1][0]), "| max abs audio diff", float(np.abs(got[0][0, :L] - want[0][0, :L].numpy()).max()))

import sys, os, json, collections, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForMaskedLM
from melo.api import TTS
from melo_static import MeloStatic
from melo_full import BertStatic, MeloFull, host_front
N, Y, L, G = 96, 384, 32, 5
tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
name = "hfl/chinese-roberta-wwm-ext-large"; tok = AutoTokenizer.from_pretrained(name); mlm = AutoModelForMaskedLM.from_pretrained(name).eval()
full = MeloFull(BertStatic(mlm, L), MeloStatic(tts.model, N, Y, group=G)).eval()
bi, bm, E, x, tone, lang, xl = host_front(tts, tok, "今天天气真好，我们一起去公园散步吧。", N, L)
args = tuple(torch.from_numpy(a) for a in (bi, bm, E, x, tone, lang, xl)); names = ["bert_ids", "bert_mask", "E", "x", "tone", "lang", "x_len"]
with torch.no_grad(): want = full(*args)
path = "work/melo_full_N96_Y384_g5.onnx"
torch.onnx.export(full, args, path, input_names=names, output_names=["audio", "y_len"], opset_version=17, do_constant_folding=True, dynamo=False)
m = onnx.load(path, load_external_data=False); print("size GB %.2f" % (os.path.getsize(path) / 1e9), "| nodes", len(m.graph.node), "| external tensors", sum(1 for i in m.graph.initializer if i.external_data))
print("risky ops:", collections.Counter(n.op_type for n in m.graph.node if n.op_type in ("NonZero", "If", "Loop", "Range", "ScatterND", "GatherND")))
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); got = s.run(None, {k: a.numpy() for k, a in zip(names, args)}); L_ = int(want[1][0]) * 512
print("ORT vs torch: y_len", int(got[1][0]), int(want[1][0]), "| max abs diff", float(np.abs(got[0][0, :L_] - want[0][0, :L_].numpy()).max()))

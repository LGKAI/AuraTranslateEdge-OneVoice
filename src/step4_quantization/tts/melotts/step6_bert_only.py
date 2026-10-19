import sys, json, numpy as np, torch, onnx
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForMaskedLM
from melo.api import TTS
from melo_full import BertStatic, host_front
N, L = 96, 32; tts = TTS(language="ZH", device="cpu"); name = "hfl/chinese-roberta-wwm-ext-large"; tok = AutoTokenizer.from_pretrained(name); mlm = AutoModelForMaskedLM.from_pretrained(name).eval()
b = BertStatic(mlm, L).eval(); SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
ids, ms = [], []
for t in SENT: bi, bm, *_ = host_front(tts, tok, t, N, L); ids.append(bi[0]); ms.append(bm[0])
np.savez("work/bert_inputs.npz", ids=np.stack(ids)[:, None, :].reshape(len(ids), 1, L).squeeze(1)[:, None, :].reshape(len(ids), 1, L)[:, 0, :][:, :], mask=np.stack(ms))
d = np.load("work/bert_inputs.npz"); np.savez("work/bert_inputs.npz", ids=d["ids"].reshape(len(ids), 1, L).astype(np.int32), mask=np.stack(ms).reshape(len(ids), 1, L).astype(np.float32))
a = (torch.from_numpy(np.stack(ids)[:1].astype(np.int32)), torch.from_numpy(np.stack(ms)[:1]))
torch.onnx.export(b, a, "work/bert_L32.onnx", input_names=["ids", "mask"], output_names=["hidden"], opset_version=17, do_constant_folding=True, dynamo=False)
m = onnx.load("work/bert_L32.onnx", load_external_data=False); import os, collections; print("bert size GB %.2f nodes %d" % (os.path.getsize("work/bert_L32.onnx") / 1e9, len(m.graph.node)), dict(collections.Counter(n.op_type for n in m.graph.node).most_common(8)))

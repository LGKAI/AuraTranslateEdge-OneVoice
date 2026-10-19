# VITS graph that takes BERT hidden states h [L,1024] + expansion matrix E [L,N] (BERT is a separate graph)
import sys, os, json, numpy as np, torch, onnx, onnxruntime as ort
sys.path.insert(0, "MeloTTS"); sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForMaskedLM
from melo.api import TTS
from melo_static import MeloStatic
from melo_full import BertStatic, host_front
class VitsHE(torch.nn.Module):
    def __init__(s, v): super().__init__(); s.v = v
    def forward(s, h, E, x, tone, lang, x_len): return s.v(x, tone, lang, torch.matmul(h.t(), E).unsqueeze(0), x_len)
N, Y, L = 96, 384, 32; tts = TTS(language="ZH", device="cpu"); tts.model.dec.remove_weight_norm()
name = "hfl/chinese-roberta-wwm-ext-large"; tok = AutoTokenizer.from_pretrained(name); mlm = AutoModelForMaskedLM.from_pretrained(name).eval(); bst = BertStatic(mlm, L).eval()
M = VitsHE(MeloStatic(tts.model, N, Y, group=5)).eval(); SENT = [r["text"] for r in json.load(open("work/cpu_baseline.json", encoding="utf-8"))]
names = ["h", "E", "x", "tone", "lang", "x_len"]; D = {k: [] for k in names}
for t in SENT:
    bi, bm, E, x, tone, lang, xl = host_front(tts, tok, t, N, L)
    with torch.no_grad(): h = bst(torch.from_numpy(bi), torch.from_numpy(bm))[0].numpy()
    for k, v in zip(names, [h, E, x, tone, lang, xl]): D[k].append(v)
np.savez("work/vits_hE_inputs.npz", **{k: np.stack(v) for k, v in D.items()})
args = tuple(torch.from_numpy(D[k][0]) for k in names)
torch.onnx.export(M, args, "work/melo_vits_hE.onnx", input_names=names, output_names=["audio", "y_len"], opset_version=17, do_constant_folding=True, dynamo=False)
s = ort.InferenceSession("work/melo_vits_hE.onnx", providers=["CPUExecutionProvider"]); got = s.run(None, {k: D[k][0] for k in names})
with torch.no_grad(): want = M(*args)
L_ = int(want[1][0]) * 512; print("exported %.0f MB | ORT vs torch max diff %.2e | y_len %d/%d" % (os.path.getsize("work/melo_vits_hE.onnx") / 1e6, float(np.abs(got[0][0, :L_] - want[0][0, :L_].numpy()).max()), int(got[1][0]), int(want[1][0])))

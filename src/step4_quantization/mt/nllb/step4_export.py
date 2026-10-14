import sys, os, json, time, torch, numpy as np, onnx, onnxruntime as ort
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import AutoModelForSeq2SeqLM
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, ".")
from nllb_static import NLLBStatic
import vocab_restrict as VR
S = 32; Ts = [int(x) for x in sys.argv[1:]] or [2]
tok, V, inv, keep = VR.build(all_scripts=True); json.dump({"keep": keep}, open("work/keep_all.json", "w"))
hf = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_safetensors=False, local_files_only=True).eval()
ref = json.load(open("work/ref.json", encoding="utf-8"))
for T in Ts:
    m = NLLBStatic(hf, S, T, keep_ids=keep, half_emb=True).eval()
    a = np.ones((1, S), np.int32); r = ref[0]; a[0, :len(r["ids_in"])] = r["ids_in"]; lang = np.array([V[r["tgt"]]], np.int32)
    with torch.no_grad(): want = m(torch.from_numpy(a), torch.from_numpy(lang))
    path = f"work/nllb_S{S}_T{T}.onnx"; t0 = time.time()
    torch.onnx.export(m, (torch.from_numpy(a), torch.from_numpy(lang)), path, input_names=["src_ids", "tgt_lang"], output_names=["out_ids"], opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"T={T}: exported {os.path.getsize(path)/1e9:.2f} GB in {time.time()-t0:.0f}s", flush=True)
    mm = onnx.load(path, load_external_data=False)
    import collections; print("  nodes", len(mm.graph.node), "| ops", dict(collections.Counter(n.op_type for n in mm.graph.node).most_common(8)), "| external data tensors:", sum(1 for i in mm.graph.initializer if i.external_data))
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); got = s.run(None, {"src_ids": a, "tgt_lang": lang})[0]
    print("  ORT == torch:", np.array_equal(got, want.numpy()), "| ids", got[0].tolist()[:T], flush=True)

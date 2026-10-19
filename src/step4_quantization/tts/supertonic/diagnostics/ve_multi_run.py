import sys, os, json, numpy as np, onnxruntime as ort, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8")
d=np.load("ve_samples.npz"); IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
idx=[j for j in range(len(d["sent"])) if int(d["sent"][j])==0 and int(d["step"][j]) in (0,4)]; names=json.load(open("ve_multi_names.json"))
dev=hub.Device("Dragonwing IQ-9075 EVK"); cj=hub.submit_compile_job(model="ve_multi.onnx",device=dev,options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",name="ve_multi"); cj.wait(); print("compile",cj.get_status().code,(cj.get_status().message or "")[:200],flush=True)
if str(cj.get_status().code)!="SUCCESS": sys.exit(1)
ij=hub.submit_inference_job(model=cj.get_target_model(),device=dev,inputs={k:[d[k][j] for j in idx] for k in IN},name="ve_multi_infer"); ij.wait(); print("infer",ij.get_status().code,flush=True)
od=ij.download_output_data(); ks=sorted(od.keys(),key=lambda s:int(s.split("_")[-1])); s=ort.InferenceSession("ve_multi.onnx",providers=["CPUExecutionProvider"])
for pos,j in enumerate(idx):
    ref=s.run(None,{x:d[x][j] for x in IN}); print("--- step",int(d["step"][j]))
    for oi,k in enumerate(ks):
        r=np.asarray(ref[oi]).astype(np.float64); a=np.asarray(od[k][pos]).reshape(r.shape).astype(np.float64)
        print(f"  {names[oi]:10s} shape {list(r.shape)} cpu[{r.min():.3f},{r.max():.3f}] npu[{a.min():.3f},{a.max():.3f}] cos {float(r.ravel()@a.ravel()/(np.linalg.norm(r)*np.linalg.norm(a)+1e-12)):.5f} first {r.ravel()[:2].round(3).tolist()} / {a.ravel()[:2].round(3).tolist()}",flush=True)

# -*- coding: utf-8 -*-
"""Fine bisect inside main_blocks.3/attn. python ve_cut2.py make | run <i> | report"""
import sys, os, json, numpy as np, onnx, onnxruntime as ort
from onnx import utils
sys.stdout.reconfigure(encoding="utf-8")
HERE=(os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); VE=os.path.join(HERE,"static_N96_L150","vector_estimator_static.onnx"); cpu=["CPUExecutionProvider"]
IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
P="/vector_estimator/vector_field/main_blocks.3/attn/"
WANT=["Concat_3","Concat_4","MatMul","Div_4","Where","Softmax","Where_1","MatMul_1"]
if sys.argv[1]=="make":
    m=onnx.load(VE); by={n.name:n for n in m.graph.node}; cuts=[]
    for w in WANT:
        n=by[P+w]; cuts.append((w,n.output[0]))
    # block output: out_fc + residual (Mul_14 last)
    cuts.append(("Mul_14(block3 attn out)",by[P+"Mul_14"].output[0]))
    for i,(w,t) in enumerate(cuts):
        utils.extract_model(VE,os.path.join(HERE,f"ve_c2_{i}.onnx"),IN,[t]); mm=onnx.load(os.path.join(HERE,f"ve_c2_{i}.onnx")); del mm.graph.value_info[:]; onnx.save(mm,os.path.join(HERE,f"ve_c2_{i}.onnx"))
    json.dump(cuts,open(os.path.join(HERE,"ve_c2_cuts.json"),"w")); print(len(cuts),"cuts")
elif sys.argv[1]=="run":
    import qai_hub as hub
    i=int(sys.argv[2]); d=np.load(os.path.join(HERE,"ve_samples.npz")); idx=[j for j in range(len(d["sent"])) if int(d["sent"][j])==0 and int(d["step"][j]) in (0,4)]
    dev=hub.Device("Dragonwing IQ-9075 EVK"); cj=hub.submit_compile_job(model=os.path.join(HERE,f"ve_c2_{i}.onnx"),device=dev,options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",name=f"ve_c2_{i}"); print("compile",cj.job_id,flush=True)
    cj.wait(); st=cj.get_status(); print("compile",st.code,(st.message or "")[:200],flush=True)
    if str(st.code)!="SUCCESS": sys.exit(1)
    ij=hub.submit_inference_job(model=cj.get_target_model(),device=dev,inputs={k:[d[k][j] for j in idx] for k in IN},name=f"ve_c2_{i}_infer"); ij.wait(); print("infer",ij.get_status().code,flush=True)
    od=ij.download_output_data(); k=sorted(od.keys())[0]; np.save(os.path.join(HERE,f"ve_c2_{i}_hw.npy"),np.stack([np.asarray(a) for a in od[k]])); print("saved",flush=True)
else:
    d=np.load(os.path.join(HERE,"ve_samples.npz")); idx=[j for j in range(len(d["sent"])) if int(d["sent"][j])==0 and int(d["step"][j]) in (0,4)]; cuts=json.load(open(os.path.join(HERE,"ve_c2_cuts.json")))
    for i,(w,t) in enumerate(cuts):
        p=os.path.join(HERE,f"ve_c2_{i}_hw.npy")
        if not os.path.exists(p): print(i,w,"(no hw)"); continue
        h=np.load(p); s=ort.InferenceSession(os.path.join(HERE,f"ve_c2_{i}.onnx"),providers=cpu); o=[]
        for pos,j in enumerate(idx):
            r=s.run(None,{x:d[x][j] for x in IN})[0].astype(np.float64).ravel(); a=h[pos].astype(np.float64).ravel(); o.append(float(r@a/(np.linalg.norm(r)*np.linalg.norm(a)+1e-12)))
        print(f"cut {i} {w:26s} cos step0 {o[0]:.5f} step4 {o[1]:.5f}")

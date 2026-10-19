# -*- coding: utf-8 -*-
"""Prefix sub-models of the static VE (single output each). python ve_cut.py make | run <i> | report"""
import sys, os, json, numpy as np, onnx, onnxruntime as ort
from onnx import shape_inference, utils
sys.stdout.reconfigure(encoding="utf-8")
HERE=(os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); VE=os.path.join(HERE,"static_N96_L150","vector_estimator_static.onnx"); cpu=["CPUExecutionProvider"]
IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
FR=[0.12,0.3,0.5,0.7,0.88]
if sys.argv[1]=="make":
    m=shape_inference.infer_shapes(onnx.load(VE)); info={v.name:v for v in m.graph.value_info}; cuts=[]
    for f in FR:
        k=int(f*len(m.graph.node))
        while k<len(m.graph.node):
            o=[x for x in m.graph.node[k].output if x in info and info[x].type.tensor_type.elem_type==1 and len(info[x].type.tensor_type.shape.dim)==3]
            if o: cuts.append((k,m.graph.node[k].op_type,o[0])); break
            k+=1
    for i,(k,op,t) in enumerate(cuts): utils.extract_model(VE,os.path.join(HERE,f"ve_cut{i}.onnx"),IN,[t])
    json.dump(cuts,open(os.path.join(HERE,"ve_cuts.json"),"w")); print(cuts)
elif sys.argv[1]=="run":
    import qai_hub as hub
    i=int(sys.argv[2]); d=np.load(os.path.join(HERE,"ve_samples.npz")); idx=[j for j in range(len(d["sent"])) if int(d["sent"][j])==0 and int(d["step"][j]) in (0,4)]
    dev=hub.Device("Dragonwing IQ-9075 EVK"); cj=hub.submit_compile_job(model=os.path.join(HERE,f"ve_cut{i}.onnx"),device=dev,options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",name=f"ve_cut{i}"); print("compile",cj.job_id,flush=True)
    cj.wait(); st=cj.get_status(); print("compile",st.code,(st.message or "")[:200],flush=True)
    if str(st.code)!="SUCCESS": sys.exit(1)
    ij=hub.submit_inference_job(model=cj.get_target_model(),device=dev,inputs={k:[d[k][j] for j in idx] for k in IN},name=f"ve_cut{i}_infer"); ij.wait(); print("infer",ij.get_status().code,flush=True)
    od=ij.download_output_data(); k=sorted(od.keys())[0]; np.save(os.path.join(HERE,f"ve_cut{i}_hw.npy"),np.stack([np.asarray(a) for a in od[k]])); print("saved",flush=True)
else:
    d=np.load(os.path.join(HERE,"ve_samples.npz")); idx=[j for j in range(len(d["sent"])) if int(d["sent"][j])==0 and int(d["step"][j]) in (0,4)]; cuts=json.load(open(os.path.join(HERE,"ve_cuts.json")))
    for i,(k,op,t) in enumerate(cuts):
        p=os.path.join(HERE,f"ve_cut{i}_hw.npy")
        if not os.path.exists(p): print(i,k,op,"(no hw result)"); continue
        h=np.load(p); s=ort.InferenceSession(os.path.join(HERE,f"ve_cut{i}.onnx"),providers=cpu); out=[]
        for pos,j in enumerate(idx):
            r=s.run(None,{x:d[x][j] for x in IN})[0].astype(np.float64).ravel(); a=h[pos].astype(np.float64).ravel(); out.append(float(r@a/(np.linalg.norm(r)*np.linalg.norm(a)+1e-12)))
        print(f"cut {i} node {k:4d} {op:10s} {t[-50:]:50s} cos step0 {out[0]:.5f} step4 {out[1]:.5f}")

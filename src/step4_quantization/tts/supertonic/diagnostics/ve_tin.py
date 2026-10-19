# -*- coding: utf-8 -*-
"""Make the time-encoder branch (sin/cos of 1000*t, unrepresentable in fp16/int16) a precomputed INPUT/constant.
python ve_tin.py prep   -> ve_tin.onnx (time_emb replaces current_step/total_step), time_embs.npy (5 steps, fp32)
python ve_tin.py run    -> compile fp16 on NPU + one-step cos per ODE step"""
import sys, os, json, time, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto as TP, numpy_helper
sys.stdout.reconfigure(encoding="utf-8")
HERE=(os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); VE=os.path.join(HERE,"static_N96_L150","vector_estimator_static.onnx"); cpu=["CPUExecutionProvider"]
T="/vector_estimator/vector_field/time_encoder/Unsqueeze_output_0"
IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","total_step","time_emb"]
def prep():
    m=onnx.load(VE); g=m.graph
    cons={}; prod={}
    for n in g.node:
        for i in n.input: cons.setdefault(i,set()).add(id(n))
        for o in n.output: prod[o]=n
    outs={o.name for o in g.output}; rem={id(prod[T])}; changed=True
    byid={id(n):n for n in g.node}
    while changed:
        changed=False
        for n in g.node:
            if id(n) in rem: continue
            if any(o in outs for o in n.output): continue
            c=set().union(*[cons.get(o,set()) for o in n.output]) if n.output else set()
            if c and c<=rem: rem.add(id(n)); changed=True
    keep=[n for n in g.node if id(n) not in rem]; dropped=len(g.node)-len(keep)
    # time_emb value: run full model, exposing T
    m2=onnx.load(VE); vi=helper.make_tensor_value_info(T,TP.FLOAT,None); m2.graph.output.append(vi); s=ort.InferenceSession(m2.SerializeToString(),providers=cpu)
    d=np.load(os.path.join(HERE,"ve_samples.npz")); te=[]
    for st in range(5):
        j=[k for k in range(len(d["step"])) if int(d["step"][k])==st][0]
        te.append(s.run([T],{k:d[k][j] for k in ["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]})[0])
    np.save(os.path.join(HERE,"time_embs.npy"),np.stack(te)); shp=list(te[0].shape)
    # new model: T is a graph input; drop old inputs current_step / total_step if unused
    del g.node[:]; g.node.extend(keep)
    used={i for n in g.node for i in n.input}
    for nm in ("current_step","total_step"):
        if nm not in used:
            for x in list(g.input):
                if x.name==nm: g.input.remove(x)
    g.input.append(helper.make_tensor_value_info(T,TP.FLOAT,shp))
    # rename the input to time_emb
    for n in g.node:
        n.input[:]=["time_emb" if i==T else i for i in n.input]
    g.input[-1].name="time_emb"
    onnx.save(m,os.path.join(HERE,"ve_tin.onnx")); print("dropped",dropped,"time nodes; time_emb shape",shp,"| inputs",[x.name for x in g.input])
    # CPU check
    s2=ort.InferenceSession(os.path.join(HERE,"ve_tin.onnx"),providers=cpu); cs=[]
    for k in range(len(d["ref"])):
        y=s2.run(None,{**{x:d[x][k] for x in IN[:-1]},"time_emb":np.stack(te)[int(d["step"][k])]})[0]; r=d["ref"][k]; cs.append(float(np.dot(y.ravel(),r.ravel())/(np.linalg.norm(y)*np.linalg.norm(r))))
    print("CPU fp32 ve_tin vs original: min cos",min(cs))
def run():
    import qai_hub as hub
    d=np.load(os.path.join(HERE,"ve_samples.npz")); te=np.load(os.path.join(HERE,"time_embs.npy")); dev=hub.Device("Dragonwing IQ-9075 EVK")
    idx=[j for j in range(len(d["sent"])) if int(d["sent"][j]) in (0,2,6,9)]
    cj=hub.submit_compile_job(model=os.path.join(HERE,"ve_tin.onnx"),device=dev,options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",name="ve_tin_compile"); print("compile",cj.job_id,flush=True)
    cj.wait(); st=cj.get_status(); print("compile",st.code,(st.message or "")[:200],flush=True)
    if str(st.code)!="SUCCESS": return
    inp={x:[d[x][j] for j in idx] for x in IN[:-1]}; inp["time_emb"]=[te[int(d["step"][j])] for j in idx]
    ij=hub.submit_inference_job(model=cj.get_target_model(),device=dev,inputs=inp,name="ve_tin_infer"); print("infer",ij.job_id,flush=True); ij.wait(); st=ij.get_status(); print("infer",st.code,flush=True)
    od=ij.download_output_data(); k=sorted(od.keys(),key=lambda s:int(s.split("_")[-1]))[0]; cs={s:[] for s in range(5)}
    for jj,j in enumerate(idx):
        y=np.asarray(od[k][jj]).reshape(d["ref"][j].shape); r=d["ref"][j]; cs[int(d["step"][j])].append(float(np.dot(y.ravel(),r.ravel())/(np.linalg.norm(y)*np.linalg.norm(r))))
    print("ONE-STEP cos vs CPU fp32 (time_emb precomputed):",{s:round(float(np.mean(v)),5) for s,v in cs.items()},flush=True)
prep() if sys.argv[1]=="prep" else run()

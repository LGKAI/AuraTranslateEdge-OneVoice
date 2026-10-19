import sys, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto as TP
sys.stdout.reconfigure(encoding="utf-8")
IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
d=np.load("ve_samples.npz"); idx=[j for j in range(len(d["sent"])) if int(d["sent"][j]) in (0,2,6,9)]
def variant(round_in, round_out, only_rot=True):
    m=onnx.load("static_N96_L150/vector_estimator_static.onnx"); new=[]; k=0
    for n in m.graph.node:
        if n.op_type in ("Sin","Cos") and (not only_rot or "time_encoder" not in n.name):
            x=n.input[0]
            if round_in:
                a,b=f"rin{k}a",f"rin{k}b"; new+=[helper.make_node("Cast",[x],[a],to=TP.FLOAT16),helper.make_node("Cast",[a],[b],to=TP.FLOAT)]; n.input[0]=b
            if round_out:
                o=n.output[0]; n.output[0]=f"rout{k}"; new.append(n); a=f"rout{k}h"
                new+=[helper.make_node("Cast",[f"rout{k}"],[a],to=TP.FLOAT16),helper.make_node("Cast",[a],[o],to=TP.FLOAT)]; k+=1; continue
            k+=1
        new.append(n)
    del m.graph.node[:]; m.graph.node.extend(new); return ort.InferenceSession(m.SerializeToString(),providers=["CPUExecutionProvider"]),k
for name,(ri,ro,only) in {"rotary angle->fp16":(True,False,True),"rotary sin/cos out->fp16":(False,True,True),"time-encoder angle->fp16":(True,False,False)}.items():
    s,k=variant(ri,ro,only); cs={st:[] for st in range(5)}
    for j in idx:
        y=s.run(None,{x:d[x][j] for x in IN})[0]; r=d["ref"][j]; cs[int(d["step"][j])].append(float(np.dot(y.ravel(),r.ravel())/(np.linalg.norm(y)*np.linalg.norm(r))))
    print(f"{name:28s} ({k} sin/cos nodes) per-step cos:",{s_:round(float(np.mean(v)),5) for s_,v in cs.items()})
print("NPU measured                                 per-step cos: {0: 0.99765, 1: 0.9775, 2: 0.95602, 3: 0.94806, 4: 0.9407}")

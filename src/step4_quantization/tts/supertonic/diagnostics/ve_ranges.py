import sys, os, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto as TP
sys.stdout.reconfigure(encoding="utf-8")
HERE=(os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); VE=os.path.join(HERE,"static_N96_L150","vector_estimator_static.onnx")
IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
from onnx import shape_inference
m=shape_inference.infer_shapes(onnx.load(VE)); INFO={v.name:v.type.tensor_type.elem_type for v in m.graph.value_info}; names=[]
for n in m.graph.node:
    for o in n.output: names.append((o,n.op_type,n.name))
outs={o.name for o in m.graph.output}
for o,_,_ in names:
    if o not in outs and INFO.get(o)==1: m.graph.output.append(helper.make_tensor_value_info(o,TP.FLOAT,None))
del m.graph.value_info[:]
s=ort.InferenceSession(m.SerializeToString(),providers=["CPUExecutionProvider"]); d=np.load(os.path.join(HERE,"ve_samples.npz"))
j=[k for k in range(len(d["sent"])) if int(d["sent"][k])==0 and int(d["step"][k])==4][0]
res=s.run(None,{k:d[k][j] for k in IN}); onames=[o.name for o in s.get_outputs()]; R=dict(zip(onames,res))
rows=[]
for o,op,nm in names:
    if o in R and R[o].dtype==np.float32 and R[o].size>1:
        a=np.abs(R[o]); rows.append((float(a.max()),op,o))
rows.sort(reverse=True); print("top 15 tensors by max|x| (fp32 CPU, step 4):")
for mx,op,o in rows[:15]: print(f"  {mx:12.1f} {op:20s} {o[-70:]}")
# LayerNorm inputs & squares
ln=[(n.input[0],n.name) for n in m.graph.node if n.op_type=="LayerNormalization"]
print("LayerNormalization nodes:",len(ln))
mx=[(float(np.abs(R[i]).max()),float((R[i].astype(np.float64)**2).mean(axis=-1).max()),nm[-55:]) for i,nm in ln if i in R]
mx.sort(reverse=True)
for a,v,nm in mx[:6]: print(f"  LN input max|x| {a:9.1f}  max mean(x^2) {v:12.1f}  {nm}")
print("count fp16-overflow-risk tensors (max|x|>1000):",sum(1 for r in rows if r[0]>1000)," (>10000):",sum(1 for r in rows if r[0]>10000))

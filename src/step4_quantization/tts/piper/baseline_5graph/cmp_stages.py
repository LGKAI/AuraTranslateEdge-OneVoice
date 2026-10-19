import json, numpy as np, onnxruntime as ort, qai_hub as hub, sys
L=json.load(open("hw_log_hw_t2.json")); J={r["stage"]:r["job"] for r in L}
C="outputs/piper_vi_npu/components/"
def sess(f): return ort.InferenceSession(C+f, providers=["CPUExecutionProvider"])
def cos(a,b):
    a=np.asarray(a,np.float64).ravel(); b=np.asarray(b,np.float64).ravel(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))
def io(stage):
    j=hub.get_job(J[stage]); return j.download_output_data(), None
# encoder: input from its job dataset
je=hub.get_job(J["encoder"]); ein=je.inputs.download()
print("encoder input keys", list(ein.keys()) if ein else None)
eo=je.download_output_data(); se=sess("piper_vi_encoder.onnx")
ref=se.run(None,{k:np.asarray(v[0]) for k,v in ein.items()})
for i,(n,r) in enumerate(zip([o.name for o in se.get_outputs()],ref)): print("encoder",n,"cos",round(cos(r,eo["output_%d"%i][0]),6))
# sdp: feed NPU-encoder outputs to both
js=hub.get_job(J["sdp"]); sin=js.inputs.download(); so=js.download_output_data(); ss=sess("piper_vi_sdp.onnx")
ref=ss.run(None,{k:np.asarray(v[0]) for k,v in sin.items()})
for i,(n,r) in enumerate(zip([o.name for o in ss.get_outputs()],ref)): print("sdp",n,"cos",round(cos(r,so["output_%d"%i][0]),6),"ref sum",float(np.asarray(r).sum()),"npu sum",float(np.asarray(so["output_%d"%i][0]).sum()))
print("sdp inputs", {k:np.asarray(v[0]).ravel()[:3] for k,v in sin.items() if np.asarray(v[0]).size<4})
# flow
jf=hub.get_job(J["flow"]); fin=jf.inputs.download(); fo=jf.download_output_data(); sf=sess("piper_vi_flow.onnx")
ref=sf.run(None,{k:np.asarray(v[0]) for k,v in fin.items()})
print("flow cos",round(cos(ref[0],list(fo.values())[0][0]),6))
# decoder: all 7 windows
jd=hub.get_job(J["decoder"]); din=jd.inputs.download(); do=jd.download_output_data(); sd=sess("piper_vi_decoder.onnx")
k=list(do.keys())[0]; cs=[]; pr=[];pn=[]
for i in range(len(din["z"])):
    r=sd.run(None,{"z":np.asarray(din["z"][i])})[0]; n=np.asarray(do[k][i]); cs.append(cos(r,n)); pr.append(float(np.abs(r).max())); pn.append(float(np.abs(n).max()))
print("decoder cos per window",[round(c,4) for c in cs]); print("peak ref",[round(p,3) for p in pr]); print("peak npu",[round(p,3) for p in pn])

# -*- coding: utf-8 -*-
"""Synthetic HTP checks of the failing rotary multiply. python mul_synth.py make | run <i> | report"""
import sys, os, json, numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto as TP
sys.stdout.reconfigure(encoding="utf-8")
HERE=(os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); L=150
rng=np.random.RandomState(0)
A=(rng.randn(1,8,2,L,32)*10).astype(np.float32)[0]                       # [8,2,150,32] like q slice
pos=np.arange(L,dtype=np.float32).reshape(1,L,1); freq=np.array([10*(0.1**(i/31)) for i in range(32)],np.float32).reshape(1,1,32)
ANG=np.repeat((pos/62.0)*freq,2,axis=0).astype(np.float32)                 # [2,150,32], up to 24
ANGS=np.clip(ANG,0,9.8).astype(np.float32)                                 # valid-region-only range
def g(name,nodes,ins,out):
    vi=lambda n,s: helper.make_tensor_value_info(n,TP.FLOAT,s)
    m=helper.make_model(helper.make_graph(nodes,name,[vi(n,s) for n,s in ins],[vi(out[0],out[1])]),opset_imports=[helper.make_opsetid("",17)]); m.ir_version=9
    onnx.checker.check_model(m); onnx.save(m,os.path.join(HERE,f"{name}.onnx"))
if sys.argv[1]=="make":
    g("syn_t1",[helper.make_node("Mul",["a","b"],["y"])],[("a",[8,2,L,32]),("b",[2,L,32])],("y",[8,2,L,32]))                       # broadcast mul, b is an input
    g("syn_t2",[helper.make_node("Sin",["c"],["s"]),helper.make_node("Mul",["a","s"],["y"])],[("a",[8,2,L,32]),("c",[2,L,32])],("y",[8,2,L,32]))     # angle up to 24
    g("syn_t3",[helper.make_node("Sin",["c"],["s"]),helper.make_node("Mul",["a","s"],["y"])],[("a",[8,2,L,32]),("c",[2,L,32])],("y",[8,2,L,32]))     # angle <=9.8
    g("syn_t4",[helper.make_node("Mul",["a","b"],["y"])],[("a",[8,2,96,32]),("b",[2,96,32])],("y",[8,2,96,32]))                   # key-sized control
    print("made")
elif sys.argv[1]=="run":
    import qai_hub as hub
    i=int(sys.argv[2]); name=f"syn_t{i}"; dev=hub.Device("Dragonwing IQ-9075 EVK")
    if i==1: inp={"a":[A],"b":[np.cos(ANG)]}
    elif i==2: inp={"a":[A],"c":[ANG]}
    elif i==3: inp={"a":[A],"c":[ANGS]}
    else: inp={"a":[A[:,:,:96]],"b":[np.cos(ANG[:,:96])]}
    cj=hub.submit_compile_job(model=os.path.join(HERE,name+".onnx"),device=dev,options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",name=name); cj.wait()
    print(name,"compile",cj.get_status().code,flush=True)
    ij=hub.submit_inference_job(model=cj.get_target_model(),device=dev,inputs=inp,name=name+"_infer"); ij.wait(); print(name,"infer",ij.get_status().code,flush=True)
    od=ij.download_output_data(); k=sorted(od.keys())[0]; H=np.asarray(od[k][0]).astype(np.float64)
    if i==1: R=A.astype(np.float64)*np.cos(ANG)
    elif i in (2,3): R=A.astype(np.float64)*np.sin(inp["c"][0].astype(np.float64))
    else: R=A[:,:,:96].astype(np.float64)*np.cos(ANG[:,:96])
    H=H.reshape(R.shape); vs=slice(0,62) if R.shape[2]==L else slice(0,62)
    print(name,"cos all %.5f | valid-frames cos %.5f | max abs err %.3f (max|ref| %.2f)"%(float(R.ravel()@H.ravel()/np.linalg.norm(R)/np.linalg.norm(H)),float(R[:,:,vs].ravel()@H[:,:,vs].ravel()/np.linalg.norm(R[:,:,vs])/np.linalg.norm(H[:,:,vs])),float(np.abs(R-H).max()),float(np.abs(R).max())),flush=True)

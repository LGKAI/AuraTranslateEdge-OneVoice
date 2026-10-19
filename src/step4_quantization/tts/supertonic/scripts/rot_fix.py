# -*- coding: utf-8 -*-
"""Rewrite rotary Sin/Cos tables so Mul operands have equal rank (HTP mis-computes rank-4 x rank-3 broadcast here).
python rot_fix.py make <mode>   mode B: Unsqueeze(0) -> [1,2,L,32] ; mode A: Expand to [8,2,L,32]
python rot_fix.py run <name>    compile+infer the cut (Concat_3) or full VE and report cos"""
import sys, os, json, numpy as np, onnx, onnxruntime as ort
from onnx import helper, utils, numpy_helper, TensorProto as TP
sys.stdout.reconfigure(encoding="utf-8")
HERE=(os.environ.get("ST_WORK") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "work")); L=int(os.environ.get("VAR_L","150")); VE=os.path.join(HERE,f"static_N96_L{L}","vector_estimator_static.onnx")
IN=["noisy_latent","text_emb","style_ttl","latent_mask","text_mask","current_step","total_step"]
SMP=os.path.join(HERE,"ve_samples.npz" if L==150 else f"ve_samples_L{L}.npz")
def fix_table():
    """replace query-rotary Sin/Cos (angle = pos/len*freq, len = runtime integer) by a precomputed fp32 table selected with a one-hot of len."""
    m=onnx.load(VE); g=m.graph; by={n.name:n for n in g.node}; W={i.name:numpy_helper.to_array(i) for i in g.initializer}
    P="/vector_estimator/vector_field/main_blocks.3/attn/"
    len3=by[P+"Reshape"].output[0]; pos=W[by[P+"Div"].input[0]].astype(np.float32).reshape(-1); freq=W[by[P+"Mul"].input[1]].astype(np.float32).reshape(-1)
    n_pos=pos.shape[0]; F=freq.shape[0]
    COS=np.zeros((L,n_pos*F),np.float32); SIN=np.zeros((L,n_pos*F),np.float32)
    for ln in range(1,L+1):
        ang=(pos/np.float32(ln)).reshape(-1,1)*freq.reshape(1,-1); COS[ln-1]=np.cos(ang).ravel(); SIN[ln-1]=np.sin(ang).ravel()
    g.initializer.extend([numpy_helper.from_array(COS,"rt_cos"),numpy_helper.from_array(SIN,"rt_sin"),numpy_helper.from_array(np.arange(1,L+1,dtype=np.float32).reshape(1,1,L),"rt_grid"),
        numpy_helper.from_array(np.array(0.5,np.float32),"rt_half"),numpy_helper.from_array(np.array([2,n_pos,F],np.int64),"rt_shape")])
    cosn,sinn=by[P+"Cos"],by[P+"Sin"]; oc,os_=cosn.output[0],sinn.output[0]
    new=[]
    for n in g.node:
        if n is cosn or n is sinn: continue
        new.append(n)
        if n.name==by[P+"Reshape"].name:
            new+=[helper.make_node("Sub",[len3,"rt_grid"],["rt_d"]),helper.make_node("Abs",["rt_d"],["rt_a"]),helper.make_node("Less",["rt_a","rt_half"],["rt_b"]),helper.make_node("Cast",["rt_b"],["rt_oh"],to=TP.FLOAT),
                  helper.make_node("MatMul",["rt_oh","rt_cos"],["rt_cf"]),helper.make_node("Reshape",["rt_cf","rt_shape"],[oc]),
                  helper.make_node("MatMul",["rt_oh","rt_sin"],["rt_sf"]),helper.make_node("Reshape",["rt_sf","rt_shape"],[os_])]
    del g.node[:]; g.node.extend(new)
    # dead-node elimination
    need={o.name for o in g.output}; keep=[]
    for n in reversed(list(g.node)):
        if any(o in need for o in n.output): keep.append(n); need.update(i for i in n.input if i)
    keep.reverse(); del g.node[:]; g.node.extend(keep); del g.value_info[:]
    return m,2
def fix_slice():
    """replace the half-slices of q/k in rotary attention by MatMul with a constant 0/1 selection matrix (avoids Slice->Mul fusion bug)."""
    m=onnx.load(VE); g=m.graph; W={i.name:numpy_helper.to_array(i) for i in g.initializer}; new=[]; k=0
    S1=np.zeros((64,32),np.float32); S1[np.arange(32),np.arange(32)]=1; S2=np.zeros((64,32),np.float32); S2[32+np.arange(32),np.arange(32)]=1
    g.initializer.extend([numpy_helper.from_array(S1,"sel_lo"),numpy_helper.from_array(S2,"sel_hi")])
    for n in g.node:
        if n.op_type=="Slice" and "/attn/" in n.name and len(n.input)>=4 and all(x in W for x in n.input[1:4]):
            st,en,ax=[int(W[x].ravel()[0]) for x in n.input[1:4]]
            if ax==-1 and (st,en) in ((0,32),(32,64)):
                new.append(helper.make_node("MatMul",[n.input[0],"sel_lo" if st==0 else "sel_hi"],[n.output[0]],name=f"selmm{k}")); k+=1; continue
        new.append(n)
    del g.node[:]; g.node.extend(new); del g.value_info[:]; return m,k
def fix_reducesum():
    """ReduceSum(axes=[1,2]) over the tiled masks returns only the first elements on HTP (62 -> 1). Replace by Reshape + MatMul with ones."""
    from onnx import shape_inference
    m=onnx.load(VE); mi=shape_inference.infer_shapes(m); shp={v.name:[dd.dim_value for dd in v.type.tensor_type.shape.dim] for v in mi.graph.value_info}
    g=m.graph; W={i.name:numpy_helper.to_array(i) for i in g.initializer}; new=[]; k=0; info=[]
    for n in g.node:
        if n.op_type=="ReduceSum" and len(n.input)>=2 and n.input[1] in W and sorted(W[n.input[1]].ravel().tolist())==[1,2]:
            x=n.input[0]; sh=shp[x]; assert len(sh)==3 and sh[1]==1, sh; B,_,Lx=sh
            g.initializer.extend([numpy_helper.from_array(np.ones((Lx,1),np.float32),f"rs_ones{k}"),numpy_helper.from_array(np.array([B,Lx],np.int64),f"rs_sh{k}"),numpy_helper.from_array(np.array([B],np.int64),f"rs_o{k}")])
            new+=[helper.make_node("Reshape",[x,f"rs_sh{k}"],[f"rs_x{k}"]),helper.make_node("MatMul",[f"rs_x{k}",f"rs_ones{k}"],[f"rs_y{k}"]),helper.make_node("Reshape",[f"rs_y{k}",f"rs_o{k}"],[n.output[0]])]; info.append((n.name[-40:],sh)); k+=1
        else: new.append(n)
    print("ReduceSum replaced:",info)
    del g.node[:]; g.node.extend(new); del g.value_info[:]; return m,k
def fix(mode):
    if mode=="R": return fix_reducesum()
    if mode=="S": return fix_slice()
    if mode=="T": return fix_table()
    m=onnx.load(VE); new=[]; k=0
    for n in m.graph.node:
        if n.op_type in ("Sin","Cos") and "time_encoder" not in n.name:
            o=n.output[0]; raw=f"rotraw{k}"; n.output[0]=raw; new.append(n)
            if mode=="B":
                new.append(helper.make_node("Unsqueeze",[raw,"rf_ax0"],[o],name=f"rotfix{k}"))
            else:
                new.append(helper.make_node("Unsqueeze",[raw,"rf_ax0"],[raw+"u"],name=f"rotfix{k}u")); new.append(helper.make_node("Expand",[raw+"u","rf_shape"],[o],name=f"rotfix{k}"))
            k+=1
        else: new.append(n)
    m.graph.initializer.extend([numpy_helper.from_array(np.array([0],np.int64),"rf_ax0"),numpy_helper.from_array(np.array([8,2,L,32],np.int64),"rf_shape")])
    del m.graph.node[:]; m.graph.node.extend(new); del m.graph.value_info[:]; return m,k
if sys.argv[1]=="make":
    mode=sys.argv[2]; m,k=fix(mode); p=os.path.join(HERE,f"ve_rot{mode}_L{L}.onnx"); onnx.save(m,p); print("fixed",k,"sin/cos nodes ->",p)
    by={n.name:n for n in m.graph.node}; t=by["/vector_estimator/vector_field/main_blocks.3/attn/Concat_3"].output[0]
    c=os.path.join(HERE,f"ve_rot{mode}_cut_L{L}.onnx"); utils.extract_model(p,c,IN,[t]); mm=onnx.load(c); del mm.graph.value_info[:]; onnx.save(mm,c)
    # CPU equivalence
    d=np.load(SMP); s=ort.InferenceSession(p,providers=["CPUExecutionProvider"]); cs=[]
    for j in range(0,len(d["ref"]),7):
        y=s.run(None,{x:d[x][j] for x in IN})[0]; r=d["ref"][j]; cs.append(float(np.dot(y.ravel(),r.ravel())/(np.linalg.norm(y)*np.linalg.norm(r))))
    print("CPU equivalence min cos",min(cs))
else:
    import qai_hub as hub
    name=sys.argv[2]; d=np.load(SMP); idx=[j for j in range(len(d["sent"])) if int(d["sent"][j])==0 and int(d["step"][j]) in (0,4)] if "cut" in name else [j for j in range(len(d["sent"])) if int(d["sent"][j]) in (0,2,6,9)]
    dev=hub.Device("Dragonwing IQ-9075 EVK"); cj=hub.submit_compile_job(model=os.path.join(HERE,f"{name}.onnx"),device=dev,options="--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",name=name); print("compile",cj.job_id,flush=True)
    cj.wait(); st=cj.get_status(); print("compile",st.code,(st.message or "")[:200],flush=True)
    if str(st.code)!="SUCCESS": sys.exit(1)
    ij=hub.submit_inference_job(model=cj.get_target_model(),device=dev,inputs={k:[d[k][j] for j in idx] for k in IN},name=name+"_infer"); ij.wait(); print("infer",ij.get_status().code,flush=True)
    od=ij.download_output_data(); k=sorted(od.keys())[0]; H=[np.asarray(a) for a in od[k]]
    s=ort.InferenceSession(os.path.join(HERE,f"{name}.onnx"),providers=["CPUExecutionProvider"])
    if "cut" in name:
        for pos,j in enumerate(idx):
            r=s.run(None,{x:d[x][j] for x in IN})[0].astype(np.float64); a=H[pos].reshape(r.shape).astype(np.float64); ax=[q for q,sz in enumerate(r.shape) if sz==L]; sl=[slice(None)]*r.ndim; sl[ax[0]]=slice(0,62)
            print(f"{name} step {int(d['step'][j])}: Concat_3 valid-frame cos {float(r[tuple(sl)].ravel()@a[tuple(sl)].ravel()/(np.linalg.norm(r[tuple(sl)])*np.linalg.norm(a[tuple(sl)]))):.5f}",flush=True)
    else:
        cs={q:[] for q in range(5)}
        for pos,j in enumerate(idx):
            y=H[pos].reshape(d["ref"][j].shape); r=d["ref"][j]; cs[int(d["step"][j])].append(float(np.dot(y.ravel(),r.ravel())/(np.linalg.norm(y)*np.linalg.norm(r))))
        print(name,"ONE-STEP cos per ODE step:",{q:round(float(np.mean(v)),5) for q,v in cs.items()},flush=True)

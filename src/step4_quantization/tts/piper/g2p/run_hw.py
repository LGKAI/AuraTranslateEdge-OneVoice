# Usage: python run_hw.py <key>  (key from build/jobs.json). Waits for the compile job, then runs profile + inference on the Dragonwing IQ-9075.
import sys, json, time, numpy as np, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0,".")
import full_graph as F
k=sys.argv[1]; J=json.load(open("build/jobs.json")); dev=hub.Device("Dragonwing IQ-9075 EVK")
texts=["Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé!","Xin chào, tôi là trợ lý ảo chạy hoàn toàn trên chip.","Việt Nam là một đất nước xinh đẹp với nhiều danh lam thắng cảnh nổi tiếng."]
cj=hub.get_job(J[k]); cj.wait(); st=cj.get_status(); print(k,"compile",st.code,(st.message or "")[:300],flush=True)
if str(st.code)!="SUCCESS": sys.exit(1)
tm=cj.get_target_model()
pj=hub.submit_profile_job(model=tm,device=dev,name=f"piper_vi_{k}_v2_profile"); print(k,"profile",pj.job_id,flush=True)
if k.startswith("full") or k=="g2p": inputs={"text_bytes":[F.text_to_bytes(t) for t in texts]}
else: inputs={"zb":[np.random.RandomState(0).randn(8 if k=="decb8" else 39,192,64).astype(np.float32)*0.5]}
ij=hub.submit_inference_job(model=tm,device=dev,inputs=inputs,name=f"piper_vi_{k}_v2_infer"); print(k,"infer",ij.job_id,flush=True)
t0=time.time(); ij.wait(); st=ij.get_status(); print(k,"infer",st.code,"wall",round(time.time()-t0),"s",(st.message or "")[:200],flush=True)
if str(st.code)=="SUCCESS":
    od=ij.download_output_data(); np.savez(f"build/hw_{k}_out.npz",**{n:np.stack([np.asarray(a) for a in v]) for n,v in od.items()}); print(k,"saved",{n:np.asarray(v[0]).shape for n,v in od.items()},flush=True)
pj.wait(); print(k,"profile",pj.get_status().code,flush=True)
json.dump({"profile":pj.job_id,"infer":ij.job_id},open(f"build/run_{k}.json","w"))

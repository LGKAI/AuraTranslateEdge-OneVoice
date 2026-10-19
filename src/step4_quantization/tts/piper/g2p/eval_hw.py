# Usage: python eval_hw.py fullg8  -> NPU outputs vs fp32 torch reference; wavs -> ../results/full_e2e_npu
import sys, json, wave, numpy as np, torch, qai_hub as hub
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0,".")
import full_graph as F
texts=["Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé!","Xin chào, tôi là trợ lý ảo chạy hoàn toàn trên chip.","Việt Nam là một đất nước xinh đẹp với nhiều danh lam thắng cảnh nổi tiếng."]
K=sys.argv[1] if len(sys.argv)>1 else "fullg8"; R=json.load(open(f"build/run_{K}.json")); ij=hub.get_job(R["infer"]); od=ij.download_output_data(); print({k:np.asarray(v[0]).shape for k,v in od.items()})
names={k:k for k in od}; 
A=np.stack([np.asarray(a) for a in od["output_0"]]).reshape(3,-1); YL=np.stack([np.asarray(a) for a in od["output_1"]]).reshape(3); OOV=np.stack([np.asarray(a) for a in od["output_2"]]).reshape(3)
m,tab=F.build(int(K[-1]) if K[-1].isdigit() else 8)
import os; os.makedirs("../results/full_e2e_npu",exist_ok=True)
for i,t in enumerate(texts):
    with torch.no_grad(): ra,ry,ro=m(torch.from_numpy(F.text_to_bytes(t)))
    ra=ra[0].numpy(); n=int(YL[i])*256; a=A[i]
    L=min(n,int(ry[0])*256); c=float(np.dot(a[:L],ra[:L])/(np.linalg.norm(a[:L])*np.linalg.norm(ra[:L])+1e-12))
    print(f"[{i}] y_len NPU {int(YL[i])} vs fp32 {int(ry[0])} | oov {OOV[i]:.0f} | audio cos {c:.5f} | peak NPU {np.abs(a[:n]).max():.3f} fp32 {np.abs(ra[:L]).max():.3f} | tail max {np.abs(a[n:]).max():.1e} | {n/22050:.2f}s")
    for tag,x in (("NPU",a[:n]),("fp32",ra[:int(ry[0])*256])):
        with wave.open(f"../results/full_e2e_npu/cau{i+1}_{tag}.wav","wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes((np.clip(x,-1,1)*32767).astype(np.int16).tobytes())
pr=hub.get_job(R["profile"]).download_profile(); es=pr["execution_summary"]; u={}
for l in pr["execution_detail"]: u[l.get("compute_unit","?")]=u.get(l.get("compute_unit","?"),0)+1
print("PROFILE fullg8: infer %.1f ms | layers %s | peak mem %.0f MB"%(es["estimated_inference_time"]/1000,u,es.get("estimated_inference_peak_memory",0)/1e6))

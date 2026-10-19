import sys, time, json, wave, numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8")
import full_graph as F
from piper import PiperVoice
m, tab = F.build()
V=F.ONNX; voice=PiperVoice.load(V,config_path=V+".json")
def ref_audio(text):
    ids=voice.phonemes_to_ids(voice.phonemize(text)[0])
    x=np.zeros((1,512),np.int32); x[0,:len(ids)]=ids; xl=np.array([len(ids)],np.int32)
    with torch.no_grad():
        xe,mp,lp,xm=m.enc(torch.from_numpy(x),torch.from_numpy(xl))
        yl,wc=m.sdp(xe,xm,m.ls,m.nw); yl=int(torch.clamp(yl,max=1536).item())
        at,ym=m.align(wc,xm,torch.tensor([yl],dtype=torch.int32))
        z=m.flow(mp,lp,ym,at,m.ns)[0:1]
        first=min(52,yl); zb=torch.zeros(1,192,64); zb[:,:,:first]=z[:,:,:first]
        audio=m.dec(zb).reshape(-1)[:40*256]; total=40
        while total<yl:
            s=total-12; e=min(total+52,yl,z.shape[2]); zb=torch.zeros(1,192,64); zb[:,:,:e-s]=z[:,:,s:e]
            a=m.dec(zb).reshape(-1); cv=min(40,max(0,yl-total)); audio=torch.cat([audio,a[12*256:(12+cv)*256]]); total+=40
    return audio[:yl*256].numpy(), yl
for text in ["Xin chào, tôi là trợ lý ảo chạy hoàn toàn trên chip Qualcomm.","Hôm nay trời đẹp quá, chúng ta cùng đi dạo nhé!"]:
    t0=time.time()
    with torch.no_grad(): a,yl,oov=m(torch.from_numpy(F.text_to_bytes(text)))
    t1=time.time()-t0
    a=a[0].numpy(); n=int(yl[0])*256
    r,ryl=ref_audio(text)
    L=min(n,len(r)); c=float(np.dot(a[:L],r[:L])/(np.linalg.norm(a[:L])*np.linalg.norm(r[:L])))
    print(f"{text[:40]!r}: full y_len {int(yl[0])} ref {ryl} | oov {int(oov[0])} | cos {c:.6f} maxdiff {np.abs(a[:L]-r[:L]).max():.2e} | tail energy {np.abs(a[n:]).max():.1e} | cpu {t1:.1f}s")

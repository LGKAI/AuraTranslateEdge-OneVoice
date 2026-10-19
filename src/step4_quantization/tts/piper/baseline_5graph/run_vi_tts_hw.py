# -*- coding: utf-8 -*-
"""Vietnamese text -> audio. Phoneme/glue on host CPU; encoder/sdp/aligner/flow/decoder on AI Hub Dragonwing IQ-9075 NPU.
   mode=ref: same glue with ONNX Runtime fp32 CPU (reference).  mode=hw: every neural stage = AI Hub inference job."""
import sys, json, time, wave
sys.path.insert(0, "src/step4_npu")
import numpy as np
from pathlib import Path
import piper_components_pipeline as P
P.logging.disable(P.logging.WARNING)
mode, text = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "t1"
OUT = Path("outputs/piper_vi_npu"); COMP = OUT/"components"
import onnxruntime as ort
names = {}
def onnx_io(f):
    s = ort.InferenceSession(str(COMP/f), providers=["CPUExecutionProvider"])
    return [i.name for i in s.get_inputs()], [o.name for o in s.get_outputs()], s
FILES = {"encoder":"piper_vi_encoder.onnx","sdp":"piper_vi_sdp.onnx","monotonic_aligner":"monotonic_aligner.onnx",
         "flow":"piper_vi_flow.onnx","decoder":"piper_vi_decoder.onnx"}
io = {c:onnx_io(f) for c,f in FILES.items()}
log = []
if mode == "hw":
    import qai_hub as hub
    jobs = json.load(open("compile_jobs.json")); dev = hub.Device("Dragonwing IQ-9075 EVK")
    tm = {c: hub.get_job(jobs[c]).get_target_model() for c in FILES}
def run(comp, feeds_list):
    """feeds_list: list of dict(name->array). returns list of list(outputs in onnx order)"""
    if mode == "ref":
        return [io[comp][2].run(None, f) for f in feeds_list]
    t0 = time.time()
    inputs = {k:[f[k] for f in feeds_list] for k in feeds_list[0]}
    j = hub.submit_inference_job(model=tm[comp], device=dev, inputs=inputs, name=f"piper_vi_{comp}_{tag}")
    print(f"[{comp}] job {j.job_id} {j.url}", flush=True); j.wait()
    st = j.get_status(); dt = time.time()-t0
    print(f"[{comp}] {st.code} wall {dt:.0f}s", flush=True)
    log.append({"stage":comp,"job":j.job_id,"status":str(st.code),"wall_s":round(dt,1),"n":len(feeds_list)})
    if str(st.code) != "SUCCESS": raise SystemExit(f"{comp} failed: {st.message}")
    od = j.download_output_data(); outn = io[comp][1]
    # Hub names outputs output_0..output_N in ONNX output order (dict iteration order is NOT reliable)
    keys = sorted(od.keys(), key=lambda k: int(k.split("_")[-1]))
    return [[np.asarray(od[k][i]) for k in keys] for i in range(len(feeds_list))]
ids = P.phonemize_vi(text); print("phoneme ids:", len(ids), flush=True)
x, xl = P.prepare_input(ids)
x_enc, m_p, logs_p, x_mask = [o for o in run("encoder", [{"x":x,"x_lengths":xl}])[0]]
def shp(n,a): print(f"  {n}: {np.asarray(a).shape} {np.asarray(a).dtype}", flush=True)
for n,a in [("x_enc",x_enc),("m_p",m_p),("x_mask",x_mask)]: shp(n,a)
yl, w_ceil = run("sdp", [{"x_encoded":x_enc.reshape(1,192,512),"x_mask":x_mask.reshape(1,1,512),
        "length_scale":np.array([1.0],np.float32),"noise_scale_w":np.array([0.8],np.float32)}])[0]
yl_int = int(np.asarray(yl).flatten()[0]); print("y_len frames:", yl_int, flush=True)
attn, y_mask = run("monotonic_aligner", [{"w_ceil":np.asarray(w_ceil).reshape(1,1,512).astype(np.float32),
        "x_mask":np.asarray(x_mask).reshape(1,1,512).astype(np.float32),"y_lengths":np.array([yl_int],np.int32)}])[0]
if np.asarray(attn).size != 1536*512: attn, y_mask = y_mask, attn
attn = np.asarray(attn).reshape(1,1536,512).astype(np.float32); y_mask = np.asarray(y_mask).reshape(1,1,1536).astype(np.float32)
z = run("flow", [{"m_p":np.asarray(m_p).reshape(1,192,512).astype(np.float32),"logs_p":np.asarray(logs_p).reshape(1,192,512).astype(np.float32),
        "y_mask":y_mask,"attn_squeezed":attn,"noise_scale":np.array([0.667],np.float32)}])[0][0]
z = np.asarray(z).reshape(1,192,1536).astype(np.float32)
# decoder windows (same slicing as reference synthesize), all in ONE job
wins = []; first = min(40+12, yl_int)
zb = np.zeros((1,192,64),np.float32); zb[:,:,:first] = z[:,:,:first]; wins.append(zb); total = 40
while total < yl_int:
    s=total-12; e=min(total+40+12, yl_int, z.shape[2]); zb=np.zeros((1,192,64),np.float32)
    if e>s: zb[:,:,:e-s]=z[:,:,s:e]
    wins.append(zb); total += 40
outs = run("decoder", [{"z":w} for w in wins]); print("decoder windows:", len(wins), flush=True)
audio = np.asarray(outs[0][0]).reshape(-1)[:40*256]; total = 40
for k in range(1, len(wins)):
    cv = min(40, max(0, yl_int-total)); a = np.asarray(outs[k][0]).reshape(-1)
    audio = np.concatenate([audio, a[12*256:(12+cv)*256]]); total += 40
audio = audio[:yl_int*256].astype(np.float32)
pk = float(np.abs(audio).max()); print(f"audio: {len(audio)} samples = {len(audio)/22050:.2f}s @22.05k, peak {pk:.3f}, nan={np.isnan(audio).any()}")
np.save(f"audio_{mode}_{tag}.npy", audio)
with wave.open(f"audio_{mode}_{tag}.wav","wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes((np.clip(audio,-1,1)*32767).astype(np.int16).tobytes())
json.dump(log, open(f"hw_log_{mode}_{tag}.json","w"), indent=1)

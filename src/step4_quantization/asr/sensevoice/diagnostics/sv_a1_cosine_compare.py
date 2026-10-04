import os
# -*- coding: utf-8 -*-
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import onnxruntime as ort
import torchaudio.compliance.kaldi as K

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
DATA_DIR = os.path.join(ROOT, "data", "asr")
QUANT_MODEL = os.environ.get("SV_QDQ_DEBUG_MODEL", "job_jprlmdevp_qdq_onnx/model_a4_layernorm_debug.onnx")  # QDQ onnx tai tu quantize job + debug outputs LayerNorm
CALIB_NPZ = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx", "calib_data_unified_fixed.npz")

FS=16000; N_MELS=80; FRAME_SAMPLES=400; HOP_SAMPLES=160; N_FFT=512; LFR_M,LFR_N=7,6; MAX_LFR=500
window = torch.hamming_window(FRAME_SAMPLES, periodic=False)
dft = torch.fft.rfft(torch.eye(N_FFT)); dft_real, dft_imag = dft.real, dft.imag
mel_fb,_ = K.get_mel_banks(N_MELS, N_FFT, FS, 20.0, 0.0, 100.0, -500.0, 1.0)
LID_DICT = {"zh":3,"en":4,"ko":12}

class FE(nn.Module):
    def __init__(self, cmvn):
        super().__init__()
        self.register_buffer('window', window); self.register_buffer('dft_real', dft_real); self.register_buffer('dft_imag', dft_imag)
        self.register_buffer('mel_fb', mel_fb)
        self.register_buffer('cmvn_mean', cmvn[0:1,:]); self.register_buffer('cmvn_scale', cmvn[1:2,:])
    def forward(self, wav):
        wav = wav*float(1<<15)
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES,1), stride=(HOP_SAMPLES,1)).squeeze(0).transpose(0,1)
        fp = frames.clone(); fp[:,1:] = frames[:,1:] - 0.97*frames[:,:-1]; fp[:,0] = frames[:,0]*(1-0.97)
        windowed = fp*self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT-FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real); imag = torch.matmul(padded, self.dft_imag)
        power = real[:,:-1]**2+imag[:,:-1]**2
        mel = torch.matmul(power, self.mel_fb.T); mel_log = torch.clamp(mel,min=1e-10).log()
        left_pad = mel_log[0:1].expand(LFR_M//2,-1); mel_padded = torch.cat([left_pad, mel_log], dim=0)
        mel_img = mel_padded.T.unsqueeze(0).unsqueeze(-1)
        mel_uf = F.unfold(mel_img, kernel_size=(LFR_M,1), stride=(LFR_N,1)).squeeze(0).view(80,LFR_M,-1)
        mel_uf = mel_uf.permute(1,0,2).reshape(560,-1).transpose(0,1)
        mel_uf = (mel_uf+self.cmvn_mean)*self.cmvn_scale
        actual_len = mel_uf.shape[0]
        if actual_len>=MAX_LFR: mel_uf = mel_uf[:MAX_LFR,:]
        else: mel_uf = F.pad(mel_uf,(0,0,0,MAX_LFR-actual_len))
        return mel_uf.unsqueeze(0), actual_len

print("[1] Load fp32 SenseVoice...", flush=True)
from funasr import AutoModel
from funasr.models.sense_voice.export_meta import export_rebuild_model
am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
orig_fe = am.kwargs.get("frontend"); orig_fe.dither = 0.0
sv = am.model; sv.eval()
export_model = export_rebuild_model(sv, device="cpu", max_seq_len=512)
class StaticSinusoidalPositionEncoder(torch.nn.Module):
    def __init__(self, timesteps=504, depth=560):
        super().__init__()
        positions = torch.arange(1, timesteps + 1).float()[None, :]
        log_timescale_increment = torch.log(torch.tensor([10000.0])) / (depth / 2 - 1)
        inv_timescales = torch.exp(torch.arange(depth / 2).float() * (-log_timescale_increment))
        scaled_time = positions.unsqueeze(-1) * inv_timescales.unsqueeze(0).unsqueeze(0)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        self.register_buffer("pe", encoding.float())
    def forward(self, x):
        return x + self.pe
export_model.encoder.embed = StaticSinusoidalPositionEncoder(timesteps=504, depth=560)
export_model.eval()
fe = FE(orig_fe.cmvn); fe.eval()

# dang ky hook tren tat ca LayerNorm, theo dung thu tu named_modules (da xac nhan khop encoders0->encoders->tp_encoders->after_norm->tp_norm)
captured = []
def make_hook():
    def hook(module, inp, out):
        captured.append(out.detach().numpy())
    return hook
ln_modules = [(name, m) for name, m in export_model.named_modules() if isinstance(m, torch.nn.LayerNorm)]
print(f"  n LayerNorm modules: {len(ln_modules)}", flush=True)
for name, m in ln_modules:
    m.register_forward_hook(make_hook())

# chay fp32 tren zh_2 that
item_path = os.path.join(DATA_DIR, "zh", "zh_2.wav")
wav, sr = sf.read(item_path, dtype="float32")
wav_t = torch.from_numpy(wav).unsqueeze(0)
with torch.no_grad():
    feat, actual_len = fe(wav_t)
    lfr_len = actual_len
    sl = torch.tensor([lfr_len], dtype=torch.int32)
    lid = torch.tensor([LID_DICT["zh"]], dtype=torch.long)
    tn = torch.tensor([15], dtype=torch.long)
    logits_fp32, out_lens_fp32 = export_model(feat, sl, lid, tn)
print(f"  fp32: lfr_len={lfr_len} out_lens={int(out_lens_fp32[0])} n_captured={len(captured)}", flush=True)

fp32_acts = captured  # list of 142 numpy arrays [1, T, C], T = lfr_len (khong padding, vi model chay tren feat that khong padding thua... )
for i, a in enumerate(fp32_acts[:3]):
    print(f"  fp32 layer{i} shape={a.shape}")

print("\n[2] Load quantized ONNX (da co debug outputs)...", flush=True)
sess = ort.InferenceSession(QUANT_MODEL, providers=["CPUExecutionProvider"])
out_names = [o.name for o in sess.get_outputs()]
import re
ln_names = sorted([n for n in out_names if n.startswith("layer_norm")], key=lambda n: int(re.match(r'layer_norm(_(\d+))?$', n).group(2) or 0))
print(f"  n layer_norm outputs: {len(ln_names)}", flush=True)

calib = np.load(CALIB_NPZ)
sv_manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8"))
zh_items = [it for it in sv_manifest if it["lang"] == "zh"]
i_zh2 = 2  # zh_2.wav la item thu 3 (index 2) trong danh sach zh
wav_len_zh2 = calib["wav_len"][5 + i_zh2]  # offset 5 (en) + index trong zh
wav_arr = calib["wav"][5 + i_zh2]
language = calib["language"][5 + i_zh2]; textnorm = calib["textnorm"][5 + i_zh2]
out = sess.run(None, {
    "wav": wav_arr.astype(np.float32), "wav_len": wav_len_zh2.astype(np.int32),
    "language": language.astype(np.int32), "textnorm": textnorm.astype(np.int32),
})
out_map = dict(zip(out_names, out))
quant_acts = [out_map[n] for n in ln_names]  # list of 142 [1,504,C]
print(f"  quant wav_len={wav_len_zh2[0]}", flush=True)

print("\n[3] So sanh cosine similarity tung layer (chi trong vung valid)...", flush=True)
valid_len = lfr_len  # dung do dai LFR that (khong tinh +4 dac biet, vi fp32 hook khong co 4 token dau)
print(f"  valid_len (LFR, khong tinh +4 token dac biet) = {valid_len}")

def cos_sim(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    return float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

print(f"{'idx':4s} {'name':30s} {'cos_sim':>10s} {'rel_l2_err':>12s}")
for i, (name, _) in enumerate(ln_modules):
    fp32_a = fp32_acts[i][0]  # [T_fp32, C]
    quant_a = quant_acts[i][0]  # [504, C] -- nhung 4 vi tri dau la special token, that su offset
    # quant co 4 token dac biet o dau (lang/emo/event/itn), fp32 KHONG co (fp32 forward tra ve logits voi
    # ca 4 token do da duoc model tu them vao ben trong, nhung cac LAYER NORM ben trong encoder co the van
    # xu ly tren chieu dai co san 4 token do tu dau -- can kiem tra T_fp32 co bao gom +4 khong)
    T_fp32 = fp32_a.shape[0]
    T_quant = quant_a.shape[0]
    n = min(T_fp32, valid_len+4, T_quant)
    # thu offset 0 truoc
    fa = fp32_a[:n]
    qa = quant_a[:n]
    if fa.shape[1] != qa.shape[1]:
        print(f'{i:4d} {name:30s} SHAPE MISMATCH fp32={fa.shape} quant={qa.shape}')
        continue
    cs = cos_sim(fa, qa)
    rel_err = float(np.linalg.norm(fa-qa) / (np.linalg.norm(fa)+1e-9))
    flag = '  <-- THAP' if cs < 0.9 else ''
    print(f'{i:4d} {name:30s} {cs:10.4f} {rel_err:12.4f}{flag}')

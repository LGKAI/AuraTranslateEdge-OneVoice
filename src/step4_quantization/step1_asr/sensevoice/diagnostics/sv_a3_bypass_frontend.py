import os
# -*- coding: utf-8 -*-
import sys, json, os, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
torch.set_num_threads(4)
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import onnx
import onnx.numpy_helper as nph
import onnxruntime as ort
import torchaudio.compliance.kaldi as K
import jiwer

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
DATA_DIR = os.path.join(ROOT, "data", "asr")
QUANT_MODEL_PATH = os.environ.get("SV_QDQ_MODEL", "job_jprlmdevp_qdq_onnx/model.onnx")  # QDQ onnx tai tu quantize job (graph gop cu)
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

class StaticPE(nn.Module):
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

print("[1] Load fp32 frontend + acoustic model that (de lay dung add_6 that, co xu ly 4 token dac biet)...", flush=True)
from funasr import AutoModel
from funasr.models.sense_voice.export_meta import export_rebuild_model
am = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)
orig_fe = am.kwargs.get("frontend")
fe = FE(orig_fe.cmvn); fe.eval()
sv = am.model; sv.eval()
export_model_ref = export_rebuild_model(sv, device="cpu", max_seq_len=512)
export_model_ref.encoder.embed = StaticPE(timesteps=504, depth=560)
export_model_ref.eval()

_captured_add6 = {}
def _hook(module, inp, out):
    _captured_add6["v"] = out.detach()
export_model_ref.encoder.embed.register_forward_hook(_hook)

def norm_wer(s): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip())).strip()
def cer(ref, hyp):
    ref_c = re.sub(r"\s+", "", ref); hyp_c = re.sub(r"\s+", "", hyp)
    return jiwer.cer(ref_c, hyp_c) if ref_c else 1.0
def decode_bytes(arr):
    arr = np.asarray(arr).astype(np.int64).reshape(-1)
    raw = bytes(int(b) & 0xFF for b in arr)
    return raw.replace(b"\x00", b"").decode("utf-8", errors="replace")

manifest = json.load(open(os.path.join(DATA_DIR, "manifest.json"), encoding="utf-8"))
sv_items = [it for it in manifest if it["lang"] in ("en","zh","ko")]
calib = np.load(CALIB_NPZ)

print("[2] Graph surgery: thay add_6 bang tensor fp32 that (bypass toan bo frontend quantize)...", flush=True)
base_model = onnx.load(QUANT_MODEL_PATH, load_external_data=True)

targets = [2, 5, 6, 7, 10]  # en_2(tot), zh_0, zh_1, zh_2(loi), ko_0(loi)
for idx in targets:
    item = sv_items[idx]
    wav_np = calib["wav"][idx]; wav_len_np = calib["wav_len"][idx]
    language = calib["language"][idx]; textnorm = calib["textnorm"][idx]

    wav_t = torch.from_numpy(wav_np[0, :int(wav_len_np[0])]).unsqueeze(0)
    with torch.no_grad():
        feat, actual_len = fe(wav_t)
        sl = torch.tensor([actual_len], dtype=torch.int32)
        lid_t = torch.tensor([int(language[0])], dtype=torch.long)
        tn_t = torch.tensor([int(textnorm[0])], dtype=torch.long)
        _ = export_model_ref(feat, sl, lid_t, tn_t)
        add6_true = _captured_add6["v"]  # [1,504,560] -- gia tri fp32 THAT cua add_6 (dung 4 token dac biet)

    model = onnx.ModelProto(); model.CopyFrom(base_model)
    graph = model.graph
    # xoa initializer 'add_6' cu neu co, them moi
    to_remove = [i for i in graph.initializer if i.name == "add_6_bypass_fp32"]
    for i in to_remove: graph.initializer.remove(i)
    new_init = nph.from_array(add6_true.numpy().astype(np.float32), name="add_6_bypass_fp32")
    graph.initializer.append(new_init)
    for n in graph.node:
        if n.name == "node_add_6":
            n.output[0] = "add_6_DISABLED"  # vo hieu hoa output goc (tranh trung ten)
    for n in graph.node:
        for i, inp in enumerate(n.input):
            if inp == "add_6":
                n.input[i] = "add_6_bypass_fp32"

    tmp_path = os.path.join(__import__("tempfile").gettempdir(), f"_tmp_bypass_fe_{idx}.onnx")
    onnx.save_model(model, tmp_path, save_as_external_data=False)
    sess = ort.InferenceSession(tmp_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {
        "wav": wav_np.astype(np.float32), "wav_len": wav_len_np.astype(np.int32),
        "language": language.astype(np.int32), "textnorm": textnorm.astype(np.int32),
    })
    hyp = decode_bytes(out[0])
    lang = item["lang"]; ref = item["transcript"]
    metric = cer(ref, hyp) if lang == "zh" else (jiwer.wer(norm_wer(ref), norm_wer(hyp)) if norm_wer(hyp) else 1.0)
    print(f"[{lang}] {os.path.basename(item['path'])} (frontend bypass -> fp32) {'CER' if lang=='zh' else 'WER'}={metric*100:5.1f}%  HYP={hyp[:70]!r}", flush=True)
    os.remove(tmp_path)

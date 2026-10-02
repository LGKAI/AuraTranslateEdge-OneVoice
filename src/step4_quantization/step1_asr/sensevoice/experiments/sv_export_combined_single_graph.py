# -*- coding: utf-8 -*-
"""Gop Frontend (DA FIX loi tran so fp16) + Encoder+Classifier thanh 1 GRAPH ONNX DUY NHAT:
wav, wav_len, language, textnorm -> byte_stream.

Ly do gop: giam tu 2 AI Hub inference job (2 lan tai model "nguoi" len thiet bi, moi lan co
the mat toi hang tram giay) xuong con 1 job duy nhat -> giam ~50% so lan tai nguoi.

Khong quantize encoder (dung fp16 thuan) -- da chung minh trong bao cao: fp16 encoder cho
KET QUA GIONG HET encoder quantize W8A16 (khong mat chat luong), va thuc te compute con
NHANH HON (~211ms vs ~269ms do qua profile job). Nen gop toan bo thanh 1 graph fp16, khong
can buoc quantize rieng cho encoder nua.
"""
import os, sys, time, json, math
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
from step4_s1_export_sensevoice_e2e_unified import (
    build_mel_filterbank, MAX_WAV_SAMPLES, MAX_LFR_FRAMES, MAX_SEQ_FRAMES, L_MAX, BYTE_STREAM_LEN,
    FRAME_SAMPLES, HOP_SAMPLES, N_FFT, N_MELS, LFR_M, LFR_N,
    StaticSinusoidalPositionEncoder, StaticCTCCollapseAndDetokenizer, build_static_byte_table,
    MODEL_ID_HF, MODEL_ID_MS,
)

OUT_DIR = os.path.join(os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice"), "outputs", "sensevoice-e2e-onnx")
SCALE = float(1 << 15)
LOG_SCALE_COMPENSATION = 2.0 * math.log(SCALE)
SAFE_CLAMP_MIN_FP16 = 1e-10 / (SCALE ** 2)


class TraceableFrontendFixed(nn.Module):
    """Frontend DA FIX loi tran so fp16 (bo scale x32768 truoc FFT-matmul, bu bang hang so
    cong trong khong gian log) -- da xac nhan tren NPU that: cos_sim tu 0.24 len 1.000000."""
    def __init__(self, cmvn: torch.Tensor):
        super().__init__()
        self.register_buffer("window", torch.hamming_window(FRAME_SAMPLES))
        dft_complex = torch.fft.rfft(torch.eye(N_FFT))
        self.register_buffer("dft_real", dft_complex.real)
        self.register_buffer("dft_imag", dft_complex.imag)
        mel_fb = build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)
        self.register_buffer("cmvn_mean", cmvn[0:1, :])
        self.register_buffer("cmvn_scale", cmvn[1:2, :])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav_img = wav.unsqueeze(1).unsqueeze(-1)
        frames = F.unfold(wav_img, kernel_size=(FRAME_SAMPLES, 1), stride=(HOP_SAMPLES, 1))
        frames = frames.squeeze(0).transpose(0, 1)
        frames_pe = frames.clone()
        frames_pe[:, 1:] = frames[:, 1:] - 0.97 * frames[:, :-1]
        frames_pe[:, 0] = frames[:, 0] * (1.0 - 0.97)
        windowed = frames_pe * self.window.unsqueeze(0)
        padded = F.pad(windowed, (0, N_FFT - FRAME_SAMPLES))
        real = torch.matmul(padded, self.dft_real)
        imag = torch.matmul(padded, self.dft_imag)
        power_no_dc = real[:, :-1] ** 2 + imag[:, :-1] ** 2

        mel = torch.matmul(power_no_dc, self.mel_fb.T)
        mel_log = torch.clamp(mel, min=SAFE_CLAMP_MIN_FP16).log() + LOG_SCALE_COMPENSATION

        left_pad = mel_log[0:1].expand(LFR_M // 2, -1)
        mel_padded = torch.cat([left_pad, mel_log], dim=0)
        mel_img = mel_padded.T.unsqueeze(0).unsqueeze(-1)
        mel_uf = F.unfold(mel_img, kernel_size=(LFR_M, 1), stride=(LFR_N, 1))
        mel_uf = mel_uf.squeeze(0).view(80, LFR_M, -1)
        mel_uf = mel_uf.permute(1, 0, 2).reshape(560, -1).transpose(0, 1)

        mel_uf = (mel_uf + self.cmvn_mean) * self.cmvn_scale

        actual_len = mel_uf.shape[0]
        if actual_len >= MAX_LFR_FRAMES:
            mel_uf = mel_uf[:MAX_LFR_FRAMES, :]
        else:
            mel_uf = F.pad(mel_uf, (0, 0, 0, MAX_LFR_FRAMES - actual_len))
        return mel_uf.unsqueeze(0)


class CombinedGraph(nn.Module):
    """1 graph DUY NHAT: wav, wav_len, language, textnorm -> byte_stream.
    Khong quantize -- chay fp16 thuan tren NPU (da chung minh khong mat chat luong)."""
    def __init__(self, frontend, acoustic_model, decoder):
        super().__init__()
        self.frontend = frontend
        self.acoustic_model = acoustic_model
        self.decoder = decoder

    def forward(self, wav, wav_len, language, textnorm):
        fbank = self.frontend(wav)
        wav_len_i = wav_len.to(torch.int64)
        fbank_len = (wav_len_i - FRAME_SAMPLES) // HOP_SAMPLES + 1
        fbank_len = torch.clamp(fbank_len, min=0)
        lfr_len = (fbank_len + (LFR_M // 2) - LFR_M) // LFR_N + 1
        lfr_len = torch.clamp(lfr_len, min=0, max=MAX_LFR_FRAMES)
        speech_lengths = lfr_len.to(torch.int32).reshape(1)

        logits, out_lens = self.acoustic_model(fbank, speech_lengths, language, textnorm)
        valid_len = speech_lengths.to(torch.int64) + 4
        byte_stream = self.decoder(logits, valid_len)
        return byte_stream


def main():
    print("[1] Tai SenseVoice-Small tu FunASR...")
    from funasr import AutoModel
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    sv = am.model
    sv.eval()
    tokenizer_sp = am.kwargs.get("tokenizer").sp
    vocab_size = tokenizer_sp.get_piece_size()

    print("[2] Xay dung Byte Table + Static Position Encoder...")
    byte_table_tensor = build_static_byte_table(tokenizer_sp, vocab_size=vocab_size, l_max=L_MAX)
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    import types
    sv_exported = export_rebuild_model(sv, device="cpu", max_seq_len=512)
    sv_exported.export_dynamic_axes = types.MethodType(lambda self: {}, sv_exported)
    sv_exported.encoder.embed = StaticSinusoidalPositionEncoder(timesteps=MAX_SEQ_FRAMES, depth=560)

    orig_fe = am.kwargs.get("frontend")
    frontend = TraceableFrontendFixed(orig_fe.cmvn)
    frontend.eval()
    decoder = StaticCTCCollapseAndDetokenizer(byte_table_tensor, max_frames=MAX_SEQ_FRAMES, l_max=L_MAX)
    decoder.eval()

    print("[3] Gop thanh 1 graph duy nhat...")
    combined = CombinedGraph(frontend, sv_exported, decoder)
    combined.eval()

    dummy_wav = torch.randn(1, MAX_WAV_SAMPLES, dtype=torch.float32)
    dummy_wav_len = torch.tensor([MAX_WAV_SAMPLES], dtype=torch.int32)
    dummy_lang = torch.tensor([4], dtype=torch.int32)
    dummy_tn = torch.tensor([14], dtype=torch.int32)

    onnx_path = os.path.join(OUT_DIR, "model_sv_combined_single.onnx")
    print(f"\n[4] Xuat ONNX sang: {onnx_path} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            combined, (dummy_wav, dummy_wav_len, dummy_lang, dummy_tn), onnx_path,
            input_names=["wav", "wav_len", "language", "textnorm"],
            output_names=["byte_stream"],
            dynamic_axes={}, opset_version=17, do_constant_folding=True,
        )
    print(f"  Xuat thanh cong sau {time.time()-t0:.1f}s! Size: {os.path.getsize(onnx_path)/1e6:.1f} MB")

    print("\n[5] Graph surgery (bom Conv bias, tranh loi khi khong co bias)...")
    import onnx.numpy_helper as nph
    import numpy as np
    model = onnx.load(onnx_path, load_external_data=True)
    graph = model.graph
    existing_inputs = {init.name for init in graph.initializer}
    patched = 0
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) >= 3 and node.input[2]:
            continue
        weight_name = node.input[1]
        weight_init = next((i for i in graph.initializer if i.name == weight_name), None)
        if weight_init is None:
            continue
        out_channels = weight_init.dims[0]
        bias_name = f"{weight_name}_dummy_bias"
        if bias_name not in existing_inputs:
            bias_np = np.zeros(out_channels, dtype=np.float32)
            bias_tensor = nph.from_array(bias_np, name=bias_name)
            graph.initializer.append(bias_tensor)
            existing_inputs.add(bias_name)
        while len(node.input) < 3:
            node.input.append("")
        node.input[2] = bias_name
        patched += 1
    for o in graph.output:
        if o.name == "byte_stream":
            o.type.tensor_type.shape.dim[0].dim_value = 1
            o.type.tensor_type.shape.dim[0].ClearField('dim_param')
            o.type.tensor_type.shape.dim[1].dim_value = BYTE_STREAM_LEN
            o.type.tensor_type.shape.dim[1].ClearField('dim_param')
    print(f"  Patched {patched} Conv nodes.")

    print("\n[6] Resave INLINE (tranh loi external-data khi upload AI Hub)...")
    inline_path = os.path.join(OUT_DIR, "model_sv_combined_single_inline.onnx")
    onnx.save_model(model, inline_path, save_as_external_data=False)
    print(f"  Saved: {inline_path}  size={os.path.getsize(inline_path)/1e6:.1f}MB")

    cfg_path = os.path.join(OUT_DIR, "combined_single_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"combined_onnx": onnx_path, "combined_inline_onnx": inline_path}, f, indent=2)
    print("Saved config:", cfg_path)


if __name__ == "__main__":
    main()

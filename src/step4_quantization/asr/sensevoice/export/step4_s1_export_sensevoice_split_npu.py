# -*- coding: utf-8 -*-
"""step4_s1_export_sensevoice_split_npu.py — Xuất SenseVoice-Small thành 2 graph NPU RIÊNG BIỆT
thay vì 1 graph gộp duy nhất.

LÝ DO (đã xác nhận bằng thực nghiệm, xem ZIPFORMER_CTC_RESET_REPORT.md / báo cáo phiên làm việc):
  Khi gộp Frontend (tính fbank: DFT matmul + mel filterbank + CMVN) vào CHUNG 1 graph với
  Encoder (50 lớp transformer) rồi quantize W8A16 CẢ GRAPH, phần Frontend bị quantize sai
  nghiêm trọng (dải động log-mel spectrum quá rộng cho 1 scale quantize duy nhất), làm hỏng
  dữ liệu đầu vào Encoder ngay từ đầu -- Encoder/Classifier dù tốt đến đâu cũng vô ích.
  Đã kiểm chứng: bypass Frontend về fp32 thật (giữ Encoder+Classifier quantize) đưa zh_2 từ
  100% lỗi (rỗng hoàn toàn) xuống 8.1% CER -- xác nhận Frontend chính là thủ phạm.

GIẢI PHÁP: tách thành 2 graph NPU compile RIÊNG:
  Graph 1 (frontend): wav, wav_len -> fbank[1,500,560], speech_lengths[1]
    KHÔNG quantize (hoặc quantize rất nhẹ) -- vẫn compile lên NPU chạy fp16, KHÔNG phải CPU.
  Graph 2 (encoder_classifier): fbank, speech_lengths, language, textnorm -> byte_stream
    Quantize W8A16 như cũ (đã xác nhận encoder+classifier chịu quantize tốt qua thực nghiệm A2).
  Host chỉ chuyển tensor fbank+speech_lengths giữa 2 lần gọi NPU -- vẫn là "100% NPU" theo
  đúng nghĩa (toàn bộ phép tính neural đều chạy NPU, host không tính toán gì, chỉ định tuyến).

Cách chạy:
  python step4_s1_export_sensevoice_split_npu.py
"""
import os
import sys
import types
import time
import json
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# tai su dung toan bo class/ham da co san trong file goc, khong viet lai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_s1_export_sensevoice_e2e_unified import (
    TraceableFrontend, StaticSinusoidalPositionEncoder,
    StaticCTCCollapseAndDetokenizer, build_static_byte_table,
    MAX_WAV_SAMPLES, MAX_LFR_FRAMES, MAX_SEQ_FRAMES, L_MAX, BYTE_STREAM_LEN,
    FRAME_SAMPLES, HOP_SAMPLES, LFR_M, LFR_N,
    MODEL_ID_HF, MODEL_ID_MS,
)

import onnx
import onnx.numpy_helper as nph
import numpy as np


def patch_conv_bias(onnx_path: str, patched_path: str) -> int:
    """Ban sua: dung save_as_external_data=True de tranh loi vuot gioi han 2GB protobuf
    (khac ham goc trong file unified, chi can cho graph nho hon; graph nay lon hon can external data)."""
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
    for o in model.graph.output:
        if o.name == "byte_stream":
            o.type.tensor_type.shape.dim[0].dim_value = 1
            o.type.tensor_type.shape.dim[0].ClearField('dim_param')
            o.type.tensor_type.shape.dim[1].dim_value = BYTE_STREAM_LEN
            o.type.tensor_type.shape.dim[1].ClearField('dim_param')
    onnx.save_model(model, patched_path, save_as_external_data=True, all_tensors_to_one_file=True,
                     location=os.path.basename(patched_path) + ".data")
    print(f"[patch_conv_bias] Patched {patched} Conv nodes -> Saved {patched_path}")
    return patched


def patch_mask_outliers(onnx_path: str, patched_path: str, clip_val: float = -30.0) -> int:
    model = onnx.load(onnx_path, load_external_data=True)
    patched = 0
    for i, init in enumerate(model.graph.initializer):
        arr = nph.to_array(init)
        if arr.size == 0 or arr.dtype not in [np.float32, np.float16]:
            continue
        min_val = float(arr.min())
        if min_val < clip_val:
            arr_clipped = np.clip(arr, clip_val, None)
            new_init = nph.from_array(arr_clipped.astype(arr.dtype), name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, new_init)
            patched += 1
    onnx.save_model(model, patched_path, save_as_external_data=True, all_tensors_to_one_file=True,
                     location=os.path.basename(patched_path) + ".data")
    print(f"[patch_mask] Clipped {patched} outlier initializers to {clip_val} -> Saved {patched_path}")
    return patched

ROOT = os.environ.get("SV_ROOT", r"C:\Users\Admin\Downloads\AuraTranslateEdge-OneVoice")
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-e2e-onnx")


class FrontendGraph(torch.nn.Module):
    """Graph 1: wav, wav_len -> fbank, speech_lengths (tinh toan do dai bang so nguyen thuan,
    KHONG cast qua float -- da xac nhan giu duoc tensor nay ngoai pham vi quantize)."""
    def __init__(self, frontend: TraceableFrontend):
        super().__init__()
        self.frontend = frontend

    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor):
        fbank = self.frontend(wav)  # [1, 500, 560]
        wav_len_i = wav_len.to(torch.int64)
        fbank_len = (wav_len_i - FRAME_SAMPLES) // HOP_SAMPLES + 1
        fbank_len = torch.clamp(fbank_len, min=0)
        lfr_len = (fbank_len + (LFR_M // 2) - LFR_M) // LFR_N + 1
        lfr_len = torch.clamp(lfr_len, min=0, max=MAX_LFR_FRAMES)
        speech_lengths = lfr_len.to(torch.int32).reshape(1)
        return fbank, speech_lengths


class EncoderClassifierGraph(torch.nn.Module):
    """Graph 2: fbank, speech_lengths, language, textnorm -> byte_stream."""
    def __init__(self, acoustic_model, decoder: StaticCTCCollapseAndDetokenizer):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.decoder = decoder

    def forward(self, fbank: torch.Tensor, speech_lengths: torch.Tensor,
                language: torch.Tensor, textnorm: torch.Tensor):
        logits, out_lens = self.acoustic_model(fbank, speech_lengths, language, textnorm)
        # FIX BUG #5 (da xac nhan): khong dung out_lens model tu tra ve (bi quantize sai,
        # bao hoa o 180) -- tu tinh valid_len = speech_lengths + 4 (offset da do dac chinh xac).
        valid_len = speech_lengths.to(torch.int64) + 4
        byte_stream = self.decoder(logits, valid_len)
        return byte_stream


def export_frontend_graph(frontend):
    graph1 = FrontendGraph(frontend)
    graph1.eval()
    dummy_wav = torch.randn(1, MAX_WAV_SAMPLES, dtype=torch.float32)
    dummy_wav_len = torch.tensor([MAX_WAV_SAMPLES], dtype=torch.int32)

    onnx_path = os.path.join(OUT_DIR, "model_sv_frontend.onnx")
    print(f"\n[Graph 1/2] Xuất Frontend ONNX sang: {onnx_path} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            graph1,
            (dummy_wav, dummy_wav_len),
            onnx_path,
            input_names=["wav", "wav_len"],
            output_names=["fbank", "speech_lengths"],
            dynamic_axes={},
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Xuất thành công sau {time.time()-t0:.1f}s! Size: {os.path.getsize(onnx_path)/1e6:.1f} MB")
    return onnx_path


def export_encoder_classifier_graph(sv_exported, decoder):
    graph2 = EncoderClassifierGraph(sv_exported, decoder)
    graph2.eval()
    dummy_fbank = torch.randn(1, MAX_LFR_FRAMES, 560, dtype=torch.float32)
    dummy_sl = torch.tensor([MAX_LFR_FRAMES], dtype=torch.int32)
    dummy_lang = torch.tensor([4], dtype=torch.int32)
    dummy_tn = torch.tensor([14], dtype=torch.int32)

    onnx_raw = os.path.join(OUT_DIR, "model_sv_enc_cls_raw.onnx")
    onnx_patched = os.path.join(OUT_DIR, "model_sv_enc_cls_patched.onnx")
    print(f"\n[Graph 2/2] Xuất Encoder+Classifier ONNX sang: {onnx_raw} ...")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            graph2,
            (dummy_fbank, dummy_sl, dummy_lang, dummy_tn),
            onnx_raw,
            input_names=["fbank", "speech_lengths", "language", "textnorm"],
            output_names=["byte_stream"],
            dynamic_axes={},
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"  Xuất thành công sau {time.time()-t0:.1f}s!")

    print("  Graph Surgery (Bơm Conv Bias + Clamp Outlier)...")
    onnx_bias_patched = os.path.join(OUT_DIR, "model_sv_enc_cls_temp_bias.onnx")
    patch_conv_bias(onnx_raw, onnx_bias_patched)
    patch_mask_outliers(onnx_bias_patched, onnx_patched, clip_val=-30.0)
    if os.path.exists(onnx_bias_patched):
        os.remove(onnx_bias_patched)
    print(f"  Size: {os.path.getsize(onnx_patched)/1e6:.1f} MB")
    return onnx_patched


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print("XUẤT SENSEVOICE-SMALL THÀNH 2 GRAPH NPU RIÊNG (FIX LỖI QUANTIZE FRONTEND)")
    print("=" * 70)

    print("\n[1/4] Đang tải SenseVoice-Small từ FunASR ...")
    from funasr import AutoModel
    try:
        am = AutoModel(model=MODEL_ID_HF, hub="hf", device="cpu", disable_update=True)
    except TypeError:
        am = AutoModel(model=MODEL_ID_MS, device="cpu", disable_update=True)
    sv = am.model
    sv.eval()
    tokenizer_sp = am.kwargs.get("tokenizer").sp
    vocab_size = tokenizer_sp.get_piece_size()

    print("[2/4] Đang xây dựng Byte Table + Static Position Encoder ...")
    byte_table_tensor = build_static_byte_table(tokenizer_sp, vocab_size=vocab_size, l_max=L_MAX)
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    sv_exported = export_rebuild_model(sv, device="cpu", max_seq_len=512)
    sv_exported.export_dynamic_axes = types.MethodType(lambda self: {}, sv_exported)
    sv_exported.encoder.embed = StaticSinusoidalPositionEncoder(timesteps=MAX_SEQ_FRAMES, depth=560)

    orig_fe = am.kwargs.get("frontend")
    frontend = TraceableFrontend(orig_fe.cmvn)
    frontend.eval()
    decoder = StaticCTCCollapseAndDetokenizer(byte_table_tensor, max_frames=MAX_SEQ_FRAMES, l_max=L_MAX)
    decoder.eval()

    print("[3/4] Xuất 2 graph ONNX ...")
    frontend_onnx = export_frontend_graph(frontend)
    enc_cls_onnx = export_encoder_classifier_graph(sv_exported, decoder)

    print("\n[4/4] Lưu config ...")
    config = {
        "frontend_onnx": frontend_onnx,
        "encoder_classifier_onnx": enc_cls_onnx,
        "frontend_input_shapes": {"wav": [1, MAX_WAV_SAMPLES], "wav_len": [1]},
        "frontend_output_shapes": {"fbank": [1, MAX_LFR_FRAMES, 560], "speech_lengths": [1]},
        "enc_cls_input_shapes": {
            "fbank": [1, MAX_LFR_FRAMES, 560], "speech_lengths": [1],
            "language": [1], "textnorm": [1],
        },
        "enc_cls_output_shapes": {"byte_stream": [1, BYTE_STREAM_LEN]},
        "target_hardware": "Qualcomm Dragonwing IQ-9075 EVK (Hexagon NPU v73)",
        "recipe": "Frontend: KHONG quantize (fp16 tren NPU) | Encoder+Classifier: W8A16",
    }
    cfg_path = os.path.join(OUT_DIR, "split_npu_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Đã lưu: {cfg_path}")

    print("\n" + "=" * 70)
    print("✅ HOÀN TẤT XUẤT 2 GRAPH NPU RIÊNG BIỆT!")
    print(f"  Frontend:          {frontend_onnx}")
    print(f"  Encoder+Classifier: {enc_cls_onnx}")
    print("=" * 70)


if __name__ == "__main__":
    main()

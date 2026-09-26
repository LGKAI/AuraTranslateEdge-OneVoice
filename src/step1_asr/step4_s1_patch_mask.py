"""Step 4 — Bước 2/6: Patch mask outlier trong ONNX graph

Mục tiêu:
  - Tìm và clamp giá trị cực đoan trong attention mask (VD: -inf, -3.4e38)
  - Thay bằng -30.0 (đủ âm để softmax triệt tiêu, nhưng không nuốt độ phân giải)
  - Xử lý cả static outliers (initializers) lẫn dynamic outliers (qua graph surgery)
  - Verify output của model patched ≈ output của model gốc (cosine sim > 0.999)

Input:  outputs/sensevoice-onnx/model.onnx
Output: outputs/sensevoice-onnx/model_patched.onnx

Cách chạy:
  python src/step1_asr/step4_s1_patch_mask.py [--clip-val -30.0]

Tài liệu tham khảo:
  meeting_prep_quantization.md §4.3, §5.2 — outlier nuốt độ phân giải quantize
  NLLB: sửa mask -3.4e38 → -30 đạt cos_sim=0.9998
"""
import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "sensevoice-onnx")
SRC_ONNX = os.path.join(OUT_DIR, "model.onnx")
DST_ONNX = os.path.join(OUT_DIR, "model_patched.onnx")
CLIP_VAL = -30.0   # giá trị clamp mặc định (e^-30 ≈ 9e-14 ≈ 0, đủ cho softmax)


def patch_static_initializers(model, clip_val: float):
    """Clamp outlier trong tất cả initializers (trọng số tĩnh trong graph)."""
    import onnx
    patched_count = 0
    for i, init in enumerate(model.graph.initializer):
        arr = onnx.numpy_helper.to_array(init)
        if arr.size == 0 or arr.dtype not in [np.float32, np.float16]:
            continue
        min_val = float(arr.min())
        if min_val < clip_val:
            print(f"  [patch_init] '{init.name[:60]}' "
                  f"shape={list(arr.shape)}  min={min_val:.2e} → clip to {clip_val}")
            arr_clipped = np.clip(arr, clip_val, None)
            new_init = onnx.numpy_helper.from_array(arr_clipped.astype(arr.dtype),
                                                     name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.insert(i, new_init)
            patched_count += 1
    return patched_count


def add_clip_node_after_mask_nodes(model, clip_val: float):
    """
    ONNX graph surgery: thêm Clip node ngay sau các node tạo ra mask cực đoan.
    
    SenseVoice có thể tạo mask động (không phải hằng số tĩnh). Chiến lược:
    - Tìm các node có output tên chứa "mask" hoặc output có giá trị phạm vi rất âm
    - Thêm Clip(min=clip_val) ngay sau node đó
    - Rewire tất cả consumer của output gốc → dùng output đã clip
    
    Lưu ý: Nếu SenseVoice không dùng dynamic mask (chỉ dùng static initializer),
    hàm này sẽ không thêm node nào (safe no-op).
    """
    import onnx
    from onnx import helper, TensorProto

    graph = model.graph
    nodes_to_add = []
    rewire_map = {}  # old_output_name → new_clipped_output_name

    clip_min_name = f"__clip_mask_min_{abs(int(clip_val))}"

    # Tìm node có output tên nghi là mask
    mask_keywords = ["mask", "bias", "attn_bias", "pos_bias", "key_mask"]

    for node in graph.node:
        for out_name in node.output:
            if any(kw in out_name.lower() for kw in mask_keywords):
                if out_name in rewire_map:
                    continue
                clipped_name = out_name + "_clipped"
                # Tạo Clip node
                clip_node = helper.make_node(
                    "Clip",
                    inputs=[out_name, clip_min_name, ""],  # min=clip_val, max=unbounded
                    outputs=[clipped_name],
                    name=f"__auto_clip_{out_name[:30]}",
                )
                nodes_to_add.append(clip_node)
                rewire_map[out_name] = clipped_name
                print(f"  [graph_surgery] Adding Clip node after '{out_name[:60]}'")

    if not nodes_to_add:
        print("  [graph_surgery] No dynamic mask outputs detected by name heuristic — "
              "static initializer patch may be sufficient")
        return 0

    # Thêm hằng số clip_min vào initializers
    clip_min_arr = np.array([clip_val], dtype=np.float32)
    clip_min_init = onnx.numpy_helper.from_array(clip_min_arr, name=clip_min_name)
    graph.initializer.append(clip_min_init)

    # Thêm các Clip nodes vào graph
    for cn in nodes_to_add:
        graph.node.append(cn)

    # Rewire: thay thế tất cả references đến output gốc → output đã clip
    for node in graph.node:
        new_inputs = []
        for inp in node.input:
            new_inputs.append(rewire_map.get(inp, inp))
        if list(node.input) != new_inputs:
            del node.input[:]
            node.input.extend(new_inputs)

    return len(nodes_to_add)


def patch_missing_conv_biases(model):
    """
    QNN / QAIRT requires all Conv nodes to have a bias tensor for per-channel w8a16 quantization.
    This function finds Conv nodes with 2 inputs (no bias) and adds a dummy zero-bias.
    """
    import onnx
    patched_count = 0
    # Create a map of initializers for quick lookup
    init_map = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) == 2:
            weight_name = node.input[1]
            if weight_name in init_map:
                weight_tensor = init_map[weight_name]
                weight_arr = onnx.numpy_helper.to_array(weight_tensor)
                # Conv weight shape in ONNX: [out_channels, in_channels/group, k1, k2]
                out_channels = weight_arr.shape[0]
                
                bias_name = node.name + "_dummy_bias"
                bias_arr = np.zeros((out_channels,), dtype=np.float32)
                bias_init = onnx.numpy_helper.from_array(bias_arr, name=bias_name)
                
                model.graph.initializer.append(bias_init)
                node.input.append(bias_name)
                patched_count += 1
                
    return patched_count



def verify_patch(src_onnx, dst_onnx):
    """So sánh output gốc vs patched trên sample thật."""
    print("\n[verify_patch] Comparing original vs patched ONNX outputs...")
    try:
        import onnxruntime as ort
        import soundfile as sf
        from scipy.spatial.distance import cosine as cos_dist

        data_dir = os.path.join(ROOT, "data", "asr")
        wav_path = os.path.join(data_dir, "en", "en_0.wav")
        wav, sr = sf.read(wav_path)
        wav_np = wav.astype(np.float32)[np.newaxis, :]

        sess_orig = ort.InferenceSession(src_onnx, providers=["CPUExecutionProvider"])
        sess_patch = ort.InferenceSession(dst_onnx, providers=["CPUExecutionProvider"])

        input_names = [inp.name for inp in sess_orig.get_inputs()]
        lang_map = {"speech": wav_np,
                    "speech_lengths": np.array([wav.shape[0]], dtype=np.int32),
                    "language": np.array([3], dtype=np.int32),
                    "textnorm": np.array([15], dtype=np.int32)}
        feed = {k: v for k, v in lang_map.items() if k in input_names}

        out_orig = sess_orig.run(None, feed)[0].flatten()
        out_patch = sess_patch.run(None, feed)[0].flatten()

        cos_sim = 1 - cos_dist(out_orig.astype(np.float64),
                                out_patch.astype(np.float64))
        print(f"[verify_patch] cosine_similarity(original, patched) = {cos_sim:.6f}")

        if cos_sim > 0.9999:
            print("[verify_patch] ✅ Patch does not change output — safe to proceed")
        elif cos_sim > 0.999:
            print("[verify_patch] ⚠️  Minor difference — acceptable for mask-only patch")
        else:
            print(f"[verify_patch] ⚠️  Significant diff ({cos_sim:.4f}) — investigate!")

        return cos_sim
    except Exception as e:
        import traceback
        print(f"[verify_patch] Failed: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-val", type=float, default=CLIP_VAL,
                        help="Clamp negative outliers to this value (default: -30.0)")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    clip_val = args.clip_val

    print("=" * 60)
    print("Step 4 — Bước 2: Patch mask outlier trong ONNX graph")
    print("=" * 60)
    print(f"  Source:    {SRC_ONNX}")
    print(f"  Target:    {DST_ONNX}")
    print(f"  Clip val:  {clip_val}")

    if not os.path.exists(SRC_ONNX):
        print(f"\n❌ model.onnx not found at {SRC_ONNX}")
        print("   Hãy chạy step4_s1_export_sensevoice_onnx.py trước.")
        sys.exit(1)

    import onnx
    print(f"\n[patch] Loading {SRC_ONNX} ({os.path.getsize(SRC_ONNX)/1e6:.1f}MB)...")
    model = onnx.load(SRC_ONNX)

    print("\n[patch] Step A — Patching static initializer outliers...")
    n_init_patched = patch_static_initializers(model, clip_val)
    print(f"  → Patched {n_init_patched} initializer(s)")

    print("\n[patch] Step B — ONNX graph surgery for dynamic mask nodes...")
    n_nodes_added = add_clip_node_after_mask_nodes(model, clip_val)
    print(f"  → Added {n_nodes_added} Clip node(s)")

    print("\n[patch] Step C — Patching missing Conv biases for QNN compatibility...")
    n_biases_added = patch_missing_conv_biases(model)
    print(f"  → Added {n_biases_added} dummy zero-bias(es) to Conv nodes")

    if n_init_patched == 0 and n_nodes_added == 0 and n_biases_added == 0:
        print("\n[patch] ℹ️  No patches needed — SenseVoice may not use extreme mask values")
        print("  Saving copy as model_patched.onnx anyway for consistency...")
        import shutil
        shutil.copy2(SRC_ONNX, DST_ONNX)
    else:
        print(f"\n[patch] Saving patched model → {DST_ONNX}...")
        onnx.save(model, DST_ONNX)

    print(f"[patch] ✅ Saved: {DST_ONNX} ({os.path.getsize(DST_ONNX)/1e6:.1f}MB)")

    # Load outlier report từ bước 1 nếu có
    report_path = os.path.join(OUT_DIR, "outlier_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        print(f"\n[patch] Outlier report from step 1: {len(report)} outliers")
        for r in report:
            print(f"  {r['name'][:60]}: min={r['min']:.2e}")

    if not args.skip_verify:
        cos_sim = verify_patch(SRC_ONNX, DST_ONNX)
    else:
        cos_sim = None

    print("\n" + "=" * 60)
    print("SUMMARY — Bước 2/6")
    print("=" * 60)
    print(f"  Initializers patched:  {n_init_patched}")
    print(f"  Clip nodes added:      {n_nodes_added}")
    print(f"  Conv biases added:     {n_biases_added}")
    if cos_sim is not None:
        print(f"  Patch cosine sim:      {cos_sim:.6f}")
    print(f"  Output:                {DST_ONNX}")
    print()
    print("Bước tiếp theo: chạy step4_s1_prepare_calib.py để tạo calibration data")


if __name__ == "__main__":
    main()

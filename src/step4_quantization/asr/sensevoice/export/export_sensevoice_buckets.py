# -*- coding: utf-8 -*-
"""SenseVoice-Small encoder+CTC graphs for SEVERAL static windows ("buckets"), same graph as step4_s1_export_sensevoice_split_npu.py
(static position encoding, CTC argmax + collapse + byte table inside the graph, conv-bias / mask-outlier patches) but with the window as a
parameter, cross-platform paths, and an optional parity check:

  python export_sensevoice_buckets.py --model FunAudioLLM/SenseVoiceSmall --out-dir out/sv_buckets --frames 50,100,200,500 [--check-wavs dir]

Inputs per graph: fbank (1, L, 560) float32, speech_lengths (1,) int32, language (1,) int32, textnorm (1,) int32; output byte_stream (1, (L+4)*24).
One LFR frame = 60 ms, so L = 50 / 100 / 200 / 500 hold 3 / 6 / 12 / 30 s. s2s_edge runs the smallest window that holds the utterance
(models.json "asr_xx_encoder" + "asr_xx_encoder@<L>") and computes the features on the host (s2s_edge.asr.sv_features, exact FunASR front end),
so no frontend graph is needed. Precision on the chip: W8A16 (as the shipped 500-frame graph, calibration fbank cut to L frames) or fp16
(weights shared across windows in one link) - not measured yet, A/B both (docs/asr_optimization.md)."""
import argparse, glob, os, sys, time, types
import numpy as np
import torch, onnx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_s1_export_sensevoice_e2e_unified import StaticSinusoidalPositionEncoder, StaticCTCCollapseAndDetokenizer, build_static_byte_table, L_MAX
from step4_s1_export_sensevoice_split_npu import EncoderClassifierGraph, patch_conv_bias, patch_mask_outliers


def export_window(sv, byte_table, L, out_dir):
    sv.encoder.embed = StaticSinusoidalPositionEncoder(timesteps=L + 4, depth=560)
    g = EncoderClassifierGraph(sv, StaticCTCCollapseAndDetokenizer(byte_table, max_frames=L + 4, l_max=L_MAX).eval()).eval()
    raw = os.path.join(out_dir, f"sv_enc_cls_{L}_raw.onnx"); tmp = os.path.join(out_dir, f"sv_enc_cls_{L}_bias.onnx"); dst = os.path.join(out_dir, f"sv_enc_cls_{L}.onnx")
    args = (torch.randn(1, L, 560), torch.tensor([L], dtype=torch.int32), torch.tensor([4], dtype=torch.int32), torch.tensor([14], dtype=torch.int32))
    with torch.no_grad():
        torch.onnx.export(g, args, raw, input_names=["fbank", "speech_lengths", "language", "textnorm"], output_names=["byte_stream"],
                          dynamic_axes={}, opset_version=17, do_constant_folding=True, dynamo=False)
    patch_conv_bias(raw, tmp); patch_mask_outliers(tmp, dst, clip_val=-30.0)
    m = onnx.load(dst, load_external_data=True)
    from onnx import numpy_helper as nph
    for node in m.graph.node:                                          # newer torch exports masked_fill(-inf) as Constant nodes, which
        if node.op_type != "Constant": continue                       # patch_mask_outliers (initializers only) misses: clip them too
        for at in node.attribute:
            if at.name == "value":
                v = nph.to_array(at.t)
                if v.dtype.kind == "f" and v.size and float(v.min()) < -30.0: at.t.CopyFrom(nph.from_array(np.maximum(v, -30.0).astype(v.dtype), at.t.name))
    # patch_conv_bias pins the output to the 500-frame length: set this window's
    d = m.graph.output[0].type.tensor_type.shape.dim; d[0].ClearField("dim_param"); d[0].dim_value = 1; d[1].ClearField("dim_param"); d[1].dim_value = (L + 4) * L_MAX
    for p in (raw, tmp, tmp + ".data", dst + ".data"):                  # onnx appends to an existing .data file: remove it, then save one
        if os.path.exists(p): os.remove(p)                             # self-contained file (~0.94 GB < 2 GB protobuf limit, AI Hub upload as is)
    onnx.save_model(m, dst + ".tmp", save_as_external_data=False); os.replace(dst + ".tmp", dst)
    return dst


def check(paths, cmvn_path, wav_dir):
    """Every window vs the largest one on clips that fit, with the host features of s2s_edge: identical transcript expected."""
    import onnxruntime as ort, soundfile as sf
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5, "s2s_edge"))
    from s2s_edge.asr import sv_features, load_cmvn
    cmvn = load_cmvn(cmvn_path); so = ort.SessionOptions(); so.log_severity_level = 3
    sess = {L: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for L, p in paths.items()}; Lmax = max(sess)
    def run(L, x):
        n = min(len(x), L); b = np.zeros((1, L, 560), np.float32); b[0, :n] = x[:n]
        a = sess[L].run(None, {"fbank": b, "speech_lengths": np.array([n], np.int32), "language": np.array([4], np.int32), "textnorm": np.array([14], np.int32)})[0]
        return bytes(int(v) & 0xFF for v in a.reshape(-1)).replace(b"\x00", b"").decode("utf-8", errors="replace").strip()
    stats = {L: [0, 0] for L in sess if L != Lmax}
    for p in sorted(glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True))[:60]:
        w, sr = sf.read(p, dtype="float32"); x = sv_features(w if w.ndim == 1 else w.mean(1), cmvn); ref = run(Lmax, x)
        for L in stats:
            if len(x) <= L: s = stats[L]; s[0] += 1; h = run(L, x); s[1] += h == ref; (h != ref) and print(f"    {os.path.basename(p)} L={L}: {h!r} vs {ref!r}")
    for L, (c, same) in sorted(stats.items()): print(f"  window {L:4d}: {c} clips fit, identical transcript {same}/{c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="FunAudioLLM/SenseVoiceSmall", help="hub id or a local folder with model.pt / config.yaml / am.mvn")
    ap.add_argument("--out-dir", required=True); ap.add_argument("--frames", default="50,100,200,500"); ap.add_argument("--check-wavs", default="")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    from funasr import AutoModel
    am = AutoModel(model=a.model, hub="hf", device="cpu", disable_update=True) if not os.path.isdir(a.model) else AutoModel(model=a.model, device="cpu", disable_update=True)
    sv = am.model.eval(); sp = am.kwargs.get("tokenizer").sp
    byte_table = build_static_byte_table(sp, vocab_size=sp.get_piece_size(), l_max=L_MAX)
    from funasr.models.sense_voice.export_meta import export_rebuild_model
    sv = export_rebuild_model(sv, device="cpu", max_seq_len=512); sv.export_dynamic_axes = types.MethodType(lambda self: {}, sv)
    paths = {}
    for L in sorted(int(x) for x in a.frames.split(",")):
        t = time.time(); paths[L] = export_window(sv, byte_table, L, a.out_dir)
        print(f"saved {paths[L]} window {L} LFR frames = {L * 0.06:.1f} s ({time.time() - t:.0f} s)", flush=True)
    if a.check_wavs:
        cm = os.path.join(a.model, "am.mvn") if os.path.isdir(a.model) else os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5, "s2s_edge", "assets", "sensevoice_am.mvn")
        check(paths, cm, a.check_wavs)


if __name__ == "__main__":
    main()

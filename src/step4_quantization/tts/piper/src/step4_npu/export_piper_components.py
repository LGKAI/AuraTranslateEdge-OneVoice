#!/usr/bin/env python3
"""Step 4 NPU -- Export Piper (vi) as 4 static-shape ONNX components (NPU recipe).

Ports Qualcomm's official PiperTTS recipe (qualcomm/ai-hub-models,
pipertts_*) to the Vietnamese voice. The full end-to-end Piper VITS graph is
NOT HTP-compilable (dynamic shapes: NonZero/Range/ScatterND in the flow
decoder, dynamic audio length, 1-D conv layout issues). Qualcomm solves this
by splitting the model into FOUR fixed-shape sub-models, each compiled
separately for HTP, with the (non-neural) alignment/window glue on the host:

  1. encoder   : text encoder        (1,512) int32  -> x_encoded/m_p/logs_p/x_mask
  2. sdp       : duration predictor  (deterministic noise; one flow layer dropped)
  3. flow      : normalizing flow    (reverse, static attn_squeezed input)
  4. decoder   : HiFi-GAN vocoder    (1,192,64) chunk -> audio chunk

Everything neural runs on the NPU; the host only does phonemization (espeak),
argmax alignment (generate_path) and window assembly.

Usage:
    python export_piper_components.py --onnx outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx \
        --config outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx.json \
        --out_dir outputs/piper_vi_npu/components
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fixed shapes from the official recipe
MAX_SEQ_LEN = 512
ENCODER_HIDDEN_DIM = 192
UPSAMPLED_MAX_SEQ_LEN = 1536  # MAX_SEQ_LEN * 3
DEC_SEQ_OVERLAP = 12
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_LEN = MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP  # 64
UPSAMPLE_FACTOR = 256
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_SCALE_W = 0.8

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_train.vits.models import SynthesizerTrn  # noqa: E402


def build_model_from_onnx(onnx_path: str, config_path: str) -> SynthesizerTrn:
    """Reconstruct the PyTorch SynthesizerTrn from the piper ONNX + config.

    Ported from qualcomm/ai-hub-models .../_shared/pipertts/util.py (the
    Italian ONNX-based path), which is exactly our case (vi voice ships as
    ONNX, no .ckpt checkpoint).
    """
    import onnx
    from onnx import numpy_helper

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    num_symbols = config["num_symbols"]  # 256
    num_speakers = config["num_speakers"]  # 1
    sample_rate = config["audio"]["sample_rate"]  # 22050
    logger.info("Config: num_symbols=%d, num_speakers=%d, sample_rate=%d",
                num_symbols, num_speakers, sample_rate)

    model_g = SynthesizerTrn(
        n_vocab=num_symbols,
        spec_channels=513,
        segment_size=32,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=768,
        n_heads=2,
        n_layers=6,
        kernel_size=3,
        p_dropout=0.1,
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        n_speakers=num_speakers,
        gin_channels=0,
        use_sdp=True,
    )

    model_g.eval()
    with torch.no_grad():
        model_g.dec.remove_weight_norm()
        for flow_layer in model_g.flow.flows:
            if hasattr(flow_layer, "enc"):
                flow_layer.enc.remove_weight_norm()

    logger.info("Loading weights from ONNX: %s", onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx_weights = {
        init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
        for init in onnx_model.graph.initializer
    }
    logger.info("Found %d weight tensors in ONNX", len(onnx_weights))

    state_dict = model_g.state_dict()
    name_map = {}
    special = {}

    # 1. Direct name matches
    for pt_key in state_dict:
        if pt_key in onnx_weights:
            name_map[pt_key] = pt_key

    # dp -> sdp / sdp -> dp prefix fallback (whichever direction the export used)
    for pt_key in state_dict:
        if pt_key not in name_map and pt_key.startswith("dp."):
            alt = "sdp." + pt_key[3:]
            if alt in onnx_weights:
                name_map[pt_key] = alt
        if pt_key not in name_map and pt_key.startswith("sdp."):
            alt = "dp." + pt_key[4:]
            if alt in onnx_weights:
                name_map[pt_key] = alt

    # 3. Embedding stored as 'sid'
    emb_key = "enc_p.emb.weight"
    if emb_key not in name_map:
        target = [num_symbols, 192]
        hits = [(k, v) for k, v in onnx_weights.items()
                if list(v.shape) == target and k not in name_map.values()]
        if hits:
            found, _ = hits[0]
            name_map[emb_key] = found
            logger.info("Embedding -> ONNX key '%s'", found)
        else:
            logger.warning("Embedding not found - audio will be noise!")

    # 4. Flow WN weights: piper's ONNX export names these onnx::Conv_*.
    #    The vi_VN-vais1000-medium export shares the exact auto-numbering of
    #    Qualcomm's IT voice (onnx::Conv_8168..8261, verified against the file),
    #    so the official hardcoded mapping applies unchanged.
    flow_onnx_keys = [
        "onnx::Conv_8168", "onnx::Conv_8171", "onnx::Conv_8174", "onnx::Conv_8177",
        "onnx::Conv_8180", "onnx::Conv_8183", "onnx::Conv_8186", "onnx::Conv_8189",
        "onnx::Conv_8192", "onnx::Conv_8195", "onnx::Conv_8198", "onnx::Conv_8201",
        "onnx::Conv_8204", "onnx::Conv_8207", "onnx::Conv_8210", "onnx::Conv_8213",
        "onnx::Conv_8216", "onnx::Conv_8219", "onnx::Conv_8222", "onnx::Conv_8225",
        "onnx::Conv_8228", "onnx::Conv_8231", "onnx::Conv_8234", "onnx::Conv_8237",
        "onnx::Conv_8240", "onnx::Conv_8243", "onnx::Conv_8246", "onnx::Conv_8249",
        "onnx::Conv_8252", "onnx::Conv_8255", "onnx::Conv_8258", "onnx::Conv_8261",
    ]
    flow_pt_keys = []
    for fi in [6, 4, 2, 0]:
        for li in range(4):
            flow_pt_keys.append(f"flow.flows.{fi}.enc.in_layers.{li}.weight")
            flow_pt_keys.append(f"flow.flows.{fi}.enc.res_skip_layers.{li}.weight")
    for pt_key, onnx_key in zip(flow_pt_keys, flow_onnx_keys, strict=False):
        if onnx_key in onnx_weights:
            name_map[pt_key] = onnx_key

    # 5. dp.flows.0.logs stored as exp(-logs) -> recover with -log().
    #    In the IT/EN exports this is onnx::Exp_8159; the vi export carries it
    #    as an onnx::Exp_* constant too. Search for any scalar 'Exp'-named init.
    exp_cands = [k for k in onnx_weights if "Exp" in k]
    if exp_cands:
        cand = exp_cands[0]
        arr = onnx_weights[cand]
        if arr.numel() == 1:
            special["dp.flows.0.logs"] = -torch.log(arr.flatten()[0])
            logger.info("dp.flows.0.logs recovered from %s = %s", cand, float(arr.flatten()[0]))

    # Load state dict
    new_state_dict = {}
    missing = []
    for pt_key in state_dict:
        if pt_key in special:
            new_state_dict[pt_key] = special[pt_key]
        elif pt_key in name_map:
            new_state_dict[pt_key] = onnx_weights[name_map[pt_key]]
        else:
            missing.append(pt_key)
            new_state_dict[pt_key] = state_dict[pt_key]

    logger.info("Loaded %d / %d parameters",
                len(state_dict) - len(missing), len(state_dict))

    expected_missing = {"enc_q.", "dp.post_", "dp.flows.1.", "dp.flows.0.logs"}
    real_missing = [k for k in missing if not any(k.startswith(p) for p in expected_missing)]
    if real_missing:
        raise RuntimeError(f"Missing {len(real_missing)} inference-critical parameters during weight conversion: {real_missing[:10]}")
    logger.info("All inference-critical weights loaded successfully!")

    model_g.load_state_dict(new_state_dict, strict=True)
    model_g.eval()
    return model_g


class Encoder(torch.nn.Module):
    def __init__(self, gen: SynthesizerTrn):
        super().__init__()
        self.gen = gen

    def forward(self, x, x_lengths):
        x_encoded, m_p, logs_p, x_mask = self.gen.enc_p(x, x_lengths)
        return x_encoded, m_p, logs_p, x_mask


class SDP(torch.nn.Module):
    """Duration predictor with deterministic noise (constant pattern)."""

    def __init__(self, gen: SynthesizerTrn, speed_adjustment: float = 1.0):
        super().__init__()
        self.gen = gen
        self.register_buffer("sdp_noise_pattern", torch.ones(1, 2, MAX_SEQ_LEN) * 0.5)
        self.scale = 1.0 / speed_adjustment

    def forward(self, x_encoded, x_mask, length_scale, noise_scale_w):
        gen = self.gen
        g = None
        if gen.use_sdp:
            dp = gen.dp
            dp_x = dp.pre(torch.detach(x_encoded))
            dp_x = dp.convs(dp_x, x_mask)
            dp_x = dp.proj(dp_x) * x_mask

            sdp_noise = self.sdp_noise_pattern[:, :, : x_encoded.shape[2]]
            z = sdp_noise.expand(x_encoded.size(0), -1, -1).to(x_encoded.device) * noise_scale_w

            flows = list(reversed(dp.flows))
            flows = [*flows[:-2], flows[-1]]  # drop flows[-2] for speed (official recipe)
            for flow in flows:
                z = flow(z, x_mask, g=dp_x, reverse=True)
            z0, _ = torch.split(z, [1, 1], 1)
            logw = z0
        else:
            logw = gen.dp(x_encoded, x_mask, g=g)

        logw = logw + x_encoded.sum() * 0  # keep x_encoded in the graph
        w = torch.exp(logw + torch.log(self.scale * length_scale)) * x_mask
        w_ceil = torch.ceil(w)
        y_lengths = torch.sum(torch.sum(w_ceil, dim=2), dim=1)
        return y_lengths, w_ceil


class Flow(torch.nn.Module):
    def __init__(self, gen: SynthesizerTrn):
        super().__init__()
        self.flow = gen.flow
        hidden_channels = gen.hidden_channels
        self.register_buffer(
            "fixed_noise", torch.randn(1, hidden_channels, UPSAMPLED_MAX_SEQ_LEN) * 0.5
        )

    def forward(self, m_p, logs_p, y_mask, attn_squeezed, noise_scale):
        m_p = torch.matmul(m_p, attn_squeezed.transpose(1, 2))
        logs_p = torch.matmul(logs_p, attn_squeezed.transpose(1, 2))
        z_p = m_p + self.fixed_noise * torch.exp(logs_p) * noise_scale
        return self.flow(z_p, y_mask, g=None, reverse=True)


class Decoder(torch.nn.Module):
    """HiFi-GAN vocoder (original piper Generator, conv_post bias=False).

    Note: the official Qualcomm repo defines a Generator_Mod wrapper that adds
    a bias to conv_post, but it is not actually used by the Decoder component
    -- the export uses the original Generator with ONNX-loaded weights.
    """

    def __init__(self, gen: SynthesizerTrn):
        super().__init__()
        self.dec = gen.dec

    def forward(self, z):
        return self.dec(z, g=None)


def export_component(model, path: Path, inputs: dict, name: str, opset: int = 15):
    """Export a component to a static-shape ONNX file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        str(path),
        input_names=list(inputs.keys()),
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Exported %s -> %s", name, path)


def main():
    parser = argparse.ArgumentParser(description="Export Piper (vi) as 4 NPU components")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/piper_vi_npu/components"))
    args = parser.parse_args()

    model_g = build_model_from_onnx(str(args.onnx), str(args.config))

    torch.manual_seed(0)
    np.random.seed(0)

    # Sanity: run a forward pass of enc_p to make sure weights are loaded
    with torch.no_grad():
        x = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
        x[0, :5] = torch.tensor([1, 14, 15, 16, 2])
        x_len = torch.tensor([5], dtype=torch.int64)
        x_encoded, m_p, logs_p, x_mask = model_g.enc_p(x, x_len)
        logger.info("enc_p forward OK: %s %s", tuple(x_encoded.shape), tuple(x_mask.shape))

    enc = Encoder(model_g)
    sdp = SDP(model_g, speed_adjustment=1.0)
    flow = Flow(model_g)
    dec = Decoder(model_g)

    # Exports (fixed shapes, matching the official recipe)
    export_component(enc, args.out_dir / "piper_vi_encoder.onnx",
                     {"x": torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int32),
                      "x_lengths": torch.zeros(1, dtype=torch.int32)},
                     "encoder")
    export_component(sdp, args.out_dir / "piper_vi_sdp.onnx",
                     {"x_encoded": torch.zeros(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                      "x_mask": torch.zeros(1, 1, MAX_SEQ_LEN),
                      "length_scale": torch.zeros(1),
                      "noise_scale_w": torch.zeros(1)},
                     "sdp")
    export_component(flow, args.out_dir / "piper_vi_flow.onnx",
                     {"m_p": torch.zeros(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                      "logs_p": torch.zeros(1, ENCODER_HIDDEN_DIM, MAX_SEQ_LEN),
                      "y_mask": torch.zeros(1, 1, UPSAMPLED_MAX_SEQ_LEN),
                      "attn_squeezed": torch.zeros(1, UPSAMPLED_MAX_SEQ_LEN, MAX_SEQ_LEN),
                      "noise_scale": torch.zeros(1)},
                     "flow")
    export_component(dec, args.out_dir / "piper_vi_decoder.onnx",
                     {"z": torch.zeros(1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN)},
                     "decoder")

    logger.info("All 4 components exported to %s", args.out_dir)


if __name__ == "__main__":
    main()

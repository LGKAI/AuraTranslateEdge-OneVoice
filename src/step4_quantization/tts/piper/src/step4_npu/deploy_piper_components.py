#!/usr/bin/env python3
"""Step 4 NPU -- Deploy Piper (vi) 4 components to Qualcomm AI Hub (IQ-9075 EVK).

Compiles the 4 static-shape ONNX components (encoder/sdp/flow/decoder) for the
Dragonwing IQ-9075 EVK HTP (Hexagon v73+) and runs hardware inference.

Strategy (official Qualcomm PiperTTS recipe adapted to vi):
  - All 4 components compile to qnn_context_binary (HTP/NPU-only).
  - float precision -> fp16 on HTP (no calibration needed).
  - encoder/sdp/flow: int32/float32 I/O -> --truncate_64bit_tensors
    --truncate_64bit_io (official options).
  - decoder: --quantize_io (official option for the vocoder).

Usage:
    python deploy_piper_components.py --step compile [--only encoder]
    python deploy_piper_components.py --step quantize     # w8a16 (optional)
    python deploy_piper_components.py --step inference [--only encoder]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# Windows sandbox: tempfile.TemporaryDirectory cleanup can fail with
# PermissionError when a file handle is still held (qai-hub keeps handles
# open during upload/download). The temp data is only scratch space, so
# ignoring cleanup errors is safe.
_orig_td = tempfile.TemporaryDirectory


class _SafeTemporaryDirectory(_orig_td):
    def cleanup(self):
        try:
            super().cleanup()
        except Exception:  # noqa: BLE001
            pass


tempfile.TemporaryDirectory = _SafeTemporaryDirectory

# Under the DSH file sandbox, directories created by tempfile.mkdtemp get a
# restrictive ACL (Windows) that even the owner cannot write into, which makes
# qai-hub's upload/download temp files fail with PermissionError. Create temp
# dirs with plain os.makedirs (which produces a normal, writable ACL) instead.
import uuid as _uuid


def _safe_mkdtemp(*args, **kwargs):
    import tempfile as _tf

    base = kwargs.pop("dir", None) or _tf.gettempdir()
    prefix = kwargs.pop("prefix", "tmp")
    suffix = kwargs.pop("suffix", "")
    d = os.path.join(base, f"{prefix}{_uuid.uuid4().hex[:10]}{suffix}")
    os.makedirs(d, exist_ok=False)
    return d


tempfile.mkdtemp = _safe_mkdtemp

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE_NAME = "Dragonwing IQ-9075 EVK"
COMPONENTS = [
    "byte_text_encoder",
    "encoder",
    "sdp",
    "monotonic_aligner",
    "flow",
    "decoder",
    "overlap_add",
    "audio_resampler",
]

# Official compile options per component
COMPILE_OPTIONS = {
    "byte_text_encoder": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
    "encoder": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
    "sdp": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
    "monotonic_aligner": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
    "flow": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
    "decoder": "--target_runtime qnn_dlc --quantize_io",
    "overlap_add": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
    "audio_resampler": "--target_runtime qnn_dlc --truncate_64bit_tensors --truncate_64bit_io",
}

MODEL_FILES = {
    "byte_text_encoder": "byte_text_encoder.onnx",
    "encoder": "piper_vi_encoder.onnx",
    "sdp": "piper_vi_sdp.onnx",
    "monotonic_aligner": "monotonic_aligner.onnx",
    "flow": "piper_vi_flow.onnx",
    "decoder": "piper_vi_decoder.onnx",
    "overlap_add": "overlap_add.onnx",
    "audio_resampler": "audio_resampler.onnx",
}


def _is_success(status) -> bool:
    code = getattr(status, "code", None) or str(status)
    return str(code).lower() in ("completed", "success", "successful")


def _save_job_log(output_dir: Path, record: dict):
    log_path = output_dir / "job_log_components.json"
    log = []
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            log = []
    log.append(record)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _fail_and_exit(job, output_dir: Path, prefix: str, status):
    code = getattr(status, "code", None) or str(status)
    logger.error("%s job failed: %s (status=%s)", prefix, job.url, code)
    try:
        job.download_job_logs(str(output_dir / f"{prefix}_logs"))
    except Exception as e:
        logger.error("log download failed: %s", e)
    sys.exit(1)


def step_compile(output_dir: Path, only: str | None = None):
    import qai_hub as hub

    comps = [only] if only else COMPONENTS
    device = hub.Device(DEVICE_NAME)
    for comp in comps:
        model_path = output_dir / "components" / MODEL_FILES[comp]
        if not model_path.exists():
            logger.error("model missing: %s", model_path)
            sys.exit(1)
        logger.info("Compiling %s for %s", comp, DEVICE_NAME)
        logger.info("  options: %s", COMPILE_OPTIONS[comp])
        job = hub.submit_compile_job(
            model=str(model_path),
            device=device,
            options=COMPILE_OPTIONS[comp],
            name=f"piper_vi_{comp}_iq9075",
        )
        logger.info("  submitted: %s (%s)", job.job_id, job.url)
        _save_job_log(output_dir, {"job_type": "compile", "component": comp,
                                   "job_id": job.job_id, "url": job.url,
                                   "status": "submitted", "options": COMPILE_OPTIONS[comp]})
        logger.info("  waiting...")
        job.wait()
        status = job.get_status()
        code = getattr(status, "code", None) or str(status)
        logger.info("  status code: %s", code)
        if not _is_success(status):
            _fail_and_exit(job, output_dir, f"compile_{comp}", status)
        bin_path = output_dir / "components" / f"piper_vi_{comp}.iq9075.bin"
        job.download_target_model(str(bin_path))
        logger.info("  QNN binary saved: %s (%.1f MB)",
                    bin_path, bin_path.stat().st_size / 1e6)
        _save_job_log(output_dir, {"job_type": "compile", "component": comp,
                                   "job_id": job.job_id, "status": str(code),
                                   "binary": str(bin_path)})


def step_quantize(output_dir: Path, only: str | None = None):
    """Optional w8a16 quantization (weights int8, activations int16)."""
    import qai_hub as hub

    comps = [only] if only else COMPONENTS
    for comp in comps:
        model_path = output_dir / "components" / MODEL_FILES[comp]
        calib_path = output_dir / "calib" / f"calib_{comp}.npz"
        if not model_path.exists():
            logger.error("model missing: %s", model_path)
            sys.exit(1)
        if not calib_path.exists():
            logger.warning("calibration missing for %s (generate_component_calibration.py)", comp)
            continue
        data = np.load(calib_path, allow_pickle=True)
        calib = {k: [data[k][i] for i in range(len(data[k]))] for k in data.files}
        logger.info("Quantizing %s (w8a16)", comp)
        job = hub.submit_quantize_job(
            model=str(model_path),
            calibration_data=calib,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,
            name=f"piper_vi_{comp}_w8a16",
        )
        logger.info("  submitted: %s (%s)", job.job_id, job.url)
        _save_job_log(output_dir, {"job_type": "quantize", "component": comp,
                                   "job_id": job.job_id, "url": job.url,
                                   "status": "submitted"})
        job.wait()
        status = job.get_status()
        code = getattr(status, "code", None) or str(status)
        logger.info("  status: %s", code)
        if not _is_success(status):
            _fail_and_exit(job, output_dir, f"quantize_{comp}", status)
        target = output_dir / "components" / f"piper_vi_{comp}_w8a16.onnx"
        job.download_target_model(str(target))
        logger.info("  quantized model saved: %s (%.1f MB)",
                    target, target.stat().st_size / 1e6)
        _save_job_log(output_dir, {"job_type": "quantize", "component": comp,
                                   "job_id": job.job_id, "status": str(code),
                                   "target": str(target)})


def load_test_inputs(output_dir: Path, comp: str):
    """Load hardware test inputs for a component (from calib/test npz)."""
    test_path = output_dir / "calib" / f"test_{comp}.npz"
    if not test_path.exists():
        logger.error("test inputs missing: %s", test_path)
        sys.exit(1)
    data = np.load(test_path, allow_pickle=True)
    return {k: [data[k][i] for i in range(len(data[k]))] for k in data.files}


def step_inference(output_dir: Path, only: str | None = None):
    import qai_hub as hub

    comps = [only] if only else COMPONENTS
    device = hub.Device(DEVICE_NAME)
    for comp in comps:
        # prefer compiled binary
        model_path = output_dir / "components" / f"piper_vi_{comp}.iq9075.bin"
        if not model_path.exists():
            model_path = output_dir / "components" / MODEL_FILES[comp]
        inputs = load_test_inputs(output_dir, comp)
        logger.info("Inference %s on %s (%d samples)", comp, DEVICE_NAME, len(list(inputs.values())[0]))
        job = hub.submit_inference_job(
            model=str(model_path),
            device=device,
            inputs=inputs,
            name=f"piper_vi_{comp}_infer_iq9075",
        )
        logger.info("  submitted: %s (%s)", job.job_id, job.url)
        _save_job_log(output_dir, {"job_type": "inference", "component": comp,
                                   "job_id": job.job_id, "url": job.url,
                                   "status": "submitted"})
        job.wait()
        status = job.get_status()
        code = getattr(status, "code", None) or str(status)
        logger.info("  status: %s", code)
        if not _is_success(status):
            _fail_and_exit(job, output_dir, f"infer_{comp}", status)
        out_dir = output_dir / "hw_outputs" / comp
        out_dir.mkdir(parents=True, exist_ok=True)
        job.download_output_data(str(out_dir))
        logger.info("  outputs downloaded to %s", out_dir)
        _save_job_log(output_dir, {"job_type": "inference", "component": comp,
                                   "job_id": job.job_id, "status": str(code),
                                   "output_dir": str(out_dir)})


def main():
    parser = argparse.ArgumentParser(description="Deploy Piper (vi) components to AI Hub")
    parser.add_argument("--step", required=True, choices=["compile", "quantize", "inference"])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/piper_vi_npu"))
    parser.add_argument("--only", choices=COMPONENTS, help="Only one component")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.step == "compile":
        step_compile(args.output_dir, args.only)
    elif args.step == "quantize":
        step_quantize(args.output_dir, args.only)
    elif args.step == "inference":
        step_inference(args.output_dir, args.only)


if __name__ == "__main__":
    main()

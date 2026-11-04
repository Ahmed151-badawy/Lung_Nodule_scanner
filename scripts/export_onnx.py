"""
ONNX export and INT8 quantization (research prototype)
ONNX Export, Opset 17 Conformance & Dynamic INT8 Quantization
Targeting Sub-20ms CPU Execution
"""

import time
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure standard output can handle UTF-8 symbols across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import torch
import numpy as np

from model.model import NoduleCNN


def export_and_quantize(
    model_checkpoint: str = "model/best_model.pth",
    output_dir: str = "dist/onnx",
    input_resolution: tuple = (1, 1, 256, 256)
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fp32_onnx = str(out_path / "nodule_cnn_fp32.onnx")
    int8_onnx = str(out_path / "nodule_cnn_int8.onnx")

    # 1. Load PyTorch model
    print(f"[1/4] Loading PyTorch model for architecture (input channels: {input_resolution[1]})...")
    model = NoduleCNN(in_channels=input_resolution[1], num_classes=1)

    ckpt_p = Path(model_checkpoint)
    if ckpt_p.exists():
        try:
            ckpt = torch.load(ckpt_p, map_location="cpu", weights_only=True)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"[OK] Loaded weights from {model_checkpoint}")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint weights ({e}). Exporting baseline architecture.")
    else:
        print(f"[INFO] Checkpoint '{model_checkpoint}' not found. Exporting calibrated baseline architecture.")

    model.eval()

    # 2. Export FP32 ONNX
    dummy_input = torch.randn(*input_resolution, dtype=torch.float32)
    print(f"[2/4] Exporting FP32 ONNX graph to {fp32_onnx}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            fp32_onnx,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False
        )
        print(f"[OK] FP32 model created: {fp32_onnx} ({Path(fp32_onnx).stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"[WARN] ONNX export encountered dependency requirement: {e}")
        print("       Run: pip install onnx onnxruntime onnxscript to enable ONNX runtime execution.")
        return

    # 3. Dynamic INT8 Quantization via onnxruntime
    print("[3/4] Applying Dynamic INT8 Quantization...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=fp32_onnx,
            model_output=int8_onnx,
            op_types_to_quantize=['MatMul', 'Gemm'],
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True
        )
        print(f"[OK] Quantized model created: {int8_onnx} ({Path(int8_onnx).stat().st_size / 1024:.1f} KB)")
        
        # 4. Latency Benchmark Comparison
        benchmark_runtimes(fp32_onnx, int8_onnx, dummy_input.numpy())

    except ImportError:
        print("[INFO] onnxruntime or onnxruntime.quantization not installed. Skipping INT8 quantization step.")


def benchmark_runtimes(fp32_path: str, int8_path: str, sample_input: np.ndarray, iterations: int = 50):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skipping latency benchmark.")
        return

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    # Warmup
    for _ in range(5):
        sess_fp32.run(None, {"input": sample_input})
        sess_int8.run(None, {"input": sample_input})

    # Benchmark FP32
    t0 = time.perf_counter()
    for _ in range(iterations):
        sess_fp32.run(None, {"input": sample_input})
    fp32_time = (time.perf_counter() - t0) / iterations * 1000.0

    # Benchmark INT8
    t0 = time.perf_counter()
    for _ in range(iterations):
        sess_int8.run(None, {"input": sample_input})
    int8_time = (time.perf_counter() - t0) / iterations * 1000.0

    fp32_sz = Path(fp32_path).stat().st_size / (1024 * 1024)
    int8_sz = Path(int8_path).stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("CLINICAL INFERENCE BENCHMARK (CPU Execution Provider)")
    print("=" * 60)
    print(f"FP32 Model:  {fp32_sz:.2f} MB  |  Latency: {fp32_time:.2f} ms / slice")
    print(f"INT8 Model:  {int8_sz:.2f} MB  |  Latency: {int8_time:.2f} ms / slice")
    print(f"Speedup:     {fp32_time / max(int8_time, 1e-6):.2f}x faster  |  Compression: {fp32_sz / max(int8_sz, 1e-6):.2f}x smaller")
    print("═" * 60)


if __name__ == "__main__":
    export_and_quantize()

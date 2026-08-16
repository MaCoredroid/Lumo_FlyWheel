"""NVFP4 kernel smoke on GB10/sm_121a inside the pinned vLLM image.

Go/no-go for the FR14 Qwen3.8-27B NVFP4 port: exercises scaled_fp4_quant and
cutlass_scaled_fp4_mm (the ops nvfp4_scaled_mm_sm120_kernels.cu.o provides) on
toy tensors and checks numerical sanity against a bf16 reference. The recorded
Sprint-0.5-era risk is an ARM64 CUDA illegal-instruction crash; a clean PASS
here retires it for the pinned image.
"""
import json
import sys
import traceback

out = {"stage": "import", "pass": False}

def emit():
    print("NVFP4_SMOKE_RESULT " + json.dumps(out), flush=True)

try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = torch.version.cuda
    out["device"] = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    out["capability"] = list(cap)

    import vllm
    out["vllm"] = vllm.__version__
    from vllm import _custom_ops as ops

    out["stage"] = "supports_fp4_probe"
    cc_int = cap[0] * 10 + cap[1]
    try:
        out["cutlass_scaled_mm_supports_fp4"] = bool(
            ops.cutlass_scaled_mm_supports_fp4(cc_int)
        )
    except Exception as e:  # helper may not exist in this build
        out["cutlass_scaled_mm_supports_fp4"] = f"probe-error: {e}"

    out["stage"] = "quant"
    torch.manual_seed(0)
    m, n, k = 32, 256, 512
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = 448.0
    a_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a.abs().amax().to(torch.float32)
    b_gs = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / b.abs().amax().to(torch.float32)
    a_q, a_sf = ops.scaled_fp4_quant(a, a_gs)
    b_q, b_sf = ops.scaled_fp4_quant(b, b_gs)
    torch.cuda.synchronize()
    out["a_q_shape"] = list(a_q.shape)
    out["a_sf_shape"] = list(a_sf.shape)

    out["stage"] = "gemm"
    alpha = 1.0 / (a_gs * b_gs)
    res = ops.cutlass_scaled_fp4_mm(a_q, b_q, a_sf, b_sf, alpha, torch.bfloat16)
    torch.cuda.synchronize()
    out["out_shape"] = list(res.shape)

    out["stage"] = "check"
    ref = (a.to(torch.float32) @ b.to(torch.float32).t())
    rel = ((res.to(torch.float32) - ref).abs().mean() / ref.abs().mean()).item()
    out["mean_rel_err"] = rel
    # both operands fp4-quantized -> expect a few percent; anything sane < 0.15
    out["pass"] = rel < 0.15

    out["stage"] = "done"
except Exception:
    out["error"] = traceback.format_exc()
finally:
    emit()
    sys.exit(0 if out.get("pass") else 1)

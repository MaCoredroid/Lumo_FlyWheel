#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


VOCAB = 65_536
HIDDEN = 5_120
WEIGHT_BLOCK = [128, 128]
CALLS_PER_EVENT = 5
FP8_WEIGHT_BYTES = VOCAB * HIDDEN
FP32_SCALE_BYTES = (VOCAB // 128) * (HIDDEN // 128) * 4
MANDATORY_BYTES_PER_CALL = FP8_WEIGHT_BYTES + FP32_SCALE_BYTES


def _event_ms(fn, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / repeats


def _tensor_contract(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "element_size": tensor.element_size(),
        "contiguous": tensor.is_contiguous(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("this smoke is qualified only on SM121")

    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        cutlass_scaled_mm,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        CUTLASS_BLOCK_FP8_SUPPORTED,
    )
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    if not CUTLASS_BLOCK_FP8_SUPPORTED:
        raise RuntimeError("vLLM CUTLASS block-FP8 support is unavailable")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    with torch.inference_mode():
        weight = torch.empty(
            (VOCAB, HIDDEN), dtype=torch.bfloat16, device=device
        )
        weight.normal_(mean=0.0, std=0.02)
        torch.cuda.synchronize()
        quant_start = time.monotonic()
        qweight, weight_scale = per_block_cast_to_fp8(
            weight, block_size=WEIGHT_BLOCK, use_ue8m0=False
        )
        torch.cuda.synchronize()
        quant_ms = (time.monotonic() - quant_start) * 1000.0

        if (
            tuple(qweight.shape) != (VOCAB, HIDDEN)
            or tuple(qweight.stride()) != (HIDDEN, 1)
            or qweight.element_size() != 1
            or tuple(weight_scale.shape) != (VOCAB // 128, HIDDEN // 128)
            or tuple(weight_scale.stride()) != (HIDDEN // 128, 1)
            or weight_scale.dtype != torch.float32
        ):
            raise RuntimeError("quantized weight contract drifted")

        batches: dict[str, object] = {}
        for batch in (1, 4):
            hidden = torch.randn(
                (batch, HIDDEN), dtype=torch.bfloat16, device=device
            )

            def quantize_activation():
                return per_token_group_quant_fp8(
                    hidden,
                    128,
                    column_major_scales=True,
                    use_ue8m0=False,
                )

            aq, activation_scale = quantize_activation()
            if (
                tuple(aq.shape) != (batch, HIDDEN)
                or tuple(aq.stride()) != (HIDDEN, 1)
                or aq.element_size() != 1
                or tuple(activation_scale.shape) != (batch, HIDDEN // 128)
                or tuple(activation_scale.stride()) != (1, batch)
                or activation_scale.dtype != torch.float32
            ):
                raise RuntimeError(f"B{batch} activation contract drifted")

            def fp8_gemm():
                return cutlass_scaled_mm(
                    aq,
                    qweight,
                    activation_scale,
                    weight_scale,
                    WEIGHT_BLOCK,
                    torch.bfloat16,
                )

            def fp8_full_call():
                live_aq, live_as = quantize_activation()
                return cutlass_scaled_mm(
                    live_aq,
                    qweight,
                    live_as,
                    weight_scale,
                    WEIGHT_BLOCK,
                    torch.bfloat16,
                )

            def bf16_gemm():
                return torch.nn.functional.linear(hidden, weight)

            fp8_output = fp8_gemm()
            bf16_output = bf16_gemm()
            torch.cuda.synchronize()
            if (
                tuple(fp8_output.shape) != (batch, VOCAB)
                or tuple(fp8_output.stride()) != (VOCAB, 1)
                or fp8_output.dtype != torch.bfloat16
                or not bool(torch.isfinite(fp8_output).all().item())
            ):
                raise RuntimeError(f"B{batch} FP8 output contract drifted")

            fp8_gemm_ms = _event_ms(fp8_gemm, args.repeats)
            fp8_full_call_ms = _event_ms(fp8_full_call, args.repeats)
            bf16_gemm_ms = _event_ms(bf16_gemm, args.repeats)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    fp8_output.float().flatten(),
                    bf16_output.float().flatten(),
                    dim=0,
                ).item()
            )
            batches[str(batch)] = {
                "activation_q": _tensor_contract(aq),
                "activation_scale": _tensor_contract(activation_scale),
                "output": _tensor_contract(fp8_output),
                "fp8_gemm_ms_per_call": fp8_gemm_ms,
                "fp8_quant_plus_gemm_ms_per_call": fp8_full_call_ms,
                "fp8_five_call_ms": fp8_full_call_ms * CALLS_PER_EVENT,
                "bf16_gemm_ms_per_call": bf16_gemm_ms,
                "bf16_five_call_ms": bf16_gemm_ms * CALLS_PER_EVENT,
                "fp8_vs_bf16_speedup": bf16_gemm_ms / fp8_full_call_ms,
                "mandatory_weight_bandwidth_gbps": (
                    MANDATORY_BYTES_PER_CALL / fp8_gemm_ms / 1_000_000
                ),
                "fp8_bf16_output_cosine_informational": cosine,
                "argmax_equal_informational": bool(
                    torch.equal(
                        fp8_output.argmax(dim=-1),
                        bf16_output.argmax(dim=-1),
                    )
                ),
            }

    source = Path(__file__).read_bytes()
    payload = {
        "schema": "fr13.draft_head_fp8_sm121_smoke.v1",
        "status": "PASS",
        "classification": "kernel_smoke_not_acceptance",
        "production_default_enabled": False,
        "lossless_acceptance_claimed": False,
        "script_sha256": hashlib.sha256(source).hexdigest(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(),
        "device_capability": list(torch.cuda.get_device_capability()),
        "cutlass_block_fp8_supported": bool(CUTLASS_BLOCK_FP8_SUPPORTED),
        "seed": args.seed,
        "repeats": args.repeats,
        "weight_quantization_ms_one_time": quant_ms,
        "weight": _tensor_contract(weight),
        "qweight": _tensor_contract(qweight),
        "weight_scale": _tensor_contract(weight_scale),
        "traffic": {
            "calls_per_event": CALLS_PER_EVENT,
            "bf16_weight_bytes_per_call_removed": VOCAB * HIDDEN * 2,
            "fp8_weight_bytes_per_call": FP8_WEIGHT_BYTES,
            "fp32_weight_scale_bytes_per_call": FP32_SCALE_BYTES,
            "mandatory_bytes_per_call": MANDATORY_BYTES_PER_CALL,
            "mandatory_bytes_per_event": (
                CALLS_PER_EVENT * MANDATORY_BYTES_PER_CALL
            ),
            "candidate_full_step_mandatory_bytes": 30_989_326_208,
            "candidate_weight_floor_ms": 113.514015414,
            "one_sided_u95_cap_ms": 130.541117726,
        },
        "batches": batches,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

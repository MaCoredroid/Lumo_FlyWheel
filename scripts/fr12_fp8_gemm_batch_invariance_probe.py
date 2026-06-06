#!/usr/bin/env python3
"""Probe fp8 activation quantization batch-invariance on FR12 captures.

This is intentionally boot-free: it consumes saved sub-kernel tensors and does
not load model weights. It can prove whether the live per-token-group activation
quantizer changes a row when co-resident rows change. It cannot replay a full
fp8 GEMM unless a capture also includes the projection weights/scales.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


FALLBACK_TREE_SPINE_ROWS = [0, 1, 2, 4, 6]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _stage_tensor(capture: dict[str, Any], stage: str) -> torch.Tensor:
    try:
        return capture["stages"][stage]["tensor"]
    except KeyError as exc:
        raise KeyError(f"{capture.get('call_path', capture.get('path'))} missing {stage}") from exc


def _tensor_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list):
        return [int(x) for x in value]
    return None


def _tree_path0_rows(capture: dict[str, Any], depth_count: int) -> list[int]:
    detail = capture.get("meta", {}).get("tree_conv_detail", {})
    path0 = _tensor_list(detail.get("path0_nodes"))
    if path0:
        return path0[:depth_count]
    return FALLBACK_TREE_SPINE_ROWS[:depth_count]


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def _nonzero(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.numel() == 0:
        return 0
    return int((a.to(torch.float32) != b.to(torch.float32)).sum().item())


def _q_bytes(q: torch.Tensor) -> torch.Tensor:
    return q.contiguous().view(torch.uint8)


def _quantize(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    return fp8_utils.per_token_group_quant_fp8(
        x.contiguous(),
        group_size=group_size,
    )


def _quant_stats(
    full_x: torch.Tensor,
    rows: list[int],
    group_size: int,
    *,
    reverse_context: bool,
) -> dict[str, Any]:
    selected = torch.tensor(rows, device=full_x.device, dtype=torch.long)
    q_full, s_full = _quantize(full_x, group_size)
    q_rows, s_rows = _quantize(full_x.index_select(0, selected).contiguous(), group_size)

    q_full_rows = q_full.index_select(0, selected)
    s_full_rows = s_full.index_select(0, selected)
    out: dict[str, Any] = {
        "rows": [int(x) for x in rows],
        "full_shape": list(full_x.shape),
        "row_only_shape": list(q_rows.shape),
        "q_byte_mismatches_full_vs_row_only": int(
            (_q_bytes(q_full_rows) != _q_bytes(q_rows)).sum().item()
        ),
        "scale_max_abs_full_vs_row_only": _max_abs(s_full_rows, s_rows),
        "scale_mean_abs_full_vs_row_only": _mean_abs(s_full_rows, s_rows),
    }

    if reverse_context:
        reversed_x = full_x.flip(0).contiguous()
        reverse_rows = [int(full_x.shape[0] - 1 - r) for r in rows]
        reverse_selected = torch.tensor(reverse_rows, device=full_x.device, dtype=torch.long)
        q_rev, s_rev = _quantize(reversed_x, group_size)
        q_rev_rows = q_rev.index_select(0, reverse_selected)
        s_rev_rows = s_rev.index_select(0, reverse_selected)
        out.update(
            {
                "q_byte_mismatches_full_vs_reversed_context": int(
                    (_q_bytes(q_full_rows) != _q_bytes(q_rev_rows)).sum().item()
                ),
                "scale_max_abs_full_vs_reversed_context": _max_abs(
                    s_full_rows, s_rev_rows
                ),
                "scale_mean_abs_full_vs_reversed_context": _mean_abs(
                    s_full_rows, s_rev_rows
                ),
            }
        )

    return out


def _cross_quant_stats(
    tree_x: torch.Tensor,
    native_x: torch.Tensor,
    tree_rows: list[int],
    native_rows: list[int],
    group_size: int,
) -> dict[str, Any]:
    tree_idx = torch.tensor(tree_rows, device=tree_x.device, dtype=torch.long)
    native_idx = torch.tensor(native_rows, device=native_x.device, dtype=torch.long)
    q_tree, s_tree = _quantize(tree_x.index_select(0, tree_idx).contiguous(), group_size)
    q_native, s_native = _quantize(
        native_x.index_select(0, native_idx).contiguous(), group_size
    )
    return {
        "q_byte_mismatches_tree_vs_native_row_only": int(
            (_q_bytes(q_tree) != _q_bytes(q_native)).sum().item()
        ),
        "scale_max_abs_tree_vs_native_row_only": _max_abs(s_tree, s_native),
        "scale_mean_abs_tree_vs_native_row_only": _mean_abs(s_tree, s_native),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-capture", required=True, type=Path)
    parser.add_argument("--native-capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--input-stage", default="gate_out")
    parser.add_argument("--output-stage", default="o_proj_out")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--depth-count", type=int, default=5)
    parser.add_argument("--tree-rows", default="")
    parser.add_argument("--native-rows", default="")
    args = parser.parse_args()

    tree_cap = _load(args.tree_capture)
    native_cap = _load(args.native_capture)

    depth_count = int(args.depth_count)
    tree_rows = (
        [int(x) for x in args.tree_rows.split(",") if x.strip()]
        if args.tree_rows
        else _tree_path0_rows(tree_cap, depth_count)
    )
    native_rows = (
        [int(x) for x in args.native_rows.split(",") if x.strip()]
        if args.native_rows
        else list(range(depth_count))
    )
    if len(tree_rows) != len(native_rows):
        raise ValueError(f"row length mismatch: tree={tree_rows}, native={native_rows}")

    tree_in_cpu = _stage_tensor(tree_cap, args.input_stage).to(torch.bfloat16)
    native_in_cpu = _stage_tensor(native_cap, args.input_stage).to(torch.bfloat16)
    tree_out_cpu = _stage_tensor(tree_cap, args.output_stage).to(torch.bfloat16)
    native_out_cpu = _stage_tensor(native_cap, args.output_stage).to(torch.bfloat16)

    tree_idx_cpu = torch.tensor(tree_rows, dtype=torch.long)
    native_idx_cpu = torch.tensor(native_rows, dtype=torch.long)
    tree_in_rows = tree_in_cpu.index_select(0, tree_idx_cpu)
    native_in_rows = native_in_cpu.index_select(0, native_idx_cpu)
    tree_out_rows = tree_out_cpu.index_select(0, tree_idx_cpu)
    native_out_rows = native_out_cpu.index_select(0, native_idx_cpu)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: probe must exercise the live CUDA fp8 quantizer")
    device = torch.device("cuda")

    tree_in = tree_in_cpu.to(device)
    native_in = native_in_cpu.to(device)

    result: dict[str, Any] = {
        "schema": "fr12.fp8_gemm_batch_invariance_probe.v1",
        "tree_capture": str(args.tree_capture),
        "native_capture": str(args.native_capture),
        "input_stage": args.input_stage,
        "output_stage": args.output_stage,
        "group_size": int(args.group_size),
        "tree_rows": tree_rows,
        "native_rows": native_rows,
        "o_proj_boundary": {
            "input_max_abs_tree_vs_native": _max_abs(tree_in_rows, native_in_rows),
            "input_mean_abs_tree_vs_native": _mean_abs(tree_in_rows, native_in_rows),
            "input_nonzero_tree_vs_native": _nonzero(tree_in_rows, native_in_rows),
            "output_max_abs_tree_vs_native": _max_abs(tree_out_rows, native_out_rows),
            "output_mean_abs_tree_vs_native": _mean_abs(tree_out_rows, native_out_rows),
            "output_nonzero_tree_vs_native": _nonzero(tree_out_rows, native_out_rows),
        },
        "activation_quantizer": {
            "tree_full_vs_row_only": _quant_stats(
                tree_in, tree_rows, args.group_size, reverse_context=True
            ),
            "native_full_vs_row_only": _quant_stats(
                native_in, native_rows, args.group_size, reverse_context=True
            ),
            "tree_vs_native_row_only": _cross_quant_stats(
                tree_in, native_in, tree_rows, native_rows, args.group_size
            ),
        },
        "in_proj_boundary": {
            "status": "not_measured",
            "reason": (
                "existing subkernel captures start after in_proj (pre_conv) and do "
                "not include the decoder-layer hidden input or fp8 projection "
                "weights/scales"
            ),
        },
        "full_fp8_gemm_replay": {
            "status": "not_measured",
            "reason": (
                "captures contain o_proj input/output tensors but not RowParallelLinear "
                "fp8 weights or block scales; this probe only tests the live activation "
                "quantizer's batch invariance"
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

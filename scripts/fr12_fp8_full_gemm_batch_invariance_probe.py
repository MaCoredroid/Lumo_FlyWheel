#!/usr/bin/env python3
"""Boot-free Cutlass block-fp8 GEMM batch-invariance probe.

The earlier FR12 fp8 probe tested only activation quantization. This script
loads real Qwen3.6-FP8 layer-0 block-fp8 weights/scales and saved tree/native
inputs, then replays the live vLLM W8A8 block-fp8 Cutlass path in three M
contexts:

* full captured batch
* aligned spine rows only
* reversed full-batch row order

The gate is whether the same spine row's GEMM output changes when co-resident
rows change. This does not start a vLLM server or load the full model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


FALLBACK_TREE_SPINE_ROWS = [0, 1, 2, 4, 6]
DEFAULT_BRANCH_NODES = [3, 5, 7, 9]
WEIGHT_BLOCK = [128, 128]


def _csv_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


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


def _tree_parent(capture: dict[str, Any]) -> list[int] | None:
    for stage in ("gdn_scan_out", "conv1d_out", "pre_conv"):
        try:
            parent = capture["stages"][stage].get("extra", {}).get("tree_parent")
        except KeyError:
            continue
        parsed = _tensor_list(parent)
        if parsed:
            return parsed
    return None


def _node_path(parent: list[int], node: int) -> list[int]:
    path: list[int] = []
    cur = int(node)
    while cur >= 0:
        path.append(cur)
        cur = int(parent[cur])
    return list(reversed(path))


def _stage_tensor(capture: dict[str, Any], stage: str) -> torch.Tensor:
    try:
        return capture["stages"][stage]["tensor"]
    except KeyError as exc:
        raise KeyError(f"{capture.get('path')} missing stage {stage}") from exc


def _has_stage(capture: dict[str, Any], stage: str) -> bool:
    return stage in capture.get("stages", {})


def _layer_input(capture: dict[str, Any]) -> torch.Tensor:
    if "input_hidden" not in capture:
        raise KeyError(f"{capture.get('path', capture.keys())} missing input_hidden")
    return capture["input_hidden"]


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


def _load_weight_pair(
    model_dir: Path,
    layer: int,
    module: str,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    shard = model_dir / f"layers-{layer}.safetensors"
    weight_key = f"model.language_model.layers.{layer}.linear_attn.{module}.weight"
    scale_key = (
        f"model.language_model.layers.{layer}.linear_attn.{module}.weight_scale_inv"
    )
    with safe_open(shard, framework="pt", device="cpu") as f:
        if weight_key not in f.keys():
            raise KeyError(f"{weight_key} absent from {shard}")
        if scale_key not in f.keys():
            raise KeyError(f"{scale_key} absent from {shard}")
        weight = f.get_tensor(weight_key)
        scale = f.get_tensor(scale_key)
    return weight, scale, weight_key, scale_key


def _apply_cutlass_block_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    import vllm.model_executor.kernels.linear.scaled_mm.cutlass  # noqa: F401
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.platforms import current_platform

    q_input, input_scale = fp8_utils.per_token_group_quant_fp8(
        x.contiguous(),
        group_size=128,
        column_major_scales=True,
        dtype=current_platform.fp8_dtype(),
    )
    return torch.ops.vllm.padded_cutlass(
        q_input,
        weight,
        input_scale,
        scale,
        WEIGHT_BLOCK,
        torch.bfloat16,
    )


def _run_contexts(
    x_cpu: torch.Tensor,
    rows: list[int],
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_reference: torch.Tensor | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda")
    x_full = x_cpu.to(torch.bfloat16).contiguous().to(device)
    row_idx = torch.tensor(rows, device=device, dtype=torch.long)
    weight_gpu = weight.contiguous().to(device)
    scale_gpu = scale.contiguous().to(device=device, dtype=torch.float32)

    out_full = _apply_cutlass_block_fp8(x_full, weight_gpu, scale_gpu)
    out_rows = _apply_cutlass_block_fp8(
        x_full.index_select(0, row_idx).contiguous(), weight_gpu, scale_gpu
    )

    x_reversed = x_full.flip(0).contiguous()
    reverse_rows = [int(x_full.shape[0] - 1 - r) for r in rows]
    reverse_idx = torch.tensor(reverse_rows, device=device, dtype=torch.long)
    out_reversed = _apply_cutlass_block_fp8(x_reversed, weight_gpu, scale_gpu)

    full_rows = out_full.index_select(0, row_idx).detach().cpu()
    row_only = out_rows.detach().cpu()
    reversed_rows = out_reversed.index_select(0, reverse_idx).detach().cpu()

    out: dict[str, Any] = {
        "input_shape": list(x_cpu.shape),
        "rows": [int(x) for x in rows],
        "weight_shape": list(weight.shape),
        "weight_dtype": str(weight.dtype),
        "scale_shape": list(scale.shape),
        "scale_dtype": str(scale.dtype),
        "full_vs_row_only": {
            "max_abs": _max_abs(full_rows, row_only),
            "mean_abs": _mean_abs(full_rows, row_only),
            "nonzero": _nonzero(full_rows, row_only),
        },
        "full_vs_reversed_context": {
            "max_abs": _max_abs(full_rows, reversed_rows),
            "mean_abs": _mean_abs(full_rows, reversed_rows),
            "nonzero": _nonzero(full_rows, reversed_rows),
        },
    }
    if output_reference is not None:
        ref = output_reference.index_select(0, torch.tensor(rows, dtype=torch.long))
        out["full_vs_captured_output"] = {
            "max_abs": _max_abs(full_rows, ref),
            "mean_abs": _mean_abs(full_rows, ref),
            "nonzero": _nonzero(full_rows, ref),
        }
    return out


def _cross_rows(
    tree_x_cpu: torch.Tensor,
    native_x_cpu: torch.Tensor,
    tree_rows: list[int],
    native_rows: list[int],
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device("cuda")
    weight_gpu = weight.contiguous().to(device)
    scale_gpu = scale.contiguous().to(device=device, dtype=torch.float32)
    tree_idx = torch.tensor(tree_rows, device=device, dtype=torch.long)
    native_idx = torch.tensor(native_rows, device=device, dtype=torch.long)
    tree_x = tree_x_cpu.to(torch.bfloat16).contiguous().to(device).index_select(
        0, tree_idx
    )
    native_x = native_x_cpu.to(torch.bfloat16).contiguous().to(device).index_select(
        0, native_idx
    )
    tree_out = _apply_cutlass_block_fp8(tree_x, weight_gpu, scale_gpu).detach().cpu()
    native_out = _apply_cutlass_block_fp8(native_x, weight_gpu, scale_gpu).detach().cpu()
    return {
        "input_max_abs": _max_abs(tree_x.detach().cpu(), native_x.detach().cpu()),
        "input_mean_abs": _mean_abs(tree_x.detach().cpu(), native_x.detach().cpu()),
        "input_nonzero": _nonzero(tree_x.detach().cpu(), native_x.detach().cpu()),
        "output_max_abs": _max_abs(tree_out, native_out),
        "output_mean_abs": _mean_abs(tree_out, native_out),
        "output_nonzero": _nonzero(tree_out, native_out),
    }


def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    return x


def _path_terminal_contexts(
    x_cpu: torch.Tensor,
    paths: dict[int, list[int]],
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    captured_output: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device("cuda")
    x_full = x_cpu.to(torch.bfloat16).contiguous().to(device)
    weight_gpu = weight.contiguous().to(device)
    scale_gpu = scale.contiguous().to(device=device, dtype=torch.float32)
    captured_output = _flatten_rows(captured_output.to(torch.bfloat16))

    full_out = _apply_cutlass_block_fp8(x_full, weight_gpu, scale_gpu).detach().cpu()
    nodes: dict[str, Any] = {}
    max_path_vs_full = 0.0
    max_path_vs_capture = 0.0
    max_row_vs_capture = 0.0
    for node, path in paths.items():
        path_idx = torch.tensor(path, device=device, dtype=torch.long)
        node_idx = torch.tensor([node], device=device, dtype=torch.long)
        path_x = x_full.index_select(0, path_idx).contiguous()
        row_x = x_full.index_select(0, node_idx).contiguous()
        path_out = _apply_cutlass_block_fp8(path_x, weight_gpu, scale_gpu)[-1:].detach().cpu()
        row_out = _apply_cutlass_block_fp8(row_x, weight_gpu, scale_gpu).detach().cpu()
        full_node = full_out[node : node + 1]
        captured_node = captured_output[node : node + 1]
        path_vs_full = {
            "max_abs": _max_abs(path_out, full_node),
            "mean_abs": _mean_abs(path_out, full_node),
            "nonzero": _nonzero(path_out, full_node),
        }
        path_vs_capture = {
            "max_abs": _max_abs(path_out, captured_node),
            "mean_abs": _mean_abs(path_out, captured_node),
            "nonzero": _nonzero(path_out, captured_node),
        }
        row_vs_capture = {
            "max_abs": _max_abs(row_out, captured_node),
            "mean_abs": _mean_abs(row_out, captured_node),
            "nonzero": _nonzero(row_out, captured_node),
        }
        max_path_vs_full = max(max_path_vs_full, float(path_vs_full["max_abs"]))
        max_path_vs_capture = max(max_path_vs_capture, float(path_vs_capture["max_abs"]))
        max_row_vs_capture = max(max_row_vs_capture, float(row_vs_capture["max_abs"]))
        nodes[str(int(node))] = {
            "path": [int(x) for x in path],
            "depth_position": int(len(path) - 1),
            "path_terminal_vs_full_tree": path_vs_full,
            "path_terminal_vs_captured_output": path_vs_capture,
            "row_only_vs_captured_output": row_vs_capture,
        }
    return {
        "nodes": nodes,
        "summary": {
            "path_terminal_vs_full_tree_max_abs": float(max_path_vs_full),
            "path_terminal_vs_captured_output_max_abs": float(max_path_vs_capture),
            "row_only_vs_captured_output_max_abs": float(max_row_vs_capture),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("/models/qwen3.6-27b-fp8"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tree-subkernel", required=True, type=Path)
    parser.add_argument("--native-subkernel", required=True, type=Path)
    parser.add_argument("--tree-layer", required=True, type=Path)
    parser.add_argument("--native-layer", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--depth-count", type=int, default=5)
    parser.add_argument("--branch-nodes", default=",".join(map(str, DEFAULT_BRANCH_NODES)))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for live Cutlass block-fp8 replay")

    tree_sub = _load_pt(args.tree_subkernel)
    native_sub = _load_pt(args.native_subkernel)
    tree_layer = _load_pt(args.tree_layer)
    native_layer = _load_pt(args.native_layer)

    tree_rows = _tree_path0_rows(tree_sub, args.depth_count)
    native_rows = list(range(args.depth_count))
    parent = _tree_parent(tree_sub)
    branch_nodes = _csv_ints(args.branch_nodes, DEFAULT_BRANCH_NODES)
    branch_paths = {}
    if parent is not None:
        branch_paths = {
            int(node): _node_path(parent, int(node))
            for node in branch_nodes
            if 0 <= int(node) < len(parent)
        }
    out_w, out_s, out_w_key, out_s_key = _load_weight_pair(
        args.model_dir, args.layer, "out_proj"
    )
    qkv_w, qkv_s, qkv_w_key, qkv_s_key = _load_weight_pair(
        args.model_dir, args.layer, "in_proj_qkv"
    )
    z_w, z_s, z_w_key, z_s_key = _load_weight_pair(
        args.model_dir, args.layer, "in_proj_z"
    )

    tree_gate = _stage_tensor(tree_sub, "gate_out")
    native_gate = _stage_tensor(native_sub, "gate_out")
    tree_o_ref = _stage_tensor(tree_sub, "o_proj_out").to(torch.bfloat16)
    native_o_ref = _stage_tensor(native_sub, "o_proj_out").to(torch.bfloat16)
    tree_hidden_from_subkernel = _has_stage(tree_sub, "input_hidden")
    tree_hidden = (
        _stage_tensor(tree_sub, "input_hidden")
        if tree_hidden_from_subkernel
        else _layer_input(tree_layer)
    )
    native_hidden = _layer_input(native_layer)
    tree_pre_conv = _stage_tensor(tree_sub, "pre_conv").to(torch.bfloat16)
    native_pre_conv = _stage_tensor(native_sub, "pre_conv").to(torch.bfloat16)
    tree_z_ref = None
    if "gate_z" in tree_sub.get("stages", {}):
        tree_z_ref = _flatten_rows(_stage_tensor(tree_sub, "gate_z")).to(torch.bfloat16)

    result: dict[str, Any] = {
        "schema": "fr12.fp8_full_gemm_batch_invariance_probe.v1",
        "model_dir": str(args.model_dir),
        "layer": int(args.layer),
        "tree_rows": tree_rows,
        "native_rows": native_rows,
        "branch_nodes": [int(x) for x in branch_nodes],
        "tree_parent": parent,
        "weight_block": WEIGHT_BLOCK,
        "source": {
            "live_cutlass_path": (
                "vllm.model_executor.kernels.linear.scaled_mm.cutlass."
                "CutlassFp8BlockScaledMMKernel via torch.ops.vllm.padded_cutlass"
            ),
            "forced_dispatch": "direct live Cutlass block-fp8 op; no DeepGEMM/FlashInfer/server",
            "scale_compute_dtype": "torch.float32",
            "tree_hidden_source": (
                "tree_subkernel.stages.input_hidden"
                if tree_hidden_from_subkernel
                else "tree_layer.input_hidden"
            ),
        },
        "o_proj": {
            "weight_key": out_w_key,
            "scale_key": out_s_key,
            "tree_contexts": _run_contexts(
                tree_gate,
                tree_rows,
                out_w,
                out_s,
                output_reference=tree_o_ref,
            ),
            "native_contexts": _run_contexts(
                native_gate,
                native_rows,
                out_w,
                out_s,
                output_reference=native_o_ref,
            ),
            "tree_vs_native_row_only": _cross_rows(
                tree_gate, native_gate, tree_rows, native_rows, out_w, out_s
            ),
        },
        "in_proj_qkv": {
            "weight_key": qkv_w_key,
            "scale_key": qkv_s_key,
            "tree_contexts": _run_contexts(
                tree_hidden,
                tree_rows,
                qkv_w,
                qkv_s,
                output_reference=tree_pre_conv,
            ),
            "native_contexts": _run_contexts(native_hidden, native_rows, qkv_w, qkv_s),
            "tree_vs_native_row_only": (
                {
                    "not_applicable": True,
                    "reason": (
                        "tree hidden comes from the MTP GDN subkernel capture; "
                        "native-layer fixture is target-model hidden, so this "
                        "legacy cross-row comparison would mix sources"
                    ),
                }
                if tree_hidden_from_subkernel
                else _cross_rows(
                    tree_hidden, native_hidden, tree_rows, native_rows, qkv_w, qkv_s
                )
            ),
        },
        "in_proj_z": {
            "weight_key": z_w_key,
            "scale_key": z_s_key,
            "tree_contexts": _run_contexts(
                tree_hidden,
                tree_rows,
                z_w,
                z_s,
                output_reference=tree_z_ref,
            ),
            "native_contexts": _run_contexts(native_hidden, native_rows, z_w, z_s),
            "tree_vs_native_row_only": (
                {
                    "not_applicable": True,
                    "reason": (
                        "tree hidden comes from the MTP GDN subkernel capture; "
                        "native-layer fixture is target-model hidden, so this "
                        "legacy cross-row comparison would mix sources"
                    ),
                }
                if tree_hidden_from_subkernel
                else _cross_rows(
                    tree_hidden, native_hidden, tree_rows, native_rows, z_w, z_s
                )
            ),
        },
        "pre_conv_boundary": {
            "note": (
                "pre_conv is the post-layout qkv tensor used by the conv path; "
                "Qwen3-Next qkv/z are interleaved/rearranged after the two fp8 "
                "input projections, so this is the serving boundary available "
                "in the existing subkernel capture."
            ),
            "tree_vs_native": {
                "max_abs": _max_abs(
                    tree_pre_conv.index_select(0, torch.tensor(tree_rows)),
                    native_pre_conv.index_select(0, torch.tensor(native_rows)),
                ),
                "mean_abs": _mean_abs(
                    tree_pre_conv.index_select(0, torch.tensor(tree_rows)),
                    native_pre_conv.index_select(0, torch.tensor(native_rows)),
                ),
                "nonzero": _nonzero(
                    tree_pre_conv.index_select(0, torch.tensor(tree_rows)),
                    native_pre_conv.index_select(0, torch.tensor(native_rows)),
                ),
            },
        },
    }
    if branch_paths:
        result["branch_input_oracle"] = {
            "native_reference": (
                "L0 native branch-path in_proj sees the same hidden row for each "
                "root-to-node token path; live fp8 GEMM is replayed on that path "
                "and compared to the captured tree terminal row."
            ),
            "branch_nodes": [int(x) for x in branch_paths],
            "paths": {str(k): [int(x) for x in v] for k, v in branch_paths.items()},
            "in_proj_qkv": _path_terminal_contexts(
                tree_hidden,
                branch_paths,
                qkv_w,
                qkv_s,
                captured_output=tree_pre_conv,
            ),
        }
        if tree_z_ref is not None:
            result["branch_input_oracle"]["in_proj_z"] = _path_terminal_contexts(
                tree_hidden,
                branch_paths,
                z_w,
                z_s,
                captured_output=tree_z_ref,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

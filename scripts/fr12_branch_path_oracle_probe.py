#!/usr/bin/env python3
"""Boot-free L0 branch-path oracle for GDN scan -> gate -> o_proj.

For every requested node, this script builds the native linear path
root -> ... -> node from the captured tree payload, replays the live vLLM
recurrent GDN update on that path, then applies the same RMSNormGated and
block-fp8 out_proj kernels as serving.  The last row of that native path is
compared to the captured tree row for scan/gate/o_proj.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

WEIGHT_BLOCK = [128, 128]
DEFAULT_BRANCH_NODES = [3, 5, 7, 9]
DEFAULT_SPINE_NODES = [0, 1, 2, 4, 6]


def _csv_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _stage_tensor(capture: dict[str, Any], stage: str) -> torch.Tensor:
    try:
        return capture["stages"][stage]["tensor"]
    except KeyError as exc:
        raise KeyError(f"{capture.get('path', '<capture>')} missing stage {stage}") from exc


def _node_path(parent: list[int], node: int) -> list[int]:
    path: list[int] = []
    cur = int(node)
    while cur >= 0:
        path.append(cur)
        cur = int(parent[cur])
    return list(reversed(path))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().mean().item())


def _nonzero(a: torch.Tensor, b: torch.Tensor) -> int:
    if a.numel() == 0:
        return 0
    return int(((a.float() - b.float()).abs() != 0).sum().item())


def _first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    diff = (a.float() - b.float()).abs()
    if diff.numel() == 0:
        return None
    flat = int(torch.argmax(diff).item())
    val = float(diff.reshape(-1)[flat].item())
    if val == 0.0:
        return None
    idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), diff.shape)]
    return {
        "index": idx,
        "tree": float(a[tuple(idx)].float().item()),
        "native_path": float(b[tuple(idx)].float().item()),
        "abs": val,
    }


def _metrics(tree: torch.Tensor, native: torch.Tensor) -> dict[str, Any]:
    return {
        "max_abs": _max_abs(tree, native),
        "mean_abs": _mean_abs(tree, native),
        "nonzero": _nonzero(tree, native),
        "first_mismatch": _first_mismatch(tree, native),
    }


def _rows(payload: dict[str, Any], key: str, nodes: list[int], device: torch.device) -> torch.Tensor:
    tensor = payload[key]
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    idx = torch.tensor(nodes, dtype=torch.long)
    return tensor.index_select(0, idx).contiguous().to(device)


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
        keys = set(f.keys())
        if weight_key not in keys:
            raise KeyError(f"{weight_key} absent from {shard}")
        if scale_key not in keys:
            raise KeyError(f"{scale_key} absent from {shard}")
        weight = f.get_tensor(weight_key)
        scale = f.get_tensor(scale_key)
    return weight, scale, weight_key, scale_key


def _load_embedding_rows(
    model_dir: Path, token_ids: list[int], device: torch.device
) -> tuple[torch.Tensor, str]:
    shard = model_dir / "outside.safetensors"
    key = "model.language_model.embed_tokens.weight"
    with safe_open(shard, framework="pt", device="cpu") as f:
        if key not in set(f.keys()):
            raise KeyError(f"{key} absent from {shard}")
        weight = f.get_tensor(key)
    idx = torch.tensor(token_ids, dtype=torch.long)
    return weight.index_select(0, idx).to(device=device, dtype=torch.bfloat16), key


def _load_tree_token_ids(path: Path, *, expected_drafts: int) -> list[int]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "tree_path_lcp_max":
                continue
            emitted = row.get("emitted_tokens") or []
            drafts = row.get("draft_token_ids") or []
            if emitted and len(drafts) >= expected_drafts:
                return [int(emitted[0])] + [int(x) for x in drafts[:expected_drafts]]
    raise RuntimeError(f"no usable tree_path_lcp_max row in {path}")


def _load_conv_weight(model_dir: Path, layer: int) -> tuple[torch.Tensor, str]:
    shard = model_dir / f"layers-{layer}.safetensors"
    key = f"model.language_model.layers.{layer}.linear_attn.conv1d.weight"
    with safe_open(shard, framework="pt", device="cpu") as f:
        if key not in set(f.keys()):
            raise KeyError(f"{key} absent from {shard}")
        weight = f.get_tensor(key)
    return weight, key


def _load_norm_weight(model_dir: Path, layer: int) -> tuple[torch.Tensor, str]:
    shard = model_dir / f"layers-{layer}.safetensors"
    key = f"model.language_model.layers.{layer}.linear_attn.norm.weight"
    with safe_open(shard, framework="pt", device="cpu") as f:
        if key not in set(f.keys()):
            raise KeyError(f"{key} absent from {shard}")
        weight = f.get_tensor(key)
    return weight, key


def _apply_rmsnorm_gated(
    scan: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    use_live_kernel: bool,
) -> torch.Tensor:
    scan_2d = scan.to(torch.bfloat16).reshape(-1, scan.shape[-1]).contiguous()
    z_2d = z.to(torch.bfloat16).reshape(-1, z.shape[-1]).contiguous()
    if use_live_kernel:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        out = rmsnorm_fn(
            scan_2d,
            weight.to(device=scan.device, dtype=torch.bfloat16).contiguous(),
            None,
            z=z_2d,
            eps=eps,
            group_size=None,
            norm_before_gate=True,
            activation="swish",
        )
    else:
        x = scan_2d.float()
        w = weight.to(device=scan.device).float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        out = (x * torch.rsqrt(variance + eps) * w) * F.silu(z_2d.float())
        out = out.to(torch.bfloat16)
    return out.reshape(scan.shape[0], -1).contiguous()


def _apply_cutlass_block_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    import vllm.model_executor.kernels.linear.scaled_mm.cutlass  # noqa: F401
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.platforms import current_platform

    q_input, input_scale = fp8_utils.per_token_group_quant_fp8(
        x.to(torch.bfloat16).contiguous(),
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


def _native_path_scan(
    payload: dict[str, Any],
    path: list[int],
    *,
    device: torch.device,
) -> torch.Tensor:
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    native_out, _native_state = fused_sigmoid_gating_delta_rule_update(
        A_log=payload["A_log"].to(device).contiguous(),
        a=_rows(payload, "a", path, device),
        b=_rows(payload, "b", path, device),
        dt_bias=payload["dt_bias"].to(device).contiguous(),
        q=_rows(payload, "query_spec", path, device).unsqueeze(0),
        k=_rows(payload, "key_spec", path, device).unsqueeze(0),
        v=_rows(payload, "value_spec", path, device).unsqueeze(0),
        scale=float(payload["output_scale"]),
        initial_state=payload["h0"].to(device).unsqueeze(0).contiguous(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return native_out.squeeze(0).contiguous()


def _native_preconv_from_token_ids(
    *,
    model_dir: Path,
    layer: int,
    token_ids: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, str]]:
    hidden, embed_key = _load_embedding_rows(model_dir, token_ids, device)
    qkv_w, qkv_s, qkv_w_key, qkv_s_key = _load_weight_pair(
        model_dir, layer, "in_proj_qkv"
    )
    qkv = _apply_cutlass_block_fp8(
        hidden,
        qkv_w.contiguous().to(device),
        qkv_s.contiguous().to(device=device, dtype=torch.float32),
    )
    torch.cuda.synchronize()
    return qkv.detach().cpu(), {
        "embed": embed_key,
        "in_proj_qkv": qkv_w_key,
        "in_proj_qkv_scale": qkv_s_key,
    }


def _native_path_conv_from_preconv(
    *,
    pre_conv: torch.Tensor,
    prior_window: torch.Tensor,
    conv_weight: torch.Tensor,
    path: list[int],
) -> torch.Tensor:
    if prior_window.ndim != 2:
        raise ValueError(f"prior_window must be 2D, got {list(prior_window.shape)}")
    if conv_weight.ndim == 3 and conv_weight.shape[1] == 1:
        conv_weight = conv_weight[:, 0, :]
    if conv_weight.ndim != 2:
        raise ValueError(f"conv_weight must be [hidden,width], got {list(conv_weight.shape)}")
    width = int(conv_weight.shape[1])
    if int(prior_window.shape[1]) != width - 1:
        raise ValueError(
            f"prior_window has {prior_window.shape[1]} cols but conv width is {width}"
        )
    path_pre = pre_conv.index_select(0, torch.tensor(path, dtype=torch.long))
    source = torch.cat((prior_window.transpose(0, 1), path_pre), dim=0)
    windows = []
    for pos in range(len(path)):
        windows.append(source[pos : pos + width])
    window = torch.stack(windows, dim=0)
    acc = torch.zeros(
        (len(path), pre_conv.shape[1]), dtype=torch.float32, device=pre_conv.device
    )
    for col in range(width):
        product = (
            window[:, col, :].to(torch.bfloat16)
            * conv_weight[:, col].unsqueeze(0).to(torch.bfloat16)
        ).to(torch.bfloat16)
        acc = acc + product.to(torch.float32)
    return F.silu(acc).to(torch.bfloat16).contiguous()


def _load_z(payload: dict[str, Any], subkernel: dict[str, Any], device: torch.device) -> torch.Tensor:
    if "z_spec" in payload:
        return payload["z_spec"].to(device).contiguous()
    return _stage_tensor(subkernel, "gate_z").to(device).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--tree-subkernel", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/models/qwen3.6-27b-fp8"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--branch-nodes", default=",".join(map(str, DEFAULT_BRANCH_NODES)))
    parser.add_argument("--spine-nodes", default=",".join(map(str, DEFAULT_SPINE_NODES)))
    parser.add_argument("--tree-path-lcp-log", type=Path)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--torch-rmsnorm", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    payload = _load_pt(args.payload)
    subkernel = _load_pt(args.tree_subkernel)
    parent = [int(x) for x in payload["tree_parent"]]
    sub_parent = (
        _stage_tensor(subkernel, "gdn_scan_out")
        .new_tensor(
            subkernel["stages"]["gdn_scan_out"].get("extra", {}).get("tree_parent", []),
            dtype=torch.int64,
        )
        .tolist()
    )
    if sub_parent and [int(x) for x in sub_parent] != parent:
        raise RuntimeError(f"tree_parent mismatch: payload={parent} subkernel={sub_parent}")

    branch_nodes = _csv_ints(args.branch_nodes, DEFAULT_BRANCH_NODES)
    spine_nodes = _csv_ints(args.spine_nodes, DEFAULT_SPINE_NODES)
    requested_nodes = spine_nodes + [node for node in branch_nodes if node not in spine_nodes]
    for node in requested_nodes:
        if node < 0 or node >= len(parent):
            raise ValueError(f"node {node} outside tree of {len(parent)} rows")

    device = torch.device("cuda")
    norm_weight, norm_key = _load_norm_weight(args.model_dir, args.layer)
    conv_weight, conv_key = _load_conv_weight(args.model_dir, args.layer)
    out_w, out_s, out_w_key, out_s_key = _load_weight_pair(
        args.model_dir, args.layer, "out_proj"
    )
    norm_weight_gpu = norm_weight.to(device=device, dtype=torch.bfloat16).contiguous()
    out_w_gpu = out_w.contiguous().to(device)
    out_s_gpu = out_s.contiguous().to(device=device, dtype=torch.float32)

    tree_scan = _stage_tensor(subkernel, "gdn_scan_out").to(device)
    tree_gate = _stage_tensor(subkernel, "gate_out").to(device)
    tree_o_proj = _stage_tensor(subkernel, "o_proj_out").to(device)
    tree_pre_conv_cpu = _stage_tensor(subkernel, "pre_conv").to(torch.bfloat16)
    tree_conv_cpu = _stage_tensor(subkernel, "conv1d_out").to(torch.bfloat16)
    tree_detail = subkernel.get("meta", {}).get("tree_conv_detail", {})
    prior_window_cpu = tree_detail.get("prior_window")
    if not torch.is_tensor(prior_window_cpu):
        raise KeyError("tree subkernel capture missing meta.tree_conv_detail.prior_window")
    prior_window_cpu = prior_window_cpu.to(torch.bfloat16)
    conv_weight_cpu = conv_weight.to(torch.bfloat16)
    z_all = _load_z(payload, subkernel, device)
    pre_conv_oracle_cpu = tree_pre_conv_cpu
    pre_conv_source: dict[str, Any] = {
        "reference": "captured_tree_rows",
        "note": (
            "pass --tree-path-lcp-log to replay L0 pre_conv from traced token "
            "ids using embed_tokens + in_proj_qkv"
        ),
    }
    if args.tree_path_lcp_log is not None:
        token_ids = _load_tree_token_ids(
            args.tree_path_lcp_log, expected_drafts=len(parent) - 1
        )
        pre_conv_oracle_cpu, preconv_weight_keys = _native_preconv_from_token_ids(
            model_dir=args.model_dir,
            layer=args.layer,
            token_ids=token_ids,
            device=device,
        )
        pre_conv_source = {
            "reference": "embed_tokens + in_proj_qkv replay from traced tree token ids",
            "tree_path_lcp_log": str(args.tree_path_lcp_log),
            "token_ids_by_node": [int(x) for x in token_ids],
            "weights": preconv_weight_keys,
            "position_note": (
                "L0 pre_conv is token-embedding projection before positional "
                "operators; depth-position ancestry is verified at conv/scan."
            ),
        }

    full_gate_replay = _apply_rmsnorm_gated(
        tree_scan,
        z_all,
        norm_weight_gpu,
        eps=float(args.rms_eps),
        use_live_kernel=not args.torch_rmsnorm,
    )
    full_o_replay = _apply_cutlass_block_fp8(full_gate_replay, out_w_gpu, out_s_gpu)
    torch.cuda.synchronize()

    node_results: list[dict[str, Any]] = []
    native_scan_rows: list[torch.Tensor] = []
    native_z_rows: list[torch.Tensor] = []
    for node in requested_nodes:
        path = _node_path(parent, node)
        native_scan_path = _native_path_scan(payload, path, device=device)
        native_last_scan = native_scan_path[-1:].contiguous()
        native_scan_rows.append(native_last_scan)
        native_z_rows.append(z_all[node : node + 1].contiguous())

    native_scan = torch.cat(native_scan_rows, dim=0).contiguous()
    native_z = torch.cat(native_z_rows, dim=0).contiguous()
    native_gate = _apply_rmsnorm_gated(
        native_scan,
        native_z,
        norm_weight_gpu,
        eps=float(args.rms_eps),
        use_live_kernel=not args.torch_rmsnorm,
    )
    native_o_proj = _apply_cutlass_block_fp8(native_gate, out_w_gpu, out_s_gpu)
    torch.cuda.synchronize()

    for out_index, node in enumerate(requested_nodes):
        path = _node_path(parent, node)
        node_results.append(
            {
                "node": int(node),
                "kind": "spine" if node in spine_nodes else "branch",
                "path": [int(x) for x in path],
                "depth_position": int(len(path) - 1),
                "comparisons": {
                    "pre_conv": _metrics(
                        tree_pre_conv_cpu[node : node + 1],
                        pre_conv_oracle_cpu.index_select(
                            0, torch.tensor([path[-1]], dtype=torch.long)
                        ),
                    ),
                    "conv1d_out": _metrics(
                        tree_conv_cpu[node : node + 1],
                        _native_path_conv_from_preconv(
                            pre_conv=tree_pre_conv_cpu,
                            prior_window=prior_window_cpu,
                            conv_weight=conv_weight_cpu,
                            path=path,
                        )[-1:],
                    ),
                    "scan": _metrics(
                        tree_scan[node : node + 1].detach().cpu(),
                        native_scan[out_index : out_index + 1].detach().cpu(),
                    ),
                    "gate": _metrics(
                        tree_gate[node : node + 1].detach().cpu(),
                        native_gate[out_index : out_index + 1].detach().cpu(),
                    ),
                    "o_proj": _metrics(
                        tree_o_proj[node : node + 1].detach().cpu(),
                        native_o_proj[out_index : out_index + 1].detach().cpu(),
                    ),
                },
            }
        )

    result = {
        "schema": "fr12.branch_path_oracle_probe.v1",
        "payload": str(args.payload),
        "tree_subkernel": str(args.tree_subkernel),
        "model_dir": str(args.model_dir),
        "layer": int(args.layer),
        "tree_parent": parent,
        "spine_nodes": [int(x) for x in spine_nodes],
        "branch_nodes": [int(x) for x in branch_nodes],
        "native_reference": "vllm.fused_sigmoid_gating_delta_rule_update on root-to-node path",
        "position_basis": "depth position within each native path",
        "gate_reference": (
            "vllm RMSNormGated live kernel"
            if not args.torch_rmsnorm
            else "torch RMSNormGated formula"
        ),
        "o_proj_reference": "vllm padded_cutlass block-fp8",
        "pre_conv_reference": pre_conv_source,
        "conv_reference": (
            "native linear branch-path causal conv replay from captured pre_conv "
            "rows plus captured prior window and real conv1d weights"
        ),
        "weights": {
            "conv1d": conv_key,
            "norm": norm_key,
            "out_proj": out_w_key,
            "out_proj_scale": out_s_key,
        },
        "rms_eps": float(args.rms_eps),
        "self_checks": {
            "payload_serving_scan_vs_subkernel_scan": _metrics(
                payload["serving_out"], _stage_tensor(subkernel, "gdn_scan_out")
            )
            if "serving_out" in payload
            else None,
            "tree_scan_plus_gate_z_vs_captured_gate": _metrics(
                full_gate_replay.detach().cpu(), _stage_tensor(subkernel, "gate_out")
            ),
            "tree_gate_replay_vs_captured_o_proj": _metrics(
                full_o_replay.detach().cpu(), _stage_tensor(subkernel, "o_proj_out")
            ),
        },
        "nodes": node_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

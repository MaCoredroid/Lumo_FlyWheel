#!/usr/bin/env python3
"""Replay FR13 L12 tree-GDN handoff payload against native-on-path FLA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (  # noqa: E402
    Tree,
    launch_tree_gdn_prepared,
    padded_nodes,
)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _nonzero(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(((a.float() - b.float()).abs() != 0).sum().item())


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "torch_equal": bool(torch.equal(a, b)),
        "max_abs": _max_abs(a, b),
        "nonzero": _nonzero(a, b),
    }


def _rows(payload: dict[str, Any], key: str, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    tensor = payload[key]
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    return tensor.index_select(0, rows.cpu()).to(device).contiguous()


def evaluate(handoff_path: Path, scan_path: Path | None, out_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    payload = torch.load(handoff_path, map_location="cpu")
    device = torch.device("cuda")
    parent = [int(x) for x in payload["tree_parent"]]
    n_actual = len(parent)
    n_pad = padded_nodes(n_actual)
    tree = Tree(tuple(parent))
    strict, visible = tree.masks(device, n_pad)
    output_scale = float(payload["output_scale"])

    replay_out, replay_state = launch_tree_gdn_prepared(
        q=payload["query_spec"].to(device).contiguous(),
        k=payload["key_spec"].to(device).contiguous(),
        v=payload["value_tree"].to(device).contiguous(),
        g=payload["g_tree"].to(device).contiguous(),
        beta=payload["beta_tree"].to(device).contiguous(),
        raw_a=payload["a"].to(device).contiguous(),
        raw_b=payload["b"].to(device).contiguous(),
        A_log=payload["A_log"].to(device).contiguous(),
        dt_bias=payload["dt_bias"].to(device).contiguous(),
        h0=payload["prev_h0"].to(device).contiguous(),
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
    )

    path = [int(x) for x in payload["accepted_node_path"]]
    path_rows = torch.tensor(path, dtype=torch.long)
    native_out, native_state = fused_sigmoid_gating_delta_rule_update(
        A_log=payload["A_log"].to(device).contiguous(),
        a=_rows(payload, "a", path_rows, device),
        b=_rows(payload, "b", path_rows, device),
        dt_bias=payload["dt_bias"].to(device).contiguous(),
        q=_rows(payload, "query_spec", path_rows, device).unsqueeze(0),
        k=_rows(payload, "key_spec", path_rows, device).unsqueeze(0),
        v=_rows(payload, "value_spec", path_rows, device).unsqueeze(0),
        scale=output_scale,
        initial_state=payload["prev_h0"].to(device).unsqueeze(0).contiguous(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    accepted_node = int(payload["accepted_node_id"])
    serving_state = payload["serving_tree_state"].to(device).contiguous()
    next_read_ssm = payload["next_read_ssm_state"].to(device).contiguous()
    accepted_serving_state = serving_state[accepted_node]
    native_state_seq = native_state.squeeze(0)
    native_final_state = native_state_seq[-1]
    native_final_out = native_out.squeeze(0)[-1]
    replay_state_n = replay_state[:n_actual].contiguous()
    replay_out_n = replay_out[:n_actual].contiguous()

    result: dict[str, Any] = {
        "schema": "fr13.l12_handoff_replay.v1",
        "handoff_payload": str(handoff_path),
        "scan_payload": None if scan_path is None else str(scan_path),
        "layer_prefix": payload.get("layer_prefix"),
        "accepted_len": int(payload["accepted_len"]),
        "accepted_node_id": accepted_node,
        "accepted_node_path": path,
        "accepted_token_ids": [int(x) for x in payload.get("accepted_token_ids", [])],
        "tree_parent": parent,
        "tree_replay_vs_serving_state": _metrics(replay_state_n, serving_state),
        "accepted_state_replay_vs_serving": _metrics(
            replay_state_n[accepted_node], accepted_serving_state
        ),
        "accepted_state_native_path_vs_tree_replay": _metrics(
            native_final_state, replay_state_n[accepted_node]
        ),
        "accepted_state_native_path_vs_serving": _metrics(
            native_final_state, accepted_serving_state
        ),
        "accepted_state_native_path_vs_next_read": _metrics(
            native_final_state, next_read_ssm
        ),
        "accepted_state_serving_vs_next_read": _metrics(
            accepted_serving_state, next_read_ssm
        ),
        "accepted_out_native_path_vs_tree_replay": _metrics(
            native_final_out, replay_out_n[accepted_node]
        ),
    }
    next_conv = payload.get("next_read_conv_state")
    if next_conv is not None:
        serving_conv = payload["serving_conv_rows"].to(device).contiguous()
        next_conv_t = next_conv.to(device).contiguous()
        result["accepted_conv_serving_vs_next_read"] = _metrics(
            serving_conv[accepted_node], next_conv_t
        )

    if scan_path is not None:
        scan = torch.load(scan_path, map_location="cpu")
        result["scan_payload_h0_vs_handoff_prev_h0"] = _metrics(
            scan["h0"], payload["prev_h0"]
        )
        result["scan_payload_serving_state_vs_handoff_serving_state"] = _metrics(
            scan["serving_state"], payload["serving_tree_state"]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--scan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.handoff, args.scan, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

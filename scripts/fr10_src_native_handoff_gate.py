#!/usr/bin/env python3
"""Replay the FR10 cross-step src==native handoff payload.

The live patcher captures the state that the next serving step actually reads
plus the previous event's fixed GDN inputs and accepted tree path. This gate
runs the native recurrent update over only that accepted path and compares the
result with the next-read state. A neighbor accepted-row replay is the powered
negative control.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr10_gdn_tree_kernel import Tree  # noqa: E402

_VALIDATION_PATH = REPO / "scripts" / "fr10_phase4_real_tensor_validation.py"
_validation_spec = importlib.util.spec_from_file_location(
    "fr10_phase4_real_tensor_validation", _VALIDATION_PATH
)
if _validation_spec is None or _validation_spec.loader is None:
    raise RuntimeError(f"cannot load {_VALIDATION_PATH}")
_validation = importlib.util.module_from_spec(_validation_spec)
sys.modules["fr10_phase4_real_tensor_validation"] = _validation
_validation_spec.loader.exec_module(_validation)
native_update_serial_per_path = _validation.native_update_serial_per_path


def _path_to(parent: list[int], node: int) -> list[int]:
    path: list[int] = []
    cur = int(node)
    guard = 0
    while cur >= 0 and guard <= len(parent):
        path.append(cur)
        cur = int(parent[cur])
        guard += 1
    path.reverse()
    return path


def _linear_parent(n: int) -> list[int]:
    return [-1] + [idx - 1 for idx in range(1, n)]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _native_final_state_for_path(
    payload: dict[str, Any],
    path: list[int],
    device: torch.device,
) -> torch.Tensor:
    if not path:
        return payload["prev_h0"].to(device).contiguous()
    index = torch.tensor(path, device=device, dtype=torch.long)
    linear = Tree(tuple(_linear_parent(len(path))))
    _, states = native_update_serial_per_path(
        linear,
        payload["query_spec"].to(device).index_select(0, index).contiguous(),
        payload["key_spec"].to(device).index_select(0, index).contiguous(),
        payload["value_spec"].to(device).index_select(0, index).contiguous(),
        payload["a"].to(device).index_select(0, index).contiguous(),
        payload["b"].to(device).index_select(0, index).contiguous(),
        payload["A_log"].to(device).contiguous(),
        payload["dt_bias"].to(device).contiguous(),
        payload["prev_h0"].to(device).contiguous(),
        float(payload["output_scale"]),
    )
    return states[-1].contiguous()


def evaluate(
    payload_path: Path,
    *,
    out_path: Path | None = None,
    state_atol: float = 1.0e-4,
    conv_atol: float = 0.0,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FR10 src==native handoff replay")
    payload = torch.load(payload_path, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    accepted_len = int(payload["accepted_len"])
    accepted_node = int(payload["accepted_node_id"])
    accepted_path = [int(x) for x in payload.get("accepted_node_path") or []]
    if accepted_len <= 0 or not accepted_path:
        raise ValueError("src==native handoff payload must contain an accepted path")
    if accepted_path[-1] != accepted_node:
        raise ValueError(
            f"accepted_node_id {accepted_node} does not match accepted path {accepted_path}"
        )

    device = torch.device("cuda")
    native_state = _native_final_state_for_path(payload, accepted_path, device)
    next_read_ssm = payload["next_read_ssm_state"].to(device).contiguous()
    serving_tree_state = payload["serving_tree_state"].to(device).contiguous()
    serving_conv_rows = payload["serving_conv_rows"].to(device).contiguous()
    next_read_conv = payload["next_read_conv_state"]
    if next_read_conv is not None:
        next_read_conv = next_read_conv.to(device).contiguous()

    tree_row_state = serving_tree_state[accepted_node].contiguous()
    conv_oracle = serving_conv_rows[accepted_node].contiguous()
    torch.cuda.synchronize()

    native_vs_next = _max_abs(native_state, next_read_ssm)
    native_vs_tree_row = _max_abs(native_state, tree_row_state)
    conv_vs_next = None if next_read_conv is None else _max_abs(conv_oracle, next_read_conv)

    neg_node = accepted_node + 1 if accepted_node + 1 < len(parent) else accepted_node - 1
    negative_powered = False
    negative_node = None
    negative_ssm_max = None
    negative_conv_max = None
    if 0 <= neg_node < len(parent):
        negative_node = int(neg_node)
        negative_path = _path_to(parent, negative_node)
        neg_native_state = _native_final_state_for_path(payload, negative_path, device)
        negative_ssm_max = _max_abs(neg_native_state, next_read_ssm)
        negative_powered = negative_powered or negative_ssm_max > state_atol
        if next_read_conv is not None:
            negative_conv_max = _max_abs(serving_conv_rows[negative_node], next_read_conv)
            negative_powered = negative_powered or negative_conv_max > conv_atol

    state_pass = native_vs_next <= state_atol
    conv_pass = next_read_conv is not None and conv_vs_next is not None and conv_vs_next <= conv_atol
    result = {
        "schema": "fr10.src_native_handoff_gate.v1",
        "regime": "deterministic_captured_replay_byte_exact",
        "payload": str(payload_path),
        "layer_prefix": payload.get("layer_prefix"),
        "batch_index": int(payload.get("batch_index", 0)),
        "accepted_len": accepted_len,
        "accepted_node_id": accepted_node,
        "accepted_node_path": accepted_path,
        "accepted_token_ids": [int(x) for x in payload.get("accepted_token_ids") or []],
        "accepted_spec_state_bank_row": payload.get("accepted_spec_state_bank_row"),
        "next_read_bank_row": payload.get("next_read_bank_row"),
        "state_atol": state_atol,
        "conv_atol": conv_atol,
        "ssm_next_vs_native_max_abs": native_vs_next,
        "ssm_next_vs_native_bit_exact": bool(torch.equal(next_read_ssm, native_state)),
        "ssm_tree_row_vs_native_max_abs": native_vs_tree_row,
        "ssm_tree_row_vs_native_bit_exact": bool(torch.equal(tree_row_state, native_state)),
        "conv_next_vs_native_max_abs": conv_vs_next,
        "conv_next_vs_native_bit_exact": (
            False if next_read_conv is None else bool(torch.equal(next_read_conv, conv_oracle))
        ),
        "negative_node_id": negative_node,
        "negative_ssm_max_abs": negative_ssm_max,
        "negative_conv_max_abs": negative_conv_max,
        "negative_powered": bool(negative_powered),
        "src_native_pass": bool(state_pass and conv_pass and negative_powered),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-atol", type=float, default=1.0e-4)
    parser.add_argument("--conv-atol", type=float, default=0.0)
    args = parser.parse_args()
    result = evaluate(
        args.payload,
        out_path=args.out,
        state_atol=args.state_atol,
        conv_atol=args.conv_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["src_native_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

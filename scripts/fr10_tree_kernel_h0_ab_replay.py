#!/usr/bin/env python3
"""Replay FR10 tree GDN with a known-correct h0 from a handoff payload."""
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

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (  # noqa: E402
    Tree,
    launch_tree_gdn_prepared,
    padded_nodes,
)

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


def _linear_parent(n: int) -> list[int]:
    return [-1] + [idx - 1 for idx in range(1, n)]


def _node_major_spec_tensor(payload: dict[str, Any], name: str) -> torch.Tensor:
    tensor = payload[name]
    if name == "value_spec" and tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    q_rows = int(payload["query_spec"].shape[0])
    if tensor.shape[0] > q_rows:
        tensor = tensor[:q_rows]
    return tensor


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def evaluate(payload_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FR10 tree kernel h0 replay")
    payload = torch.load(payload_path, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    accepted_node = int(payload.get("accepted_gdn_node_id", payload["accepted_node_id"]))
    accepted_path = [int(x) for x in payload.get("accepted_gdn_node_path") or []]
    if not accepted_path:
        accepted_path = [int(x) for x in payload.get("accepted_node_path") or []]
    if not accepted_path or accepted_path[-1] != accepted_node:
        raise ValueError(
            f"accepted path {accepted_path} does not end at accepted node {accepted_node}"
        )

    device = torch.device("cuda")
    tree = Tree(tuple(parent))
    n_actual = len(parent)
    n_pad = padded_nodes(n_actual)
    strict, visible = tree.masks(device, n_pad)

    q = _node_major_spec_tensor(payload, "query_spec").to(device).contiguous()
    k = _node_major_spec_tensor(payload, "key_spec").to(device).contiguous()
    v = _node_major_spec_tensor(payload, "value_tree").to(device).contiguous()
    g = _node_major_spec_tensor(payload, "g_tree").to(device).contiguous()
    beta = _node_major_spec_tensor(payload, "beta_tree").to(device).contiguous()
    h0 = payload["prev_h0"].to(device).contiguous()
    serving_tree_state = payload["serving_tree_state"].to(device).contiguous()
    output_scale = float(payload["output_scale"])

    _, replay_state = launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        h0=h0,
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
    )

    index = torch.tensor(accepted_path, device=device, dtype=torch.long)
    _, serial_states = native_update_serial_per_path(
        Tree(tuple(_linear_parent(len(accepted_path)))),
        q.index_select(0, index).contiguous(),
        k.index_select(0, index).contiguous(),
        v.index_select(0, index).contiguous(),
        g.index_select(0, index).contiguous(),
        beta.index_select(0, index).contiguous(),
        payload["A_log"].to(device).contiguous(),
        payload["dt_bias"].to(device).contiguous(),
        h0,
        output_scale,
    )
    torch.cuda.synchronize()

    replay_row = replay_state[accepted_node].contiguous()
    serving_row = serving_tree_state[accepted_node].contiguous()
    native_final = serial_states[-1].contiguous()
    replay_vs_native = _max_abs(replay_row, native_final)
    serving_vs_replay = _max_abs(serving_row, replay_row)
    serving_vs_native = _max_abs(serving_row, native_final)
    state_atol = 1.0e-4
    branch_replay_pass = replay_vs_native <= state_atol
    result = {
        "schema": "fr10.tree_kernel_h0_ab_replay.v1",
        "payload": str(payload_path),
        "accepted_len": int(payload["accepted_len"]),
        "accepted_node_id": int(payload["accepted_node_id"]),
        "accepted_node_path": [int(x) for x in payload.get("accepted_node_path") or []],
        "accepted_gdn_node_id": accepted_node,
        "accepted_gdn_node_path": accepted_path,
        "state_atol": state_atol,
        "kernel_replay_row_vs_native_max_abs": replay_vs_native,
        "serving_tree_row_vs_kernel_replay_max_abs": serving_vs_replay,
        "serving_tree_row_vs_native_max_abs": serving_vs_native,
        "kernel_replay_matches_native": bool(branch_replay_pass),
        "branch_split": "A" if branch_replay_pass else "B",
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.payload, args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["kernel_replay_matches_native"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

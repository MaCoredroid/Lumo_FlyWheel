#!/usr/bin/env python3
"""Replay a captured FR10 serving tree-GDN scan payload.

The live patcher writes this payload only in debug/eager capture mode. This
script runs the same serving tree kernel and the serial-per-path native update
on the fixed captured inputs, then reports exact replay and numeric parity.
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

from lumo_flywheel_serving.fr10_gdn_tree_kernel import Tree, launch_tree_gdn_prepared  # noqa: E402

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


def _tree_masks(parent: list[int], device: torch.device, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
    return Tree(tuple(parent)).masks(device, n_pad)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _per_node_max_abs(left: torch.Tensor, right: torch.Tensor) -> list[float]:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}")
    return [
        float((left[idx].float() - right[idx].float()).abs().max().item())
        for idx in range(int(left.shape[0]))
    ]


def evaluate(payload_path: Path, *, out_path: Path | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for scan capture replay")
    payload = torch.load(payload_path, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    n_actual = int(payload["n_actual"])
    n_pad = int(payload["n_pad"])
    device = torch.device("cuda")
    strict, visible = _tree_masks(parent, device, n_pad)

    q = payload["query_spec"].to(device).contiguous()
    k = payload["key_spec"].to(device).contiguous()
    value_spec = payload["value_spec"].to(device).contiguous()
    value_tree = payload["value_tree"].to(device).contiguous()
    g_tree = payload["g_tree"].to(device).contiguous()
    beta_tree = payload["beta_tree"].to(device).contiguous()
    h0 = payload["h0"].to(device).contiguous()
    a = payload["a"].to(device).contiguous()
    b = payload["b"].to(device).contiguous()
    A_log = payload["A_log"].to(device).contiguous()
    dt_bias = payload["dt_bias"].to(device).contiguous()
    serving_out = payload["serving_out"].to(device).contiguous()
    serving_state = payload["serving_state"].to(device).contiguous()
    output_scale = float(payload["output_scale"])

    replay_out, replay_state = launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=value_tree,
        g=g_tree,
        beta=beta_tree,
        h0=h0,
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
    )
    serial_out, serial_state = native_update_serial_per_path(
        Tree(tuple(parent)),
        q,
        k,
        value_spec,
        a,
        b,
        A_log,
        dt_bias,
        h0,
        output_scale,
    )
    flat_parent = [-1] + list(range(n_actual - 1))
    flat_out, flat_state = native_update_serial_per_path(
        Tree(tuple(flat_parent)),
        q,
        k,
        value_spec,
        a,
        b,
        A_log,
        dt_bias,
        h0,
        output_scale,
    )
    torch.cuda.synchronize()

    replay_out_n = replay_out[:n_actual]
    replay_state_n = replay_state[:n_actual]
    flat_out_delta = _max_abs(flat_out, serial_out)
    flat_state_delta = _max_abs(flat_state, serial_state)
    serving_serial_out_by_node = _per_node_max_abs(serving_out, serial_out)
    serving_serial_state_by_node = _per_node_max_abs(serving_state, serial_state)
    replay_serial_out_by_node = _per_node_max_abs(replay_out_n, serial_out)
    replay_serial_state_by_node = _per_node_max_abs(replay_state_n, serial_state)
    result = {
        "schema": "fr10.scan_capture_replay_gate.v1",
        "payload": str(payload_path),
        "layer_prefix": payload.get("layer_prefix"),
        "tree_parent": parent,
        "flat_negative_parent": flat_parent,
        "n_actual": n_actual,
        "n_pad": n_pad,
        "serving_vs_replay_out_bit_exact": bool(torch.equal(serving_out, replay_out_n)),
        "serving_vs_replay_state_bit_exact": bool(torch.equal(serving_state, replay_state_n)),
        "serving_vs_replay_out_max_abs": _max_abs(serving_out, replay_out_n),
        "serving_vs_replay_state_max_abs": _max_abs(serving_state, replay_state_n),
        "replay_vs_serial_out_max_abs": _max_abs(replay_out_n, serial_out),
        "replay_vs_serial_state_max_abs": _max_abs(replay_state_n, serial_state),
        "serving_vs_serial_out_max_abs": _max_abs(serving_out, serial_out),
        "serving_vs_serial_state_max_abs": _max_abs(serving_state, serial_state),
        "serving_vs_serial_out_max_abs_by_node": serving_serial_out_by_node,
        "serving_vs_serial_state_max_abs_by_node": serving_serial_state_by_node,
        "replay_vs_serial_out_max_abs_by_node": replay_serial_out_by_node,
        "replay_vs_serial_state_max_abs_by_node": replay_serial_state_by_node,
        "node0_serving_vs_serial_out_max_abs": serving_serial_out_by_node[0],
        "node0_serving_vs_serial_state_max_abs": serving_serial_state_by_node[0],
        "flat_negative_vs_serial_out_max_abs": flat_out_delta,
        "flat_negative_vs_serial_state_max_abs": flat_state_delta,
        "flat_negative_control_pass": bool(
            flat_out_delta > 0.0 or flat_state_delta > 0.0
        ),
        "serving_replay_pass": bool(
            torch.equal(serving_out, replay_out_n)
            and torch.equal(serving_state, replay_state_n)
        ),
        "serial_parity_pass": bool(
            _max_abs(replay_out_n, serial_out) <= 6.2e-5
            and _max_abs(replay_state_n, serial_state) <= 1.0e-4
        ),
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.payload, out_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if (
        result["serving_replay_pass"]
        and result["serial_parity_pass"]
        and result["flat_negative_control_pass"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())

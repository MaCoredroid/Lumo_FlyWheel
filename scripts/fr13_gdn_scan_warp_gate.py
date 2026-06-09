#!/usr/bin/env python3
"""Validate the FR13 GDN tree-scan launch warp setting on real captured tensors.

This is a lightweight CUDA replay, not a model-server boot. It uses a captured
tree-GDN payload and compares the repo tree kernel against the native per-path
FLA update at two padding regimes:

* N_PAD=1: a single root row, sensitive to small leading extent changes.
* N_PAD=16: the deployed MTP-5 tree size that motivated num_warps=8.

The raw output max_abs values are the authoritative gate; do not use the older
scan-output replay script's loose threshold as a pass/fail signal.
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

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (  # noqa: E402
    Tree,
    launch_tree_gdn_prepared,
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


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _first_mismatch(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any] | None:
    diff = (left.float() - right.float()).abs()
    flat = int(torch.argmax(diff.reshape(-1)).item())
    value = float(diff.reshape(-1)[flat].item())
    if value == 0.0:
        return None
    idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), diff.shape)]
    return {
        "index": idx,
        "left": float(left[tuple(idx)].float().item()),
        "right": float(right[tuple(idx)].float().item()),
        "abs": value,
    }


def _select_rows(tensor: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    return tensor.index_select(0, rows.cpu()).contiguous()


def _run_case(
    payload: dict[str, Any],
    *,
    name: str,
    rows_list: list[int],
    n_pad: int,
    parent: list[int] | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda")
    rows = torch.tensor(rows_list, dtype=torch.long)
    n_actual = int(rows.numel())
    if parent is None:
        parent = [-1] + [idx - 1 for idx in range(1, n_actual)]
    if len(parent) != n_actual:
        raise ValueError(f"parent length {len(parent)} does not match n_actual {n_actual}")
    tree = Tree(tuple(parent))
    strict, visible = tree.masks(device, n_pad)

    q = _select_rows(payload["query_spec"], rows).to(device).contiguous()
    k = _select_rows(payload["key_spec"], rows).to(device).contiguous()
    value_spec = _select_rows(payload["value_spec"], rows).to(device).contiguous()
    value_tree = _select_rows(payload["value_tree"], rows).to(device).contiguous()
    g_tree = _select_rows(payload["g_tree"], rows).to(device).contiguous()
    beta_tree = _select_rows(payload["beta_tree"], rows).to(device).contiguous()
    a = _select_rows(payload["a"], rows).to(device).contiguous()
    b = _select_rows(payload["b"], rows).to(device).contiguous()
    A_log = payload["A_log"].to(device).contiguous()
    dt_bias = payload["dt_bias"].to(device).contiguous()
    h0 = payload["h0"].to(device).contiguous()
    output_scale = float(payload["output_scale"])

    tree_out, tree_state = launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=value_tree,
        g=g_tree,
        beta=beta_tree,
        raw_a=a,
        raw_b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        h0=h0,
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
    )
    native_out, native_state = native_update_serial_per_path(
        tree,
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
    tree_out = tree_out[:n_actual].contiguous()
    tree_state = tree_state[:n_actual].contiguous()

    by_node = []
    for idx in range(n_actual):
        by_node.append(
            {
                "node": idx,
                "source_row": int(rows_list[idx]),
                "out_vs_native_max_abs": _max_abs(tree_out[idx], native_out[idx]),
                "state_vs_native_max_abs": _max_abs(tree_state[idx], native_state[idx]),
            }
        )
    out_max = _max_abs(tree_out, native_out)
    state_max = _max_abs(tree_state, native_state)
    return {
        "name": name,
        "n_actual": n_actual,
        "n_pad": n_pad,
        "source_rows": rows_list,
        "parent": parent,
        "out_vs_native_max_abs": out_max,
        "state_vs_native_max_abs": state_max,
        "out_bit_exact": bool(torch.equal(tree_out, native_out.to(tree_out.dtype))),
        "state_bit_exact": bool(torch.equal(tree_state, native_state)),
        "out_first_mismatch": _first_mismatch(tree_out, native_out),
        "state_first_mismatch": _first_mismatch(tree_state, native_state),
        "by_node": by_node,
    }


def evaluate(payload_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GDN scan warp validation")
    payload = torch.load(payload_path, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    if int(payload["n_pad"]) < 16 or len(parent) < 10:
        raise RuntimeError(
            "payload must contain the deployed N_PAD=16 MTP-5 tree; "
            f"got n_pad={payload.get('n_pad')} parent_len={len(parent)}"
        )
    return {
        "schema": "fr13.gdn_scan_warp_gate.v1",
        "payload": str(payload_path),
        "layer_prefix": payload.get("layer_prefix"),
        "kernel": "lumo_flywheel_serving.fr10_gdn_tree_kernel.launch_tree_gdn_prepared",
        "launch_num_warps_expected": 8,
        "cases": [
            _run_case(payload, name="root_npad1", rows_list=[0], n_pad=1),
            _run_case(
                payload,
                name="tree_npad16",
                rows_list=list(range(10)),
                n_pad=16,
                parent=parent,
            ),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    ok = all(float(case["out_vs_native_max_abs"]) == 0.0 for case in result["cases"])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

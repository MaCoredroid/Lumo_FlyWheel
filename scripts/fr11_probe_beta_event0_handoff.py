#!/usr/bin/env python3
"""Probe beta for FR11 no-copy GDN tree verification.

This is a boot-free replay over an FR10 handoff capture. It byte-compares the
accepted tree spine state against the state that the next native speculative
forward actually read. Address equality is diagnostic only; the verdict is
based on bytes of the loaded recurrent h0 and the conv prior-state window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} did not contain a dict payload")
    schema = payload.get("schema")
    if schema != "fr10.src_native_handoff_payload.v1":
        raise ValueError(f"unsupported payload schema {schema!r}")
    return payload


def _require_tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"payload missing tensor {key!r}")
    return value.contiguous()


def _path_to(parent: list[int], node: int) -> list[int]:
    path: list[int] = []
    cur = int(node)
    for _ in range(len(parent) + 1):
        if cur < 0:
            return list(reversed(path))
        path.append(cur)
        cur = int(parent[cur])
    raise ValueError(f"parent cycle while resolving node {node}")


def _accepted_node_and_path(payload: dict[str, Any], parent: list[int]) -> tuple[int, list[int]]:
    accepted_node = int(payload["accepted_node_id"])
    accepted_path = [int(x) for x in payload.get("accepted_node_path") or []]
    if not (0 <= accepted_node < len(parent)):
        raise ValueError(f"accepted_node_id {accepted_node} outside tree size {len(parent)}")
    resolved = _path_to(parent, accepted_node)
    if accepted_path and resolved != accepted_path and resolved[1:] != accepted_path:
        # Older captures sometimes stored a draft-space node id and a GDN-space
        # path. Prefer the path when it resolves to a unique GDN node.
        matches = [
            idx
            for idx in range(len(parent))
            if _path_to(parent, idx) == accepted_path
            or _path_to(parent, idx)[1:] == accepted_path
        ]
        if len(matches) == 1:
            accepted_node = matches[0]
            resolved = _path_to(parent, accepted_node)
        else:
            raise ValueError(
                "accepted_node_path does not match accepted_node_id and cannot be "
                f"resolved uniquely: id_path={resolved}, captured_path={accepted_path}"
            )
    report_path = accepted_path or resolved
    if len(report_path) < 1:
        raise ValueError("handoff capture has no accepted tree spine beyond root")
    return accepted_node, report_path


def _tensor_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "shape_match": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "bit_exact": False,
            "num_mismatched_elements": None,
            "max_abs": None,
        }
    diff = (left.float() - right.float()).abs()
    return {
        "shape_match": True,
        "shape": list(left.shape),
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "bit_exact": bool(torch.equal(left, right)),
        "num_mismatched_elements": int((left != right).sum().item()),
        "max_abs": float(diff.max().item()) if left.numel() else 0.0,
    }


def _first_mismatch_column(left: torch.Tensor, right: torch.Tensor) -> int | None:
    if left.shape != right.shape or left.ndim < 2:
        return None
    columns = int(left.shape[-1])
    for col in range(columns):
        if not torch.equal(left[..., col], right[..., col]):
            return col
    return None


def _per_column_stats(left: torch.Tensor, right: torch.Tensor) -> list[dict[str, Any]]:
    if left.shape != right.shape or left.ndim < 2:
        return []
    rows: list[dict[str, Any]] = []
    for col in range(int(left.shape[-1])):
        l_col = left[..., col]
        r_col = right[..., col]
        diff = (l_col.float() - r_col.float()).abs()
        rows.append(
            {
                "column": col,
                "bit_exact": bool(torch.equal(l_col, r_col)),
                "num_mismatched_elements": int((l_col != r_col).sum().item()),
                "max_abs": float(diff.max().item()) if l_col.numel() else 0.0,
            }
        )
    return rows


def _best_column_shift(left: torch.Tensor, right: torch.Tensor, max_shift: int = 4) -> dict[str, Any] | None:
    if left.shape != right.shape or left.ndim < 2:
        return None
    best: dict[str, Any] | None = None
    width = int(left.shape[-1])
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            l_slice = left[..., : width + shift]
            r_slice = right[..., -shift:]
        elif shift > 0:
            l_slice = left[..., shift:]
            r_slice = right[..., : width - shift]
        else:
            l_slice = left
            r_slice = right
        if l_slice.numel() == 0:
            continue
        diff = (l_slice.float() - r_slice.float()).abs()
        row = {
            "shift": shift,
            "overlap_columns": int(l_slice.shape[-1]),
            "bit_exact": bool(torch.equal(l_slice, r_slice)),
            "num_mismatched_elements": int((l_slice != r_slice).sum().item()),
            "max_abs": float(diff.max().item()),
        }
        if best is None or (row["num_mismatched_elements"], row["max_abs"]) < (
            best["num_mismatched_elements"],
            best["max_abs"],
        ):
            best = row
    return best


def evaluate(
    payload_path: Path,
    *,
    conv_width: int = 4,
    out_path: Path | None = None,
) -> dict[str, Any]:
    payload = _load_payload(payload_path)
    parent = [int(x) for x in payload.get("tree_parent") or []]
    if len(parent) < 2:
        raise ValueError("tree_parent missing or not engaged")
    accepted_len = int(payload["accepted_len"])
    if accepted_len <= 0:
        raise ValueError("accepted_len must be positive for event-0 handoff probe")
    accepted_node, accepted_path = _accepted_node_and_path(payload, parent)
    if len(accepted_path) != accepted_len:
        raise ValueError(
            "accepted_len/path mismatch: "
            f"accepted_len={accepted_len}, accepted_path_len={len(accepted_path)}"
        )

    serving_tree_state = _require_tensor(payload, "serving_tree_state")
    next_read_ssm = _require_tensor(payload, "next_read_ssm_state")
    serving_conv_rows = _require_tensor(payload, "serving_conv_rows")
    next_read_conv = _require_tensor(payload, "next_read_conv_state")
    if accepted_node >= int(serving_tree_state.shape[0]):
        raise ValueError("accepted node outside serving_tree_state rows")
    if accepted_node >= int(serving_conv_rows.shape[0]):
        raise ValueError("accepted node outside serving_conv_rows rows")

    tree_h0 = serving_tree_state[accepted_node].contiguous()
    tree_conv_row = serving_conv_rows[accepted_node].contiguous()
    native_h0 = next_read_ssm.contiguous()
    native_conv_row = next_read_conv.contiguous()

    if tree_conv_row.ndim != 2:
        raise ValueError(f"expected conv rows [D, state_len], got {tuple(tree_conv_row.shape)}")
    if native_conv_row.shape != tree_conv_row.shape:
        raise ValueError(
            f"conv row shape mismatch: tree={tuple(tree_conv_row.shape)} "
            f"native={tuple(native_conv_row.shape)}"
        )
    state_len = int(tree_conv_row.shape[1])
    if conv_width < 2:
        raise ValueError("--conv-width must be >= 2")
    read_col = max(0, accepted_len - 1)
    window_cols = conv_width - 1
    if read_col + window_cols > state_len:
        raise ValueError(
            f"native conv read window [{read_col}, {read_col + window_cols}) "
            f"exceeds state_len={state_len}"
        )
    tree_conv_window = tree_conv_row[:, read_col : read_col + window_cols].contiguous()
    native_conv_window = native_conv_row[:, read_col : read_col + window_cols].contiguous()

    h0_stats = _tensor_stats(tree_h0, native_h0)
    conv_window_stats = _tensor_stats(tree_conv_window, native_conv_window)
    conv_row_stats = _tensor_stats(tree_conv_row, native_conv_row)
    h0_match = bool(h0_stats["shape_match"] and h0_stats["bit_exact"])
    conv_window_match = bool(
        conv_window_stats["shape_match"] and conv_window_stats["bit_exact"]
    )
    wrong_initial_state = not (h0_match and conv_window_match)
    result = {
        "schema": "fr11.probe_beta_event0_state_handoff.v1",
        "payload": str(payload_path),
        "layer_prefix": payload.get("layer_prefix"),
        "batch_index": int(payload.get("batch_index", 0)),
        "tree_engaged": True,
        "tree_parent": parent,
        "accepted_len": accepted_len,
        "accepted_node_id": int(payload["accepted_node_id"]),
        "accepted_gdn_node_id": accepted_node,
        "accepted_gdn_node_path": accepted_path,
        "accepted_spec_state_bank_row": payload.get("accepted_spec_state_bank_row"),
        "accepted_bank_row": payload.get("accepted_bank_row"),
        "next_read_bank_row": payload.get("next_read_bank_row"),
        "address_coincide": bool(
            payload.get("accepted_bank_row") is not None
            and payload.get("next_read_bank_row") is not None
            and int(payload["accepted_bank_row"]) == int(payload["next_read_bank_row"])
        ),
        "conv_width": conv_width,
        "conv_state_len": state_len,
        "native_conv_state_token_offset": read_col,
        "conv_prior_window_columns": [read_col, read_col + window_cols],
        "h0_tree_vs_native": h0_stats,
        "conv_prior_window_tree_vs_native": conv_window_stats,
        "conv_full_row_tree_vs_native": conv_row_stats,
        "conv_full_row_first_mismatch_column": _first_mismatch_column(
            tree_conv_row, native_conv_row
        ),
        "conv_window_by_column": _per_column_stats(tree_conv_window, native_conv_window),
        "conv_full_row_best_column_shift": _best_column_shift(tree_conv_row, native_conv_row),
        "probe_beta_match": bool(h0_match and conv_window_match),
        "verdict": (
            "MATCH_WRONG_INITIAL_STATE_EXCLUDED"
            if not wrong_initial_state
            else "BUG_FIXABLE_WRONG_INITIAL_STATE"
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
    parser.add_argument("--conv-width", type=int, default=4)
    args = parser.parse_args()
    result = evaluate(args.payload, conv_width=args.conv_width, out_path=args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["probe_beta_match"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

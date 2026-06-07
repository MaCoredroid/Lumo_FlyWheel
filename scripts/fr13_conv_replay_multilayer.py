#!/usr/bin/env python3
"""Replay GDN causal-conv arithmetic across multiple captured layers.

This is boot-free. It uses saved FR12 subkernel capture payloads and compares
candidate manual conv arithmetic against native causal_conv1d_update outputs
where a native target row is available. Branch rows are carried through the
report; they require a native-on-path oracle capture to become targetable.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _layer_idx(payload: dict[str, Any]) -> int:
    match = re.search(r"layers\.(\d+)\.", str(payload.get("layer_prefix", "")))
    if not match:
        raise ValueError(f"cannot parse layer from {payload.get('layer_prefix')!r}")
    return int(match.group(1))


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }


def _first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    d = (a.float() - b.float()).abs()
    if not d.numel():
        return None
    flat = int(torch.argmax(d).item())
    val = float(d.reshape(-1)[flat].item())
    if val == 0.0:
        return None
    idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), d.shape)]
    return {
        "index": idx,
        "lhs": float(a[tuple(idx)].float().item()),
        "rhs": float(b[tuple(idx)].float().item()),
        "abs": val,
    }


def _by_layer(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out = {}
    for path in paths:
        payload = _load(path)
        out[_layer_idx(payload)] = payload | {"_path": str(path)}
    return out


def _detail(payload: dict[str, Any], key: str) -> dict[str, Any]:
    meta = payload.get("meta", {})
    if key not in meta:
        raise KeyError(f"{payload.get('_path', payload.get('call_path'))}: missing {key}")
    return meta[key]


def _order_sum(taps: torch.Tensor, order: str) -> torch.Tensor:
    if order == "0123":
        return ((taps[:, 0] + taps[:, 1]) + taps[:, 2]) + taps[:, 3]
    if order == "3210":
        return ((taps[:, 3] + taps[:, 2]) + taps[:, 1]) + taps[:, 0]
    if order == "pair01_23":
        return (taps[:, 0] + taps[:, 1]) + (taps[:, 2] + taps[:, 3])
    if order == "pair03_12":
        return (taps[:, 0] + taps[:, 3]) + (taps[:, 1] + taps[:, 2])
    raise ValueError(order)


def _silu(acc: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "torch":
        return torch.nn.functional.silu(acc)
    if mode == "exp":
        return acc / (1.0 + torch.exp(-acc))
    if mode == "exp2":
        return acc / (1.0 + torch.exp2(acc.new_tensor(-1.4426950408889634) * acc))
    raise ValueError(mode)


def _store(out: torch.Tensor, mode: str) -> torch.Tensor:
    rounded = out.to(torch.bfloat16)
    if mode == "torch_bf16":
        return rounded
    rounded_f = rounded.float()
    lower = torch.nextafter(rounded, torch.full_like(rounded, -float("inf")))
    lower_f = lower.float()
    midpoint = (rounded_f + lower_f) * 0.5
    tie = (rounded_f > out) & (out == midpoint)
    if mode == "tie_all_down":
        return torch.where(tie, lower, rounded)
    if mode == "tie_positive_down":
        return torch.where(tie & (out > 0), lower, rounded)
    raise ValueError(mode)


def _variant_output(
    detail: dict[str, Any],
    *,
    tap_product: str,
    order: str,
    silu: str,
    store: str,
) -> torch.Tensor:
    if tap_product == "bf16":
        taps = detail.get(
            "tap_products_bf16_selected", detail["tap_products_bf16_path0"]
        ).float()
    elif tap_product == "fp32":
        taps = detail.get(
            "tap_products_fp32_selected", detail["tap_products_fp32_path0"]
        ).float()
    else:
        raise ValueError(tap_product)
    acc = _order_sum(taps, order)
    if detail.get("conv_bias") is not None:
        acc = acc + detail["conv_bias"].float().unsqueeze(0)
    if str(detail.get("activation", "silu")) in ("True", "silu", "swish"):
        acc = _silu(acc, silu)
    return _store(acc, store)


def _target_rows(
    tree_payload: dict[str, Any],
    native_payload: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    tree_detail = _detail(tree_payload, "tree_conv_detail")
    native_detail = _detail(native_payload, "native_conv_detail")
    selected_nodes = tree_detail.get("selected_nodes", tree_detail["path0_nodes"])
    selected_nodes_list = [int(x) for x in selected_nodes.reshape(-1).tolist()]
    path0_nodes = [int(x) for x in tree_detail["path0_nodes"].reshape(-1).tolist()]
    path0_pos = {node: idx for idx, node in enumerate(path0_nodes)}
    tree_conv = tree_payload["stages"]["conv1d_out"]["tensor"]
    native_conv = native_payload["stages"]["conv1d_out"]["tensor"]

    tree_rows = []
    native_rows = []
    mapped = []
    missing = []
    for selected_pos, node in enumerate(selected_nodes_list):
        if node in path0_pos:
            tree_rows.append(selected_pos)
            native_rows.append(path0_pos[node])
            mapped.append(
                {
                    "selected_pos": int(selected_pos),
                    "tree_node": int(node),
                    "native_row": int(path0_pos[node]),
                    "target_source": "native_path0_capture",
                }
            )
        else:
            missing.append(
                {
                    "selected_pos": int(selected_pos),
                    "tree_node": int(node),
                    "target_source": "missing_native_on_path_oracle",
                }
            )
    selected_tree = tree_conv.index_select(0, selected_nodes.to(torch.long))
    native_target = native_conv.index_select(0, torch.tensor(native_rows, dtype=torch.long))
    return selected_tree, native_target, mapped, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", nargs="+", type=Path, required=True)
    parser.add_argument("--native", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tree_layers = _by_layer(args.tree)
    native_layers = _by_layer(args.native)
    variants = [
        {
            "tap_product": tap,
            "order": order,
            "silu": silu,
            "store": store,
        }
        for tap in ("bf16", "fp32")
        for order in ("0123", "3210", "pair01_23", "pair03_12")
        for silu in ("torch", "exp", "exp2")
        for store in ("torch_bf16", "tie_positive_down", "tie_all_down")
    ]

    layer_reports = []
    aggregate: dict[str, list[torch.Tensor]] = {
        json.dumps(v, sort_keys=True): [] for v in variants
    }
    aggregate_targets: list[torch.Tensor] = []

    for layer in sorted(set(tree_layers) & set(native_layers)):
        tree_payload = tree_layers[layer]
        native_payload = native_layers[layer]
        tree_detail = _detail(tree_payload, "tree_conv_detail")
        native_detail = _detail(native_payload, "native_conv_detail")
        selected_tree, native_target, mapped, missing = _target_rows(
            tree_payload, native_payload
        )
        target_positions = [int(row["selected_pos"]) for row in mapped]
        target_pos_tensor = torch.tensor(target_positions, dtype=torch.long)
        selected_target_tree = selected_tree.index_select(0, target_pos_tensor)
        baseline = _metrics(selected_target_tree, native_target)
        aggregate_targets.append(native_target)

        variant_rows = []
        for variant in variants:
            key = json.dumps(variant, sort_keys=True)
            out = _variant_output(tree_detail, **variant)
            out_target = out.index_select(0, target_pos_tensor)
            aggregate[key].append(out_target)
            stats = _metrics(out_target, native_target)
            variant_rows.append(
                {
                    **variant,
                    **stats,
                    "first_mismatch": _first_mismatch(out_target, native_target),
                }
            )

        weight_stats = None
        if "conv_weights" in tree_detail and "conv_weights" in native_detail:
            weight_stats = _metrics(
                tree_detail["conv_weights"], native_detail["conv_weights"]
            )
        bias_stats = None
        if tree_detail.get("conv_bias") is not None and native_detail.get("conv_bias") is not None:
            bias_stats = _metrics(tree_detail["conv_bias"], native_detail["conv_bias"])

        layer_reports.append(
            {
                "layer": int(layer),
                "tree_capture": tree_payload["_path"],
                "native_capture": native_payload["_path"],
                "target_mapped_rows": mapped,
                "missing_target_rows": missing,
                "captured_tree_vs_native": {
                    **baseline,
                    "first_mismatch": _first_mismatch(selected_target_tree, native_target),
                },
                "conv_weights": weight_stats,
                "conv_bias": bias_stats,
                "best_variant": min(
                    variant_rows,
                    key=lambda row: (row["max_abs"], row["mean_abs"], row["nonzero"]),
                ),
                "variants": variant_rows,
            }
        )

    aggregate_rows = []
    if aggregate_targets:
        target_all = torch.cat(aggregate_targets, dim=0)
        for key, chunks in aggregate.items():
            if not chunks:
                continue
            variant = json.loads(key)
            out_all = torch.cat(chunks, dim=0)
            stats = _metrics(out_all, target_all)
            aggregate_rows.append(
                {
                    **variant,
                    **stats,
                    "first_mismatch": _first_mismatch(out_all, target_all),
                }
            )

    payload = {
        "schema": "fr13.conv_replay_multilayer.v1",
        "tree_captures": [str(p) for p in args.tree],
        "native_captures": [str(p) for p in args.native],
        "targetable_layers": sorted(set(tree_layers) & set(native_layers)),
        "aggregate_best_variant": None
        if not aggregate_rows
        else min(
            aggregate_rows,
            key=lambda row: (row["max_abs"], row["mean_abs"], row["nonzero"]),
        ),
        "aggregate_variants": aggregate_rows,
        "layers": layer_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "aggregate_best_variant": payload["aggregate_best_variant"],
                "layers": [
                    {
                        "layer": row["layer"],
                        "captured_tree_vs_native": row["captured_tree_vs_native"],
                        "best_variant": row["best_variant"],
                        "missing_target_rows": row["missing_target_rows"],
                    }
                    for row in layer_reports
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

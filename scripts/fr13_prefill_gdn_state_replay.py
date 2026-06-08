#!/usr/bin/env python3
"""Compare prefill GDN captures and localize recurrent-state divergence.

The capture is emitted by FR13_PREFILL_GDN_CAPTURE from the patched vLLM
GDN layer.  It records the prefill conv input/output, post-conv q/k/v/g/beta,
scan initial state, chunk output, and final recurrent state.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


STAGES = [
    "pre_conv",
    "conv_out",
    "a_non_spec",
    "b_non_spec",
    "query",
    "key",
    "value",
    "g",
    "beta",
    "initial_state",
    "core_out",
    "final_state",
]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False) | {"_path": str(path)}


def _layer_idx(payload: dict[str, Any]) -> int:
    match = re.search(r"layers\.(\d+)\.", str(payload.get("layer_prefix", "")))
    if not match:
        raise ValueError(f"cannot parse layer from {payload.get('layer_prefix')!r}")
    return int(match.group(1))


def _by_layer(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out = {}
    for path in paths:
        payload = _load(path)
        out[_layer_idx(payload)] = payload
    return out


def _metrics(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True}
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_mismatch": [list(a.shape), list(b.shape)],
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
        }
    d = (a.float() - b.float()).abs()
    out: dict[str, Any] = {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "torch_equal": bool(torch.equal(a, b)),
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }
    if out["nonzero"]:
        flat = int(torch.argmax(d).item())
        idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), d.shape)]
        out["first_mismatch"] = {
            "index": idx,
            "lhs": float(a[tuple(idx)].float().item()),
            "rhs": float(b[tuple(idx)].float().item()),
            "abs": float(d[tuple(idx)].item()),
        }
    else:
        out["first_mismatch"] = None
    return out


def _token_metrics(a: torch.Tensor, b: torch.Tensor) -> list[dict[str, Any]]:
    if a.ndim == 0 or b.ndim == 0 or a.shape[0] != b.shape[0]:
        return []
    out = []
    for tok in range(int(a.shape[0])):
        m = _metrics(a[tok], b[tok])
        out.append(
            {
                "token": tok,
                "max_abs": m.get("max_abs"),
                "nonzero": m.get("nonzero"),
                "first_mismatch": m.get("first_mismatch"),
            }
        )
    return out


def _sequence_ranges(payload: dict[str, Any]) -> list[tuple[int, int]]:
    qsl = payload.get("query_start_loc")
    if qsl is None:
        n = int(payload["query"].shape[0])
        return [(0, n)]
    vals = [int(x) for x in qsl.reshape(-1).tolist()]
    return [(vals[i], vals[i + 1]) for i in range(max(0, len(vals) - 1))]


def _manual_scan_states(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    q = payload["query"].float()
    k = payload["key"].float()
    v = payload["value"].float()
    g = payload["g"].float()
    beta = payload["beta"].float()
    initial = payload["initial_state"].float()
    num_kh = int(q.shape[1])
    num_vh = int(v.shape[1])
    head_group = num_vh // num_kh
    out = torch.empty_like(v, dtype=torch.float32)
    states = torch.empty(
        (int(q.shape[0]), num_vh, int(v.shape[2]), int(q.shape[2])),
        dtype=torch.float32,
    )
    for seq_i, (start, end) in enumerate(_sequence_ranges(payload)):
        state = initial[seq_i].clone()
        for tok in range(start, end):
            for vh in range(num_vh):
                kh = vh // head_group
                k_t = k[tok, kh]
                q_t = q[tok, kh]
                decayed = state[vh] * torch.exp(g[tok, vh])
                delta_v = v[tok, vh] - torch.sum(decayed * k_t.view(1, -1), dim=1)
                delta_v = delta_v * beta[tok, vh]
                state[vh] = decayed + delta_v.view(-1, 1) * k_t.view(1, -1)
                out[tok, vh] = torch.sum(state[vh] * q_t.view(1, -1), dim=1)
            states[tok] = state
    return out, states


def _try_fla_replay(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule

    device = torch.device("cuda")
    q = payload["query"].to(device).unsqueeze(0).contiguous()
    k = payload["key"].to(device).unsqueeze(0).contiguous()
    v = payload["value"].to(device).unsqueeze(0).contiguous()
    g = payload["g"].to(device).unsqueeze(0).contiguous()
    beta = payload["beta"].to(device).unsqueeze(0).contiguous()
    initial = payload["initial_state"].to(device).contiguous()
    qsl = payload.get("query_start_loc")
    cu = None if qsl is None else qsl.to(device).contiguous()
    chunk_indices = payload.get("chunk_indices")
    chunk_offsets = payload.get("chunk_offsets")
    replay_out, replay_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=None if chunk_indices is None else chunk_indices.to(device),
        chunk_offsets=None if chunk_offsets is None else chunk_offsets.to(device),
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    return {
        "core_out": replay_out.squeeze(0).detach().cpu(),
        "final_state": replay_state.detach().cpu(),
    }


def _compare_layer(tree: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    stage_metrics = {name: _metrics(tree.get(name), native.get(name)) for name in STAGES}
    first = None
    for name in STAGES:
        m = stage_metrics[name]
        if m.get("shape_mismatch") or (m.get("max_abs") not in (None, 0.0)):
            first = name
            break
    token_breakdown = {}
    for name in ("pre_conv", "conv_out", "query", "key", "value", "g", "beta", "core_out"):
        if name in tree and name in native and tuple(tree[name].shape) == tuple(native[name].shape):
            token_breakdown[name] = _token_metrics(tree[name], native[name])
    manual_tree_out, manual_tree_states = _manual_scan_states(tree)
    manual_native_out, manual_native_states = _manual_scan_states(native)
    replay: dict[str, Any] = {
        "manual_tree_vs_native": {
            "core_out": _metrics(manual_tree_out, manual_native_out),
            "per_token_state": _metrics(manual_tree_states, manual_native_states),
            "tree_manual_vs_capture_core_out": _metrics(manual_tree_out, tree["core_out"]),
            "native_manual_vs_capture_core_out": _metrics(manual_native_out, native["core_out"]),
            "tree_manual_final_vs_capture_final": _metrics(
                manual_tree_states[-1:].reshape_as(tree["final_state"]),
                tree["final_state"],
            ) if tree["final_state"].numel() == manual_tree_states[-1:].numel() else {"shape_mismatch": [list(manual_tree_states[-1:].shape), list(tree["final_state"].shape)]},
            "native_manual_final_vs_capture_final": _metrics(
                manual_native_states[-1:].reshape_as(native["final_state"]),
                native["final_state"],
            ) if native["final_state"].numel() == manual_native_states[-1:].numel() else {"shape_mismatch": [list(manual_native_states[-1:].shape), list(native["final_state"].shape)]},
        }
    }
    try:
        fla_tree = _try_fla_replay(tree)
        fla_native = _try_fla_replay(native)
        if fla_tree is not None and fla_native is not None:
            replay["fla_replay"] = {
                "tree_replay_vs_capture_core_out": _metrics(
                    fla_tree["core_out"], tree["core_out"]
                ),
                "native_replay_vs_capture_core_out": _metrics(
                    fla_native["core_out"], native["core_out"]
                ),
                "tree_replay_vs_native_replay_core_out": _metrics(
                    fla_tree["core_out"], fla_native["core_out"]
                ),
                "tree_replay_vs_capture_final_state": _metrics(
                    fla_tree["final_state"], tree["final_state"]
                ),
                "native_replay_vs_capture_final_state": _metrics(
                    fla_native["final_state"], native["final_state"]
                ),
                "tree_replay_vs_native_replay_final_state": _metrics(
                    fla_tree["final_state"], fla_native["final_state"]
                ),
            }
    except Exception as exc:
        replay["fla_replay_error"] = type(exc).__name__ + ":" + str(exc)
    return {
        "layer": _layer_idx(tree),
        "tree_capture": tree["_path"],
        "native_capture": native["_path"],
        "num_actual_tokens": int(tree.get("num_actual_tokens", -1)),
        "tree_query_start_loc": (
            None
            if tree.get("query_start_loc") is None
            else [int(x) for x in tree["query_start_loc"].reshape(-1).tolist()]
        ),
        "native_query_start_loc": (
            None
            if native.get("query_start_loc") is None
            else [int(x) for x in native["query_start_loc"].reshape(-1).tolist()]
        ),
        "first_diverging_stage": first,
        "stages": stage_metrics,
        "token_breakdown": token_breakdown,
        "replay": replay,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", nargs="+", type=Path, required=True)
    parser.add_argument("--native", nargs="+", type=Path, required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tree_layers = _by_layer(args.tree)
    native_layers = _by_layer(args.native)
    layers = sorted(set(tree_layers) & set(native_layers))
    if args.layers.strip():
        wanted = {int(x.strip()) for x in args.layers.split(",") if x.strip()}
        layers = [layer for layer in layers if layer in wanted]
    results = [_compare_layer(tree_layers[layer], native_layers[layer]) for layer in layers]
    first_layer = None
    for row in results:
        if row["first_diverging_stage"] is not None:
            first_layer = row
            break
    payload = {
        "schema": "fr13.prefill_gdn_state_replay.v1",
        "tree_captures": [str(p) for p in args.tree],
        "native_captures": [str(p) for p in args.native],
        "layers": [row["layer"] for row in results],
        "first_diverging_layer": None if first_layer is None else first_layer["layer"],
        "first_diverging_stage": None if first_layer is None else first_layer["first_diverging_stage"],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        "layers": payload["layers"],
        "first_diverging_layer": payload["first_diverging_layer"],
        "first_diverging_stage": payload["first_diverging_stage"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    for row in results:
        print(
            json.dumps(
                {
                    "layer": row["layer"],
                    "first": row["first_diverging_stage"],
                    "stage_max": {
                        name: row["stages"][name].get("max_abs")
                        for name in STAGES
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if first_layer is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

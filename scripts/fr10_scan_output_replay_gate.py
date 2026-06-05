#!/usr/bin/env python3
"""FR10 scan-OUTPUT replay gate (the seam FR10_STATUS flags as MISSING).

The src_native gate proved the recurrent STATE handoff byte-native. It did NOT
prove the per-position scan OUTPUT (out[i]) that actually feeds the logit. This
gate replays the live captured tree-GDN inputs through the kernel and compares
each node's OUTPUT to the native serial-per-path reference -- with special focus
on node 0 (the bonus/root = the depth-0 target source).

Runs in a LIGHTWEIGHT GPU container (torch+triton+kernel, NO model load) in
seconds:
    docker run --rm --gpus all <vllm-image> \
        python /work/scripts/fr10_scan_output_replay_gate.py \
        --payload <output/.../logs/fr10_src_native_handoff.pt> --out <gate.json>

Verdict:
  - node0 (root) out_vs_native ~0  -> GDN scan output is native at the root ->
    the depth-0 logit bug is DOWNSTREAM (full-attention / position / LM-head).
  - node0 out_vs_native large       -> the GDN scan OUTPUT itself is wrong at the
    root (kernel-fixable; iterate here without a 30-min model boot).
Powered negative control: a neighbor-row output must NOT match (else vacuous).
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
    padded_nodes,
)

_VALIDATION_PATH = REPO / "scripts" / "fr10_phase4_real_tensor_validation.py"
_spec = importlib.util.spec_from_file_location(
    "fr10_phase4_real_tensor_validation", _VALIDATION_PATH
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {_VALIDATION_PATH}")
_val = importlib.util.module_from_spec(_spec)
sys.modules["fr10_phase4_real_tensor_validation"] = _val
_spec.loader.exec_module(_val)
native_update_serial_per_path = _val.native_update_serial_per_path


def _linear_parent(n: int) -> list[int]:
    return [-1] + [idx - 1 for idx in range(1, n)]


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


def _node_major(payload: dict[str, Any], name: str) -> torch.Tensor:
    tensor = payload[name]
    if name == "value_spec" and tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    q_rows = int(payload["query_spec"].shape[0])
    if tensor.shape[0] > q_rows:
        tensor = tensor[:q_rows]
    return tensor


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _native_out_for_path(
    payload, q, k, value_spec, a, b, A_log, dt_bias, h0, output_scale, path, device
):
    idx = torch.tensor(path, device=device, dtype=torch.long)
    outs, _states = native_update_serial_per_path(
        Tree(tuple(_linear_parent(len(path)))),
        q.index_select(0, idx).contiguous(),
        k.index_select(0, idx).contiguous(),
        value_spec.index_select(0, idx).contiguous(),
        a.index_select(0, idx).contiguous(),
        b.index_select(0, idx).contiguous(),
        A_log,
        dt_bias,
        h0,
        output_scale,
    )
    # outs is a per-step list/stack; the LAST entry is this path's final node output.
    last = outs[-1] if isinstance(outs, (list, tuple)) else outs[-1]
    return last.contiguous()


def evaluate(payload_path: Path, out_path: Path | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FR10 scan-output replay")
    payload = torch.load(payload_path, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    n_actual = len(parent)
    n_pad = padded_nodes(n_actual)
    device = torch.device("cuda")
    tree = Tree(tuple(parent))
    strict, visible = tree.masks(device, n_pad)

    q = _node_major(payload, "query_spec").to(device).contiguous()
    k = _node_major(payload, "key_spec").to(device).contiguous()
    v = _node_major(payload, "value_tree").to(device).contiguous()
    g = _node_major(payload, "g_tree").to(device).contiguous()
    beta = _node_major(payload, "beta_tree").to(device).contiguous()
    value_spec = _node_major(payload, "value_spec").to(device).contiguous()
    a = _node_major(payload, "a").to(device).contiguous()
    b = _node_major(payload, "b").to(device).contiguous()
    h0 = payload["prev_h0"].to(device).contiguous()
    output_scale = float(payload["output_scale"])
    A_log = payload["A_log"].to(device).contiguous()
    dt_bias = payload["dt_bias"].to(device).contiguous()

    replay_out, _replay_state = launch_tree_gdn_prepared(
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
    torch.cuda.synchronize()

    rows = []
    for node in range(n_actual):
        path = _path_to(parent, node)
        native_out = _native_out_for_path(
            payload, q, k, value_spec, a, b, A_log, dt_bias, h0, output_scale, path, device
        )
        rows.append(
            {
                "node": node,
                "depth": len(path) - 1,
                "parent": parent[node],
                "out_vs_native_max_abs": _max_abs(replay_out[node], native_out),
            }
        )

    root = rows[0]
    # Powered negative control: root output must NOT match a DIFFERENT node's native out.
    neg_node = 1 if n_actual > 1 else 0
    neg_native = _native_out_for_path(
        payload, q, k, value_spec, a, b, A_log, dt_bias, h0, output_scale,
        _path_to(parent, neg_node), device,
    )
    negative_root_vs_other = _max_abs(replay_out[0], neg_native)

    atol = 1e-3
    result = {
        "schema": "fr10.scan_output_replay_gate.v1",
        "payload": str(payload_path),
        "layer_prefix": payload.get("layer_prefix"),
        "accepted_len": int(payload.get("accepted_len", -1)),
        "n_actual": n_actual,
        "atol": atol,
        "root_out_vs_native_max_abs": root["out_vs_native_max_abs"],
        "root_pass": bool(root["out_vs_native_max_abs"] <= atol),
        "max_node_out_vs_native": max(r["out_vs_native_max_abs"] for r in rows),
        "negative_control_root_vs_other_node": negative_root_vs_other,
        "negative_control_powered": bool(negative_root_vs_other > atol),
        "per_node": rows,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    res = evaluate(args.payload, args.out)
    print(json.dumps(res, indent=2, sort_keys=True))
    # exit 0 only if root output is native AND the negative control is powered
    ok = res["root_pass"] and res["negative_control_powered"]
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

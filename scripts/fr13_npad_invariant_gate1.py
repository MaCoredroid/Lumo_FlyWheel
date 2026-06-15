#!/usr/bin/env python3
"""FR13 GATE 1 — N_PAD-INVARIANT reduction-order DE-CONFOUND proof.

MECHANISM: the deployed ``_tree_gdn_kernel`` scan loops ``tl.static_range(0,
N_PAD)`` and reduces parents via ``tl.sum(tl.where(offs_n==j, h_cache, 0.0))``
with ``offs_n = tl.arange(0, N_PAD)``.  N_PAD = 1<<(n-1).bit_length() GROWS with
the number of co-resident tree nodes (cat9 caterpillar -> N_PAD=16; a leaf-free
5-node spine chain5 -> N_PAD=8).  A larger N_PAD recompiles the scan/reduction to
a different unrolled FMA/reduction tree, so the SAME spine nodes (identical q/k/
v/g/beta inputs) get a different ROUNDING ORDER (bug-class #10 codegen identity;
MEASURED state gap ~0.0289).

This gate runs the SHARED depth-5 spine chain under TWO topologies built from the
SAME payload rows:
  * cat9   : caterpillar [-1,0,1,2,3,0,1,2,3]  (spine 0..4 + 4 leaves), N_PAD=16
  * chain5 : leaf-free   [-1,0,1,2,3]          (spine 0..4 only),       N_PAD=8
The spine rows (payload rows 0..4) are byte-identical between the two trees, so
the spine STATES can differ ONLY through the N_PAD-dependent reduction order.

FR13_NPAD_INVARIANT ON pins BOTH trees' scan span to the fixed N_FIXED=16, so the
reduction order is identical and the spine states must int-view-match (0.0).
OFF: cat9 keeps N_PAD=16, chain5 keeps N_PAD=8 -> the order differs -> the spine
states int-view-MISMATCH with a ~0.0289 max_abs (the NEGATIVE CONTROL that powers
the gate; without a measurable OFF gap the ON 0.0 is vacuous, bug-class #9).

num_warps stays the DEPLOYED 8 in every arm (geometry HELD; this is NOT the
refuted recompute route that changed geometry to native BV32/w1/s3).  int-view
equality, NEVER atol (#10).

Usage (inside the pinned GPU container):
  python3 scripts/fr13_npad_invariant_gate1.py --payload <tree_gdn_payload.pt> --out <gate1.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (  # noqa: E402
    Tree,
    launch_tree_gdn_prepared,
    npad_invariant_on,
)

# cat9 caterpillar parent array (root chain 0..4 + leaves 5..8 hanging off 0..3)
# and the leaf-free 5-node spine chain.  The first 5 nodes (0..4) are the SHARED
# spine in BOTH trees and consume IDENTICAL payload rows.
CAT9_PARENT = [-1, 0, 1, 2, 3, 0, 1, 2, 3]      # 9 nodes, N_PAD=16
CHAIN5_PARENT = [-1, 0, 1, 2, 3]                 # 5 nodes, N_PAD=8
SPINE_NODES = [0, 1, 2, 3, 4]                    # depth-5 shared spine


def _n_pad(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _int_view_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    a = a.contiguous()
    b = b.to(a.dtype).contiguous()
    if a.shape != b.shape:
        return False
    if a.dtype == torch.float32:
        return bool(torch.equal(a.view(torch.int32), b.view(torch.int32)))
    if a.dtype in (torch.bfloat16, torch.float16):
        return bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    if a.dtype == torch.float64:
        return bool(torch.equal(a.view(torch.int64), b.view(torch.int64)))
    return bool(torch.equal(a, b))


def _select(t: torch.Tensor, rows: list[int]) -> torch.Tensor:
    if t.ndim >= 1 and t.size(0) >= max(rows) + 1 and t.ndim in (2, 3):
        return t.index_select(0, torch.tensor(rows, device=t.device)).contiguous()
    return t.contiguous()


def _run(payload: dict[str, Any], parent: list[int], rows: list[int], device) -> torch.Tensor:
    """Run the deployed scan on `parent` topology with payload rows `rows`.

    Returns the per-node STATE tensor [n_actual, NUM_VH, DIM_V, DIM_K].
    """
    n_actual = len(parent)
    n_pad = _n_pad(n_actual)
    tree = Tree(tuple(parent))
    strict, visible = tree.masks(device, n_pad)
    # per-node row tensors (q/k/v/g/beta) selected to the spine+leaf rows
    q = _select(payload["query_spec"].to(device), rows)
    k = _select(payload["key_spec"].to(device), rows)
    v = _select(payload["value_tree"].to(device), rows)
    g = _select(payload["g_tree"].to(device), rows)
    beta = _select(payload["beta_tree"].to(device), rows)
    a = _select(payload["a"].to(device), rows)
    b = _select(payload["b"].to(device), rows)
    A_log = payload["A_log"].to(device).contiguous()
    dt_bias = payload["dt_bias"].to(device).contiguous()
    h0 = payload["h0"].to(device).contiguous()
    out, state = launch_tree_gdn_prepared(
        q=q, k=k, v=v, g=g, beta=beta, raw_a=a, raw_b=b,
        A_log=A_log, dt_bias=dt_bias, h0=h0,
        n_actual=n_actual, n_pad=n_pad,
        strict_mask=strict, visible_mask=visible,
        output_scale=float(payload["output_scale"]),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    if state is None:
        raise RuntimeError("kernel returned no per-node state (STORE_NODE_STATES off)")
    return state[:n_actual].contiguous()


def _spine_compare(cat9_state: torch.Tensor, chain5_state: torch.Tensor) -> dict[str, Any]:
    cs = cat9_state[SPINE_NODES].contiguous()
    hs = chain5_state[SPINE_NODES].contiguous()
    diff = (cs.float() - hs.float()).abs()
    return {
        "int_view_equal": _int_view_equal(cs, hs),
        "max_abs": float(diff.max().item()),
        "cat9_spine_norm": float(cs.float().norm().item()),
        "chain5_spine_norm": float(hs.float().norm().item()),
        "both_norms_positive": bool(cs.float().norm() > 0 and hs.float().norm() > 0),
        "rel_err": float((diff.max() / (hs.float().abs().max() + 1e-12)).item()),
    }


def _arm(payload, device, label: str) -> dict[str, Any]:
    cat9_state = _run(payload, CAT9_PARENT, list(range(9)), device)
    # chain5 uses the SAME spine rows 0..4 (byte-identical inputs to the spine).
    chain5_state = _run(payload, CHAIN5_PARENT, list(range(5)), device)
    cmp = _spine_compare(cat9_state, chain5_state)
    cmp["arm"] = label
    cmp["npad_invariant_env"] = os.environ.get("FR13_NPAD_INVARIANT", "")
    cmp["npad_invariant_on"] = bool(npad_invariant_on())
    cmp["cat9_n_pad"] = _n_pad(9)
    cmp["chain5_n_pad"] = _n_pad(5)
    return cmp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--payload", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("GATE1 requires CUDA")
    device = torch.device("cuda")
    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    if int(payload.get("n_pad", 0)) < 16:
        raise SystemExit(f"payload n_pad={payload.get('n_pad')} < 16; need a cat9-capacity payload")

    result: dict[str, Any] = {"payload": str(args.payload), "spine_nodes": SPINE_NODES}

    # Arm A: WITH the flag (both trees pinned to N_FIXED=16) -> expect int-view 0.0.
    os.environ["FR13_NPAD_INVARIANT"] = "1"
    assert npad_invariant_on(), "flag fn did not read FR13_NPAD_INVARIANT=1"
    result["flag_on"] = _arm(payload, device, "FR13_NPAD_INVARIANT=1")

    # Arm B: WITHOUT the flag (cat9 N_PAD=16, chain5 N_PAD=8) -> expect ~0.0289 gap
    # (the powered negative control).
    os.environ["FR13_NPAD_INVARIANT"] = "0"
    assert not npad_invariant_on(), "flag fn still ON after FR13_NPAD_INVARIANT=0"
    result["flag_off"] = _arm(payload, device, "FR13_NPAD_INVARIANT=0")

    on = result["flag_on"]
    off = result["flag_off"]
    result["gate1_pass"] = bool(
        on["int_view_equal"]
        and on["both_norms_positive"]
        and (not off["int_view_equal"])          # neg-control must be powered
        and off["both_norms_positive"]
        and off["max_abs"] > 0.0
    )
    result["neg_control_powered"] = bool(
        (not off["int_view_equal"]) and off["max_abs"] > 0.0 and off["both_norms_positive"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

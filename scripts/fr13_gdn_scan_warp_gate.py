#!/usr/bin/env python3
"""FR13 GDN tree-scan A/B gate vs the NATIVE PACKED-DECODE incumbent SASS.

THE DECISIVE carrier test. The deployed scan ``_tree_gdn_kernel`` was only ever
A/B'd vs ``native_update_serial_per_path`` -- a SERIAL TORCH reference using
``fused_sigmoid_gating_delta_rule_update`` -- NEVER vs the actual Triton kernel
the no-spec B=1 decode oracle dispatches:
``fused_recurrent_gated_delta_rule_packed_decode`` (confirmed from the live
dispatch in gdn_linear_attn.py _forward_core -> _forward_core_decode_non_spec).
bit-exact-to-a-serial-ref != bit-exact-to-the-incumbent-SASS (bug-class #10
codegen identity). This gate uses the RIGHT reference.

DECISIVE COMPARAND = the DURABLE GDN STATE (h / ssm_state), NOT the output o.
The native packed-decode kernel writes the recurrent STATE (b_h -> ht) IN PLACE
and ALSO emits an output o = sum(b_h * b_q); the durable carrier is the STATE.
An earlier version of this gate keyed its verdict on the OUTPUT and was VACUOUS
(bug-class #9): the native ref passed ssm_state_indices=0, hitting the kernel's
`if state_idx <= 0: store zeros; return` short-circuit -> the output was all
ZEROS *and* the state was never updated. Fixed: the ref now uses a valid cache
slot (>= 1) so the kernel runs the real rank-1 update and writes the durable
STATE, which this gate extracts and A/Bs.

For each arm (BV/num_warps geometry of the deployed scan) we compare, at
N_PAD={1,16}, on the SPINE and a BRANCH winner, OUR per-node tree-scan STATE vs
the native packed-decode STATE (and, secondarily, the outputs), using:
  * int-view equality  a.view(int32).eq(b.view(int32)).all()  (NEVER atol)
  * raw max_abs
  * rel_err = max_abs / (||native||_inf + eps)
  * norm_ours / norm_native
  * first-mismatch index
A POWERED NEGATIVE CONTROL (bug-class #9) perturbs the root value row that feeds
OUR recurrent STATE update, so OUR STATE MUST int-view-mismatch native's
unperturbed STATE -- an all-match elsewhere is then non-vacuous. The verdict is
keyed on the STATE comparison (negative_control_powered) AND requires both
states to be non-zero (norm-ratio finite) so we never re-key off a zeros tensor.

The serial-torch reference (native_update_serial_per_path) is still recorded as
a SECONDARY column for continuity, but the packed-decode reference is the gate.

Geometry arms are driven by FR13_TREE_GDN_GEOM_OVERRIDE (test-only, value-
neutral; the deployed default is BV=16/num_warps=8 with this env UNSET).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (  # noqa: E402
    Tree,
    launch_tree_gdn_prepared,
)
from fr13_native_packed_decode_ref import (  # noqa: E402
    native_packed_decode_per_path,
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


# ---------------------------------------------------------------------------
# Geometry arms. Each arm sets FR13_TREE_GDN_GEOM_OVERRIDE; the DEPLOYED arm
# leaves it unset so the served byte-identical launch is exercised directly.
# ---------------------------------------------------------------------------
# Each arm is one (geometry, scan-align) configuration of the SAME served
# launch. The DEPLOYED arm leaves both env vars unset (the byte-identical locked
# path). The try-order from the FR13_SCAN_ALIGNMENT_MATH plan:
#   (1) GEOMETRY pre-test  : value-neutral BV32/w1/s3 launch-override on the
#       UNCHANGED scan (geom only; no body seams). If this matches native at
#       N_PAD=1, geometry IS the dominant seam.
#   (2) BODY SEAMS         : FR13_SCAN_ALIGN=1 (l2norm div-by-sqrt + beta bf16).
#   (3) RECOMPUTE-FROM-SPINE: FR13_SCAN_ALIGN=1 + MODE=recompute at native
#       BV32/w1/s3 -- the DEPLOYABLE geometry fix (spill-free, no co-residency).
ARMS = [
    {"name": "BV16_w8_DEPLOYED", "override": None, "scan_align": None},
    # (1) geometry pre-test, native packed-decode launch on the unchanged body.
    {
        "name": "BV32_w1_s3_native_geom",
        "override": "BV=32,num_warps=1,num_stages=3",
        "scan_align": None,
    },
    # Legacy w4 geom arm retained for continuity with the prior gate.
    {
        "name": "BV32_w4_native_geom",
        "override": "BV=32,num_warps=4,num_stages=3",
        "scan_align": None,
    },
    # (2) body seams only, deployed geometry.
    {"name": "BODY_SEAMS_BV16_w8", "override": None, "scan_align": "body"},
    # (2)+(1) body seams at native geom.
    {
        "name": "BODY_SEAMS_BV32_w1_s3",
        "override": "BV=32,num_warps=1,num_stages=3",
        "scan_align": "body",
    },
    # (3) recompute-from-spine: native geom is selected inside the launcher.
    {"name": "RECOMPUTE_FROM_SPINE", "override": None, "scan_align": "recompute"},
]


def _set_geom(override: str | None) -> None:
    if override is None:
        os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)
    else:
        os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = override


def _set_scan_align(mode: str | None) -> None:
    """mode None => SCAN_ALIGN off; 'body'/'recompute' => on with that mode."""
    if mode is None:
        os.environ.pop("FR13_SCAN_ALIGN", None)
        os.environ.pop("FR13_SCAN_ALIGN_MODE", None)
    else:
        os.environ["FR13_SCAN_ALIGN"] = "1"
        os.environ["FR13_SCAN_ALIGN_MODE"] = mode


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _int_view_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Bit-exact equality via integer reinterpretation; NEVER atol.

    Both tensors are cast to the same dtype, then their raw bits are compared.
    This catches a single-ULP difference that any tolerance would hide.
    """
    a = left.contiguous()
    b = right.to(a.dtype).contiguous()
    if a.shape != b.shape:
        return False
    if a.dtype == torch.bfloat16:
        ai = a.view(torch.int16)
        bi = b.view(torch.int16)
    elif a.dtype == torch.float16:
        ai = a.view(torch.int16)
        bi = b.view(torch.int16)
    elif a.dtype == torch.float32:
        ai = a.view(torch.int32)
        bi = b.view(torch.int32)
    elif a.dtype == torch.float64:
        ai = a.view(torch.int64)
        bi = b.view(torch.int64)
    else:
        ai = a
        bi = b
    return bool(torch.equal(ai, bi))


def _rel_err(left: torch.Tensor, right: torch.Tensor) -> float:
    """max_abs / (||native||_inf + eps); native is the RIGHT (reference) arg."""
    max_abs = (left.float() - right.float()).abs().max()
    ref_inf = right.float().abs().max()
    return float((max_abs / (ref_inf + 1e-12)).item())


def _norm_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    no = left.float().norm()
    nn = right.float().norm()
    return float((no / (nn + 1e-12)).item())


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


def _compare(name: str, ours: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    return {
        "ref": name,
        "int_view_equal": _int_view_equal(ours, ref),
        "max_abs": _max_abs(ours, ref),
        "rel_err": _rel_err(ours, ref),
        "norm_ratio": _norm_ratio(ours, ref),
        # Explicit per-side norms so the verdict can require the NATIVE (ref) state
        # to be genuinely non-zero -- norm_ratio alone is vacuous-passable (a zeros
        # ref gives ours.norm()/1e-12 = huge-but-FINITE, which passes 0<r<inf). The
        # old all-zeros-ref bug (state_idx==0 short-circuit) is caught only by ref_norm>0.
        "ref_norm": float(ref.float().norm().item()),
        "ours_norm": float(ours.float().norm().item()),
        "first_mismatch": _first_mismatch(ours, ref),
    }


def _build_inputs(payload: dict[str, Any], rows: torch.Tensor, device: torch.device):
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
    return dict(
        q=q,
        k=k,
        value_spec=value_spec,
        value_tree=value_tree,
        g_tree=g_tree,
        beta_tree=beta_tree,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        h0=h0,
        output_scale=output_scale,
    )


def _run_tree(arm_override, inp, tree, n_actual, n_pad, strict, visible, scan_align=None):
    """Launch the deployed scan under one (geometry, scan-align) arm.

    Returns (out, state). The geometry override and the FR13_SCAN_ALIGN body
    seams / recompute route are set in env around the launch and always cleared
    in `finally`, so the deployed arm (both None) is the byte-identical path.
    """
    _set_geom(arm_override)
    _set_scan_align(scan_align)
    try:
        tree_out, tree_state = launch_tree_gdn_prepared(
            q=inp["q"],
            k=inp["k"],
            v=inp["value_tree"],
            g=inp["g_tree"],
            beta=inp["beta_tree"],
            raw_a=inp["a"],
            raw_b=inp["b"],
            A_log=inp["A_log"],
            dt_bias=inp["dt_bias"],
            h0=inp["h0"],
            n_actual=n_actual,
            n_pad=n_pad,
            strict_mask=strict,
            visible_mask=visible,
            output_scale=inp["output_scale"],
            use_qk_l2norm_in_kernel=True,
        )
    finally:
        _set_geom(None)
        _set_scan_align(None)
    torch.cuda.synchronize()
    state_out = (
        tree_state[:n_actual].contiguous() if tree_state is not None else None
    )
    return tree_out[:n_actual].contiguous(), state_out


def _run_case(
    payload: dict[str, Any],
    *,
    name: str,
    rows_list: list[int],
    n_pad: int,
    parent: list[int] | None,
    negative_control: bool = False,
) -> dict[str, Any]:
    device = torch.device("cuda")
    rows = torch.tensor(rows_list, dtype=torch.long)
    n_actual = int(rows.numel())
    if parent is None:
        parent = [-1] + [idx - 1 for idx in range(1, n_actual)]
    if len(parent) != n_actual:
        raise ValueError(
            f"parent length {len(parent)} does not match n_actual {n_actual}"
        )
    tree = Tree(tuple(parent))
    strict, visible = tree.masks(device, n_pad)
    inp = _build_inputs(payload, rows, device)

    # NEGATIVE CONTROL (bug-class #9): perturb the root value-tree row that
    # feeds OUR scan (value_tree) while leaving the native reference's
    # value_spec UNCHANGED. value_spec and value_tree are byte-identical in the
    # captured payload, so this guarantees a real ours-vs-native divergence in
    # the DURABLE STATE -- v feeds the recurrent update directly
    # (b_v -= sum(state*k); b_v *= beta; state += b_v*k), so a perturbed root v
    # changes state_i and propagates forward to every descendant's state. The
    # neg-control's STATE int-view comparison MUST therefore be False, proving
    # the STATE comparator is LIVE (a clean all-match elsewhere is then
    # non-vacuous). A value-add (+0.5, not a single ULP XOR) makes the control
    # unambiguously powered against the STATE, not just the output.
    if negative_control:
        vt = inp["value_tree"].clone()
        vt[0, 0, 0] = vt[0, 0, 0] + torch.tensor(
            0.5, dtype=vt.dtype, device=vt.device
        )
        inp["value_tree"] = vt.contiguous()

    # Native packed-decode reference (the incumbent decode SASS).
    native_out, native_state = native_packed_decode_per_path(
        tree=tree,
        query_spec=inp["q"],
        key_spec=inp["k"],
        value_spec=inp["value_spec"],
        a=inp["a"],
        b=inp["b"],
        A_log=inp["A_log"],
        dt_bias=inp["dt_bias"],
        h0=inp["h0"],
        scale=inp["output_scale"],
        use_qk_l2norm_in_kernel=True,
    )
    # Secondary serial-torch reference (continuity column only).
    serial_out, serial_state = native_update_serial_per_path(
        tree,
        inp["q"],
        inp["k"],
        inp["value_spec"],
        inp["a"],
        inp["b"],
        inp["A_log"],
        inp["dt_bias"],
        inp["h0"],
        inp["output_scale"],
    )
    torch.cuda.synchronize()

    arms_out: list[dict[str, Any]] = []
    for arm in ARMS:
        tree_out, tree_state = _run_tree(
            arm["override"],
            inp,
            tree,
            n_actual,
            n_pad,
            strict,
            visible,
            scan_align=arm.get("scan_align"),
        )
        state_cmp = (
            _compare("native_packed_decode", tree_state, native_state)
            if tree_state is not None
            else {"ref": "native_packed_decode", "skipped": "no state returned"}
        )
        arms_out.append(
            {
                "arm": arm["name"],
                "override": arm["override"],
                "scan_align": arm.get("scan_align"),
                "out_vs_native_packed": _compare(
                    "native_packed_decode", tree_out, native_out
                ),
                "state_vs_native_packed": state_cmp,
                "out_vs_serial_torch": _compare(
                    "native_update_serial_per_path", tree_out, serial_out
                ),
            }
        )

    return {
        "name": name,
        "n_actual": n_actual,
        "n_pad": n_pad,
        "source_rows": rows_list,
        "parent": parent,
        "negative_control": negative_control,
        "arms": arms_out,
    }


def evaluate(payload_path: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GDN scan A/B gate")
    payload = torch.load(payload_path, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    if int(payload["n_pad"]) < 16 or len(parent) < 10:
        raise RuntimeError(
            "payload must contain the deployed N_PAD=16 MTP-5 tree; "
            f"got n_pad={payload.get('n_pad')} parent_len={len(parent)}"
        )
    tree = Tree(tuple(parent))

    # Branch winners: nodes whose path is NOT a pure spine prefix. Pick the
    # deepest off-spine leaf for the branch case. The reference for those nodes
    # is native-on-path-to-root (handled inside native_packed_decode_per_path).
    # The spine case uses the linear-chain prefix [0,1,2,...].
    n = len(parent)
    spine_rows = list(range(n))

    cases = [
        # SPINE, N_PAD=1: root only.
        _run_case(payload, name="spine_npad1", rows_list=[0], n_pad=1, parent=[-1]),
        # SPINE, N_PAD=16: full captured tree (parent topology drives paths).
        _run_case(
            payload,
            name="spine_npad16",
            rows_list=spine_rows,
            n_pad=16,
            parent=parent,
        ),
        # BRANCH winner [0,2] (off-spine child of root via the captured tree).
        # n_pad must be a warmed power-of-2 >= node count (the masks tensor is
        # [n_pad,n_pad]); 2 nodes -> padded block 4 (smallest no-spill tile).
        _run_case(
            payload, name="branch_0_2", rows_list=[0, 2], n_pad=4, parent=[-1, 0]
        ),
        # BRANCH winner [0,1,4] (deeper off-spine path; 3 nodes -> padded 4).
        _run_case(
            payload,
            name="branch_0_1_4",
            rows_list=[0, 1, 4],
            n_pad=4,
            parent=[-1, 0, 1],
        ),
        # POWERED NEGATIVE CONTROL: deployed geometry, perturbed row -> MUST
        # int-view-mismatch on the native-packed comparison.
        _run_case(
            payload,
            name="neg_control_npad16",
            rows_list=spine_rows,
            n_pad=16,
            parent=parent,
            negative_control=True,
        ),
    ]

    # POWERED on the durable STATE (h), not the output (o). The directive: the
    # gate must compare the DURABLE GDN STATE the kernel writes in place (b_h /
    # ssm_state), and the neg-control must FLIP that STATE comparison. The
    # neg-control perturbs value_tree[0,0,0] (an INPUT to OUR scan's v), which
    # propagates into the recurrent STATE (b_v -= sum(state*k); state += b_v*k),
    # so OUR state MUST diverge from native's unperturbed state -> the STATE
    # int-view comparison MUST be False. A clean all-match elsewhere is then
    # non-vacuous (bug-class #9).
    neg = cases[-1]
    deployed = next(a for a in neg["arms"] if a["arm"] == "BV16_w8_DEPLOYED")
    neg_state = deployed["state_vs_native_packed"]
    # The STATE comparison must be present + non-vacuous (states are non-zero):
    neg_state_int_eq = neg_state.get("int_view_equal", None)
    neg_state_norm_ratio = neg_state.get("norm_ratio", None)
    neg_state_ref_norm = neg_state.get("ref_norm", None)
    neg_state_ours_norm = neg_state.get("ours_norm", None)
    # Powered iff (a) the STATE comparison ran (state returned), (b) it FLIPPED
    # to mismatch (int-view False), (c) norm ratio finite & > 0, AND (d) BOTH the
    # native (ref) state AND our state are EXPLICITLY non-zero. (d) is the bulletproof
    # guard against the old vacuous bug (zeros native ref from the state_idx==0
    # short-circuit) -- norm_ratio alone passes vacuously off a zeros ref (huge-finite).
    neg_powered_state = bool(
        neg_state_int_eq is False
        and neg_state_norm_ratio is not None
        and 0.0 < float(neg_state_norm_ratio) < float("inf")
        and neg_state_ref_norm is not None
        and float(neg_state_ref_norm) > 0.0
        and neg_state_ours_norm is not None
        and float(neg_state_ours_norm) > 0.0
    )

    # Sanity: the NATIVE STATE must be non-zero (the old ref returned the
    # all-zeros output because state_idx==0 short-circuited the kernel). Surface
    # the OFF-arm STATE comparison on the clean spine case as the carrier number.
    spine = next(c for c in cases if c["name"] == "spine_npad16")
    spine_off = next(a for a in spine["arms"] if a["arm"] == "BV16_w8_DEPLOYED")
    spine_recompute = next(
        a for a in spine["arms"] if a["arm"] == "RECOMPUTE_FROM_SPINE"
    )

    return {
        "schema": "fr13.gdn_scan_packed_ab_gate.v2_state",
        "payload": str(payload_path),
        "layer_prefix": payload.get("layer_prefix"),
        "native_ref_kernel": (
            "fused_recurrent_gated_delta_rule_packed_decode "
            "(vllm fla ops; the no-spec B=1 decode oracle dispatches this via "
            "gdn_linear_attn._forward_core -> _forward_core_decode_non_spec)"
        ),
        "deployed_scan": "lumo_flywheel_serving.fr10_gdn_tree_kernel.launch_tree_gdn_prepared",
        "arms": [a["name"] for a in ARMS],
        # The decisive verdict is now keyed on the DURABLE STATE comparison.
        "negative_control_powered": bool(neg_powered_state),
        "negative_control_state_int_view_equal": neg_state_int_eq,
        "negative_control_state_norm_ratio": neg_state_norm_ratio,
        "negative_control_native_state_norm": neg_state_ref_norm,
        "negative_control_our_state_norm": neg_state_ours_norm,
        # The carrier measurement we never got: OFF scan-STATE vs native-STATE.
        "off_arm_spine_state_vs_native": spine_off.get("state_vs_native_packed"),
        "recompute_arm_spine_state_vs_native": spine_recompute.get(
            "state_vs_native_packed"
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    # Gate is informational (it discriminates branches); fail only if the
    # negative control did not power against the DURABLE STATE (vacuous). A
    # powered control = the STATE int-view comparison FLIPPED to mismatch AND
    # both states are non-zero, so the OFF/recompute STATE results are real.
    return 0 if result["negative_control_powered"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

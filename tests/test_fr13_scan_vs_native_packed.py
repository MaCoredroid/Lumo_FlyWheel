"""CPU wiring + AST gates for the FR13 scan-vs-native-packed-decode A/B.

These tests run with NO GPU. They prove:
  1. The TEST-ONLY geometry override (FR13_TREE_GDN_GEOM_OVERRIDE) is flag-gated
     and the served launch is BYTE-IDENTICAL when the env is unset.
  2. The override parser accepts BV/num_warps/num_stages and rejects junk.
  3. The packed-decode reference layout helper packs (q,k,v) correctly.
  4. The extended gate uses INT-VIEW equality (NEVER atol), invokes the
     native-packed-decode reference, and carries a powered negative control.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

KERNEL_SRC = REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
GATE_SRC = REPO / "scripts" / "fr13_gdn_scan_warp_gate.py"
REF_SRC = REPO / "scripts" / "fr13_native_packed_decode_ref.py"


# ---------------------------------------------------------------------------
# 1. Default-off byte-identity of the served launch (AST).
# ---------------------------------------------------------------------------
def _find_launch_call(tree: ast.AST) -> ast.Call:
    """Return the _tree_gdn_kernel[grid](...) call inside launch_tree_gdn_prepared."""
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "launch_tree_gdn_prepared":
            func = node
            break
    assert func is not None, "launch_tree_gdn_prepared not found"
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            sub = node.func
            if isinstance(sub.value, ast.Name) and sub.value.id == "_tree_gdn_kernel":
                return node
    raise AssertionError("_tree_gdn_kernel[grid](...) launch call not found")


def test_override_is_flag_gated_default_off_byte_identical():
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    call = _find_launch_call(tree)
    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}

    # BLOCK_V must be the resolved local _bv (defaults to module BV when unset),
    # NOT a literal -- so the override can move it without touching the served
    # default (which resolves to BV).
    assert "BLOCK_V" in kw
    assert isinstance(kw["BLOCK_V"], ast.Name) and kw["BLOCK_V"].id == "_bv"

    # num_warps must be the resolved local _num_warps (defaults to 8).
    assert "num_warps" in kw
    assert isinstance(kw["num_warps"], ast.Name) and kw["num_warps"].id == "_num_warps"

    # num_stages must NOT be a fixed launch kwarg -- it is only injected through
    # **_extra_launch_kwargs, which is empty when the override is unset, so the
    # served default keeps the Triton-default num_stages (byte-identical).
    assert "num_stages" not in kw, "num_stages must not be a hard launch kwarg"
    star_kwargs = [k for k in call.keywords if k.arg is None]
    assert len(star_kwargs) == 1, "expected exactly one **kwargs spread"
    assert isinstance(star_kwargs[0].value, ast.Name)
    assert star_kwargs[0].value.id == "_extra_launch_kwargs"


# ---------------------------------------------------------------------------
# 2. Override parser. The kernel module imports triton (GPU-only), so we exec
#    ONLY the pure-python helper + constants from source -- no triton import.
# ---------------------------------------------------------------------------
def _load_geom_helper():
    """Exec just _read_tree_gdn_geom_override + constants out of the source."""
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_read_tree_gdn_geom_override":
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("BV", "_DEPLOYED_NUM_WARPS"):
                    wanted.append(node)
    mod = ast.Module(body=wanted, type_ignores=[])
    ns: dict = {"os": os}
    exec(compile(mod, str(KERNEL_SRC), "exec"), ns)  # noqa: S102
    return ns


def test_default_geom_resolves_to_deployed_values():
    """When the env is unset, the resolved geometry == deployed (BV=16, w8)."""
    os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)
    ns = _load_geom_helper()
    assert ns["_read_tree_gdn_geom_override"]() is None
    assert ns["BV"] == 16
    assert ns["_DEPLOYED_NUM_WARPS"] == 8


def test_override_parser_accepts_known_keys():
    os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = "BV=32,num_warps=4,num_stages=3"
    try:
        ns = _load_geom_helper()
        assert ns["_read_tree_gdn_geom_override"]() == {
            "BV": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    finally:
        os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)


def test_override_parser_partial_and_whitespace():
    os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = " BV=8 , num_warps=8 "
    try:
        ns = _load_geom_helper()
        assert ns["_read_tree_gdn_geom_override"]() == {"BV": 8, "num_warps": 8}
    finally:
        os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)


def test_override_parser_rejects_unknown_key():
    os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = "BLOCK=16"
    try:
        ns = _load_geom_helper()
        with pytest.raises(ValueError):
            ns["_read_tree_gdn_geom_override"]()
    finally:
        os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)


def test_override_parser_rejects_malformed_token():
    os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = "BV"
    try:
        ns = _load_geom_helper()
        with pytest.raises(ValueError):
            ns["_read_tree_gdn_geom_override"]()
    finally:
        os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)


# ---------------------------------------------------------------------------
# 3. Packed mixed_qkv layout helper (CPU only).
# ---------------------------------------------------------------------------
def test_build_packed_mixed_qkv_layout():
    from fr13_native_packed_decode_ref import build_packed_mixed_qkv

    b, h, k_dim = 3, 16, 128
    hv, v_dim = 48, 128
    q = torch.arange(b * h * k_dim, dtype=torch.float32).reshape(b, h, k_dim)
    k = torch.arange(b * h * k_dim, dtype=torch.float32).reshape(b, h, k_dim) + 1000
    v = torch.arange(b * hv * v_dim, dtype=torch.float32).reshape(b, hv, v_dim) + 1e6

    packed = build_packed_mixed_qkv(q, k, v)
    assert packed.shape == (b, h * k_dim + h * k_dim + hv * v_dim)
    assert packed.is_contiguous()

    # Verify the native packed-decode offsets (fused_recurrent.py L305-307):
    #   q_off = i_h*K, k_off = H*K + i_h*K, v_off = 2*H*K + i_hv*V.
    row = 1
    for i_h in range(h):
        q_off = i_h * k_dim
        assert torch.equal(packed[row, q_off : q_off + k_dim], q[row, i_h])
        k_off = h * k_dim + i_h * k_dim
        assert torch.equal(packed[row, k_off : k_off + k_dim], k[row, i_h])
    for i_hv in range(hv):
        v_off = 2 * h * k_dim + i_hv * v_dim
        assert torch.equal(packed[row, v_off : v_off + v_dim], v[row, i_hv])


def test_build_packed_mixed_qkv_rejects_bad_ndim():
    from fr13_native_packed_decode_ref import build_packed_mixed_qkv

    with pytest.raises(ValueError):
        build_packed_mixed_qkv(
            torch.zeros(2, 16), torch.zeros(2, 16, 128), torch.zeros(2, 48, 128)
        )


# ---------------------------------------------------------------------------
# 4. Gate AST: int-view (not atol), native-packed reference, neg control.
# ---------------------------------------------------------------------------
def test_gate_uses_int_view_not_atol():
    src = GATE_SRC.read_text(encoding="utf-8")
    # int-view comparator present.
    assert "_int_view_equal" in src
    assert "view(torch.int32)" in src or "view(torch.int16)" in src
    # No atol/rtol/allclose USAGE in the gate comparisons (docstrings may say
    # "NEVER atol"; what is banned is calling them).
    assert "atol=" not in src, "gate must not use atol="
    assert "rtol=" not in src, "gate must not use rtol="
    assert "allclose" not in src, "gate must not use allclose"


def test_gate_invokes_native_packed_reference_and_arms():
    src = GATE_SRC.read_text(encoding="utf-8")
    assert "native_packed_decode_per_path" in src
    # The deployed arm + the geometry pre-test + body-seam + recompute arms.
    for arm in (
        "BV16_w8_DEPLOYED",
        "BV32_w1_s3_native_geom",
        "BODY_SEAMS_BV16_w8",
        "BODY_SEAMS_BV32_w1_s3",
        "RECOMPUTE_FROM_SPINE",
    ):
        assert arm in src, f"missing arm {arm}"
    # rel_err + norm ratio + first mismatch.
    assert "_rel_err" in src and "_norm_ratio" in src and "_first_mismatch" in src
    # Powered negative control.
    assert "negative_control" in src
    assert "negative_control_powered" in src


def test_gate_wires_scan_align_env_and_clears_it():
    """The gate must drive the body seams / recompute via the SCAN_ALIGN env,
    and ALWAYS clear it in finally so the deployed arm stays byte-identical."""
    src = GATE_SRC.read_text(encoding="utf-8")
    assert "_set_scan_align" in src
    assert 'os.environ["FR13_SCAN_ALIGN"] = "1"' in src
    assert 'os.environ["FR13_SCAN_ALIGN_MODE"] = mode' in src
    # cleared in finally (popped to None) alongside the geom override.
    assert 'os.environ.pop("FR13_SCAN_ALIGN", None)' in src
    assert 'os.environ.pop("FR13_SCAN_ALIGN_MODE", None)' in src
    # The finally block clears both.
    gtree = ast.parse(src)
    run_fn = None
    for node in ast.walk(gtree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_tree":
            run_fn = node
            break
    assert run_fn is not None
    finals = [n for n in ast.walk(run_fn) if isinstance(n, ast.Try)]
    assert finals, "_run_tree must use try/finally"
    final_src = "\n".join(
        ast.get_source_segment(src, stmt) or "" for stmt in finals[0].finalbody
    )
    assert "_set_geom(None)" in final_src
    assert "_set_scan_align(None)" in final_src


def test_ref_module_confirms_dispatch_in_docstring():
    src = REF_SRC.read_text(encoding="utf-8")
    # The reference module must name the EXACT live dispatch chain it mirrors.
    assert "_forward_core_decode_non_spec" in src
    assert "fused_recurrent_gated_delta_rule_packed_decode" in src
    assert "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE" in src


def test_ref_uses_valid_state_slot_not_null_block():
    """The native packed-decode ref MUST index a valid cache slot (>= 1), not
    NULL_BLOCK_ID=0. The kernel short-circuits `if state_idx <= 0` (zeros to
    out + state never updated), which made the prior ref vacuous. The fix uses
    slot 1; ssm_state_indices must be 1, NOT zeros."""
    src = REF_SRC.read_text(encoding="utf-8")
    # The vacuous zeros(1) ssm_state_indices must be GONE.
    assert "torch.zeros(1, dtype=torch.int32" not in src, (
        "ref still passes ssm_state_indices=zeros(1) -> state_idx==0 -> kernel "
        "short-circuits (zeros output + un-updated state); use slot >= 1"
    )
    # A valid slot is used and the durable STATE at that slot is extracted.
    assert "state_slot = 1" in src
    assert "torch.full(" in src and "state_slot" in src
    assert "state[state_slot].copy_(h0" in src
    assert "states.append(state[state_slot].clone())" in src
    # The kernel short-circuit is documented so the fix can't silently regress.
    assert "state_idx <= 0" in src or "NULL_BLOCK_ID" in src


def test_gate_verdict_is_keyed_on_durable_state_not_output():
    """The decisive verdict must compare the DURABLE STATE (h), not the zeros-
    prone output. negative_control_powered must read state_vs_native_packed and
    require both states non-zero (norm-ratio finite) so it can never re-vacuum
    off a zeros tensor (bug-class #9)."""
    src = GATE_SRC.read_text(encoding="utf-8")
    gtree = ast.parse(src)
    ev = None
    for node in ast.walk(gtree):
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate":
            ev = node
            break
    assert ev is not None, "evaluate not found"
    ev_src = ast.get_source_segment(src, ev)
    assert ev_src is not None
    # The verdict variable must derive from the STATE comparison.
    assert 'deployed["state_vs_native_packed"]' in ev_src
    assert "neg_powered_state" in ev_src
    # It must require the STATE int-view to FLIP to mismatch (False).
    assert "neg_state_int_eq is False" in ev_src
    # And require both states non-zero via the norm ratio (no zeros tensor).
    assert "neg_state_norm_ratio" in ev_src
    assert 'result["negative_control_powered"]' or "negative_control_powered" in ev_src
    # The schema bumped to the state-keyed version.
    assert "fr13.gdn_scan_packed_ab_gate.v2_state" in src
    # The OFF and recompute spine STATE-vs-native carriers are surfaced.
    assert "off_arm_spine_state_vs_native" in src
    assert "recompute_arm_spine_state_vs_native" in src


# ---------------------------------------------------------------------------
# 5. FR13_SCAN_ALIGN flag helpers (CPU exec of the pure-python helpers).
# ---------------------------------------------------------------------------
def _load_scan_align_helpers():
    """Exec scan_align_on / scan_align_mode + native-geom constants from source,
    without importing triton (GPU-only)."""
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "scan_align_on",
            "scan_align_mode",
        ):
            wanted.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in (
                    "_NATIVE_PACKED_BV",
                    "_NATIVE_PACKED_NUM_WARPS",
                    "_NATIVE_PACKED_NUM_STAGES",
                ):
                    wanted.append(node)
    mod = ast.Module(body=wanted, type_ignores=[])
    ns: dict = {"os": os}
    exec(compile(mod, str(KERNEL_SRC), "exec"), ns)  # noqa: S102
    return ns


def test_scan_align_default_off():
    os.environ.pop("FR13_SCAN_ALIGN", None)
    os.environ.pop("FR13_SCAN_ALIGN_MODE", None)
    ns = _load_scan_align_helpers()
    assert ns["scan_align_on"]() is False
    assert ns["scan_align_mode"]() == "body"


def test_scan_align_on_truthy_values():
    ns = _load_scan_align_helpers()
    try:
        for v in ("1", "true", "TRUE", "Yes", "on", " on "):
            os.environ["FR13_SCAN_ALIGN"] = v
            assert ns["scan_align_on"]() is True, v
        for v in ("0", "false", "no", "off", "", "  "):
            os.environ["FR13_SCAN_ALIGN"] = v
            assert ns["scan_align_on"]() is False, repr(v)
    finally:
        os.environ.pop("FR13_SCAN_ALIGN", None)


def test_scan_align_mode_values_and_reject():
    ns = _load_scan_align_helpers()
    try:
        os.environ["FR13_SCAN_ALIGN_MODE"] = "recompute"
        assert ns["scan_align_mode"]() == "recompute"
        os.environ["FR13_SCAN_ALIGN_MODE"] = "body"
        assert ns["scan_align_mode"]() == "body"
        os.environ["FR13_SCAN_ALIGN_MODE"] = "bogus"
        with pytest.raises(ValueError):
            ns["scan_align_mode"]()
    finally:
        os.environ.pop("FR13_SCAN_ALIGN_MODE", None)


def test_native_packed_geom_constants_match_image():
    """The recompute route's geometry must equal the native packed-decode
    launch re-grounded on the pinned image (fused_recurrent.py:437-439):
    BV=min(next_pow2(V),32)=32, num_warps=1, num_stages=3."""
    ns = _load_scan_align_helpers()
    assert ns["_NATIVE_PACKED_BV"] == 32
    assert ns["_NATIVE_PACKED_NUM_WARPS"] == 1
    assert ns["_NATIVE_PACKED_NUM_STAGES"] == 3


# ---------------------------------------------------------------------------
# 6. _gdn_node_step: both body seams are SCAN_ALIGN-gated; the OFF branch keeps
#    the original ops (rsqrt l2norm + plain fp32 sigmoid). Bug-class #10:
#    default-OFF must be byte-identical, so the aligned ops must live behind an
#    `if SCAN_ALIGN:` and the off branch must be the unchanged source.
# ---------------------------------------------------------------------------
def _func_def(name: str) -> ast.FunctionDef:
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_gdn_node_step_has_scan_align_constexpr_default_false():
    fn = _func_def("_gdn_node_step")
    # SCAN_ALIGN is the last arg with default False.
    arg_names = [a.arg for a in fn.args.args]
    assert "SCAN_ALIGN" in arg_names
    # default False (defaults align to the trailing args).
    n_defaults = len(fn.args.defaults)
    assert n_defaults >= 1
    last_default = fn.args.defaults[-1]
    assert isinstance(last_default, ast.Constant) and last_default.value is False


def test_gdn_node_step_seams_are_gated_and_off_branch_is_original():
    """The l2norm and beta seams must be `if SCAN_ALIGN: <native>; else:
    <original>`. The else-branch must contain rsqrt (l2norm) and a plain
    sigmoid (beta) with NO bf16 round-trip, so OFF == locked path."""
    src = KERNEL_SRC.read_text(encoding="utf-8")
    fn = _func_def("_gdn_node_step")
    fn_src = ast.get_source_segment(src, fn)
    assert fn_src is not None
    # Aligned (native) forms appear (inside the SCAN_ALIGN branches):
    assert "b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)" in fn_src
    assert "b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)" in fn_src
    assert "tl.sigmoid(b_raw_b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)" in fn_src
    # Original (OFF) forms still present (the else branches):
    assert "b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)" in fn_src
    assert "b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)" in fn_src
    # plain fp32 sigmoid (no bf16 cast) must remain as the OFF beta.
    assert "b_beta = tl.sigmoid(b_raw_b.to(tl.float32))\n" in (fn_src + "\n")
    # Both seams are guarded by `if SCAN_ALIGN`.
    n_if_scan_align = sum(
        1
        for node in ast.walk(fn)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "SCAN_ALIGN"
    )
    assert n_if_scan_align == 2, f"expected 2 SCAN_ALIGN ifs, got {n_if_scan_align}"


def test_all_gdn_node_step_calls_pass_scan_align():
    """Every in-kernel _gdn_node_step call must forward SCAN_ALIGN, or the
    aligned arm cannot reach the body (and the OFF arm cannot stay byte-id)."""
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_gdn_node_step"
    ]
    assert len(calls) == 4, f"expected 4 _gdn_node_step calls, got {len(calls)}"
    for call in calls:
        kw = {k.arg for k in call.keywords if k.arg is not None}
        assert "SCAN_ALIGN" in kw, "a _gdn_node_step call drops SCAN_ALIGN"


# ---------------------------------------------------------------------------
# 7. Recompute-from-spine: exists, is no-h_cache (one tile), reuses the shared
#    body, and launches at native BV32/w1/s3 only when SCAN_ALIGN + mode.
# ---------------------------------------------------------------------------
def test_recompute_kernel_is_no_hcache_one_tile():
    fn = _func_def("_tree_gdn_recompute_kernel")
    fn_src = ast.get_source_segment(KERNEL_SRC.read_text(encoding="utf-8"), fn)
    assert fn_src is not None
    # The h_cache (N_PAD, BLOCK_V, DIM_K) tile of the scan must NOT be allocated
    # here -- no `tl.zeros((N_PAD, ...))` 3-D cache. The replay/recompute hold a
    # single (BLOCK_V, DIM_K) tile that re-replays ancestry. We assert via the
    # AST: no assignment whose value is a tl.zeros with an (N_PAD, ...) shape and
    # no name `h_cache` is bound.
    bound_names = {
        t.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "h_cache" not in bound_names, "recompute must not bind an h_cache tile"
    # No tl.zeros((N_PAD, ...)) 3-D cache allocation.
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "zeros"
            and node.args
            and isinstance(node.args[0], ast.Tuple)
        ):
            elts = node.args[0].elts
            first = elts[0]
            assert not (
                isinstance(first, ast.Name) and first.id == "N_PAD"
            ), "recompute allocated an (N_PAD, ...) cache tile (spill risk)"
    # It must reuse the shared per-node body.
    assert "_gdn_node_step(" in fn_src
    # It produces per-node out + (optional) state, like the scan.
    assert "tl.store(" in fn_src
    # SCAN_ALIGN constexpr threaded.
    assert "SCAN_ALIGN" in [a.arg for a in fn.args.args]


def test_recompute_dispatch_gated_and_native_geom():
    """launch_tree_gdn_prepared dispatches the recompute kernel only when
    scan_align_on() and mode=='recompute', at native w1/s3."""
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "launch_tree_gdn_prepared":
            fn = node
            break
    assert fn is not None
    fn_src = ast.get_source_segment(src, fn)
    assert fn_src is not None
    # Gated dispatch condition.
    assert 'scan_align_mode() == "recompute"' in fn_src
    assert "_scan_align and scan_align_mode() == \"recompute\"" in fn_src
    # The recompute launch uses the native packed geom constants.
    assert "_NATIVE_PACKED_NUM_WARPS" in fn_src
    assert "_NATIVE_PACKED_NUM_STAGES" in fn_src
    assert "_NATIVE_PACKED_BV" in fn_src
    # Find the _tree_gdn_recompute_kernel[...] launch and check its geom kwargs.
    rcall = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_tree_gdn_recompute_kernel"
        ):
            rcall = node
            break
    assert rcall is not None, "recompute kernel launch not found"
    kw = {k.arg: k.value for k in rcall.keywords if k.arg is not None}
    assert isinstance(kw["num_warps"], ast.Name) and kw["num_warps"].id == "_NATIVE_PACKED_NUM_WARPS"
    assert isinstance(kw["num_stages"], ast.Name) and kw["num_stages"].id == "_NATIVE_PACKED_NUM_STAGES"
    assert "SCAN_ALIGN" in kw


def test_served_scan_launch_passes_scan_align_and_default_off_is_identity():
    """The served _tree_gdn_kernel launch must pass SCAN_ALIGN=_scan_align, and
    _scan_align must resolve from scan_align_on() (default False => byte-id)."""
    src = KERNEL_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    call = _find_launch_call(tree)
    kw = {k.arg: k.value for k in call.keywords if k.arg is not None}
    assert "SCAN_ALIGN" in kw
    assert isinstance(kw["SCAN_ALIGN"], ast.Name) and kw["SCAN_ALIGN"].id == "_scan_align"
    # _scan_align is assigned from scan_align_on() in the function body.
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "launch_tree_gdn_prepared":
            fn = node
            break
    assert fn is not None
    found = False
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_scan_align" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "scan_align_on"
        ):
            found = True
    assert found, "_scan_align must be assigned from scan_align_on()"

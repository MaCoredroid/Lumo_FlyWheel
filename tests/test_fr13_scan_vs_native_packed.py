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
    # The four geometry arms.
    for arm in ("BV16_w8_DEPLOYED", "BV32_w4_native_geom", "BV8_w8", "BV8_w4"):
        assert arm in src, f"missing arm {arm}"
    # rel_err + norm ratio + first mismatch.
    assert "_rel_err" in src and "_norm_ratio" in src and "_first_mismatch" in src
    # Powered negative control.
    assert "negative_control" in src
    assert "negative_control_powered" in src


def test_ref_module_confirms_dispatch_in_docstring():
    src = REF_SRC.read_text(encoding="utf-8")
    # The reference module must name the EXACT live dispatch chain it mirrors.
    assert "_forward_core_decode_non_spec" in src
    assert "fused_recurrent_gated_delta_rule_packed_decode" in src
    assert "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE" in src

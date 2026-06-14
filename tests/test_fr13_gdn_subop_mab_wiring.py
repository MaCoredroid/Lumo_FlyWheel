"""FR13_GDN_SUBOP_MAB wiring + default-OFF byte-identity gate (CPU).

The L0-GDN sub-op M=10/M=5/M=1 A/B (FR13_GDN_SUBOP_MAB, default OFF) is an
OBSERVE-ONLY in-process hook: at a captured tree-verify forward it re-runs the
GDN sub-op sequence (conv1d_update -> fused_sigmoid_gating scan) on the
deep-spine row at three row-counts on the SAME captured input + h0 + conv
prior-window, to localize WHERE co-residency (branch rows) first perturbs the
committed-spine row.

This CPU test pins:
  1. The patcher (scripts/fr10_phase4_patch_vllm_tree_gdn.py) emits the MAB
     helper + call-site + conv-state-snapshot stash, ALL behind the
     FR13_GDN_SUBOP_MAB flag (class-9 engagement asserts present).
  2. The patch is PURELY ADDITIVE (no needle text deleted): the default-OFF
     path is byte-identical to the locked cat9 build.
  3. The emitted helper compiles, returns immediately when the flag != "1"
     (default-OFF no-op), and -- when ON -- runs three arms, finds pre_conv
     M-invariant, and writes exactly one JSONL record per event with the v2
     schema + the M1 receptive-field caveat.

The live B=1 same-seed repeat on the GPU host remains the first binding gate
(class 8: offline proofs are necessary, not sufficient).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
SRC = PATCHER.read_text()


# --------------------------------------------------------------------------- #
# 1. The patcher emits the MAB behind the default-OFF flag.
# --------------------------------------------------------------------------- #
def test_flag_guard_present_default_off() -> None:
    # Module-scope default OFF read via env (flag plan).
    assert 'os.environ.get("FR13_GDN_SUBOP_MAB", "0") != "1"' in SRC
    assert 'os.environ.get("FR13_GDN_SUBOP_MAB", "0") == "1"' in SRC
    # The call site is inside an explicit flag guard.
    assert re.search(
        r'if os\.environ\.get\("FR13_GDN_SUBOP_MAB", "0"\) == "1":\n'
        r"\s+# FR13_GDN_SUBOP_MAB.*?_fr13_gdn_subop_mab\(\s*\n\s+self,",
        SRC,
        re.DOTALL,
    )
    # The conv-state snapshot + pre_conv stash is gated by the same flag.
    assert "self._fr13_gdn_subop_mab_pre_conv = _fr12_pre_conv_spec" in SRC
    assert "self._fr13_gdn_subop_mab_conv_state = (" in SRC


def test_engagement_asserts_present_class9() -> None:
    # Fail loud on tree disengagement / silent shape change (class 9).
    assert "tree DISENGAGED (no path0)" in SRC
    assert "tree DISENGAGED (no parent)" in SRC
    assert "silent tree-shape change; class 9" in SRC
    assert "conv_state snapshot missing (stash disengaged)" in SRC
    # No reward-hacks: observe-only, the helper never mutates the live forward.
    assert "splice" in SRC.lower()  # documented ban in the helper docstring
    assert "copy-recurrent" in SRC
    assert "dense-route" in SRC


def test_patch_is_purely_additive_no_needle_deleted() -> None:
    # The two needles whose surrounding strings the MAB extends must still be
    # present verbatim (we only INSERTED lines around them -> default path byte-
    # identical). conv pre-clone condition + scan needle marker.
    assert (
        '_fr12_subkernel_capture_enabled = bool(os.environ.get("FR12_SUBKERNEL_CAPTURE"))'
        in SRC
    )
    assert 'or os.environ.get("FR12_TREE_CONV_NATIVE_PRIOR_READ", "0") == "1"' in SRC
    # The fused_post_conv_prep call (right after our scan-site call) is intact.
    assert "_, _, value_tree, g_tree, beta_tree = fused_post_conv_prep(" in SRC


# --------------------------------------------------------------------------- #
# 2. Extract + compile + exercise the emitted helper on CPU with stub kernels.
# --------------------------------------------------------------------------- #
def _extract_helper() -> str:
    m = re.search(r"mab_helper = '''(.*?)'''", SRC, re.DOTALL)
    assert m, "mab_helper triple-quoted block not found in patcher"
    # The patcher writes the literal body into gdn_linear_attn.py; the only
    # escape in the body is the JSONL newline written as a python string escape.
    return m.group(1)


def _make_module(tmp_path: Path):
    """Exec the emitted helper in a namespace with stub kernels + logger."""
    body = _extract_helper()
    records: list[dict] = []

    class _Logger:
        def warning(self, *a, **k):  # pragma: no cover - trivial
            pass

    def stub_conv(x, cs, w, bias, act, **k):
        # Depthwise-ish stub: output is row-independent of M for a fixed row's
        # content (here just a deterministic transform of x), so the pre_conv
        # control + a row-local conv would be M-invariant; we make it depend
        # ONLY on the row's own content to model a row-parallel op.
        return x.clone() * 2.0 + 0.5

    def stub_scan(*, A_log, a, b, dt_bias, q, k, v, initial_state, **k2):
        # q is [1, m, ...]; return [1, m, vh, vd]-ish keyed to the row's own q so
        # each row maps to itself (M-invariant per-row), plus None final state.
        out = (q * 3.0)[..., : v.shape[-1]] if q.shape[-1] >= v.shape[-1] else q * 3.0
        return out, None

    ns = {
        "os": os,
        "json": json,
        "torch": torch,
        "logger": _Logger(),
        "causal_conv1d_update": stub_conv,
        "fused_sigmoid_gating_delta_rule_update": stub_scan,
    }
    compile(body, "<emitted_mab_helper>", "exec")
    exec(body, ns)
    return ns, records


class _FakeAttnMeta:
    def __init__(self, tree_n, spine_rows, parent):
        self.num_spec_decodes = 1
        self.fr10_tree_path0_nodes = torch.tensor(spine_rows, dtype=torch.long)
        self.fr10_tree_parent = torch.tensor(parent, dtype=torch.int32)


class _FakeLayer:
    prefix = "language_model.model.layers.0.linear_attn"

    class _Conv1d:
        bias = None

    conv1d = _Conv1d()
    activation = "silu"
    A_log = torch.zeros(16)
    dt_bias = torch.zeros(16)


def _cat9_geometry():
    # parent/path0 for the locked cat9 tree.
    parent = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]
    spine = [0, 1, 2, 4, 6, 8]  # path0_nodes (deep node = 8)
    return parent, spine


def test_default_off_helper_is_noop(tmp_path, monkeypatch) -> None:
    ns, _ = _make_module(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.delenv("FR13_GDN_SUBOP_MAB", raising=False)  # default OFF
    dump = tmp_path / "noop.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    # Flag OFF: the helper must return immediately and write NOTHING.
    fn(
        _FakeLayer(),
        torch.randn(10, 8),  # pre_conv
        torch.randn(1, 10, 8),  # mixed_qkv_spec (unused on OFF)
        torch.randn(10, 8),  # a
        torch.randn(10, 8),  # b
        torch.randn(1, 10, 8),  # query_spec
        torch.randn(1, 10, 8),  # key_spec
        torch.randn(10, 8),  # value_spec
        torch.randn(20, 8, 4),  # conv_state_snapshot
        torch.randn(8, 4),  # conv_weights
        torch.randn(20, 8, 8),  # ssm_state
        torch.zeros(1, 16, dtype=torch.long),  # spec_state_indices_tensor
        torch.tensor([5], dtype=torch.long),  # num_accepted_tokens
        torch.tensor([0, 10], dtype=torch.long),  # spec_query_start_loc
        _FakeAttnMeta(10, _cat9_geometry()[1], _cat9_geometry()[0]),  # attn_metadata
        10,  # num_actual_tokens
    )
    assert not dump.exists(), "default-OFF helper wrote an artifact (must be no-op)"


def test_helper_on_writes_one_record_pre_conv_m_invariant(tmp_path, monkeypatch) -> None:
    ns, _ = _make_module(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    parent, spine = _cat9_geometry()
    tree_n = 10
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "on.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")

    H = 8
    pre_conv = torch.randn(tree_n, H)
    fn(
        _FakeLayer(),
        pre_conv,
        torch.randn(tree_n, H),  # mixed_qkv_spec: live shape [num_spec*tree_n, dim]
        torch.randn(tree_n, H),  # a
        torch.randn(tree_n, H),  # b
        torch.randn(1, tree_n, H),  # query_spec
        torch.randn(1, tree_n, H),  # key_spec
        torch.randn(tree_n, H),  # value_spec
        torch.randn(20, H, 4),  # conv_state_snapshot
        torch.randn(H, 4),  # conv_weights
        torch.randn(20, H, H),  # ssm_state
        torch.zeros(1, 16, dtype=torch.long),  # spec_state_indices_tensor
        torch.tensor([5], dtype=torch.long),  # num_accepted_tokens
        torch.tensor([0, tree_n], dtype=torch.long),  # spec_query_start_loc
        _FakeAttnMeta(
            tree_n,
            spine,  # fr10_tree_path0_nodes
            parent,
        ),
        tree_n,
    )
    assert dump.exists(), "flag ON wrote no record"
    recs = [json.loads(l) for l in dump.read_text().splitlines() if l.strip()]
    assert len(recs) == 1, f"expected exactly 1 record, got {len(recs)}"
    rec = recs[0]
    assert rec["schema"] == "fr13.gdn_subop_mab.v2"
    assert rec["tree_n"] == tree_n
    assert rec["spine_rows"] == spine
    assert rec["deep_row"] == spine[-1]
    assert rec["m5_deep_index"] == spine.index(spine[-1])
    # pre_conv is a pure row-gather -> M-invariant by construction (bit-exact).
    pc = rec["subops"]["pre_conv"]
    assert pc["m10_deep_vs_m5_deep_max_abs"] == 0.0
    assert pc["m10_deep_vs_m1_max_abs"] == 0.0
    assert rec["pre_conv_m_invariant"] is True
    # All three sub-ops present with the full pairwise triplet.
    for name in ("pre_conv", "conv1d_out", "scan_out"):
        s = rec["subops"][name]
        for key in (
            "m10_deep_vs_m5_deep_max_abs",
            "m5_deep_vs_m1_max_abs",
            "m10_deep_vs_m1_max_abs",
        ):
            assert key in s, f"{name} missing {key}"
    assert "m1_receptive_field_caveat" in rec
    assert "first_coresidency_subop_m10_vs_m5" in rec
    assert "first_subop_m5_vs_m1" in rec
    # Second axis: the served-fused-vs-native realization seam on conv1d_out.
    # With the live-shaped 2D mixed_qkv_spec this must be a real number.
    seam = rec["subops"]["conv1d_out"]["served_fused_vs_native_decode_max_abs"]
    assert seam is not None and isinstance(seam, float)


def test_helper_on_disengaged_tree_fails_loud(tmp_path, monkeypatch) -> None:
    """No path0 on attn_metadata => the A/B must NOT silently record a number."""
    ns, _ = _make_module(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "diseng.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")

    class _NoTree:
        num_spec_decodes = 1
        fr10_tree_path0_nodes = None
        fr10_tree_parent = None

    # The helper swallows the RuntimeError internally (diagnostic-only) and logs,
    # but must NOT write a record on a disengaged tree.
    fn(
        _FakeLayer(),
        torch.randn(10, 8),
        torch.randn(1, 10, 8),
        torch.randn(10, 8),
        torch.randn(10, 8),
        torch.randn(1, 10, 8),
        torch.randn(1, 10, 8),
        torch.randn(10, 8),
        torch.randn(20, 8, 4),
        torch.randn(8, 4),
        torch.randn(20, 8, 8),
        torch.zeros(1, 16, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0, 10], dtype=torch.long),
        _NoTree(),
        10,
    )
    assert not dump.exists(), "disengaged tree wrote a vacuous record (class 9)"


def test_missing_conv_state_snapshot_no_record(tmp_path, monkeypatch) -> None:
    ns, _ = _make_module(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    parent, spine = _cat9_geometry()
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "nosnap.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    fn(
        _FakeLayer(),
        torch.randn(10, 8),
        torch.randn(1, 10, 8),
        torch.randn(10, 8),
        torch.randn(10, 8),
        torch.randn(1, 10, 8),
        torch.randn(1, 10, 8),
        torch.randn(10, 8),
        None,  # conv_state_snapshot MISSING -> must bail, no record
        torch.randn(8, 4),
        torch.randn(20, 8, 8),
        torch.zeros(1, 16, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0, 10], dtype=torch.long),
        _FakeAttnMeta(10, spine, parent),
        10,
    )
    assert not dump.exists()


# --------------------------------------------------------------------------- #
# 3. CUDA-SAFETY bounds-guards: a device-side assert is a GPU-runtime fault CPU
#    tests cannot catch, so the helper must raise a CLEAN Python error BEFORE
#    any native-kernel launch on a bad bank / row / index.  These tests use
#    kernel stubs that RECORD whether they were called and assert the guard
#    fired FIRST (kernel never reached -> no OOB global read -> no CUDA assert).
# --------------------------------------------------------------------------- #
def _make_module_kernel_tracking(tmp_path: Path):
    """Like _make_module but the stub kernels set a 'called' flag so a test can
    assert the bounds-guard fired BEFORE the native kernel was launched."""
    body = _extract_helper()
    calls = {"conv": 0, "scan": 0}

    class _Logger:
        def warning(self, *a, **k):  # pragma: no cover - trivial
            pass

    def stub_conv(x, cs, w, bias, act, **k):
        calls["conv"] += 1
        return x.clone() * 2.0 + 0.5

    def stub_scan(*, A_log, a, b, dt_bias, q, k, v, initial_state, **k2):
        calls["scan"] += 1
        out = (q * 3.0)[..., : v.shape[-1]] if q.shape[-1] >= v.shape[-1] else q * 3.0
        return out, None

    ns = {
        "os": os,
        "json": json,
        "torch": torch,
        "logger": _Logger(),
        "causal_conv1d_update": stub_conv,
        "fused_sigmoid_gating_delta_rule_update": stub_scan,
    }
    compile(body, "<emitted_mab_helper>", "exec")
    exec(body, ns)
    return ns, calls


def _call(fn, *, ssi, ssm, conv_snap, max_path_len=16, tree_n=10, nacc=5):
    parent, spine = _cat9_geometry()
    H = 8
    return fn(
        _FakeLayer(),
        torch.randn(tree_n, H),  # pre_conv
        torch.randn(tree_n, H),  # mixed_qkv_spec
        torch.randn(tree_n, H),  # a
        torch.randn(tree_n, H),  # b
        torch.randn(1, tree_n, H),  # query_spec
        torch.randn(1, tree_n, H),  # key_spec
        torch.randn(tree_n, H),  # value_spec
        conv_snap,
        torch.randn(H, 4),  # conv_weights (width 4)
        ssm,
        ssi,
        torch.tensor([nacc], dtype=torch.long),
        torch.tensor([0, tree_n], dtype=torch.long),
        _FakeAttnMeta(tree_n, spine, parent),
        tree_n,
    )


def test_oob_prior_bank_fires_guard_no_kernel(tmp_path, monkeypatch) -> None:
    """ssi0[0,0] points at a bank >= n_ssm_banks: the prior-bank guard must
    raise BEFORE any kernel launch (would be the OOB h0 deref -> CUDA assert)."""
    ns, calls = _make_module_kernel_tracking(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "oobbank.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    ssi = torch.zeros(1, 16, dtype=torch.long)
    ssi[0, 0] = 999  # committed prior bank way out of range
    _call(
        fn,
        ssi=ssi,
        ssm=torch.randn(20, 8, 8),  # only 20 banks
        conv_snap=torch.randn(20, 8, 4),
    )
    assert not dump.exists(), "OOB prior bank wrote a record (guard did not fire)"
    assert calls["conv"] == 0 and calls["scan"] == 0, (
        "a native kernel was launched on an OOB bank -> would be a CUDA assert"
    )


def test_row_ge_max_path_len_fires_guard_no_kernel(tmp_path, monkeypatch) -> None:
    """A served tree wider than the index table's column count: index_select(1,.)
    would be a CUDA assert.  _guard_rows must raise first (no kernel)."""
    ns, calls = _make_module_kernel_tracking(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "oobrow.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    # max_path_len = 4 but tree_n = 10 -> full_rows reaches column 9 >= 4.
    ssi = torch.zeros(1, 4, dtype=torch.long)
    _call(
        fn,
        ssi=ssi,
        ssm=torch.randn(20, 8, 8),
        conv_snap=torch.randn(20, 8, 4),
        max_path_len=4,
    )
    assert not dump.exists(), "row>=max_path_len wrote a record (guard did not fire)"
    assert calls["conv"] == 0 and calls["scan"] == 0, (
        "index_select(1,.) on an OOB column was reached -> would be a CUDA assert"
    )


def test_served_table_branch_bank_oob_fires_guard(tmp_path, monkeypatch) -> None:
    """M10 served arm reuses the per-node bank table; if a node bank is OOB the
    final bank-range guard must catch it (the kernel would OOB-read h0).  -1
    (PAD_SLOT_ID) is allowed by the guard (kernel skips it)."""
    ns, calls = _make_module_kernel_tracking(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "branchbank.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    ssi = torch.zeros(1, 16, dtype=torch.long)  # column 0 valid (bank 0)
    ssi[0, 7] = 12345  # a branch/leaf node column holds an OOB bank
    _call(
        fn,
        ssi=ssi,
        ssm=torch.randn(20, 8, 8),  # 20 banks; 12345 is OOB
        conv_snap=torch.randn(20, 8, 4),
    )
    assert not dump.exists(), "served OOB node bank wrote a record (guard missed it)"
    # The conv arm runs before the scan arm; the guard that catches the served
    # branch bank is inside _scan_arm (M10). The conv arm uses only ssi0[:,0]
    # (valid) so it legitimately runs; the scan-table guard then fires and the
    # event is aborted with no record.  Assert the SCAN kernel was never reached.
    assert calls["scan"] == 0, (
        "scan kernel launched on an OOB served node bank -> would be a CUDA assert"
    )


def test_pad_slot_minus_one_allowed_in_scan_table(tmp_path, monkeypatch) -> None:
    """PAD_SLOT_ID=-1 in the served bank table must NOT trip the guard (the
    kernel checks state_idx<0 and skips); a record is still written."""
    ns, calls = _make_module_kernel_tracking(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "padslot.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    ssi = torch.zeros(1, 16, dtype=torch.long)
    ssi[0, 9] = -1  # a non-spine node padded out (allowed)
    _call(
        fn,
        ssi=ssi,
        ssm=torch.randn(20, 8, 8),
        conv_snap=torch.randn(20, 8, 4),
    )
    assert dump.exists(), "PAD_SLOT_ID=-1 wrongly tripped the bank guard"
    recs = [json.loads(l) for l in dump.read_text().splitlines() if l.strip()]
    assert len(recs) == 1
    assert calls["scan"] >= 1, "scan arms did not run on a valid (PAD-allowed) table"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

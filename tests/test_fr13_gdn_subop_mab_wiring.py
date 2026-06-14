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
    # The three worker-side gate reads resolve via the worker-env-safe helper,
    # NOT a bare os.environ.get (the bare master switch is curated out of the
    # mp/spawn EngineCore worker -- measured 2026-06-14, class-9 vacuous boot).
    assert "def _fr13_gdn_subop_mab_enabled():" in SRC
    assert "if not _fr13_gdn_subop_mab_enabled():" in SRC  # helper top-guard
    assert "_fr13_gdn_subop_mab_on = _fr13_gdn_subop_mab_enabled()" in SRC
    # The call site is inside the resolver guard.
    assert re.search(
        r"if _fr13_gdn_subop_mab_enabled\(\):\n"
        r"\s+# FR13_GDN_SUBOP_MAB.*?_fr13_gdn_subop_mab\(\s*\n\s+self,",
        SRC,
        re.DOTALL,
    )
    # No bare worker-time master env read survives at the GDN gate sites; the
    # only direct master env read is the patch-time sidecar writer (pid 1).
    assert SRC.count('os.environ.get("FR13_GDN_SUBOP_MAB", "0")') == 1
    # The patch-time sidecar bridge (pid 1 -> worker via /logs bind mount).
    assert "_fr13_write_subop_mab_sidecar" in SRC
    assert "/logs/fr13_gdn_subop_mab.flag" in SRC
    # The conv-state snapshot + pre_conv stash is gated by the same resolver.
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
    # Hermetic: point the sidecar bridge at a guaranteed-absent path so the
    # resolver cannot pick up a stray /logs flag from a co-located run.
    monkeypatch.setenv(
        "FR13_GDN_SUBOP_MAB_FLAG_FILE", str(tmp_path / "absent.flag")
    )
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


# --------------------------------------------------------------------------- #
# 4. Worker-env bridge: the resolver reads env OR the patch-time sidecar so the
#    master switch reaches the curated mp/spawn EngineCore worker (class-9 fix).
# --------------------------------------------------------------------------- #
def _make_resolver(tmp_path: Path):
    """Exec just the emitted helper body and return the resolver fn + namespace.

    Each call gets a fresh namespace so the module-global cache resets.
    """
    body = _extract_helper()
    ns = {"os": os, "json": json, "torch": torch, "logger": object()}
    exec(body, ns)
    return ns["_fr13_gdn_subop_mab_enabled"], ns


def test_resolver_env_on_short_circuits(tmp_path, monkeypatch) -> None:
    fn, _ = _make_resolver(tmp_path)
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    monkeypatch.setenv(
        "FR13_GDN_SUBOP_MAB_FLAG_FILE", str(tmp_path / "nope.flag")
    )
    assert fn() is True


def test_resolver_sidecar_bridge_when_env_absent(tmp_path, monkeypatch) -> None:
    # The mp/spawn worker case: master env is ABSENT, the patch-time sidecar
    # (written in pid 1) carries the flag through the /logs bind mount.
    fn, _ = _make_resolver(tmp_path)
    monkeypatch.delenv("FR13_GDN_SUBOP_MAB", raising=False)
    flag = tmp_path / "fr13_gdn_subop_mab.flag"
    flag.write_text("1\n")
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_FLAG_FILE", str(flag))
    assert fn() is True


def test_resolver_env_off_overrides_sidecar(tmp_path, monkeypatch) -> None:
    # An explicit OFF env must win over a stale sidecar (no leak ON).
    fn, _ = _make_resolver(tmp_path)
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "0")
    flag = tmp_path / "fr13_gdn_subop_mab.flag"
    flag.write_text("1\n")
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_FLAG_FILE", str(flag))
    assert fn() is False


def test_resolver_default_off_no_env_no_sidecar(tmp_path, monkeypatch) -> None:
    fn, _ = _make_resolver(tmp_path)
    monkeypatch.delenv("FR13_GDN_SUBOP_MAB", raising=False)
    monkeypatch.setenv(
        "FR13_GDN_SUBOP_MAB_FLAG_FILE", str(tmp_path / "absent.flag")
    )
    assert fn() is False


def test_patcher_sidecar_writer_on_off(tmp_path, monkeypatch) -> None:
    # Exec the patch-time sidecar writer (pid-1 side) in isolation: ON writes
    # "1", OFF deletes any stale flag (default-safe, no leak).
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fr13_patcher_mod", str(PATCHER))
    assert spec and spec.loader
    flag = tmp_path / "fr13_gdn_subop_mab.flag"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_FLAG_FILE", str(flag))
    # Import the patcher WITHOUT running main(); grab the writer.
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    writer = mod._fr13_write_subop_mab_sidecar
    # OFF: no flag file written.
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "0")
    writer()
    assert not flag.exists()
    # ON: flag file written with "1".
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    writer()
    assert flag.exists() and flag.read_text().strip() == "1"
    # Back to OFF: stale flag is removed (no leak into an OFF boot).
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "0")
    writer()
    assert not flag.exists()


# --------------------------------------------------------------------------- #
# 5. FR13_SUBOP_STAGE markers (the rebuild: NEVER silently vacuous, class-9).
#    These pin the stage-marker helper + the call-site engagement probe + the
#    hoisted engagement asserts, then exercise the emitted helpers on CPU stubs.
# --------------------------------------------------------------------------- #
# 5.1 Patcher-text assertions (cheap; catch a needle drift).
def test_stage_marker_helper_emitted() -> None:
    assert "def _fr13_subop_stage(" in SRC
    assert "FR13_SUBOP_STAGE=%s %s" in SRC
    assert "def _fr13_subop_worker_env_gate(" in SRC
    assert "/proc/self/environ" in SRC


def test_callsite_skip_marker_above_use_fr10_tree() -> None:
    # The call-site engagement probe must sit BEFORE the `if use_fr10_tree:`
    # branch at the scan site (failure-#4 hole).
    assert "callsite-skip" in SRC
    assert re.search(
        r"if _fr13_gdn_subop_mab_enabled\(\) and not use_fr10_tree:.*?"
        r"_fr13_subop_stage\(\s*\n\s+\"callsite-skip\"",
        SRC, re.DOTALL,
    )
    # The orphaned stash is released on the skip (no leak into a later event).
    assert SRC.count("self._fr13_gdn_subop_mab_pre_conv = None") >= 2


def test_engagement_asserts_hoisted_out_of_try() -> None:
    # The disengagement guards now emit a loud `engage-fail` marker and RETURN,
    # they are no longer bare `raise` inside the swallowing try.
    assert "engage-fail" in SRC
    assert "def _engage_fail(reason):" in SRC
    # The swallow handler is now a distinct ERROR tag, not the success-level
    # warning.
    assert 'FR13_GDN_SUBOP_MAB A/B failed' not in SRC  # old swallow text gone
    assert '"arm-fail"' in SRC
    assert '"record-written"' in SRC
    assert '"engaged"' in SRC


# 5.2 Emitted-helper behavioral tests (exec the helper on CPU stubs).
class _ErrLogger:
    """A logger stub that captures `.error` calls so a test can assert which
    FR13_SUBOP_STAGE markers fired (the chase is never silent)."""

    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, *a, **k):  # pragma: no cover - trivial
        self.warnings.append(a)

    def error(self, *a, **k):
        self.errors.append(a)


def _stage_tags(err_logger):
    """Extract the FR13_SUBOP_STAGE=<tag> tag from each captured error call."""
    tags = []
    for a in err_logger.errors:
        # `_fr13_subop_stage` logs as ("FR13_SUBOP_STAGE=%s %s", tag, msg)
        if a and a[0] == "FR13_SUBOP_STAGE=%s %s" and len(a) >= 2:
            tags.append(a[1])
    return tags


def _make_module_err(tmp_path: Path):
    """Like _make_module but the logger captures `.error` so the stage markers
    are observable.  Returns (ns, err_logger)."""
    body = _extract_helper()
    err_logger = _ErrLogger()

    def stub_conv(x, cs, w, bias, act, **k):
        return x.clone() * 2.0 + 0.5

    def stub_scan(*, A_log, a, b, dt_bias, q, k, v, initial_state, **k2):
        out = (q * 3.0)[..., : v.shape[-1]] if q.shape[-1] >= v.shape[-1] else q * 3.0
        return out, None

    ns = {
        "os": os,
        "json": json,
        "torch": torch,
        "logger": err_logger,
        "causal_conv1d_update": stub_conv,
        "fused_sigmoid_gating_delta_rule_update": stub_scan,
    }
    compile(body, "<emitted_mab_helper>", "exec")
    exec(body, ns)
    return ns, err_logger


def test_engage_marker_on_disengaged_tree_no_record(tmp_path, monkeypatch) -> None:
    # Same as the disengaged-tree test, but ALSO assert an `engage-fail` stage
    # marker fired (the chase is never silent) and NO record was written.
    ns, err_logger = _make_module_err(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "diseng.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")

    class _NoTree:
        num_spec_decodes = 1
        fr10_tree_path0_nodes = None
        fr10_tree_parent = None

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
    tags = _stage_tags(err_logger)
    assert "engage-fail" in tags, f"no engage-fail marker fired; tags={tags}"
    assert "record-written" not in tags


def test_record_written_marker_on_success(tmp_path, monkeypatch) -> None:
    # The ON happy-path additionally fires a `record-written` (and `engaged`)
    # stage marker exactly once for the event.
    ns, err_logger = _make_module_err(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    parent, spine = _cat9_geometry()
    tree_n = 10
    H = 8
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "on.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    fn(
        _FakeLayer(),
        torch.randn(tree_n, H),
        torch.randn(tree_n, H),
        torch.randn(tree_n, H),
        torch.randn(tree_n, H),
        torch.randn(1, tree_n, H),
        torch.randn(1, tree_n, H),
        torch.randn(tree_n, H),
        torch.randn(20, H, 4),
        torch.randn(H, 4),
        torch.randn(20, H, H),
        torch.zeros(1, 16, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        torch.tensor([0, tree_n], dtype=torch.long),
        _FakeAttnMeta(tree_n, spine, parent),
        tree_n,
    )
    assert dump.exists(), "flag ON wrote no record"
    tags = _stage_tags(err_logger)
    assert tags.count("record-written") == 1, f"tags={tags}"
    assert "engaged" in tags
    assert "engage-fail" not in tags
    assert "arm-fail" not in tags


def test_worker_env_gate_marker_env_channel(tmp_path, monkeypatch) -> None:
    # FR13_GDN_SUBOP_MAB=1 in env + absent sidecar -> the gate logs
    # `FR13_SUBOP_STAGE=worker-env ... channel=env`.
    ns, err_logger = _make_module_err(tmp_path)
    gate = ns["_fr13_subop_worker_env_gate"]
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    monkeypatch.setenv(
        "FR13_GDN_SUBOP_MAB_FLAG_FILE", str(tmp_path / "absent.flag")
    )
    gate()
    assert "worker-env" in _stage_tags(err_logger)
    joined = " ".join(str(a) for a in err_logger.errors)
    assert "channel=env" in joined, joined


def test_worker_env_gate_marker_sidecar_channel(tmp_path, monkeypatch) -> None:
    # env absent, sidecar flag=1 -> channel=sidecar.
    ns, err_logger = _make_module_err(tmp_path)
    gate = ns["_fr13_subop_worker_env_gate"]
    monkeypatch.delenv("FR13_GDN_SUBOP_MAB", raising=False)
    flag = tmp_path / "fr13_gdn_subop_mab.flag"
    flag.write_text("1\n")
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_FLAG_FILE", str(flag))
    gate()
    assert "worker-env" in _stage_tags(err_logger)
    joined = " ".join(str(a) for a in err_logger.errors)
    assert "channel=sidecar" in joined, joined


def test_stage_marker_once_dedup(tmp_path, monkeypatch) -> None:
    # Two calls with once=True on the same tag -> exactly one logger.error; the
    # counter increments to 2 (record-count discriminator stays live).
    ns, err_logger = _make_module_err(tmp_path)
    stage = ns["_fr13_subop_stage"]
    stage("dedup-tag", "first", once=True)
    stage("dedup-tag", "second", once=True)
    assert _stage_tags(err_logger).count("dedup-tag") == 1
    assert ns["_FR13_SUBOP_STAGE_COUNTS"]["dedup-tag"] == 2


# 5.3 Default-OFF byte-identity (the new markers are wholly behind enabled()).
def test_stage_markers_are_noop_when_off() -> None:
    # The call-site probe block is wholly inside an enabled() guard; OFF => no-op.
    assert re.search(
        r"if _fr13_gdn_subop_mab_enabled\(\) and not use_fr10_tree:",
        SRC,
    )


# --------------------------------------------------------------------------- #
# 6. KERNEL-VALID REDUCED-ROW CONV GEOMETRY (the residual device-assert fix).
#    The reduced (M5/M1) conv arm previously passed num_accepted_tokens =
#    nacc.clamp(min=1,max=m) (the spec path), which drives the FLA conv-update
#    kernel to state_len = width-1+(m-1) and conv_state_token_offset = nacc-1.
#    On a reduced arm that span/offset can run OFF the committed conv-state bank
#    -> a device-side assert INSIDE the kernel (beyond the host-side
#    _guard_rows).  The fix: the reduced arms pass num_accepted_tokens=None so
#    the kernel uses the non-spec path (state_len = width-1, offset = 0), a
#    footprint INDEPENDENT of nacc / max_path_len / the physical column count.
#    The M10 (served-geometry) reference arm KEEPS the spec geometry (faithful
#    co-residency).  These CPU tests capture the kwargs the helper passes to
#    causal_conv1d_update per arm and assert the geometry, plus model the
#    kernel's state_len arithmetic to prove the reduced footprint is bounded.
# --------------------------------------------------------------------------- #
def _make_module_conv_capture(tmp_path: Path):
    """Exec the emitted helper with a conv stub that RECORDS, per call, the
    geometry-defining kwargs (num_accepted_tokens, max_query_len) + row count."""
    body = _extract_helper()
    conv_calls: list[dict] = []

    class _Logger:
        def warning(self, *a, **k):  # pragma: no cover - trivial
            pass

    def stub_conv(x, cs, w, bias, act, **k):
        nacc = k.get("num_accepted_tokens", None)
        conv_calls.append(
            {
                "m": int(x.shape[0]),
                "max_query_len": int(k.get("max_query_len", -1)),
                "nacc_is_none": nacc is None,
                "nacc_vals": (
                    None if nacc is None else [int(v) for v in nacc.reshape(-1).tolist()]
                ),
            }
        )
        return x.clone() * 2.0 + 0.5

    def stub_scan(*, A_log, a, b, dt_bias, q, k, v, initial_state, **k2):
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
    return ns, conv_calls


def test_conv_arm_has_served_geom_split() -> None:
    """The patcher must thread served_geom through _conv_arm (mirrors _scan_arm),
    and the reduced call-sites must pass served_geom=False."""
    helper = _extract_helper()
    assert "def _conv_arm(rows, served_geom):" in helper
    assert "conv_m10 = _conv_arm(full_rows, served_geom=True)" in helper
    assert "conv_m5 = _conv_arm(spine_rows, served_geom=False)" in helper
    assert "conv_m1 = _conv_arm(m1_rows, served_geom=False)" in helper
    # Reduced arms disable the spec sliding-window (num_accepted_tokens=None).
    assert "            else:\n                conv_nacc = None\n" in helper


def test_reduced_conv_arm_passes_nacc_none_full_keeps_spec(tmp_path, monkeypatch) -> None:
    """Live-shaped capture: the M10 conv arm (10 rows) keeps the served spec
    geometry (num_accepted_tokens != None); the reduced M5 (5) + M1 (1) arms
    pass num_accepted_tokens=None -> the kernel's non-spec path (no OOB)."""
    ns, conv_calls = _make_module_conv_capture(tmp_path)
    fn = ns["_fr13_gdn_subop_mab"]
    parent, spine = _cat9_geometry()
    tree_n = 10
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB", "1")
    dump = tmp_path / "convcap.jsonl"
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_DUMP", str(dump))
    monkeypatch.setenv("FR13_GDN_SUBOP_MAB_LAYER", "*")
    _call(
        fn,
        ssi=torch.zeros(1, 16, dtype=torch.long),  # all banks valid (bank 0)
        ssm=torch.randn(20, 8, 8),
        conv_snap=torch.randn(20, 8, 4),  # width 4 -> physical state_len 4
        nacc=5,  # served num_accepted (the OLD reduced-arm assert trigger)
    )
    assert dump.exists(), "live geometry wrote no record"
    # Three conv arms ran in M10, M5, M1 order.
    by_m = {c["m"]: c for c in conv_calls}
    assert set(by_m) == {tree_n, len(spine), 1}, f"arms ran: {[c['m'] for c in conv_calls]}"
    # M10 (full tree) = the faithful reference: spec geometry retained.
    assert not by_m[tree_n]["nacc_is_none"], "M10 arm dropped the served spec geometry"
    # M5 / M1 (reduced) = kernel-valid non-spec geometry.
    assert by_m[len(spine)]["nacc_is_none"], "M5 reduced arm still passes spec nacc (would assert)"
    assert by_m[1]["nacc_is_none"], "M1 reduced arm still passes spec nacc (would assert)"


def _conv_kernel_state_len(width: int, m: int, nacc_is_none: bool) -> tuple[int, int]:
    """Model causal_conv1d_update's host-side state_len + the kernel's
    conv_state_token_offset (causal_conv1d.py wrapper L1233-1236 + kernel
    L859-863).  Returns (state_len, conv_state_token_offset) for the arm; the
    reduced arm (nacc_is_none) must have a footprint bounded by width-1 with
    offset 0, INDEPENDENT of m and nacc -> provably in-range on any committed
    bank with >= width-1 physical columns."""
    if nacc_is_none:
        state_len = width - 1
        offset = 0
    else:
        seqlen = m
        state_len = width - 1 + (seqlen - 1)
        # offset = nacc - 1 (clamped to [1, m] by the served arm); the reference
        # arm's geometry equals the live forward's and is in-range by construction.
        offset = m - 1
    return state_len, offset


def test_reduced_conv_footprint_is_kernel_bounded() -> None:
    """The reduced-arm footprint is width-1 with offset 0 for ANY m / nacc, so
    the kernel's read (offset .. offset+width-1) and store (0 .. state_len-1)
    stay inside the committed bank's width-1 physical window -> no OOB assert.
    The OLD reduced geometry (spec path) grew state_len with m (M5 -> 7 cols),
    which over-ran a width-1 (3-col) committed window = the device assert."""
    width = 4  # cat9 GDN conv width
    phys_cols = width - 1  # a committed conv-state bank's physical column count
    # Reduced arms (the fix): footprint pinned to width-1, offset 0, for all m.
    for m in (1, 5, 10):
        state_len, offset = _conv_kernel_state_len(width, m, nacc_is_none=True)
        assert state_len == phys_cols, f"reduced m={m} state_len={state_len} != {phys_cols}"
        assert offset == 0, f"reduced m={m} offset={offset} != 0"
        # store span [0, state_len) and read window [offset, offset+width-1)
        # both fit the physical bank.
        assert state_len <= phys_cols
        assert offset + (width - 1) <= phys_cols + (width - 2)  # read uses masking
    # The OLD (spec) reduced geometry over-ran the committed width-1 window:
    old_state_len, _ = _conv_kernel_state_len(width, 5, nacc_is_none=False)
    assert old_state_len == 7 and old_state_len > phys_cols, (
        "the spec reduced geometry no longer over-runs the committed bank "
        "(the device-assert precondition is gone) -- recheck the model"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))

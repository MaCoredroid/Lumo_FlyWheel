"""FR13_REPLAY_DURABLE_AB wiring + default-OFF byte-identity gate (CPU).

The replay-durable-state A/B (FR13_REPLAY_DURABLE_AB, default OFF) is the PRIME
lead from FR13_TOTAL_DRIFT_REANALYSIS_LEADS_BIND.md: an OBSERVE-ONLY in-process
hook that, on every commit event, after OUR replay kernel publishes the durable
GDN recurrent state (H_ours) into the served ssm_state bank, ALSO runs NATIVE
MTP's sequential recurrent kernel (fused_sigmoid_gating_delta_rule_update) over
the SAME root+accepted token chain from the SAME CLONED h0, and records
max_abs(H_ours - H_native_seq) per GDN layer + first-nonzero + the per-EVENT
index (back-loading). The served stream is NEVER touched (native runs on a
detached clone with inplace_final_state=False, writing into a fresh tensor).

This CPU test pins (live GPU boot remains the binding gate, class 8):
  1. The patcher emits the durable-AB resolver + emit + native A/B helper +
     call sites, ALL behind the FR13_REPLAY_DURABLE_AB flag (class-9
     engagement / stage markers present).
  2. The patch is PURELY ADDITIVE (no served replay-launch needle deleted):
     default-OFF path is byte-identical to the locked cat9 build.
  3. The emitted helper compiles, the resolver returns False when the flag is
     off (default-OFF no-op), and -- when forced ON via a fake resolver -- the
     native A/B runs on CLONES, calls native with a B=1 linear chain
     (cu_seqlens, inplace_final_state=False), writes exactly one JSONL record
     per (layer, row), and does NOT mutate the served bank (observe-only).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
SRC = PATCHER.read_text()


# --------------------------------------------------------------------------- #
# 1. Flag-gated, sidecar-bridged, default-OFF.
# --------------------------------------------------------------------------- #
def test_flag_guard_present_default_off() -> None:
    # Worker-env-safe resolver (the bare master is curated out of the mp/spawn
    # EngineCore worker -- same as SUBOP_MAB, measured 2026-06-14).
    assert "def _fr13_replay_durable_ab_enabled():" in SRC
    # Helper top-guard returns immediately when OFF.
    assert "if not _fr13_replay_durable_ab_enabled():" in SRC
    # Both committer call sites resolve via the helper (never a bare env read).
    assert SRC.count("_fr13_rdab_on = _fr13_replay_durable_ab_enabled()") == 2
    assert SRC.count("if _fr13_rdab_on:") == 2
    # Patch-time sidecar bridge (pid 1 -> worker via /logs bind mount), and it
    # is registered in main() right after the SUBOP_MAB sidecar.
    assert "def _fr13_write_replay_durable_ab_sidecar()" in SRC
    assert "_fr13_write_replay_durable_ab_sidecar()" in SRC
    assert "/logs/fr13_replay_durable_ab.flag" in SRC
    # Default-safe: only the patch-time sidecar writer reads the bare master env
    # (pid 1); the worker-side resolver consults env-or-sidecar.
    assert SRC.count('os.environ.get("FR13_REPLAY_DURABLE_AB", "0")') == 1


def test_stage_markers_class9() -> None:
    # Loud grep-able stage markers so the run can NEVER be silently vacuous.
    assert "FR13_RDAB_STAGE=%s %s" in SRC
    assert '"record-written"' in SRC
    assert '"arm-fail"' in SRC
    # The header records the native reference kernel identity + flag state.
    assert '"ref_kernel": "fused_sigmoid_gating_delta_rule_update"' in SRC


def test_observe_only_documented_bans() -> None:
    # No reward-hacks: the helper docstring documents the observe-only contract
    # (clone, inplace_final_state=False, served bank read-only, NOT a reroute).
    blob = SRC
    assert "OBSERVE-ONLY" in blob
    assert "inplace_final_state=False" in blob
    assert "NOT a reroute" in blob
    # native arm runs on a CLONE of h0 + a flat gather of the rings.
    assert ".clone()" in blob


# --------------------------------------------------------------------------- #
# 2. Purely additive -> default path byte-identical to locked cat9.
# --------------------------------------------------------------------------- #
def test_patch_is_purely_additive_no_replay_needle_deleted() -> None:
    # The served replay launches (the thing we must NOT touch) are intact in
    # BOTH committer variants.
    assert SRC.count("_fr13_replay_launch(") == 2
    assert "state_bank=_fr13_ssm_bank," in SRC
    # The EAGER_PACK fast loop is only bypassed when the AB (or boundary) flag
    # is ON -- default OFF keeps the fast path.
    assert "if _ep_active and not _fr13_bnd_on and not _fr13_rdab_on:" in SRC
    # The shared per-event counter is incremented for boundary OR durable-AB.
    assert SRC.count("if _fr13_bnd_on or _fr13_rdab_on:") == 2


# --------------------------------------------------------------------------- #
# 3. Extract + compile + exercise the emitted helper on CPU.
# --------------------------------------------------------------------------- #
def _extract_helper() -> str:
    # The committer + durable-AB helpers live in the rejection_sampler LCP
    # helper raw string: helper = r''' ... '''.
    m = re.search(r"\n    helper = r'''(.*?)\n'''\n", SRC, re.DOTALL)
    assert m, "LCP helper raw string not found in patcher"
    return m.group(1)


def _exec_helper(stub_native, fake_enabled):
    body = _extract_helper()

    class _Logger:
        def __getattr__(self, n):
            return lambda *a, **k: None

    ns = {
        "os": os,
        "json": json,
        "torch": torch,
        "hashlib": __import__("hashlib"),
        "time": __import__("time"),
        "logger": _Logger(),
    }
    compile(body, "<lcp_helper>", "exec")
    exec(body, ns)
    # Inject a stub native kernel + force the resolver ON, so the exec'd helper
    # does not need a GPU / vLLM import.
    ns["_FR13_RDAB_FLAG"] = bool(fake_enabled)

    def _patched_native(**kw):
        return stub_native(**kw)

    # Monkeypatch the module-namespace name the helper imports lazily by
    # pre-seeding sys.modules with a fake fused_sigmoid_gating module.
    import sys
    import types

    fake_mod = types.ModuleType(
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating"
    )
    fake_mod.fused_sigmoid_gating_delta_rule_update = _patched_native
    # ensure the package path exists for the `from ... import` to resolve
    for pkg in [
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.fla",
        "vllm.model_executor.layers.fla.ops",
    ]:
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    sys.modules[
        "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating"
    ] = fake_mod
    return ns


class _FakeLayer:
    prefix = "language_model.model.layers.0.linear_attn"


def _make_fake_layer(num_vh, dim_v, dim_k, n_pad, rows):
    lyr = _FakeLayer()
    # ring layout k(B,N,KH,DK) v(B,N,VH,DV) a/b(B,N,VH); single k head group.
    lyr._fr13_replay_ring_k = torch.randn(rows, n_pad, 1, dim_k)
    lyr._fr13_replay_ring_v = torch.randn(rows, n_pad, num_vh, dim_v)
    lyr._fr13_replay_ring_a = torch.randn(rows, n_pad, num_vh)
    lyr._fr13_replay_ring_b = torch.randn(rows, n_pad, num_vh)
    lyr.A_log = torch.randn(num_vh)
    lyr.dt_bias = torch.randn(num_vh)
    # spec_idx maps each row's columns to bank rows; prev_lens = scan snapshot.
    lyr._fr13_replay_spec_idx = torch.tensor(
        [[10 + c for c in range(n_pad)] for _ in range(rows)], dtype=torch.long
    )
    lyr._fr13_replay_prev_lens = torch.tensor([1] * rows, dtype=torch.long)
    return lyr


def test_resolver_default_off_no_op(monkeypatch) -> None:
    # When neither env nor sidecar is set, the resolver is False and the helper
    # is a no-op (default-OFF byte-identical path).
    monkeypatch.delenv("FR13_REPLAY_DURABLE_AB", raising=False)
    monkeypatch.setenv("FR13_REPLAY_DURABLE_AB_FLAG_FILE", "/nonexistent/flag")

    calls = {"n": 0}

    def stub_native(**kw):
        calls["n"] += 1
        T = kw["k"].shape[1]
        HV = kw["v"].shape[2]
        V = kw["v"].shape[3]
        K = kw["k"].shape[3]
        return None, torch.zeros(T, HV, V, K)

    ns = _exec_helper(stub_native, fake_enabled=False)
    # force a fresh resolution (resolver caches): with flag forced None it reads
    # env/sidecar -> both absent -> False.
    ns["_FR13_RDAB_FLAG"] = None
    assert ns["_fr13_replay_durable_ab_enabled"]() is False
    lyr = _make_fake_layer(num_vh=2, dim_v=4, dim_k=4, n_pad=8, rows=1)
    bank = torch.randn(64, 2, 4, 4)
    bank_before = bank.clone()

    class _GdnMod:
        pass

    ns["_fr13_replay_durable_ab"](
        _GdnMod(), lyr.prefix, lyr, bank, 1,
        [[1, 2]], [2], 0,
    )
    assert calls["n"] == 0  # OFF -> native never called
    assert torch.equal(bank, bank_before)  # nothing touched


def test_native_ab_runs_observe_only_and_records(tmp_path, monkeypatch) -> None:
    out = tmp_path / "rdab.jsonl"
    monkeypatch.setenv("FR13_REPLAY_DURABLE_AB", "1")
    monkeypatch.setenv("FR13_REPLAY_DURABLE_AB_PATH", str(out))

    captured = {"shapes": None, "cu": None, "inplace": None, "h0": None}

    def stub_native(**kw):
        # Record what the helper fed native: a B=1 linear chain (cu_seqlens),
        # inplace_final_state=False (observe-only fresh output).
        captured["shapes"] = (
            tuple(kw["k"].shape), tuple(kw["v"].shape),
            tuple(kw["a"].shape), tuple(kw["b"].shape),
        )
        captured["cu"] = kw["cu_seqlens"].tolist()
        captured["inplace"] = kw.get("inplace_final_state")
        captured["h0"] = kw["initial_state"].clone()
        T = kw["k"].shape[1]
        HV = kw["v"].shape[2]
        V = kw["v"].shape[3]
        K = kw["k"].shape[3]
        # Return a deterministic "native" final state; final = ht[T-1].
        ht = torch.arange(T * HV * V * K, dtype=torch.float32).reshape(T, HV, V, K)
        return None, ht

    ns = _exec_helper(stub_native, fake_enabled=True)
    ns["_FR13_RDAB_FLAG"] = True

    num_vh, dim_v, dim_k, n_pad, rows = 2, 4, 4, 8, 1
    lyr = _make_fake_layer(num_vh, dim_v, dim_k, n_pad, rows)
    bank = torch.randn(64, num_vh, dim_v, dim_k)
    bank_before = bank.clone()

    class _GdnMod:
        pass

    # accepted path [1,2], len 2 -> chain = root(0)+[1,2] => M=3.
    ns["_fr13_replay_durable_ab"](
        _GdnMod(), lyr.prefix, lyr, bank, rows,
        [[1, 2]], [2], 7,
    )

    # observe-only: the served bank is byte-untouched.
    assert torch.equal(bank, bank_before)
    # native fed a B=1 linear chain of length M=3, cu_seqlens=[0,3].
    assert captured["cu"] == [0, 3]
    assert captured["inplace"] is False
    kshp, vshp, ashp, bshp = captured["shapes"]
    assert kshp == (1, 3, 1, dim_k)
    assert vshp == (1, 3, num_vh, dim_v)
    assert ashp == (1, 3, num_vh)
    assert bshp == (1, 3, num_vh)
    # h0 is a CLONE of the bank's h0 source row (prev_len=1 -> col 0 -> row 10).
    assert torch.equal(captured["h0"][0], bank[10].to(torch.float32))

    # exactly one record with the back-loading event index + max_abs field.
    lines = [json.loads(_ln) for _ln in out.read_text().splitlines()]
    recs = [r for r in lines if r.get("tap") == "durable_ab"]
    assert len(recs) == 1
    r = recs[0]
    assert r["event"] == 7
    assert r["accepted_len"] == 2
    assert r["M_chain"] == 3
    assert r["node_chain"] == [0, 1, 2]
    assert r["h0_src_row"] == 10
    # final dst row = spec_idx col clamp(acc_len-1,0)=1 -> row 11.
    assert r["final_dst_row"] == 11
    assert "max_abs_h_ours_vs_native" in r
    assert isinstance(r["max_abs_h_ours_vs_native"], float)


def test_zero_accept_chain_is_root_only(tmp_path, monkeypatch) -> None:
    # acc_len == 0 -> chain = root(0) only, M=1, final col clamps to 0.
    out = tmp_path / "rdab0.jsonl"
    monkeypatch.setenv("FR13_REPLAY_DURABLE_AB", "1")
    monkeypatch.setenv("FR13_REPLAY_DURABLE_AB_PATH", str(out))

    seen = {"cu": None}

    def stub_native(**kw):
        seen["cu"] = kw["cu_seqlens"].tolist()
        T = kw["k"].shape[1]
        HV = kw["v"].shape[2]
        V = kw["v"].shape[3]
        K = kw["k"].shape[3]
        return None, torch.zeros(T, HV, V, K)

    ns = _exec_helper(stub_native, fake_enabled=True)
    ns["_FR13_RDAB_FLAG"] = True
    lyr = _make_fake_layer(2, 4, 4, 8, 1)
    bank = torch.randn(64, 2, 4, 4)

    class _GdnMod:
        pass

    ns["_fr13_replay_durable_ab"](
        _GdnMod(), lyr.prefix, lyr, bank, 1, [[]], [0], 3,
    )
    assert seen["cu"] == [0, 1]
    recs = [json.loads(_ln) for _ln in out.read_text().splitlines()]
    r = [x for x in recs if x.get("tap") == "durable_ab"][0]
    assert r["M_chain"] == 1
    assert r["node_chain"] == [0]

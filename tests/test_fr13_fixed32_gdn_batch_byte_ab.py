from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")
    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    triton_stub.next_power_of_2 = lambda value: 1 << (value - 1).bit_length()
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel


def test_real_event_control_is_worker_safe_and_rejects_probe_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled = tmp_path / "enabled"
    event = tmp_path / "real_event.arm"
    monkeypatch.delenv("FR13_FIXED32_BATCH_GDN_BYTE_AB", raising=False)
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED_PATH", str(enabled)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH", str(event)
    )

    assert kernel._fr13_fixed32_batch_gdn_byte_ab_control() == (False, None)
    enabled.write_text("1\n", encoding="ascii")
    assert kernel._fr13_fixed32_batch_gdn_byte_ab_control() == (True, None)

    event.write_text("probe:synthetic\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="swe_verified:<task_id>"):
        kernel._fr13_fixed32_batch_gdn_byte_ab_control()

    event.write_text("swe_verified:django__django-12345\n", encoding="ascii")
    assert kernel._fr13_fixed32_batch_gdn_byte_ab_control() == (
        True,
        "swe_verified:django__django-12345",
    )


def test_selector_is_default_off_and_production_requires_live_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = tmp_path / "diagnostic.enabled"
    event = tmp_path / "real_event.arm"
    production = tmp_path / "production.arm"
    live_pass = tmp_path / "pass.json"
    for name in (
        "FR13_FIXED32_BATCH_GDN_BYTE_AB",
        "FR13_FIXED32_BATCH_GDN_PRODUCTION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED_PATH", str(diagnostic)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH", str(event)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_PRODUCTION_ARM_PATH", str(production)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(live_pass)
    )

    assert kernel.fixed32_batch_gdn_selector(1) is None
    assert kernel.fixed32_batch_gdn_selector(4) is None

    diagnostic.write_text("1\n", encoding="ascii")
    assert kernel.fixed32_batch_gdn_selector(4) == "diagnostic"

    production.write_text("1\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="requires a readable live-gate PASS"):
        kernel.fixed32_batch_gdn_selector(4)

    live_pass.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.batch_gdn.live_pass.v1",
                "status": "pass",
                "task_marker": "swe_verified:django__django-12345",
                "batch": 4,
                "layer_count": 48,
                "layer_keys": [f"0x{index:x}" for index in range(48)],
                "reference_always_served": True,
            }
        ),
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        kernel.fixed32_batch_gdn_selector(4)

    diagnostic.unlink()
    assert kernel.fixed32_batch_gdn_selector(4) == "production"
    with pytest.raises(RuntimeError, match="batch does not match"):
        kernel.fixed32_batch_gdn_selector(3)


def test_live_pass_emits_only_after_all_48_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pass.json"
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(path)
    )
    kernel._fr13_fixed32_batch_gdn_live_pass_emit(
        task_marker="swe_verified:django__django-12345",
        batch=4,
        layer_keys=set(range(47)),
    )
    assert not path.exists()
    kernel._fr13_fixed32_batch_gdn_live_pass_emit(
        task_marker="swe_verified:django__django-12345",
        batch=4,
        layer_keys=set(range(48)),
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["status"] == "pass"
    assert payload["layer_count"] == 48
    assert payload["reference_always_served"] is True


def test_byte_diff_reports_first_nonzero_byte() -> None:
    reference = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    equal = kernel._fr13_fixed32_batch_gdn_byte_diff(
        "equal", reference, reference.clone()
    )
    assert equal["byte_equal"] is True
    assert equal["differing_bytes"] == 0
    assert equal["first_nonzero_byte"] is None

    candidate = reference.clone()
    candidate[2] = 9
    mismatch = kernel._fr13_fixed32_batch_gdn_byte_diff(
        "mismatch", reference, candidate
    )
    assert mismatch["byte_equal"] is False
    assert mismatch["differing_bytes"] == 1
    assert mismatch["first_nonzero_byte"] == 2
    assert mismatch["reference_byte"] == 3
    assert mismatch["candidate_byte"] == 9


class _FakeBatchKernel:
    def __init__(self, launch) -> None:
        self.launch = launch

    def __getitem__(self, _grid):
        return self.launch


@pytest.mark.parametrize("candidate_mismatch", [False, True])
def test_real_event_gate_restores_and_serves_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_mismatch: bool,
) -> None:
    batch = 2
    n_actual = n_pad = 32
    rows = batch * n_actual
    q = torch.empty((rows, 1, 1), dtype=torch.float32)
    q[:n_actual].fill_(1.0)
    q[n_actual:].fill_(2.0)
    k = q.clone().add_(10.0)
    v = q.clone().add_(20.0)
    g = q.reshape(rows, 1).clone().add_(30.0)
    beta = g.clone().add_(10.0)
    raw_a = g.clone().add_(20.0)
    raw_b = g.clone().add_(30.0)
    out = torch.full_like(q, -1.0)
    ring_k = torch.full_like(k, -2.0)
    ring_v = torch.full_like(v, -3.0)
    ring_a = torch.full_like(raw_a, -4.0)
    ring_b = torch.full_like(raw_b, -5.0)
    flags = torch.tensor([-6, -7], dtype=torch.int32)
    counter = torch.tensor(11, dtype=torch.int32)
    strict = torch.zeros((n_pad, n_pad), dtype=torch.int32)
    visible = torch.zeros_like(strict)
    h0 = torch.zeros((4, 1, 1, 1), dtype=torch.float32)
    h0_indices = torch.zeros((batch, 1), dtype=torch.int64)
    accepted = torch.zeros((batch,), dtype=torch.int32)
    a_log = torch.zeros((1,), dtype=torch.float32)
    dt_bias = torch.zeros((1,), dtype=torch.float32)
    export = torch.arange(32, dtype=torch.float32).reshape(32, 1, 1, 1)
    export_before = export.clone()
    level = (
        torch.zeros((1, 1), dtype=torch.int32),
        torch.zeros((1,), dtype=torch.int32),
        1,
        1,
        torch.ones((1,), dtype=torch.int32),
    )
    subtree_state = {
        "schedule": "fixed32",
        "fixed32_contract": {"launches": 2},
        "fixed32_parent_slots": (
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        ),
        "route_armed": True,
        "selfcheck_armed": False,
        "levels": (level, level),
        "export": export,
        "batch_engaged_announced": False,
    }
    calls = {"legacy": 0, "candidate": 0}
    records: list[dict[str, object]] = []

    def fake_legacy(**kwargs) -> tuple[torch.Tensor, None]:
        request = int(kwargs["q"][0, 0, 0].item()) - 1
        calls["legacy"] += 1
        kwargs["out"].copy_(kwargs["q"] + 100.0)
        kwargs["ring_k"].copy_(kwargs["k"])
        kwargs["ring_v"].copy_(kwargs["v"])
        kwargs["ring_a"].copy_(kwargs["raw_a"])
        kwargs["ring_b"].copy_(kwargs["raw_b"])
        kwargs["staging_flags"][0] = 1
        kwargs["staging_flags"][1] = batch
        kwargs["invocation_counter"].add_(1)
        for slot, node in enumerate(kernel._FR13_FIXED32_EXPORT_NODES):
            export[node].fill_(request * 10 + slot)
        return kwargs["out"], None

    def fake_candidate(*_args, **kwargs) -> None:
        calls["candidate"] += 1
        if kwargs["STATE_SOURCE"] != 1:
            return
        out.copy_(q + 100.0)
        if candidate_mismatch:
            out[0, 0, 0] += 1.0
        ring_k.copy_(k)
        ring_v.copy_(v)
        ring_a.copy_(raw_a)
        ring_b.copy_(raw_b)
        flags[0] = 1
        flags[1] = batch
        counter.add_(batch)
        for request in range(batch):
            for slot in range(kernel._FR13_FIXED32_EXPORT_SLOTS):
                export[request * kernel._FR13_FIXED32_EXPORT_SLOTS + slot].fill_(
                    request * 10 + slot
                )

    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "subtree_get", lambda *_args: subtree_state)
    monkeypatch.setattr(
        kernel,
        "_read_tree_gdn_geom_override",
        lambda: {"BV": 1, "num_warps": 1},
    )
    monkeypatch.setattr(
        kernel.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_byte_ab_control",
        lambda: (True, "swe_verified:django__django-12345"),
    )
    monkeypatch.setattr(kernel, "launch_tree_gdn_prepared", fake_legacy)
    monkeypatch.setattr(
        kernel,
        "_tree_gdn_path_kernel_fixed32_batch",
        _FakeBatchKernel(fake_candidate),
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_byte_ab_emit",
        lambda record: records.append(record),
    )
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_BATCH_GDN_BYTE_AB_STATE",
        {"passed": set(), "attempts": {}, "waiting_announced": set()},
    )

    kernel.launch_tree_gdn_prepared_fixed32_batch(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        h0=h0,
        batch_size=batch,
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        out=out,
        h0_indices=h0_indices,
        h0_num_accepted_tokens=accepted,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=a_log,
        dt_bias=dt_bias,
        output_scale=1.0,
        use_qk_l2norm_in_kernel=True,
        invocation_counter=counter,
        ring_k=ring_k,
        ring_v=ring_v,
        ring_a=ring_a,
        ring_b=ring_b,
        staging_flags=flags,
        staging_rows=batch,
    )

    assert calls == {"legacy": batch, "candidate": 2}
    assert torch.equal(out, q + 100.0)
    assert torch.equal(ring_k, k)
    assert torch.equal(ring_v, v)
    assert torch.equal(ring_a, raw_a)
    assert torch.equal(ring_b, raw_b)
    assert flags.tolist() == [1, batch]
    assert counter.item() == 11 + batch
    expected_export = export_before
    for slot, node in enumerate(kernel._FR13_FIXED32_EXPORT_NODES):
        expected_export[node].fill_(10 + slot)
    assert torch.equal(export, expected_export)

    assert len(records) == 1
    record = records[0]
    assert record["reference_restored_and_served"] is True
    assert record["candidate_physical_launches"] == 2
    assert record["legacy_physical_launches"] == 2 * batch
    assert record["zero_diff"] is (not candidate_mismatch)
    if candidate_mismatch:
        assert record["first_nonzero"]["name"] == "out"
        assert not kernel._FR13_FIXED32_BATCH_GDN_BYTE_AB_STATE["passed"]
    else:
        assert record["first_nonzero"] is None
        assert kernel._FR13_FIXED32_BATCH_GDN_BYTE_AB_STATE["passed"] == {
            int(a_log.data_ptr())
        }


def test_launcher_requires_eager_real_task_gate() -> None:
    launcher = (
        ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")
    assert (
        "FR13_FIXED32_BATCH_GDN_BYTE_AB requires ENFORCE_EAGER=1" in launcher
    )
    assert (
        "FR13_FIXED32_BATCH_GDN_BYTE_AB requires FR10_METRICS=1" in launcher
    )
    assert "fr13_fixed32_batch_gdn_byte_ab.enabled" in launcher
    assert "fr13_fixed32_batch_gdn_byte_ab.real_event.arm" in launcher
    assert "FR13_FIXED32_BATCH_GDN_PRODUCTION:-0" in launcher
    assert "diagnostic and production are mutually exclusive" in launcher
    assert "requires a regular live-gate PASS record" in launcher
    assert "fr13_fixed32_batch_gdn_production.arm" in launcher


def test_passed_diagnostic_layers_still_serve_reference() -> None:
    source = (
        ROOT
        / "src"
        / "lumo_flywheel_serving"
        / "fr10_gdn_tree_kernel.py"
    ).read_text(encoding="utf-8")
    passed = source.index('if layer_key in gate_state["passed"]:')
    next_branch = source.index("elif real_event_marker is None:", passed)
    branch = source[passed:next_branch]
    assert "_launch_reference(collect_export=False)" in branch
    assert "_launch_batched()" not in branch
    assert "return out, None" in branch

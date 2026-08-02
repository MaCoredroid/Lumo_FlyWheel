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

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel  # noqa: E402


def _graph_record(
    layer_key: int,
    *,
    mismatch: bool = False,
    shared_export: torch.Tensor | None = None,
    candidate_structure: str = "fixed32_batch_tree_gdn_path",
):
    surfaces = kernel._FR13_FIXED32_BATCH_GDN_GRAPH_SURFACES
    state = {
        "out": torch.full((2,), 7, dtype=torch.uint8),
        "export": (
            torch.arange(32, dtype=torch.uint8)
            if shared_export is None
            else shared_export
        ),
        "ring_k": torch.full((2,), 11, dtype=torch.uint8),
        "ring_v": torch.full((2,), 12, dtype=torch.uint8),
        "ring_a": torch.full((2,), 13, dtype=torch.uint8),
        "ring_b": torch.full((2,), 14, dtype=torch.uint8),
        "flags": torch.tensor([1, 1, 1, 1], dtype=torch.int32),
        "invocation_counter": torch.tensor(9, dtype=torch.int32),
    }
    served = {name: tensor.clone() for name, tensor in state.items()}
    compact = torch.arange(20, dtype=torch.uint8)

    def snapshot():
        return {name: state[name].clone() for name in surfaces}

    def restore(value):
        for name in surfaces:
            state[name].copy_(value[name])

    def run_reference():
        state["invocation_counter"].fill_(13)
        state["export"][:5].fill_(layer_key)
        return {
            "block_v": 8,
            "physical_launches": 8,
            "kernel_structure": "per_request_tree_gdn_path",
            "compact_export": compact.clone(),
        }

    def run_candidate(block_v: int):
        state["invocation_counter"].fill_(13)
        state["export"][:20].copy_(compact)
        if mismatch:
            state["out"][0] = 99
        return {
            "block_v": block_v,
            "physical_launches": 2,
            "kernel_structure": candidate_structure,
            "compact_export": compact.clone(),
        }

    return (
        {
            "layer_key": layer_key,
            "snapshot": snapshot,
            "restore": restore,
            "run_reference": run_reference,
            "run_candidate": run_candidate,
            "carrier_nonzero": lambda: True,
            "byte_equal": torch.equal,
            "surface_names": surfaces,
        },
        state,
        served,
    )


def test_graph_selector_is_separate_and_only_routes_final_b4_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eager = tmp_path / "eager.enabled"
    graph = tmp_path / "graph.enabled"
    production = tmp_path / "production.arm"
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED_PATH", str(eager))
    monkeypatch.setenv(
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB_ENABLED_PATH", str(graph)
    )
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_PRODUCTION_ARM_PATH", str(production))
    monkeypatch.delenv("FR13_FIXED32_BATCH_GDN_BYTE_AB", raising=False)
    monkeypatch.delenv("FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB", raising=False)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", 64)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", None)
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT", None
    )

    graph.write_text("1\n", encoding="ascii")
    assert kernel.fixed32_batch_gdn_selector(2) is None
    assert kernel.fixed32_batch_gdn_selector(4) is None
    kernel.fixed32_batch_gdn_graph_live_capture_begin(401, 4)
    assert kernel.fixed32_batch_gdn_selector(4) == "graph_capture"
    assert kernel.fixed32_batch_gdn_selector(3) is None

    eager.write_text("1\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="eager and graph-replay"):
        kernel.fixed32_batch_gdn_selector(4)


def test_graph_capture_requires_48_unique_layer_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_graph_byte_ab_control", lambda: True
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", 64)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURE_CONTEXT", None
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES", {})
    kernel.fixed32_batch_gdn_graph_live_capture_begin(402, 4)
    for layer_key in range(1, 49):
        record, _state, _served = _graph_record(layer_key)
        kernel._fr13_fixed32_batch_gdn_graph_live_capture_register(record)
    kernel.fixed32_batch_gdn_graph_live_capture_end(402, 4, "a" * 64, 48)

    capture = kernel._FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES[402]
    assert capture["batch_size"] == 4
    assert capture["graph_signature"] == "a" * 64
    assert len(capture["records"]) == 48
    assert capture["layer_keys"] == frozenset(range(1, 49))


@pytest.mark.parametrize("mismatch", (False, True))
def test_graph_shadow_compare_always_restores_graph_served_bytes(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: bool,
) -> None:
    record, state, served = _graph_record(1, mismatch=mismatch)
    emitted = []
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_byte_ab_emit",
        lambda payload: emitted.append(payload),
    )
    compare = kernel._fr13_fixed32_batch_gdn_graph_compare_records
    kwargs = {
        "candidate_bv": 64,
        "graph_id": 403,
        "graph_signature": "b" * 64,
        "task_marker": "swe_verified:django__django-12345",
    }
    if mismatch:
        with pytest.raises(RuntimeError, match="byte mismatch.*candidate"):
            compare((record,), **kwargs)
    else:
        result = compare((record,), **kwargs)
        assert result["records"] == 1
        assert result["layer_keys"] == {1}
    assert all(torch.equal(state[name], served[name]) for name in served)
    assert emitted[0]["gate_mode"] == "post_replay_shadow"
    assert emitted[0]["graph_baseline_byte_equal"] is True
    assert emitted[0]["status"] == ("mismatch_reference_served" if mismatch else "pass")


def test_graph_shadow_bv8_requires_distinct_batched_kernel_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record, state, served = _graph_record(1)
    emitted = []
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_byte_ab_emit",
        lambda payload: emitted.append(payload),
    )

    result = kernel._fr13_fixed32_batch_gdn_graph_compare_records(
        (record,),
        candidate_bv=8,
        graph_id=408,
        graph_signature="2" * 64,
        task_marker="swe_verified:astropy__astropy-12907",
    )
    assert result["reference_bv"] == result["candidate_bv"] == 8
    assert result["reference_kernel_structure"] == "per_request_tree_gdn_path"
    assert result["candidate_kernel_structure"] == "fixed32_batch_tree_gdn_path"
    assert emitted[0]["legacy_physical_launches"] == 8
    assert emitted[0]["candidate_physical_launches"] == 2
    assert emitted[0]["reference_kernel_structure"] != emitted[0][
        "candidate_kernel_structure"
    ]
    assert all(torch.equal(state[name], served[name]) for name in served)

    invalid, _invalid_state, _invalid_served = _graph_record(
        2, candidate_structure="per_request_tree_gdn_path"
    )
    with pytest.raises(RuntimeError, match="launch metadata drift"):
        kernel._fr13_fixed32_batch_gdn_graph_compare_records(
            (invalid,),
            candidate_bv=8,
            graph_id=409,
            graph_signature="3" * 64,
            task_marker="swe_verified:astropy__astropy-12907",
        )


def test_graph_shadow_does_not_treat_shared_export_as_layer_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_export = torch.arange(32, dtype=torch.uint8)
    served_export = shared_export.clone()
    first, _first_state, _first_served = _graph_record(
        1, shared_export=shared_export
    )
    second, _second_state, _second_served = _graph_record(
        2, shared_export=shared_export
    )
    emitted = []
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_byte_ab_emit",
        lambda payload: emitted.append(payload),
    )

    result = kernel._fr13_fixed32_batch_gdn_graph_compare_records(
        (first, second),
        candidate_bv=64,
        graph_id=406,
        graph_signature="f" * 64,
        task_marker="swe_verified:astropy__astropy-12907",
    )
    assert result["records"] == 2
    assert torch.equal(shared_export, served_export)
    assert len(emitted) == 2
    assert all(
        "graph_baseline_export"
        not in {comparison["name"] for comparison in record["graph_comparisons"]}
        for record in emitted
    )


def test_graph_live_pass_is_graph_task_and_capture_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pass.json"
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(path))
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", 64)
    kernel._fr13_fixed32_batch_gdn_live_pass_emit(
        task_marker="swe_verified:django__django-12345",
        batch=4,
        layer_keys=set(range(1, 49)),
        reference_bv=8,
        candidate_bv=64,
        graph_id=404,
        graph_signature="c" * 64,
        capture_records=48,
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["schema"] == "fr13.fixed32.batch_gdn.graph_live_pass.v1"
    assert payload["gate_mode"] == "post_replay_shadow"
    assert payload["graph_id"] == 404
    assert payload["graph_signature"] == "c" * 64
    assert payload["capture_records"] == 48
    assert payload["real_task_authenticated"] is True
    assert payload["graph_baseline_byte_equal"] is True
    assert payload["state_restored"] is True


def test_graph_bv8_live_pass_is_explicit_and_requires_credential_to_produce(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "pass.json"
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(path))
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    kernel._fr13_fixed32_batch_gdn_live_pass_emit(
        task_marker="swe_verified:astropy__astropy-12907",
        batch=4,
        layer_keys=set(range(1, 49)),
        reference_bv=8,
        candidate_bv=8,
        graph_id=410,
        graph_signature="4" * 64,
        capture_records=48,
    )
    payload = json.loads(path.read_text(encoding="ascii"))
    assert payload["schema"] == "fr13.fixed32.batch_gdn.graph_live_pass.v1"
    assert payload["candidate"] == "fixed32_batch_gdn_bv8_v1"
    assert payload["reference_bv"] == payload["candidate_bv"] == 8
    assert payload["reference_kernel_structure"] == "per_request_tree_gdn_path"
    assert payload["candidate_kernel_structure"] == "fixed32_batch_tree_gdn_path"
    assert payload["reference_physical_launches_per_layer"] == 8
    assert payload["candidate_physical_launches_per_layer"] == 2
    assert payload["production_eligible"] is True

    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_PRODUCTION", "1")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", None)
    with pytest.raises(RuntimeError, match="PASS record is invalid"):
        kernel._fr13_fixed32_batch_gdn_production_control()
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", 8)
    with pytest.raises(RuntimeError, match="production credential is invalid"):
        kernel._fr13_fixed32_batch_gdn_production_control()


def test_graph_gate_runs_once_and_requires_the_captured_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = tuple(object() for _ in range(48))
    signature = "d" * 64
    calls = []
    emitted = []
    state = {
        "status": "armed",
        "candidate_bv": 64,
        "graph_id": None,
        "graph_signature": None,
        "batch_size": None,
        "records": 0,
    }
    captures = {
        405: {
            "batch_size": 4,
            "graph_signature": signature,
            "records": records,
            "layer_keys": frozenset(range(1, 49)),
        }
    }
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_graph_byte_ab_control", lambda: True
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", 64)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_STATE", state)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES", captures)
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_real_event_marker",
        lambda: "swe_verified:astropy__astropy-12907",
    )

    def compare(actual, **kwargs):
        calls.append((actual, kwargs))
        return {
            "records": 48,
            "layer_keys": set(range(1, 49)),
            "reference_bv": 8,
            "candidate_bv": 64,
        }

    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_graph_compare_records", compare
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_live_pass_emit",
        lambda **payload: emitted.append(payload),
    )
    gate = kernel.fixed32_batch_gdn_graph_live_gate_on_replay
    with pytest.raises(RuntimeError, match="replay/capture drift"):
        gate(405, "e" * 64, 4, 48)

    first = gate(405, signature, 4, 48)
    second = gate(405, signature, 4, 48)
    assert first == second == {
        "status": "passed",
        "candidate_bv": 64,
        "graph_id": 405,
        "graph_signature": signature,
        "batch_size": 4,
        "records": 48,
    }
    assert len(calls) == 1
    assert calls[0][1]["task_marker"] == "swe_verified:astropy__astropy-12907"
    assert emitted == [
        {
            "task_marker": "swe_verified:astropy__astropy-12907",
            "batch": 4,
            "layer_keys": set(range(1, 49)),
            "reference_bv": 8,
            "candidate_bv": 64,
            "graph_id": 405,
            "graph_signature": signature,
            "capture_records": 48,
        }
    ]
    assert captures == {}


def test_graph_gate_cannot_pass_if_pass_artifact_publish_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = "1" * 64
    records = tuple(object() for _ in range(48))
    state = {
        "status": "armed",
        "candidate_bv": 64,
        "graph_id": None,
        "graph_signature": None,
        "batch_size": None,
        "records": 0,
    }
    captures = {
        407: {
            "batch_size": 4,
            "graph_signature": signature,
            "records": records,
            "layer_keys": frozenset(range(1, 49)),
        }
    }
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_graph_byte_ab_control", lambda: True
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", 64)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_STATE", state)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_GRAPH_CAPTURES", captures)
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_real_event_marker",
        lambda: "swe_verified:astropy__astropy-12907",
    )
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_graph_compare_records",
        lambda *_args, **_kwargs: {
            "records": 48,
            "layer_keys": set(range(1, 49)),
            "reference_bv": 8,
            "candidate_bv": 64,
        },
    )

    def fail_publish(**_payload):
        raise OSError("read-only artifact directory")

    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_live_pass_emit", fail_publish
    )
    with pytest.raises(OSError, match="read-only artifact"):
        kernel.fixed32_batch_gdn_graph_live_gate_on_replay(
            407, signature, 4, 48
        )
    assert state["status"] == "failed"
    assert 407 in captures


def test_patcher_wires_graph_capture_signature_and_unconditional_replay_hook() -> None:
    source = (ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py").read_text(
        encoding="utf-8"
    )
    assert "fixed32_batch_gdn_graph_live_capture_begin(identity, batch)" in source
    assert "fixed32_batch_gdn_graph_live_capture_end(" in source
    assert "fixed32_batch_gdn_graph_live_gate_on_replay(" in source
    assert "_FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB == \"1\"" in source
    assert "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_BYTE_AB == \"1\"" in source
    assert "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_BYTE_AB == \"1\"" in source
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_BYTE_AB_ENABLED_PATH" in source
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_BYTE_AB_ENABLED_PATH" in source

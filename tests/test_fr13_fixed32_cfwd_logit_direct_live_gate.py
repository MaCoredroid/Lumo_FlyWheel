from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
DEVICE_PATH = ROOT / "scripts/fr13_device_multidraft_cfwd_packed_v3.py"
BASE_PATH = ROOT / "scripts/fr13_device_multidraft_kernel.py"
PATCHER_PATH = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"


def _load_device_module():
    name = "fr13_device_multidraft_kernel_cfwd_live_test"
    spec = importlib.util.spec_from_file_location(name, DEVICE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._base


def test_live_selector_is_strict_and_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_device_module()
    monkeypatch.delenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", raising=False)
    assert module._fr13_cfwd_logit_direct_requested() is False
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    assert module._fr13_cfwd_logit_direct_requested() is True
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "yes")
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        module._fr13_cfwd_logit_direct_requested()


def test_wrapper_preserves_certified_commit_and_reuses_its_uniform_buffer() -> None:
    source = BASE_PATH.read_text(encoding="utf-8")
    start = source.index("def fr13_fixed32_cfwd_logit_direct_commit(")
    stop = source.index(
        "def fr13_fixed32_cfwd_logit_direct_live_prepare_replay(", start
    )
    wrapper = source[start:stop]
    assert "reference = fr13_fixed32_taw_commit(" in wrapper
    assert 'fixed_uniforms = entry["uniforms"]' in wrapper
    assert "reference_decisions = _fr13_fixed32_taw_all_parent_decisions(" in wrapper
    assert "uniforms=fixed_uniforms" in wrapper
    assert "return reference" in wrapper
    assert '"fr13_fixed32_taw_commit",' in source
    source_functions = source[
        source.index("_FR13_FIXED32_TAW_SOURCE_FUNCTIONS = (") :
        source.index("_FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS = (")
    ]
    assert "fr13_fixed32_cfwd_logit_direct_commit" not in source_functions
    kernel_functions = source[
        source.index("_FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS = (") :
        source.index("_FR13_FIXED32_TAW_GEOMETRY = {")
    ]
    assert "_fr13_fixed32_taw_physical_slot_commit_kernel" not in kernel_functions


def test_runtime_hooks_cover_capture_replay_and_authenticated_final_flush() -> None:
    patcher = PATCHER_PATH.read_text(encoding="utf-8")
    warm = patcher.index("fr13_fixed32_cfwd_logit_direct_warm_execute(")
    capture_begin = patcher.index("fr13_fixed32_cfwd_logit_direct_capture_begin(")
    assert warm < capture_begin
    assert "fr13_fixed32_cfwd_logit_direct_capture_begin(" in patcher
    assert "fr13_fixed32_cfwd_logit_direct_capture_end(" in patcher
    assert patcher.count(
        "fr13_fixed32_cfwd_logit_direct_live_prepare_replay("
    ) == 2
    graph_replay = patcher.index("def _fr13_fixed32_observed_graph_replay(")
    drafter_begin = patcher.index("def _fr13_fixed32_drafter_proposal_begin(")
    assert "fr13_fixed32_cfwd_logit_direct_live_prepare_replay(" not in (
        patcher[graph_replay:drafter_begin]
    )
    commit = patcher.index("_fr13_f32_commit_result = (")
    replay_end = patcher.index(
        "fr13_fixed32_cfwd_logit_direct_live_prepare_replay(", commit
    )
    route = patcher.index(
        "_fr13_f32_output = _fr13_fixed32_device_commit_route(", replay_end
    )
    assert commit < replay_end < route
    flush_binding = patcher.index("flush_binding = {")
    finalize = patcher.index(
        "fr13_fixed32_cfwd_logit_direct_live_finalize(", flush_binding
    )
    ack = patcher.index("_fr13_f32_flush_write_ack(", finalize)
    assert flush_binding < finalize < ack
    assert "fr13_fixed32_cfwd_logit_direct_commit(" in patcher


def test_launcher_keeps_gate_off_and_requires_full_graph_native_production() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "FR13_CFWD_LOGIT_DIRECT_BYTE_AB=${FR13_CFWD_LOGIT_DIRECT_BYTE_AB:-0}" in launcher
    assert 'case "$FR13_CFWD_LOGIT_DIRECT_BYTE_AB" in' in launcher
    assert '"$FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION" == "1"' in launcher
    assert '"${ENFORCE_EAGER:-0}" == "0"' in launcher
    assert "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0" in launcher
    assert "/workspace/scripts/fr13_device_multidraft_cfwd_packed_v3.py" in launcher
    assert '-e FR13_CFWD_LOGIT_DIRECT_BYTE_AB="$FR13_CFWD_LOGIT_DIRECT_BYTE_AB"' in launcher


def test_graph_workspace_is_owned_and_sealed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_device_module()
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    device = torch.device("cpu")
    valid_mask = 0x3F
    entry = {
        "mode": "tail6_fixed32",
        "batch_size": 1,
        "child_table": torch.empty(0, device=device),
    }
    monkeypatch.setattr(
        module,
        "_fr13_cfwd_logit_direct_entry",
        lambda mode, batch_size: entry,
    )
    monkeypatch.setattr(
        module,
        "_fr13_fixed32_runtime_contract",
        lambda mode: (object(), valid_mask),
    )
    key = module.fr13_fixed32_taw_cache_key(
        "tail6_fixed32", valid_mask, 1, device
    )
    state = {
        "graph_id": None,
        "mode": "tail6_fixed32",
        "batch_size": 1,
        "device": device,
        "bound_calls": 0,
    }
    module._FR13_CFWD_LOGIT_DIRECT_WARM[key] = state
    module.fr13_fixed32_cfwd_logit_direct_capture_begin(
        17, mode="tail6_fixed32", batch_size=1
    )
    assert module._FR13_CFWD_LOGIT_DIRECT_CAPTURE is state
    assert module._FR13_CFWD_LOGIT_DIRECT_GRAPHS[17] is state
    assert state["graph_id"] == 17
    assert state["bound_calls"] == 0
    state["bound_calls"] = 1
    with pytest.raises(RuntimeError, match="capture binding drift"):
        module.fr13_fixed32_cfwd_logit_direct_capture_end(
            17, mode="tail6_fixed32", batch_size=1
        )
    assert module._FR13_CFWD_LOGIT_DIRECT_CAPTURE is state
    state["bound_calls"] = 0
    module.fr13_fixed32_cfwd_logit_direct_capture_end(
        17, mode="tail6_fixed32", batch_size=1
    )
    assert module._FR13_CFWD_LOGIT_DIRECT_CAPTURE is None
    assert state["bound_calls"] == 1
    with pytest.raises(RuntimeError, match="identity was reused"):
        module.fr13_fixed32_cfwd_logit_direct_capture_begin(
            17, mode="tail6_fixed32", batch_size=1
        )


def test_final_flush_emits_only_zero_mismatch_real_task_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_device_module()
    marker = tmp_path / "real.arm"
    marker.write_text("swe_verified:astropy__astropy-12907\n", encoding="ascii")
    marker.chmod(0o444)
    output = tmp_path / "live.json"
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_REAL_EVENT_PATH", str(marker))
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_REAL_EVENT_UID", str(marker.stat().st_uid))
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_LIVE_JSON", str(output))
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_SOURCE_COMMIT", "a" * 40)
    module._FR13_CFWD_LOGIT_DIRECT_GRAPHS[41] = {
        "graph_id": 41,
        "mode": "tail6_fixed32",
        "batch_size": 1,
        "bound_calls": 1,
        "count_enable": torch.zeros(1, dtype=torch.int32),
        "compared_events": torch.tensor([2], dtype=torch.int64),
        "decision_mismatches": torch.zeros(5, dtype=torch.int64),
        "walk_mismatches": torch.zeros(5, dtype=torch.int64),
        "workspace": {"invalid": torch.zeros(1, dtype=torch.int32)},
    }
    events = [
        {
            "schema": "fr13-fixed32-work-census-v12",
            "event_complete": True,
            "event_index": index,
            "producer_pid": 123,
            "mode": "tail6_fixed32",
            "batch_size": 1,
        }
        for index in range(2)
    ]
    events_sha = hashlib.sha256(
        json.dumps(
            events, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    binding = {
        "action": "final",
        "boundary_snapshot_sha256": "b" * 64,
        "complete_work_census_events": 2,
        "events_sha256": events_sha,
        "generation": 1,
        "nonce": "c" * 64,
        "producer_pid": 123,
    }
    module.fr13_fixed32_cfwd_logit_direct_live_finalize(events, binding)
    record = json.loads(output.read_text(encoding="ascii"))
    assert record["schema"] == "fr13.fixed32.cfwd_logit_direct_live_ab.v2"
    assert record["status"] == "PASS"
    assert record["integration_source_schema"] == (
        module._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SCHEMA
    )
    assert record["integration_source_sha256"] == (
        module._FR13_CFWD_LOGIT_DIRECT_INTEGRATION_SOURCE_SHA256
    )
    assert record["incumbent_source_sha256"] == (
        "6ffe57287e768bfee5e2e72f10de0dfea6fb3e6c0fa50f32b6c099c63fa916a2"
    )
    assert record["counted_graph_replays"] == 2
    assert record["decision_values_compared"] == 162
    assert record["walk_values_compared"] == 102
    assert record["decision_mismatches"] == [0] * 5
    assert record["walk_mismatches"] == [0] * 5
    assert record["candidate_invalid"] == 0
    assert record["served_return"] == "reference all-parent products unchanged"
    assert record["performance_measurement"] is False
    assert record["finalized_by_fixed32_flush"] is True


def test_final_flush_rejects_any_sticky_candidate_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_device_module()
    monkeypatch.setenv("FR13_CFWD_LOGIT_DIRECT_BYTE_AB", "1")
    module._FR13_CFWD_LOGIT_DIRECT_GRAPHS[9] = {
        "graph_id": 9,
        "mode": "hydra27_fixed32",
        "batch_size": 4,
        "bound_calls": 1,
        "count_enable": torch.zeros(1, dtype=torch.int32),
        "compared_events": torch.ones(1, dtype=torch.int64),
        "decision_mismatches": torch.zeros(5, dtype=torch.int64),
        "walk_mismatches": torch.zeros(5, dtype=torch.int64),
        "workspace": {"invalid": torch.ones(1, dtype=torch.int32)},
    }
    events = [
        {
            "schema": "fr13-fixed32-work-census-v12",
            "event_complete": True,
            "event_index": 0,
            "producer_pid": 55,
            "mode": "hydra27_fixed32",
            "batch_size": 4,
        }
    ]
    digest = hashlib.sha256(
        json.dumps(
            events, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(RuntimeError, match="byte comparison failed"):
        module.fr13_fixed32_cfwd_logit_direct_live_finalize(
            events,
            {
                "action": "final",
                "boundary_snapshot_sha256": "d" * 64,
                "complete_work_census_events": 1,
                "events_sha256": digest,
                "generation": 2,
                "nonce": "e" * 64,
                "producer_pid": 55,
            },
        )

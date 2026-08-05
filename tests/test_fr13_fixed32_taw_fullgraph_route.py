from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py").resolve()
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_fullgraph_route_patcher", PATCHER_PATH
)
assert SPEC is not None and SPEC.loader is not None
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)


def _runtime() -> dict[str, object]:
    namespace: dict[str, object] = {
        "_FR13_FIXED32_MODE": "hydra27_fixed32",
        "_FR13_FIXED32_VALID_MASK": 0x7ABDFFFF,
    }
    exec(PATCHER._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE, namespace)
    return namespace


class _FakeTawModule:
    def __init__(self) -> None:
        self.entry = {
            "native_ab_live_gate_pending": False,
            "native_ab_live_pass_emitted": True,
        }
        self.begin_calls = 0
        self.replay_calls = 0

    def _fr13_fixed32_taw_native_live_entry(self, *, mode, batch_size):
        assert (mode, batch_size) == ("hydra27_fixed32", 2)
        return self.entry

    def fr13_fixed32_taw_native_live_gate_begin(self, *, mode, batch_size):
        assert (mode, batch_size) == ("hydra27_fixed32", 2)
        self.begin_calls += 1
        if self.entry["native_ab_live_pass_emitted"]:
            return {"status": "passed", "batch_size": batch_size}
        self.entry["native_ab_live_gate_pending"] = True
        return {"status": "armed", "batch_size": batch_size}

    def fr13_fixed32_taw_native_live_gate_on_replay(
        self, *, mode, batch_size
    ):
        assert (mode, batch_size) == ("hydra27_fixed32", 2)
        self.replay_calls += 1
        if self.entry["native_ab_live_pass_emitted"]:
            return {"status": "passed", "batch_size": batch_size}
        assert self.entry["native_ab_live_gate_pending"] is True
        self.entry["native_ab_live_gate_pending"] = False
        self.entry["native_ab_live_pass_emitted"] = True
        return {"status": "passed", "batch_size": batch_size}


def test_uncaptured_record_cannot_satisfy_first_full_graph_replay() -> None:
    runtime = _runtime()
    module = _FakeTawModule()

    report = runtime["_fr13_fixed32_taw_full_graph_begin"](
        module, "hydra27_fixed32", 2
    )
    assert report == {"status": "armed", "batch_size": 2}
    assert module.entry["native_ab_live_gate_pending"] is True
    assert module.entry["native_ab_live_pass_emitted"] is False

    report = runtime["_fr13_fixed32_taw_full_graph_on_replay"](
        module, "hydra27_fixed32", 2
    )
    assert report == {"status": "passed", "batch_size": 2}
    assert runtime["_FR13_FIXED32_TAW_FULL_GRAPH_PASSES"] == {
        ("hydra27_fixed32", 2)
    }

    report = runtime["_fr13_fixed32_taw_full_graph_begin"](
        module, "hydra27_fixed32", 2
    )
    assert report == {"status": "passed", "batch_size": 2}
    assert module.begin_calls == 2
    assert module.replay_calls == 1


def test_full_graph_begin_fails_closed_on_pending_gate() -> None:
    runtime = _runtime()
    module = _FakeTawModule()
    module.entry["native_ab_live_gate_pending"] = True

    with pytest.raises(RuntimeError, match="already pending"):
        runtime["_fr13_fixed32_taw_full_graph_begin"](
            module, "hydra27_fixed32", 2
        )


def test_b4_runner_rejects_uncaptured_batch_records() -> None:
    runner = Path(
        "scripts/fr13_run_b4_tail23_all_parent_live_gate.sh"
    ).read_text(encoding="utf-8")
    assert 'record.get("evidence_route") != "full_graph_replay"' in runner

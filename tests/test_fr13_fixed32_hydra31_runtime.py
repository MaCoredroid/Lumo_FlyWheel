from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if os.fspath(SCRIPTS) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPTS))

flush = importlib.import_module("fr13_fixed32_flush_protocol")
topology = importlib.import_module("fr13_fixed32_topology")
census = importlib.import_module("fr13_fixed32_work_census")


PATCHER_PATH = SCRIPTS / "fr10_phase4_patch_vllm_tree_gdn.py"
SPEC = importlib.util.spec_from_file_location("hydra31_runtime_patcher", PATCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


def test_hydra31_runtime_identity_preserves_existing_modes() -> None:
    assert topology.VALID_MASK_BY_MODE == {
        "tail6_fixed32": 0x7A9CE73F,
        "hydra27_fixed32": 0x7ABDFFFF,
        "hydra31_fixed32": 0x7FFFFFFF,
    }
    assert tuple(map(sum, topology.VALID_BY_MODE.values())) == (21, 27, 31)
    assert topology.active_choices(topology.HYDRA31_MODE) == (
        topology.FIXED32_CHOICES
    )
    table, counts = topology.sampler_child_table(topology.HYDRA31_MODE)
    assert (len(table), len(table[0])) == (32, 3)
    assert len(counts) == 32
    assert sum(counts) == 31
    assert topology.HYDRA31_RUNTIME_GATE_MANIFEST[
        "fixed_execution_signature_sha256"
    ] == topology.FIXED_EXECUTION_SIGNATURE_SHA256


@pytest.mark.parametrize("batch_size", (1, 4))
def test_hydra31_census_uses_the_same_fixed_physical_work(batch_size: int) -> None:
    raw_events = [
        census.reference_event(mode, batch_size, f"{mode}-b{batch_size}")
        for mode in (
            census.TAIL_MODE,
            census.HYDRA_MODE,
            census.HYDRA31_MODE,
        )
    ]
    events = [
        census.validate_event(
            raw,
            source=f"{raw['mode']}-b{batch_size}",
        )
        for raw in raw_events
    ]

    assert census.MODE_SEMANTICS[census.HYDRA31_MODE] == {
        "active_nodes": 31,
        "valid_mask": 0x7FFFFFFF,
    }
    assert events[0].normalized_work == events[1].normalized_work
    assert events[1].normalized_work == events[2].normalized_work
    assert raw_events[2]["verify_rows"] == 32 * batch_size
    assert raw_events[2]["physical_drafts"] == 31


def _write_gate(path: Path, payload: str | None = None) -> None:
    path.write_text(
        (payload if payload is not None else topology.hydra31_runtime_gate_json())
        + "\n",
        encoding="ascii",
    )
    path.chmod(0o400)


def _set_gate_env(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("FR13_HYDRA31_ENABLE", "1")
    monkeypatch.setenv("FR13_HYDRA31_GATE_SIDECAR", os.fspath(path))
    monkeypatch.setattr(patcher, "_FR13_HYDRA31_GATE_SIDECAR", path)


def test_hydra31_runtime_gate_accepts_only_canonical_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "hydra31.json"
    _write_gate(sidecar)
    _set_gate_env(monkeypatch, sidecar)

    patcher._fr13_fixed32_validate_hydra31_gate("hydra31_fixed32")

    sidecar.chmod(0o600)
    with pytest.raises(RuntimeError, match="mode-400 regular file"):
        patcher._fr13_fixed32_validate_hydra31_gate("hydra31_fixed32")
    _write_gate(sidecar, '{"schema":"tampered"}')
    with pytest.raises(RuntimeError, match="content mismatch"):
        patcher._fr13_fixed32_validate_hydra31_gate("hydra31_fixed32")


def test_hydra31_runtime_gate_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sidecar = tmp_path / "missing.json"
    monkeypatch.setenv("FR13_HYDRA31_ENABLE", "0")
    monkeypatch.setenv("FR13_HYDRA31_GATE_SIDECAR", "")
    monkeypatch.setattr(patcher, "_FR13_HYDRA31_GATE_SIDECAR", sidecar)

    with pytest.raises(RuntimeError, match="requires FR13_HYDRA31_ENABLE=1"):
        patcher._fr13_fixed32_validate_hydra31_gate("hydra31_fixed32")

    monkeypatch.setenv("FR13_HYDRA31_ENABLE", "1")
    monkeypatch.setenv("FR13_HYDRA31_GATE_SIDECAR", os.fspath(sidecar))
    with pytest.raises(RuntimeError, match="sidecar is unavailable"):
        patcher._fr13_fixed32_validate_hydra31_gate("hydra31_fixed32")

    monkeypatch.setenv("FR13_HYDRA31_GATE_SIDECAR", "")
    with pytest.raises(RuntimeError, match="forbidden outside"):
        patcher._fr13_fixed32_validate_hydra31_gate("hydra27_fixed32")


def test_runtime_and_launcher_registries_are_complete() -> None:
    serve = (SCRIPTS / "fr13_bigdenom_swe_serve_variant.sh").read_text(
        encoding="utf-8"
    )
    launcher = (SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    runner = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")

    assert patcher._FR13_FIXED32_MODES["hydra31_fixed32"] == (0x7FFFFFFF, 31)
    assert "hydra31_fixed32)" in serve
    assert "hydra31_fixed32 is default-off" in serve
    assert "FR13_HYDRA31_ENABLE=1" in serve
    assert "FR13_FIXED32_MODE=hydra31_fixed32" in serve
    assert "fr13_fixed32_hydra31_runtime_gate.json" in launcher
    assert "os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW" in launcher
    assert "hydra31_fixed32" in runner
    assert "hydra31_fixed32" in flush.FIXED32_MODES

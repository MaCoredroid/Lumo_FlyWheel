from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fr13_depth_acceptance as depth_acceptance  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402


DIAGNOSTIC = REPO / "config/fr13_fixed32/subset_b1_diagnostic_one.json"
EXACT4 = REPO / "config/fr13_fixed32/subset_b4_four.json"


def test_b1_diagnostic_subset_is_pinned_and_not_formal_evidence() -> None:
    binding = floor_gate.validate_fixed32_run_subset(
        DIAGNOSTIC,
        b1_diagnostic=True,
    )

    assert binding == {
        "task_count": 1,
        "path": str(DIAGNOSTIC),
        "sha256": floor_gate.B1_DIAGNOSTIC_SUBSET["sha256"],
        "task_ids": ["astropy__astropy-12907"],
        "run_classification": "b1_diagnostic",
        "gate_eligible": False,
        "floor_acceptance_eligible": False,
    }
    assert hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest() == binding["sha256"]

    with pytest.raises(
        floor_gate.GateError,
        match="not canonical exact4/exact16",
    ):
        floor_gate.validate_canonical_subset(DIAGNOSTIC)


def test_b1_diagnostic_mode_cannot_consume_exact4_or_tampered_bytes(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        floor_gate.GateError,
        match="B1 diagnostic subset SHA-256 mismatch",
    ):
        floor_gate.validate_fixed32_run_subset(EXACT4, b1_diagnostic=True)

    tampered = tmp_path / DIAGNOSTIC.name
    tampered.write_bytes(DIAGNOSTIC.read_bytes() + b"\n")
    with pytest.raises(
        floor_gate.GateError,
        match="B1 diagnostic subset SHA-256 mismatch",
    ):
        floor_gate.validate_fixed32_run_subset(tampered, b1_diagnostic=True)


def test_formal_exact4_binding_is_unchanged() -> None:
    assert floor_gate.validate_fixed32_run_subset(
        EXACT4,
        b1_diagnostic=False,
    ) == floor_gate.validate_canonical_subset(EXACT4)


def test_acceptance_reducer_rejects_b1_diagnostic_binding() -> None:
    binding = floor_gate.validate_fixed32_run_subset(
        DIAGNOSTIC,
        b1_diagnostic=True,
    )
    reducer_binding = {
        key: binding[key]
        for key in depth_acceptance.FIXED32_FLOOR_SUBSET_KEYS
    }

    with pytest.raises(ValueError, match="canonical exact4 subset binding differs"):
        depth_acceptance.validate_canonical_subset_binding(
            reducer_binding,
            required_task_count=4,
            label="B1 diagnostic exclusion",
        )


def test_b1_diagnostic_is_guarded_across_runtime_ingress() -> None:
    serve = (REPO / "scripts/fr13_bigdenom_swe_serve_variant.sh").read_text()
    launcher = (REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh").read_text()
    offload = (
        REPO / "scripts/swe_x86_helpers/offload_codex_proxy.sh"
    ).read_text()
    runner = (REPO / "scripts/run_swe_bench_q36_a.py").read_text()

    assert serve.count("validate_fixed32_run_subset") >= 3
    assert "MAX_NUM_SEQS_OVR=1 and SWE_CONCURRENCY=1" in serve
    assert '"gate_eligible": False' in serve
    assert '"timing_eligible": False' in serve
    assert "fixed32 B1 diagnostic requires MAX_NUM_SEQS=1" in launcher
    assert "fixed32 B1 diagnostic ingress task ID is not pinned" in launcher
    assert "fixed32 B1 diagnostic offload task ID is not pinned" in offload
    assert "fixed32 B1 diagnostic proxy-control task ID is not pinned" in offload
    assert "fixed32 B1 diagnostic requires concurrency and serving batch exactly 1" in runner
    assert "fixed32 TAW native campaign arm requires exact B4 concurrency" in runner
    assert "--fixed32-taw-real-event-arm" in serve
    assert "fixed32 TAW native campaign arm requires exact B4 concurrency" in serve
    assert '"gate_eligible": False' in runner


def test_b1_diagnostic_can_time_an_authenticated_streamk_production_arm() -> None:
    launcher = (
        REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")
    timing = (
        REPO / "scripts/fr13_run_b1_cutlass_streamk_timing.sh"
    ).read_text(encoding="utf-8")

    assert "CUTLASS Stream-K production requires fixed32 B1 and a pinned live PASS" in launcher
    assert 'case "${FR13_FIXED32_B1_DIAGNOSTIC:-0}" in' in launcher
    assert "B1_DIAGNOSTIC=1" in timing
    assert "TIMING_ELIGIBLE=0" in timing
    assert "floor_acceptance_eligible=0" in timing

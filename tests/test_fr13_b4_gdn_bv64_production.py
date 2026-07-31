from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PASS_SCRIPT = ROOT / "scripts" / "fr13_b4_gdn_bv64_pass.py"
KERNEL_SOURCE = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
RUNNER = ROOT / "scripts/fr13_run_b4_gdn_bv64_timing.sh"
GATE_RUNNER = ROOT / "scripts/fr13_run_b4_gdn_wide_live_gate.sh"


def _module():
    spec = importlib.util.spec_from_file_location("fr13_b4_gdn_bv64_pass", PASS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_payload() -> dict[str, object]:
    source_sha256 = hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()
    return {
        "schema": "fr13.fixed32.batch_gdn.graph_live_pass.v1",
        "status": "pass",
        "task_marker": "swe_verified:astropy__astropy-12907",
        "batch": 4,
        "layer_count": 48,
        "layer_keys": [f"0x{index:x}" for index in range(1, 49)],
        "reference_always_served": True,
        "candidate": "fixed32_batch_gdn_bv_v2",
        "source_sha256": source_sha256,
        "mode": "tail6_fixed32",
        "physical_rows_per_request": 32,
        "reference_bv": 8,
        "candidate_bv": 64,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "compared_byte_surfaces": [
            "out",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "state_export_compact",
            "state_export_untouched_tail",
            "flags",
            "invocation_counter",
        ],
        "raw_byte_equal": True,
        "state_restored": True,
        "gate_mode": "post_replay_shadow",
        "graph_id": 407,
        "graph_signature": "a" * 64,
        "capture_records": 48,
        "real_task_authenticated": True,
        "graph_baseline_byte_equal": True,
    }


def _write_payload(path: Path, payload: dict[str, object]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _valid_verdict(
    live_payload: dict[str, object], live_sha256: str
) -> dict[str, object]:
    return {
        "schema": "fr13.fixed32.batch_gdn.b4_diagnostic.v1",
        "status": "pass",
        "run_classification": "exact4_b4_graph_byte_diagnostic",
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "subset_sha256": (
            "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
        ),
        "task_ids": [
            "astropy__astropy-12907",
            "astropy__astropy-13033",
            "astropy__astropy-13236",
            "astropy__astropy-13398",
        ],
        "task_marker": live_payload["task_marker"],
        "gate_mode": "post_replay_shadow",
        "graph_id": live_payload["graph_id"],
        "graph_signature": live_payload["graph_signature"],
        "candidate_bv": 64,
        "b4_layer_passes": 48,
        "observed_pass_layers_by_batch": {"2": 0, "3": 0, "4": 48},
        "engine_ledger_chain_head_sha256": "d" * 64,
        "graph_live_pass_sha256": live_sha256,
        "kernel_source_sha256": live_payload["source_sha256"],
        "raw_byte_equal": True,
        "reference_always_served": True,
        "production_default_enabled": False,
    }


def test_graph_pass_is_source_bound_and_installed_read_only(tmp_path: Path) -> None:
    module = _module()
    live_payload = _valid_payload()
    live = tmp_path / "graph.pass.json"
    expected_sha256 = _write_payload(live, live_payload)
    verdict = tmp_path / "gate.verdict.json"
    verdict_sha256 = _write_payload(
        verdict, _valid_verdict(live_payload, expected_sha256)
    )
    out_dir = tmp_path / "logs"
    out_dir.mkdir()
    out = out_dir / "fr13_fixed32_batch_gdn_byte_ab.pass.json"

    summary = module.install_pass(
        live_result=live,
        expected_live_sha256=expected_sha256,
        gate_verdict=verdict,
        expected_gate_verdict_sha256=verdict_sha256,
        kernel_source=KERNEL_SOURCE,
        out=out,
    )

    assert summary["candidate_bv"] == 64
    assert summary["layer_count"] == 48
    assert summary["live_result_sha256"] == expected_sha256
    assert out.read_bytes() == live.read_bytes()
    assert stat.S_IMODE(out.stat().st_mode) == 0o400
    with pytest.raises(ValueError, match="refusing to replace"):
        module.install_pass(
            live_result=live,
            expected_live_sha256=expected_sha256,
            gate_verdict=verdict,
            expected_gate_verdict_sha256=verdict_sha256,
            kernel_source=KERNEL_SOURCE,
            out=out,
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("schema", "fr13.fixed32.batch_gdn.live_pass.v2", "schema"),
        ("status", "rejected", "status"),
        ("batch", 3, "batch"),
        ("candidate_bv", 32, "candidate_bv"),
        ("mode", "hydra27_fixed32", "mode"),
        ("source_sha256", "b" * 64, "source_sha256"),
        ("reference_always_served", False, "reference_always_served"),
        ("graph_baseline_byte_equal", False, "graph_baseline_byte_equal"),
        ("state_restored", False, "state_restored"),
    ),
)
def test_graph_pass_rejects_nonqualifying_or_failed_gate(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    module = _module()
    payload = _valid_payload()
    payload[key] = value
    live = tmp_path / "graph.pass.json"
    expected_sha256 = _write_payload(live, payload)
    verdict = tmp_path / "gate.verdict.json"
    verdict_sha256 = _write_payload(
        verdict, _valid_verdict(payload, expected_sha256)
    )

    with pytest.raises(ValueError, match=message):
        module.validate_file(
            live_result=live,
            expected_live_sha256=expected_sha256,
            gate_verdict=verdict,
            expected_gate_verdict_sha256=verdict_sha256,
            kernel_source=KERNEL_SOURCE,
        )


def test_graph_pass_rejects_missing_wrong_sha_and_symlink(tmp_path: Path) -> None:
    module = _module()
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="cannot securely open"):
        module.validate_file(
            live_result=missing,
            expected_live_sha256="a" * 64,
            gate_verdict=missing,
            expected_gate_verdict_sha256="a" * 64,
            kernel_source=KERNEL_SOURCE,
        )

    live = tmp_path / "graph.pass.json"
    live_payload = _valid_payload()
    expected_sha256 = _write_payload(live, live_payload)
    verdict = tmp_path / "gate.verdict.json"
    verdict_sha256 = _write_payload(
        verdict, _valid_verdict(live_payload, expected_sha256)
    )
    with pytest.raises(ValueError, match="raw SHA-256 mismatch"):
        module.validate_file(
            live_result=live,
            expected_live_sha256="b" * 64,
            gate_verdict=verdict,
            expected_gate_verdict_sha256=verdict_sha256,
            kernel_source=KERNEL_SOURCE,
        )
    alias = tmp_path / "alias.json"
    alias.symlink_to(live)
    with pytest.raises(ValueError, match="securely open"):
        module.validate_file(
            live_result=alias,
            expected_live_sha256=expected_sha256,
            gate_verdict=verdict,
            expected_gate_verdict_sha256=verdict_sha256,
            kernel_source=KERNEL_SOURCE,
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("status", "rejected", "status"),
        ("task_ids", ["astropy__astropy-12907"], "task_ids"),
        ("subset_sha256", "e" * 64, "subset_sha256"),
        ("graph_live_pass_sha256", "e" * 64, "graph_live_pass_sha256"),
        ("kernel_source_sha256", "e" * 64, "kernel_source_sha256"),
        ("b4_layer_passes", 47, "b4_layer_passes"),
    ),
)
def test_graph_gate_verdict_must_prove_completed_exact4(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    module = _module()
    live_payload = _valid_payload()
    live = tmp_path / "graph.pass.json"
    live_sha256 = _write_payload(live, live_payload)
    verdict_payload = _valid_verdict(live_payload, live_sha256)
    verdict_payload[key] = value
    verdict = tmp_path / "gate.verdict.json"
    verdict_sha256 = _write_payload(verdict, verdict_payload)

    with pytest.raises(ValueError, match=message):
        module.validate_file(
            live_result=live,
            expected_live_sha256=live_sha256,
            gate_verdict=verdict,
            expected_gate_verdict_sha256=verdict_sha256,
            kernel_source=KERNEL_SOURCE,
        )


def test_launcher_restricts_wide_production_to_exact_b4_bv64_graph_pass() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    for needle in (
        '"$_fr13_batch_gdn_bv_production" == "64"',
        '"$MAX_NUM_SEQS" == "4"',
        '"${FR13_FIXED32_MODE:-}" == "tail6_fixed32"',
        '"${ENFORCE_EAGER:-0}" == "0"',
        '"${CUDAGRAPH_MODE:-}" == "FULL_AND_PIECEWISE"',
        "FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_JSON",
        "FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_SHA256",
        "FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_JSON",
        "FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_SHA256",
        "scripts/fr13_b4_gdn_bv64_pass.py install",
        "fr13_fixed32_batch_gdn_production.arm",
    ):
        assert needle in launcher
    assert launcher.index("scripts/fr13_b4_gdn_bv64_pass.py install") < launcher.index(
        'echo "1" > "$LOG_DIR/fr13_fixed32_batch_gdn_production.arm"'
    )
    assert launcher.index("scripts/fr13_b4_gdn_bv64_pass.py install") < launcher.index(
        "docker run"
    )
    assert (
        'if [[ "$_fr13_batch_gdn_production" != "1"' in launcher
    )


def test_timing_runner_is_exact4_full_wall_stock_first_and_floor_ineligible() -> None:
    runner = RUNNER.read_text(encoding="utf-8")

    for needle in (
        "config/fr13_fixed32/subset_b4_four.json",
        "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5",
        "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d",
        "export BSIZE=4",
        "export CONC=4",
        "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4",
        "FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1",
        "FR13_FIXED32_BATCH_GDN_BYTE_AB=0",
        "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0",
        'FR13_FIXED32_BATCH_GDN_PRODUCTION="$production"',
        'FR13_FIXED32_BATCH_GDN_BV_PRODUCTION="$production_bv"',
        'FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_JSON="$verdict_json"',
        "FR13_FIXED32_CUTLASS_WAVE=stock",
        "FR13_DFWD_UNIFIED_BM8_PRODUCTION=0",
        "FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0",
        '"decision_metric": "measured_tps_fullstep_wall"',
        '"step_wall_to_optimistic_floor_ratio"',
        '"optimistic_floor_is_full_step_hardware_floor": False',
        '"formal_floor_acceptance_eligible": False',
        "timing_eligible=0",
        "production_default_enabled=0",
    ):
        assert needle in runner
    preflight = runner.index("scripts/fr13_b4_gdn_bv64_pass.py validate")
    first_docker = runner.index("docker ps -aq")
    assert preflight < first_docker
    assert runner.index('run_arm "$STOCK_ARM" 0') < runner.index(
        'run_arm "$CANDIDATE_ARM" 1'
    )
    assert "scripts/fr13_floor_gate.py" not in runner


def test_graph_gate_verdict_is_exact4_and_binds_the_live_pass() -> None:
    gate_runner = GATE_RUNNER.read_text(encoding="utf-8")
    assert '"run_classification": "exact4_b4_graph_byte_diagnostic"' in gate_runner
    assert '"graph_live_pass_sha256": pass_sha256' in gate_runner
    assert '"kernel_source_sha256": source_sha256' in gate_runner


def test_pass_attestor_is_in_the_fixed32_runtime_manifest() -> None:
    manifest = (ROOT / "scripts/fr13_runtime_manifest.py").read_text(
        encoding="utf-8"
    )
    assert '"scripts/fr13_b4_gdn_bv64_pass.py"' in manifest

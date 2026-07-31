from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PASS_SCRIPT = ROOT / "scripts/fr13_b4_gdn_bv8_pass.py"
KERNEL_SOURCE = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
RUNNER = ROOT / "scripts/fr13_run_b4_gdn_bv8_timing.sh"
GATE_RUNNER = ROOT / "scripts/fr13_run_b4_gdn_wide_live_gate.sh"
PATCHER = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"

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


def _module():
    spec = importlib.util.spec_from_file_location(
        "fr13_b4_gdn_bv8_pass", PASS_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_payload(path: Path, payload: dict[str, object]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _write_runtime_manifest(path: Path) -> tuple[str, str]:
    gate_raw = GATE_RUNNER.read_bytes()
    timing_raw = RUNNER.read_bytes()
    gate_sha256 = hashlib.sha256(gate_raw).hexdigest()
    unsigned = {
        "canonical_format": "utf8-json-sort-keys-compact-v1",
        "closures": {
            "host_script_source": [
                {
                    "path": "scripts/fr13_run_b4_gdn_bv8_timing.sh",
                    "sha256": hashlib.sha256(timing_raw).hexdigest(),
                    "size": len(timing_raw),
                },
                {
                    "path": "scripts/fr13_run_b4_gdn_wide_live_gate.sh",
                    "sha256": gate_sha256,
                    "size": len(gate_raw),
                },
            ]
        },
        "profile": "fixed32",
        "required_absence": [],
        "schema": "fr13-runtime-manifest-v1",
        "sequence": "scripts/fr13_fixed32_floor_timers_seq.sh",
        "summary": {"file_count": 2},
    }
    digest = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(
            {**unsigned, "overall_canonical_sha256": digest},
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return digest, gate_sha256


def _valid_live() -> dict[str, object]:
    return {
        "schema": "fr13.fixed32.batch_gdn.graph_live_pass.v1",
        "status": "pass",
        "task_marker": "swe_verified:astropy__astropy-12907",
        "batch": 4,
        "layer_count": 48,
        "layer_keys": [f"0x{index:x}" for index in range(1, 49)],
        "reference_always_served": True,
        "candidate": "fixed32_batch_gdn_bv8_v1",
        "source_sha256": hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
        "mode": "tail6_fixed32",
        "physical_rows_per_request": 32,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "count_invocation": True,
        "ring_export": True,
        "flags_inkernel": True,
        "scan_align": False,
        "npad_invariant": False,
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
        "reference_kernel_structure": "per_request_tree_gdn_path",
        "candidate_kernel_structure": "fixed32_batch_tree_gdn_path",
        "production_eligible": True,
        "gate_mode": "post_replay_shadow",
        "graph_id": 407,
        "graph_signature": "a" * 64,
        "capture_records": 48,
        "real_task_authenticated": True,
        "graph_baseline_byte_equal": True,
    }


def _valid_verdict(
    live_payload: dict[str, object],
    live_sha256: str,
    runtime_manifest_sha256: str,
    gate_runner_sha256: str,
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
        "candidate": "fixed32_batch_gdn_bv8_v1",
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_kernel_structure": "per_request_tree_gdn_path",
        "candidate_kernel_structure": "fixed32_batch_tree_gdn_path",
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "count_invocation": True,
        "ring_export": True,
        "flags_inkernel": True,
        "scan_align": False,
        "npad_invariant": False,
        "tree_gdn_geom_override": "BV=8",
        "enforce_eager": 0,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "production_eligible": True,
        "b4_layer_passes": 48,
        "observed_pass_layers_by_batch": {"2": 0, "3": 0, "4": 48},
        "engine_ledger_chain_head_sha256": "d" * 64,
        "graph_live_pass_sha256": live_sha256,
        "kernel_source_sha256": live_payload["source_sha256"],
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "gate_runner_sha256": gate_runner_sha256,
        "raw_byte_equal": True,
        "reference_always_served": True,
        "production_default_enabled": False,
    }


def _install_valid_credential(tmp_path: Path):
    module = _module()
    live_payload = _valid_live()
    live = tmp_path / "graph.pass.json"
    live_sha256 = _write_payload(live, live_payload)
    runtime_manifest = tmp_path / "runtime_manifest.json"
    runtime_manifest_sha256, gate_runner_sha256 = _write_runtime_manifest(
        runtime_manifest
    )
    verdict = tmp_path / "gate.verdict.json"
    verdict_sha256 = _write_payload(
        verdict,
        _valid_verdict(
            live_payload,
            live_sha256,
            runtime_manifest_sha256,
            gate_runner_sha256,
        ),
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    sidecar = logs / "fr13_fixed32_batch_gdn_byte_ab.pass.json"
    summary = module.install_pass(
        live_result=live,
        expected_live_sha256=live_sha256,
        gate_verdict=verdict,
        expected_gate_verdict_sha256=verdict_sha256,
        kernel_source=KERNEL_SOURCE,
        runtime_manifest=runtime_manifest,
        gate_runner=GATE_RUNNER,
        out=sidecar,
    )
    return module, sidecar, live_sha256, verdict_sha256, summary


def test_exact4_artifacts_install_one_source_bound_read_only_credential(
    tmp_path: Path,
) -> None:
    module, sidecar, live_sha256, verdict_sha256, summary = (
        _install_valid_credential(tmp_path)
    )
    credential = json.loads(sidecar.read_text(encoding="ascii"))

    assert credential["schema"] == (
        "fr13.fixed32.batch_gdn.bv8.production_sidecar.v1"
    )
    assert credential["candidate"] == "fixed32_batch_gdn_bv8_v1"
    assert credential["live_result_sha256"] == live_sha256
    assert credential["gate_verdict_sha256"] == verdict_sha256
    assert credential["runtime_manifest_sha256"] == summary[
        "runtime_manifest_sha256"
    ]
    assert credential["gate_runner_sha256"] == summary["gate_runner_sha256"]
    assert credential["live_result"]["candidate_physical_launches_per_layer"] == 2
    assert credential["gate_verdict"]["task_ids"] == list(
        module.EXPECTED_TASK_IDS
    )
    assert summary["production_sidecar_sha256"] == hashlib.sha256(
        sidecar.read_bytes()
    ).hexdigest()
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o400


@pytest.mark.parametrize(
    ("artifact", "key", "value", "message"),
    (
        ("live", "candidate", "fixed32_batch_gdn_bv_v2", "candidate"),
        ("live", "candidate_kernel_structure", "per_request_tree_gdn_path", "structure"),
        ("live", "candidate_physical_launches_per_layer", 8, "launches"),
        ("live", "production_eligible", False, "production_eligible"),
        ("verdict", "task_ids", ["astropy__astropy-12907"], "task_ids"),
        ("verdict", "graph_live_pass_sha256", "e" * 64, "graph_live_pass_sha256"),
        ("verdict", "b4_layer_passes", 47, "b4_layer_passes"),
        ("verdict", "production_eligible", False, "production_eligible"),
        ("verdict", "runtime_manifest_sha256", "e" * 64, "runtime_manifest_sha256"),
    ),
)
def test_credential_rejects_structure_launch_task_or_hash_drift(
    tmp_path: Path,
    artifact: str,
    key: str,
    value: object,
    message: str,
) -> None:
    module = _module()
    live_payload = _valid_live()
    if artifact == "live":
        live_payload[key] = value
    live = tmp_path / "graph.pass.json"
    live_sha256 = _write_payload(live, live_payload)
    runtime_manifest = tmp_path / "runtime_manifest.json"
    runtime_manifest_sha256, gate_runner_sha256 = _write_runtime_manifest(
        runtime_manifest
    )
    verdict_payload = _valid_verdict(
        live_payload,
        live_sha256,
        runtime_manifest_sha256,
        gate_runner_sha256,
    )
    if artifact == "verdict":
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
            runtime_manifest=runtime_manifest,
            gate_runner=GATE_RUNNER,
        )


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("manifest", "runtime manifest canonical digest mismatch"),
        ("gate_runner", "gate runner differs"),
    ),
)
def test_credential_rejects_current_runtime_closure_drift(
    tmp_path: Path, drift: str, message: str
) -> None:
    module, _sidecar, live_sha256, verdict_sha256, _summary = (
        _install_valid_credential(tmp_path)
    )
    runtime_manifest = tmp_path / "runtime_manifest.json"
    gate_runner = GATE_RUNNER
    if drift == "manifest":
        payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        payload["overall_canonical_sha256"] = "0" * 64
        runtime_manifest.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    else:
        gate_runner = tmp_path / "drifted_gate.sh"
        gate_runner.write_bytes(GATE_RUNNER.read_bytes() + b"# drift\n")
    with pytest.raises(ValueError, match=message):
        module.validate_file(
            live_result=tmp_path / "graph.pass.json",
            expected_live_sha256=live_sha256,
            gate_verdict=tmp_path / "gate.verdict.json",
            expected_gate_verdict_sha256=verdict_sha256,
            kernel_source=KERNEL_SOURCE,
            runtime_manifest=runtime_manifest,
            gate_runner=gate_runner,
        )


def test_kernel_credential_routes_only_b4_batched_bv8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _module_value, sidecar, live_sha256, verdict_sha256, _summary = (
        _install_valid_credential(tmp_path)
    )
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(sidecar))
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_PRODUCTION", "1")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", 8)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", None)
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_byte_ab_control", lambda: (False, None)
    )
    monkeypatch.setattr(
        kernel, "_fr13_fixed32_batch_gdn_graph_byte_ab_control", lambda: False
    )

    payload = kernel._fr13_fixed32_batch_gdn_production_control()
    assert payload is not None
    assert payload["candidate"] == "fixed32_batch_gdn_bv8_v1"
    assert payload["_graph_pass_sha256"] == live_sha256
    assert payload["_gate_verdict_sha256"] == verdict_sha256
    assert kernel.fixed32_batch_gdn_selector(1) is None
    assert kernel.fixed32_batch_gdn_selector(2) is None
    assert kernel.fixed32_batch_gdn_selector(3) is None
    assert kernel.fixed32_batch_gdn_selector(4) == "production"


def test_bv8_engagement_publishes_only_after_b4_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, sidecar, live_sha256, verdict_sha256, summary = (
        _install_valid_credential(tmp_path)
    )
    engagement_path = tmp_path / "bv8.engagement.json"
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_BYTE_AB_PASS_PATH", str(sidecar))
    monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_PRODUCTION", "1")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "hydra27_fixed32")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", 8)
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURE_CONTEXT", None
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_CAPTURES", {}
    )
    monkeypatch.setattr(
        kernel, "_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED", False
    )
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_ENGAGEMENT",
        str(engagement_path),
    )
    monkeypatch.setattr(kernel.torch.cuda, "is_available", lambda: False)

    for batch in (4, 3, 2, 1):
        graph_id = 100 + batch
        signature = str(batch) * 64
        kernel.fixed32_batch_gdn_bv8_production_capture_begin(graph_id, batch)
        if batch == 4:
            for layer_key in range(1, 49):
                kernel._fr13_fixed32_batch_gdn_bv8_production_capture_register(
                    batch_size=4, layer_key=layer_key, candidate_bv=8
                )
        kernel.fixed32_batch_gdn_bv8_production_capture_end(
            graph_id, batch, signature, 48 * batch
        )

    lower = kernel.fixed32_batch_gdn_bv8_production_replay_engaged(
        102, 2, "2" * 64, 96
    )
    assert lower == {
        "status": "legacy_lower_batch",
        "batch_size": 2,
        "batched_route_capture_layers": 0,
    }
    assert not engagement_path.exists()
    report = kernel.fixed32_batch_gdn_bv8_production_replay_engaged(
        104, 4, "4" * 64, 192
    )
    assert report["status"] == "ENGAGED"
    assert report["candidate"] == "fixed32_batch_gdn_bv8_v1"
    assert report["candidate_bv"] == 8
    assert report["mode"] == "hydra27_fixed32"
    assert report["candidate_physical_launches_per_layer"] == 2
    assert report["count_invocation"] is True
    assert report["ring_export"] is True
    assert report["flags_inkernel"] is True
    assert report["scan_align"] is False
    assert report["npad_invariant"] is False
    assert report["batched_route_capture_layers_by_batch"] == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 48,
    }
    assert report["qualified_batch_sizes"] == [4]
    assert report["lower_batch_route"] == "legacy_per_request_bv8"
    assert report["physical_launches_per_layer_by_batch"] == {
        "1": 2,
        "2": 4,
        "3": 6,
        "4": 2,
    }
    assert report["all_b_le_4_launch_invariant"] is False
    assert report["graph_pass_sha256"] == live_sha256
    assert report["gate_verdict_sha256"] == verdict_sha256
    assert report["runtime_manifest_sha256"] == summary[
        "runtime_manifest_sha256"
    ]
    assert report["gate_runner_sha256"] == summary["gate_runner_sha256"]
    assert report["production_sidecar_sha256"] == summary[
        "production_sidecar_sha256"
    ]
    assert stat.S_IMODE(engagement_path.stat().st_mode) == 0o444
    validated, _raw = module.validate_engagement_file(
        engagement=engagement_path,
        expected_live_sha256=live_sha256,
        expected_gate_verdict_sha256=verdict_sha256,
        expected_production_sidecar_sha256=summary[
            "production_sidecar_sha256"
        ],
        expected_runtime_manifest_sha256=summary[
            "runtime_manifest_sha256"
        ],
        expected_gate_runner_sha256=summary["gate_runner_sha256"],
        kernel_source=KERNEL_SOURCE,
    )
    assert validated["status"] == "ENGAGED"
    assert validated["lower_batch_batched_capture_layers"] == 0


def test_launcher_installs_bv8_credential_before_arm_and_docker() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for needle in (
        '""|8|64)',
        '"$_fr13_batch_gdn_bv_production" == "8"',
        "scripts/fr13_b4_gdn_bv8_pass.py install",
        "FR13_FIXED32_BATCH_GDN_GRAPH_LIVE_PASS_SHA256",
        "FR13_FIXED32_BATCH_GDN_GRAPH_GATE_VERDICT_SHA256",
        "fr13_fixed32_batch_gdn_bv8.production_engagement.json",
        "fr13_fixed32_batch_gdn_production.arm",
    ):
        assert needle in launcher
    install = launcher.index("scripts/fr13_b4_gdn_bv8_pass.py install")
    assert install < launcher.index(
        'echo "1" > "$LOG_DIR/fr13_fixed32_batch_gdn_production.arm"'
    )
    assert install < launcher.index("docker run")


def test_timing_runner_is_stock_first_exact4_full_wall_and_fail_closed() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    for needle in (
        "config/fr13_fixed32/subset_b4_four.json",
        "export BSIZE=4",
        "export CONC=4",
        "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4",
        "B4_KV_CACHE_MEMORY_BYTES=42949672960",
        "production_bv=8",
        "TIMING_KIND=hydra27_fixed32",
        'STOCK_ARM="hydra27_fixed32_stock_${TAG}"',
        'CANDIDATE_ARM="hydra27_fixed32_gdn_bv8_${TAG}"',
        "arm=hydra27_fixed32",
        "lineage=successor_to_legacy_hydra23_not_same_topology",
        "valid_mask=0x7abdffff",
        "scripts/fr13_b4_gdn_bv8_pass.py validate",
        "scripts/fr13_b4_gdn_bv8_pass.py engagement",
        'FR13_FIXED32_BATCH_GDN_PRODUCTION="$production"',
        'FR13_FIXED32_BATCH_GDN_BV_PRODUCTION="$production_bv"',
        'FR13_FIXED32_BATCH_GDN_RUNTIME_MANIFEST_JSON="$runtime_manifest"',
        'FR13_FIXED32_BATCH_GDN_GATE_RUNNER="$gate_runner"',
        "FR13_FIXED32_BATCH_GDN_BV8_TIMING=1",
        '"decision_metric": "measured_tps_fullstep_wall"',
        '"formal_floor_acceptance_eligible": False',
        "fr13_fixed32_batch_gdn_bv8.production_engagement.json",
        'engagement.get("candidate") != "fixed32_batch_gdn_bv8_v1"',
        '"candidate_physical_launches_per_layer") != 2',
        '"lower_batch_batched_capture_layers"',
        '"all_b_le_4_launch_invariant": False',
    ):
        assert needle in runner
    manifest = runner.index("scripts/fr13_runtime_manifest.py")
    validate = runner.index("scripts/fr13_b4_gdn_bv8_pass.py validate")
    docker = runner.index("docker ps -aq")
    assert manifest < validate < docker
    assert runner.index('run_arm "$STOCK_ARM" 0') < runner.index(
        'run_arm "$CANDIDATE_ARM" 1'
    )
    assert "scripts/fr13_floor_gate.py" not in runner


def test_gate_and_timing_pin_the_same_bv8_kernel_specialization() -> None:
    gate = GATE_RUNNER.read_text(encoding="utf-8")
    timing = RUNNER.read_text(encoding="utf-8")
    shared = (
        "FR10_METRICS=1 ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
        "FR13_RING_EXPORT=1 FR13_FLAGS_INKERNEL=1",
        "FR13_TREE_GDN_GEOM_OVERRIDE=BV=8",
        "FR13_SCAN_ALIGN=0 FR13_NPAD_INVARIANT=0",
    )
    for assignment in shared:
        assert assignment in gate
        assert assignment in timing
    assert "FR10_METRICS=0" not in timing
    assert 'FR13_GATE_BATCH_GDN_BV=${FR13_GATE_BATCH_GDN_BV:-8}' in gate
    assert 'FR13_FIXED32_BATCH_GDN_BV_CANDIDATE="$FR13_GATE_BATCH_GDN_BV"' in gate
    assert "production_bv=8" in timing
    assert 'FR13_FIXED32_BATCH_GDN_BV_PRODUCTION="$production_bv"' in timing
    assert "FR13_FIXED32_BATCH_GDN_BV8_TIMING=1" in timing
    assert "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=1" in gate
    assert "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0" in timing

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert '"FR13_SCAN_ALIGN|${FR13_SCAN_ALIGN:-0}|0"' in launcher
    assert '"FR13_NPAD_INVARIANT|${FR13_NPAD_INVARIANT:-0}|0"' in launcher
    assert "batched BV8 production requires FR10_METRICS=1" in launcher
    assert "requires exact B4 Hydra fixed32 FULL-graph" in launcher

    kernel_source = KERNEL_SOURCE.read_text(encoding="utf-8")
    assert "batched BV8 production specialization" in kernel_source
    for check in (
        "not count_invocation",
        "not ring_export",
        "not flags_export",
        "scan_align_on()",
        "npad_invariant_on()",
    ):
        assert check in kernel_source


def test_patcher_dispatches_bv8_without_removing_bv64() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    for needle in (
        "fixed32_batch_gdn_bv8_production_capture_begin",
        "fixed32_batch_gdn_bv8_production_capture_end",
        "fixed32_batch_gdn_bv8_production_replay_engaged",
        "_FR13_FIXED32_BATCH_GDN_BV8_PRODUCTION_PUBLISHED",
        "fixed32_batch_gdn_bv64_production_capture_begin",
        "fixed32_batch_gdn_bv64_production_capture_end",
        "fixed32_batch_gdn_bv64_production_replay_engaged",
    ):
        assert needle in patcher
    assert patcher.count("production_bv == 8") >= 3


def test_bv8_attestor_is_in_the_fixed32_runtime_manifest() -> None:
    manifest = (ROOT / "scripts/fr13_runtime_manifest.py").read_text(
        encoding="utf-8"
    )
    assert '"scripts/fr13_b4_gdn_bv8_pass.py"' in manifest
    assert '"scripts/fr13_run_b4_gdn_bv8_timing.sh"' in manifest
    assert '"scripts/fr13_run_b4_gdn_wide_live_gate.sh"' in manifest

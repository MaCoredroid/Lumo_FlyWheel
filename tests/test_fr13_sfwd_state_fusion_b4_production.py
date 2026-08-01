from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PASS_PATH = ROOT / "scripts" / "fr13_sfwd_state_fusion_b4_pass.py"
RUNNER_PATH = ROOT / "scripts" / "fr13_run_b4_sfwd_state_fusion_live_gate.sh"
LAUNCHER_PATH = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
SERVE_VARIANT_PATH = ROOT / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
KERNEL_PATH = ROOT / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
CUTLASS_PASS_PATH = ROOT / "scripts" / "fr13_cutlass_b4_pass.py"
ORCHESTRATOR_PATH = ROOT / "scripts" / "run_swe_bench_q36_a.py"

spec = importlib.util.spec_from_file_location("sfwd_b4_pass", PASS_PATH)
assert spec is not None and spec.loader is not None
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def _write_json(path: Path, payload: dict[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("ascii") + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _b4_live_payload() -> dict[str, object]:
    return {
        "schema": qualification.B4_LIVE_SCHEMA,
        "status": "pass",
        "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
        "candidate": qualification.CANDIDATE,
        "task_set": "canonical real SWE-Verified exact4 B4",
        "task_count": 4,
        "task_ids": list(qualification.EXPECTED_TASK_IDS),
        "task_markers": list(qualification.EXPECTED_TASK_MARKERS),
        "subset_sha256": qualification.EXPECTED_SUBSET_SHA256,
        "real_task_authenticated": True,
        "batch_size": 4,
        "concurrency": 4,
        "physical_rows_per_request": 32,
        "physical_rows_total": 128,
        "layer_count": 48,
        "layer_keys": [f"0x{index + 1:x}" for index in range(48)],
        "comparison_records": 48,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "mismatching_records": 0,
        "differing_bytes": 0,
        "draft_vocab_root": 0,
        "draft_vocab_k": 0,
        "candidate_shadow_only": True,
        "served_result": "reference",
        "reference_always_served": True,
        "probe_inputs": False,
        "synthetic_inputs": False,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "production_eligible": False,
        "kernel_source_sha256": hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest(),
        "patcher_source_sha256": hashlib.sha256(PATCHER_PATH.read_bytes()).hexdigest(),
        "engine_ledger_chain_head_sha256": "1" * 64,
        "real_task_arm_sha256": "2" * 64,
        "runtime_manifest_sha256": "3" * 64,
        "runner_sha256": "4" * 64,
        "source_commit": "5" * 40,
        "errors": [],
    }


def _b1_live_payload() -> dict[str, object]:
    return {
        "schema": qualification.B1_LIVE_SCHEMA,
        "status": "byte_pass_source_only",
        "run_classification": (
            "one_real_swe_verified_full_vocab_b1_byte_timing_diagnostic"
        ),
        "candidate": qualification.CANDIDATE,
        "source_sha256": hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest(),
        "task_marker": qualification.B1_TASK_MARKER,
        "batch": 1,
        "layer_count": 48,
        "layer_keys": [f"0x{index + 1:x}" for index in range(48)],
        "physical_rows_per_request": 32,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }


def test_b4_qualification_is_exact4_source_bound_and_nonproduction(
    tmp_path: Path,
) -> None:
    live = tmp_path / "live.json"
    digest = _write_json(live, _b4_live_payload())
    result = qualification.validate_b4_live_result(
        live,
        expected_live_sha256=digest,
        kernel_source=KERNEL_PATH,
        patcher_source=PATCHER_PATH,
        expected_source_commit="5" * 40,
    )
    assert result["status"] == "QUALIFIED_BYTE_ONLY"
    assert result["qualification_task_ids"] == list(qualification.EXPECTED_TASK_IDS)
    assert result["qualification_batch_size"] == 4
    assert result["qualification_concurrency"] == 4
    assert result["qualification_physical_rows_per_request"] == 32
    assert result["qualification_layer_count"] == 48
    assert result["served_result_during_qualification"] == "reference"
    assert result["timing_eligible"] is False
    assert result["acceptance_valid"] is False
    assert result["production_default_enabled"] is False
    assert result["candidate_serving_permitted"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task_count", 16),
        ("batch_size", 1),
        ("concurrency", 16),
        ("physical_rows_per_request", 31),
        ("layer_count", 47),
        ("draft_vocab_k", 65536),
        ("served_result", "candidate"),
        ("candidate_shadow_only", False),
        ("probe_inputs", True),
        ("synthetic_inputs", True),
        ("acceptance_valid", True),
        ("timing_eligible", True),
        ("production_enabled", True),
    ),
)
def test_b4_qualification_fails_closed_on_contract_drift(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _b4_live_payload()
    payload[field] = value
    live = tmp_path / f"bad-{field}.json"
    digest = _write_json(live, payload)
    with pytest.raises(qualification.QualificationError, match=field):
        qualification.validate_b4_live_result(
            live,
            expected_live_sha256=digest,
            kernel_source=KERNEL_PATH,
            patcher_source=PATCHER_PATH,
        )


def test_b1_b4_binding_is_a_prerequisite_not_a_serving_credential(
    tmp_path: Path,
) -> None:
    live = tmp_path / "b4-live.json"
    live_sha256 = _write_json(live, _b4_live_payload())
    b4 = qualification.validate_b4_live_result(
        live,
        expected_live_sha256=live_sha256,
        kernel_source=KERNEL_PATH,
        patcher_source=PATCHER_PATH,
    )
    b4_path = tmp_path / "b4-qualification.json"
    b4_sha256 = _write_json(b4_path, b4)
    b1_path = tmp_path / "b1-live.json"
    b1_sha256 = _write_json(b1_path, _b1_live_payload())

    binding = qualification.bind_prerequisites(
        b1_live_result=b1_path,
        expected_b1_sha256=b1_sha256,
        b1_kernel_source=KERNEL_PATH,
        b4_qualification=b4_path,
        expected_b4_qualification_sha256=b4_sha256,
        b4_kernel_source=KERNEL_PATH,
    )
    assert binding["byte_prerequisites_satisfied"] is True
    assert binding["b1_batch_size"] == 1
    assert binding["b4_batch_size"] == 4
    assert binding["b4_concurrency"] == 4
    assert binding["acceptance_valid"] is False
    assert binding["timing_eligible"] is False
    assert binding["production_default_enabled"] is False
    assert binding["candidate_serving_permitted"] is False


def test_runner_and_launcher_are_exact4_b4_shadow_only() -> None:
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    serve_variant = SERVE_VARIANT_PATH.read_text(encoding="utf-8")
    orchestrator = ORCHESTRATOR_PATH.read_text(encoding="utf-8")
    pass_source = PASS_PATH.read_text(encoding="utf-8")

    assert "subset_b4_four.json" in runner
    assert "subset_b16" not in runner
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in runner
    assert "FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0" in runner
    assert "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=1" in runner
    assert "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0" in runner
    assert "FR13_FIXED32_CUTLASS_WAVE=stock" in runner
    assert (
        '"${FR13_FIXED32_CUTLASS_WAVE:-stock}" == "stock"'
        in serve_variant
    )
    assert "FR13_CONV_WB_BATCHED=1" in runner
    assert "FR13_FIXED32_CONV_SOURCE_BATCH=1" in runner
    assert (
        "FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER=1 FR13_CFWD_GPU_TIMER=1"
        in runner
    )
    assert "candidate_shadow_only=1" in runner
    assert "reference_always_served=1" in runner
    assert "acceptance_valid=0" in runner
    assert "timing_eligible=0" in runner
    assert "stat.S_IMODE(marker_info.st_mode) != 0o444" in runner
    assert (
        '"${FR13_FIXED32_CUTLASS_WAVE:-stock}" != "stock"'
        in launcher
    )
    assert '-n "${FR13_FIXED32_CUTLASS_WAVE:-}"' not in launcher
    assert "CAPTURE_ONLY" not in runner
    assert "PROBE_ONLY" not in runner
    assert "ACCEPT_SPEED_PROBE" not in runner
    assert "must be the only kernel candidate" in launcher
    assert "requires exact4 B4 full-vocabulary eager fixed32" in launcher
    assert '-e ENFORCE_EAGER="${ENFORCE_EAGER:-0}"' in launcher
    assert "_Fixed32EagerKernelDiagnosticTaskBracket" in orchestrator
    assert 'or sfwd_state_fusion_eager_diagnostic == "1"' in orchestrator
    assert "SFWD production requires exact K64 B1 eager fixed32" in launcher
    assert "SFWD production permits qrow16 and source-gated TAW production only" in launcher
    assert "bind-prerequisites" in pass_source
    assert '"candidate_serving_permitted": False' in pass_source


def test_launcher_explicitly_propagates_sfwd_b4_patch_contract() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    auto_forward = launcher.index('"${FR13_ENV_FORWARD_ARGS[@]}"')
    selector = launcher.index(
        '-e FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB='
        '"$FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB"'
    )
    production = launcher.index(
        '-e FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION='
        '"$FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION"'
    )

    assert auto_forward < selector < production
    for assignment in (
        "FR13_FIXED32_SFWD_STATE_FUSION_ENABLED_PATH="
        "/logs/fr13_fixed32_sfwd_state_fusion_byte_ab.enabled",
        "FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH="
        '"${FR13_FIXED32_SFWD_STATE_FUSION_REAL_EVENT_PATH:-}"',
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB_PATH="
        "/logs/fr13_fixed32_sfwd_state_fusion.byte_ab.jsonl",
        "FR13_FIXED32_SFWD_STATE_FUSION_PASS_PATH="
        "/logs/fr13_fixed32_sfwd_state_fusion.live_pass.json",
    ):
        assert f"-e {assignment}" in launcher


def test_patcher_import_selects_sfwd_b4_eager_lifecycle() -> None:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("FR13_")
    }
    environment.update(
        {
            "ENFORCE_EAGER": "1",
            "FR13_FIXED32_BATCH_GDN_BYTE_AB": "0",
            "FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB": "0",
            "FR13_FIXED32_CUTLASS_WAVE": "stock",
            "FR13_FIXED32_MODE": "hydra27_fixed32",
            "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB": "1",
            "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION": "0",
        }
    )
    program = """
import importlib.util
import json
import pathlib

path = pathlib.Path(r"%s")
spec = importlib.util.spec_from_file_location("sfwd_patch_time", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    "selector": module._FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB,
    "contract": module._fr13_fixed32_eager_boot_warm_contract(),
    "bindings": module._fr13_fixed32_runtime_bindings("hydra27_fixed32"),
}))
""" % os.fspath(PATCHER_PATH)

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["selector"] == "1"
    assert payload["contract"] == [
        "SFWD B4 byte diagnostic",
        4,
        "FR13_FIXED32_EAGER_SFWD_B4_BOOT_WARM",
    ]
    assert "_FR13_FIXED32_EAGER_KERNEL_DIAGNOSTIC = True" in payload["bindings"]


def test_existing_cutlass_five_shape_cap320_contract_is_unchanged() -> None:
    source = CUTLASS_PASS_PATH.read_text(encoding="utf-8")
    assert "MAX_COMPARISONS = 320" in source
    for shape in (
        "(5120, 6144)",
        "(5120, 17408)",
        "(14336, 5120)",
        "(16384, 5120)",
        "(34816, 5120)",
    ):
        assert shape in source

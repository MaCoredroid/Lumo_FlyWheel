from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
TIMING = SCRIPTS / "fr13_qrow32_split2_timing.py"
SUBSET = REPO / "config/fr13_fixed32/subset_b4_four.json"
BASELINE = (
    REPO
    / "results/fr13_fixed32_qrow16_prod_exact4_b1_20260731T182827Z"
    / "hydra_valid/deploy_speed_qrow16_prod_exact4_b1_20260731T182827Z.json"
)


def _module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("qrow32_timing", TIMING)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return path


def _inputs(tmp_path: Path):
    module = _module()
    arm = "hydra27_fixed32_k64_qrow32_nosplit_exact4_test"
    source = "1" * 40
    patch = "2" * 64
    sidecar = "3" * 64
    rows = []
    for index, task_id in enumerate(module.EXACT4_TASK_IDS):
        # Synthetic step wall placed just ABOVE the weight floor and well
        # BELOW the U95 cap, which is what makes the eligibility assertion
        # below informative. FR13 used 120-123 ms against a 119.658 floor /
        # 137.607 cap; FR14's floor is 102.480 and its cap 117.852, so the
        # fixture moves with them (a fixed 120 ms would now sit ABOVE the cap
        # and the test would only be asserting that the cap works).
        wall_ms = 103.0 + index
        rows.append(
            {
                "instance_id": task_id,
                "drafts": 100,
                "fwd_gpu_steps": 100,
                "wall_seconds": wall_ms / 10.0,
                "wall_steps": 100,
                "tok_per_draft": 31.0,
            }
        )
    measure = {
        "schema": module.MEASURE_SCHEMA,
        "instrument": "OFF",
        "regime": "deployment",
        "arm": arm,
        "batch_size": 1,
        "n_tasks": 4,
        "task_instance_ids": list(module.EXACT4_TASK_IDS),
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "mandatory_weight_bytes": 27977022848,
        "per_task": rows,
    }
    engagement = {
        "schema": module.ENGAGEMENT_SCHEMA,
        "status": "ENGAGED",
        "runtime_mode": "FULL",
        "batch_size": 1,
        "physical_rows": 32,
        "arm": module.ARM,
        "num_splits": 0,
        "layer_count": 16,
        "candidate_served": True,
        "fallback_allowed": False,
        "candidate_so_sha256": module.CANDIDATE_SHA256,
        "candidate_so_size": module.CANDIDATE_SIZE,
        "fa2_head": module.FA2_HEAD,
        "fa2_source_closure_sha256": module.SOURCE_CLOSURE_SHA256,
        "source_commit": source,
        "patch_source_sha256": patch,
        "pass_sidecar_sha256": sidecar,
        "task_ids": list(module.EXACT4_TASK_IDS),
        "subset_sha256": module.EXACT4_SUBSET_SHA256,
    }
    health = {
        "swe_orchestrator_rc": 0,
        "tasks": [
            {
                "instance_id": task_id,
                "codex_timed_out": False,
                "verdict": "resolved",
            }
            for task_id in module.EXACT4_TASK_IDS
        ],
    }
    audit = {
        "schema": "fr13-fixed32-chat-task-provenance-audit-v3",
        "mode": "hydra27_fixed32",
        "subset": {
            "sha256": module.EXACT4_SUBSET_SHA256,
            "task_count": 4,
            "task_ids": list(module.EXACT4_TASK_IDS),
        },
        "checks": {"all_canonical_tasks_validated": True},
    }
    paths = {
        "measure_path": _write(tmp_path / "measure.json", measure),
        "engagement_path": _write(tmp_path / "engagement.json", engagement),
        "health_path": _write(tmp_path / "health.json", health),
        "traffic_audit_path": _write(tmp_path / "audit.json", audit),
    }
    return module, arm, source, patch, sidecar, paths


def test_exact4_u95_controls_exact16_eligibility(tmp_path: Path) -> None:
    module, arm, source, patch, sidecar, paths = _inputs(tmp_path)
    result = module.reduce_timing(
        subset_path=SUBSET,
        baseline_path=BASELINE,
        source_commit=source,
        patch_source_sha256=patch,
        pass_sha256="4" * 64,
        pass_sidecar_sha256=sidecar,
        runner_sha256="5" * 64,
        block_map_sha256="6" * 64,
        floor_ms=102.479937172,
        cap_ms=117.8519277478,
        arm=arm,
        **paths,
    )
    assert result["timing_eligible"] is True
    assert result["formal_floor_acceptance_eligible"] is False
    assert result["exact16_eligible"] is True
    assert (
        result["descriptive_equal_task_one_sided_u95"]["u95_ms"]
        <= result["descriptive_equal_task_one_sided_u95"]["cap_ms"]
    )


def test_incomplete_wall_retention_is_rejected(tmp_path: Path) -> None:
    module, arm, source, patch, sidecar, paths = _inputs(tmp_path)
    measure = json.loads(paths["measure_path"].read_text(encoding="ascii"))
    measure["per_task"][0]["wall_steps"] = 90
    _write(paths["measure_path"], measure)
    with pytest.raises(ValueError, match="retained wall fraction"):
        module.reduce_timing(
            subset_path=SUBSET,
            baseline_path=BASELINE,
            source_commit=source,
            patch_source_sha256=patch,
            pass_sha256="4" * 64,
            pass_sidecar_sha256=sidecar,
            runner_sha256="5" * 64,
            block_map_sha256="6" * 64,
            floor_ms=102.479937172,
            cap_ms=117.8519277478,
            arm=arm,
            **paths,
        )

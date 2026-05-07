from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_track_b_e2e_readiness_manifest as readiness  # noqa: E402


def test_readiness_manifest_reports_round0_blocked(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "round0_may_run": False,
                "blocking_reasons": [
                    "vllm_request_metrics_join_available",
                    "codex_trace_out_supported",
                    "dcgm_profile_fields_available",
                ],
                "checks": {
                    "codex_trace_out_supported": {"ok": False},
                    "dcgm_sampler_runs": {"ok": True},
                    "dcgm_profile_fields_available": {"ok": False},
                    "pynvml_available": {"ok": True},
                    "vllm_request_id_labels_exposed": {"ok": False},
                    "vllm_request_metrics_side_channel": {"ok": False},
                    "vllm_request_metrics_join_available": {"ok": False},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert manifest["round0_ready"] is False
    assert manifest["decision"] == "round0_blocked"
    statuses = {step["step"]: step["status"] for step in manifest["implementation_steps"]}
    assert statuses["A"] == "blocked"
    assert statuses["B"] == "blocked"
    assert statuses["C"] == "complete"
    assert statuses["D"] == "blocked"
    assert statuses["E"] == "complete"
    assert statuses["F"] == "complete"
    assert statuses["G"] == "blocked"
    assert manifest["hard_gates"]["round0_summary_verified"] is False
    assert manifest["hard_gates"]["ncu_profiles_verified"] is False


def test_readiness_manifest_requires_round0_artifacts_even_if_preflight_passes(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                "round0_may_run": True,
                "blocking_reasons": [],
                "checks": {
                    "codex_trace_out_supported": {"ok": True},
                    "dcgm_sampler_runs": {"ok": True},
                    "dcgm_profile_fields_available": {"ok": True},
                    "pynvml_available": {"ok": True},
                    "vllm_request_id_labels_exposed": {"ok": True},
                    "vllm_request_metrics_side_channel": {"ok": False},
                    "vllm_request_metrics_join_available": {"ok": True},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert manifest["hard_gates"]["preflight_round0_may_run"] is True
    assert manifest["hard_gates"]["round0_summary_verified"] is False
    assert manifest["round0_ready"] is False


def test_trace_correctness_verification_requires_three_matching_tasks(tmp_path: Path) -> None:
    artifact = tmp_path / "codex_trace_emitter_correctness.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.codex_trace_correctness.v1",
                "verified_at": "2026-05-07T20:00:00Z",
                "trace_out_supported": True,
                "tasks": [
                    {
                        "task_id": "task-a",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                    },
                    {
                        "task_id": "task-b",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                    },
                    {
                        "task_id": "task-c",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = readiness._trace_correctness_verification(artifact)

    assert result["ok"] is True
    assert result["task_count"] == 3
    assert result["reasons"] == []


def test_trace_correctness_verification_rejects_weak_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "codex_trace_emitter_correctness.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.codex_trace_correctness.v1",
                "verified_at": "2026-05-07T20:00:00Z",
                "trace_out_supported": True,
                "tasks": [
                    {
                        "task_id": "task-a",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": False,
                        "milestone_scores_identical": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = readiness._trace_correctness_verification(artifact)

    assert result["ok"] is False
    assert "too_few_tasks" in result["reasons"]
    assert "task_0_failed" in result["reasons"]


def test_round0_summary_verification_requires_trusted_completed_tasks(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "sha256:test",
                "sample_hash": "sha256:sample",
                "trusted_task_count": 12,
                "tasks_completed": 12,
                "tasks_correctness_passed": 12,
                "median_wallclock_s": 187.4,
                "aggregate_wallclock_s": 2618.1,
                "diagnosis_distribution": {"memory-bw-headroom": 12},
            }
        ),
        encoding="utf-8",
    )

    result = readiness._round0_summary_verification(summary)

    assert result["ok"] is True
    assert result["trusted_task_count"] == 12
    assert result["reasons"] == []


def test_round0_summary_verification_rejects_existence_only_summary(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "sha256:test",
                "trusted_task_count": 1,
            }
        ),
        encoding="utf-8",
    )

    result = readiness._round0_summary_verification(summary)

    assert result["ok"] is False
    assert "too_few_trusted_tasks" in result["reasons"]
    assert "sample_hash_missing" in result["reasons"]
    assert "diagnosis_distribution_missing" in result["reasons"]


def test_ncu_profile_verification_requires_named_nonempty_archetypes(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text("Metric Name,Metric Value\n", encoding="utf-8")

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is True
    assert result["profile_count"] == 5
    assert result["reasons"] == []


def test_ncu_profile_verification_rejects_wrong_or_empty_files(tmp_path: Path) -> None:
    (tmp_path / "ncu_unrelated.csv").write_text("Metric Name,Metric Value\n", encoding="utf-8")
    (tmp_path / "ncu_long-text.csv").write_text("", encoding="utf-8")

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert result["profile_count"] == 0
    assert "long-text_missing_or_empty" in result["reasons"]
    assert "tool-call-frame_missing_or_empty" in result["reasons"]

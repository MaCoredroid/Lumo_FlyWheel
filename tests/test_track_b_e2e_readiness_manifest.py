from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_track_b_e2e_readiness_manifest as readiness  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _missing_round0_summary(_path: Path) -> dict[str, object]:
    return {
        "ok": False,
        "runtime_config_hash": None,
        "trusted_task_count": None,
        "trusted_unique_task_count": None,
        "tasks_completed": None,
        "tasks_correctness_passed": None,
        "runtime_config_hash_mismatch_count": None,
        "task_summary_schema_mismatch_count": None,
        "task_summary_round_mismatch_count": None,
        "reasons": ["summary_missing_or_invalid_json"],
    }


def test_readiness_manifest_reports_round0_blocked(tmp_path: Path, monkeypatch) -> None:
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

    monkeypatch.setattr(readiness, "_round0_summary_verification", _missing_round0_summary)

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
    steps = {step["step"]: step for step in manifest["implementation_steps"]}
    assert steps["C"]["evidence"]["runner_script_exists"] is True
    assert steps["C"]["evidence"]["round_driver_script_exists"] is True
    assert steps["D"]["evidence"]["vllm_request_metrics_side_channel_ok"] is False
    assert steps["D"]["evidence"]["vllm_request_metrics_side_channel"] == {"ok": False}
    assert steps["F"]["evidence"]["required"]["uses_hard_gated_round_driver"] is True
    assert steps["F"]["evidence"]["required"]["checks_all_spec_decode_counters"] is True
    assert steps["F"]["evidence"]["forbidden"]["direct_repeat3_task_measurement"] is False
    assert steps["G"]["evidence"]["ncu_profile_driver_exists"] is True
    assert steps["G"]["evidence"]["ncu_profile_driver_server_launch_supported"] is True
    assert steps["G"]["evidence"]["ncu_profile_driver_container_server_launch_supported"] is True
    assert steps["G"]["evidence"]["ncu_profile_required_topology"] == "server-launch"
    assert manifest["hard_gates"]["round_proposal_prompt_verified"] is True
    assert manifest["hard_gates"]["round0_summary_verified"] is False
    assert manifest["hard_gates"]["ncu_profiles_verified"] is False


def test_readiness_manifest_reports_deferred_preflight_gaps(tmp_path: Path, monkeypatch) -> None:
    preflight_path = tmp_path / "preflight.json"
    deferred = [
        "vllm_request_metrics_join_available",
        "codex_trace_out_supported",
        "dcgm_profile_fields_available",
    ]
    preflight_path.write_text(
        json.dumps(
            {
                "round0_may_run": True,
                "blocking_reasons": [],
                "deferred_reasons": deferred,
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

    monkeypatch.setattr(readiness, "_round0_summary_verification", _missing_round0_summary)

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert manifest["round0_ready"] is False
    assert manifest["deferred_reasons"] == deferred
    statuses = {step["step"]: step["status"] for step in manifest["implementation_steps"]}
    assert statuses["A"] == "deferred"
    assert statuses["B"] == "deferred"
    assert statuses["D"] == "deferred"
    steps = {step["step"]: step for step in manifest["implementation_steps"]}
    assert steps["A"]["evidence"]["deferred_by_preflight"] is True
    assert steps["B"]["evidence"]["deferred_by_preflight"] is True
    assert steps["D"]["evidence"]["deferred_by_preflight"] is True
    assert manifest["hard_gates"]["all_implementation_steps_complete"] is False


def test_round_proposal_prompt_uses_hard_gated_round_driver() -> None:
    prompt = (REPO_ROOT / "prompts" / "track_b_e2e_round_proposal.md").read_text(encoding="utf-8")

    assert "scripts/run_track_b_e2e_round.py" in prompt
    assert "--runtime-config-hash {{runtime_config_hash}}" in prompt
    assert "--protocol-hash-match" in prompt
    assert "--trace-correctness-artifact output/track_b_e2e/codex_trace_emitter_correctness.json" in prompt
    assert "spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total" in prompt
    assert "run_track_b_e2e_task.py --round {{round}} --tasks all --repeat 3" not in prompt


def test_round_proposal_prompt_verification_rejects_legacy_direct_measurement(
    tmp_path: Path, monkeypatch
) -> None:
    prompt = tmp_path / "prompts" / "track_b_e2e_round_proposal.md"
    prompt.parent.mkdir()
    prompt.write_text(
        "\n".join(
            [
                "scripts/run_track_b_e2e_round.py",
                "--runtime-config-hash {{runtime_config_hash}}",
                "--protocol-hash-match",
                "--trace-correctness-artifact output/track_b_e2e/codex_trace_emitter_correctness.json",
                "scripts/preflight_track_b_e2e.py",
                "spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total",
                "run_track_b_e2e_task.py --round {{round}} --tasks all --repeat 3",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path)

    result = readiness._round_proposal_prompt_verification()

    assert result["ok"] is False
    assert "forbidden_direct_repeat3_task_measurement_present" in result["reasons"]


def test_readiness_manifest_requires_round0_artifacts_even_if_preflight_passes(
    tmp_path: Path, monkeypatch
) -> None:
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

    monkeypatch.setattr(readiness, "_round0_summary_verification", _missing_round0_summary)

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert manifest["hard_gates"]["preflight_round0_may_run"] is True
    assert manifest["hard_gates"]["round_proposal_prompt_verified"] is True
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
                        "trace_schema_valid": True,
                    },
                    {
                        "task_id": "task-b",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                        "trace_schema_valid": True,
                    },
                    {
                        "task_id": "task-c",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                        "trace_schema_valid": True,
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
                        "trace_schema_valid": True,
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


def test_trace_correctness_verification_rejects_missing_trace_schema_status(tmp_path: Path) -> None:
    artifact = tmp_path / "codex_trace_emitter_correctness.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.codex_trace_correctness.v1",
                "verified_at": "2026-05-07T20:00:00Z",
                "trace_out_supported": True,
                "tasks": [
                    {
                        "task_id": f"task-{index}",
                        "trace_out_enabled_exit_code": 0,
                        "trace_out_disabled_exit_code": 0,
                        "model_outputs_byte_identical": True,
                        "tool_call_sequences_byte_identical": True,
                        "milestone_scores_identical": True,
                    }
                    for index in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )

    result = readiness._trace_correctness_verification(artifact)

    assert result["ok"] is False
    assert result["tasks"][0]["missing_fields"] == ["trace_schema_valid"]


def test_trace_patch_verification_rejects_placeholder_patch(tmp_path: Path, monkeypatch) -> None:
    patch_path = tmp_path / "patches" / "codex" / "trace_emitter.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text("TODO: add trace-out support later\n", encoding="utf-8")
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path)

    result = readiness._trace_patch_verification()

    assert result["ok"] is False
    assert result["reasons"] == ["trace_patch_content_invalid"]
    patch = next(patch for patch in result["patches"] if patch["exists"])
    assert "unified_diff" in patch["missing_required"]
    assert "trace_out_flag" in patch["missing_required"]


def test_trace_patch_verification_requires_trace_emitter_surface(tmp_path: Path, monkeypatch) -> None:
    patch_path = tmp_path / "vendor" / "codex-cli" / "patches" / "trace_emitter.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/codex-rs/exec/src/main.rs b/codex-rs/exec/src/main.rs",
                "+    --trace-out <PATH>",
                "+    emit_jsonl(\"task_start\", runtime_config_hash);",
                "+    emit_jsonl(\"turn_start\", vllm_request_id);",
                "+    emit_jsonl(\"turn_end\", completion_tokens);",
                "+    emit_jsonl(\"tool_call\", tool_name);",
                "+    emit_jsonl(\"task_end\", exit_code);",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(readiness, "REPO_ROOT", tmp_path)

    result = readiness._trace_patch_verification()

    assert result["ok"] is True
    assert result["reasons"] == []
    assert result["patches"][0]["ok"] is True


def test_round0_summary_verification_requires_trusted_completed_tasks(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "sample_hash": "sha256:sample",
                "trusted_task_count": 12,
                "trusted_unique_task_count": 12,
                "duplicate_trusted_task_ids": [],
                "unexpected_trusted_task_ids": [],
                "sample_hash_mismatch_count": 0,
                "runtime_config_hash_mismatch_count": 0,
                "task_summary_schema_mismatch_count": 0,
                "task_summary_round_mismatch_count": 0,
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
    assert result["trusted_unique_task_count"] == 12
    assert result["runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert result["reasons"] == []


def test_readiness_manifest_passes_round0_runtime_hash_to_ncu_verifier(tmp_path: Path, monkeypatch) -> None:
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps({"round0_may_run": True, "blocking_reasons": [], "checks": {}}),
        encoding="utf-8",
    )
    expected_hash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    captured: dict[str, object] = {}

    def fake_round0_summary(_path: Path) -> dict[str, object]:
        return {
            "ok": True,
            "runtime_config_hash": expected_hash,
            "trusted_task_count": 12,
            "trusted_unique_task_count": 12,
            "tasks_completed": 12,
            "tasks_correctness_passed": 12,
            "runtime_config_hash_mismatch_count": 0,
            "task_summary_schema_mismatch_count": 0,
            "task_summary_round_mismatch_count": 0,
            "reasons": [],
        }

    def fake_ncu_profiles(_output_dir: Path, *, expected_runtime_config_hash: str = "") -> dict[str, object]:
        captured["expected_runtime_config_hash"] = expected_runtime_config_hash
        return {
            "ok": False,
            "profile_count": 0,
            "expected_profile_count": len(readiness.NCU_ARCHETYPES),
            "expected_runtime_config_hash": expected_runtime_config_hash,
            "failure_metadata_count": 0,
            "failure_reasons": {},
            "profiles": [],
            "reasons": ["long-text_missing_or_empty"],
        }

    monkeypatch.setattr(readiness, "_round0_summary_verification", fake_round0_summary)
    monkeypatch.setattr(readiness, "_ncu_profile_verification", fake_ncu_profiles)

    manifest = readiness.build_manifest(Namespace(preflight_json=str(preflight_path), out=""))

    assert captured["expected_runtime_config_hash"] == expected_hash
    steps = {step["step"]: step for step in manifest["implementation_steps"]}
    assert steps["G"]["evidence"]["ncu_profiles"]["expected_runtime_config_hash"] == expected_hash


def test_round0_summary_verification_rejects_existence_only_summary(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "trusted_task_count": 1,
                "trusted_unique_task_count": 1,
                "duplicate_trusted_task_ids": ["task-a"],
                "unexpected_trusted_task_ids": ["off-sample/v1"],
                "sample_hash_mismatch_count": 1,
                "runtime_config_hash_mismatch_count": 1,
                "task_summary_schema_mismatch_count": 1,
                "task_summary_round_mismatch_count": 1,
            }
        ),
        encoding="utf-8",
    )

    result = readiness._round0_summary_verification(summary)

    assert result["ok"] is False
    assert "too_few_trusted_tasks" in result["reasons"]
    assert "too_few_unique_trusted_tasks" in result["reasons"]
    assert "duplicate_trusted_task_ids_present" in result["reasons"]
    assert "unexpected_trusted_task_ids_present" in result["reasons"]
    assert "sample_hash_mismatch" in result["reasons"]
    assert "runtime_config_hash_mismatch" in result["reasons"]
    assert "task_summary_schema_mismatch" in result["reasons"]
    assert "task_summary_round_mismatch" in result["reasons"]
    assert "sample_hash_missing" in result["reasons"]
    assert "diagnosis_distribution_missing" in result["reasons"]


def test_round0_summary_verification_rejects_invalid_runtime_hash(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "not-a-runtime-hash",
                "sample_hash": "sha256:sample",
                "trusted_task_count": 12,
                "trusted_unique_task_count": 12,
                "duplicate_trusted_task_ids": [],
                "unexpected_trusted_task_ids": [],
                "sample_hash_mismatch_count": 0,
                "runtime_config_hash_mismatch_count": 0,
                "task_summary_schema_mismatch_count": 0,
                "task_summary_round_mismatch_count": 0,
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

    assert result["ok"] is False
    assert "runtime_config_hash_invalid" in result["reasons"]


def test_round0_summary_verification_rejects_nonfinite_wallclocks(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "sample_hash": "sha256:sample",
                "trusted_task_count": 12,
                "trusted_unique_task_count": 12,
                "duplicate_trusted_task_ids": [],
                "unexpected_trusted_task_ids": [],
                "sample_hash_mismatch_count": 0,
                "runtime_config_hash_mismatch_count": 0,
                "task_summary_schema_mismatch_count": 0,
                "task_summary_round_mismatch_count": 0,
                "tasks_completed": 12,
                "tasks_correctness_passed": 12,
                "median_wallclock_s": float("nan"),
                "aggregate_wallclock_s": 0,
                "diagnosis_distribution": {"memory-bw-headroom": 12},
            }
        ),
        encoding="utf-8",
    )

    result = readiness._round0_summary_verification(summary)

    assert result["ok"] is False
    assert "median_wallclock_missing" in result["reasons"]
    assert "aggregate_wallclock_missing" in result["reasons"]


def test_round0_summary_verification_requires_explicit_zero_mismatch_counts(tmp_path: Path) -> None:
    summary = tmp_path / "round_summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.e2e_round_summary.v1",
                "round": 0,
                "runtime_config_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "sample_hash": "sha256:sample",
                "trusted_task_count": 12,
                "trusted_unique_task_count": 12,
                "duplicate_trusted_task_ids": [],
                "unexpected_trusted_task_ids": [],
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

    assert result["ok"] is False
    assert "sample_hash_mismatch" in result["reasons"]
    assert "runtime_config_hash_mismatch" in result["reasons"]
    assert "task_summary_schema_mismatch" in result["reasons"]
    assert "task_summary_round_mismatch" in result["reasons"]


def _ncu_csv_text() -> str:
    return (
        '"Metric Name","Metric Value"\n'
        + "\n".join(f'"{metric}",1.0' for metric in readiness.NCU_REQUIRED_METRICS)
        + "\n"
    )


def _write_ncu_metadata(root: Path, archetype: str, *, runtime_config_hash: str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") -> None:
    (root / f"ncu_{archetype}.json").write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.ncu_archetype_profile.v1",
                "round": 0,
                "archetype": archetype,
                "task_id": readiness.NCU_ARCHETYPE_TASKS[archetype],
                "runtime_config_hash": runtime_config_hash,
                "profile_csv": str(root / f"ncu_{archetype}.csv"),
                "required_metrics": list(readiness.NCU_REQUIRED_METRICS),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_ncu_profile_verification_requires_named_metric_complete_archetypes(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)

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


def test_ncu_profile_verification_rejects_missing_required_metrics(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        text = _ncu_csv_text()
        if archetype == "long-text":
            text = text.replace("gpu__time_duration.sum", "some_other_metric")
        (tmp_path / f"ncu_{archetype}.csv").write_text(text, encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert result["profile_count"] == 4
    assert "long-text_missing_required_metrics" in result["reasons"]
    long_text = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert long_text["missing_metrics"] == ["gpu__time_duration.sum"]


def test_ncu_profile_verification_reports_no_kernel_warning(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        text = _ncu_csv_text()
        if archetype == "long-text":
            text = "==WARNING== No kernels were profiled.\n"
        (tmp_path / f"ncu_{archetype}.csv").write_text(text, encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert "long-text_no_kernels_profiled" in result["reasons"]
    long_text = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert long_text["no_kernels_profiled"] is True


def test_ncu_profile_verification_reports_application_error(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        text = _ncu_csv_text()
        if archetype == "long-text":
            text = "==ERROR== The application returned an error code (1).\n"
        (tmp_path / f"ncu_{archetype}.csv").write_text(text, encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert "long-text_application_error" in result["reasons"]
    long_text = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert long_text["application_error"] is True


def test_ncu_profile_verification_reports_failure_sidecar(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text("", encoding="utf-8")
    (tmp_path / "ncu_long-text_failure.json").write_text(
        json.dumps(
            {
                "schema": "lumo.track_b.ncu_archetype_profile_failure.v1",
                "reason": "server_not_ready",
                "profile_target": "container-server-launch",
                "server_returncode": 1,
                "task_returncode": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert result["failure_metadata_count"] == 1
    assert result["failure_reasons"] == {"long-text": "server_not_ready"}
    long_text = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert long_text["failure_metadata_exists"] is True
    assert long_text["failure_metadata_schema"] == "lumo.track_b.ncu_archetype_profile_failure.v1"
    assert long_text["failure_profile_target"] == "container-server-launch"
    assert long_text["failure_server_returncode"] == 1


def test_ncu_profile_verification_rejects_nonfinite_required_metric_values(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        text = _ncu_csv_text()
        if archetype == "long-text":
            text = text.replace('"gpu__time_duration.sum",1.0', '"gpu__time_duration.sum",NaN')
        (tmp_path / f"ncu_{archetype}.csv").write_text(text, encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert "long-text_nonfinite_required_metrics" in result["reasons"]
    long_text = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert long_text["nonfinite_metrics"] == ["gpu__time_duration.sum"]
    assert long_text["required_metric_values"]["gpu__time_duration.sum"] == []


def test_ncu_profile_verification_rejects_missing_metadata(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        if archetype != "long-text":
            _write_ncu_metadata(tmp_path, archetype)

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert "long-text_metadata_invalid" in result["reasons"]
    long_text = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert "schema_mismatch" in long_text["metadata_reasons"]
    assert "runtime_config_hash_missing" in long_text["metadata_reasons"]
    assert "task_id_mismatch" in long_text["metadata_reasons"]
    assert "required_metrics_mismatch" in long_text["metadata_reasons"]
    assert "round_missing" in long_text["metadata_reasons"]


def test_ncu_profile_verification_rejects_wrong_archetype_task_metadata(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)
    metadata_path = tmp_path / "ncu_tool-call-frame.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["task_id"] = "wrong-task/v1-clean-baseline"
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    assert "tool-call-frame_metadata_invalid" in result["reasons"]
    profile = next(profile for profile in result["profiles"] if profile["archetype"] == "tool-call-frame")
    assert profile["expected_task_id"] == "policy-aware-request-resolution/v1-clean-baseline"
    assert "task_id_mismatch" in profile["metadata_reasons"]


def test_ncu_profile_verification_rejects_required_metric_metadata_drift(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)
    metadata_path = tmp_path / "ncu_long-text.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["required_metrics"] = list(readiness.NCU_REQUIRED_METRICS[:-1])
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    profile = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert profile["required_metrics_metadata_match"] is False
    assert "required_metrics_mismatch" in profile["metadata_reasons"]


def test_ncu_profile_verification_rejects_missing_round_metadata(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        _write_ncu_metadata(tmp_path, archetype)
    metadata_path = tmp_path / "ncu_long-text.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("round")
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    profile = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert "round_missing" in profile["metadata_reasons"]


def test_ncu_profile_verification_rejects_invalid_runtime_hash_metadata(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        _write_ncu_metadata(
            tmp_path,
            archetype,
            runtime_config_hash="not-a-runtime-hash" if archetype == "long-text" else "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

    result = readiness._ncu_profile_verification(tmp_path)

    assert result["ok"] is False
    profile = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert "runtime_config_hash_invalid" in profile["metadata_reasons"]


def test_ncu_profile_verification_rejects_runtime_hash_drift_when_expected(tmp_path: Path) -> None:
    for archetype in readiness.NCU_ARCHETYPES:
        (tmp_path / f"ncu_{archetype}.csv").write_text(_ncu_csv_text(), encoding="utf-8")
        _write_ncu_metadata(
            tmp_path,
            archetype,
            runtime_config_hash="sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" if archetype == "long-text" else "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

    result = readiness._ncu_profile_verification(tmp_path, expected_runtime_config_hash="sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

    assert result["ok"] is False
    assert result["expected_runtime_config_hash"] == "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    profile = next(profile for profile in result["profiles"] if profile["archetype"] == "long-text")
    assert "runtime_config_hash_mismatch" in profile["metadata_reasons"]

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from lumo_flywheel_serving import sprint0_closeout
from lumo_flywheel_serving.tuned_config import RuntimeStateStore, make_tuned_config_bundle, persist_tuned_config_bundle


def _write_registry(path: Path) -> None:
    path.write_text(
        """
models:
  qwen3.5-27b:
    hf_repo: Qwen/Qwen3.5-27B-FP8
    hf_revision: 2e1b21350ce589fcaafbb3c7d7eac526a7aed582
    local_path: /models/qwen3.5-27b-fp8
    quantization: fp8
    dtype: auto
    kv_cache_dtype: fp8_e5m2
    max_model_len: 131072
    gpu_memory_utilization: 0.90
    max_num_batched_tokens: 8192
    max_num_seqs: 4
""",
        encoding="utf-8",
    )


def _write_workload(path: Path) -> None:
    path.write_text(
        """
family_id: proposal-ranking-manager-judgment
workload_distribution_id: prmj-v1-live
latency_ceiling_ms: 35000
p99_context_tokens: 24576
avg_prompt_tokens: 4096
avg_output_tokens: 1200
rollout_baseline: 10.0
measurement_window_minutes: 25
gpu_memory_utilization_cap: 0.08
""",
        encoding="utf-8",
    )


def _baseline_bundle(root: Path) -> Path:
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id="prmj-v1-live",
        vllm_config={
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 65536,
            "kv_cache_dtype": "auto",
        },
        objective={"metric": "sustained_concurrent_eval_threads_at_L_ceiling", "value": 12},
        measurement_trace_ref=str(root / "measurement.json"),
        search_trace_ref=str(root / "search.json"),
        baseline_bundle_id=None,
        regression_guard={"baseline_value": 8, "delta": 4},
        safety_rails={"regression_guard_passed": True},
    )
    return persist_tuned_config_bundle(bundle, root / "bundles")


def test_capture_resume_cycle_artifact_records_expected_reload(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    _write_registry(registry_path)
    bundle_path = _baseline_bundle(tmp_path)
    state_root = tmp_path / "state"
    store = RuntimeStateStore(state_root)
    bundle = sprint0_closeout.load_baseline_bundle(bundle_path)
    assert bundle is not None
    store.activate_bundle(bundle_path, bundle)
    store.clear_active_bundle()

    def fake_run(command: list[str], cwd: Path, capture_output: bool, text: bool, check: bool):
        command_tail = command[-1]
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="test-sha\n", stderr="")
        if command[:3] == ["git", "status", "--short"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        if command_tail == "resume":
            store.activate_bundle(bundle_path, bundle)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "status": "resumed",
                        "model_id": "qwen3.5-27b",
                        "weight_version_id": "2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
                        "tuned_config_path": str(bundle_path),
                    }
                ),
                stderr="",
            )
        if command_tail == "status":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"health": 200, "models": {"data": [{"id": "qwen3.5-27b"}]}}),
                stderr="",
            )
        if command_tail == "stop":
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(sprint0_closeout.subprocess, "run", fake_run)

    artifact_path = sprint0_closeout.capture_resume_cycle_artifact(
        repo_root=tmp_path,
        registry_path=registry_path,
        state_root=state_root,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
        artifact_path=tmp_path / "artifacts" / "resume_cycle" / "recorded_session.json",
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["assertions"]["reloaded_expected_tuned_config_path"] is True
    assert payload["assertions"]["reloaded_expected_weight_version_id"] is True


def test_parse_json_object_accepts_prefixed_cli_output() -> None:
    payload = sprint0_closeout._parse_json_object(
        "docker-id\ncontainer-name\n{\n  \"status\": \"resumed\",\n  \"model_id\": \"qwen3.5-27b\"\n}"
    )

    assert payload == {"status": "resumed", "model_id": "qwen3.5-27b"}


def test_generate_sprint0_closeout_artifacts_emits_safety_rail_logs(tmp_path: Path, monkeypatch) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    workload_path = tmp_path / "serving_workload.yaml"
    _write_registry(registry_path)
    _write_workload(workload_path)
    bundle_path = _baseline_bundle(tmp_path)
    state_root = tmp_path / "state"
    store = RuntimeStateStore(state_root)
    bundle = sprint0_closeout.load_baseline_bundle(bundle_path)
    assert bundle is not None
    store.activate_bundle(bundle_path, bundle)

    def fake_capture_resume_cycle_artifact(**kwargs):
        artifact_path = Path(kwargs["artifact_path"])
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps({"pass": True}), encoding="utf-8")
        return artifact_path

    monkeypatch.setattr(sprint0_closeout, "_git_metadata", lambda repo_root: {"git_sha": "test-sha", "git_dirty": False})
    monkeypatch.setattr(sprint0_closeout, "capture_resume_cycle_artifact", fake_capture_resume_cycle_artifact)

    artifact_paths = sprint0_closeout.generate_sprint0_closeout_artifacts(
        repo_root=tmp_path,
        registry_path=registry_path,
        state_root=state_root,
        logs_root=tmp_path / "logs",
        triton_cache_root=tmp_path / "triton",
        workload_file=workload_path,
        artifacts_root=tmp_path / "artifacts",
    )

    regression_guard = json.loads(artifact_paths["regression_guard"].read_text(encoding="utf-8"))
    determinism_check = json.loads(artifact_paths["determinism_check"].read_text(encoding="utf-8"))
    oom_handling = json.loads(artifact_paths["oom_handling"].read_text(encoding="utf-8"))

    assert regression_guard["pass"] is True
    assert regression_guard["run_result"]["status"] == "retained_baseline"
    assert determinism_check["pass"] is True
    assert determinism_check["run_result"]["stopping_reason"] == "hard_infeasibility_determinism"
    assert oom_handling["pass"] is True
    assert oom_handling["run_result"]["stopping_reason"] == "hard_infeasibility_oom"

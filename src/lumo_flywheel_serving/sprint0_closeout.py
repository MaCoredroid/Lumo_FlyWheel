from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .auto_research import OfflineAutoResearchRunner, SyntheticWorkloadDistribution, load_baseline_bundle
from .registry import load_registry
from .tuned_config import RuntimeStateStore


def _isoformat_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    return {"git_sha": sha, "git_dirty": dirty}


def _run_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    started_at = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    duration_seconds = round(time.monotonic() - started_at, 3)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parsed_stdout = _parse_json_object(stdout)
    return {
        "command": shlex.join(command),
        "returncode": completed.returncode,
        "duration_seconds": duration_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "parsed_stdout": parsed_stdout,
    }


def _parse_json_object(stdout: str) -> dict[str, Any] | None:
    if not stdout:
        return None
    candidates = [stdout]
    last_brace = stdout.rfind("{")
    if last_brace > 0:
        candidates.append(stdout[last_brace:])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def capture_resume_cycle_artifact(
    *,
    repo_root: str | Path,
    registry_path: str | Path,
    state_root: str | Path,
    logs_root: str | Path,
    triton_cache_root: str | Path,
    artifact_path: str | Path,
    container_name: str = "lumo-vllm-sprint0-artifact",
) -> Path:
    repo_root = Path(repo_root).resolve()
    registry_path = Path(registry_path).resolve()
    state_root = Path(state_root).resolve()
    logs_root = Path(logs_root).resolve()
    triton_cache_root = Path(triton_cache_root).resolve()
    artifact_path = Path(artifact_path).resolve()

    state_store = RuntimeStateStore(state_root)
    pre_state = state_store.load()
    expected_tuned_config_path = pre_state.last_known_good_tuned_config_path
    expected_weight_version_id = pre_state.last_known_good_weight_version_id
    expected_model_id = pre_state.last_known_good_model_id or pre_state.current_model_id
    if expected_model_id is None:
        raise RuntimeError("Sprint 0 resume artifact requires a last-known-good model id in runtime state.")
    if expected_tuned_config_path is None:
        raise RuntimeError("Sprint 0 resume artifact requires a last-known-good tuned-config path in runtime state.")
    if expected_weight_version_id is None:
        raise RuntimeError("Sprint 0 resume artifact requires a last-known-good weight version id in runtime state.")

    command_prefix = [
        sys.executable,
        "-m",
        "lumo_flywheel_serving.cli",
        "--registry",
        str(registry_path),
        "--logs-root",
        str(logs_root),
        "--triton-cache-root",
        str(triton_cache_root),
        "--state-root",
        str(state_root),
        "--container-name",
        container_name,
    ]

    resume_result = _run_command([*command_prefix, "resume"], cwd=repo_root)
    status_result: dict[str, Any] | None = None
    stop_result: dict[str, Any] | None = None
    if resume_result["returncode"] == 0:
        status_result = _run_command([*command_prefix, "status"], cwd=repo_root)
    try:
        stop_result = _run_command([*command_prefix, "stop"], cwd=repo_root)
    except Exception:
        stop_result = None

    post_state = state_store.load()
    status_payload = status_result["parsed_stdout"] if status_result is not None else None
    served_model_ids: list[str] = []
    if isinstance(status_payload, dict):
        models = status_payload.get("models")
        if isinstance(models, dict):
            data = models.get("data")
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        served_model_ids.append(entry["id"])

    assertions = {
        "resume_command_succeeded": resume_result["returncode"] == 0,
        "status_command_succeeded": status_result is not None and status_result["returncode"] == 0,
        "status_health_200": isinstance(status_payload, dict) and status_payload.get("health") == 200,
        "served_model_contains_expected": expected_model_id in served_model_ids,
        "reloaded_expected_weight_version_id": post_state.current_weight_version_id == expected_weight_version_id,
        "reloaded_expected_tuned_config_path": post_state.active_tuned_config_path == expected_tuned_config_path,
        "runtime_state_ready": post_state.status == "READY",
    }
    payload = {
        "spec_item": "9.5.1",
        "generated_at": _isoformat_now(),
        **_git_metadata(repo_root),
        "inputs": {
            "registry_path": str(registry_path),
            "state_root": str(state_root),
            "logs_root": str(logs_root),
            "triton_cache_root": str(triton_cache_root),
            "container_name": container_name,
        },
        "expected_last_known_good": {
            "model_id": expected_model_id,
            "weight_version_id": expected_weight_version_id,
            "tuned_config_path": expected_tuned_config_path,
        },
        "pre_state": pre_state.as_dict(),
        "resume_result": resume_result,
        "status_result": status_result,
        "stop_result": stop_result,
        "post_state": post_state.as_dict(),
        "assertions": assertions,
        "pass": all(assertions.values()),
    }
    _write_json(artifact_path, payload)
    if not payload["pass"]:
        raise RuntimeError(f"Resume-cycle artifact failed verification. See {artifact_path}")
    return artifact_path


def _regression_candidate(
    *,
    baseline_vllm_config: dict[str, Any],
    workload: SyntheticWorkloadDistribution,
) -> dict[str, Any]:
    return {
        **baseline_vllm_config,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "max_num_batched_tokens": min(int(baseline_vllm_config["max_num_batched_tokens"]), 4096),
        "max_model_len": workload.p99_context_tokens,
    }


def _check_safety_rail(
    *,
    repo_root: Path,
    registry_path: Path,
    family_id: str,
    workload_file: Path,
    output_root: Path,
    artifact_path: Path,
    baseline_bundle_path: Path | None,
    rail_name: str,
    candidate_overrides: list[dict[str, Any]],
) -> Path:
    registry = load_registry(registry_path)
    model_config = registry["qwen3.5-27b"]
    workload = SyntheticWorkloadDistribution.from_file(
        workload_file,
        model_config=model_config,
        family_id=family_id,
    )
    baseline_bundle = load_baseline_bundle(baseline_bundle_path)
    runner = OfflineAutoResearchRunner(
        model_config=model_config,
        family_id=family_id,
        output_root=output_root,
        workload=workload,
        baseline_bundle=baseline_bundle,
        iteration_cap=len(candidate_overrides),
        candidate_overrides=candidate_overrides,
    )
    result = runner.run()
    search_trace = json.loads(result.search_trace_path.read_text(encoding="utf-8"))
    run_log = json.loads(result.run_log_path.read_text(encoding="utf-8"))

    assertions: dict[str, bool]
    if rail_name == "regression_guard":
        baseline_value = int(result.baseline_value)
        non_baseline_values = [int(entry["objective_value"]) for entry in search_trace[1:] if isinstance(entry, dict)]
        assertions = {
            "baseline_retained": result.status == "retained_baseline",
            "no_bundle_emitted": result.bundle_path is None,
            "candidate_did_not_beat_baseline": all(value <= baseline_value for value in non_baseline_values),
        }
    elif rail_name == "determinism_check":
        determinism_failures = [
            entry
            for entry in search_trace[1:]
            if isinstance(entry, dict) and entry.get("reason") == "determinism_check_failed"
        ]
        assertions = {
            "run_aborted_on_determinism_failures": result.stopping_reason == "hard_infeasibility_determinism",
            "three_failures_recorded": len(determinism_failures) >= 3,
        }
    elif rail_name == "oom_handling":
        oom_failures = [entry for entry in search_trace[1:] if isinstance(entry, dict) and entry.get("reason") == "oom"]
        assertions = {
            "run_aborted_on_oom_streak": result.stopping_reason == "hard_infeasibility_oom",
            "three_oom_failures_recorded": len(oom_failures) >= 3,
        }
    else:
        raise ValueError(f"Unknown safety rail {rail_name!r}")

    payload = {
        "spec_item": "9.3.3",
        "rail": rail_name,
        "generated_at": _isoformat_now(),
        **_git_metadata(repo_root),
        "baseline_bundle_path": str(baseline_bundle_path) if baseline_bundle_path is not None else None,
        "candidate_overrides": candidate_overrides,
        "run_result": {
            "status": result.status,
            "stopping_reason": result.stopping_reason,
            "run_dir": str(result.run_dir),
            "search_trace_path": str(result.search_trace_path),
            "measurement_trace_path": str(result.measurement_trace_path),
            "run_log_path": str(result.run_log_path),
            "bundle_path": str(result.bundle_path) if result.bundle_path is not None else None,
            "baseline_value": result.baseline_value,
            "best_value": result.best_value,
            "best_candidate_label": result.best_candidate_label,
        },
        "run_log": run_log,
        "search_trace_excerpt": search_trace[:4],
        "assertions": assertions,
        "pass": all(assertions.values()),
    }
    _write_json(artifact_path, payload)
    if not payload["pass"]:
        raise RuntimeError(f"Safety-rail artifact {rail_name} failed verification. See {artifact_path}")
    return artifact_path


def generate_sprint0_closeout_artifacts(
    *,
    repo_root: str | Path,
    registry_path: str | Path,
    state_root: str | Path,
    logs_root: str | Path,
    triton_cache_root: str | Path,
    workload_file: str | Path,
    artifacts_root: str | Path,
    family_id: str = "proposal-ranking-manager-judgment",
    baseline_bundle_path: str | Path | None = None,
    container_name: str = "lumo-vllm-sprint0-artifact",
) -> dict[str, Path]:
    repo_root = Path(repo_root).resolve()
    registry_path = Path(registry_path).resolve()
    state_root = Path(state_root).resolve()
    logs_root = Path(logs_root).resolve()
    triton_cache_root = Path(triton_cache_root).resolve()
    workload_file = Path(workload_file).resolve()
    artifacts_root = Path(artifacts_root).resolve()

    state = RuntimeStateStore(state_root).load()
    resolved_baseline_bundle_path = (
        Path(baseline_bundle_path).resolve()
        if baseline_bundle_path is not None
        else (
            Path(state.last_known_good_tuned_config_path).resolve()
            if state.last_known_good_tuned_config_path is not None
            else None
        )
    )
    if resolved_baseline_bundle_path is None:
        raise RuntimeError("Sprint 0 safety-rail artifacts require a baseline tuned-config bundle.")

    resume_artifact = capture_resume_cycle_artifact(
        repo_root=repo_root,
        registry_path=registry_path,
        state_root=state_root,
        logs_root=logs_root,
        triton_cache_root=triton_cache_root,
        artifact_path=artifacts_root / "resume_cycle" / "recorded_session.json",
        container_name=container_name,
    )

    baseline_bundle = load_baseline_bundle(resolved_baseline_bundle_path)
    if baseline_bundle is None:
        raise RuntimeError("Expected a loadable baseline tuned-config bundle for Sprint 0 closeout.")
    workload = SyntheticWorkloadDistribution.from_file(
        workload_file,
        model_config=load_registry(registry_path)["qwen3.5-27b"],
        family_id=family_id,
    )
    baseline_vllm_config = dict(baseline_bundle.vllm_config)
    safety_output_root = artifacts_root / "auto_research_runs"
    regression_guard_artifact = _check_safety_rail(
        repo_root=repo_root,
        registry_path=registry_path,
        family_id=family_id,
        workload_file=workload_file,
        output_root=safety_output_root / "regression_guard",
        artifact_path=artifacts_root / "safety_rails" / "regression_guard.json",
        baseline_bundle_path=resolved_baseline_bundle_path,
        rail_name="regression_guard",
        candidate_overrides=[_regression_candidate(baseline_vllm_config=baseline_vllm_config, workload=workload)],
    )
    determinism_artifact = _check_safety_rail(
        repo_root=repo_root,
        registry_path=registry_path,
        family_id=family_id,
        workload_file=workload_file,
        output_root=safety_output_root / "determinism_check",
        artifact_path=artifacts_root / "safety_rails" / "determinism_check.json",
        baseline_bundle_path=resolved_baseline_bundle_path,
        rail_name="determinism_check",
        candidate_overrides=[
            {**baseline_vllm_config, "harness_overrides": {"inject_nondeterminism": True}}
            for _ in range(3)
        ],
    )
    oom_artifact = _check_safety_rail(
        repo_root=repo_root,
        registry_path=registry_path,
        family_id=family_id,
        workload_file=workload_file,
        output_root=safety_output_root / "oom_handling",
        artifact_path=artifacts_root / "safety_rails" / "oom_handling.json",
        baseline_bundle_path=resolved_baseline_bundle_path,
        rail_name="oom_handling",
        candidate_overrides=[{**baseline_vllm_config, "harness_overrides": {"force_oom": True}} for _ in range(3)],
    )

    manifest_path = _write_json(
        artifacts_root / "manifest.json",
        {
            "generated_at": _isoformat_now(),
            **_git_metadata(repo_root),
            "artifacts": {
                "resume_cycle": str(resume_artifact),
                "regression_guard": str(regression_guard_artifact),
                "determinism_check": str(determinism_artifact),
                "oom_handling": str(oom_artifact),
            },
        },
    )
    return {
        "resume_cycle": resume_artifact,
        "regression_guard": regression_guard_artifact,
        "determinism_check": determinism_artifact,
        "oom_handling": oom_artifact,
        "manifest": manifest_path,
    }

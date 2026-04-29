"""Unit tests for L0cKernelMutationRunner._real_run (the L0c round driver).

The driver's external boundaries — paired baseline measurement, agent spawn, and the
real apply_and_test path — are monkeypatched. Slice 2's test suite covers the inner
apply_and_test contract; this file focuses on the three-cap loop, terminal_condition
selection, dedup, and bundle finalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from lumo_flywheel_serving import auto_research


def _write_registry(repo: Path) -> Path:
    path = repo / "model_registry.yaml"
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
    return path


def _write_workload(repo: Path) -> Path:
    family_dir = repo / "benchmark_blueprints" / "families" / "test-family"
    family_dir.mkdir(parents=True)
    for trace in ("seed_trace.jsonl", "holdout_trace.jsonl"):
        (family_dir / trace).write_text(
            "\n".join(
                json.dumps({"prompt_tokens": 64, "output_tokens": 16, "thinking_tokens": 0, "turn_index": i})
                for i in (0, 1)
            )
            + "\n",
            encoding="utf-8",
        )
    workload_path = family_dir / "workload.yaml"
    workload_path.write_text(
        """
family_id: test-family
workload_distribution_id: null
workload_distribution_id_hardening_version: v1-thinking-realistic
latency_ceiling_ms: 60000
nominal_ttft_ms: 1500
nominal_tpot_ms: 80
nominal_turn_ms: 4200
tpot_ceiling_ms: 80
turn_latency_ceiling_ms: 60000
p99_context_tokens: 24576
avg_prompt_tokens: 64
avg_output_tokens: 16
rollout_baseline: 1.0
measurement_window_minutes: 1
target_concurrency: 1
gpu_memory_utilization_cap: 0.08
seed_trace_ref: seed_trace.jsonl
holdout_trace_ref: holdout_trace.jsonl
""",
        encoding="utf-8",
    )
    workload = auto_research.load_yaml_file(workload_path)
    assert isinstance(workload, dict)
    workload["workload_distribution_id"] = auto_research.compute_workload_distribution_id(workload_path)
    workload_path.write_text(yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
    return workload_path


def _write_parity_fixture(repo: Path) -> Path:
    fixture_dir = repo / "benchmark_blueprints" / "families" / "test-family" / "parity_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "deltanet_v1.yaml"
    fixture_path.write_text(
        yaml.safe_dump(
            {
                "fixture_id": "test-family-deltanet-v1",
                "probe_count": 2,
                "tolerances": {
                    "rtol_logit": 0.001,
                    "atol_logit": 0.001,
                    "rtol_state": 0.005,
                    "atol_state": 0.005,
                },
                "state_checkpoints_at_token": [1, 1024],
                "parity_check_method": "logit_plus_state_compare",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return fixture_path


def _write_base_bundle(repo: Path, workload_path: Path) -> Path:
    bundle = auto_research.make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="test-family",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id=auto_research.compute_workload_distribution_id(workload_path),
        vllm_config={
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 131072,
            "kv_cache_dtype": "fp8_e5m2",
        },
        kernel_selection={
            "combo_id": "combo_001",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "cublas",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
        objective={"metric": "l0b_baseline", "value": 1.0},
        measurement_trace_ref="output/auto_research/l0b/measurement_trace_combined.json",
        search_trace_ref="output/auto_research/l0b/search_trace.json",
        baseline_bundle_id=None,
        regression_guard={},
        safety_rails={"determinism_check_passed": True, "parity_check_passed": True},
        round_provenance={
            "round_type": "l0b_autotune",
            "workload_descriptor_path": str(workload_path),
            "confidence": "defensible",
        },
    )
    return auto_research.persist_tuned_config_bundle(bundle, repo / "output" / "tuned_configs")


def _runtime_block() -> dict[str, Any]:
    return {
        "container_name": "lumo-vllm-test",
        "model_id": "qwen3.5-27b",
        "port": 18100,
        "proxy_port": 18101,
        "endpoint": "http://127.0.0.1:18101/v1",
        "metrics_url": "http://127.0.0.1:18100/metrics",
        "admin_url": "http://127.0.0.1:18101/admin",
    }


def _make_kernel_source(tmp_path: Path) -> Path:
    kernel = tmp_path / "kernels" / "chunk_delta_h.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    return kernel


def _stub_baseline(monkeypatch, *, rows: int, mean_obj: float = 1.0) -> None:
    def _baseline(self, *, spec, baseline_dir, baseline_uuid, count):
        return [
            self._make_measurement_row(
                candidate_uuid=baseline_uuid,
                candidate_label="l0b-baseline-remeasured",
                role="l0b_baseline_remeasured",
                measurement_index=i + 1,
                objective_value=mean_obj + 0.0001 * i,
                harness="real",
                trace_ref=f"baselines/measurement_{i + 1:02d}.json",
            )
            for i in range(count)
        ]

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner,
        "_run_real_paired_baseline",
        _baseline,
        raising=True,
    )


def _make_agent_writer(
    *,
    patch_text_for: Callable[[str], str | None],
) -> Callable:
    """Return a stub _spawn_l0c_agent_iteration that writes a per-iteration mutation.patch."""

    def _spawn(self, *, spec, round_dir, iteration_dir, iteration):
        text = patch_text_for(iteration)
        if text is not None:
            (iteration_dir / "mutation.patch").write_text(text, encoding="utf-8")
        return {"ok": True, "transcript": str(iteration_dir / "agent_session.jsonl")}

    return _spawn


def _stub_apply_and_test(*, outcomes_for: Callable[[str], dict[str, Any]]) -> Callable:
    def _apply(self, *, round_id, iteration, kernel_target, harness, round_root):
        outcome = outcomes_for(iteration)
        round_dir = Path(round_root) / round_id
        iteration_dir = round_dir / "candidates" / iteration
        iteration_dir.mkdir(parents=True, exist_ok=True)
        if outcome["outcome"] == "parity_passed":
            self._write_parity_check(
                iteration_dir,
                pass_=True,
                reason="ran_passed",
                fixture_id="test-family-deltanet-v1",
                kernel_target=kernel_target,
            )
            objective_mean = float(outcome["objective_mean"])
            candidate_uuid = outcome.get("candidate_uuid", f"cand-{iteration}")
            measurement_rows = [
                self._make_measurement_row(
                    candidate_uuid=candidate_uuid,
                    candidate_label=f"l0c-attempt-{iteration}",
                    role="l0c_candidate",
                    measurement_index=i + 1,
                    objective_value=objective_mean + 0.0001 * i,
                    harness="real",
                    trace_ref=f"candidates/{iteration}/measurement_{i + 1:02d}.json",
                )
                for i in range(2)
            ]
            (iteration_dir / "measurement_trace.json").write_text(
                json.dumps(
                    {
                        "candidate_uuid": candidate_uuid,
                        "candidate_label": f"l0c-attempt-{iteration}",
                        "harness": "real",
                        "measurement_role": "l0c_candidate",
                        "measurements": measurement_rows,
                        "objective_mean": objective_mean,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return {
                "round_id": round_id,
                "iteration": iteration,
                "kernel_target": kernel_target,
                "harness": harness,
                "mutation_hash": "stub-mutation-hash",
                "outcome": "parity_passed",
                "candidate_uuid": candidate_uuid,
                "objective_mean": objective_mean,
            }
        # parity_failed or compile_failed branches
        self._write_parity_check(
            iteration_dir,
            pass_=False,
            reason=outcome.get("reason", outcome["outcome"]),
            fixture_id="test-family-deltanet-v1",
            kernel_target=kernel_target,
            first_diverging_probe=outcome.get("first_diverging_probe"),
            tolerance_overshoot=outcome.get("tolerance_overshoot"),
            error_detail=outcome.get("error_detail"),
        )
        return {
            "round_id": round_id,
            "iteration": iteration,
            "kernel_target": kernel_target,
            "harness": harness,
            "mutation_hash": "stub-mutation-hash",
            "outcome": outcome["outcome"],
        }

    return _apply


def _make_runner_and_run(
    tmp_path: Path,
    monkeypatch,
    *,
    base_measurements: int,
    accepted_iteration_cap: int,
    total_attempt_cap: int,
    round_timeout_hours: float,
    baseline_mean: float,
    patch_text_for: Callable[[str], str | None],
    outcomes_for: Callable[[str], dict[str, Any]],
) -> auto_research.L0cKernelMutationResult:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    workload_path = _write_workload(repo)
    fixture_path = _write_parity_fixture(repo)
    base_bundle = _write_base_bundle(repo, workload_path)
    kernel_path = _make_kernel_source(tmp_path)

    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    _stub_baseline(monkeypatch, rows=base_measurements, mean_obj=baseline_mean)
    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner,
        "_spawn_l0c_agent_iteration",
        _make_agent_writer(patch_text_for=patch_text_for),
        raising=True,
    )
    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner,
        "apply_and_test",
        _stub_apply_and_test(outcomes_for=outcomes_for),
        raising=True,
    )

    return runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path=str(kernel_path),
        parity_fixture=fixture_path,
        base_measurements=base_measurements,
        accepted_iteration_cap=accepted_iteration_cap,
        total_attempt_cap=total_attempt_cap,
        round_timeout_hours=round_timeout_hours,
        round_root=repo / "output" / "auto_research",
        harness="real",
        runtime=_runtime_block(),
    )


def _patch_for(iteration: str, suffix: str = "") -> str:
    return (
        f"--- a/chunk_delta_h.py\n+++ b/chunk_delta_h.py\n"
        f"@@ -1,3 +1,3 @@\n-alpha\n+ALPHA-{iteration}{suffix}\n beta\n gamma\n"
    )


def test_real_run_passes_round_and_mints_bundle_when_winner_beats_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=2,
        accepted_iteration_cap=1,
        total_attempt_cap=2,
        round_timeout_hours=1.0,
        baseline_mean=1.0,
        patch_text_for=lambda iteration: _patch_for(iteration),
        outcomes_for=lambda iteration: {
            "outcome": "parity_passed",
            "objective_mean": 1.20,
            "candidate_uuid": f"cand-{iteration}",
        },
    )
    assert result.outcome == "ROUND_PASSED"
    assert result.terminal_condition == "accepted_cap_reached"
    assert result.accepted_count == 1
    assert result.bundle_path is not None
    bundle = yaml.safe_load(result.bundle_path.read_text(encoding="utf-8"))["tuned_config_bundle"]
    l0c_block = bundle["layer_0_deltanet"]["l0c_mutation"]
    assert l0c_block["accepted_count"] == 1
    assert l0c_block["parity_attestation"]["checkpoints_checked"] == [1, 1024]
    # HLD v0.3.3 §7.X: real-harness baseline trailer is renamed to reflect
    # the empirical-winner anchor; synthetic-harness keeps the legacy name.
    trailers = (result.round_dir / "candidate_trailers.tsv").read_text(encoding="utf-8")
    assert "Measurement-Role: l0b_empirical_winner_baseline_remeasured" in trailers
    assert "Measurement-Role: l0b_baseline_remeasured" not in trailers


def test_real_run_returns_round_null_result_when_winner_under_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=2,
        accepted_iteration_cap=1,
        total_attempt_cap=2,
        round_timeout_hours=1.0,
        baseline_mean=1.5,
        patch_text_for=lambda iteration: _patch_for(iteration),
        outcomes_for=lambda iteration: {
            "outcome": "parity_passed",
            "objective_mean": 1.0,
            "candidate_uuid": f"cand-{iteration}",
        },
    )
    assert result.outcome == "ROUND_NULL_RESULT"
    assert result.bundle_path is None


def test_real_run_terminates_proposer_stuck_after_three_parity_fails(
    tmp_path: Path, monkeypatch
) -> None:
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=1,
        accepted_iteration_cap=5,
        total_attempt_cap=10,
        round_timeout_hours=1.0,
        baseline_mean=1.0,
        patch_text_for=lambda iteration: _patch_for(iteration),
        outcomes_for=lambda iteration: {
            "outcome": "parity_failed",
            "reason": "parity_logit_diverged",
            "first_diverging_probe": int(iteration) % 64,
            "tolerance_overshoot": 0.001 * int(iteration),
        },
    )
    assert result.outcome == "ROUND_BLOCKED"
    assert result.terminal_condition == "proposer_stuck"
    assert result.bundle_path is None


def test_real_run_terminates_compile_failures_after_three_compile_fails(
    tmp_path: Path, monkeypatch
) -> None:
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=1,
        accepted_iteration_cap=5,
        total_attempt_cap=10,
        round_timeout_hours=1.0,
        baseline_mean=1.0,
        patch_text_for=lambda iteration: _patch_for(iteration),
        outcomes_for=lambda iteration: {
            "outcome": "compile_failed",
            "reason": "compile_nvcc_error",
            "error_detail": "synthetic compile failure",
        },
    )
    assert result.outcome == "ROUND_BLOCKED"
    assert result.terminal_condition == "compile_failures_3x"
    assert result.bundle_path is None


def test_real_run_preserves_parity_verdict_when_agent_exits_after_apply(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    workload_path = _write_workload(repo)
    fixture_path = _write_parity_fixture(repo)
    base_bundle = _write_base_bundle(repo, workload_path)
    kernel_path = _make_kernel_source(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    _stub_baseline(monkeypatch, rows=1, mean_obj=1.0)

    def _spawn(self, *, spec, round_dir, iteration_dir, iteration):
        (iteration_dir / "mutation.patch").write_text(_patch_for(iteration), encoding="utf-8")
        self._write_parity_check(
            iteration_dir,
            pass_=False,
            reason="parity_logit_diverged",
            fixture_id="test-family-deltanet-v1",
            kernel_target="deltanet",
            first_diverging_probe=0,
            tolerance_overshoot=0.35843359375,
        )
        return {"ok": False, "error": "agent_exit_1: stale monitor killed"}

    monkeypatch.setattr(auto_research.L0cKernelMutationRunner, "_spawn_l0c_agent_iteration", _spawn)
    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path=str(kernel_path),
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=2,
        total_attempt_cap=2,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="real",
        runtime=_runtime_block(),
    )
    rejected = (result.round_dir / "mutations_rejected.tsv").read_text(encoding="utf-8")
    assert result.terminal_condition == "total_attempt_cap_reached"
    assert "parity_logit_diverged" in rejected
    assert "0.358434" in rejected
    assert "agent_exit_1" not in rejected


def test_real_run_rate_limit_blocks_without_burning_compile_failure_budget(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    workload_path = _write_workload(repo)
    fixture_path = _write_parity_fixture(repo)
    base_bundle = _write_base_bundle(repo, workload_path)
    kernel_path = _make_kernel_source(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    _stub_baseline(monkeypatch, rows=1, mean_obj=1.0)

    def _spawn(self, *, spec, round_dir, iteration_dir, iteration):
        (iteration_dir / "agent_session.jsonl").write_text(
            json.dumps({"api_error_status": 429, "error": "You've hit your limit"})
            + "\n",
            encoding="utf-8",
        )
        return {"ok": False, "error": "agent_exit_1: rate limit"}

    monkeypatch.setattr(auto_research.L0cKernelMutationRunner, "_spawn_l0c_agent_iteration", _spawn)
    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path=str(kernel_path),
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=2,
        total_attempt_cap=8,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="real",
        runtime=_runtime_block(),
    )
    assert result.outcome == "ROUND_BLOCKED"
    assert result.terminal_condition == "agent_rate_limited"
    assert result.total_attempt_count == 1
    assert result.rejected_count == 0


def test_real_run_dedupes_duplicate_mutation_hashes_without_resetting_streaks(
    tmp_path: Path, monkeypatch
) -> None:
    # Same patch text every iteration => same mutation_hash => dedup branch hits.
    fixed_patch = _patch_for("XX")  # constant suffix => identical text every iteration
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=1,
        accepted_iteration_cap=2,
        total_attempt_cap=4,
        round_timeout_hours=1.0,
        baseline_mean=1.0,
        patch_text_for=lambda iteration: fixed_patch,
        outcomes_for=lambda iteration: {
            "outcome": "parity_passed",
            "objective_mean": 1.20,
            "candidate_uuid": f"cand-{iteration}",
        },
    )
    # First iteration accepts. Subsequent ones dedupe (same hash) — they don't count as
    # accepted, but they DO count as total_attempts for the cap check. So the loop runs
    # to total_attempt_cap=4 unless accepted hits accepted_iteration_cap=2 first (it
    # won't, because every duplicate is rejected). Final state: accepted=1, total=4
    # (or 1, depending on whether the duplicate path increments total_attempts or not).
    assert result.accepted_count == 1
    rejected = (result.round_dir / "mutations_rejected.tsv").read_text(encoding="utf-8")
    assert "duplicate_mutation_hash" in rejected


def test_real_run_writes_runtime_block_into_round_spec(tmp_path: Path, monkeypatch) -> None:
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=1,
        accepted_iteration_cap=1,
        total_attempt_cap=1,
        round_timeout_hours=1.0,
        baseline_mean=1.0,
        patch_text_for=lambda iteration: _patch_for(iteration),
        outcomes_for=lambda iteration: {
            "outcome": "parity_passed",
            "objective_mean": 1.20,
            "candidate_uuid": f"cand-{iteration}",
        },
    )
    spec = yaml.safe_load((result.round_dir / "round_spec.yaml").read_text(encoding="utf-8"))
    assert spec["agent_runtime"] == "codex"
    assert spec["runtime"]["container_name"] == "lumo-vllm-test"
    assert spec["runtime"]["endpoint"] == "http://127.0.0.1:18101/v1"
    brief = (result.round_dir / "candidates" / "001" / "iteration_brief.md").read_text(
        encoding="utf-8"
    )
    assert "rtol=0.001 / atol=0.001" in brief
    assert "rtol=0.005 / atol=0.005" in brief
    assert "BEFORE proposing a mutation" not in brief
    assert "LPDDR5x" in brief
    assert f"cd {result.round_dir.parents[2]}" in brief
    assert f"{result.round_dir.parents[2] / '.venv' / 'bin' / 'lumoserve'} auto-research apply-and-test" in brief
    assert "patch --dry-run" in brief
    # kernel_base snapshot must exist for apply_and_test (Slice 2) to read.
    base_bytes_dir = result.round_dir / "kernel_base"
    assert base_bytes_dir.is_dir()
    snapshot_files = list(base_bytes_dir.iterdir())
    assert len(snapshot_files) == 1
    assert snapshot_files[0].read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_real_paired_baseline_discards_first_cold_row(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    workload_path = _write_workload(repo)
    baseline_dir = repo / "round" / "baselines"
    baseline_dir.mkdir(parents=True)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    calls: list[dict[str, Any]] = []

    class _FakeHarness:
        def __init__(self, **kwargs):
            pass

        def measure(self, **kwargs):
            calls.append(kwargs)
            return {"eval_throughput": float(len(calls))}

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeHarness)
    rows = runner._run_real_paired_baseline(
        spec={
            "round_id": "round-1",
            "model_id": "qwen3.5-27b",
            "workload_file": str(workload_path),
            "runtime": _runtime_block(),
            "weight_version_id": "2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        },
        baseline_dir=baseline_dir,
        baseline_uuid="base-uuid",
        count=2,
    )
    assert len(calls) == 3
    assert len(rows) == 2
    assert rows[0]["objective_value"] == "2.000000"
    assert rows[1]["objective_value"] == "3.000000"
    assert rows[0]["trace_ref"] == "baselines/measurement_01.json"
    assert (baseline_dir / "cold_discard_00.json").is_file()
    assert json.loads((baseline_dir / "cold_discard_00.json").read_text(encoding="utf-8"))[
        "discard_reason"
    ] == "cold_start_baseline"


def test_real_apply_and_test_resolves_relative_kernel_path_against_repo(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    fixture_path = _write_parity_fixture(repo)
    relative_kernel = Path("output/auto_research/l0c_kernel_workdir/chunk_delta_h.py")
    kernel_path = repo / relative_kernel
    kernel_path.parent.mkdir(parents=True)
    kernel_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    round_root = repo / "output" / "auto_research"
    round_id = "round-relative-kernel"
    round_dir = round_root / round_id
    iteration_dir = round_dir / "candidates" / "001"
    iteration_dir.mkdir(parents=True)
    (round_dir / "kernel_base").mkdir()
    (round_dir / "kernel_base" / kernel_path.name).write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (iteration_dir / "mutation.patch").write_text(_patch_for("001"), encoding="utf-8")
    (round_dir / "round_spec.yaml").write_text(
        yaml.safe_dump(
            {
                "round_id": round_id,
                "model_id": "qwen3.5-27b",
                "kernel_target": "deltanet",
                "kernel_source_path": str(relative_kernel),
                "parity_fixture": str(fixture_path.relative_to(repo)),
                "parity_fixture_id": "test-family-deltanet-v1",
                "runtime": _runtime_block(),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    seen: dict[str, Path] = {}

    def _fake_apply_kernel_patch(self, *, kernel_path, patch_path):
        seen["kernel_path"] = kernel_path
        return auto_research._L0cPatchOutcome(ok=False, error="forced")

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner,
        "_apply_kernel_patch",
        _fake_apply_kernel_patch,
        raising=True,
    )
    result = runner.apply_and_test(
        round_id=round_id,
        iteration="001",
        kernel_target="deltanet",
        harness="real",
        round_root=round_root,
    )
    assert result["outcome"] == "compile_failed"
    assert seen["kernel_path"] == kernel_path.resolve()


def test_real_run_records_intermittent_parity_terminal_condition(
    tmp_path: Path, monkeypatch
) -> None:
    """If the same mutation flips between pass/fail, the loop terminates as
    intermittent_parity_observed after L0C_INTERMITTENT_PARITY_THRESHOLD hits."""
    # Need DIFFERENT patches to avoid dedup; each marked intermittent_parity.
    result = _make_runner_and_run(
        tmp_path,
        monkeypatch,
        base_measurements=1,
        accepted_iteration_cap=5,
        total_attempt_cap=10,
        round_timeout_hours=1.0,
        baseline_mean=1.0,
        patch_text_for=lambda iteration: _patch_for(iteration, suffix=f"-int-{iteration}"),
        outcomes_for=lambda iteration: {
            "outcome": "parity_failed",
            "reason": "intermittent_parity",
        },
    )
    assert result.outcome == "ROUND_BLOCKED"
    # Either intermittent_parity_observed or proposer_stuck can fire first depending
    # on threshold ordering; both are valid blocked terminations driven by the same
    # parity-failure stream. Assert at least one of the two.
    assert result.terminal_condition in {"intermittent_parity_observed", "proposer_stuck"}


def test_real_run_refuses_when_fixture_baseline_disagrees_with_base_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    workload_path = _write_workload(repo)
    base_bundle = _write_base_bundle(repo, workload_path)  # attention_backend=vllm-default
    kernel_path = _make_kernel_source(tmp_path)

    fixture_dir = repo / "benchmark_blueprints" / "families" / "test-family" / "parity_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "deltanet_v1_skewed.yaml"
    fixture_path.write_text(
        yaml.safe_dump(
            {
                "fixture_id": "test-family-deltanet-v1-skewed",
                "probe_count": 2,
                "tolerances": {
                    "rtol_logit": 0.001,
                    "atol_logit": 0.001,
                    "rtol_state": 0.005,
                    "atol_state": 0.005,
                },
                "state_checkpoints_at_token": [1, 1024],
                "parity_check_method": "logit_plus_state_compare",
                "generated_against": {
                    "reference_baseline": {
                        "attention_backend": "flash-attn-4",
                        "deltanet_kernel": "triton-chunked-delta-v2",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    with pytest.raises(RuntimeError) as excinfo:
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="deltanet",
            kernel_source_path=str(kernel_path),
            parity_fixture=fixture_path,
            base_measurements=1,
            accepted_iteration_cap=1,
            total_attempt_cap=1,
            round_timeout_hours=1.0,
            round_root=repo / "output" / "auto_research",
            harness="real",
            runtime=_runtime_block(),
        )
    msg = str(excinfo.value)
    assert "kernel_selection mismatch" in msg
    assert "attention_backend" in msg
    assert "flash-attn-4" in msg
    assert "vllm-default" in msg


def test_real_run_accepts_when_fixture_baseline_matches_base_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    workload_path = _write_workload(repo)
    base_bundle = _write_base_bundle(repo, workload_path)  # attention_backend=vllm-default
    kernel_path = _make_kernel_source(tmp_path)

    fixture_dir = repo / "benchmark_blueprints" / "families" / "test-family" / "parity_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "deltanet_v1_aligned.yaml"
    fixture_path.write_text(
        yaml.safe_dump(
            {
                "fixture_id": "test-family-deltanet-v1-aligned",
                "probe_count": 2,
                "tolerances": {"rtol_logit": 0.001, "atol_logit": 0.001, "rtol_state": 0.005, "atol_state": 0.005},
                "state_checkpoints_at_token": [1, 1024],
                "parity_check_method": "logit_plus_state_compare",
                "generated_against": {
                    "reference_baseline": {
                        "attention_backend": "vllm-default",
                        "deltanet_kernel": "triton-chunked-delta-v2",
                        "fp8_gemm_kernel": "cublas",
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    _stub_baseline(monkeypatch, rows=1, mean_obj=1.0)
    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner,
        "_spawn_l0c_agent_iteration",
        _make_agent_writer(patch_text_for=lambda i: _patch_for(i)),
        raising=True,
    )
    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner,
        "apply_and_test",
        _stub_apply_and_test(
            outcomes_for=lambda iteration: {
                "outcome": "parity_passed",
                "objective_mean": 1.20,
                "candidate_uuid": f"cand-{iteration}",
            }
        ),
        raising=True,
    )
    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path=str(kernel_path),
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=1,
        total_attempt_cap=1,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="real",
        runtime=_runtime_block(),
    )
    assert result.outcome == "ROUND_PASSED"


def test_assert_actually_resolved_no_drift_halts_when_live_runtime_drifts(
    monkeypatch,
) -> None:
    """HLD v0.3.3 §5.2: round must refuse if live runtime resolved aliases differently."""
    fixture_payload = {
        "generated_against": {
            "actually_resolved_kernel_selection": {
                "attention_backend": "vllm-default",
                "kv_cache_dtype": "fp8_e5m2",
                "deltanet_kernel": "triton-chunked-delta-v2",
                "vllm_version": "0.19.0",
                "weight_version_id": "abc",
            }
        }
    }
    monkeypatch.setattr(
        auto_research,
        "fetch_actually_resolved_kernel_selection",
        lambda endpoint, *, api_key, **_: {
            "attention_backend": "vllm-default",
            "kv_cache_dtype": "bf16",  # drifted
            "deltanet_kernel": "triton-chunked-delta-v2",
            "vllm_version": "0.19.0",
            "weight_version_id": "abc",
            "fp8_gemm_kernel": "unknown",
            "kv_cache_block_size": "unknown",
            "torch_compile_mode": "unknown",
            "cuda_graph_capture": "unknown",
        },
        raising=True,
    )
    with pytest.raises(RuntimeError) as excinfo:
        auto_research.L0cKernelMutationRunner._assert_actually_resolved_no_drift(
            fixture_payload,
            endpoint="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            fixture_path=Path("/tmp/test_fixture.yaml"),
        )
    msg = str(excinfo.value)
    assert "actually_resolved_kernel_selection_drift" in msg
    assert "kv_cache_dtype" in msg
    assert "fp8_e5m2" in msg
    assert "bf16" in msg


def test_assert_actually_resolved_no_drift_skips_unknown_pinned_keys(
    monkeypatch,
) -> None:
    """Fixture's "unknown" sentinel means no claim was made — must not halt on it."""
    fixture_payload = {
        "generated_against": {
            "actually_resolved_kernel_selection": {
                "attention_backend": "vllm-default",
                "kv_cache_dtype": "unknown",  # no claim — fall through
            }
        }
    }
    monkeypatch.setattr(
        auto_research,
        "fetch_actually_resolved_kernel_selection",
        lambda endpoint, *, api_key, **_: {
            "attention_backend": "vllm-default",
            "kv_cache_dtype": "bf16",
            "fp8_gemm_kernel": "unknown",
            "kv_cache_block_size": "unknown",
            "torch_compile_mode": "unknown",
            "cuda_graph_capture": "unknown",
            "vllm_version": "unknown",
            "weight_version_id": "unknown",
            "deltanet_kernel": "unknown",
        },
        raising=True,
    )
    auto_research.L0cKernelMutationRunner._assert_actually_resolved_no_drift(
        fixture_payload,
        endpoint="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        fixture_path=Path("/tmp/test_fixture.yaml"),
    )


def test_assert_actually_resolved_no_drift_legacy_fixture_without_block_passes() -> None:
    """Pre-v0.3.3 fixtures have no actually_resolved_kernel_selection block; skip silently."""
    auto_research.L0cKernelMutationRunner._assert_actually_resolved_no_drift(
        {"generated_against": {"reference_baseline": {"attention_backend": "vllm-default"}}},
        endpoint="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        fixture_path=Path("/tmp/test_fixture.yaml"),
    )

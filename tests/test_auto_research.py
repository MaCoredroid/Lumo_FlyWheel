from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from lumo_flywheel_serving import auto_research


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


def _write_trace(path: Path, *, prompt_tokens: int, output_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "thinking_tokens": 0,
            "turn_index": 0,
        },
        {
            "prompt_tokens": prompt_tokens // 2,
            "output_tokens": max(1, output_tokens // 2),
            "thinking_tokens": 0,
            "turn_index": 1,
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _write_workload(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_trace(path.parent / "seed_trace.jsonl", prompt_tokens=4096, output_tokens=1200)
    _write_trace(path.parent / "holdout_trace.jsonl", prompt_tokens=3072, output_tokens=900)
    path.write_text(
        """
family_id: proposal-ranking-manager-judgment
workload_distribution_id: prmj-v1-live
latency_ceiling_ms: 35000
tpot_ceiling_ms: 80
turn_latency_ceiling_ms: 35000
p99_context_tokens: 24576
avg_prompt_tokens: 4096
avg_output_tokens: 1200
rollout_baseline: 10.0
measurement_window_minutes: 25
gpu_memory_utilization_cap: 0.08
seed_trace_ref: seed_trace.jsonl
holdout_trace_ref: holdout_trace.jsonl
""",
        encoding="utf-8",
    )


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo / "model_registry.yaml")
    _write_workload(
        repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"
    )
    fixture_src = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_measurement.py"
    fixture_dst = repo / "tests" / "fixtures" / "synthetic_measurement.py"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture_src, fixture_dst)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


def test_bootstrap_measure_commit_finalize_round(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def fake_measure(self, candidate_vllm_config, *, warmup_s, window_s, target_concurrency_sweep):
        del self, candidate_vllm_config, warmup_s, window_s, target_concurrency_sweep
        return {
            "generator": "RealMeasurementHarness v0.1.0",
            "candidate_vllm_config": {
                "max_num_seqs": 4,
                "max_num_batched_tokens": 8192,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "gpu_memory_utilization": 0.9,
                "max_model_len": 131072,
                "kv_cache_dtype": "fp8_e5m2",
            },
            "resolved": {
                "attention_backend": "flash-attn-4",
                "deltanet_kernel": "triton-chunked-delta-v2",
                "torch_compile_mode": "default",
            },
            "cache_isolation": {
                "cache_salt": "",
                "prefix_cache_reset_at_bootstrap": True,
                "first_10_req_prefix_cache_hit_rate": 0.02,
                "last_10_req_prefix_cache_hit_rate": 0.71,
            },
            "windows": {"warmup_s": 120, "measurement_s": 600},
            "per_request_latencies": [
                {
                    "req_id": "req-001",
                    "ttft_ms": 1500.0,
                    "tpot_ms": 12.0,
                    "turn_latency_ms": 4200.0,
                    "thinking_tokens": 0,
                    "response_tokens": 1200,
                    "concurrency_when_dispatched": 4,
                }
            ],
            "ttft_p95_ms": {"driver": 1500.0, "promql": 1500.0, "delta_pct": 0.0},
            "tpot_p95_ms": {"driver": 12.0, "promql": 12.0, "delta_pct": 0.0},
            "turn_latency_p95_ms": {"driver": 4200.0, "promql": 4200.0, "delta_pct": 0.0},
            "sustained_concurrency": 9,
            "rollout_throughput": 12.5,
            "reasoning_content_purity": 1.0,
            "determinism_pass_rate": 1.0,
            "no_oom_events": True,
            "feasible": True,
            "feasibility_failures": [],
            "vllm_metrics_snapshot_ref": "",
            "seed_trace_replay_ref": "",
        }

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", fake_measure)

    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
    )
    round_id = bootstrap["round_id"]
    round_dir = Path(bootstrap["round_dir"])

    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    (candidate_dir / "candidate.yaml").write_text(
        """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
""",
        encoding="utf-8",
    )

    measure = manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
    assert measure["feasible"] is True

    commit = manager.commit_candidate(
        round_id=round_id,
        iteration="001",
        status="keep",
        notes="beats baseline in unit test",
    )
    assert commit["status"] == "keep"

    rescreen = manager.rescreen(round_id=round_id, top_k=1)
    assert rescreen["rescreened"][0]["parent_candidate_uuid"] == measure["candidate_uuid"]

    holdout = manager.validate_holdout(round_id=round_id, candidate_uuid=measure["candidate_uuid"])
    assert holdout["pass"] is True

    finalized = manager.finalize_round(round_id=round_id, dry_run=False)
    assert Path(finalized["bundle_path"]).is_file()
    status = manager.status(round_id=round_id)
    assert status["phase"] == "finalized"


def test_commit_candidate_rejects_synthetic_measurement_trace(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        harness_type="synthetic",
    )
    round_dir = Path(bootstrap["round_dir"])
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
""",
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")

    with pytest.raises(RuntimeError, match="production trace"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should be refused",
        )


def test_measure_rejects_duplicate_iteration_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def fake_measure(self, candidate_vllm_config, *, warmup_s, window_s, target_concurrency_sweep):
        del self, candidate_vllm_config, warmup_s, window_s, target_concurrency_sweep
        return {
            "generator": "RealMeasurementHarness v0.1.0",
            "candidate_vllm_config": {},
            "resolved": {
                "attention_backend": "flash-attn-4",
                "deltanet_kernel": "triton-chunked-delta-v2",
                "torch_compile_mode": "default",
            },
            "cache_isolation": {
                "cache_salt": "",
                "prefix_cache_reset_at_bootstrap": True,
                "first_10_req_prefix_cache_hit_rate": 0.02,
                "last_10_req_prefix_cache_hit_rate": 0.71,
            },
            "windows": {"warmup_s": 120, "measurement_s": 600},
            "per_request_latencies": [],
            "ttft_p95_ms": {"driver": 1500.0, "promql": 1500.0, "delta_pct": 0.0},
            "tpot_p95_ms": {"driver": 12.0, "promql": 12.0, "delta_pct": 0.0},
            "turn_latency_p95_ms": {"driver": 4200.0, "promql": 4200.0, "delta_pct": 0.0},
            "sustained_concurrency": 9,
            "rollout_throughput": 12.5,
            "reasoning_content_purity": 1.0,
            "determinism_pass_rate": 1.0,
            "no_oom_events": True,
            "feasible": True,
            "feasibility_failures": [],
            "vllm_metrics_snapshot_ref": "",
            "seed_trace_replay_ref": "",
        }

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", fake_measure)
    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
    )
    round_dir = Path(bootstrap["round_dir"])
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
""",
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")

    with pytest.raises(RuntimeError, match="results row already exists"):
        manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")


def test_commit_candidate_rejects_malformed_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def fake_measure(self, candidate_vllm_config, *, warmup_s, window_s, target_concurrency_sweep):
        del self, candidate_vllm_config, warmup_s, window_s, target_concurrency_sweep
        return {
            "generator": "RealMeasurementHarness v0.1.0",
            "candidate_vllm_config": {},
            "resolved": {
                "attention_backend": "flash-attn-4",
                "deltanet_kernel": "triton-chunked-delta-v2",
                "torch_compile_mode": "default",
            },
            "cache_isolation": {
                "cache_salt": "",
                "prefix_cache_reset_at_bootstrap": True,
                "first_10_req_prefix_cache_hit_rate": 0.02,
                "last_10_req_prefix_cache_hit_rate": 0.71,
            },
            "windows": {"warmup_s": 120, "measurement_s": 600},
            "per_request_latencies": [],
            "ttft_p95_ms": {"driver": 1500.0, "promql": 1500.0, "delta_pct": 0.0},
            "tpot_p95_ms": {"driver": 12.0, "promql": 12.0, "delta_pct": 0.0},
            "turn_latency_p95_ms": {"driver": 4200.0, "promql": 4200.0, "delta_pct": 0.0},
            "sustained_concurrency": 9,
            "rollout_throughput": 12.5,
            "reasoning_content_purity": 1.0,
            "determinism_pass_rate": 1.0,
            "no_oom_events": True,
            "feasible": True,
            "feasibility_failures": [],
            "vllm_metrics_snapshot_ref": "",
            "seed_trace_replay_ref": "",
        }

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", fake_measure)
    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
    )
    round_dir = Path(bootstrap["round_dir"])
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
""",
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    trace_path = candidate_dir / "measurement_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["cache_isolation"]["first_10_req_prefix_cache_hit_rate"] = 0.50
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed_trace"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should be refused",
        )


def test_bootstrap_round_rejects_dry_run_bundle(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    registry = auto_research.load_registry(repo / "model_registry.yaml")
    weight_version_id = auto_research.default_weight_version_id(registry["qwen3.5-27b"])
    bundle_dir = repo / "output" / "tuned_configs" / "proposal-ranking-manager-judgment" / weight_version_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "dry-run.yaml").write_text(
        """
tuned_config_bundle:
  bundle_id: dry-run-bundle
  produced_at: 2026-04-23T00:00:00+00:00
  weight_version_id: 2e1b21350ce589fcaafbb3c7d7eac526a7aed582
  model_id: qwen3.5-27b
  family_id: proposal-ranking-manager-judgment
  workload_distribution_id: prmj-v1-live
  vllm_config:
    max_num_seqs: 4
    max_num_batched_tokens: 8192
    enable_chunked_prefill: true
    enable_prefix_caching: true
    gpu_memory_utilization: 0.9
    max_model_len: 131072
    kv_cache_dtype: fp8_e5m2
  objective: {}
  measurement_trace_ref: trace.json
  search_trace_ref: search.json
  regression_guard: {}
  safety_rails: {}
  round_provenance:
    dry_run: true
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "add dry-run bundle"], cwd=repo, check=True, capture_output=True, text=True)

    with pytest.raises(RuntimeError, match="dry_run_bundle_exists"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="proposal-ranking-manager-judgment",
            sprint="sprint-0",
            workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
        )


def test_offline_auto_research_runner_backward_compatibility(tmp_path: Path) -> None:
    registry_path = tmp_path / "model_registry.yaml"
    workload_path = (
        tmp_path / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"
    )
    _write_registry(registry_path)
    _write_workload(workload_path)
    workload = auto_research.SyntheticWorkloadDistribution.from_file(
        workload_path,
        model_config=auto_research.load_registry(registry_path)["qwen3.5-27b"],
        family_id="proposal-ranking-manager-judgment",
    )
    runner = auto_research.OfflineAutoResearchRunner(
        model_config=auto_research.load_registry(registry_path)["qwen3.5-27b"],
        family_id="proposal-ranking-manager-judgment",
        output_root=tmp_path / "tuned_configs",
        workload=workload,
        iteration_cap=2,
    )

    result = runner.run()

    assert result.status in {"retained_baseline", "produced_bundle"}
    assert result.run_log_path.is_file()

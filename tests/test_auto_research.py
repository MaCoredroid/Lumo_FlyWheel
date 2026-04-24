from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from lumo_flywheel_serving import auto_research, measurement_harness, round_driver
from lumo_flywheel_serving.round_driver import RoundContext, RoundResult, run_round, run_round_exit_code


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
target_concurrency: 4
gpu_memory_utilization_cap: 0.08
seed_trace_ref: seed_trace.jsonl
holdout_trace_ref: holdout_trace.jsonl
""",
        encoding="utf-8",
    )


def _write_thinking_probe(repo: Path, *, outcome: str = "row-3", captured_at: datetime | None = None) -> Path:
    capture_date = captured_at or datetime.now(UTC)
    report = repo / "reports" / f"thinking-probe-{capture_date.strftime('%Y%m%d')}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Serving Thinking Probe",
                "",
                f"- capture_date: {capture_date.isoformat().replace('+00:00', 'Z')}",
                f"- outcome: {outcome}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def _write_l1_bundle(repo: Path, *, request_shaping: dict | None = None) -> Path:
    bundle_dir = (
        repo
        / "output"
        / "tuned_configs"
        / "proposal-ranking-manager-judgment"
        / "2e1b21350ce589fcaafbb3c7d7eac526a7aed582"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = bundle_dir / "l1-bundle.yaml"
    bundle_path.write_text(
        auto_research.yaml.safe_dump(
            {
                "tuned_config_bundle": {
                    "bundle_id": "l1-bundle-for-l2",
                    "produced_at": "2026-04-24T10:18:15+00:00",
                    "weight_version_id": "2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
                    "model_id": "qwen3.5-27b",
                    "family_id": "proposal-ranking-manager-judgment",
                    "workload_distribution_id": "prmj-v1-live",
                    "vllm_config": {
                        "max_num_seqs": 4,
                        "max_num_batched_tokens": 12288,
                        "enable_chunked_prefill": True,
                        "enable_prefix_caching": False,
                        "gpu_memory_utilization": 0.92,
                        "max_model_len": 131072,
                        "kv_cache_dtype": "fp8_e5m2",
                    },
                    "request_shaping": request_shaping or {},
                    "kernel_selection": {"attention_backend": "flash-attn-4"},
                    "lora_policy": {"adapter_mode": "runtime-apply"},
                    "objective": {"metric": "eval_throughput", "value": 1.0},
                    "measurement_trace_ref": "trace.json",
                    "search_trace_ref": "search.json",
                    "baseline_bundle_id": None,
                    "regression_guard": {},
                    "safety_rails": {},
                    "round_provenance": {"dry_run": False, "active_layer": "L1"},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return bundle_path


def test_capture_seed_workload_updates_seed_and_holdout_refs(tmp_path: Path) -> None:
    workload_path = tmp_path / "serving_workload.yaml"
    _write_workload(workload_path)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "capture_seed_workload.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--workload-file",
            str(workload_path),
            "--count",
            "10",
            "--split-seed",
            "17",
            "--update-workload",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    workload = auto_research.load_yaml_file(workload_path)
    assert isinstance(workload, dict)
    seed_path = workload_path.with_name("seed_trace.jsonl")
    holdout_path = workload_path.with_name("holdout_trace.jsonl")

    assert seed_path.is_file()
    assert holdout_path.is_file()
    assert payload["seed_count"] == 9
    assert payload["holdout_count"] == 1
    assert len(payload["workload_distribution_id"]) == 64
    assert workload["seed_trace_ref"] == "seed_trace.jsonl"
    assert workload["holdout_trace_ref"] == "holdout_trace.jsonl"
    assert workload["workload_distribution_id"] == payload["seed_sha256"] == payload["workload_distribution_id"]


def test_capture_seed_workload_overwrites_stale_distribution_id(tmp_path: Path) -> None:
    workload_path = tmp_path / "serving_workload.yaml"
    _write_workload(workload_path)
    workload = auto_research.load_yaml_file(workload_path)
    assert isinstance(workload, dict)
    workload["workload_distribution_id"] = "stale-id"
    workload_path.write_text(auto_research.yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "capture_seed_workload.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--workload-file",
            str(workload_path),
            "--count",
            "8",
            "--update-workload",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    updated = auto_research.load_yaml_file(workload_path)
    assert isinstance(updated, dict)
    assert updated["workload_distribution_id"] != "stale-id"
    assert updated["workload_distribution_id"] == payload["workload_distribution_id"]


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("output/\n", encoding="utf-8")
    _write_registry(repo / "model_registry.yaml")
    _write_workload(
        repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"
    )
    _write_thinking_probe(repo)
    fixture_src = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic_measurement.py"
    fixture_dst = repo / "tests" / "fixtures" / "synthetic_measurement.py"
    fixture_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fixture_src, fixture_dst)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True, text=True)
    return repo


class _HTTPResponse:
    def __init__(self, *, payload: dict | None = None, text: str = "", status_code: int = 200) -> None:
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise auto_research.requests.HTTPError(f"{self.status_code} error")

    def json(self) -> dict:
        return self._payload


def _real_trace(
    *,
    objective: int = 9,
    ttft: float = 1500.0,
    tpot: float = 12.0,
    turn: float = 4200.0,
    generator: str = "RealMeasurementHarness v0.1.0",
) -> dict[str, object]:
    return {
        "generator": generator,
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
        "per_request_latencies": [],
        "ttft_p95_ms": {"driver": ttft, "promql": ttft, "delta_pct": 0.0},
        "tpot_p95_ms": {"driver": tpot, "promql": tpot, "delta_pct": 0.0},
        "turn_latency_p95_ms": {"driver": turn, "promql": turn, "delta_pct": 0.0},
        "sustained_concurrency": objective,
        "rollout_throughput": 12.5,
        "reasoning_content_purity": 1.0,
        "determinism_pass_rate": 1.0,
        "no_oom_events": True,
        "feasible": True,
        "feasibility_failures": [],
        "vllm_metrics_snapshot_ref": "",
        "seed_trace_replay_ref": "",
    }


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
    assert subprocess.run(["git", "status", "--short"], cwd=repo, check=True, capture_output=True, text=True).stdout == ""
    assert (
        subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == "main"
    )
    assert (
        subprocess.run(
            ["git", "rev-list", "--count", f"main..{bootstrap['round_branch']}"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        != "0"
    )
    status = manager.status(round_id=round_id)
    assert status["phase"] == "finalized"


def test_bootstrap_round_creates_dedicated_round_branch(tmp_path: Path) -> None:
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

    current_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert current_branch == "main"
    assert bootstrap["round_branch"].startswith(
        "autoresearch/qwen3.5-27b/proposal-ranking-manager-judgment/sprint-0/"
    )
    round_branch_head = subprocess.run(
        ["git", "rev-parse", bootstrap["round_branch"]],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    main_head = subprocess.run(
        ["git", "rev-parse", "main"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert round_branch_head == main_head


def test_bootstrap_round_writes_spec_brief_templates(tmp_path: Path) -> None:
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
    impl_brief = (round_dir / "impl_brief.md").read_text(encoding="utf-8")
    iteration_brief = (round_dir / "iteration_brief.md").read_text(encoding="utf-8")

    assert "## Context docs (read all three first)" in impl_brief
    assert "validate-holdout" in impl_brief
    assert "A dry-run round against SyntheticMeasurementFixture completes" in impl_brief
    assert "## Hard rules (sub-spec §6 — verified by watchdog + CLI)" in iteration_brief
    assert "{{per_candidate_wall_clock_minutes}}" in iteration_brief
    assert "{{next_iteration}}" in iteration_brief
    assert "{{workload_file}}" in iteration_brief
    assert "--harness {{harness_mode}}" in iteration_brief
    assert 'generator starting with "{{harness_generator_prefix}}"' in iteration_brief
    assert "synthetic fixture commits also carry `Fixture-Mode: true`" in iteration_brief
    assert "R8. If a CLI call returns non-zero" in iteration_brief

    ctx = RoundContext.from_bootstrap_json(
        bootstrap,
        harness_mode="synthetic",
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    prompt = round_driver._iteration_prompt(ctx, iteration="001", next_iteration="002")
    assert "--harness synthetic" in prompt
    assert "generator starting with \"SyntheticMeasurementFixture\"" in prompt
    assert "{{harness_mode}}" not in prompt
    assert "{{harness_generator_prefix}}" not in prompt


def test_bootstrap_round_records_serving_thinking_probe_for_real_round(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    probe_path = next((repo / "reports").glob("thinking-probe-*.md"))
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
        serving_thinking_probe=probe_path,
    )

    spec = auto_research.load_yaml_file(Path(bootstrap["round_spec_path"]))
    assert isinstance(spec, dict)
    assert spec["serving_thinking_probe"]["path"] == f"reports/{probe_path.name}"
    assert spec["serving_thinking_probe"]["outcome"] == "row-3"
    assert spec["serving_thinking_probe"]["capture_date"].endswith("Z")


def test_bootstrap_round_rejects_blocking_serving_thinking_probe(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    probe_path = _write_thinking_probe(repo, outcome="row-2")
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="serving_thinking_probe_blocking_outcome:row-2"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="proposal-ranking-manager-judgment",
            sprint="sprint-0",
            workload_file=repo
            / "benchmark_blueprints"
            / "families"
            / "proposal-ranking-manager-judgment"
            / "serving_workload.yaml",
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
            serving_thinking_probe=probe_path,
        )

    assert list((repo / "output" / "auto_research").glob("*")) == []


def test_l2_bootstrap_requires_and_records_lower_layer_bundle(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    bundle_path = _write_l1_bundle(repo)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="L2 bootstrap requires --baseline-bundle"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="proposal-ranking-manager-judgment",
            sprint="sprint-0",
            workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
            harness_type="synthetic",
            active_layer="L2",
        )

    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        harness_type="synthetic",
        active_layer="L2",
        baseline_bundle=bundle_path,
    )
    round_dir = Path(bootstrap["round_dir"])
    spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    baseline_candidate = auto_research.load_yaml_file(round_dir / "candidates" / "baseline_a" / "candidate.yaml")

    assert spec["active_layer"] == "L2"
    assert spec["baseline_bundle_id"] == "l1-bundle-for-l2"
    assert spec["baseline_bundle_path"] == str(bundle_path.resolve())
    assert spec["frozen_vllm_config"]["max_num_batched_tokens"] == 12288
    assert baseline_candidate == {
        "concurrency_cap_eval": 4,
        "concurrency_cap_rollout": 0,
        "admission_queue_depth_max": 128,
        "per_request_kv_budget": 131072,
        "priority_preemption": "off",
    }


def test_l2_candidate_validation_rejects_l1_and_l3_keys(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    bundle_path = _write_l1_bundle(repo)
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
        active_layer="L2",
        baseline_bundle=bundle_path,
    )
    round_dir = Path(bootstrap["round_dir"])
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
concurrency_cap_eval: 3
concurrency_cap_rollout: 1
admission_queue_depth_max: 64
per_request_kv_budget: 65536
priority_preemption: strict
max_num_seqs: 4
adapter_mode: runtime-apply
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported keys for L2"):
        manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")


def test_l2_measurement_composes_frozen_vllm_config_with_request_shaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    bundle_path = _write_l1_bundle(repo)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    seen: dict[str, object] = {}

    def fake_measure(self, candidate_vllm_config, **kwargs):
        del self
        seen["candidate_vllm_config"] = dict(candidate_vllm_config)
        seen["kwargs"] = dict(kwargs)
        return _real_trace(objective=3)

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", fake_measure)
    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        active_layer="L2",
        baseline_bundle=bundle_path,
    )
    round_dir = Path(bootstrap["round_dir"])
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
concurrency_cap_eval: 3
concurrency_cap_rollout: 1
admission_queue_depth_max: 64
per_request_kv_budget: 65536
priority_preemption: strict
""",
        encoding="utf-8",
    )

    measured = manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    trace = json.loads((candidate_dir / "measurement_trace.json").read_text(encoding="utf-8"))

    assert measured["feasible"] is True
    assert seen["candidate_vllm_config"]["max_num_batched_tokens"] == 12288
    assert seen["kwargs"]["target_concurrency"] == 3
    assert trace["active_layer"] == "L2"
    assert trace["candidate_request_shaping"]["priority_preemption"] == "strict"
    assert trace["frozen_lower_layer"]["source_bundle_id"] == "l1-bundle-for-l2"
    assert trace["request_shaping_enforcement"]["real_proxy_enforcement"] is False


def test_l2_finalize_emits_bundle_with_frozen_vllm_and_request_shaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    bundle_path = _write_l1_bundle(repo)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace(objective=3))
    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        active_layer="L2",
        baseline_bundle=bundle_path,
    )
    round_id = bootstrap["round_id"]
    round_dir = Path(bootstrap["round_dir"])
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
concurrency_cap_eval: 3
concurrency_cap_rollout: 1
admission_queue_depth_max: 64
per_request_kv_budget: 65536
priority_preemption: strict
""",
        encoding="utf-8",
    )
    measured = manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=round_id, iteration="001", status="keep", notes="l2 winner")
    manager.rescreen(round_id=round_id, top_k=1)
    holdout = manager.validate_holdout(round_id=round_id, candidate_uuid=measured["candidate_uuid"])
    assert holdout["pass"] is True

    finalized = manager.finalize_round(round_id=round_id, dry_run=False)
    bundle = auto_research.load_yaml_file(finalized["bundle_path"])["tuned_config_bundle"]

    assert bundle["vllm_config"]["max_num_batched_tokens"] == 12288
    assert bundle["request_shaping"] == {
        "concurrency_cap_eval": 3,
        "concurrency_cap_rollout": 1,
        "admission_queue_depth_max": 64,
        "per_request_kv_budget": 65536,
        "priority_preemption": "strict",
    }
    assert bundle["kernel_selection"] == {"attention_backend": "flash-attn-4"}
    assert bundle["lora_policy"] == {"adapter_mode": "runtime-apply"}
    assert bundle["baseline_bundle_id"] == "l1-bundle-for-l2"
    assert bundle["round_provenance"]["active_layer"] == "L2"
    assert bundle["round_provenance"]["request_shaping_enforcement"]["real_proxy_enforcement"] is False


def test_commit_candidate_rejects_synthetic_measurement_trace_in_real_mode(tmp_path: Path) -> None:
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
            harness="real",
        )


def test_commit_candidate_rejects_real_measurement_trace_in_synthetic_mode(tmp_path: Path) -> None:
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
    trace_path = candidate_dir / "measurement_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["generator"] = "RealMeasurementHarness v0.1.0"
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="synthetic fixture trace"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should be refused",
        )


def test_commit_candidate_refuses_when_unexpected_paths_are_staged(tmp_path: Path) -> None:
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
    (repo / "README.md").write_text("staged outside round scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True, text=True)

    with pytest.raises(RuntimeError, match=r"commit_refused: staged paths outside allow-list: README.md"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should refuse staged spillover",
            allow_synthetic=True,
        )


def test_commit_candidate_tracks_bootstrap_artifacts_and_leaves_worktree_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace())
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
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=bootstrap["round_id"], iteration="001", status="keep", notes="tracks bootstrap")

    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", bootstrap["round_branch"], str(round_dir.relative_to(repo))],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "round_spec.yaml" in tracked
    assert "impl_brief.md" in tracked
    assert "iteration_brief.md" in tracked
    assert "candidates/baseline_a/candidate.yaml" in tracked
    assert "candidates/baseline_b/candidate.yaml" in tracked
    assert "codex-home/.codex/config.toml" not in tracked
    assert subprocess.run(["git", "status", "--short"], cwd=repo, check=True, capture_output=True, text=True).stdout == ""


def test_commit_candidate_refuses_duplicate_commit_for_same_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace())
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
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=bootstrap["round_id"], iteration="001", status="keep", notes="initial commit")

    with pytest.raises(RuntimeError, match="results row already finalized"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="discard",
            notes="should not be allowed",
        )


def test_commit_candidate_refuses_when_git_index_has_stale_allowed_path(tmp_path: Path) -> None:
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
    results_path = round_dir / "results.tsv"
    results_path.write_text(results_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", str(results_path.relative_to(repo))],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match=r"commit_refused: git index not clean: .*results.tsv"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should refuse stale index",
            allow_synthetic=True,
        )


def test_commit_candidate_refuses_when_bootstrap_artifact_is_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace())
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
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    results_before = (round_dir / "results.tsv").read_text(encoding="utf-8")
    round_spec_before = (round_dir / "round_spec.yaml").read_text(encoding="utf-8")
    (round_dir / "impl_brief.md").write_text("corrupted brief\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"commit_refused: immutable round artifact changed: impl_brief.md"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should refuse dirty bootstrap artifact",
        )
    assert (round_dir / "results.tsv").read_text(encoding="utf-8") == results_before
    assert (round_dir / "round_spec.yaml").read_text(encoding="utf-8") == round_spec_before


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


def test_measure_rejects_iteration_past_round_caps(tmp_path: Path) -> None:
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

    too_far_main = round_dir / "candidates" / "013"
    too_far_main.mkdir()
    too_far_main.joinpath("candidate.yaml").write_text(
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

    with pytest.raises(RuntimeError, match="exceeds iteration_cap 12"):
        manager.measure(round_id=bootstrap["round_id"], candidate_path=too_far_main / "candidate.yaml")

    too_far_rescreen = round_dir / "candidates" / "rescreen_04"
    too_far_rescreen.mkdir()
    too_far_rescreen.joinpath("candidate.yaml").write_text(
        too_far_main.joinpath("candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exceeds rescreen_top_k 3"):
        manager.measure(round_id=bootstrap["round_id"], candidate_path=too_far_rescreen / "candidate.yaml")


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


def test_commit_candidate_rejects_missing_required_cache_isolation_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace())
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
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    trace_path = candidate_dir / "measurement_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["cache_isolation"]["prefix_cache_reset_at_bootstrap"] = False
    trace["cache_isolation"].pop("last_10_req_prefix_cache_hit_rate")
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="malformed_trace"):
        manager.commit_candidate(
            round_id=bootstrap["round_id"],
            iteration="001",
            status="keep",
            notes="should be refused",
        )


def test_measure_and_commit_candidate_surface_promql_mismatch_as_harness_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def fake_measure(self, candidate_vllm_config, *, warmup_s, window_s, target_concurrency_sweep):
        del self, candidate_vllm_config, warmup_s, window_s, target_concurrency_sweep
        trace = _real_trace()
        trace["ttft_p95_ms"]["delta_pct"] = 12.5
        return trace

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
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    measure = manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    rows = manager._read_results(round_dir / "results.tsv")

    assert measure["recommended_status"] is None
    assert measure["notes"] == "promql_mismatch"
    assert rows[0].feasible is True

    committed = manager.commit_candidate(
        round_id=bootstrap["round_id"],
        iteration="001",
        status="keep",
        notes="latency promql mismatch recorded as warning",
    )
    updated_rows = manager._read_results(round_dir / "results.tsv")

    assert committed["status"] == "keep"
    assert updated_rows[0].status == "keep"
    assert updated_rows[0].notes == "latency promql mismatch recorded as warning"


def test_baseline_commits_persist_noise_floor_into_round_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    measurements = iter(
        [
            _real_trace(objective=9, ttft=1400.0, tpot=12.0, turn=4000.0),
            _real_trace(objective=11, ttft=1500.0, tpot=12.5, turn=4200.0),
        ]
    )
    monkeypatch.setattr(
        auto_research.RealMeasurementHarness,
        "measure",
        lambda self, candidate_vllm_config, **kwargs: next(measurements),
    )
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

    manager.measure(round_id=round_id, candidate_path=round_dir / "candidates" / "baseline_a" / "candidate.yaml")
    manager.commit_candidate(
        round_id=round_id,
        iteration="baseline_a",
        status="baseline",
        notes="default baseline replay a",
    )
    round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["noise_floor"] == 0.0

    manager.measure(round_id=round_id, candidate_path=round_dir / "candidates" / "baseline_b" / "candidate.yaml")
    manager.commit_candidate(
        round_id=round_id,
        iteration="baseline_b",
        status="baseline",
        notes="default baseline replay b",
    )

    updated_round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert isinstance(updated_round_spec, dict)
    assert updated_round_spec["noise_floor"] == pytest.approx(4.0)
    assert manager.status(round_id=round_id)["noise_floor"] == pytest.approx(4.0)


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
    subprocess.run(["git", "add", "-f", "."], cwd=repo, check=True, capture_output=True, text=True)
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


def test_bootstrap_round_rejects_incompatible_codex_cli_version_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    real_subprocess_run = auto_research.subprocess.run

    def fake_run(*args, **kwargs):
        cmd = args[0]
        if cmd == ["codex", "--version"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="codex-cli 0.119.9\n", stderr="")
        return real_subprocess_run(*args, **kwargs)

    monkeypatch.setattr(auto_research.shutil, "which", lambda name: "/usr/bin/codex" if name == "codex" else None)
    monkeypatch.setattr(auto_research.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"need >= 0\.120\.0, found 0\.119\.9"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="proposal-ranking-manager-judgment",
            sprint="sprint-0",
            workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
        )

    assert list((repo / "output" / "auto_research").glob("*")) == []
    branches = subprocess.run(
        ["git", "branch", "--format=%(refname:short)"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert branches == ["main"]


def test_real_measurement_harness_loads_candidate_and_flushes_prefix_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_trace = tmp_path / "seed_trace.jsonl"
    _write_trace(seed_trace, prompt_tokens=64, output_tokens=32)
    workload_spec = measurement_harness.WorkloadSpec(
        family_id="proposal-ranking-manager-judgment",
        workload_distribution_id="prmj-v1-live",
        seed_trace_ref=seed_trace,
        holdout_trace_ref=None,
        latency_ceiling_ms=35000,
        tpot_ceiling_ms=80,
        turn_latency_ceiling_ms=35000,
        avg_prompt_tokens=64,
        avg_output_tokens=32,
        measurement_window_minutes=1,
        rollout_baseline=0.01,
    )
    harness = measurement_harness.RealMeasurementHarness(
        workload_spec=workload_spec,
        seed_trace_path=seed_trace,
        slo=measurement_harness.SLO(ttft_ms=35000, tpot_ms=80, turn_ms=35000),
        endpoint="http://127.0.0.1:8001/v1",
        metrics_scrape_url="http://127.0.0.1:8000/metrics",
        admin_url="http://127.0.0.1:8001/admin",
        model_id="qwen3.5-27b",
        weight_version_id="rev-123",
        bundle_staging_dir=tmp_path / "measure-staging",
        round_id="round-123",
    )

    events: list[tuple[str, str]] = []
    loaded_bundle_path: Path | None = None

    def fake_post(url: str, **kwargs):
        nonlocal loaded_bundle_path
        events.append(("POST", url))
        payload = kwargs.get("json")
        if url.endswith("/admin/load_tuned_config"):
            loaded_bundle_path = Path(str(payload["bundle_path"]))
            assert loaded_bundle_path.is_file()
        return _HTTPResponse()

    def fake_get(url: str, **kwargs):
        del kwargs
        events.append(("GET", url))
        if url.endswith("/metrics"):
            return _HTTPResponse(text="vllm:prefix_cache_queries_total 0\nvllm:prefix_cache_hits_total 0\n")
        return _HTTPResponse()

    monkeypatch.setattr(measurement_harness.requests, "post", fake_post)
    monkeypatch.setattr(measurement_harness.requests, "get", fake_get)

    trace = harness.measure(
        {
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 131072,
            "kv_cache_dtype": "fp8_e5m2",
        },
        warmup_s=1,
        window_s=2,
        target_concurrency_sweep=[1, 2],
    )

    assert loaded_bundle_path is not None
    assert not loaded_bundle_path.exists()
    assert events[0] == ("POST", "http://127.0.0.1:8001/admin/load_tuned_config")
    assert events[1] == ("POST", "http://127.0.0.1:8000/reset_prefix_cache")
    assert ("GET", "http://127.0.0.1:8000/health") in events
    assert trace["sustained_concurrency"] == 2
    assert {entry["concurrency_when_dispatched"] for entry in trace["per_request_latencies"]} == {2}


def test_real_measurement_harness_throughput_uses_elapsed_replay_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_trace = tmp_path / "seed_trace.jsonl"
    _write_trace(seed_trace, prompt_tokens=64, output_tokens=32)
    workload_spec = measurement_harness.WorkloadSpec(
        family_id="proposal-ranking-manager-judgment",
        workload_distribution_id="prmj-v1-live",
        seed_trace_ref=seed_trace,
        holdout_trace_ref=None,
        latency_ceiling_ms=35000,
        tpot_ceiling_ms=80,
        turn_latency_ceiling_ms=35000,
        avg_prompt_tokens=64,
        avg_output_tokens=32,
        measurement_window_minutes=1,
        rollout_baseline=0.01,
    )
    harness = measurement_harness.RealMeasurementHarness(
        workload_spec=workload_spec,
        seed_trace_path=seed_trace,
        slo=measurement_harness.SLO(ttft_ms=35000, tpot_ms=80, turn_ms=35000),
        endpoint="http://127.0.0.1:8001/v1",
        metrics_scrape_url="http://127.0.0.1:8000/metrics",
        admin_url="http://127.0.0.1:8001/admin",
        model_id="qwen3.5-27b",
        weight_version_id="rev-123",
        bundle_staging_dir=tmp_path / "measure-staging",
        round_id="round-123",
    )
    replay = [
        {
            "req_id": f"req-{index:04d}",
            "ttft_ms": 10.0,
            "tpot_ms": 1.0,
            "turn_latency_ms": 100.0,
            "thinking_tokens": 0,
            "response_tokens": 100,
            "concurrency_when_dispatched": 1,
        }
        for index in range(1, 5)
    ]
    clock = iter([100.0, 104.0, 200.0, 204.0])

    monkeypatch.setattr(harness, "_activate_candidate", lambda candidate_vllm_config: None)
    monkeypatch.setattr(harness, "_metrics_snapshot", lambda: {})
    monkeypatch.setattr(
        harness,
        "_replay_requests",
        lambda replay_entries, candidate_vllm_config, *, target_concurrency: replay,
    )
    monkeypatch.setattr(measurement_harness.time, "time", lambda: next(clock))

    screen = harness.measure({}, warmup_s=120, window_s=600, target_concurrency=4)
    full = harness.measure({}, warmup_s=300, window_s=1500, target_concurrency=4)

    assert screen["windows"]["measurement_s"] == 600
    assert full["windows"]["measurement_s"] == 1500
    assert screen["windows"]["measurement_elapsed_s"] == 4.0
    assert full["windows"]["measurement_elapsed_s"] == 4.0
    # eval_throughput is completed requests/s; rollout_throughput is response tokens/s.
    assert screen["eval_throughput"] == 1.0
    assert full["eval_throughput"] == 1.0
    assert screen["rollout_throughput"] == 100.0
    assert full["rollout_throughput"] == 100.0
    assert screen["diagnostics"]["rollout_throughput"] == 100.0
    assert full["diagnostics"]["rollout_throughput"] == 100.0


def test_measure_uses_round_target_concurrency_not_candidate_max_num_seqs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    seen_target_concurrency: list[int] = []

    def fake_measure(self, candidate_vllm_config, **kwargs):
        del self, candidate_vllm_config
        seen_target_concurrency.append(kwargs["target_concurrency"])
        return _real_trace()

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
    round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["target_concurrency"] == 4
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        """
max_num_seqs: 32
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.92
max_model_len: 32768
kv_cache_dtype: fp8_e5m2
""",
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")

    assert seen_target_concurrency == [4]


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


def test_run_non_agent_threads_harness_type_to_bootstrap_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    monkeypatch.setenv("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT", "1")

    captured: dict[str, object] = {}

    def fake_bootstrap_round(self, **kwargs):
        captured["bootstrap"] = kwargs
        return {
            "round_id": "round-123",
            "round_dir": str(repo / "output" / "auto_research" / "round-123"),
            "round_branch": "autoresearch/test",
            "round_spec_path": str(repo / "output" / "auto_research" / "round-123" / "round_spec.yaml"),
        }

    def fake_finalize_round(self, *, round_id: str, dry_run: bool = False):
        captured["finalize"] = {"round_id": round_id, "dry_run": dry_run}
        return {
            "round_id": round_id,
            "bundle_path": str(repo / "output" / "tuned_configs" / "bundle.yaml"),
            "finalize_commit_sha": "synthetic-sha",
        }

    monkeypatch.setattr(auto_research.AutoResearchRoundManager, "bootstrap_round", fake_bootstrap_round)
    monkeypatch.setattr(auto_research.AutoResearchRoundManager, "finalize_round", fake_finalize_round)
    monkeypatch.setattr(auto_research.AutoResearchRoundManager, "measure", lambda self, **kwargs: {"candidate_uuid": "uuid"})
    monkeypatch.setattr(
        auto_research.AutoResearchRoundManager,
        "commit_candidate",
        lambda self, **kwargs: {"iteration": kwargs["iteration"], "candidate_uuid": "uuid", "status": kwargs["status"]},
    )
    monkeypatch.setattr(auto_research.OfflineAutoResearchRunner, "_candidate_plan", lambda self: [])

    round_dir = repo / "output" / "auto_research" / "round-123"
    (round_dir / "candidates" / "baseline_a").mkdir(parents=True, exist_ok=True)
    (round_dir / "candidates" / "baseline_b").mkdir(parents=True, exist_ok=True)

    result = manager.run_non_agent(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        baseline_bundle=None,
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        iteration_cap=1,
        harness_type="synthetic",
    )

    assert captured["bootstrap"] == {
        "model_id": "qwen3.5-27b",
        "family_id": "proposal-ranking-manager-judgment",
        "sprint": "sprint-0",
        "workload_file": repo
        / "benchmark_blueprints"
        / "families"
        / "proposal-ranking-manager-judgment"
        / "serving_workload.yaml",
        "weight_version_id": None,
        "round_root": repo / "output" / "auto_research",
        "harness_type": "synthetic",
        "skip_preflight": True,
    }
    assert captured["finalize"] == {"round_id": "round-123", "dry_run": True}
    assert result["bundle_path"] == str(repo / "output" / "tuned_configs" / "bundle.yaml")


def test_run_non_agent_real_harness_skips_production_bootstrap_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    monkeypatch.setenv("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT", "1")
    monkeypatch.setattr(
        auto_research.AutoResearchRoundManager,
        "_run_bootstrap_preflight",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("production preflight should be skipped")),
    )
    monkeypatch.setattr(auto_research.AutoResearchRoundManager, "measure", lambda self, **kwargs: {"candidate_uuid": "uuid"})
    monkeypatch.setattr(
        auto_research.AutoResearchRoundManager,
        "commit_candidate",
        lambda self, **kwargs: {"iteration": kwargs["iteration"], "candidate_uuid": "uuid", "status": kwargs["status"]},
    )
    monkeypatch.setattr(
        auto_research.AutoResearchRoundManager,
        "finalize_round",
        lambda self, *, round_id, dry_run=False: {
            "round_id": round_id,
            "bundle_path": str(repo / "output" / "tuned_configs" / "bundle.yaml"),
            "finalize_commit_sha": "synthetic-sha",
        },
    )
    monkeypatch.setattr(auto_research.OfflineAutoResearchRunner, "_candidate_plan", lambda self: [])

    result = manager.run_non_agent(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        baseline_bundle=None,
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        iteration_cap=1,
        harness_type="real",
    )

    assert result["round_id"].startswith("qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-")


def test_finalize_round_refuses_without_rescreen_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace())
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
    candidate_dir.joinpath("candidate.yaml").write_text(
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    measured = manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=round_id, iteration="001", status="keep", notes="winner")
    rescreen = manager.rescreen(round_id=round_id, top_k=1)
    assert rescreen["rescreened"]
    (round_dir / "holdout_trace.json").write_text(
        json.dumps({"pass": True, "candidate_uuid": measured["candidate_uuid"]}, indent=2),
        encoding="utf-8",
    )
    (round_dir / "rescreen_trace.json").unlink()

    with pytest.raises(RuntimeError, match="rescreen_trace.json"):
        manager.finalize_round(round_id=round_id, dry_run=False)


def test_rescreen_and_finalize_allow_baseline_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    measurements = iter(
        [
            _real_trace(objective=9, ttft=1400.0, tpot=12.0, turn=4000.0),
            _real_trace(objective=9, ttft=1600.0, tpot=12.5, turn=4300.0),
            _real_trace(objective=9, ttft=1450.0, tpot=11.5, turn=4050.0),
            _real_trace(objective=9, ttft=1425.0, tpot=11.0, turn=3950.0),
        ]
    )

    monkeypatch.setattr(
        auto_research.RealMeasurementHarness,
        "measure",
        lambda self, candidate_vllm_config, **kwargs: next(measurements),
    )
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

    baseline_uuids: dict[str, str] = {}
    for suffix in ("a", "b"):
        result = manager.measure(
            round_id=round_id,
            candidate_path=round_dir / "candidates" / f"baseline_{suffix}" / "candidate.yaml",
        )
        baseline_uuids[suffix] = result["candidate_uuid"]
        manager.commit_candidate(
            round_id=round_id,
            iteration=f"baseline_{suffix}",
            status="baseline",
            notes=f"default baseline replay {suffix}",
        )

    rescreen = manager.rescreen(round_id=round_id, top_k=1)
    assert len(rescreen["rescreened"]) == 1
    assert rescreen["rescreened"][0]["parent_candidate_uuid"] == baseline_uuids["a"]

    holdout = manager.validate_holdout(round_id=round_id, candidate_uuid=baseline_uuids["a"])
    assert holdout["pass"] is True

    finalized = manager.finalize_round(round_id=round_id, dry_run=False)

    assert finalized["winner_iteration"] == "baseline_a"
    assert finalized["winner_candidate_uuid"] == baseline_uuids["a"]


def test_validate_holdout_accepts_rescreen_uuid_and_canonicalizes_to_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    measurements = iter([_real_trace(objective=9), _real_trace(objective=9), _real_trace(objective=9)])
    monkeypatch.setattr(
        auto_research.RealMeasurementHarness,
        "measure",
        lambda self, candidate_vllm_config, **kwargs: next(measurements),
    )
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
    candidate_dir.joinpath("candidate.yaml").write_text(
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    measured = manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=round_id, iteration="001", status="keep", notes="winner")
    rescreen = manager.rescreen(round_id=round_id, top_k=1)

    holdout = manager.validate_holdout(
        round_id=round_id,
        candidate_uuid=rescreen["rescreened"][0]["candidate_uuid"],
    )

    assert holdout["pass"] is True
    assert holdout["candidate_uuid"] == measured["candidate_uuid"]

    finalized = manager.finalize_round(round_id=round_id, dry_run=False)
    assert finalized["winner_candidate_uuid"] == measured["candidate_uuid"]


def test_finalize_round_refuses_when_holdout_uuid_does_not_match_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    monkeypatch.setattr(auto_research.RealMeasurementHarness, "measure", lambda self, candidate_vllm_config, **kwargs: _real_trace())
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
    candidate_dir.joinpath("candidate.yaml").write_text(
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=round_id, iteration="001", status="keep", notes="winner")
    manager.rescreen(round_id=round_id, top_k=1)
    (round_dir / "holdout_trace.json").write_text(
        json.dumps({"pass": True, "candidate_uuid": "not-the-winner"}, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="holdout candidate_uuid"):
        manager.finalize_round(round_id=round_id, dry_run=False)


def test_finalize_round_breaks_rescreen_ties_with_latency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    measurements = iter(
        [
            {"objective": 9, "ttft": 1900.0, "tpot": 12.0, "turn": 4200.0},
            {"objective": 10, "ttft": 1800.0, "tpot": 11.0, "turn": 4100.0},
            {"objective": 11, "ttft": 2600.0, "tpot": 14.0, "turn": 4700.0},
            {"objective": 10, "ttft": 1200.0, "tpot": 10.0, "turn": 3900.0},
        ]
    )

    def fake_measure(self, candidate_vllm_config, *, warmup_s, window_s, target_concurrency_sweep):
        del self, candidate_vllm_config, warmup_s, window_s, target_concurrency_sweep
        sample = next(measurements)
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
            "per_request_latencies": [],
            "ttft_p95_ms": {"driver": sample["ttft"], "promql": sample["ttft"], "delta_pct": 0.0},
            "tpot_p95_ms": {"driver": sample["tpot"], "promql": sample["tpot"], "delta_pct": 0.0},
            "turn_latency_p95_ms": {"driver": sample["turn"], "promql": sample["turn"], "delta_pct": 0.0},
            "sustained_concurrency": sample["objective"],
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

    candidate_payload = """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
"""
    for iteration in ("001", "002"):
        candidate_dir = round_dir / "candidates" / iteration
        candidate_dir.mkdir()
        candidate_dir.joinpath("candidate.yaml").write_text(candidate_payload, encoding="utf-8")
        manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
        manager.commit_candidate(
            round_id=round_id,
            iteration=iteration,
            status="keep",
            notes=f"candidate {iteration}",
        )

    round_spec_path = round_dir / "round_spec.yaml"
    round_spec = auto_research.load_yaml_file(round_spec_path)
    assert isinstance(round_spec, dict)
    round_spec["noise_floor"] = 5.0
    round_spec_path.write_text(
        auto_research.yaml.safe_dump(round_spec, sort_keys=False),
        encoding="utf-8",
    )

    rescreen = manager.rescreen(round_id=round_id, top_k=2)
    assert len(rescreen["rescreened"]) == 2
    second_parent_uuid = manager._read_results(round_dir / "results.tsv")[1].candidate_uuid

    (round_dir / "holdout_trace.json").write_text(
        json.dumps({"pass": True, "candidate_uuid": second_parent_uuid}, indent=2),
        encoding="utf-8",
    )

    finalized = manager.finalize_round(round_id=round_id, dry_run=False)

    winner_commit = subprocess.run(
        ["git", "log", bootstrap["round_branch"], "-1", "--format=%(trailers:key=Winner-Candidate-UUID,valueonly)"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert finalized["winner_iteration"] == "002"
    assert finalized["winner_candidate_uuid"] == second_parent_uuid
    assert winner_commit == second_parent_uuid


def test_rescreen_refuses_non_production_trace_on_real_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    measurements = iter(
        [
            _real_trace(objective=9),
            _real_trace(objective=9, generator="SyntheticMeasurementFixture v0.1.0"),
        ]
    )
    monkeypatch.setattr(
        auto_research.RealMeasurementHarness,
        "measure",
        lambda self, candidate_vllm_config, **kwargs: next(measurements),
    )
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
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=bootstrap["round_id"], iteration="001", status="keep", notes="winner")

    with pytest.raises(RuntimeError, match="production trace"):
        manager.rescreen(round_id=bootstrap["round_id"], top_k=1)


def test_measure_rejects_harness_overrides_on_real_harness(tmp_path: Path) -> None:
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
harness_overrides:
  force_oom: true
""",
        encoding="utf-8",
    )
    round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    round_spec["harness_type"] = "real"
    (round_dir / "round_spec.yaml").write_text(
        auto_research.yaml.safe_dump(round_spec, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unsupported keys: \\['harness_overrides'\\]"):
        manager.measure(round_id=bootstrap["round_id"], candidate_path=candidate_dir / "candidate.yaml")


def test_finalize_round_refuses_when_unexpected_paths_are_staged(tmp_path: Path) -> None:
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
    manager.commit_candidate(
        round_id=bootstrap["round_id"],
        iteration="001",
        status="keep",
        notes="dry-run winner",
        allow_synthetic=True,
    )
    (repo / "README.md").write_text("staged outside finalize scope\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True, text=True)

    with pytest.raises(RuntimeError, match=r"finalize-round refuses: staged paths outside allow-list: README.md"):
        manager.finalize_round(round_id=bootstrap["round_id"], dry_run=True)
    assert not (round_dir / "run_log.json").exists()
    assert not (round_dir / "search_trace.json").exists()
    assert not (round_dir / "measurement_trace_combined.json").exists()
    assert (round_dir / ".round.lock").exists()
    assert list((repo / "output" / "tuned_configs").glob("**/*.yaml")) == []


def test_finalize_round_refuses_when_round_artifact_is_dirty(tmp_path: Path) -> None:
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
    manager.commit_candidate(
        round_id=bootstrap["round_id"],
        iteration="001",
        status="keep",
        notes="dry-run winner",
        allow_synthetic=True,
    )
    (round_dir / "iteration_brief.md").write_text("corrupted brief\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"finalize-round refuses: immutable round artifact changed: iteration_brief.md"):
        manager.finalize_round(round_id=bootstrap["round_id"], dry_run=True)


def test_finalize_round_dry_run_refuses_mixed_measurement_generators(tmp_path: Path) -> None:
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
    round_id = bootstrap["round_id"]
    round_dir = Path(bootstrap["round_dir"])

    candidate_payload = """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
"""
    for iteration in ("001", "002"):
        candidate_dir = round_dir / "candidates" / iteration
        candidate_dir.mkdir()
        candidate_dir.joinpath("candidate.yaml").write_text(candidate_payload, encoding="utf-8")
        manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
        manager.commit_candidate(
            round_id=round_id,
            iteration=iteration,
            status="keep",
            notes=f"synthetic candidate {iteration}",
            allow_synthetic=True,
        )

    second_trace_path = round_dir / "candidates" / "002" / "measurement_trace.json"
    second_trace = json.loads(second_trace_path.read_text(encoding="utf-8"))
    second_trace["generator"] = "RealMeasurementHarness v0.1.0"
    second_trace_path.write_text(json.dumps(second_trace, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="synthetic fixture trace"):
        manager.finalize_round(round_id=round_id, dry_run=True)


def test_finalize_round_dry_run_refuses_terminal_harness_fault_row(tmp_path: Path) -> None:
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
    round_id = bootstrap["round_id"]
    round_dir = Path(bootstrap["round_dir"])

    candidate_payload = """
max_num_seqs: 4
max_num_batched_tokens: 8192
enable_chunked_prefill: true
enable_prefix_caching: true
gpu_memory_utilization: 0.90
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
"""
    first_candidate_dir = round_dir / "candidates" / "001"
    first_candidate_dir.mkdir()
    first_candidate_dir.joinpath("candidate.yaml").write_text(candidate_payload, encoding="utf-8")
    manager.measure(round_id=round_id, candidate_path=first_candidate_dir / "candidate.yaml")
    manager.commit_candidate(
        round_id=round_id,
        iteration="001",
        status="keep",
        notes="first synthetic candidate",
        allow_synthetic=True,
    )

    second_candidate_dir = round_dir / "candidates" / "002"
    second_candidate_dir.mkdir()
    second_candidate_dir.joinpath("candidate.yaml").write_text(candidate_payload, encoding="utf-8")
    manager.measure(round_id=round_id, candidate_path=second_candidate_dir / "candidate.yaml")
    second_trace_path = second_candidate_dir / "measurement_trace.json"
    second_trace = json.loads(second_trace_path.read_text(encoding="utf-8"))
    second_trace["eval_throughput"] = -1.0
    second_trace["feasible"] = False
    second_trace["feasibility_failures"] = ["harness_fault"]
    second_trace_path.write_text(json.dumps(second_trace, indent=2), encoding="utf-8")
    manager.commit_candidate(
        round_id=round_id,
        iteration="002",
        status="harness_fault",
        notes="promql_mismatch",
        allow_synthetic=True,
    )

    with pytest.raises(RuntimeError, match="harness_fault row has no successor feasible run"):
        manager.finalize_round(round_id=round_id, dry_run=True)


def test_run_round_synthetic_completes_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    monkeypatch.setenv("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT", "1")
    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        sprint="sprint-0",
        workload_file=repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml",
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        harness_type="synthetic",
        skip_preflight=True,
    )
    ctx = RoundContext.from_bootstrap_json(
        bootstrap,
        harness_mode="synthetic",
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
        iteration_cap=1,
    )

    result = run_round(ctx)

    assert result.outcome == "ROUND_BUNDLE_READY"
    assert result.live_gate == "skipped_fixture_mode"
    assert result.bundle_path is not None
    assert Path(result.bundle_path).is_file()
    report = json.loads((ctx.round_dir / "round_result.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "lumo.auto_research.round_result.v1"
    assert report["outcome"] == "ROUND_BUNDLE_READY"
    assert report["round_id"] == result.round_id
    generators = {
        json.loads(path.read_text(encoding="utf-8"))["generator"]
        for path in sorted((ctx.round_dir / "candidates").glob("*/measurement_trace.json"))
    }
    assert generators
    assert all(generator.startswith("SyntheticMeasurementFixture") for generator in generators)
    fixture_trailers = subprocess.run(
        [
            "git",
            "log",
            "--format=%H%x00%(trailers:key=Fixture-Mode,valueonly)",
            f"main..{bootstrap['round_branch']}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    commit_trailers = {
        line.split("\0", 1)[0]: line.split("\0", 1)[1]
        for line in fixture_trailers
        if "\0" in line
    }
    assert commit_trailers
    assert all(value == "true" for value in commit_trailers.values())


def test_run_round_exit_code_distinguishes_honest_terminal_outcomes() -> None:
    def result(outcome: str, live_gate: str = "not_run") -> RoundResult:
        return RoundResult(
            round_id="round",
            round_branch="autoresearch/test",
            outcome=outcome,
            stopping_reason="test",
            bundle_path=None,
            iterations_total=0,
            feasible_count=0,
            rescreened_count=0,
            holdout_validation="not_run",
            live_gate=live_gate,
        )

    assert run_round_exit_code(result("ROUND_PASSED", live_gate="pass")) == 0
    assert run_round_exit_code(result("ROUND_INFEASIBLE", live_gate="skipped_no_bundle")) == 0
    assert run_round_exit_code(result("ROUND_BUNDLE_READY", live_gate="skipped_fixture_mode")) == 0
    assert run_round_exit_code(result("ROUND_BUNDLE_READY", live_gate="not_run")) == 1
    assert run_round_exit_code(result("ROUND_BLOCKED")) == 1
    assert run_round_exit_code(result("ROUND_BUNDLE_REJECTED", live_gate="fail")) == 1

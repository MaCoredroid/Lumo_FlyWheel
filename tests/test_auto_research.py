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
from lumo_flywheel_serving.tuned_config import StructuredValidationError, load_tuned_config_bundle, validate_bundle_load_policy
from lumo_flywheel_serving.workload_p1 import write_heavy_workload_descriptor


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
workload_distribution_id: null
workload_distribution_id_hardening_version: v1-thinking-realistic
latency_ceiling_ms: 35000
nominal_ttft_ms: 2000
nominal_tpot_ms: 80
nominal_turn_ms: 30000
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
    workload = auto_research.load_yaml_file(path)
    assert isinstance(workload, dict)
    workload["workload_distribution_id"] = auto_research.compute_workload_distribution_id(path)
    path.write_text(auto_research.yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")


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


def _write_composite_workload(repo: Path, family_id: str = "multi-family-v5") -> Path:
    workload_dir = repo / "benchmark_blueprints" / "workloads" / family_id
    workload_path = workload_dir / "workload.yaml"
    _write_workload(workload_path)
    workload = auto_research.load_yaml_file(workload_path)
    assert isinstance(workload, dict)
    workload["family_id"] = family_id
    workload["workload_distribution_id_hardening_version"] = auto_research.HARDENED_COMPOSITE_WORKLOAD_VERSION
    workload["workload_distribution_id"] = None
    workload_path.write_text(auto_research.yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
    workload["workload_distribution_id"] = auto_research.compute_workload_distribution_id(workload_path)
    workload_path.write_text(auto_research.yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
    return workload_path


def _write_l0_heavy_workload(repo: Path) -> Path:
    source_family = "responses-sdk-adapter-cutover"
    family_dir = repo / "benchmark_blueprints" / "families" / source_family
    _write_trace(family_dir / "seed_trace_v5.jsonl", prompt_tokens=4096, output_tokens=1200)
    workload_dir = repo / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy"
    _write_trace(workload_dir / "seed_trace.jsonl", prompt_tokens=4096, output_tokens=1200)
    _write_trace(workload_dir / "holdout_trace.jsonl", prompt_tokens=3072, output_tokens=900)
    for trace_path in (workload_dir / "seed_trace.jsonl", workload_dir / "holdout_trace.jsonl"):
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["family_id"] = source_family
            row["thinking_tokens"] = max(1, int(row.get("output_tokens", 1)))
        trace_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    source_rows = [json.loads(line) for line in (family_dir / "seed_trace_v5.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in source_rows:
        row["family_id"] = source_family
        row["thinking_tokens"] = max(1, int(row.get("output_tokens", 1)))
    (family_dir / "seed_trace_v5.jsonl").write_text(
        "\n".join(json.dumps(row) for row in source_rows) + "\n",
        encoding="utf-8",
    )
    return write_heavy_workload_descriptor(
        repo_root=repo,
        capture_date="2026-04-25T00:00:00Z",
        thinking_probe_ref="reports/thinking-probe-20260424.md",
    )


def _write_l0a_fixture_pair(repo: Path, source_family: str = "responses-sdk-adapter-cutover") -> None:
    fixture_dir = repo / "benchmark_blueprints" / "families" / source_family / "parity_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "probes_input.jsonl").write_text(
        "\n".join(json.dumps({"probe_index": index, "prompt": f"probe {index}"}) for index in range(4)) + "\n",
        encoding="utf-8",
    )
    for kernel in ("deltanet", "gatedattn"):
        (fixture_dir / f"{kernel}_reference_logits.npz").write_bytes(b"dummy logits")
        payload = {
            "fixture_id": f"{source_family}-{kernel}-v1",
            "probe_input_ref": "probes_input.jsonl",
            "reference_logits_ref": f"{kernel}_reference_logits.npz",
            "generated_against": {
                "weight_version_id": "2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
            },
        }
        if kernel == "deltanet":
            (fixture_dir / "deltanet_reference_state.npz").write_bytes(b"dummy state")
            payload["reference_state_snapshots_ref"] = "deltanet_reference_state.npz"
        (fixture_dir / f"{kernel}_v1.yaml").write_text(
            auto_research.yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )


def _write_l0a_workload(repo: Path) -> Path:
    workload_dir = repo / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy"
    workload_dir.mkdir(parents=True, exist_ok=True)
    _write_trace(workload_dir / "seed_trace.jsonl", prompt_tokens=4096, output_tokens=1200)
    _write_trace(workload_dir / "holdout_trace.jsonl", prompt_tokens=3072, output_tokens=900)
    workload_path = workload_dir / "workload.yaml"
    workload_path.write_text(
        auto_research.yaml.safe_dump(
            {
                "family_id": "responses-sdk-adapter-cutover-heavy",
                "source_family": "responses-sdk-adapter-cutover",
                "workload_distribution_id": None,
                "workload_distribution_id_hardening_version": "v2-l0-kernel-heavy",
                "seed_trace_ref": "seed_trace.jsonl",
                "holdout_trace_ref": "holdout_trace.jsonl",
                "parity_fixture_refs": {
                    "deltanet": "parity_fixture/deltanet_v1.yaml",
                    "gatedattn": "parity_fixture/gatedattn_v1.yaml",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = auto_research.load_yaml_file(workload_path)
    assert isinstance(payload, dict)
    payload["workload_distribution_id"] = auto_research.compute_workload_distribution_id(workload_path)
    workload_path.write_text(auto_research.yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return workload_path


def _write_l0a_action_space(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
axes:
  attention_backend: [vllm-default, flash-attn-4, flashinfer]
  deltanet_kernel: [triton-chunked-delta-v2, triton-experimental-scan]
  fp8_gemm_kernel: [cublas, cutlass]
  torch_compile_mode: [default, reduce-overhead]
  cuda_graph_capture: ['off', 'on']
""",
        encoding="utf-8",
    )
    return path


def _write_l0a_bundle(repo: Path, *, kernel_selection: dict | None = None) -> Path:
    bundle = auto_research.make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="responses-sdk-adapter-cutover-heavy",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id=auto_research.compute_workload_distribution_id(
            repo / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy" / "workload.yaml"
        ),
        vllm_config={
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 131072,
            "kv_cache_dtype": "fp8_e5m2",
        },
        kernel_selection=kernel_selection
        or {
            "combo_id": "combo_001",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "cublas",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
        objective={"metric": "l0a_rescreen_objective_mean", "value": 1.20},
        measurement_trace_ref="output/auto_research/l0a/measurement_trace_combined.json",
        search_trace_ref="output/auto_research/l0a/search_trace.json",
        baseline_bundle_id=None,
        regression_guard={},
        safety_rails={"determinism_check_passed": True, "parity_check_passed": True},
        round_provenance={
            "round_type": "l0a_select_only",
            "workload_descriptor_path": str(
                repo / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy" / "workload.yaml"
            ),
            "confidence": "defensible",
        },
    )
    return auto_research.persist_tuned_config_bundle(bundle, repo / "output" / "tuned_configs")


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
    assert workload["workload_distribution_id"] == payload["workload_distribution_id"]
    assert workload["workload_distribution_id"] != payload["seed_sha256"]
    assert workload["workload_distribution_id"] == auto_research.compute_workload_distribution_id(workload_path)


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


def test_l2_candidate_plan_varies_only_enforced_fields() -> None:
    frozen_vllm_config = {
        "max_num_seqs": 4,
        "max_model_len": 131072,
    }

    candidates = auto_research.AutoResearchRoundManager._request_shaping_candidate_plan(frozen_vllm_config)

    assert candidates
    assert {candidate["per_request_kv_budget"] for candidate in candidates} == {131072}
    assert {candidate["priority_preemption"] for candidate in candidates} == {"off"}
    assert len(
        {
            (
                candidate["concurrency_cap_eval"],
                candidate["concurrency_cap_rollout"],
                candidate["admission_queue_depth_max"],
            )
            for candidate in candidates
        }
    ) == len(candidates)


def test_l2_iteration_prompt_keeps_advisory_fields_out_of_action_space(tmp_path: Path) -> None:
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
    ctx = RoundContext.from_bootstrap_json(
        bootstrap,
        harness_mode="synthetic",
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    prompt = round_driver._iteration_prompt(ctx, iteration="001", next_iteration="002")

    assert "Vary only the three enforced fields" in prompt
    assert "concurrency_cap_eval, concurrency_cap_rollout, admission_queue_depth_max" in prompt
    assert "Keep advisory fields fixed as metadata: per_request_kv_budget=131072, priority_preemption=off" in prompt


def test_l2_enforcement_validation_rejects_missing_advisory_metadata() -> None:
    request_shaping = {
        "concurrency_cap_eval": 3,
        "concurrency_cap_rollout": 1,
        "admission_queue_depth_max": 64,
        "per_request_kv_budget": 65536,
        "priority_preemption": "strict",
    }
    record = {
        "mode": "enforced",
        "real_proxy_enforcement": True,
        "enforced_fields": [
            "concurrency_cap_eval",
            "concurrency_cap_rollout",
            "admission_queue_depth_max",
        ],
        "advisory_fields": [],
        "field_values": {
            "concurrency_cap_eval": {"value": 3, "enforcement": "enforced"},
            "concurrency_cap_rollout": {"value": 1, "enforcement": "enforced"},
            "admission_queue_depth_max": {"value": 64, "enforcement": "enforced"},
        },
    }

    with pytest.raises(RuntimeError, match="advisory_fields mismatch"):
        auto_research.AutoResearchRoundManager._validate_l2_enforcement_record(
            record,
            context="AR.28 L2 enforcement coverage for candidate-001",
            request_shaping=request_shaping,
        )


def test_bootstrap_prefers_composite_descriptor_and_enforces_version_pin(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    workload_path = _write_composite_workload(repo)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "add composite workload"], cwd=repo, check=True, capture_output=True, text=True)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="multi-family-v5",
        sprint="sprint-0",
        workload_file=None,
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        harness_type="synthetic",
    )
    round_spec = auto_research.load_yaml_file(Path(bootstrap["round_spec_path"]))

    assert round_spec["workload_descriptor_path"] == str(workload_path.resolve())
    assert round_spec["workload_distribution_id_hardening_version"] == auto_research.HARDENED_COMPOSITE_WORKLOAD_VERSION
    assert round_spec["workload_distribution_id"] == auto_research.compute_workload_distribution_id(workload_path)

    workload = auto_research.load_yaml_file(workload_path)
    assert isinstance(workload, dict)
    workload["workload_distribution_id_hardening_version"] = "legacy-version"
    workload_path.write_text(auto_research.yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")
    workload["workload_distribution_id"] = auto_research.compute_workload_distribution_id(workload_path)
    workload_path.write_text(auto_research.yaml.safe_dump(workload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="descriptor_stale_workload_distribution_id_hardening_version"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="multi-family-v5",
            sprint="sprint-1",
            workload_file=None,
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
            harness_type="synthetic",
        )

    legacy_bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="multi-family-v5",
        sprint="sprint-legacy",
        workload_file=None,
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        harness_type="synthetic",
        allow_legacy_workload=True,
    )
    legacy_round_spec = auto_research.load_yaml_file(Path(legacy_bootstrap["round_spec_path"]))
    assert legacy_round_spec["workload_distribution_id_hardening_version"] == "legacy-version"


def test_bootstrap_accepts_l0_heavy_workload_descriptor_version(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    workload_path = _write_l0_heavy_workload(repo)
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "add l0 heavy workload"], cwd=repo, check=True, capture_output=True, text=True)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    bootstrap = manager.bootstrap_round(
        model_id="qwen3.5-27b",
        family_id="responses-sdk-adapter-cutover-heavy",
        sprint="sprint-0",
        workload_file=None,
        weight_version_id=None,
        round_root=repo / "output" / "auto_research",
        harness_type="synthetic",
    )
    round_spec = auto_research.load_yaml_file(Path(bootstrap["round_spec_path"]))

    assert round_spec["workload_descriptor_path"] == str(workload_path.resolve())
    assert round_spec["workload_distribution_id_hardening_version"] == "v2-l0-kernel-heavy"
    assert round_spec["workload_distribution_id"] == auto_research.compute_workload_distribution_id(workload_path)


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
    enforcement = trace["request_shaping_enforcement"]
    assert enforcement["mode"] == "enforced_minus_advisory"
    assert enforcement["real_proxy_enforcement"] is True
    assert enforcement["enforced_fields"] == [
        "concurrency_cap_eval",
        "concurrency_cap_rollout",
        "admission_queue_depth_max",
    ]
    assert enforcement["advisory_fields"] == ["per_request_kv_budget", "priority_preemption"]
    assert enforcement["field_values"]["per_request_kv_budget"] == {
        "value": 65536,
        "enforcement": "advisory",
        "reason": (
            "v0.2 records and validates this field, but the proxy does not enforce it until "
            "real KV accounting and scheduler preemption hooks exist."
        ),
    }


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
    assert bundle["round_provenance"]["request_shaping_enforcement"]["real_proxy_enforcement"] is True
    assert bundle["round_provenance"]["l2_enforcement_coverage"]["mode"] == "enforced_minus_advisory"
    assert bundle["round_provenance"]["l2_enforcement_coverage"]["real_proxy_enforcement"] is True
    assert bundle["round_provenance"]["l2_enforcement_coverage"]["enforced_fields"] == [
        "concurrency_cap_eval",
        "concurrency_cap_rollout",
        "admission_queue_depth_max",
    ]
    assert bundle["round_provenance"]["l2_enforcement_coverage"]["advisory_fields"] == [
        "per_request_kv_budget",
        "priority_preemption",
    ]
    assert bundle["round_provenance"]["l2_enforcement_coverage"]["field_values"]["priority_preemption"][
        "enforcement"
    ] == "advisory"


def test_l2_finalize_rejects_stale_trace_enforcement_metadata(
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

    trace_path = round_dir / "candidates" / "rescreen_01_screen_1" / "measurement_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["request_shaping_enforcement"] = {
        "mode": "substrate_measurement_only",
        "real_proxy_enforcement": False,
        "enforced_fields": ["concurrency_cap_eval"],
        "advisory_fields": [],
    }
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="AR\\.28 L2 enforcement coverage"):
        manager.finalize_round(round_id=round_id, dry_run=False)


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
            _real_trace(objective=10, ttft=1450.0, tpot=12.0, turn=4100.0),
            _real_trace(objective=10, ttft=1460.0, tpot=12.0, turn=4100.0),
            _real_trace(objective=10, ttft=1470.0, tpot=12.0, turn=4100.0),
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

    for iteration in ("baseline_b", "baseline_c", "baseline_d", "baseline_e"):
        manager.measure(round_id=round_id, candidate_path=round_dir / "candidates" / iteration / "candidate.yaml")
        manager.commit_candidate(
            round_id=round_id,
            iteration=iteration,
            status="baseline",
            notes=f"default baseline replay {iteration}",
        )

    updated_round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert isinstance(updated_round_spec, dict)
    assert updated_round_spec["baseline_mean_screen"] == pytest.approx(10.0)
    assert updated_round_spec["baseline_stddev_screen"] == pytest.approx(0.70710678)
    assert updated_round_spec["noise_floor"] == pytest.approx(1.41421356)
    assert manager.status(round_id=round_id)["noise_floor"] == pytest.approx(1.41421356)


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


def test_bootstrap_round_verifies_descriptor_id_without_minting(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    workload_path = repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"
    workload = auto_research.load_yaml_file(workload_path)
    assert isinstance(workload, dict)
    original_id = workload["workload_distribution_id"]

    missing = dict(workload)
    missing.pop("workload_distribution_id")
    workload_path.write_text(auto_research.yaml.safe_dump(missing, sort_keys=False), encoding="utf-8")
    with pytest.raises(RuntimeError, match="descriptor_missing_workload_distribution_id"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="proposal-ranking-manager-judgment",
            sprint="sprint-0",
            workload_file=workload_path,
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
            harness_type="synthetic",
        )
    assert "workload_distribution_id" not in auto_research.load_yaml_file(workload_path)

    stale = dict(workload)
    stale["workload_distribution_id"] = original_id
    workload_path.write_text(auto_research.yaml.safe_dump(stale, sort_keys=False), encoding="utf-8")
    (workload_path.parent / "seed_trace.jsonl").write_text('{"turn_index": 999, "prompt_tokens": 1, "output_tokens": 1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="descriptor_workload_distribution_id_mismatch"):
        manager.bootstrap_round(
            model_id="qwen3.5-27b",
            family_id="proposal-ranking-manager-judgment",
            sprint="sprint-0",
            workload_file=workload_path,
            weight_version_id=None,
            round_root=repo / "output" / "auto_research",
            harness_type="synthetic",
        )


def test_finalize_round_populates_hardened_honesty_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    manager = auto_research.AutoResearchRoundManager(
        registry_path=repo / "model_registry.yaml",
        repo_root=repo,
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    measurements = iter(
        [
            *[_real_trace(objective=10, ttft=1500.0, turn=4000.0) for _ in range(5)],
            _real_trace(objective=14, ttft=1500.0, turn=4000.0),
            _real_trace(objective=14, ttft=2500.0, turn=4000.0),
            _real_trace(objective=14, ttft=1500.0, turn=4000.0),
            _real_trace(objective=14, ttft=1500.0, turn=4000.0),
            _real_trace(objective=5, ttft=1500.0, turn=4000.0),
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
    for iteration in auto_research.BASELINE_ITERATIONS:
        manager.measure(round_id=round_id, candidate_path=round_dir / "candidates" / iteration / "candidate.yaml")
        manager.commit_candidate(round_id=round_id, iteration=iteration, status="baseline", notes=iteration)
    candidate_dir = round_dir / "candidates" / "001"
    candidate_dir.mkdir()
    candidate_dir.joinpath("candidate.yaml").write_text(
        (round_dir / "candidates" / "baseline_a" / "candidate.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    measured = manager.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
    manager.commit_candidate(round_id=round_id, iteration="001", status="keep", notes="winner")
    manager.rescreen(round_id=round_id, top_k=1)
    (round_dir / "holdout_trace.json").write_text(
        json.dumps({"pass": True, "candidate_uuid": measured["candidate_uuid"]}, indent=2),
        encoding="utf-8",
    )

    finalized = manager.finalize_round(round_id=round_id, dry_run=False)
    bundle_payload = auto_research.load_yaml_file(finalized["bundle_path"])
    assert isinstance(bundle_payload, dict)
    provenance = bundle_payload["tuned_config_bundle"]["round_provenance"]
    assert provenance["confidence"] == "defensible"
    assert provenance["improvement_over_baseline_req_per_s"] == pytest.approx(4.0)
    assert provenance["improvement_over_baseline_ci_95"] == [4.0, 4.0]
    assert provenance["latency_above_slo"] is True
    assert provenance["screen_full_consistency"] == "divergent"
    assert provenance["l2_enforcement_coverage"]["mode"] == "not_l2"
    assert provenance["workload_descriptor_path"] == str(
        repo / "benchmark_blueprints" / "families" / "proposal-ranking-manager-judgment" / "serving_workload.yaml"
    )
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert "screen_full_divergence_note" in run_log["diagnostics"]


def test_replay_round_imports_candidate_without_agent_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _init_repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT", "1")
    imported_candidate = repo / "candidate.yaml"
    imported_candidate.write_text(
        """
max_num_batched_tokens: 8192
max_num_seqs: 4
gpu_memory_utilization: 0.9
enable_chunked_prefill: true
enable_prefix_caching: true
max_model_len: 131072
kv_cache_dtype: fp8_e5m2
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "candidate.yaml"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "candidate"], cwd=repo, check=True, capture_output=True, text=True)

    result = round_driver.run_replay_round(
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
        port=8000,
        proxy_port=8001,
        workload_file=repo
        / "benchmark_blueprints"
        / "families"
        / "proposal-ranking-manager-judgment"
        / "serving_workload.yaml",
        baselines=5,
        import_candidate=imported_candidate,
        rescreens_screen=3,
        rescreens_full=1,
        holdout_rows=1,
        round_root=repo / "output" / "auto_research",
        harness_mode="synthetic",
        model_id="qwen3.5-27b",
    )

    round_dir = Path(result["round_dir"])
    assert result["outcome"] == round_driver.ROUND_BUNDLE_READY
    assert not list(round_dir.glob("candidates/*/agent_session.jsonl"))
    assert (round_dir / "candidates" / "import_001" / "candidate.yaml").read_text(encoding="utf-8") == imported_candidate.read_text(
        encoding="utf-8"
    )
    bundle_payload = auto_research.load_yaml_file(result["bundle_path"])
    provenance = bundle_payload["tuned_config_bundle"]["round_provenance"]
    assert provenance["round_type"] == "replay"
    assert provenance["imported_from_candidate"] == str(imported_candidate.resolve())
    assert provenance["imported_from_commit"]


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
    assert loaded_bundle_path.exists()
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
            _real_trace(objective=9, ttft=1435.0, tpot=11.0, turn=3975.0),
            _real_trace(objective=9, ttft=1445.0, tpot=11.0, turn=3985.0),
            _real_trace(objective=9, ttft=1460.0, tpot=11.0, turn=3990.0),
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
    assert len(rescreen["rescreened"]) == 4
    assert rescreen["rescreened"][0]["parent_candidate_uuid"] == baseline_uuids["a"]
    assert sum(1 for row in rescreen["rescreened"] if row["profile"] == "screen") == 3
    assert sum(1 for row in rescreen["rescreened"] if row["profile"] == "full") == 1

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
    measurements = iter([_real_trace(objective=9) for _ in range(6)])
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
                {"objective": 10, "ttft": 2600.0, "tpot": 14.0, "turn": 4700.0},
                {"objective": 10, "ttft": 1200.0, "tpot": 10.0, "turn": 3900.0},
                {"objective": 10, "ttft": 1250.0, "tpot": 10.0, "turn": 3950.0},
                {"objective": 10, "ttft": 1300.0, "tpot": 10.0, "turn": 4000.0},
                {"objective": 9, "ttft": 1500.0, "tpot": 10.0, "turn": 4100.0},
                {"objective": 9, "ttft": 1500.0, "tpot": 10.0, "turn": 4100.0},
                {"objective": 9, "ttft": 1500.0, "tpot": 10.0, "turn": 4100.0},
                {"objective": 9, "ttft": 1500.0, "tpot": 10.0, "turn": 4100.0},
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
    assert len(rescreen["rescreened"]) == 8
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


def test_run_round_synthetic_l2_records_ar28_enforcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    bundle_path = _write_l1_bundle(repo)
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
        active_layer="L2",
        baseline_bundle=bundle_path,
    )
    ctx = RoundContext.from_bootstrap_json(
        bootstrap,
        harness_mode="synthetic",
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
        iteration_cap=2,
    )

    result = run_round(ctx)

    assert result.outcome == "ROUND_BUNDLE_READY"
    assert result.bundle_path is not None
    traces = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(ctx.round_dir.glob("candidates/*/measurement_trace.json"))
    ]
    assert traces
    for trace in traces:
        enforcement = trace["request_shaping_enforcement"]
        assert enforcement["mode"] == "enforced_minus_advisory"
        assert enforcement["real_proxy_enforcement"] is True
        assert enforcement["enforced_fields"] == [
            "concurrency_cap_eval",
            "concurrency_cap_rollout",
            "admission_queue_depth_max",
        ]
        assert set(enforcement["advisory_fields"]).issubset({"per_request_kv_budget", "priority_preemption"})
        assert enforcement["field_values"]["per_request_kv_budget"]["enforcement"] == "advisory"
        assert enforcement["field_values"]["priority_preemption"]["enforcement"] == "advisory"
    generated_candidates = [
        auto_research.load_yaml_file(path)
        for path in sorted(ctx.round_dir.glob("candidates/[0-9][0-9][0-9]/candidate.yaml"))
    ]
    assert {candidate["per_request_kv_budget"] for candidate in generated_candidates} == {131072}
    assert {candidate["priority_preemption"] for candidate in generated_candidates} == {"off"}
    bundle = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    coverage = bundle["round_provenance"]["l2_enforcement_coverage"]
    assert coverage["mode"] == "enforced_minus_advisory"
    assert coverage["real_proxy_enforcement"] is True
    assert coverage["enforced_fields"] == [
        "concurrency_cap_eval",
        "concurrency_cap_rollout",
        "admission_queue_depth_max",
    ]
    assert coverage["advisory_fields"] == ["per_request_kv_budget", "priority_preemption"]


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


def _agent_runtime_ctx(
    tmp_path: Path,
    *,
    agent_runtime: str,
    round_spec: dict | None = None,
) -> RoundContext:
    worktree = tmp_path / "worktree"
    round_dir = tmp_path / "round"
    worktree.mkdir()
    round_dir.mkdir()
    return RoundContext(
        round_id="round-test",
        round_dir=round_dir,
        round_branch="autoresearch/test",
        worktree=worktree,
        round_spec_path=round_dir / "round_spec.yaml",
        round_spec=round_spec or {},
        harness_mode="real",
        registry_path=tmp_path / "model_registry.yaml",
        tuned_config_root=tmp_path / "output" / "tuned_configs",
        iteration_cap=1,
        agent_runtime=agent_runtime,
    )


def test_agent_invocation_codex_argv_is_unchanged(tmp_path: Path) -> None:
    ctx = _agent_runtime_ctx(tmp_path, agent_runtime="codex")
    last_message_path = ctx.round_dir / "last.txt"
    argv, timeout = round_driver._agent_invocation(
        ctx,
        iteration_dir=ctx.round_dir,
        last_message_path=last_message_path,
    )
    assert argv == [
        "codex",
        "-c",
        'model="gpt-5.4"',
        "-c",
        'model_reasoning_effort="high"',
        "exec",
        "--cd",
        str(ctx.worktree),
        "--json",
        "--output-last-message",
        str(last_message_path),
        "--skip-git-repo-check",
        "-",
    ]
    assert timeout == 45 * 60


def test_agent_invocation_claude_argv_uses_claude_cli_and_anthropic_auth(tmp_path: Path) -> None:
    ctx = _agent_runtime_ctx(tmp_path, agent_runtime="claude")
    argv, timeout = round_driver._agent_invocation(
        ctx,
        iteration_dir=ctx.round_dir,
        last_message_path=ctx.round_dir / "last.txt",
    )
    assert argv[0] == "claude"
    assert "-p" in argv
    assert ["--output-format", "stream-json"] == argv[argv.index("--output-format"):argv.index("--output-format") + 2]
    assert "--verbose" in argv
    assert ["--model", round_driver.DEFAULT_CLAUDE_MODEL] == argv[argv.index("--model"):argv.index("--model") + 2]
    assert ["--effort", round_driver.DEFAULT_CLAUDE_EFFORT] == argv[argv.index("--effort"):argv.index("--effort") + 2]
    assert ["--permission-mode", round_driver.DEFAULT_CLAUDE_PERMISSION_MODE] == argv[
        argv.index("--permission-mode"):argv.index("--permission-mode") + 2
    ]
    assert ["--add-dir", str(ctx.worktree)] == argv[argv.index("--add-dir"):argv.index("--add-dir") + 2]
    assert timeout == 45 * 60


def test_agent_invocation_claude_round_spec_overrides_apply(tmp_path: Path) -> None:
    ctx = _agent_runtime_ctx(
        tmp_path,
        agent_runtime="claude",
        round_spec={
            "claude_model": "claude-sonnet-4-6",
            "claude_effort": "medium",
            "claude_permission_mode": "acceptEdits",
            "per_iteration_claude_wall_clock_s": 600,
        },
    )
    argv, timeout = round_driver._agent_invocation(
        ctx,
        iteration_dir=ctx.round_dir,
        last_message_path=ctx.round_dir / "last.txt",
    )
    assert ["--model", "claude-sonnet-4-6"] == argv[argv.index("--model"):argv.index("--model") + 2]
    assert ["--effort", "medium"] == argv[argv.index("--effort"):argv.index("--effort") + 2]
    assert ["--permission-mode", "acceptEdits"] == argv[
        argv.index("--permission-mode"):argv.index("--permission-mode") + 2
    ]
    assert timeout == 600


def test_extract_claude_last_message_picks_final_result(tmp_path: Path) -> None:
    transcript = tmp_path / "agent_session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "system", "subtype": "init"}),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "interim text"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "FINAL_REPORT_LINE",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    last_message_path = tmp_path / "agent_last_message.txt"
    round_driver._extract_claude_last_message(transcript, last_message_path)
    assert last_message_path.read_text(encoding="utf-8") == "FINAL_REPORT_LINE"


def test_extract_claude_last_message_falls_back_to_last_assistant_text(tmp_path: Path) -> None:
    transcript = tmp_path / "agent_session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "first"}]},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"content": [{"type": "text", "text": "second_final"}]},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    last_message_path = tmp_path / "agent_last_message.txt"
    round_driver._extract_claude_last_message(transcript, last_message_path)
    assert last_message_path.read_text(encoding="utf-8") == "second_final"


def test_round_context_rejects_unknown_agent_runtime(tmp_path: Path) -> None:
    bootstrap_payload = {
        "round_id": "r-1",
        "round_dir": str(tmp_path),
        "round_branch": "branch",
        "round_spec_path": str(tmp_path / "round_spec.yaml"),
        "worktree_path": str(tmp_path),
    }
    (tmp_path / "round_spec.yaml").write_text("model_id: x\nfamily_id: y\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid agent_runtime"):
        RoundContext.from_bootstrap_json(
            bootstrap_payload,
            harness_mode="synthetic",
            registry_path=tmp_path / "registry.yaml",
            tuned_config_root=tmp_path / "tc",
            agent_runtime="bogus",
        )


def test_run_agent_main_loop_dispatches_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        harness_mode="real",
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
        iteration_cap=1,
        agent_runtime="claude",
    )

    captured: dict = {}
    real_run = round_driver.subprocess.run

    class _FakeCompleted:
        returncode = 0
        stderr = b""

    def _fake_run(argv, *args, **kwargs):
        if not argv or argv[0] not in {"codex", "claude"}:
            return real_run(argv, *args, **kwargs)
        captured["argv"] = argv
        captured["cwd"] = kwargs.get("cwd")
        captured["timeout"] = kwargs.get("timeout")
        captured["input"] = kwargs.get("input")
        stdout = kwargs.get("stdout")
        if stdout is not None:
            stdout.write(
                (
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "result": "iteration-1-final",
                        }
                    )
                    + "\n"
                ).encode("utf-8")
            )
        return _FakeCompleted()

    monkeypatch.setattr(round_driver.subprocess, "run", _fake_run)
    # Short-circuit the loop after one iteration by returning a status that says we advanced.
    monkeypatch.setattr(
        manager,
        "status",
        lambda round_id: {"iterations_total": 99, "feasible_count": 0, "rescreened_count": 0},
    )

    round_driver._run_agent_main_loop(manager, ctx)

    assert captured["argv"][0] == "claude"
    assert captured["cwd"] == str(ctx.worktree)
    iteration_dir = ctx.round_dir / "candidates" / "001"
    assert (iteration_dir / "agent_session.jsonl").read_text(encoding="utf-8").strip().startswith("{")
    assert (iteration_dir / "agent_last_message.txt").read_text(encoding="utf-8") == "iteration-1-final"


def test_l0a_kernel_select_synthetic_writes_p3_artifacts_and_refuses_production_load(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        action_space_file=action_space_path,
        baselines=5,
        screen_measurements_per_combo=2,
        rescreen_top_k=8,
        rescreen_measurements_per_candidate=4,
        parallel_instances="auto",
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
    )

    assert result.total_combos == 48
    assert result.survivor_count > 0
    eliminated = (result.round_dir / "eliminated.tsv").read_text(encoding="utf-8").splitlines()
    header = eliminated[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in eliminated[1:]]
    assert {row["elimination_reason"] for row in rows} == {
        "nondeterministic",
        "parity_diverges_from_reference",
    }
    parity_rows = [row for row in rows if row["elimination_reason"] == "parity_diverges_from_reference"]
    assert parity_rows
    assert all(row["first_diverging_probe_index"] for row in parity_rows)
    assert all(float(row["tolerance_overshoot"]) > 0.0 for row in parity_rows)

    determinism_log = json.loads((result.round_dir / "determinism_log.json").read_text(encoding="utf-8"))
    parity_check = json.loads((result.round_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert determinism_log["pass"] is True
    assert determinism_log["probe_count"] == 64
    assert parity_check["pass"] is True
    assert parity_check["reason"] == "ran_passed"
    run_log = json.loads((result.round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["artifact_counts"]["baseline_rows"] == 5
    assert run_log["artifact_counts"]["rescreen_rows"] == 32

    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert bundle_payload["round_provenance"]["round_type"] == "l0a_select_only"
    assert bundle_payload["round_provenance"]["parallel_instances"] == 1
    assert bundle_payload["kernel_selection"]["attention_backend"] == "flash-attn-4"
    assert bundle_payload["kernel_selection"]["deltanet_kernel"] == "triton-chunked-delta-v2"

    with pytest.raises(StructuredValidationError, match="bundle-validity: refused"):
        validate_bundle_load_policy(load_tuned_config_bundle(result.bundle_path), bundle_confidence_policy="passthrough")


def test_l0a_kernel_select_real_dispatches_live_smoke_with_runtime_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    calls: list[dict[str, object]] = []

    class _FakeRealMeasurementHarness:
        VERSION = "RealMeasurementHarness v0.1.0"

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            return {
                "generator": self.VERSION,
                "candidate_vllm_config": candidate_vllm_config,
                "cache_isolation": {},
                "windows": {"measurement_elapsed_s": 1.0},
                "per_request_latencies": [],
                "diagnostics": {},
                "ttft_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "tpot_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "turn_latency_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "eval_throughput": 1.25,
                "rollout_throughput": 10.0,
                "window_completed": True,
                "reasoning_content_purity": 1.0,
                "determinism_pass_rate": 1.0,
                "no_oom_events": True,
                "feasible": True,
                "feasibility_failures": [],
                "harness_health_warnings": [],
            }

        def restore_runtime(self) -> None:
            return None

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        action_space_file=action_space_path,
        baselines=1,
        screen_measurements_per_combo=1,
        rescreen_top_k=1,
        rescreen_measurements_per_candidate=1,
        parallel_instances="auto",
        round_root=repo / "output" / "auto_research",
        harness="real",
        max_combos=1,
        proxy_port=8101,
    )

    round_dir = result.round_dir
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "PASS"
    assert run_log["kernel_selection_runtime_activation"] == "runtime_applied"
    assert run_log["limited_mode"] is True
    assert run_log["live_dispatch"]["baseline_rows"] == 1
    assert run_log["live_dispatch"]["screen_rows"] == 1
    assert run_log["live_dispatch"]["rescreen_rows"] == 1
    assert len(calls) == 3
    assert calls[0]["target_concurrency"] == 1
    assert calls[1]["kernel_selection"] == {
        "combo_id": "combo_001",
        "attention_backend": "vllm-default",
        "deltanet_kernel": "triton-chunked-delta-v2",
        "fp8_gemm_kernel": "cublas",
        "torch_compile_mode": "default",
        "cuda_graph_capture": "off",
    }
    measurements = (round_dir / "measurements.tsv").read_text(encoding="utf-8")
    assert "kernel_selection_applied" in measurements
    assert "\truntime" in measurements


def test_l0a_kernel_select_real_reaches_reduce_overhead_runtime_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    calls: list[dict[str, object]] = []

    class _FakeRealMeasurementHarness:
        VERSION = "RealMeasurementHarness v0.1.0"

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            return {
                "generator": self.VERSION,
                "candidate_vllm_config": candidate_vllm_config,
                "cache_isolation": {},
                "windows": {"measurement_elapsed_s": 1.0},
                "per_request_latencies": [],
                "diagnostics": {},
                "ttft_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "tpot_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "turn_latency_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "eval_throughput": 1.25,
                "rollout_throughput": 10.0,
                "window_completed": True,
                "reasoning_content_purity": 1.0,
                "determinism_pass_rate": 1.0,
                "no_oom_events": True,
                "feasible": True,
                "feasibility_failures": [],
                "harness_health_warnings": [],
            }

        def restore_runtime(self) -> None:
            return None

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        action_space_file=action_space_path,
        baselines=1,
        screen_measurements_per_combo=1,
        rescreen_top_k=2,
        rescreen_measurements_per_candidate=1,
        parallel_instances="auto",
        round_root=repo / "output" / "auto_research",
        harness="real",
        max_combos=3,
        proxy_port=8101,
    )

    assert result.survivor_count == 2
    dispatched_kernel_selections = [
        call.get("kernel_selection")
        for call in calls
        if isinstance(call.get("kernel_selection"), dict) and call["kernel_selection"]
    ]
    assert any(
        selection["combo_id"] == "combo_003"
        and selection["torch_compile_mode"] == "reduce-overhead"
        for selection in dispatched_kernel_selections
    )
    run_log = json.loads((result.round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "PASS"
    assert run_log["kernel_selection_runtime_activation"] == "runtime_applied"


def test_l0a_kernel_select_real_blocks_precisely_on_unsupported_runtime_knobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2, triton-state-update-fused, triton-experimental-scan]
  fp8_gemm_kernel: [cublas, cutlass]
  torch_compile_mode: [default, reduce-overhead]
  cuda_graph_capture: ['off', 'on']
""",
        encoding="utf-8",
    )
    harness_inits: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class _FakeRealMeasurementHarness:
        VERSION = "RealMeasurementHarness v0.1.0"

        def __init__(self, **kwargs: object) -> None:
            harness_inits.append(kwargs)
            self.kwargs = kwargs

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            return {}

        def restore_runtime(self) -> None:
            return None

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="HALT_REASON: l0a_kernel_selection_runtime_unsupported_knobs"):
        runner.run(
            workload_file=workload_path,
            action_space_file=action_space_path,
            baselines=1,
            screen_measurements_per_combo=1,
            rescreen_top_k=1,
            rescreen_measurements_per_candidate=1,
            parallel_instances="auto",
            round_root=repo / "output" / "auto_research",
            harness="real",
            max_combos=17,
            proxy_port=8101,
            runtime_unsupported_policy="strict",
        )

    round_dir = next((repo / "output" / "auto_research").glob("*-l0a-select-*"))
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "ROUND_BLOCKED"
    assert run_log["HALT_REASON"] == "l0a_kernel_selection_runtime_unsupported_knobs"
    assert run_log["live_dispatch"]["attempted"] is False
    assert run_log["runtime_activation_check_ref"] == "runtime_activation_check.json"
    assert run_log["unsupported_runtime_activation"][0]["combo_id"] == "combo_009"
    assert run_log["unsupported_runtime_activation"][0]["unsupported_knobs"][0]["axis"] == "deltanet_kernel"
    activation_check = json.loads((round_dir / "runtime_activation_check.json").read_text(encoding="utf-8"))
    assert activation_check["status"] == "blocked"
    assert activation_check["checked_combo_count"] == 17
    assert activation_check["unsupported_combo_count"] == 9
    assert activation_check["unsupported_survivor_count"] == 4
    assert activation_check["unsupported_runtime_activation"][0]["smoke_status"] == "survivor"
    assert activation_check["unsupported_runtime_activation"][-1]["combo_id"] == "combo_017"
    assert activation_check["unsupported_runtime_activation"][-1]["smoke_status"] == "eliminated"
    assert harness_inits == []
    assert calls == []


def test_l0a_kernel_select_real_partitions_unsupported_runtime_knobs_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2, triton-state-update-fused, triton-experimental-scan]
  fp8_gemm_kernel: [cublas, cutlass]
  torch_compile_mode: [default, reduce-overhead]
  cuda_graph_capture: ['off', 'on']
""",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    class _FakeRealMeasurementHarness:
        VERSION = "RealMeasurementHarness v0.1.0"

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            kernel_selection = kwargs.get("kernel_selection")
            if isinstance(kernel_selection, dict) and kernel_selection:
                assert kernel_selection["deltanet_kernel"] == "triton-chunked-delta-v2"
            return {
                "generator": self.VERSION,
                "candidate_vllm_config": candidate_vllm_config,
                "cache_isolation": {},
                "windows": {"measurement_elapsed_s": 1.0},
                "per_request_latencies": [],
                "diagnostics": {},
                "ttft_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "tpot_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "turn_latency_p95_ms": {"driver": 1.0, "promql": 1.0, "delta_pct": 0.0},
                "eval_throughput": 1.25,
                "rollout_throughput": 10.0,
                "window_completed": True,
                "reasoning_content_purity": 1.0,
                "determinism_pass_rate": 1.0,
                "no_oom_events": True,
                "feasible": True,
                "feasibility_failures": [],
                "harness_health_warnings": [],
            }

        def restore_runtime(self) -> None:
            return None

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        action_space_file=action_space_path,
        baselines=1,
        screen_measurements_per_combo=1,
        rescreen_top_k=2,
        rescreen_measurements_per_candidate=1,
        parallel_instances="auto",
        round_root=repo / "output" / "auto_research",
        harness="real",
        max_combos=17,
        proxy_port=8101,
    )

    run_log = json.loads((result.round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "PASS"
    assert run_log["runtime_unsupported_policy"] == "partition"
    assert run_log["total_combos"] == 17
    assert run_log["survivor_count"] == 8
    assert run_log["runtime_supported_survivor_count"] == 4
    assert run_log["artifact_counts"]["runtime_unsupported_rows"] == 9
    assert run_log["artifact_counts"]["screen_rows"] == 4
    assert run_log["live_dispatch"]["unsupported_runtime_excluded_rows"] == 9
    assert run_log["live_dispatch"]["unsupported_runtime_excluded_survivors"] == 4

    activation_check = json.loads((result.round_dir / "runtime_activation_check.json").read_text(encoding="utf-8"))
    assert activation_check["status"] == "partitioned"
    assert activation_check["checked_combo_count"] == 17
    assert activation_check["supported_combo_count"] == 8
    assert activation_check["unsupported_combo_count"] == 9
    assert activation_check["supported_survivor_count"] == 4
    assert activation_check["unsupported_survivor_count"] == 4
    assert activation_check["runtime_measured_survivor_combo_ids"] == [
        "combo_001",
        "combo_003",
        "combo_005",
        "combo_007",
    ]

    audit = (result.round_dir / "unsupported_runtime_candidates.tsv").read_text(encoding="utf-8").splitlines()
    assert len(audit) == 10
    assert audit[0].split("\t") == auto_research.L0aKernelSelectRunner.RUNTIME_UNSUPPORTED_COLUMNS
    assert "combo_009" in audit[1]
    assert "deltanet_kernel" in audit[1]

    supported_action_space = auto_research.load_yaml_file(result.round_dir / "action_space.runtime_supported.yaml")
    unsupported_action_space = auto_research.load_yaml_file(result.round_dir / "action_space.runtime_unsupported.yaml")
    assert len(supported_action_space) == 8
    assert len(unsupported_action_space) == 9
    assert {item["deltanet_kernel"] for item in supported_action_space} == {"triton-chunked-delta-v2"}
    assert {item["deltanet_kernel"] for item in unsupported_action_space} == {
        "triton-state-update-fused",
        "triton-experimental-scan",
    }

    dispatched = [
        call["kernel_selection"]
        for call in calls
        if isinstance(call.get("kernel_selection"), dict) and call["kernel_selection"]
    ]
    assert len(dispatched) == 6
    assert {selection["combo_id"] for selection in dispatched} <= {
        "combo_001",
        "combo_003",
        "combo_005",
        "combo_007",
    }


def test_l0a_kernel_select_refuses_missing_parity_fixture(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="HALT_REASON: l0a_precondition_missing_fixture"):
        runner.run(
            workload_file=workload_path,
            action_space_file=action_space_path,
            baselines=5,
            screen_measurements_per_combo=2,
            rescreen_top_k=8,
            rescreen_measurements_per_candidate=4,
            parallel_instances="auto",
            round_root=repo / "output" / "auto_research",
            harness="synthetic",
        )


def test_l0b_kernel_autotune_synthetic_writes_p6_artifacts(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    runner = auto_research.L0bKernelAutotuneRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        base_measurements=2,
        autotune_budget_minutes=1,
        measurement_rescreens=2,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
        max_autotune_candidates=8,
    )

    assert result.outcome == "PASS"
    measurements = (result.round_dir / "measurements.tsv").read_text(encoding="utf-8")
    assert "l0a_baseline_remeasured" in measurements
    trailers = (result.round_dir / "candidate_trailers.tsv").read_text(encoding="utf-8")
    assert "Measurement-Role: l0a_baseline_remeasured" in trailers
    warmup_trace = json.loads((result.round_dir / "warmup_stable_trace.json").read_text(encoding="utf-8"))
    assert warmup_trace["warmup_replays"] == 5
    assert warmup_trace["stable_window_replays"] == 10
    frozen = auto_research.load_yaml_file(result.round_dir / "frozen_autotune_params.yaml")
    assert frozen["frozen_at"] is True
    assert "per_kernel_params" in frozen
    determinism_log = json.loads((result.round_dir / "determinism_log.json").read_text(encoding="utf-8"))
    parity_check = json.loads((result.round_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert determinism_log["pass"] is True
    assert parity_check["pass"] is True

    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert bundle_payload["round_provenance"]["round_type"] == "l0b_autotune"
    assert bundle_payload["round_provenance"]["ROUND_NULL_RESULT"] is False
    assert bundle_payload["objective"]["paired_baseline_objective_mean"] > 0
    assert bundle_payload["objective"]["autotune_winner_objective_mean"] > bundle_payload["objective"]["paired_baseline_objective_mean"]
    assert bundle_payload["layer_0_deltanet"]["l0b_autotune"]["frozen_at"] is True
    assert bundle_payload["layer_0_deltanet"]["l0b_autotune"]["per_kernel_params"]


def test_l0b_kernel_autotune_synthetic_records_null_result_for_unsupported_target(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    runner = auto_research.L0bKernelAutotuneRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="gatedattn",
        base_measurements=1,
        autotune_budget_minutes=1,
        measurement_rescreens=1,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
        max_autotune_candidates=4,
    )

    run_log = json.loads((result.round_dir / "run_log.json").read_text(encoding="utf-8"))
    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert result.outcome == "ROUND_NULL_RESULT"
    assert run_log["outcome"] == "ROUND_NULL_RESULT"
    assert bundle_payload["round_provenance"]["ROUND_NULL_RESULT"] is True
    assert bundle_payload["round_provenance"]["null_result_reason"] == "gatedattn_autotune_requires_triton_attention_backend"


def test_l0b_kernel_autotune_real_dispatches_l0a_base_runtime_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    calls: list[dict[str, object]] = []
    restores: list[bool] = []

    class _FakeRealMeasurementHarness:
        VERSION = "RealMeasurementHarness v0.1.0"

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            return {
                "generator": self.VERSION,
                "candidate_vllm_config": candidate_vllm_config,
                "cache_isolation": {},
                "windows": {"measurement_elapsed_s": 1.0},
                "per_request_latencies": [],
                "diagnostics": {},
                "eval_throughput": 1.25,
                "rollout_throughput": 10.0,
                "window_completed": True,
                "reasoning_content_purity": 1.0,
                "determinism_pass_rate": 1.0,
                "no_oom_events": True,
                "feasible": True,
                "feasibility_failures": [],
                "harness_health_warnings": [],
            }

        def restore_runtime(self) -> None:
            restores.append(True)

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0bKernelAutotuneRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        base_measurements=1,
        autotune_budget_minutes=1,
        measurement_rescreens=1,
        round_root=repo / "output" / "auto_research",
        harness="real",
        max_autotune_candidates=4,
    )

    assert len(calls) == 2
    assert restores == [True]
    assert calls[0]["kernel_selection"] == {
        "combo_id": "combo_001",
        "attention_backend": "vllm-default",
        "deltanet_kernel": "triton-chunked-delta-v2",
        "fp8_gemm_kernel": "cublas",
        "torch_compile_mode": "default",
        "cuda_graph_capture": "off",
    }
    assert calls[1]["kernel_selection"] == calls[0]["kernel_selection"]
    run_log = json.loads((result.round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["live_dispatch"]["autotune_params_runtime_applied"] is False
    assert run_log["artifact_counts"]["baseline_rows"] == 1
    assert run_log["artifact_counts"]["winner_rows"] == 1


def test_l0b_kernel_autotune_real_blocks_unsupported_base_runtime_knobs(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(
        repo,
        kernel_selection={
            "combo_id": "combo_999",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-state-update-fused",
            "fp8_gemm_kernel": "cublas",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
    )
    runner = auto_research.L0bKernelAutotuneRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="HALT_REASON: l0a_kernel_selection_runtime_unsupported_knobs"):
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="deltanet",
            base_measurements=1,
            autotune_budget_minutes=1,
            measurement_rescreens=1,
            round_root=repo / "output" / "auto_research",
            harness="real",
            max_autotune_candidates=4,
        )

    round_dir = next((repo / "output" / "auto_research").glob("*-l0b-autotune-deltanet-*"))
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    activation_check = json.loads((round_dir / "runtime_activation_check.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "ROUND_BLOCKED"
    assert run_log["live_dispatch"]["attempted"] is False
    assert activation_check["status"] == "blocked"
    assert activation_check["unsupported_runtime_activation"][0]["axis"] == "deltanet_kernel"


def test_l0b_kernel_autotune_real_writes_halt_artifact_on_live_harness_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)

    class _FailingRealMeasurementHarness:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("live service unavailable")

        def restore_runtime(self) -> None:
            raise RuntimeError("restore unavailable")

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FailingRealMeasurementHarness)
    runner = auto_research.L0bKernelAutotuneRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="HALT_REASON: l0b_real_harness_blocked"):
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="deltanet",
            base_measurements=1,
            autotune_budget_minutes=1,
            measurement_rescreens=1,
            round_root=repo / "output" / "auto_research",
            harness="real",
            max_autotune_candidates=4,
        )

    round_dir = next((repo / "output" / "auto_research").glob("*-l0b-autotune-deltanet-*"))
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    trace = json.loads((round_dir / "measurement_trace_combined.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "ROUND_BLOCKED"
    assert run_log["HALT_REASON"] == "l0b_real_harness_blocked"
    assert "live service unavailable" in run_log["measurement_error"]
    assert "restore unavailable" in run_log["restore_error"]
    assert trace["HALT_REASON"] == "l0b_real_harness_blocked"

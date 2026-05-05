from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
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


def _write_l0a_fp8_fixture(
    repo: Path,
    source_family: str = "responses-sdk-adapter-cutover",
    *,
    omit_companion_key: str | None = None,
    reference_fp8_gemm_kernel: str = "cublas",
) -> Path:
    fixture_dir = repo / "benchmark_blueprints" / "families" / source_family / "parity_fixture"
    tier_3_dir = fixture_dir / "tier_3_inputs"
    tier_4_dir = fixture_dir / "tier_4_vllm_parity"
    tier_3_dir.mkdir(parents=True, exist_ok=True)
    tier_4_dir.mkdir(parents=True, exist_ok=True)
    refs = {
        "tier_3_probe_input_a_ref": "tier_3_inputs/gemm_input_a.npz",
        "tier_3_probe_input_b_ref": "tier_3_inputs/gemm_input_b.npz",
        "tier_3_probe_input_scale_a_ref": "tier_3_inputs/gemm_input_scale_a.npz",
        "tier_3_probe_input_scale_b_ref": "tier_3_inputs/gemm_input_scale_b.npz",
        "tier_3_reference_gemm_output_ref": "tier_3_inputs/gemm_reference_output.npz",
        "tier_4_probe_input_state_ref": "tier_4_vllm_parity/probe_state_snapshots.npz",
        "tier_4_reference_downstream_logits_ref": "tier_4_vllm_parity/reference_downstream_logits.npz",
    }
    for key, ref in refs.items():
        if key == omit_companion_key:
            continue
        with zipfile.ZipFile(fixture_dir / ref, mode="w") as archive:
            archive.writestr(f"{key}.npy", b"dummy")
    payload = {
        "fixture_id": f"{source_family}-fp8-gemm-v1",
        "kernel_target": "fp8_gemm",
        "generated_at": "2026-05-02T00:00:00Z",
        "generated_against": {
            "weight_version_id": "2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
            "reference_baseline": {
                "attention_backend": "vllm-default",
                "deltanet_kernel": "triton-chunked-delta-v2",
                "fp8_gemm_kernel": reference_fp8_gemm_kernel,
                "torch_compile_mode": "default",
                "cuda_graph_capture": "off",
            },
            "reference_reproducibility_runs": 3,
        },
        "tier_3_probe_count": 2,
        "tier_3_smoke_probe_count": 1,
        "tier_3_probe_shapes": [{"M": 1, "N": 8, "K": 4}, {"M": 16, "N": 8, "K": 4}],
        "tier_3_tolerances": {"rtol_gemm_output": 2.0e-3, "atol_gemm_output": 2.0e-3},
        "tier_3_parity_check_method": "gemm_output_compare_only",
        "tier_4_call_site_count": 2,
        "tier_4_call_site_layer_indices": [0, 4],
        "tier_4_tolerances": {"rtol_downstream_logit": 1.0e-3, "atol_downstream_logit": 1.0e-3},
        "tier_4_parity_check_method": "vllm_parity_with_downstream_logit_compounding_guard",
        **refs,
    }
    path = fixture_dir / "fp8_gemm_v1.yaml"
    path.write_text(auto_research.yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    if omit_companion_key is None:
        payload["content_hash"] = auto_research.fixture_content_hash(path)
        path.write_text(auto_research.yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _add_l0a_workload_fp8_ref(workload_path: Path) -> None:
    payload = auto_research.load_yaml_file(workload_path)
    assert isinstance(payload, dict)
    payload["parity_fixture_refs"]["fp8_gemm"] = "parity_fixture/fp8_gemm_v1.yaml"
    workload_path.write_text(auto_research.yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    payload["workload_distribution_id"] = auto_research.compute_workload_distribution_id(workload_path)
    workload_path.write_text(auto_research.yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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


def test_real_measurement_harness_records_metrics_consumption(
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
    counters = {
        "prompt": 0.0,
        "gen": 0.0,
        "kv": 0.0,
        "ttft_sum": 0.0,
        "ttft_count": 0.0,
        "prefill": 0.0,
        "decode": 0.0,
        "itl": 0.0,
        "queries": 0.0,
        "hits": 0.0,
    }

    def prom() -> str:
        return "\n".join(
            [
                f"vllm:prompt_tokens_total {counters['prompt']}",
                f"vllm:generation_tokens_total {counters['gen']}",
                f"vllm:request_prefill_kv_computed_tokens_sum {counters['kv']}",
                f"vllm:time_to_first_token_seconds_sum {counters['ttft_sum']}",
                f"vllm:time_to_first_token_seconds_count {counters['ttft_count']}",
                f"vllm:request_prefill_time_seconds_sum {counters['prefill']}",
                f"vllm:request_decode_time_seconds_sum {counters['decode']}",
                f"vllm:inter_token_latency_seconds_sum {counters['itl']}",
                f"vllm:prefix_cache_queries_total {counters['queries']}",
                f"vllm:prefix_cache_hits_total {counters['hits']}",
            ]
        )

    def fake_post(url: str, **kwargs):
        payload = kwargs.get("json") or {}
        if url.endswith("/responses"):
            prompt_tokens = len(str(payload.get("input", "")).split())
            gen_tokens = int(payload.get("max_output_tokens") or 1)
            counters["prompt"] += prompt_tokens
            counters["gen"] += gen_tokens
            counters["kv"] += prompt_tokens
            counters["ttft_sum"] += 0.2
            counters["ttft_count"] += 1
            counters["prefill"] += 0.01
            counters["decode"] += 0.04
            counters["itl"] += 0.04
            counters["queries"] += 1
            counters["hits"] += 0.5
        return _HTTPResponse()

    def fake_get(url: str, **kwargs):
        del url, kwargs
        return _HTTPResponse(text=prom())

    monkeypatch.setattr(measurement_harness.requests, "post", fake_post)
    monkeypatch.setattr(measurement_harness.requests, "get", fake_get)

    trace = harness.measure({}, warmup_s=0, window_s=0, target_concurrency=1)

    consumption = trace["metrics_consumption"]
    assert consumption["available"] is True
    assert consumption["step_consumption"]["decode_ms_per_generated_token"] is not None
    assert consumption["gb10_reference"]["theoretical_bandwidth_gb_s"] == 273.0


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
        'model="gpt-5.5"',
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
    assert timeout == 0


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
    assert timeout == 0


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
    round_spec = auto_research.load_yaml_file(result.round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert set(round_spec["parity_fixture_refs"]) == {"deltanet", "gatedattn"}
    assert set(round_spec["parity_fixture_content_hashes"]) == {"deltanet", "gatedattn"}

    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert bundle_payload["round_provenance"]["round_type"] == "l0a_select_only"
    assert bundle_payload["round_provenance"]["parallel_instances"] == 1
    assert bundle_payload["kernel_selection"]["attention_backend"] == "flash-attn-4"
    assert bundle_payload["kernel_selection"]["deltanet_kernel"] == "triton-chunked-delta-v2"

    with pytest.raises(StructuredValidationError, match="bundle-validity: refused"):
        validate_bundle_load_policy(load_tuned_config_bundle(result.bundle_path), bundle_confidence_policy="passthrough")


def test_l0a_kernel_select_phase_a_cli_prereq_metadata_and_round_prefix(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = repo / "kernel_search" / "phase_a_action_space.yaml"
    action_space_path.parent.mkdir(parents=True, exist_ok=True)
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2]
  fp8_gemm_kernel: [cublas]
  torch_compile_mode: [default]
  cuda_graph_capture: ['off']
""",
        encoding="utf-8",
    )
    base_bundle_path = _write_l0a_bundle(repo)
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
        harness="synthetic",
        base_stack_resolution="bundle",
        base_bundle_path=base_bundle_path,
        round_prefix="qwen3.5-27b-fp8-gemm-phase-a",
        phase_a_screen_method="replay",
    )

    assert result.round_id.startswith("qwen3.5-27b-fp8-gemm-phase-a-")
    assert result.round_dir.name == result.round_id
    round_spec = auto_research.load_yaml_file(result.round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["base_stack_resolution"] == "bundle"
    assert round_spec["base_bundle_path"] == str(base_bundle_path.resolve())
    assert round_spec["phase_a_screen_method"] == "replay"
    assert round_spec["phase_a_screen_method_effective"] == "synthetic_fixture"
    assert "metadata only" in round_spec["phase_a_screen_method_note"]
    assert round_spec["phase_a_backend_identities_ref"] == "phase_a_backend_identities.json"
    identities = round_spec["phase_a_backend_identities"]
    assert set(identities) == {"cublas"}
    backend_identity = identities["cublas"]
    assert backend_identity["repo_dispatch_hook_symbol"].endswith("._apply_fp8_gemm_kernel")
    assert backend_identity["repo_dispatch_hook_source_path"] == "src/lumo_flywheel_serving/kernel_activation.py"
    assert backend_identity["resolved_runtime_name"] == "torch_scaled_mm"
    assert backend_identity["support_status"] == "supported"
    assert len(backend_identity["content_hash"]) == 64

    identity_manifest = json.loads(
        (result.round_dir / "phase_a_backend_identities.json").read_text(encoding="utf-8")
    )
    assert identity_manifest["schema"] == "phase_a_backend_identity_manifest.v1"
    assert identity_manifest["identities"] == identities
    assert result.artifact_paths["phase_a_backend_identities"].endswith(
        "phase_a_backend_identities.json"
    )


def test_l0a_kernel_select_real_replay_screen_halts_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    harness_inits: list[dict[str, object]] = []

    class _FakeRealMeasurementHarness:
        def __init__(self, **kwargs: object) -> None:
            harness_inits.append(kwargs)

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="HALT_REASON: phase_a_replay_harness_not_implemented"):
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
            phase_a_screen_method="replay",
        )

    round_dir = next((repo / "output" / "auto_research").glob("*-l0a-select-*"))
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "ROUND_BLOCKED"
    assert run_log["HALT_REASON"] == "phase_a_replay_harness_not_implemented"
    assert run_log["phase_a_screen_method"] == "replay"
    assert run_log["phase_a_screen_method_effective"] == "not_implemented"
    assert run_log["smoke_attempted"] is False
    assert run_log["live_dispatch"]["attempted"] is False
    round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert round_spec["phase_a_screen_method"] == "replay"
    assert round_spec["phase_a_screen_method_effective"] == "not_implemented"
    measurement_trace = json.loads((round_dir / "measurement_trace_combined.json").read_text(encoding="utf-8"))
    assert measurement_trace["HALT_REASON"] == "phase_a_replay_harness_not_implemented"
    assert measurement_trace["measurements_attempted"] is False
    assert not (round_dir / "smoke_trace.json").exists()
    assert not (round_dir / "measurements.tsv").exists()
    assert not (round_dir / "runtime_activation_check.json").exists()
    assert not (round_dir / "live_traces").exists()
    assert harness_inits == []


def test_l0a_kernel_select_records_optional_fp8_fixture_ref_and_hash(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    fp8_fixture_path = _write_l0a_fp8_fixture(repo)
    workload_path = _write_l0a_workload(repo)
    _add_l0a_workload_fp8_ref(workload_path)
    action_space_path = repo / "kernel_search" / "phase_a_action_space.yaml"
    action_space_path.parent.mkdir(parents=True, exist_ok=True)
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2]
  fp8_gemm_kernel: [cublas]
  torch_compile_mode: [default]
  cuda_graph_capture: ['off']
""",
        encoding="utf-8",
    )
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
        harness="synthetic",
    )

    expected_hash = auto_research.fixture_content_hash(fp8_fixture_path)
    expected_ref = "benchmark_blueprints/families/responses-sdk-adapter-cutover/parity_fixture/fp8_gemm_v1.yaml"
    round_spec = auto_research.load_yaml_file(result.round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["parity_fixture_refs"]["fp8_gemm"] == expected_ref
    assert round_spec["parity_fixture_content_hashes"]["fp8_gemm"] == expected_hash
    assert set(round_spec["parity_fixture_refs"]) == {"deltanet", "gatedattn", "fp8_gemm"}

    parity_check = json.loads((result.round_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert parity_check["fixture_refs"]["fp8_gemm"] == {
        "path": expected_ref,
        "content_hash": expected_hash,
    }
    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert bundle_payload["round_provenance"]["parity_fixture_refs"]["fp8_gemm"] == expected_ref
    assert bundle_payload["round_provenance"]["parity_fixture_content_hashes"]["fp8_gemm"] == expected_hash


def test_l0a_kernel_select_fp8_fixture_missing_companion_artifact_fails_clearly(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    _write_l0a_fp8_fixture(repo, omit_companion_key="tier_4_reference_downstream_logits_ref")
    workload_path = _write_l0a_workload(repo)
    _add_l0a_workload_fp8_ref(workload_path)
    action_space_path = repo / "kernel_search" / "phase_a_action_space.yaml"
    action_space_path.parent.mkdir(parents=True, exist_ok=True)
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2]
  fp8_gemm_kernel: [cublas]
  torch_compile_mode: [default]
  cuda_graph_capture: ['off']
""",
        encoding="utf-8",
    )
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "HALT_REASON: l0a_precondition_missing_fixture; invalid parity fixture\\(s\\): "
            "fp8_gemm:.*tier_4_reference_downstream_logits_ref"
        ),
    ):
        runner.run(
            workload_file=workload_path,
            action_space_file=action_space_path,
            baselines=1,
            screen_measurements_per_combo=1,
            rescreen_top_k=1,
            rescreen_measurements_per_candidate=1,
            parallel_instances="auto",
            round_root=repo / "output" / "auto_research",
            harness="synthetic",
        )


def test_l0a_kernel_select_bundle_resolution_requires_bundle_path(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="--base-bundle-path is required"):
        runner.run(
            workload_file=workload_path,
            action_space_file=action_space_path,
            baselines=1,
            screen_measurements_per_combo=1,
            rescreen_top_k=1,
            rescreen_measurements_per_candidate=1,
            parallel_instances="auto",
            round_root=repo / "output" / "auto_research",
            harness="synthetic",
            base_stack_resolution="bundle",
        )


def test_l0a_kernel_select_real_bundle_resolution_uses_base_bundle_stack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = repo / "kernel_search" / "phase_a_action_space.yaml"
    action_space_path.parent.mkdir(parents=True, exist_ok=True)
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2]
  fp8_gemm_kernel: [cutlass]
  torch_compile_mode: [default]
  cuda_graph_capture: ['off']
""",
        encoding="utf-8",
    )
    base_bundle = _write_l0a_bundle(
        repo,
        kernel_selection={
            "combo_id": "combo_base",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "cublas",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
    )
    payload = auto_research.load_yaml_file(base_bundle)
    payload["tuned_config_bundle"]["vllm_config"]["max_num_batched_tokens"] = 4096
    base_bundle.write_text(auto_research.yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

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
        base_stack_resolution="bundle",
        base_bundle_path=base_bundle,
        proxy_port=8101,
        phase_a_screen_method="full_vllm",
    )

    assert len(calls) == 3
    assert all(call["candidate_vllm_config"]["max_num_batched_tokens"] == 4096 for call in calls)
    assert calls[0]["kernel_selection"]["fp8_gemm_kernel"] == "cublas"
    assert calls[1]["kernel_selection"]["fp8_gemm_kernel"] == "cutlass"
    round_spec = auto_research.load_yaml_file(result.round_dir / "round_spec.yaml")
    assert round_spec["base_stack_resolution"] == "bundle"
    assert round_spec["base_bundle_id"] == payload["tuned_config_bundle"]["bundle_id"]
    assert round_spec["baseline_vllm_config_source"] == "base_bundle"
    output_bundle = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert output_bundle["vllm_config"]["max_num_batched_tokens"] == 4096
    assert output_bundle["baseline_bundle_id"] == payload["tuned_config_bundle"]["bundle_id"]
    assert output_bundle["round_provenance"]["base_stack_resolution"] == "bundle"
    assert output_bundle["round_provenance"]["base_bundle_id"] == payload["tuned_config_bundle"]["bundle_id"]
    assert output_bundle["round_provenance"]["phase_a_screen_method"] == "full_vllm"
    assert output_bundle["round_provenance"]["phase_a_screen_method_effective"] == "full_vllm"


def test_l0a_kernel_select_real_replay_blocks_before_live_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = _write_l0a_action_space(repo / "kernel_search" / "l0a_action_space.yaml")
    harness_inits: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []

    class _FakeRealMeasurementHarness:
        def __init__(self, **kwargs: object) -> None:
            harness_inits.append(kwargs)

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            return {}

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _FakeRealMeasurementHarness)
    runner = auto_research.L0aKernelSelectRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(
        RuntimeError,
        match="HALT_REASON: phase_a_replay_harness_not_implemented",
    ):
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
            max_combos=1,
            proxy_port=8101,
            phase_a_screen_method="replay",
        )

    round_dir = next((repo / "output" / "auto_research").glob("*-l0a-select-*"))
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    trace = json.loads((round_dir / "measurement_trace_combined.json").read_text(encoding="utf-8"))
    round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")

    assert run_log["outcome"] == "ROUND_BLOCKED"
    assert run_log["round_status"] == "blocked"
    assert run_log["HALT_REASON"] == "phase_a_replay_harness_not_implemented"
    assert run_log["halt_reason"] == "phase_a_replay_harness_not_implemented"
    assert run_log["phase_a_screen_method"] == "replay"
    assert run_log["phase_a_screen_method_requested"] == "replay"
    assert run_log["phase_a_screen_method_effective"] == "not_implemented"
    assert run_log["live_dispatch"]["attempted"] is False
    assert run_log["smoke_attempted"] is False
    assert trace["outcome"] == "ROUND_BLOCKED"
    assert trace["round_status"] == "blocked"
    assert trace["HALT_REASON"] == "phase_a_replay_harness_not_implemented"
    assert trace["phase_a_screen_method_requested"] == "replay"
    assert trace["phase_a_screen_method_effective"] == "not_implemented"
    assert trace["baselines"] == []
    assert trace["screen"] == []
    assert trace["rescreen"] == []
    assert trace["live_dispatch"]["attempted"] is False
    assert round_spec["phase_a_screen_method"] == "replay"
    assert round_spec["phase_a_screen_method_requested"] == "replay"
    assert round_spec["phase_a_screen_method_effective"] == "not_implemented"
    assert harness_inits == []
    assert calls == []


def test_l0a_kernel_select_real_full_vllm_dispatches_live_smoke_with_runtime_activation(
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
        phase_a_screen_method="full_vllm",
    )

    round_dir = result.round_dir
    run_log = json.loads((round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["outcome"] == "PASS"
    assert run_log["kernel_selection_runtime_activation"] == "runtime_applied"
    assert run_log["phase_a_screen_method"] == "full_vllm"
    assert run_log["phase_a_screen_method_effective"] == "full_vllm"
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
        phase_a_screen_method="full_vllm",
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
            phase_a_screen_method="full_vllm",
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


def test_l0a_kernel_select_real_records_unsupported_fp8_backend_identity(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    action_space_path = repo / "kernel_search" / "phase_a_action_space.yaml"
    action_space_path.parent.mkdir(parents=True, exist_ok=True)
    action_space_path.write_text(
        """
axes:
  attention_backend: [vllm-default]
  deltanet_kernel: [triton-chunked-delta-v2]
  fp8_gemm_kernel: [cublas, triton_fp8_scaled_mm]
  torch_compile_mode: [default]
  cuda_graph_capture: ['off']
""",
        encoding="utf-8",
    )
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
            proxy_port=8101,
            runtime_unsupported_policy="strict",
            phase_a_screen_method="full_vllm",
        )

    round_dir = next((repo / "output" / "auto_research").glob("*-l0a-select-*"))
    activation_check = json.loads((round_dir / "runtime_activation_check.json").read_text(encoding="utf-8"))
    unsupported = activation_check["unsupported_runtime_activation"]
    assert len(unsupported) == 1
    assert unsupported[0]["kernel_selection"]["fp8_gemm_kernel"] == "triton_fp8_scaled_mm"
    assert unsupported[0]["unsupported_knobs"][0]["axis"] == "fp8_gemm_kernel"
    identity = unsupported[0]["activation_plan"]["resolved"]["fp8_gemm_backend_identity"]
    assert identity["support_status"] == "unsupported"
    assert identity["supported"] is False
    assert identity["resolved_runtime_name"] is None

    round_spec = auto_research.load_yaml_file(round_dir / "round_spec.yaml")
    assert round_spec["phase_a_backend_identities"]["cublas"]["support_status"] == "supported"
    assert round_spec["phase_a_backend_identities"]["triton_fp8_scaled_mm"]["support_status"] == "unsupported"


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
        phase_a_screen_method="full_vllm",
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
    assert activation_check["supported_runtime_activation"][0]["activation_plan"]["resolved"][
        "fp8_gemm_backend_identity"
    ]["support_status"] == "supported"

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
    """Real harness drives autotune-phase replays + baseline + winner, captures Triton cache."""
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
            self.triton_cache_root = Path(kwargs["triton_cache_root"])  # round-local
            self._call_index = 0

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            self._call_index += 1
            calls.append({"candidate_vllm_config": candidate_vllm_config, **kwargs})
            # Emulate Triton's first-call autotune: write 3 cache entries on the
            # very first measure(), zero on every subsequent call. The stable
            # window check must therefore trip on the second autotune-phase replay.
            if self._call_index == 1:
                for shape_id in range(3):
                    cache_path = self.triton_cache_root / f"autotune_shape_{shape_id}" / "winner.json"
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps({"BV": 64, "num_warps": 4, "num_stages": 3}),
                        encoding="utf-8",
                    )
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
        warmup_replays=2,
        stable_window_replays=2,
    )

    # 2 warmup + 2 stable_window + 1 baseline + 1 winner = 6 measure() calls.
    assert len(calls) == 6
    assert restores == [True]
    assert calls[0]["kernel_selection"]["deltanet_kernel"] == "triton-chunked-delta-v2"
    # Every replay invokes the L0a kernel selection (no per-call kernel switching).
    assert all(call["kernel_selection"] == calls[0]["kernel_selection"] for call in calls)

    warmup_trace = json.loads((result.round_dir / "warmup_stable_trace.json").read_text(encoding="utf-8"))
    assert warmup_trace["stabilized"] is True
    # First replay produced 3 cache entries; subsequent replays produced none.
    assert warmup_trace["events"][0]["phase"] == "warmup"
    final_event = warmup_trace["events"][-1]
    assert final_event["phase"] == "stable_window"
    assert final_event["new_winners"] == []

    frozen = auto_research.load_yaml_file(result.round_dir / "frozen_autotune_params.yaml")
    assert frozen["frozen_at"] is True
    assert frozen["stabilized"] is True
    assert frozen["per_kernel_params"]["captured_from"] == "upstream_triton_autotune"
    assert frozen["per_kernel_params"]["cache_file_count"] == 3
    archive = result.round_dir / frozen["frozen_triton_cache_ref"]
    assert archive.is_file()
    assert frozen["frozen_triton_cache_sha256"]

    run_log = json.loads((result.round_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["live_dispatch"]["autotune_params_runtime_applied"] is True
    assert run_log["live_dispatch"]["autotune_phase_replays"] == 4
    assert run_log["live_dispatch"]["stabilized"] is True
    assert run_log["artifact_counts"]["baseline_rows"] == 1
    assert run_log["artifact_counts"]["winner_rows"] == 1


def test_l0b_kernel_autotune_real_records_budget_exhausted_when_cache_keeps_growing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Triton keeps autotuning new shapes, autotune phase records budget_exhausted."""
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)

    class _NeverStableHarness:
        VERSION = "RealMeasurementHarness v0.1.0"

        def __init__(self, **kwargs: object) -> None:
            self.triton_cache_root = Path(kwargs["triton_cache_root"])
            self._idx = 0

        def measure(self, candidate_vllm_config: dict, **kwargs: object) -> dict[str, object]:
            self._idx += 1
            cache_path = self.triton_cache_root / f"shape_{self._idx}" / "winner.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text("{}", encoding="utf-8")
            return {
                "generator": self.VERSION,
                "candidate_vllm_config": candidate_vllm_config,
                "cache_isolation": {},
                "windows": {"measurement_elapsed_s": 1.0},
                "per_request_latencies": [],
                "diagnostics": {},
                "eval_throughput": 1.0,
                "rollout_throughput": 1.0,
                "window_completed": True,
                "reasoning_content_purity": 1.0,
                "determinism_pass_rate": 1.0,
                "no_oom_events": True,
                "feasible": True,
                "feasibility_failures": [],
                "harness_health_warnings": [],
            }

        def restore_runtime(self) -> None:
            pass

    monkeypatch.setattr(auto_research, "RealMeasurementHarness", _NeverStableHarness)
    # The runner's hard replay cap (max_extra_replays = stable_window * 4) is
    # the safety net we exercise here: every replay introduces a new cache
    # file, so consecutive_stable can never reach stable_window_replays.

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
        max_autotune_candidates=2,
        warmup_replays=2,
        stable_window_replays=2,
    )

    warmup_trace = json.loads((result.round_dir / "warmup_stable_trace.json").read_text(encoding="utf-8"))
    assert warmup_trace["stabilized"] is False
    assert warmup_trace["stable_window_condition"] == "budget_exhausted_before_stable"
    # Hard replay cap: stable_window_replays * 4 = 8 extra replays.
    extra_events = [event for event in warmup_trace["events"] if event["phase"] == "stable_window"]
    assert len(extra_events) == 8
    frozen = auto_research.load_yaml_file(result.round_dir / "frozen_autotune_params.yaml")
    assert frozen["freeze_reason"] == "budget_exhausted_before_stable"
    assert frozen["stabilized"] is False


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


def test_l0c_kernel_mutation_synthetic_writes_p5_artifacts_and_passes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    fixture_path = (
        repo
        / "benchmark_blueprints"
        / "families"
        / "responses-sdk-adapter-cutover"
        / "parity_fixture"
        / "deltanet_v1.yaml"
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path="kernels/deltanet/chunked_delta.py",
        parity_fixture=fixture_path,
        base_measurements=2,
        accepted_iteration_cap=4,
        total_attempt_cap=12,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
    )

    assert result.outcome == "ROUND_PASSED"
    assert result.terminal_condition == "accepted_cap_reached"
    assert result.accepted_count == 4
    # Default outcome rejects every 3rd attempt; the cap check is evaluated BEFORE
    # the next attempt runs, so reaching 4 accepted finishes after attempt 005
    # (001/002/004/005 pass; 003 fails) — the runner does not spawn 006.
    assert result.total_attempt_count == 5
    assert result.rejected_count == 1

    measurements = (result.round_dir / "measurements.tsv").read_text(encoding="utf-8")
    assert "l0b_baseline_remeasured" in measurements
    assert "l0c_candidate" in measurements
    trailers = (result.round_dir / "candidate_trailers.tsv").read_text(encoding="utf-8")
    assert "Measurement-Role: l0b_baseline_remeasured" in trailers
    assert "Mutation-Hash:" in trailers

    rejected = (result.round_dir / "mutations_rejected.tsv").read_text(encoding="utf-8").splitlines()
    assert rejected[0].split("\t")[0] == "iteration"
    rejected_iterations = [line.split("\t")[0] for line in rejected[1:]]
    assert rejected_iterations == ["003"]

    iteration_labels = sorted(p.name for p in (result.round_dir / "candidates").iterdir())
    assert iteration_labels == ["001", "002", "003", "004", "005"]
    # Per AR.43: parity_check.json present every iteration; measurement_trace.json
    # only when parity passed (no faster-but-wrong leakage).
    for label in iteration_labels:
        cand = result.round_dir / "candidates" / label
        parity = json.loads((cand / "parity_check.json").read_text(encoding="utf-8"))
        if label == "003":
            assert parity["pass"] is False
            assert parity["reason"] == "parity_logit_diverged"
            assert not (cand / "measurement_trace.json").exists()
        else:
            assert parity["pass"] is True
            assert parity["reason"] == "ran_passed"
            assert parity["checkpoints_checked"] == [1, 1024]
            assert (cand / "measurement_trace.json").exists()
        assert (cand / "mutation.patch").exists()

    assert result.bundle_path is not None
    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert bundle_payload["round_provenance"]["round_type"] == "l0c_mutation"
    assert bundle_payload["round_provenance"]["terminal_condition"] == "accepted_cap_reached"
    l0c_block = bundle_payload["layer_0_deltanet"]["l0c_mutation"]
    assert l0c_block["accepted_count"] == 4
    assert l0c_block["total_attempt_count"] == 5
    assert l0c_block["rejected_count"] == 1
    assert l0c_block["parity_attestation"]["checkpoints_checked"] == [1, 1024]
    assert bundle_payload["layer_0_fp8_gemm"] == {}
    # AR.48b counter audit: accepted + rejected == total.
    assert (
        l0c_block["accepted_count"] + l0c_block["rejected_count"]
        == l0c_block["total_attempt_count"]
    )


def test_l0c_kernel_mutation_synthetic_fp8_gemm_bootstraps_and_populates_bundle(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    fp8_fixture_path = _write_l0a_fp8_fixture(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    p3a_dir = repo / "output" / "p3a_roofline_probe_20260429T193758Z"
    p3a_dir.mkdir(parents=True)
    (p3a_dir / "p3a_roofline_probe.json").write_text(
        json.dumps(
            {
                "probe_count": 1,
                "wall_clock_s": 1.0,
                "derived": {},
                "gpu_poll_stats": {},
                "p3a_decision": {
                    "basis": (
                        "HLD prior plus live decode-dominant single probe; no counter "
                        "evidence contradicts DeltaNet-first canary ordering."
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="fp8_gemm",
        kernel_source_path="kernels/fp8_gemm/triton_scaled_mm.py",
        parity_fixture=fp8_fixture_path,
        base_measurements=2,
        accepted_iteration_cap=2,
        total_attempt_cap=4,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
    )

    assert result.outcome == "ROUND_PASSED"
    assert result.kernel_target == "fp8_gemm"
    round_spec = auto_research.load_yaml_file(result.round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["kernel_target"] == "fp8_gemm"
    assert round_spec["parity_fixture_id"] == "responses-sdk-adapter-cutover-fp8-gemm-v1"
    assert round_spec["parity_fixture_content_hash"] == auto_research.fixture_content_hash(
        fp8_fixture_path
    )

    parity = json.loads(
        (result.round_dir / "candidates" / "001" / "parity_check.json").read_text(
            encoding="utf-8"
        )
    )
    assert parity["pass"] is True
    assert parity["kernel_target"] == "fp8_gemm"
    assert parity["fixture_id"] == "responses-sdk-adapter-cutover-fp8-gemm-v1"
    assert parity["checkpoints_checked"] == []
    assert parity["tiers_checked"] == [
        "tier_3_gemm_output_compare",
        "tier_4_downstream_logit_guard",
    ]

    brief = (result.round_dir / "iteration_brief.md").read_text(encoding="utf-8")
    assert "Tier 3 GEMM-output tolerance" in brief
    assert "g`/`gk`" not in brief
    strategy = (result.round_dir / "strategy_brief.md").read_text(encoding="utf-8")
    assert "DeltaNet-first ordering" not in strategy
    assert "Triton FP8 GEMM call boundary" in strategy
    assert "FFN GEMM pivot brief" in strategy

    assert result.bundle_path is not None
    bundle_payload = auto_research.load_yaml_file(result.bundle_path)["tuned_config_bundle"]
    assert bundle_payload["layer_0_deltanet"] == {}
    assert bundle_payload["layer_0_gatedattn"] == {}
    fp8_l0c = bundle_payload["layer_0_fp8_gemm"]["l0c_mutation"]
    assert fp8_l0c["accepted_count"] == 2
    assert fp8_l0c["parity_attestation"]["fixture_content_hash"] == round_spec[
        "parity_fixture_content_hash"
    ]
    assert fp8_l0c["parity_attestation"]["tiers_checked"] == [
        "tier_3_gemm_output_compare",
        "tier_4_downstream_logit_guard",
    ]


def test_l0c_kernel_mutation_fp8_gemm_validates_fixture_schema(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    fp8_fixture_path = _write_l0a_fp8_fixture(
        repo,
        omit_companion_key="tier_4_reference_downstream_logits_ref",
    )
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "HALT_REASON: l0c_precondition_missing_fixture; invalid parity fixture: "
            "fp8_gemm:.*tier_4_reference_downstream_logits_ref"
        ),
    ):
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="fp8_gemm",
            kernel_source_path="kernels/fp8_gemm/triton_scaled_mm.py",
            parity_fixture=fp8_fixture_path,
            base_measurements=1,
            accepted_iteration_cap=1,
            total_attempt_cap=1,
            round_timeout_hours=1.0,
            round_root=repo / "output" / "auto_research",
            harness="synthetic",
        )


def test_l0c_fp8_gemm_real_cutlass_missing_fixture_halts_after_round_metadata(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(
        repo,
        kernel_selection={
            "combo_id": "combo_002",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "cutlass",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(
        RuntimeError,
        match="HALT_REASON: l0c_precondition_missing_fixture; missing fp8_gemm parity fixture",
    ):
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="fp8_gemm",
            kernel_source_path=None,
            parity_fixture=None,
            base_measurements=1,
            accepted_iteration_cap=1,
            total_attempt_cap=1,
            round_timeout_hours=1.0,
            round_root=repo / "output" / "auto_research",
            harness="real",
            runtime={
                "container_name": "test",
                "port": 8100,
                "proxy_port": 8101,
                "endpoint": "http://127.0.0.1:8101/v1",
                "metrics_url": "http://127.0.0.1:8100/metrics",
                "admin_url": "http://127.0.0.1:8101/admin",
            },
        )

    rounds = list((repo / "output" / "auto_research").iterdir())
    assert len(rounds) == 1
    run_log = json.loads((rounds[0] / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["HALT_REASON"] == "l0c_precondition_missing_fixture"
    round_spec = auto_research.load_yaml_file(rounds[0] / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["kernel_target"] == "fp8_gemm"
    assert round_spec["mutation_surface"]["kind"] == "cutlass_source_workspace"
    assert round_spec["mutation_surface"]["runtime_wired"] is True


def test_l0c_fp8_gemm_real_cutlass_bootstrap_reaches_candidate_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    fp8_fixture_path = _write_l0a_fp8_fixture(repo, reference_fp8_gemm_kernel="cutlass")
    workload_path = _write_l0a_workload(repo)
    _add_l0a_workload_fp8_ref(workload_path)
    base_bundle = _write_l0a_bundle(
        repo,
        kernel_selection={
            "combo_id": "combo_002",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "cutlass",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
    )
    timing_dir = repo / "output" / "p3a_agent_flow_roofline_20260501T192001Z"
    timing_dir.mkdir(parents=True)
    (timing_dir / "p3a_agent_flow_roofline_full10_summary.json").write_text(
        json.dumps(
            {
                "aggregate_timing": {
                    "categories": [
                        {
                            "category": "gatedattn_attention_with_kv_read",
                            "leaf_share": 0.67,
                            "self_time_ms": 100.0,
                            "ms_per_requested_output_token": 10.0,
                        },
                        {
                            "category": "ffn_linear",
                            "leaf_share": 0.20,
                            "self_time_ms": 30.0,
                            "ms_per_requested_output_token": 3.0,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def _no_drift(*args, **kwargs):
        return None

    def _baseline(*, spec, baseline_dir, baseline_uuid, count):
        return [
            runner._make_measurement_row(
                candidate_uuid=baseline_uuid,
                candidate_label="l0b-empirical-winner-baseline-remeasured",
                role="l0b_empirical_winner_baseline_remeasured",
                measurement_index=1,
                objective_value=1.0,
                harness="real",
                trace_ref="baselines/measurement_01.json",
            )
        ]

    def _agent_unavailable(*, spec, round_dir, iteration_dir, iteration):
        return {"ok": False, "error": "agent_binary_missing: codex"}

    def _stage_workspace(*, round_dir, spec):
        surface = dict(spec["mutation_surface"])
        workspace = round_dir / "cutlass_source_workspace"
        source = workspace / "vllm-source"
        python_source = source / "vllm" / "model_executor" / "kernels" / "linear" / "scaled_mm"
        cxx_source = source / "csrc" / "quantization" / "w8a8" / "cutlass"
        python_source.mkdir(parents=True)
        cxx_source.mkdir(parents=True)
        (python_source / "cutlass.py").write_text("def scaled_mm():\n    return None\n", encoding="utf-8")
        (cxx_source / "scaled_mm_entry.cu").write_text("// entry\n", encoding="utf-8")
        return {
            **surface,
            "workspace_path": str(workspace),
            "workspace_source_path": str(source),
            "workspace_python_source_path": str(python_source),
            "workspace_cxx_source_path": str(cxx_source),
            "workspace_base_path": str(round_dir / "cutlass_source_base"),
            "workspace_base_source_path": str(round_dir / "cutlass_source_base" / "vllm-source"),
            "container_vllm_source_dir": "/opt/vllm-source",
            "round_dir": str(round_dir),
        }

    monkeypatch.setattr(runner, "_assert_actually_resolved_no_drift", _no_drift)
    monkeypatch.setattr(runner, "_run_real_paired_baseline", _baseline)
    monkeypatch.setattr(runner, "_spawn_l0c_agent_iteration", _agent_unavailable)
    monkeypatch.setattr(runner, "_stage_fp8_cutlass_source_workspace", _stage_workspace)

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="fp8_gemm",
        kernel_source_path=None,
        parity_fixture=fp8_fixture_path,
        base_measurements=1,
        accepted_iteration_cap=1,
        total_attempt_cap=1,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="real",
        runtime={
            "container_name": "test",
            "port": 8100,
            "proxy_port": 8101,
            "endpoint": "http://127.0.0.1:8101/v1",
            "metrics_url": "http://127.0.0.1:8100/metrics",
            "admin_url": "http://127.0.0.1:8101/admin",
        },
        per_iteration_wall_clock_s=1,
    )

    assert result.outcome == "ROUND_BLOCKED"
    assert result.terminal_condition == "agent_unavailable"
    assert (result.round_dir / "candidates" / "001" / "iteration_brief.md").is_file()
    assert (result.round_dir / "prior_research_memory.tsv").is_file()
    assert (result.round_dir / "research_memory.tsv").is_file()
    assert (result.round_dir / "research_memory.md").is_file()
    strategy_brief = (result.round_dir / "strategy_brief.md").read_text(encoding="utf-8")
    assert "Prior-Art Memory Contract" in strategy_brief
    assert "Prior Measured-Trial Memory" in strategy_brief
    assert "compact config/schedule traces" in strategy_brief
    assert "mutation_features" in strategy_brief
    assert "GB10 CUTLASS Timing Breakdown" in strategy_brief
    assert "ffn_linear" in strategy_brief
    assert "which timing component it expects to reduce" in strategy_brief
    assert "structured compute/bandwidth accounting block" in strategy_brief
    assert "expected end-to-end tok/s delta" in strategy_brief
    assert "pre-change CUTLASS timing baseline" in strategy_brief
    assert "low-level CUTLASS sub-kernel timing" in strategy_brief
    assert "low-level evidence block" in strategy_brief
    assert "live warm shape dispatch hits that path" in strategy_brief
    assert "B-weight bytes change" in strategy_brief
    assert "3% end-to-end warm decode lift" in strategy_brief
    assert "10.1 tok/s full-model stream ceiling" in strategy_brief
    assert "not proof of achieved memory bandwidth" in strategy_brief
    candidate_brief = (result.round_dir / "candidates" / "001" / "iteration_brief.md").read_text(
        encoding="utf-8"
    )
    assert "research_memory.tsv" in candidate_brief
    assert "DGX Spark GB10" in candidate_brief
    assert "baseline timing breakdown" in candidate_brief
    assert "candidate_analysis.md" in candidate_brief
    assert "compute/bandwidth breakdown" in candidate_brief
    assert "FLOP or arithmetic-intensity sanity check" in candidate_brief
    assert "representative shape(s) as M/N/K" in candidate_brief
    assert "expected end-to-end tok/s delta" in candidate_brief
    assert "which CUTLASS time component" in candidate_brief
    assert "CUTLASS-internal timing/proxy" in candidate_brief
    assert "`ffn_linear` proxy" in candidate_brief
    assert "low-level evidence table" in candidate_brief
    assert "live-shape dispatch-hit proof" in candidate_brief
    assert "byte-component split" in candidate_brief
    assert "at least 3%" in candidate_brief
    assert "auto-research warm-diagnostic" in candidate_brief
    assert "MUST run a cheap warm-request diagnostic" in candidate_brief
    assert "warm_pre_mutation.json" in candidate_brief
    assert "aggregate_consumption.step_consumption" in candidate_brief
    assert "per_step_consumption" in candidate_brief
    assert "warm_diagnostic_skipped.json" in candidate_brief
    assert "per-step token/time/cache consumption" in candidate_brief
    assert "GB10 bandwidth roofline context" in candidate_brief
    research_memory_header = (result.round_dir / "research_memory.tsv").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    assert "workload_key" in research_memory_header
    assert "schedule_trace" in research_memory_header
    assert "patch_diff" in research_memory_header
    assert "failure_class" in research_memory_header
    assert "search_bias" in research_memory_header
    round_spec = auto_research.load_yaml_file(result.round_dir / "round_spec.yaml")
    assert isinstance(round_spec, dict)
    assert round_spec["mutation_surface"]["kind"] == "cutlass_source_workspace"
    assert round_spec["mutation_surface"]["runtime_wired"] is True
    assert "workspace_path" in round_spec["mutation_surface"]
    assert "prelaunch_shell" in round_spec["runtime"]
    assert "cmake --build /opt/vllm-source/build/lumo_cutlass_research --target _C" in round_spec[
        "runtime"
    ]["prelaunch_shell"]
    assert "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" in round_spec["runtime"]["prelaunch_shell"]
    assert "/opt/vllm-source" in round_spec["runtime"]["prelaunch_shell"]
    assert round_spec["parity_fixture_id"] == "responses-sdk-adapter-cutover-fp8-gemm-v1"


def test_l0c_research_memory_refresh_records_patch_diff_and_failure_class(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    round_dir = repo / "output" / "auto_research" / (
        "qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T000000Z"
    )
    (round_dir / "baselines").mkdir(parents=True)
    (round_dir / "candidates" / "001").mkdir(parents=True)
    (round_dir / "candidates" / "002").mkdir(parents=True)
    (round_dir / "candidates" / "003").mkdir(parents=True)
    runner._write_tsv(round_dir / "prior_research_memory.tsv", auto_research.L0C_RESEARCH_MEMORY_COLUMNS, [])
    (round_dir / "baselines" / "measurement_01.json").write_text(
        json.dumps({"eval_throughput": 0.056}, indent=2),
        encoding="utf-8",
    )
    patch_text = "\n".join(
        [
            "--- cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh",
            "+++ cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh",
            "@@ -1,1 +1,1 @@",
            "-  using TileShape = Shape<_16, _64, _128>;",
            "+  using TileShape = Shape<_16, _128, _128>;",
            "",
        ]
    )
    candidate_001 = round_dir / "candidates" / "001"
    candidate_002 = round_dir / "candidates" / "002"
    candidate_003 = round_dir / "candidates" / "003"
    (candidate_001 / "mutation.patch").write_text(patch_text, encoding="utf-8")
    (candidate_001 / "parity_check.json").write_text(
        json.dumps({"pass": True, "reason": "ran_passed_with_tier4_downstream_logit_diagnostic"}),
        encoding="utf-8",
    )
    (candidate_001 / "measurement_trace.json").write_text(
        json.dumps(
            {
                "measurements": [
                    {"objective_value": "0.034"},
                    {"objective_value": "0.045"},
                ],
                "objective_mean": 0.0395,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (candidate_002 / "mutation.patch").write_text(patch_text, encoding="utf-8")
    (candidate_002 / "parity_check.json").write_text(
        json.dumps(
            {
                "pass": False,
                "reason": "parity_fp8_tier4_downstream_logit_diverged",
                "tolerance_overshoot": 0.35,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (candidate_003 / "BLOCKED.md").write_text(
        "\n".join(
            [
                "# Candidate 003 Blocked",
                "",
                "Tried a compile preflight mutation in",
                "`csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8_dispatch.cuh`:",
                "change `TileShape = Shape<_64, _128, _128>` to `Shape<_64, _128, _256>`.",
                "Targeted compile preflight failed before submission, so no mutation.patch remains.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (candidate_003 / "agent_last_message.txt").write_text("blocked\n", encoding="utf-8")

    rows = runner._refresh_l0c_research_memory_from_artifacts(round_dir)

    assert rows[0]["outcome"] == "discard"
    assert rows[0]["failure_class"] == "performance"
    assert "using TileShape = Shape<_16, _128, _128>" in rows[0]["patch_diff"]
    assert rows[1]["outcome"] == "parity_fp8_tier4_downstream_logit_diverged"
    assert rows[1]["failure_class"] == "correctness"
    assert rows[2]["outcome"] == "agent_blocked_compile_preflight"
    assert rows[2]["failure_class"] == "build"
    assert rows[2]["surface"] == "cutlass_sm120_dispatch"
    assert "Shape<_64, _128, _256>" in rows[2]["patch_diff"]
    memory_text = (round_dir / "research_memory.md").read_text(encoding="utf-8")
    assert "failure_class=performance" in memory_text
    assert "gate=ran_passed_with_tier4_downstream_logit_diagnostic" in memory_text
    assert "patch_diff=" in memory_text
    augmented_brief = runner._append_l0c_auto_research_memory("strategy", memory_text)
    assert "Auto-Refreshed Candidate Memory" in augmented_brief
    assert "canonical patch_diff/failure_class table" in augmented_brief
    assert "failure_class=correctness" in augmented_brief
    assert "using TileShape = Shape<_16, _128, _128>" in augmented_brief
    baseline_rows, accepted_rows, _, results_rows = runner._load_l0c_resume_ledgers(round_dir)
    assert [row["objective_value"] for row in baseline_rows] == ["0.056000"]
    assert [row["objective_value"] for row in accepted_rows] == ["0.034000", "0.045000"]
    assert results_rows[0]["iteration"] == "001"
    assert results_rows[0]["status"] == "discard"
    assert results_rows[0]["objective_mean"] == "0.039500"


def test_l0c_fp8_gemm_real_non_cutlass_non_triton_backend_blocked(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    fp8_fixture_path = _write_l0a_fp8_fixture(repo, reference_fp8_gemm_kernel="machete")
    workload_path = _write_l0a_workload(repo)
    _add_l0a_workload_fp8_ref(workload_path)
    base_bundle = _write_l0a_bundle(
        repo,
        kernel_selection={
            "combo_id": "combo_003",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "machete",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(
        RuntimeError,
        match="HALT_REASON: l0c_fp8_gemm_real_backend_unsupported",
    ):
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="fp8_gemm",
            kernel_source_path=None,
            parity_fixture=fp8_fixture_path,
            base_measurements=1,
            accepted_iteration_cap=1,
            total_attempt_cap=1,
            round_timeout_hours=1.0,
            round_root=repo / "output" / "auto_research",
            harness="real",
            runtime={
                "container_name": "test",
                "port": 8100,
                "proxy_port": 8101,
                "endpoint": "http://127.0.0.1:8101/v1",
                "metrics_url": "http://127.0.0.1:8100/metrics",
                "admin_url": "http://127.0.0.1:8101/admin",
            },
        )


def test_l0c_fp8_gemm_real_cutlass_allows_explicit_source_path(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    _write_l0a_workload(repo)
    explicit_source = repo / "vendor_cutlass_kernel.cu"
    explicit_source.write_text("// local source marker\n", encoding="utf-8")
    base_bundle = _write_l0a_bundle(
        repo,
        kernel_selection={
            "combo_id": "combo_002",
            "attention_backend": "vllm-default",
            "deltanet_kernel": "triton-chunked-delta-v2",
            "fp8_gemm_kernel": "cutlass",
            "torch_compile_mode": "default",
            "cuda_graph_capture": "off",
        },
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    base = auto_research.load_baseline_bundle(base_bundle)
    assert base is not None

    surface = runner._resolve_fp8_gemm_real_mutation_surface(
        base=base,
        kernel_source_path=explicit_source,
    )

    assert surface["kind"] == "cutlass_source_workspace"
    assert surface["kernel_source_path"] == str(explicit_source.resolve())
    assert surface["source_mutability"] == "staged_vllm_cutlass_source_tree"


def test_l0c_kernel_mutation_synthetic_halts_on_proposer_stuck(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    fixture_path = (
        repo
        / "benchmark_blueprints"
        / "families"
        / "responses-sdk-adapter-cutover"
        / "parity_fixture"
        / "deltanet_v1.yaml"
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def _always_parity_fail(attempt_index: int) -> dict:
        return {"stage": "parity_fail"}

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path="kernels/deltanet/chunked_delta.py",
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=4,
        total_attempt_cap=12,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
        attempt_outcome_fn=_always_parity_fail,
    )

    assert result.terminal_condition == "proposer_stuck"
    assert result.accepted_count == 0
    assert result.total_attempt_count == 3
    assert result.outcome == "ROUND_BLOCKED"
    assert result.bundle_path is None


def test_l0c_kernel_mutation_synthetic_halts_on_compile_failures(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    fixture_path = (
        repo
        / "benchmark_blueprints"
        / "families"
        / "responses-sdk-adapter-cutover"
        / "parity_fixture"
        / "deltanet_v1.yaml"
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    def _always_compile_fail(attempt_index: int) -> dict:
        return {"stage": "compile_failed"}

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path="kernels/deltanet/chunked_delta.py",
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=4,
        total_attempt_cap=12,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
        attempt_outcome_fn=_always_compile_fail,
    )

    assert result.terminal_condition == "compile_failures_3x"
    assert result.outcome == "ROUND_BLOCKED"


def test_l0c_kernel_mutation_real_harness_requires_runtime_block(tmp_path: Path) -> None:
    """Real-harness L0c rounds need a runtime block (container/port/endpoints).

    Slice 3 wired the round driver; the old HALT_REASON gate is gone. The remaining
    pre-flight check is that callers explicitly pass runtime info — without it we
    can't restart vLLM or talk to the parity probe endpoint.
    """
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    fixture_path = (
        repo
        / "benchmark_blueprints"
        / "families"
        / "responses-sdk-adapter-cutover"
        / "parity_fixture"
        / "deltanet_v1.yaml"
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    with pytest.raises(RuntimeError, match="real-harness L0c rounds require --runtime"):
        runner.run(
            workload_file=workload_path,
            base_bundle=base_bundle,
            kernel_target="deltanet",
            kernel_source_path="kernels/deltanet/chunked_delta.py",
            parity_fixture=fixture_path,
            base_measurements=1,
            accepted_iteration_cap=2,
            total_attempt_cap=4,
            round_timeout_hours=0.1,
            round_root=repo / "output" / "auto_research",
            harness="real",
        )


def test_l0c_apply_and_test_synthetic_routes_parity_outcomes(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    fixture_path = (
        repo
        / "benchmark_blueprints"
        / "families"
        / "responses-sdk-adapter-cutover"
        / "parity_fixture"
        / "deltanet_v1.yaml"
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    # Bootstrap a round so the apply-and-test CLI has a round_spec.yaml to read.
    bootstrapped = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path="kernels/deltanet/chunked_delta.py",
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=1,
        total_attempt_cap=2,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
    )
    round_id = bootstrapped.round_id

    iteration_dir = bootstrapped.round_dir / "candidates" / "010"
    iteration_dir.mkdir()
    (iteration_dir / "mutation.patch").write_text(
        "--- a/kernels/deltanet/chunked_delta.py\n+++ b/kernels/deltanet/chunked_delta.py\n@@ -1 +1 @@\n-# baseline\n+# good mutation\n",
        encoding="utf-8",
    )
    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="010",
        kernel_target="deltanet",
        harness="synthetic",
        round_root=repo / "output" / "auto_research",
    )
    assert payload["outcome"] == "parity_passed"
    assert (iteration_dir / "measurement_trace.json").is_file()

    bad_dir = bootstrapped.round_dir / "candidates" / "011"
    bad_dir.mkdir()
    (bad_dir / "mutation.patch").write_text(
        "--- a/kernels/deltanet/chunked_delta.py\n+++ b/kernels/deltanet/chunked_delta.py\n@@ -1 +1 @@\n-# baseline\n+# BAD_PARITY\n",
        encoding="utf-8",
    )
    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="011",
        kernel_target="deltanet",
        harness="synthetic",
        round_root=repo / "output" / "auto_research",
    )
    assert payload["outcome"] == "parity_failed"
    assert not (bad_dir / "measurement_trace.json").exists()


def test_l0c_iteration_brief_substitutes_kernel_target_and_round_id(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_l0a_fixture_pair(repo)
    workload_path = _write_l0a_workload(repo)
    base_bundle = _write_l0a_bundle(repo)
    fixture_path = (
        repo
        / "benchmark_blueprints"
        / "families"
        / "responses-sdk-adapter-cutover"
        / "parity_fixture"
        / "deltanet_v1.yaml"
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    result = runner.run(
        workload_file=workload_path,
        base_bundle=base_bundle,
        kernel_target="deltanet",
        kernel_source_path="kernels/deltanet/chunked_delta.py",
        parity_fixture=fixture_path,
        base_measurements=1,
        accepted_iteration_cap=1,
        total_attempt_cap=2,
        round_timeout_hours=1.0,
        round_root=repo / "output" / "auto_research",
        harness="synthetic",
    )
    brief = (result.round_dir / "iteration_brief.md").read_text(encoding="utf-8")
    assert "kernels/deltanet/chunked_delta.py" in brief
    assert result.round_id in brief
    assert "auto-research apply-and-test" in brief
    assert "{{kernel_target}}" not in brief


def test_l0c_fp8_cutlass_brief_defers_apply_and_test_to_controller(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    brief = runner._render_brief(
        kernel_target="fp8_gemm",
        kernel_source_path=repo
        / "src"
        / "lumo_flywheel_serving"
        / "kernel_overlays"
        / "fp8_gemm_cutlass_overlay_bootstrap.py",
        fixture_path=repo / "parity_fixture" / "fp8_gemm_v1.yaml",
        fixture_payload={
            "tier_3_tolerances": {"rtol_gemm_output": 0.002, "atol_gemm_output": 0.002},
            "tier_4_tolerances": {
                "rtol_downstream_logit": 0.001,
                "atol_downstream_logit": 0.001,
            },
        },
        round_id="round-fp8",
        harness="real",
        mutation_surface={
            "kind": "cutlass_source_workspace",
            "workspace_path": str(repo / "round" / "cutlass_source_workspace"),
            "workspace_source_path": str(repo / "round" / "cutlass_source_workspace" / "vllm-source"),
            "workspace_python_source_path": str(
                repo
                / "round"
                / "cutlass_source_workspace"
                / "vllm-source"
                / "vllm"
                / "model_executor"
                / "kernels"
                / "linear"
                / "scaled_mm"
            ),
            "workspace_cxx_source_path": str(
                repo
                / "round"
                / "cutlass_source_workspace"
                / "vllm-source"
                / "csrc"
                / "quantization"
                / "w8a8"
                / "cutlass"
            ),
            "container_vllm_source_dir": "/opt/vllm-source",
            "round_dir": str(repo / "round"),
        },
    )

    assert "Do not run `auto-research apply-and-test`" in brief
    assert "local CUTLASS source workspace" in brief
    assert "cutlass_source_workspace" in brief
    assert "prior_research_memory.tsv" in brief
    assert "research_memory.tsv" in brief
    assert "research_memory.md" in brief
    assert "/opt/vllm-source" in brief
    assert "patch --dry-run -p0" in brief
    assert "python3 -m py_compile" in brief
    assert "auto-research preflight-patch" in brief
    assert "--workspace-source" in brief
    assert "--compile-mode targeted" in brief
    assert "--compile-jobs 1" in brief
    assert "You own authoring-time compile failures" in brief
    assert "A compiled-file mutation is not ready to submit" in brief
    assert "CUTLASS FP8/SM120 objects" in brief
    assert "compile_preflight.output_tail" in brief
    assert "matching_rule" in brief
    assert "code_snippet" in brief
    assert "evidence_snippet" in brief
    assert "nonzero exit is reserved for agent/tool infrastructure failure" in brief
    assert "short targeted research pass" in brief
    assert "primary docs/source" in brief
    assert "cheap local diagnostics" in brief
    assert "DGX Spark GB10" in brief
    assert "baseline timing breakdown" in brief
    assert "candidate_analysis.md" in brief
    assert "compute/bandwidth breakdown" in brief
    assert "FLOP or arithmetic-intensity sanity check" in brief
    assert "representative shape(s) as M/N/K" in brief
    assert "expected end-to-end tok/s delta" in brief
    assert "which CUTLASS time component" in brief
    assert "CUTLASS-internal timing/proxy" in brief
    assert "`ffn_linear` proxy" in brief
    assert "low-level evidence table" in brief
    assert "live-shape dispatch-hit proof" in brief
    assert "byte-component split" in brief
    assert "at least 3%" in brief
    assert "auto-research warm-diagnostic" in brief
    assert "MUST run a cheap warm-request diagnostic" in brief
    assert "warm_pre_mutation.json" in brief
    assert "aggregate_consumption.step_consumption" in brief
    assert "per_step_consumption" in brief
    assert "warm_diagnostic_skipped.json" in brief
    assert "per-step token/time/cache consumption" in brief
    assert "GB10 bandwidth roofline context" in brief
    assert "effective bandwidth in GB/s" in brief
    assert "percent of the 273 GB/s GB10 ceiling" in brief
    assert "10.1 tok/s full-model FP8 stream ceiling" in brief
    assert "roofline as context, not proof of achieved memory bandwidth" in brief
    assert "`ffn_linear` share of ms/token" in brief
    assert "non-FFN residual ms/token" in brief
    assert "Do not start vLLM and do not run apply-and-test" in brief
    assert "do not spend iteration budget on online research" not in brief
    assert "expected speed mechanism" in brief
    assert "Do not replace `process_weights_after_loading` wholesale" in brief
    assert "CUTLASS weight transposition" in brief
    assert "This round is CUTLASS-only" in brief
    assert "w8a8_triton_block_scaled_mm_func" in brief
    assert "CUTLASS dispatch" in brief
    assert "scale tensors" in brief
    assert "GEMM problem shape" in brief
    assert "fp8_gemm_cutlass_python_wrapper_rewrite" not in brief
    assert "auto-research apply-and-test \\" not in brief


def test_l0c_fp8_cutlass_preflight_rejects_non_cutlass_backend_route(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    patch_text = "\n".join(
        [
            "--- cutlass_source_workspace/vllm-source/vllm/model_executor/layers/quantization/utils/fp8_utils.py",
            "+++ cutlass_source_workspace/vllm-source/vllm/model_executor/layers/quantization/utils/fp8_utils.py",
            "@@ -1,1 +1,1 @@",
            "+        return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(",
            "+            q_input, weight, input_scale, weight_scale, list(self.weight_group_shape), input_2d.dtype)",
            "",
        ]
    )

    preflight = runner._preflight_l0c_patch(kernel_target="fp8_gemm", patch_text=patch_text)

    assert preflight is not None
    assert preflight["tier"] == "safety_critical"
    assert preflight["pattern_id"] == "safety_routes_cutlass_round_to_non_cutlass_backend"


def test_l0c_candidate_analysis_requires_structured_roofline_accounting(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "candidate_analysis.md").write_text(
        """
# Candidate Analysis

Warm decode rate: 7.37 generated tokens/s, 135.6 ms/generated token.
GB10 bandwidth: 273 GB/s, with a warm roofline context of 37.0 GB/token.
The 10.1 tok/s full-model FP8 stream ceiling is a roofline context number, not
proof of achieved memory bandwidth.
The CUTLASS/FP8 GEMM proxy is ffn_linear at 80.6 ms/token, so this is the
component the mutation must reduce.
The analysis reads warm_pre_mutation.json and uses aggregate_consumption,
per_step_consumption, and bottleneck_hint from the authoring-time warm
diagnostic rather than inventing a timing source.

Structured compute/bandwidth accounting:

| representative shape M/N/K | FLOPs per GEMM | estimated bytes moved | arithmetic intensity | roofline/ceiling | ffn_linear ms/token | expected changed bytes/FLOPs/overhead | expected end-to-end tok/s delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M=1, N=8192, K=8192 | 134M FLOP | 67 MB plus scales/output | about 2 FLOP/byte, memory-bound | below the theoretical stream ceiling | 80.6 | less epilogue scale overhead, unchanged bytes/FLOPs | +0.2 tok/s delta if the changed visitor is visible |

7.5 tok/s breakdown: 37.0 GB/token at 7.37 tok/s implies about 273 GB/s
effective bandwidth, roughly 100% of the 273 GB/s ceiling. `ffn_linear` share is
80.6 / 135.6 ms/token = 59% share of ms/token, leaving a non-FFN residual
ms/token of about 55.0. This patch attacks schedule/epilogue overhead, not
B-weight traffic. This is roofline context rather than proof of measured memory
bandwidth.

Low-level evidence:

| source file/symbol | live-shape dispatch-hit proof | before-mutation observation | byte-component split for A/B weights/scales/output/epilogue | B-weight bytes change? | material lift gate |
| --- | --- | --- | --- | --- | --- |
| source file csrc/quantization/w8a8/cutlass/scaled_mm_sm120_fp8.cu, symbol cutlass_scaled_mm_sm120_fp8 | dispatch-hit proof from live shape M=1,N=8192,K=8192: path is hit by CutlassFP8ScaledMMLinearKernel | warm diagnostic shows 7.37 tok/s and targeted compile/preflight passes | A bytes tiny, B-weight bytes dominate, scale bytes scalar, output store small, epilogue overhead candidate | weight bytes unchanged; B-weight bytes do not change | at least 3% end-to-end only if epilogue overhead is a measured long-tail bottleneck |

Mechanism: the mutation should reduce scalar-scale epilogue overhead without
changing the GEMM signature or scale semantics. The expected reduction is not a
weight-byte reduction, so the analysis explicitly treats the roofline as context
rather than proof. Current measurement rows are baseline_a/b and recent
candidates around 0.034/0.056 objective; the patch must lift observed warm
decode materially over the recent 7.5 tok/s level to matter.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert auto_research.L0cKernelMutationRunner._validate_l0c_candidate_analysis(candidate_dir) is None


def test_l0c_candidate_analysis_rejects_missing_shape_accounting(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "candidate_analysis.md").write_text(
        (
            "Warm decode rate 7.37 generated tokens/s and 135.6 ms/generated token. "
            "GB10 bandwidth is 273 GB/s and warm roofline budget is 37 GB/token. "
            "CUTLASS ffn_linear FP8 GEMM proxy is 80.6 ms/token. "
            "This compute sanity check says FLOPs and arithmetic intensity are important, "
            "but omits the actual representative dimensions. "
            "The mutation mechanism should reduce expected reduction overhead and lift "
            "warm decode; expected end-to-end tok/s delta is measurable. "
        )
        * 8,
        encoding="utf-8",
    )

    error = auto_research.L0cKernelMutationRunner._validate_l0c_candidate_analysis(candidate_dir)

    assert error is not None
    assert "representative_shape" in error


def test_l0c_candidate_analysis_rejects_missing_low_level_evidence(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "candidate_analysis.md").write_text(
        """
# Candidate Analysis

Warm decode rate: 7.37 generated tokens/s, 135.6 ms/generated token.
GB10 bandwidth: 273 GB/s, with a warm roofline context of 37.0 GB/token.
The 10.1 tok/s full-model FP8 stream ceiling is roofline context, not proof of
achieved memory bandwidth.
The CUTLASS/FP8 GEMM proxy is ffn_linear at 80.6 ms/token.

Structured compute/bandwidth accounting:

| representative shape M/N/K | FLOPs per GEMM | estimated bytes moved | arithmetic intensity | roofline/ceiling | ffn_linear ms/token | expected changed bytes/FLOPs/overhead | expected end-to-end tok/s delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M=1, N=8192, K=8192 | 134M FLOP | 67 MB plus scales/output | about 2 FLOP/byte, memory-bound | below the theoretical stream ceiling | 80.6 | less overhead | +0.3 tok/s delta |

7.5 tok/s breakdown: effective bandwidth is 273 GB/s observed, 100% of the
273 GB/s ceiling. ffn_linear share is 59% share of ms/token, non-FFN residual
ms/token is 55. This is not measured memory bandwidth.

Mechanism: the mutation should reduce overhead and lift warm decode. This has
the old high-level analysis shape but omits the edited file identity, dispatch
proof, byte split, B-byte statement, and before-mutation low-level observation.
""".strip()
        + "\n",
        encoding="utf-8",
    )

    error = auto_research.L0cKernelMutationRunner._validate_l0c_candidate_analysis(candidate_dir)

    assert error is not None
    assert "source_symbol" in error
    assert "dispatch_hit_proof" in error
    assert "byte_component_split" in error


def test_l0c_warm_diagnostic_records_step_consumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    workload_path = _write_l0a_workload(repo)
    round_root = repo / "output" / "auto_research"
    round_id = "round-warm"
    round_dir = round_root / round_id
    round_dir.mkdir(parents=True)
    auto_research.L0cKernelMutationRunner._write_yaml(
        round_dir / "round_spec.yaml",
        {
            "round_id": round_id,
            "workload_file": str(workload_path),
            "kernel_target": "fp8_gemm",
            "runtime": {
                "endpoint": "http://127.0.0.1:8101/v1",
                "metrics_url": "http://127.0.0.1:8100/metrics",
                "model_id": "qwen3.5-27b",
                "vllm_config": {},
            },
        },
    )
    counters = {
        "prompt": 0.0,
        "gen": 0.0,
        "kv": 0.0,
        "ttft_sum": 0.0,
        "ttft_count": 0.0,
        "prefill": 0.0,
        "decode": 0.0,
        "itl": 0.0,
        "queries": 0.0,
        "hits": 0.0,
    }

    def prom() -> str:
        return "\n".join(
            [
                f"vllm:prompt_tokens_total {counters['prompt']}",
                f"vllm:generation_tokens_total {counters['gen']}",
                f"vllm:request_prefill_kv_computed_tokens_sum {counters['kv']}",
                f"vllm:time_to_first_token_seconds_sum {counters['ttft_sum']}",
                f"vllm:time_to_first_token_seconds_count {counters['ttft_count']}",
                f"vllm:request_prefill_time_seconds_sum {counters['prefill']}",
                f"vllm:request_decode_time_seconds_sum {counters['decode']}",
                f"vllm:inter_token_latency_seconds_sum {counters['itl']}",
                f"vllm:prefix_cache_queries_total {counters['queries']}",
                f"vllm:prefix_cache_hits_total {counters['hits']}",
            ]
        )

    def fake_get(url: str, **kwargs):
        assert url == "http://127.0.0.1:8100/metrics"
        del kwargs
        return _HTTPResponse(text=prom())

    def fake_post(url: str, **kwargs):
        assert url == "http://127.0.0.1:8101/v1/responses"
        payload = kwargs.get("json") or {}
        assert payload.get("model") == "qwen3.5-27b"
        prompt_tokens = len(str(payload.get("input", "")).split())
        gen_tokens = int(payload.get("max_output_tokens") or 1)
        counters["prompt"] += prompt_tokens
        counters["gen"] += gen_tokens
        counters["kv"] += prompt_tokens
        counters["ttft_sum"] += 0.1
        counters["ttft_count"] += 1
        counters["prefill"] += 0.01
        counters["decode"] += 0.02
        counters["itl"] += 0.02
        counters["queries"] += 1
        counters["hits"] += 1
        return _HTTPResponse(payload={"id": "resp"})

    monkeypatch.setattr(auto_research.requests, "get", fake_get)
    monkeypatch.setattr(auto_research.requests, "post", fake_post)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    payload = runner.warm_diagnostic(
        round_id=round_id,
        iteration="001",
        round_root=round_root,
        phase="pre_mutation",
        request_count=2,
        warmup_requests=1,
        max_output_tokens=8,
        prompt_token_cap=16,
    )

    artifact = Path(payload["artifact_path"])
    assert artifact.is_file()
    assert payload["aggregate_consumption"]["available"] is True
    assert len(payload["per_step_consumption"]) == 2
    assert payload["gb10_reference"]["theoretical_bandwidth_gb_s"] == 273.0
    assert payload["policy"]["controller_owns_patched_vllm_restart"] is True


def test_l0c_preflight_patch_allows_fp8_cutlass_source_edits(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    patch_path = repo / "candidate.patch"
    patch_path.write_text(
        """--- cutlass_source_workspace/scaled_mm/cutlass.py
+++ cutlass_source_workspace/scaled_mm/cutlass.py
@@ -1,2 +1,2 @@
-output = ops.cutlass_scaled_mm(A, B)
+output = ops.cutlass_scaled_mm(A.contiguous(), B)
""",
        encoding="utf-8",
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    payload = runner.preflight_patch(kernel_target="fp8_gemm", patch_path=patch_path)

    assert payload["ok"] is True
    assert payload["reason"] == "preflight_passed"
    assert payload["rules"]
    assert all(rule["tier"] == "safety_critical" for rule in payload["rules"])


def test_l0c_preflight_patch_compiles_workspace_copy_without_mutating_source(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    workspace_source = repo / "round" / "cutlass_source_workspace" / "vllm-source"
    python_source = (
        workspace_source
        / "vllm"
        / "model_executor"
        / "kernels"
        / "linear"
        / "scaled_mm"
    )
    python_source.mkdir(parents=True)
    target = python_source / "cutlass.py"
    target.write_text("def scaled_mm(a, b):\n    return a + b\n", encoding="utf-8")
    patch_path = repo / "round" / "candidates" / "001" / "mutation.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        """--- cutlass_source_workspace/vllm-source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
+++ cutlass_source_workspace/vllm-source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
@@ -1,2 +1,2 @@
 def scaled_mm(a, b):
-    return a + b
+    return a - b
""",
        encoding="utf-8",
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    payload = runner.preflight_patch(
        kernel_target="fp8_gemm",
        patch_path=patch_path,
        workspace_source=workspace_source,
        compile_mode="python",
    )

    assert payload["ok"] is True
    assert payload["compile_preflight"]["reason"] == "compile_preflight_passed"
    assert target.read_text(encoding="utf-8") == "def scaled_mm(a, b):\n    return a + b\n"


def test_l0c_preflight_patch_reports_compile_failure_for_authoring_agent(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    workspace_source = repo / "round" / "cutlass_source_workspace" / "vllm-source"
    python_source = (
        workspace_source
        / "vllm"
        / "model_executor"
        / "kernels"
        / "linear"
        / "scaled_mm"
    )
    python_source.mkdir(parents=True)
    (python_source / "cutlass.py").write_text("def scaled_mm(a, b):\n    return a + b\n", encoding="utf-8")
    patch_path = repo / "round" / "candidates" / "001" / "mutation.patch"
    patch_path.parent.mkdir(parents=True)
    patch_path.write_text(
        """--- cutlass_source_workspace/vllm-source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
+++ cutlass_source_workspace/vllm-source/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py
@@ -1,2 +1,2 @@
 def scaled_mm(a, b):
-    return a + b
+    return (
""",
        encoding="utf-8",
    )
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )

    payload = runner.preflight_patch(
        kernel_target="fp8_gemm",
        patch_path=patch_path,
        workspace_source=workspace_source,
        compile_mode="python",
    )

    assert payload["ok"] is False
    assert payload["reason"] == "python_compile_failed"
    assert "py_compile_failed" in payload["compile_preflight"]["error"]


def test_cutlass_compile_preflight_shell_defaults_to_single_job() -> None:
    shell = auto_research.L0cKernelMutationRunner._cutlass_cmake_rebuild_shell(
        install=False,
        default_jobs=1,
    )

    assert "export MAX_JOBS=${MAX_JOBS:-1}" in shell
    assert "export TORCH_CUDA_ARCH_LIST=${LUMO_CUTLASS_TORCH_CUDA_ARCH_LIST:-12.0}" in shell
    assert '-DTORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"' in shell
    assert "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache" in shell
    assert "set(MARLIN_ARCHS \"\")" in shell
    assert "set(MARLIN_MOE_OTHER_ARCHS \"\")" in shell
    assert "set(SCALED_MM_2X_ARCHS \"\")" in shell
    assert "set(FP4_ARCHS \"\")" in shell
    assert "set(MLA_ARCHS \"\")" in shell
    assert "set(CUTLASS_MOE_DATA_ARCHS \"\")" in shell
    assert "set(QUTLASS_ARCHS \"\")" in shell
    assert "missing expected non-FP8 CMake fragment(s)" in shell
    assert "missing expected QuTLASS CMake fragment(s)" in shell
    assert "[LUMO-CUTLASS-REBUILD] disabled non-FP8 Marlin, C2X, NVFP4, MLA, MoE data, and QuTLASS CMake sources" in shell


def test_cutlass_targeted_compile_targets_sm120_dispatch_header() -> None:
    patch_text = """--- cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh
+++ cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh
@@ -1,2 +1,2 @@
-using TileShape = Shape<_32, _64, _128>;
+using TileShape = Shape<_32, _128, _128>;
"""

    targets, changed_paths = auto_research.L0cKernelMutationRunner._cutlass_targeted_compile_targets(
        patch_text
    )

    assert "cutlass_source_workspace/vllm-source/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8_dispatch.cuh" in changed_paths
    assert "CMakeFiles/_C.dir/csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm120_fp8.cu.o" in targets
    assert "CMakeFiles/_C.dir/csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm120.cu.o" in targets


def test_cutlass_rebuild_prelaunch_defaults_to_single_job() -> None:
    shell = auto_research.L0cKernelMutationRunner._cutlass_rebuild_prelaunch_shell()

    assert "export MAX_JOBS=${MAX_JOBS:-1}" in shell
    assert "set(MARLIN_ARCHS \"\")" in shell
    assert "set(SCALED_MM_2X_ARCHS \"\")" in shell
    assert "set(FP4_ARCHS \"\")" in shell
    assert "set(MLA_ARCHS \"\")" in shell
    assert "set(CUTLASS_MOE_DATA_ARCHS \"\")" in shell
    assert "set(QUTLASS_ARCHS \"\")" in shell
    assert "[LUMO-CUTLASS-REBUILD] disabled non-FP8 Marlin, C2X, NVFP4, MLA, MoE data, and QuTLASS CMake sources" in shell


def test_l0c_fp8_cutlass_overlay_wrapper_patch_is_not_file_forbidden(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    patch_text = """--- src/lumo_flywheel_serving/kernel_overlays/fp8_gemm_cutlass_overlay_bootstrap.py
+++ src/lumo_flywheel_serving/kernel_overlays/fp8_gemm_cutlass_overlay_bootstrap.py
@@ -18,6 +18,12 @@
-        "source_replacements": [],
+        "source_replacements": [
+            {"label": "alias", "before": "ops.cutlass_scaled_mm(", "after": "_alias("},
+        ],
"""

    preflight = runner._preflight_l0c_patch(kernel_target="fp8_gemm", patch_text=patch_text)

    assert preflight is None


def test_l0c_fp8_cutlass_overlay_comment_patch_is_not_file_forbidden(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    patch_text = """--- src/lumo_flywheel_serving/kernel_overlays/fp8_gemm_cutlass_overlay_bootstrap.py
+++ src/lumo_flywheel_serving/kernel_overlays/fp8_gemm_cutlass_overlay_bootstrap.py
@@ -1,3 +1,4 @@
 # Existing comment
+# Candidate note
 """

    preflight = runner._preflight_l0c_patch(kernel_target="fp8_gemm", patch_text=patch_text)

    assert preflight is None


def test_l0c_fp8_cutlass_overlay_patch_with_timestamp_header_is_not_file_forbidden(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    patch_text = """--- /home/mark/shared/lumoFlyWheel/src/lumo_flywheel_serving/kernel_overlays/fp8_gemm_cutlass_overlay_bootstrap.py\t2026-05-03 03:37:44.132445545 +0000
+++ /dev/fd/63\t2026-05-03 03:41:34.977865197 +0000
@@ -18,6 +18,12 @@
-        "source_replacements": [],
+        "source_replacements": [
+            {"label": "comment_only", "before": "# Fused GEMM_DQ", "after": "# Fused GEMM_DQ # note"},
+        ],
"""

    preflight = runner._preflight_l0c_patch(kernel_target="fp8_gemm", patch_text=patch_text)

    assert preflight is None

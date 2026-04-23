from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .registry import ModelConfig
from .tuned_config import TunedConfigBundle, default_weight_version_id, make_tuned_config_bundle, persist_tuned_config_bundle
from .yaml_utils import load_yaml_file

MIN_LIVE_STARTUP_GPU_MEMORY_UTILIZATION = 0.30


@dataclass(frozen=True)
class SyntheticWorkloadDistribution:
    family_id: str
    workload_distribution_id: str
    latency_ceiling_ms: int
    p99_context_tokens: int
    avg_prompt_tokens: int
    avg_output_tokens: int
    rollout_baseline: float
    measurement_window_minutes: int = 30
    gpu_memory_utilization_cap: float | None = None

    @classmethod
    def default_for(cls, *, model_config: ModelConfig, family_id: str) -> "SyntheticWorkloadDistribution":
        payload = {
            "family_id": family_id,
            "p99_context_tokens": min(model_config.max_model_len // 2, 32768),
            "avg_prompt_tokens": min(model_config.max_model_len // 16, 4096),
            "avg_output_tokens": 768,
            "latency_ceiling_ms": 650,
            "rollout_baseline": 14.0,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return cls(
            family_id=family_id,
            workload_distribution_id=digest,
            latency_ceiling_ms=payload["latency_ceiling_ms"],
            p99_context_tokens=payload["p99_context_tokens"],
            avg_prompt_tokens=payload["avg_prompt_tokens"],
            avg_output_tokens=payload["avg_output_tokens"],
            rollout_baseline=payload["rollout_baseline"],
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        model_config: ModelConfig,
        family_id: str,
    ) -> "SyntheticWorkloadDistribution":
        raw = load_yaml_file(path)
        if not isinstance(raw, dict):
            raise ValueError(f"Workload file {path} must be a mapping")
        payload = dict(raw)
        payload.setdefault("family_id", family_id)
        payload.setdefault("p99_context_tokens", min(model_config.max_model_len // 2, 32768))
        payload.setdefault("avg_prompt_tokens", min(model_config.max_model_len // 16, 4096))
        payload.setdefault("avg_output_tokens", 768)
        payload.setdefault("latency_ceiling_ms", 650)
        payload.setdefault("rollout_baseline", 14.0)
        payload.setdefault(
            "workload_distribution_id",
            hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12],
        )
        return cls(
            family_id=str(payload["family_id"]),
            workload_distribution_id=str(payload["workload_distribution_id"]),
            latency_ceiling_ms=int(payload["latency_ceiling_ms"]),
            p99_context_tokens=int(payload["p99_context_tokens"]),
            avg_prompt_tokens=int(payload["avg_prompt_tokens"]),
            avg_output_tokens=int(payload["avg_output_tokens"]),
            rollout_baseline=float(payload["rollout_baseline"]),
            measurement_window_minutes=int(payload.get("measurement_window_minutes", 30)),
            gpu_memory_utilization_cap=(
                float(payload["gpu_memory_utilization_cap"])
                if payload.get("gpu_memory_utilization_cap") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    iteration: int
    label: str
    candidate: dict[str, Any]
    feasible: bool
    objective_value: int
    p95_latency_ms: float
    rollout_throughput: float
    determinism_pass_rate: float
    kv_probe_passed: bool
    oom: bool
    reason: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "label": self.label,
            "candidate": self.candidate,
            "feasible": self.feasible,
            "objective_value": self.objective_value,
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "rollout_throughput": round(self.rollout_throughput, 3),
            "determinism_pass_rate": round(self.determinism_pass_rate, 6),
            "kv_probe_passed": self.kv_probe_passed,
            "oom": self.oom,
            "reason": self.reason,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class OfflineAutoResearchResult:
    status: str
    stopping_reason: str
    run_dir: Path
    search_trace_path: Path
    measurement_trace_path: Path
    run_log_path: Path
    bundle_path: Path | None
    baseline_value: int
    best_value: int
    best_candidate_label: str | None


class SyntheticMeasurementHarness:
    def __init__(self, workload: SyntheticWorkloadDistribution) -> None:
        self.workload = workload

    def evaluate(self, candidate: dict[str, Any], *, iteration: int, label: str) -> CandidateEvaluation:
        vllm_config = dict(candidate)
        overrides = vllm_config.pop("harness_overrides", {})
        max_num_seqs = int(vllm_config["max_num_seqs"])
        max_num_batched_tokens = int(vllm_config["max_num_batched_tokens"])
        enable_chunked_prefill = bool(vllm_config["enable_chunked_prefill"])
        enable_prefix_caching = bool(vllm_config["enable_prefix_caching"])
        gpu_memory_utilization = float(vllm_config["gpu_memory_utilization"])
        max_model_len = int(vllm_config["max_model_len"])

        memory_load = (
            (max_num_seqs * max_model_len / 131072) * 0.045
            + (max_num_batched_tokens / 16384) * 0.18
            + (0.08 if enable_prefix_caching else 0.0)
            + (0.05 if enable_chunked_prefill else 0.0)
        )
        force_oom = bool(overrides.get("force_oom"))
        if force_oom or memory_load > gpu_memory_utilization + 0.05:
            return CandidateEvaluation(
                iteration=iteration,
                label=label,
                candidate=candidate,
                feasible=False,
                objective_value=0,
                p95_latency_ms=float(self.workload.latency_ceiling_ms) * 1.4,
                rollout_throughput=0.0,
                determinism_pass_rate=1.0,
                kv_probe_passed=True,
                oom=True,
                reason="oom",
                metrics={"memory_load": round(memory_load, 4), "gpu_memory_utilization": gpu_memory_utilization},
            )

        batch_factor = min(max_num_batched_tokens / max(self.workload.avg_prompt_tokens * 2, 1), 2.5)
        cache_factor = 1.15 if enable_prefix_caching else 0.8
        prefill_factor = 1.1 if enable_chunked_prefill else 0.9
        memory_factor = max(0.2, 0.75 + ((gpu_memory_utilization - 0.70) * 1.5))
        length_factor = 1.0 if max_model_len >= self.workload.p99_context_tokens + 2048 else 0.5
        raw_capacity = max_num_seqs * batch_factor * cache_factor * prefill_factor * memory_factor * length_factor

        latency_ratio = (
            0.38
            + (max_num_seqs / 40.0)
            + (max_model_len / 262144.0)
            + (0.12 * (1.0 - (batch_factor / 2.5)))
            + (0.06 if not enable_chunked_prefill else -0.03)
            + (0.05 if not enable_prefix_caching else -0.02)
            - ((gpu_memory_utilization - 0.70) * 0.15)
        )
        p95_latency_ms = self.workload.latency_ceiling_ms * latency_ratio
        rollout_throughput = 8.0 * batch_factor * cache_factor * prefill_factor
        determinism_pass_rate = 1.0
        if overrides.get("inject_nondeterminism"):
            determinism_pass_rate = 0.95
        kv_probe_passed = not bool(overrides.get("inject_kv_poisoning"))
        feasible = (
            p95_latency_ms <= self.workload.latency_ceiling_ms
            and rollout_throughput >= (self.workload.rollout_baseline * 0.5)
            and determinism_pass_rate >= 0.999
            and kv_probe_passed
        )
        objective_value = min(max_num_seqs, int(raw_capacity / 2.0))
        if feasible and objective_value == 0:
            objective_value = 1
        feasible = feasible and objective_value > 0
        if not feasible:
            if determinism_pass_rate < 0.999:
                reason = "determinism_check_failed"
            elif not kv_probe_passed:
                reason = "kv_probe_failed"
            elif rollout_throughput < (self.workload.rollout_baseline * 0.5):
                reason = "rollout_floor_failed"
            else:
                reason = "latency_ceiling_failed"
        else:
            reason = "ok"
        return CandidateEvaluation(
            iteration=iteration,
            label=label,
            candidate=candidate,
            feasible=feasible,
            objective_value=objective_value if feasible else 0,
            p95_latency_ms=p95_latency_ms,
            rollout_throughput=rollout_throughput,
            determinism_pass_rate=determinism_pass_rate,
            kv_probe_passed=kv_probe_passed,
            oom=False,
            reason=reason,
            metrics={
                "memory_load": round(memory_load, 4),
                "batch_factor": round(batch_factor, 4),
                "cache_factor": round(cache_factor, 4),
                "prefill_factor": round(prefill_factor, 4),
                "memory_factor": round(memory_factor, 4),
                "length_factor": round(length_factor, 4),
                "raw_capacity": round(raw_capacity, 4),
            },
        )


class OfflineAutoResearchRunner:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        family_id: str,
        output_root: str | Path,
        workload: SyntheticWorkloadDistribution,
        baseline_bundle: TunedConfigBundle | None = None,
        weight_version_id: str | None = None,
        iteration_cap: int = 12,
        wall_clock_seconds: float = 4 * 60 * 60,
        diminishing_returns_window: int = 8,
        diminishing_returns_threshold: float = 0.02,
        candidate_overrides: list[dict[str, Any]] | None = None,
    ) -> None:
        self.model_config = model_config
        self.family_id = family_id
        self.output_root = Path(output_root)
        self.workload = workload
        self.baseline_bundle = baseline_bundle
        self.weight_version_id = weight_version_id or default_weight_version_id(model_config)
        self.iteration_cap = iteration_cap
        self.wall_clock_seconds = wall_clock_seconds
        self.diminishing_returns_window = diminishing_returns_window
        self.diminishing_returns_threshold = diminishing_returns_threshold
        self.candidate_overrides = candidate_overrides
        self.harness = SyntheticMeasurementHarness(workload)

    def run(self) -> OfflineAutoResearchResult:
        run_id = f"run_{int(time.time())}_{self.model_config.model_id.replace('.', '_')}"
        run_dir = self.output_root / self.family_id / self.weight_version_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        measurement_trace_path = run_dir / "measurement_trace.json"
        search_trace_path = run_dir / "search_trace.json"
        run_log_path = run_dir / "run_log.json"
        started_at = time.monotonic()

        baseline_candidate = (
            dict(self.baseline_bundle.vllm_config)
            if self.baseline_bundle is not None
            else self._baseline_candidate()
        )
        baseline_eval = self.harness.evaluate(baseline_candidate, iteration=0, label="baseline")
        baseline_value = baseline_eval.objective_value

        search_trace: list[dict[str, Any]] = [baseline_eval.as_dict()]
        measurement_trace: list[dict[str, Any]] = [baseline_eval.as_dict()]
        best_eval = baseline_eval
        if not baseline_eval.feasible:
            best_eval = CandidateEvaluation(
                iteration=0,
                label="baseline",
                candidate=baseline_candidate,
                feasible=False,
                objective_value=0,
                p95_latency_ms=baseline_eval.p95_latency_ms,
                rollout_throughput=baseline_eval.rollout_throughput,
                determinism_pass_rate=baseline_eval.determinism_pass_rate,
                kv_probe_passed=baseline_eval.kv_probe_passed,
                oom=baseline_eval.oom,
                reason=baseline_eval.reason,
                metrics=baseline_eval.metrics,
            )

        stopping_reason = "iteration_cap"
        infeasible_oom_streak = 0
        determinism_failures = 0
        best_history: list[int] = [best_eval.objective_value]

        for index, candidate in enumerate(self._candidate_plan(), start=1):
            if index > self.iteration_cap:
                break
            if time.monotonic() - started_at > self.wall_clock_seconds:
                stopping_reason = "wall_clock_cap"
                break

            evaluation = self.harness.evaluate(candidate, iteration=index, label=f"candidate-{index:02d}")
            search_trace.append(evaluation.as_dict())
            measurement_trace.append(evaluation.as_dict())

            if evaluation.oom:
                infeasible_oom_streak += 1
                if infeasible_oom_streak >= 3:
                    stopping_reason = "hard_infeasibility_oom"
                    break
                continue
            infeasible_oom_streak = 0

            if evaluation.determinism_pass_rate < 0.999:
                determinism_failures += 1
                if determinism_failures >= 3:
                    stopping_reason = "hard_infeasibility_determinism"
                    break
                continue

            determinism_failures = 0
            if evaluation.feasible and evaluation.objective_value > best_eval.objective_value:
                best_eval = evaluation
            best_history.append(best_eval.objective_value)

            if len(best_history) >= self.diminishing_returns_window:
                window = best_history[-self.diminishing_returns_window :]
                window_start = max(window[0], 1)
                improvement = (window[-1] - window[0]) / window_start
                if improvement < self.diminishing_returns_threshold:
                    stopping_reason = "diminishing_returns"
                    break
        else:
            stopping_reason = "iteration_cap"

        search_trace_path.write_text(json.dumps(search_trace, indent=2), encoding="utf-8")
        measurement_trace_path.write_text(json.dumps(measurement_trace, indent=2), encoding="utf-8")

        bundle_path: Path | None = None
        status = "retained_baseline"
        if best_eval.feasible and best_eval.objective_value > baseline_value:
            bundle = make_tuned_config_bundle(
                model_id=self.model_config.model_id,
                family_id=self.family_id,
                weight_version_id=self.weight_version_id,
                workload_distribution_id=self.workload.workload_distribution_id,
                vllm_config=dict(best_eval.candidate),
                objective={
                    "metric": "sustained_concurrent_eval_threads_at_L_ceiling",
                    "value": best_eval.objective_value,
                    "L_ceiling_ms": self.workload.latency_ceiling_ms,
                    "measurement_window_minutes": self.workload.measurement_window_minutes,
                },
                measurement_trace_ref=str(measurement_trace_path),
                search_trace_ref=str(search_trace_path),
                baseline_bundle_id=self.baseline_bundle.bundle_id if self.baseline_bundle is not None else None,
                regression_guard={
                    "baseline_value": baseline_value,
                    "delta": best_eval.objective_value - baseline_value,
                },
                safety_rails={
                    "compute_budget_cap_respected": True,
                    "regression_guard_passed": True,
                    "determinism_check_passed": True,
                    "oom_streak_abort_triggered": stopping_reason == "hard_infeasibility_oom",
                    "kv_cache_poisoning_check_passed": best_eval.kv_probe_passed,
                    "rollback_path_available": True,
                },
            )
            bundle_path = persist_tuned_config_bundle(bundle, self.output_root)
            status = "produced_bundle"
        run_log_path.write_text(
            json.dumps(
                {
                    "status": status,
                    "stopping_reason": stopping_reason,
                    "baseline_value": baseline_value,
                    "best_value": best_eval.objective_value,
                    "best_candidate_label": best_eval.label,
                    "weight_version_id": self.weight_version_id,
                    "model_id": self.model_config.model_id,
                    "family_id": self.family_id,
                    "bundle_path": str(bundle_path) if bundle_path is not None else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return OfflineAutoResearchResult(
            status=status,
            stopping_reason=stopping_reason,
            run_dir=run_dir,
            search_trace_path=search_trace_path,
            measurement_trace_path=measurement_trace_path,
            run_log_path=run_log_path,
            bundle_path=bundle_path,
            baseline_value=baseline_value,
            best_value=best_eval.objective_value,
            best_candidate_label=best_eval.label,
        )

    def _baseline_candidate(self) -> dict[str, Any]:
        return self._apply_workload_caps(self.model_config.vllm_config())

    def _candidate_plan(self) -> list[dict[str, Any]]:
        if self.candidate_overrides is not None:
            return [dict(candidate) for candidate in self.candidate_overrides]

        base = self._baseline_candidate()
        candidates = self._memory_constrained_candidates(base) + [
            {
                **base,
                "max_num_seqs": min(base["max_num_seqs"] * 2, 64),
                "max_num_batched_tokens": min(base["max_num_batched_tokens"] + 4096, 16384),
                "gpu_memory_utilization": self._clamp_gpu_memory_utilization(
                    min(round(base["gpu_memory_utilization"] + 0.03, 2), 0.95)
                ),
            },
            {
                **base,
                "max_num_seqs": min(base["max_num_seqs"] + 4, 64),
                "max_num_batched_tokens": min(base["max_num_batched_tokens"] + 2048, 16384),
            },
            {
                **base,
                "enable_chunked_prefill": False,
            },
            {
                **base,
                "enable_prefix_caching": False,
            },
            {
                **base,
                "gpu_memory_utilization": self._clamp_gpu_memory_utilization(
                    min(round(base["gpu_memory_utilization"] + 0.01, 2), 0.95)
                ),
                "max_num_batched_tokens": min(base["max_num_batched_tokens"] + 1024, 16384),
            },
            {
                **base,
                "max_num_seqs": min(base["max_num_seqs"] + 8, 64),
                "max_model_len": max(self.workload.p99_context_tokens + 2048, min(base["max_model_len"], 131072)),
            },
        ]
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = self._apply_workload_caps(candidate)
            key = json.dumps(normalized, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique.append(normalized)
        return unique

    def _clamp_gpu_memory_utilization(self, value: float) -> float:
        cap = self.workload.gpu_memory_utilization_cap
        # Sprint 0 is locked to the real Qwen 27B vLLM startup path, so do not
        # emit tuned bundles below the observed live startup floor even when the
        # synthetic workload cap is more aggressive.
        if cap is None:
            return max(value, MIN_LIVE_STARTUP_GPU_MEMORY_UTILIZATION)
        return max(min(value, cap), MIN_LIVE_STARTUP_GPU_MEMORY_UTILIZATION)

    def _apply_workload_caps(self, candidate: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(candidate)
        normalized["gpu_memory_utilization"] = self._clamp_gpu_memory_utilization(
            float(normalized["gpu_memory_utilization"])
        )
        return normalized

    def _memory_constrained_candidates(self, base: dict[str, Any]) -> list[dict[str, Any]]:
        cap = self.workload.gpu_memory_utilization_cap
        if cap is None or cap > 0.2:
            return []
        constrained_max_model_len = max(self.workload.p99_context_tokens + 2048, min(base["max_model_len"], 32768))
        return [
            {
                **base,
                "gpu_memory_utilization": cap,
                "max_num_seqs": 1,
                "max_num_batched_tokens": min(base["max_num_batched_tokens"], max((self.workload.avg_prompt_tokens * 3) // 2, 6144)),
                "enable_prefix_caching": False,
                "enable_chunked_prefill": True,
                "max_model_len": constrained_max_model_len,
            },
            {
                **base,
                "gpu_memory_utilization": cap,
                "max_num_seqs": 1,
                "max_num_batched_tokens": min(base["max_num_batched_tokens"], max(self.workload.avg_prompt_tokens, 4096)),
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "max_model_len": constrained_max_model_len,
            },
        ]


def load_baseline_bundle(path: str | Path | None) -> TunedConfigBundle | None:
    if path is None:
        return None
    from .tuned_config import load_tuned_config_bundle

    return load_tuned_config_bundle(path)

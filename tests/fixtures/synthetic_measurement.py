from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SyntheticMeasurementFixture:
    workload: Any

    VERSION: str = "SyntheticMeasurementFixture v0.1.0"

    def evaluate(
        self,
        candidate: dict[str, Any],
        *,
        iteration: int,
        label: str,
        profile: str = "screen",
    ) -> dict[str, Any]:
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
        if bool(overrides.get("force_oom")) or memory_load > gpu_memory_utilization + 0.05:
            feasible = False
            failures = ["oom"]
            eval_throughput = 0.0
            rollout_throughput = 0.0
            ttft = float(self.workload.latency_ceiling_ms) * 1.4
        else:
            batch_factor = min(max_num_batched_tokens / max(self.workload.avg_prompt_tokens * 2, 1), 2.5)
            cache_factor = 1.15 if enable_prefix_caching else 0.8
            prefill_factor = 1.1 if enable_chunked_prefill else 0.9
            memory_factor = max(0.2, 0.75 + ((gpu_memory_utilization - 0.70) * 1.5))
            length_factor = 1.0 if max_model_len >= self.workload.p99_context_tokens + 2048 else 0.5
            raw_capacity = max_num_seqs * batch_factor * cache_factor * prefill_factor * memory_factor * length_factor
            eval_throughput = max(0.0, min(float(max_num_seqs), raw_capacity / 2.0))
            rollout_throughput = 8.0 * batch_factor * cache_factor * prefill_factor
            ttft = self.workload.latency_ceiling_ms * (
                0.38
                + (max_num_seqs / 40.0)
                + (max_model_len / 262144.0)
                + (0.12 * (1.0 - (batch_factor / 2.5)))
                + (0.06 if not enable_chunked_prefill else -0.03)
                + (0.05 if not enable_prefix_caching else -0.02)
                - ((gpu_memory_utilization - 0.70) * 0.15)
            )
            failures: list[str] = []
            if bool(overrides.get("inject_nondeterminism")):
                failures.append("determinism")
            if bool(overrides.get("inject_kv_poisoning")):
                failures.append("purity")
            feasible = not failures and eval_throughput >= 0.0

        determinism_pass_rate = 0.95 if "determinism" in failures else 1.0
        reasoning_purity = 0.99 if "purity" in failures else 1.0
        turn_latency = ttft * 1.2
        tpot = ttft / max(int(self.workload.avg_output_tokens), 1)
        target_concurrency = max(max_num_seqs, 1)
        return {
            "generator": self.VERSION,
            "candidate_label": label,
            "candidate_vllm_config": dict(candidate),
            "resolved": {
                "attention_backend": "synthetic",
                "deltanet_kernel": "synthetic",
                "torch_compile_mode": "default",
            },
            "cache_isolation": {
                "cache_salt": "",
                "prefix_cache_reset_at_bootstrap": True,
                "first_10_req_prefix_cache_hit_rate": 0.0,
                "last_10_req_prefix_cache_hit_rate": 0.75,
            },
            "windows": {
                "warmup_s": 300 if profile == "full" else 120,
                "measurement_s": 1500 if profile == "full" else 600,
            },
            "per_request_latencies": [
                {
                    "req_id": f"synthetic-{iteration:03d}",
                    "ttft_ms": round(ttft, 3),
                    "tpot_ms": round(tpot, 3),
                    "turn_latency_ms": round(turn_latency, 3),
                    "thinking_tokens": 0,
                    "response_tokens": self.workload.avg_output_tokens,
                    "concurrency_when_dispatched": target_concurrency,
                }
            ],
            "diagnostics": {
                "ttft_p95_ms": {"driver": round(ttft, 3), "promql": round(ttft, 3), "delta_pct": 0.0},
                "tpot_p95_ms": {"driver": round(tpot, 3), "promql": round(tpot, 3), "delta_pct": 0.0},
                "turn_latency_p95_ms": {
                    "driver": round(turn_latency, 3),
                    "promql": round(turn_latency, 3),
                    "delta_pct": 0.0,
                },
                "rollout_throughput": round(rollout_throughput, 3),
                "target_concurrency": target_concurrency,
            },
            "ttft_p95_ms": {"driver": round(ttft, 3), "promql": round(ttft, 3), "delta_pct": 0.0},
            "tpot_p95_ms": {"driver": round(tpot, 3), "promql": round(tpot, 3), "delta_pct": 0.0},
            "turn_latency_p95_ms": {
                "driver": round(turn_latency, 3),
                "promql": round(turn_latency, 3),
                "delta_pct": 0.0,
            },
            "sustained_concurrency": target_concurrency,
            "eval_throughput": round(eval_throughput, 6),
            "rollout_throughput": round(rollout_throughput, 3),
            "window_completed": True,
            "reasoning_content_purity": reasoning_purity,
            "determinism_pass_rate": determinism_pass_rate,
            "no_oom_events": "oom" not in failures,
            "feasible": feasible,
            "feasibility_failures": failures,
            "vllm_metrics_snapshot_ref": "",
            "seed_trace_replay_ref": "",
        }

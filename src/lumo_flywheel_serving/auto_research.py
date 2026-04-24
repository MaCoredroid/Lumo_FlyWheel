from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
import yaml

from .measurement_harness import RealMeasurementHarness, SLO, WorkloadSpec
from .registry import ModelConfig, load_registry
from .tuned_config import TunedConfigBundle, default_weight_version_id, make_tuned_config_bundle, persist_tuned_config_bundle
from .yaml_utils import load_yaml_file

MIN_LIVE_STARTUP_GPU_MEMORY_UTILIZATION = 0.30
SIGNED_OFF_BY = "lumoserve-auto-research-cli <auto-research@lumo-flywheel>"
MIN_CODEX_CLI_VERSION = (0, 120, 0)
ALLOWED_VLLM_CONFIG_KEYS = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "max_model_len",
    "kv_cache_dtype",
}
PRODUCTION_AUTO_RESEARCH_SUBCOMMANDS = (
    "bootstrap-round",
    "measure",
    "commit-candidate",
    "rescreen",
    "validate-holdout",
    "finalize-round",
    "status",
    "run-round",
)
AUTO_RESEARCH_HELP_SUBCOMMANDS = PRODUCTION_AUTO_RESEARCH_SUBCOMMANDS + ("run",)
RESULTS_COLUMNS = [
    "candidate_uuid",
    "parent_candidate_uuid",
    "iteration",
    "candidate_label",
    "feasible",
    "eval_throughput",
    "objective_mean",
    "objective_ci_95",
    "measurement_count",
    "window_completed",
    "no_oom_events",
    "reasoning_content_purity",
    "determinism_pass_rate",
    "status",
    "notes",
]
ITERATION_ID_RE = re.compile(r"^(\d{3}|baseline_[ab]|rescreen_\d{2})$")

IMPL_BRIEF_TEMPLATE = """# IMPL Brief — Auto-Research Substrate (LLD-SB-06)

You are the implementation agent. Your job is to deliver the substrate
the v0.1 auto-research round will run on top of. This is a one-shot
implementation task, not a research loop.

## Context docs (read all three first)

- Parent HLD:  docs/HLD-Serving-Backend-AutoResearch-v0_1.md
- Sub-spec:    docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md
- Parent §5.6 (measurement harness), §5.9 (bundle schema), §9.3 (verification)

## Deliverables (all must land on main)

1. src/lumo_flywheel_serving/measurement_harness.py
   - class RealMeasurementHarness per sub-spec §9.1
   - measure() method per §9.1 signature
   - emits MeasuredTrace per §9.2 schema with generator =
     "RealMeasurementHarness v0.1.0"
   - implements the parent §5.6 loop: /admin/load_tuned_config,
     /health wait, seed-trace replay, per-request latency capture,
     /metrics scrape at window boundaries, PromQL-derived p95
     cross-check, purity sample, determinism probe, KV-poisoning probe

2. scripts/capture_seed_workload.py
   - runs family eval set through default-config serving stack once
   - persists per-request jsonl: prompt_tokens, output_tokens,
     thinking_tokens, turn_index
   - emits workload_distribution_id = sha256 of the persisted file

3. CLI subcommands under `lumoserve auto-research …` — all 8 required
   for Phase A completion, plus the backward-compat `run`:
   - bootstrap-round   (sub-spec §8.1)
   - measure           (sub-spec §8.2 — --harness real|synthetic)
   - commit-candidate  (sub-spec §8.3 — --harness real|synthetic)
   - rescreen          (sub-spec §8.4 — required by finalize-round)
   - validate-holdout  (sub-spec §8.5 — required by finalize-round)
   - finalize-round    (sub-spec §8.6 — refuses without rescreen + holdout
                         unless --dry-run is passed, §8.6a)
   - status            (sub-spec §8.7 — read-only round state for Python)
   - run-round         (sub-spec §8.8 — Python outer loop command)
   Existing `run` subcommand stays but is env-guarded per §8.9.

4. skills/auto-research-round-manager/SKILL.md — full rewrite
   - Python outer loop per sub-spec §11
   - spawns `codex exec` per iteration (sub-spec §2.3)
   - owns stop criteria (sub-spec §11.3)
   - calls bootstrap-round, loop-of-codex-exec, finalize-round,
     live family gate in that order

5. tests/fixtures/synthetic_measurement.py
   - move SyntheticMeasurementHarness here, rename to
     SyntheticMeasurementFixture, emit generator =
     "SyntheticMeasurementFixture v<n>"
   - commit-candidate must REFUSE this generator in real mode and accept
     it only in synthetic fixture mode per sub-spec §8.3

6. Unit + integration tests:
   - unit: each CLI subcommand
   - unit: skill watchdog paths (silence, out-of-scope write,
           unsigned commit)
   - integration: dry-run round against SyntheticMeasurementFixture
                  (allowed only under LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT=1)
   - integration: precondition refuses when harness module absent

7. Pre-flight checks for the skill (sub-spec §11.1):
   - RealMeasurementHarness imports cleanly
   - codex --version returns expected version
   - git status clean
   - workload yaml has seed_trace_ref pointing at existing jsonl
   - LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT unset

8. Codex-facing brief templates (strings in the skill):
   - impl_brief.md   (this file — you may update if you discover
                       the spec is wrong; note the update in §14)
   - iteration_brief.md (sub-spec §5.2 template — ship verbatim)

9. src/lumo_flywheel_serving/round_driver.py
   - RoundContext, RoundResult, run_round(ctx), and restore_worktree_head()
   - exposed through `lumoserve auto-research run-round`

## Done when

- All 9 items above land on main
- All unit + integration tests pass
- A dry-run round against SyntheticMeasurementFixture completes
  successfully end-to-end (demonstrates the wiring is correct,
  does not prove the real harness works)
- `python -c "from lumo_flywheel_serving.measurement_harness \
   import RealMeasurementHarness"` succeeds
- sub-spec §9.3.AR.7 and §9.3.AR.12 verification items pass

## You may

- install packages and add dependencies to pyproject.toml
  (Phase A is the only phase where this is allowed)
- modify any file in the repo
- create new files under src/, scripts/, skills/, tests/
- refactor existing code that conflicts with the new surface

## You may not

- ship a Phase A deliverable that calls SyntheticMeasurementFixture
  from production code paths
- modify docs/ without updating the corresponding sub-spec section
- leave any test failing
- declare done without running the dry-run round end-to-end

## Exit protocol

Open one PR with all 8 deliverables. Title:
  "Phase A: auto-research substrate (LLD-SB-06)"
Body: checklist from "Done when" above, all items checked.
"""

ITERATION_BRIEF_TEMPLATE = """# Auto-Research Iteration {{iteration}} of Round {{round_id}}

You are running ONE iteration of an auto-research round. You are not
running the round. Python is running the round and will spawn your
successor when you exit cleanly.

## Round identity (read-only — DO NOT edit)

- round_id:            {{round_id}}
- model_id:            {{model_id}}
- family_id:           {{family_id}}
- active_layer:        {{active_layer}}
- round_branch:        {{round_branch}}
- round_spec_ref:      {{round_dir}}/round_spec.yaml

## This iteration

- iteration:           {{iteration}}          # e.g. "007"
- iteration_dir:       {{round_dir}}/candidates/{{iteration}}/
- prior_results_ref:   {{round_dir}}/results.tsv   # all rows up to {{iteration}}-1

## Your job (exactly four steps — do them in this order)

1. Read {{round_dir}}/round_spec.yaml to understand the throughput objective,
   iteration_cap, and active_layer for this round.

2. Read {{round_dir}}/results.tsv. Look at every prior row. Study the
   pattern of feasible vs infeasible candidates, the stability gate each
   infeasible candidate tripped, and the eval_throughput trend.

3. Propose ONE candidate for this iteration. Write it to:
     {{iteration_dir}}/candidate.yaml
   Schema: parent HLD §5.3.2 L1 action space keys only. No L0, no L2,
   no L3 keys. No extra keys. The baseline case (iteration=000) is
   the default-config dict from the model registry.

4. Invoke:
     lumoserve auto-research measure \
       --round-id {{round_id}} \
       --candidate {{iteration_dir}}/candidate.yaml
   The CLI will:
     - /admin/load_tuned_config with your candidate's vllm_config
     - wait for /health
     - drive RealMeasurementHarness for warmup + measurement window
     - write measurement_trace.json next to candidate.yaml
     - append one row to results.tsv with a stable candidate_uuid
       populated (no commit_sha column — see §7.2)
     - print one JSON object to stdout including {candidate_uuid, ...}
     - exit 0 on success, non-zero with structured error on fault
   Total wall-clock: ~{{per_candidate_wall_clock_minutes}} minutes.

5. Read {{iteration_dir}}/measurement_trace.json. Pick ONE status from
   {keep, discard, crash, baseline, harness_fault}. Then invoke:
     lumoserve auto-research commit-candidate \
       --round-id {{round_id}} \
       --iteration {{iteration}} \
       --status <status> \
       --notes "<one-line rationale grounded in the trace>"
   The CLI will create one git commit with message format §7.3.

6. Exit with code 0.

## Hard rules (sub-spec §6 — verified by watchdog + CLI)

R1. You may write ONLY under {{iteration_dir}}. The CLI rejects other
    paths.
R2. You may NOT modify round_spec.yaml, iteration_brief.md, results.tsv
    (except via the CLI), or anything under src/ docs/ benchmark_blueprints/.
R3. You may NOT call `pip install` or any package-install command.
R4. You may NOT hand-compute objective values. The only source of
    truth is measurement_trace.json.
R5. You may NOT make git commits yourself — only via `commit-candidate`.
R6. You do NOT decide whether the round continues. Exit 0 when this
    iteration is done. Python decides what happens next.
R7. You do NOT call `finalize-round`. That is Python's job when the
    round is done. Calling it yourself is a R2 violation and the
    watchdog will kill the round.
R8. If a CLI call returns non-zero, read the error. Retry at most
    twice. If still failing, write a one-line explanation to
    {{iteration_dir}}/BLOCKED.md and exit with code 2.

## What "done" looks like for this iteration

- {{iteration_dir}}/candidate.yaml exists and is valid
- {{iteration_dir}}/measurement_trace.json exists with
  generator starting with "RealMeasurementHarness"
- One new row in results.tsv with a candidate_uuid column
  populated
- One new commit on {{round_branch}} whose message carries both
  a `Candidate-UUID: <uuid>` trailer (matching the results.tsv
  row) and a `Signed-off-by: lumoserve-auto-research-cli` trailer
- You have exited with code 0

## Out-of-scope for this iteration (Python handles)

- Deciding whether to run iteration {{next_iteration}}
- Detecting diminishing returns across iterations
- Detecting 3-in-a-row OOM hard-infeasibility
- Running the live family gate
- Writing the bundle yaml
- Merging the round branch

## Reference material (read if needed — do not modify)

- Parent HLD:     docs/HLD-Serving-Backend-AutoResearch-v0_1.md
- Sub-spec:       docs/HLD-Serving-Backend-AutoResearch-v0_1-SubSpec-AutoResearchAgent.md
- Workload yaml:  {{workload_file}}
- CLI help:       lumoserve auto-research --help
"""


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
    target_concurrency: int = 4
    gpu_memory_utilization_cap: float | None = None
    seed_trace_ref: str | None = None
    holdout_trace_ref: str | None = None
    tpot_ceiling_ms: int = 80
    turn_latency_ceiling_ms: int = 35000

    @classmethod
    def default_for(cls, *, model_config: ModelConfig, family_id: str) -> "SyntheticWorkloadDistribution":
        payload = {
            "family_id": family_id,
            "p99_context_tokens": min(model_config.max_model_len // 2, 32768),
            "avg_prompt_tokens": min(model_config.max_model_len // 16, 4096),
            "avg_output_tokens": 768,
            "latency_ceiling_ms": 650,
            "rollout_baseline": 14.0,
            "target_concurrency": 4,
            "tpot_ceiling_ms": 80,
            "turn_latency_ceiling_ms": 20000,
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
            target_concurrency=payload["target_concurrency"],
            tpot_ceiling_ms=payload["tpot_ceiling_ms"],
            turn_latency_ceiling_ms=payload["turn_latency_ceiling_ms"],
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
        payload.setdefault("target_concurrency", 4)
        payload.setdefault("tpot_ceiling_ms", 80)
        payload.setdefault("turn_latency_ceiling_ms", int(payload["latency_ceiling_ms"]))
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
            target_concurrency=int(payload["target_concurrency"]),
            measurement_window_minutes=int(payload.get("measurement_window_minutes", 30)),
            gpu_memory_utilization_cap=(
                float(payload["gpu_memory_utilization_cap"])
                if payload.get("gpu_memory_utilization_cap") is not None
                else None
            ),
            seed_trace_ref=(
                str(payload["seed_trace_ref"]).strip() if payload.get("seed_trace_ref") else None
            ),
            holdout_trace_ref=(
                str(payload["holdout_trace_ref"]).strip() if payload.get("holdout_trace_ref") else None
            ),
            tpot_ceiling_ms=int(payload.get("tpot_ceiling_ms", 80)),
            turn_latency_ceiling_ms=int(payload.get("turn_latency_ceiling_ms", payload["latency_ceiling_ms"])),
        )

    def to_workload_spec(self, *, base_dir: Path) -> WorkloadSpec:
        if not self.seed_trace_ref:
            raise ValueError("Workload spec is missing seed_trace_ref")
        seed_trace_path = (base_dir / self.seed_trace_ref).resolve() if not Path(self.seed_trace_ref).is_absolute() else Path(self.seed_trace_ref)
        holdout_path: Path | None = None
        if self.holdout_trace_ref:
            holdout_path = (base_dir / self.holdout_trace_ref).resolve() if not Path(self.holdout_trace_ref).is_absolute() else Path(self.holdout_trace_ref)
        return WorkloadSpec(
            family_id=self.family_id,
            workload_distribution_id=self.workload_distribution_id,
            seed_trace_ref=seed_trace_path,
            holdout_trace_ref=holdout_path,
            latency_ceiling_ms=self.latency_ceiling_ms,
            tpot_ceiling_ms=self.tpot_ceiling_ms,
            turn_latency_ceiling_ms=self.turn_latency_ceiling_ms,
            avg_prompt_tokens=self.avg_prompt_tokens,
            avg_output_tokens=self.avg_output_tokens,
            measurement_window_minutes=self.measurement_window_minutes,
            rollout_baseline=self.rollout_baseline,
            target_concurrency=self.target_concurrency,
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


class _LegacySyntheticHarness:
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
        if bool(overrides.get("force_oom")) or memory_load > gpu_memory_utilization + 0.05:
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
        determinism_pass_rate = 0.95 if overrides.get("inject_nondeterminism") else 1.0
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
        self.harness = _LegacySyntheticHarness(workload)

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
        best_eval = baseline_eval if baseline_eval.feasible else CandidateEvaluation(
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


@dataclass(frozen=True)
class ResultsRow:
    candidate_uuid: str
    parent_candidate_uuid: str
    iteration: str
    candidate_label: str
    feasible: bool
    eval_throughput: str
    objective_mean: str
    objective_ci_95: str
    measurement_count: int
    window_completed: bool
    no_oom_events: bool
    reasoning_content_purity: str
    determinism_pass_rate: str
    status: str
    notes: str

    @property
    def objective_value(self) -> str:
        return self.eval_throughput

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_uuid": self.candidate_uuid,
            "parent_candidate_uuid": self.parent_candidate_uuid,
            "iteration": self.iteration,
            "candidate_label": self.candidate_label,
            "feasible": "true" if self.feasible else "false",
            "eval_throughput": self.eval_throughput,
            "objective_mean": self.objective_mean,
            "objective_ci_95": self.objective_ci_95,
            "measurement_count": str(self.measurement_count),
            "window_completed": "true" if self.window_completed else "false",
            "no_oom_events": "true" if self.no_oom_events else "false",
            "reasoning_content_purity": self.reasoning_content_purity,
            "determinism_pass_rate": self.determinism_pass_rate,
            "status": self.status,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> "ResultsRow":
        return cls(
            candidate_uuid=payload.get("candidate_uuid", ""),
            parent_candidate_uuid=payload.get("parent_candidate_uuid", ""),
            iteration=payload.get("iteration", ""),
            candidate_label=payload.get("candidate_label", ""),
            feasible=payload.get("feasible", "").lower() == "true",
            eval_throughput=payload.get("eval_throughput", payload.get("objective_value", "")),
            objective_mean=payload.get("objective_mean", ""),
            objective_ci_95=payload.get("objective_ci_95", ""),
            measurement_count=int(payload.get("measurement_count", "0") or 0),
            window_completed=payload.get("window_completed", "true").lower() == "true",
            no_oom_events=payload.get("no_oom_events", "true").lower() == "true",
            reasoning_content_purity=payload.get("reasoning_content_purity", ""),
            determinism_pass_rate=payload.get("determinism_pass_rate", ""),
            status=payload.get("status", ""),
            notes=payload.get("notes", ""),
        )


@dataclass(frozen=True)
class RoundSpecRecord:
    round_id: str
    round_root: str
    round_dir: str
    worktree_path: str
    round_branch: str
    model_id: str
    family_id: str
    sprint: str
    weight_version_id: str
    workload_file: str
    workload_distribution_id: str
    latency_ceiling_ms: int
    tpot_ceiling_ms: int
    turn_latency_ceiling_ms: int
    active_layer: str = "L1"
    target_concurrency: int = 4
    iteration_cap: int = 12
    rescreen_top_k: int = 3
    round_wall_clock_s: int = 8 * 60 * 60
    screen_warmup_s: int = 120
    screen_measurement_s: int = 600
    full_warmup_s: int = 300
    full_measurement_s: int = 1500
    per_iteration_codex_wall_clock_s: int = 45 * 60
    diminishing_returns_window_k: int = 4
    noise_floor: float = 0.0
    parent_head_sha: str = ""
    harness_type: str = "real"
    round_started_at: float = 0.0
    sub_spec_version: str = "v0.1.11"

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_root": self.round_root,
            "round_dir": self.round_dir,
            "worktree_path": self.worktree_path,
            "round_branch": self.round_branch,
            "model_id": self.model_id,
            "family_id": self.family_id,
            "sprint": self.sprint,
            "weight_version_id": self.weight_version_id,
            "workload_file": self.workload_file,
            "workload_distribution_id": self.workload_distribution_id,
            "latency_ceiling_ms": self.latency_ceiling_ms,
            "tpot_ceiling_ms": self.tpot_ceiling_ms,
            "turn_latency_ceiling_ms": self.turn_latency_ceiling_ms,
            "active_layer": self.active_layer,
            "target_concurrency": self.target_concurrency,
            "iteration_cap": self.iteration_cap,
            "rescreen_top_k": self.rescreen_top_k,
            "round_wall_clock_s": self.round_wall_clock_s,
            "screen_warmup_s": self.screen_warmup_s,
            "screen_measurement_s": self.screen_measurement_s,
            "full_warmup_s": self.full_warmup_s,
            "full_measurement_s": self.full_measurement_s,
            "screen_profile_s": self.screen_warmup_s + self.screen_measurement_s + 180,
            "full_profile_s": self.full_warmup_s + self.full_measurement_s + 180,
            "per_iteration_codex_wall_clock_s": self.per_iteration_codex_wall_clock_s,
            "diminishing_returns_window_k": self.diminishing_returns_window_k,
            "noise_floor": self.noise_floor,
            "parent_head_sha": self.parent_head_sha,
            "harness_type": self.harness_type,
            "round_started_at": self.round_started_at,
            "sub_spec_version": self.sub_spec_version,
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "RoundSpecRecord":
        payload = load_yaml_file(path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid round spec: {path}")
        return cls(
            round_id=str(payload["round_id"]),
            round_root=str(payload["round_root"]),
            round_dir=str(payload["round_dir"]),
            worktree_path=str(payload.get("worktree_path", payload["round_dir"])),
            round_branch=str(payload["round_branch"]),
            model_id=str(payload["model_id"]),
            family_id=str(payload["family_id"]),
            sprint=str(payload["sprint"]),
            weight_version_id=str(payload["weight_version_id"]),
            workload_file=str(payload["workload_file"]),
            workload_distribution_id=str(payload["workload_distribution_id"]),
            latency_ceiling_ms=int(payload["latency_ceiling_ms"]),
            tpot_ceiling_ms=int(payload["tpot_ceiling_ms"]),
            turn_latency_ceiling_ms=int(payload["turn_latency_ceiling_ms"]),
            active_layer=str(payload.get("active_layer", "L1")),
            target_concurrency=int(payload.get("target_concurrency", 4)),
            iteration_cap=int(payload.get("iteration_cap", 12)),
            rescreen_top_k=int(payload.get("rescreen_top_k", 3)),
            round_wall_clock_s=int(payload.get("round_wall_clock_s", 8 * 60 * 60)),
            screen_warmup_s=int(payload.get("screen_warmup_s", 120)),
            screen_measurement_s=int(payload.get("screen_measurement_s", 600)),
            full_warmup_s=int(payload.get("full_warmup_s", 300)),
            full_measurement_s=int(payload.get("full_measurement_s", 1500)),
            per_iteration_codex_wall_clock_s=int(payload.get("per_iteration_codex_wall_clock_s", 45 * 60)),
            diminishing_returns_window_k=int(payload.get("diminishing_returns_window_k", 4)),
            noise_floor=float(payload.get("noise_floor", 0.0)),
            parent_head_sha=str(payload.get("parent_head_sha", "")),
            harness_type=str(payload.get("harness_type", "real")),
            round_started_at=float(payload.get("round_started_at", 0.0)),
            sub_spec_version=str(payload.get("sub_spec_version", "v0.1.11")),
        )


class AutoResearchRoundManager:
    def __init__(
        self,
        *,
        registry_path: str | Path,
        repo_root: str | Path,
        tuned_config_root: str | Path,
        port: int = 8000,
        proxy_port: int = 8001,
    ) -> None:
        self.registry_path = Path(registry_path).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.tuned_config_root = Path(tuned_config_root).resolve()
        self.port = port
        self.proxy_port = proxy_port

    def bootstrap_round(
        self,
        *,
        model_id: str,
        family_id: str,
        sprint: str,
        workload_file: str | Path,
        weight_version_id: str | None,
        round_root: str | Path,
        harness_type: str = "real",
        skip_preflight: bool = False,
    ) -> dict[str, Any]:
        workload_file = Path(workload_file).resolve()
        round_root = Path(round_root).resolve()
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id: {model_id}")
        model_config = registry[model_id]
        workload = SyntheticWorkloadDistribution.from_file(workload_file, model_config=model_config, family_id=family_id)
        if not workload.seed_trace_ref:
            raise RuntimeError("Workload file is missing seed_trace_ref")
        seed_trace_path = workload_file.parent / workload.seed_trace_ref
        if not seed_trace_path.is_file():
            raise RuntimeError(f"Seed trace file does not exist: {seed_trace_path}")
        if harness_type != "synthetic" and not skip_preflight:
            self._run_bootstrap_preflight(
                model_config=model_config,
                family_id=family_id,
                weight_version_id=weight_version_id,
                workload=workload,
            )

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        round_id = f"{model_id}-{family_id}-{sprint}-{timestamp}"
        round_branch = self._round_branch_name(
            model_id=model_id,
            family_id=family_id,
            sprint=sprint,
            timestamp=timestamp,
        )
        round_dir = round_root / round_id
        if round_dir.exists():
            raise RuntimeError(f"Round directory already exists: {round_dir}")

        parent_head_sha = self._git(["rev-parse", "HEAD"]).stdout.strip()
        self._create_branch_ref(round_branch=round_branch, start_point=parent_head_sha)

        round_dir.mkdir(parents=True)
        candidates_dir = round_dir / "candidates"
        candidates_dir.mkdir()

        spec = RoundSpecRecord(
            round_id=round_id,
            round_root=str(round_root),
            round_dir=str(round_dir),
            worktree_path=str(round_dir),
            round_branch=round_branch,
            model_id=model_id,
            family_id=family_id,
            sprint=sprint,
            weight_version_id=weight_version_id or default_weight_version_id(model_config),
            workload_file=str(workload_file),
            workload_distribution_id=workload.workload_distribution_id,
            latency_ceiling_ms=workload.latency_ceiling_ms,
            tpot_ceiling_ms=workload.tpot_ceiling_ms,
            turn_latency_ceiling_ms=workload.turn_latency_ceiling_ms,
            target_concurrency=workload.target_concurrency,
            parent_head_sha=parent_head_sha,
            harness_type=harness_type,
            round_started_at=time.time(),
        )
        self._write_yaml(round_dir / "round_spec.yaml", spec.as_dict())
        (round_dir / "impl_brief.md").write_text(IMPL_BRIEF_TEMPLATE, encoding="utf-8")
        (round_dir / "iteration_brief.md").write_text(ITERATION_BRIEF_TEMPLATE, encoding="utf-8")
        self._write_results(round_dir / "results.tsv", [])
        (round_dir / ".round.lock").write_text(json.dumps({"round_id": round_id, "created_at": time.time()}), encoding="utf-8")

        default_candidate = model_config.vllm_config()
        for suffix in ("a", "b"):
            baseline_dir = candidates_dir / f"baseline_{suffix}"
            baseline_dir.mkdir()
            self._write_yaml(baseline_dir / "candidate.yaml", default_candidate)

        if harness_type != "synthetic":
            staged_paths = [path.relative_to(self.repo_root) for path in self._bootstrap_round_artifact_paths(round_dir)]
            bootstrap_message = (
                f"AR({round_id}) BOOTSTRAP\n\n"
                f"round_branch={round_branch} model_id={model_id} family_id={family_id} sprint={sprint}\n\n"
                "Bootstrap: true\n"
                f"Signed-off-by: {SIGNED_OFF_BY}\n"
            )
            self._commit_paths(
                staged_paths,
                bootstrap_message,
                False,
                branch=round_branch,
                context="bootstrap-round refuses",
            )

        return {
            "round_id": round_id,
            "round_dir": str(round_dir),
            "worktree_path": str(round_dir),
            "round_branch": round_branch,
            "round_spec_path": str(round_dir / "round_spec.yaml"),
        }

    def measure(
        self,
        *,
        round_id: str,
        candidate_path: str | Path,
        profile: str | None = None,
        parent_candidate_uuid: str | None = None,
        harness: str | None = None,
    ) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        spec = self._spec_with_harness(spec, harness)
        candidate_path = Path(candidate_path).resolve()
        if not candidate_path.is_file():
            raise RuntimeError(f"Candidate file does not exist: {candidate_path}")
        if round_dir not in candidate_path.parents:
            raise RuntimeError("Candidate must live under the round directory")
        iteration_id = candidate_path.parent.name
        self._validate_iteration_id(iteration_id)
        selected_profile = profile or ("full" if iteration_id.startswith("rescreen_") else "screen")
        self._validate_measure_request(
            spec=spec,
            iteration_id=iteration_id,
            profile=selected_profile,
        )
        candidate = load_yaml_file(candidate_path)
        if not isinstance(candidate, dict):
            raise RuntimeError(f"Candidate yaml must be a mapping: {candidate_path}")
        allowed_extra_keys = {"harness_overrides"} if spec.harness_type == "synthetic" else set()
        unknown_keys = sorted(set(candidate) - ALLOWED_VLLM_CONFIG_KEYS - allowed_extra_keys)
        if unknown_keys:
            raise RuntimeError(f"Candidate contains unsupported keys: {unknown_keys}")
        vllm_config = {
            key: candidate[key]
            for key in candidate
            if key in ALLOWED_VLLM_CONFIG_KEYS or key in allowed_extra_keys
        }
        rows = self._read_results(round_dir / "results.tsv")
        if any(row.iteration == iteration_id for row in rows):
            raise RuntimeError(f"measure_refused: results row already exists for iteration {iteration_id}")
        candidate_uuid = str(uuid4())
        candidate_label = f"candidate-{iteration_id}" if iteration_id.isdigit() else iteration_id

        workload = SyntheticWorkloadDistribution.from_file(
            spec.workload_file,
            model_config=load_registry(self.registry_path)[spec.model_id],
            family_id=spec.family_id,
        )
        trace = self._run_harness(
            spec=spec,
            workload=workload,
            candidate_vllm_config=vllm_config,
            profile=selected_profile,
        )
        self._normalize_trace(trace)
        promql_mismatch = not self._valid_latency_cross_checks(trace)
        if promql_mismatch:
            warnings = list(trace.get("harness_health_warnings", []))
            if "promql_mismatch" not in warnings:
                warnings.append("promql_mismatch")
            trace["harness_health_warnings"] = warnings

        trace.update(
            {
                "round_id": round_id,
                "iteration": iteration_id,
                "candidate_label": candidate_label,
                "candidate_uuid": candidate_uuid,
                "parent_candidate_uuid": parent_candidate_uuid,
                "profile": selected_profile,
                "candidate_vllm_config": {key: value for key, value in vllm_config.items() if key in ALLOWED_VLLM_CONFIG_KEYS},
            }
        )
        trace["cache_isolation"]["cache_salt"] = candidate_uuid
        trace["cache_isolation"]["prefix_cache_reset_at_bootstrap"] = True

        candidate_dir = candidate_path.parent
        metrics_ref = candidate_dir / "vllm_metrics.prom"
        replay_ref = candidate_dir / "replay.jsonl"
        metrics_ref.write_text("", encoding="utf-8")
        replay_ref.write_text(
            Path(workload.to_workload_spec(base_dir=Path(spec.workload_file).parent).seed_trace_ref).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        trace["vllm_metrics_snapshot_ref"] = str(metrics_ref.relative_to(round_dir))
        trace["seed_trace_replay_ref"] = str(replay_ref.relative_to(round_dir))
        trace_path = candidate_dir / "measurement_trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")

        rows.append(self._trace_to_pending_row(trace))
        self._write_results(round_dir / "results.tsv", rows)
        return {
            "round_id": round_id,
            "iteration": iteration_id,
            "candidate_uuid": candidate_uuid,
            "feasible": bool(trace["feasible"]),
            "eval_throughput": trace.get("eval_throughput"),
            "trace_path": str(trace_path),
            "recommended_status": None,
            "notes": "promql_mismatch" if promql_mismatch else None,
        }

    def commit_candidate(
        self,
        *,
        round_id: str,
        iteration: str,
        status: str,
        notes: str,
        allow_synthetic: bool = False,
        harness: str | None = None,
    ) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        spec = self._spec_with_harness(spec, harness)
        if spec.harness_type == "synthetic":
            allow_synthetic = True
        self._validate_iteration_id(iteration)
        candidate_dir = round_dir / "candidates" / iteration
        candidate_path = candidate_dir / "candidate.yaml"
        trace_path = candidate_dir / "measurement_trace.json"
        if not candidate_path.is_file() or not trace_path.is_file():
            raise RuntimeError(f"Iteration artifacts missing for {iteration}")

        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        self._validate_measurement_trace(trace, status=status)
        generator = str(trace.get("generator", ""))
        if not generator.startswith("RealMeasurementHarness") and not allow_synthetic:
            raise RuntimeError(f"commit_refused: generator {generator!r} is not a production trace")

        rows = self._read_results(round_dir / "results.tsv")
        matched = False
        updated_rows: list[ResultsRow] = []
        for row in rows:
            if row.iteration == iteration and row.candidate_uuid == str(trace.get("candidate_uuid")):
                if row.status.strip():
                    raise RuntimeError(
                        f"commit_refused: results row already finalized for iteration {iteration}"
                    )
                updated_rows.append(
                    ResultsRow(
                        **{
                            **row.__dict__,
                            "status": status,
                            "notes": notes.strip(),
                        }
                    )
                )
                matched = True
            else:
                updated_rows.append(row)
        if not matched:
            raise RuntimeError("commit_refused: pending results row missing or candidate_uuid mismatch")
        staged_paths = [
            path.relative_to(self.repo_root)
            for path in [*self._bootstrap_round_artifact_paths(round_dir), candidate_dir, round_dir / "results.tsv"]
        ]
        self._assert_only_allowed_staged_paths(staged_paths, context="commit_refused")
        self._assert_git_index_clean(context="commit_refused")
        self._assert_immutable_round_artifacts(round_dir, spec=spec, context="commit_refused")
        self._assert_dirty_paths_match_expected(
            mutable_paths=[candidate_dir, round_dir / "results.tsv", round_dir / "round_spec.yaml"],
            bootstrap_paths=self._bootstrap_round_artifact_paths(round_dir),
            context="commit_refused",
        )
        updated_spec = self._updated_spec_with_noise_floor(spec=spec, rows=updated_rows)
        if updated_spec != spec:
            self._write_yaml(round_dir / "round_spec.yaml", updated_spec.as_dict())
        self._write_results(round_dir / "results.tsv", updated_rows)

        commit_message = self._candidate_commit_message(
            round_id=round_id,
            iteration=iteration,
            row=next(row for row in updated_rows if row.iteration == iteration and row.candidate_uuid == str(trace.get("candidate_uuid"))),
            trace_path=trace_path.relative_to(self.repo_root),
            extra_trailers=["Fixture-Mode: true"] if spec.harness_type == "synthetic" else None,
        )
        commit_sha = self._commit_paths(
            staged_paths,
            commit_message,
            spec.harness_type == "synthetic",
            branch=spec.round_branch,
            context="commit_refused",
        )
        return {
            "iteration": iteration,
            "candidate_uuid": str(trace.get("candidate_uuid")),
            "commit_sha": commit_sha,
            "status": status,
        }

    def rescreen(self, *, round_id: str, top_k: int, profile: str = "full", harness: str | None = None) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        spec = self._spec_with_harness(spec, harness)
        rows = self._read_results(round_dir / "results.tsv")
        feasible_rows = [
            row
            for row in rows
            if row.status in {"baseline", "keep"} and row.feasible and row.eval_throughput and not row.parent_candidate_uuid
        ]
        feasible_rows.sort(
            key=lambda row: (
                -self._metric_tie_break_value(row.eval_throughput),
                row.iteration,
                row.candidate_uuid,
            )
        )
        selected = feasible_rows[:top_k]
        rescreen_rows: list[dict[str, Any]] = []
        for index, parent in enumerate(selected, start=1):
            iteration = f"rescreen_{index:02d}"
            rescreen_dir = round_dir / "candidates" / iteration
            staged_paths = [
                path.relative_to(self.repo_root)
                for path in [*self._bootstrap_round_artifact_paths(round_dir), rescreen_dir, round_dir / "results.tsv"]
            ]
            self._assert_only_allowed_staged_paths(staged_paths, context="commit_refused")
            self._assert_git_index_clean(context="commit_refused")
            self._assert_immutable_round_artifacts(round_dir, spec=spec, context="commit_refused")
            self._assert_dirty_paths_match_expected(
                mutable_paths=[rescreen_dir, round_dir / "results.tsv", round_dir / "round_spec.yaml"],
                bootstrap_paths=self._bootstrap_round_artifact_paths(round_dir),
                context="commit_refused",
            )
            rescreen_dir.mkdir(parents=True, exist_ok=False)
            parent_candidate_dir = round_dir / "candidates" / parent.iteration
            shutil.copy2(parent_candidate_dir / "candidate.yaml", rescreen_dir / "candidate.yaml")
            measure_result = self.measure(
                round_id=round_id,
                candidate_path=rescreen_dir / "candidate.yaml",
                profile=profile,
                parent_candidate_uuid=parent.candidate_uuid,
                harness=spec.harness_type,
            )
            trace = json.loads((rescreen_dir / "measurement_trace.json").read_text(encoding="utf-8"))
            self._validate_measurement_trace(trace, status="rescreened")
            generator = str(trace.get("generator", ""))
            if not generator.startswith("RealMeasurementHarness") and spec.harness_type != "synthetic":
                raise RuntimeError(f"commit_refused: generator {generator!r} is not a production trace")
            objective_parent = float(parent.eval_throughput)
            objective_rescreen = float(trace["eval_throughput"])
            objective_mean = (objective_parent + objective_rescreen) / 2.0
            objective_ci_95 = self._ci95([objective_parent, objective_rescreen])
            notes = ""
            if abs(objective_rescreen - objective_parent) > float(spec.noise_floor):
                notes = "inconsistent_rescreen"
            rows = self._read_results(round_dir / "results.tsv")
            updated_rows: list[ResultsRow] = []
            for row in rows:
                if row.iteration == iteration and row.candidate_uuid == measure_result["candidate_uuid"]:
                    updated_rows.append(
                        ResultsRow(
                            candidate_uuid=row.candidate_uuid,
                            parent_candidate_uuid=parent.candidate_uuid,
                            iteration=row.iteration,
                            candidate_label=row.candidate_label,
                            feasible=row.feasible,
                            eval_throughput=row.eval_throughput,
                            objective_mean=f"{objective_mean:.3f}",
                            objective_ci_95=f"{objective_ci_95:.3f}",
                            measurement_count=2,
                            window_completed=row.window_completed,
                            no_oom_events=row.no_oom_events,
                            reasoning_content_purity=row.reasoning_content_purity,
                            determinism_pass_rate=row.determinism_pass_rate,
                            status="rescreened",
                            notes=notes,
                        )
                    )
                else:
                    updated_rows.append(row)
            self._assert_only_allowed_staged_paths(staged_paths, context="commit_refused")
            self._assert_git_index_clean(context="commit_refused")
            self._assert_immutable_round_artifacts(round_dir, spec=spec, context="commit_refused")
            self._assert_dirty_paths_match_expected(
                mutable_paths=[rescreen_dir, round_dir / "results.tsv", round_dir / "round_spec.yaml"],
                bootstrap_paths=self._bootstrap_round_artifact_paths(round_dir),
                context="commit_refused",
            )
            self._write_results(round_dir / "results.tsv", updated_rows)
            commit_message = self._candidate_commit_message(
                round_id=round_id,
                iteration=iteration,
                row=next(row for row in updated_rows if row.iteration == iteration),
                trace_path=(rescreen_dir / "measurement_trace.json").relative_to(self.repo_root),
                extra_trailers=[
                    f"Rescreen-Of-UUID: {parent.candidate_uuid}",
                    *(["Fixture-Mode: true"] if spec.harness_type == "synthetic" else []),
                ],
            )
            commit_sha = self._commit_paths(
                staged_paths,
                commit_message,
                spec.harness_type == "synthetic",
                branch=spec.round_branch,
                context="commit_refused",
            )
            rescreen_rows.append(
                {
                    "iteration": iteration,
                    "candidate_uuid": measure_result["candidate_uuid"],
                    "parent_candidate_uuid": parent.candidate_uuid,
                    "commit_sha": commit_sha,
                }
            )

        trace_path = round_dir / "rescreen_trace.json"
        trace_path.write_text(json.dumps(rescreen_rows, indent=2), encoding="utf-8")
        return {"round_id": round_id, "rescreened": rescreen_rows, "trace_path": str(trace_path)}

    def validate_holdout(self, *, round_id: str, candidate_uuid: str, harness: str | None = None) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        spec = self._spec_with_harness(spec, harness)
        workload = SyntheticWorkloadDistribution.from_file(
            spec.workload_file,
            model_config=load_registry(self.registry_path)[spec.model_id],
            family_id=spec.family_id,
        )
        if not workload.holdout_trace_ref:
            holdout_path = round_dir / "holdout_trace.json"
            if spec.harness_type != "synthetic":
                payload = {"pass": False, "reasons_failed": ["missing_holdout_trace_ref"], "candidate_uuid": candidate_uuid}
                holdout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
            payload = {"pass": True, "reasons_failed": [], "candidate_uuid": candidate_uuid, "harness": "synthetic"}
            holdout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload

        rows = self._read_results(round_dir / "results.tsv")
        row = next((entry for entry in rows if entry.candidate_uuid == candidate_uuid), None)
        if row is None:
            raise RuntimeError(f"Unknown candidate_uuid: {candidate_uuid}")
        target_row = row
        if row.parent_candidate_uuid:
            target_row = next((entry for entry in rows if entry.candidate_uuid == row.parent_candidate_uuid), None)
            if target_row is None:
                raise RuntimeError(
                    f"Unknown parent candidate_uuid for holdout validation: {row.parent_candidate_uuid}"
                )
        candidate_yaml = round_dir / "candidates" / target_row.iteration / "candidate.yaml"
        trace = self._run_harness(
            spec=spec,
            workload=workload,
            candidate_vllm_config=load_yaml_file(candidate_yaml),
            profile="full",
            use_holdout=True,
        )
        passed = bool(trace.get("feasible"))
        payload = {
            "pass": passed,
            "reasons_failed": [] if passed else list(trace.get("feasibility_failures", ["holdout_failed"])),
            "candidate_uuid": target_row.candidate_uuid,
            "harness": spec.harness_type,
            "trace": trace,
        }
        holdout_path = round_dir / "holdout_trace.json"
        holdout_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def finalize_round(self, *, round_id: str, dry_run: bool = False) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        rows = self._read_results(round_dir / "results.tsv")
        if not rows:
            raise RuntimeError("Cannot finalize an empty round")
        self._validate_finalize_generator_consistency(round_dir)
        self._validate_finalize_harness_fault_rows(rows)

        winner_row: ResultsRow | None = None
        winner_parent_uuid = ""
        winner_rescreen_uuid = ""
        if dry_run:
            rescreen_rows = [
                row
                for row in rows
                if row.status == "rescreened" and row.objective_mean and row.notes != "inconsistent_rescreen"
            ]
            if rescreen_rows:
                winner_rescreen = self._pick_winner_row(rescreen_rows, objective_field="objective_mean")
                winner_parent_uuid = winner_rescreen.parent_candidate_uuid
                winner_rescreen_uuid = winner_rescreen.candidate_uuid
                winner_row = next(
                    row for row in rows if row.candidate_uuid == winner_parent_uuid and not row.parent_candidate_uuid
                )
            else:
                feasible_rows = [row for row in rows if row.feasible and row.eval_throughput]
                if not feasible_rows:
                    raise RuntimeError("no_feasible_row_for_dry_run")
                winner_row = self._pick_winner_row(feasible_rows, objective_field="eval_throughput")
                winner_parent_uuid = winner_row.candidate_uuid
        else:
            if not (round_dir / "rescreen_trace.json").is_file():
                raise RuntimeError("finalize-round refuses without rescreen_trace.json")
            rescreen_rows = [
                row
                for row in rows
                if row.status == "rescreened" and row.objective_mean and row.notes != "inconsistent_rescreen"
            ]
            if not rescreen_rows:
                raise RuntimeError("finalize-round refuses without eligible rescreen rows")
            holdout_path = round_dir / "holdout_trace.json"
            if not holdout_path.is_file():
                raise RuntimeError("finalize-round refuses without holdout_trace.json")
            winner_rescreen = self._pick_winner_row(rescreen_rows, objective_field="objective_mean")
            winner_parent_uuid = winner_rescreen.parent_candidate_uuid
            winner_rescreen_uuid = winner_rescreen.candidate_uuid
            winner_row = next(
                row for row in rows if row.candidate_uuid == winner_parent_uuid and not row.parent_candidate_uuid
            )
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
            if not bool(holdout.get("pass")):
                raise RuntimeError("finalize-round refuses because holdout validation failed")
            if str(holdout.get("candidate_uuid") or "") != winner_parent_uuid:
                raise RuntimeError("finalize-round refuses because holdout candidate_uuid does not match the winner")

        if winner_row is None:
            raise RuntimeError("Unable to determine winner")

        measurement_trace_ref = round_dir / "measurement_trace_combined.json"
        search_trace_ref = round_dir / "search_trace.json"
        measurement_trace_combined = []
        for trace_path in sorted((round_dir / "candidates").glob("*/measurement_trace.json")):
            measurement_trace_combined.append(json.loads(trace_path.read_text(encoding="utf-8")))
        search_trace_payload = [row.as_dict() for row in rows]
        bundle = make_tuned_config_bundle(
            model_id=spec.model_id,
            family_id=spec.family_id,
            weight_version_id=spec.weight_version_id,
            workload_distribution_id=spec.workload_distribution_id,
            vllm_config=load_yaml_file(round_dir / "candidates" / winner_row.iteration / "candidate.yaml"),
            objective={
                "metric": "eval_throughput",
                "value": float(winner_row.objective_mean or winner_row.eval_throughput or "0"),
            },
            measurement_trace_ref=str(measurement_trace_ref.relative_to(self.repo_root)),
            search_trace_ref=str(search_trace_ref.relative_to(self.repo_root)),
            baseline_bundle_id=None,
            regression_guard={"noise_floor": spec.noise_floor},
            safety_rails={"regression_guard_passed": True},
            round_provenance={
                "dry_run": dry_run,
                "round_id": round_id,
                "round_branch": spec.round_branch,
                "winner_iteration": winner_row.iteration,
                "winner_candidate_uuid": winner_parent_uuid,
                "winner_rescreen_uuid": winner_rescreen_uuid or None,
                "sub_spec_version": spec.sub_spec_version,
                "agent_session_dir_ref": str((round_dir / "candidates").relative_to(self.repo_root)),
                "agent_model_pin": {"model": "gpt-5.4", "reasoning_effort": "high"},
                "results_tsv_ref": str((round_dir / "results.tsv").relative_to(self.repo_root)),
                "holdout_validation": "skipped" if dry_run else "pass",
            },
        )
        bundle_path = self._bundle_output_path(bundle)
        run_log = {
            "round_id": round_id,
            "winner_iteration": winner_row.iteration,
            "winner_candidate_uuid": winner_parent_uuid,
            "winner_rescreen_uuid": winner_rescreen_uuid or None,
            "bundle_path": str(bundle_path),
            "dry_run": dry_run,
            "feasible_count": sum(1 for row in rows if row.feasible),
            "iterations_total": len(rows),
            "rescreened_count": sum(1 for row in rows if row.status == "rescreened"),
            "holdout_validation": "skipped" if dry_run else "pass",
        }

        commit_message = (
            f"AR({round_id}) FINALIZE: {winner_row.candidate_label} - eval_throughput={winner_row.objective_mean or winner_row.eval_throughput}\n\n"
            f"winner_iteration={winner_row.iteration} winner_candidate_uuid={winner_parent_uuid} winner_rescreen_uuid={winner_rescreen_uuid or ''} bundle={bundle_path}\n"
            f"round_wall_clock_minutes={int((time.time() - spec.round_started_at) / 60)} total_iterations={len(rows)} feasible_count={sum(1 for row in rows if row.feasible)}\n"
            f"rescreened_count={sum(1 for row in rows if row.status == 'rescreened')} holdout_validation={'skipped' if dry_run else 'pass'}\n"
            f"stopping_reason=ok\n\n"
            f"Winner-Candidate-UUID: {winner_parent_uuid}\n"
            f"Signed-off-by: {SIGNED_OFF_BY}\n"
        )
        staged_paths = [
            (round_dir / "run_log.json").relative_to(self.repo_root),
            search_trace_ref.relative_to(self.repo_root),
            measurement_trace_ref.relative_to(self.repo_root),
            bundle_path.relative_to(self.repo_root),
            (round_dir / ".round.lock").relative_to(self.repo_root),
        ]
        if (round_dir / "rescreen_trace.json").exists():
            staged_paths.append((round_dir / "rescreen_trace.json").relative_to(self.repo_root))
        if (round_dir / "holdout_trace.json").exists():
            staged_paths.append((round_dir / "holdout_trace.json").relative_to(self.repo_root))
        self._assert_only_allowed_staged_paths(staged_paths, context="finalize-round refuses")
        self._assert_immutable_round_artifacts(round_dir, spec=spec, context="finalize-round refuses")
        if spec.harness_type != "synthetic":
            self._assert_dirty_paths_match_expected(
                mutable_paths=[self.repo_root / path for path in staged_paths] + [round_dir / "round_spec.yaml"],
                context="finalize-round refuses",
            )
        measurement_trace_ref.write_text(
            json.dumps(measurement_trace_combined, indent=2),
            encoding="utf-8",
        )
        search_trace_ref.write_text(
            json.dumps(search_trace_payload, indent=2),
            encoding="utf-8",
        )
        bundle_path = persist_tuned_config_bundle(bundle, self.tuned_config_root)
        (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
        lock_path = round_dir / ".round.lock"
        if lock_path.exists():
            lock_path.unlink()
        finalize_commit_sha = self._commit_paths(
            staged_paths,
            commit_message,
            spec.harness_type == "synthetic",
            branch=spec.round_branch,
            context="finalize-round refuses",
        )
        return {
            "round_id": round_id,
            "bundle_path": str(bundle_path),
            "winner_iteration": winner_row.iteration,
            "winner_candidate_uuid": winner_parent_uuid,
            "winner_rescreen_uuid": winner_rescreen_uuid or None,
            "finalize_commit_sha": finalize_commit_sha,
        }

    def status(self, *, round_id: str) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        if not round_dir.exists():
            raise FileNotFoundError(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        rows = self._read_results(round_dir / "results.tsv")
        phase = "bootstrapped"
        blocker: str | None = None
        if any(path.name == "BLOCKED.md" for path in round_dir.glob("candidates/*/BLOCKED.md")):
            phase = "blocked"
            blocker = next(round_dir.glob("candidates/*/BLOCKED.md")).read_text(encoding="utf-8").strip()
        elif (round_dir / "run_log.json").exists():
            phase = "finalized"
        elif (round_dir / "holdout_trace.json").exists():
            phase = "holdout"
        elif any(row.status == "rescreened" for row in rows):
            phase = "rescreen"
        elif any(row.iteration not in {"baseline_a", "baseline_b"} for row in rows):
            phase = "main_loop"
        elif rows:
            phase = "baseline"

        elapsed = max(0.0, time.time() - spec.round_started_at)
        return {
            "round_id": round_id,
            "phase": phase,
            "iterations_total": len(rows),
            "feasible_count": sum(1 for row in rows if row.feasible),
            "rescreened_count": sum(1 for row in rows if row.status == "rescreened"),
            "best_eval_throughput": max((float(row.eval_throughput) for row in rows if row.eval_throughput), default=0.0),
            "noise_floor": spec.noise_floor,
            "round_wall_clock_elapsed_s": round(elapsed, 3),
            "round_wall_clock_remaining_s": round(max(0.0, spec.round_wall_clock_s - elapsed), 3),
            "blocker": blocker,
        }

    def run_non_agent(
        self,
        *,
        model_id: str,
        family_id: str,
        workload_file: str | Path,
        baseline_bundle: str | Path | None,
        weight_version_id: str | None,
        round_root: str | Path,
        iteration_cap: int,
        harness_type: str = "real",
    ) -> dict[str, Any]:
        if harness_type not in {"real", "synthetic"}:
            raise RuntimeError(f"Unsupported harness_type: {harness_type}")
        if os.environ.get("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT") != "1":
            raise RuntimeError("auto-research run is CI-only unless LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT=1")
        bootstrap = self.bootstrap_round(
            model_id=model_id,
            family_id=family_id,
            sprint="sprint-0",
            workload_file=workload_file,
            weight_version_id=weight_version_id,
            round_root=round_root,
            harness_type=harness_type,
            skip_preflight=True,
        )
        round_id = str(bootstrap["round_id"])
        round_dir = Path(bootstrap["round_dir"])
        rows_created = []
        for suffix in ("a", "b"):
            self.measure(round_id=round_id, candidate_path=round_dir / "candidates" / f"baseline_{suffix}" / "candidate.yaml")
            rows_created.append(
                self.commit_candidate(
                    round_id=round_id,
                    iteration=f"baseline_{suffix}",
                    status="baseline",
                    notes=f"default-config baseline replay {suffix}",
                    allow_synthetic=harness_type == "synthetic",
                )
            )

        registry = load_registry(self.registry_path)
        model_config = registry[model_id]
        workload = SyntheticWorkloadDistribution.from_file(workload_file, model_config=model_config, family_id=family_id)
        runner = OfflineAutoResearchRunner(
            model_config=model_config,
            family_id=family_id,
            output_root=round_dir / "_legacy_plan",
            workload=workload,
            baseline_bundle=load_baseline_bundle(baseline_bundle),
            weight_version_id=weight_version_id,
            iteration_cap=iteration_cap,
        )
        for index, candidate in enumerate(runner._candidate_plan(), start=1):
            if index > iteration_cap:
                break
            iteration = f"{index:03d}"
            candidate_dir = round_dir / "candidates" / iteration
            candidate_dir.mkdir(parents=True, exist_ok=True)
            self._write_yaml(candidate_dir / "candidate.yaml", candidate)
            self.measure(round_id=round_id, candidate_path=candidate_dir / "candidate.yaml")
            trace = json.loads((candidate_dir / "measurement_trace.json").read_text(encoding="utf-8"))
            status = "keep" if bool(trace.get("feasible")) else ("crash" if "oom" in trace.get("feasibility_failures", []) else "discard")
            rows_created.append(
                self.commit_candidate(
                    round_id=round_id,
                    iteration=iteration,
                    status=status,
                    notes="non-agent dry-run candidate",
                    allow_synthetic=harness_type == "synthetic",
                )
            )
        finalized = self.finalize_round(round_id=round_id, dry_run=True)
        return {
            "round_id": round_id,
            "round_dir": str(round_dir),
            "bundle_path": finalized["bundle_path"],
            "rows_created": rows_created,
            "finalize_commit_sha": finalized["finalize_commit_sha"],
        }

    def _run_harness(
        self,
        *,
        spec: RoundSpecRecord,
        workload: SyntheticWorkloadDistribution,
        candidate_vllm_config: dict[str, Any],
        profile: str,
        use_holdout: bool = False,
    ) -> dict[str, Any]:
        if spec.harness_type == "synthetic":
            fixture_cls = self._load_synthetic_fixture()
            harness = fixture_cls(workload)
            evaluation = harness.evaluate(
                candidate_vllm_config,
                iteration=0,
                label="fixture",
                profile=profile,
            )
            return evaluation

        workload_spec = workload.to_workload_spec(base_dir=Path(spec.workload_file).parent)
        seed_trace_path = workload_spec.holdout_trace_ref if use_holdout and workload_spec.holdout_trace_ref else workload_spec.seed_trace_ref
        harness = RealMeasurementHarness(
            workload_spec=workload_spec,
            seed_trace_path=seed_trace_path,
            slo=SLO(
                ttft_ms=workload.latency_ceiling_ms,
                tpot_ms=workload.tpot_ceiling_ms,
                turn_ms=workload.turn_latency_ceiling_ms,
            ),
            endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
            metrics_scrape_url=f"http://127.0.0.1:{self.port}/metrics",
            admin_url=f"http://127.0.0.1:{self.proxy_port}/admin",
            model_id=spec.model_id,
            weight_version_id=spec.weight_version_id,
            bundle_staging_dir=Path(spec.round_dir) / ".measure-staging",
            round_id=spec.round_id,
        )
        warmup_s = spec.full_warmup_s if profile == "full" else spec.screen_warmup_s
        measurement_s = spec.full_measurement_s if profile == "full" else spec.screen_measurement_s
        measurable_config = {key: value for key, value in candidate_vllm_config.items() if key in ALLOWED_VLLM_CONFIG_KEYS}
        target_concurrency = int(spec.target_concurrency)
        try:
            return harness.measure(
                measurable_config,
                warmup_s=warmup_s,
                window_s=measurement_s,
                target_concurrency=target_concurrency,
            )
        except TypeError as exc:
            if "target_concurrency" not in str(exc):
                raise
            return harness.measure(
                measurable_config,
                warmup_s=warmup_s,
                window_s=measurement_s,
                target_concurrency_sweep=[target_concurrency],
            )

    def _spec_with_harness(self, spec: RoundSpecRecord, harness: str | None) -> RoundSpecRecord:
        if harness is None:
            return spec
        if harness not in {"real", "synthetic"}:
            raise RuntimeError(f"Unsupported harness: {harness}")
        if harness == "synthetic" and os.environ.get("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT") != "1":
            raise RuntimeError("synthetic harness requires LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT=1")
        if harness == spec.harness_type:
            return spec
        return RoundSpecRecord(**{**spec.__dict__, "harness_type": harness})

    def _load_synthetic_fixture(self):
        fixture_path = self.repo_root / "tests" / "fixtures" / "synthetic_measurement.py"
        if not fixture_path.is_file():
            raise RuntimeError(f"Synthetic fixture is unavailable: {fixture_path}")
        module_name = "_lumo_synthetic_measurement_fixture"
        spec = importlib.util.spec_from_file_location(module_name, fixture_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import synthetic fixture: {fixture_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        fixture_cls = getattr(module, "SyntheticMeasurementFixture", None)
        if fixture_cls is None:
            raise RuntimeError("SyntheticMeasurementFixture is not defined")
        return fixture_cls

    def _run_bootstrap_preflight(
        self,
        *,
        model_config: ModelConfig,
        family_id: str,
        weight_version_id: str | None,
        workload: SyntheticWorkloadDistribution,
    ) -> None:
        if RealMeasurementHarness is None:
            raise RuntimeError("bootstrap-round preflight failed: harness module missing")
        if os.environ.get("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT"):
            raise RuntimeError("bootstrap-round preflight failed: non-agent mode enabled")
        if self._git_status_short():
            raise RuntimeError("bootstrap-round requires a clean git worktree")
        if shutil.which("codex") is None:
            raise RuntimeError("bootstrap-round preflight failed: codex cli missing")
        try:
            version_result = subprocess.run(
                ["codex", "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("bootstrap-round preflight failed: codex cli missing or wrong version") from exc
        self._assert_supported_codex_cli_version(version_result.stdout)

        help_output = self._run_cli_probe(["auto-research", "--help"])
        for subcommand in AUTO_RESEARCH_HELP_SUBCOMMANDS:
            if subcommand not in help_output:
                raise RuntimeError(f"bootstrap-round preflight failed: cli subcommand missing: {subcommand}")
            payload = json.loads(self._run_cli_probe(["auto-research", subcommand, "--help-only"]))
            if payload != {"subcommand": subcommand, "status": "registered"}:
                raise RuntimeError(f"bootstrap-round preflight failed: cli subcommand missing: {subcommand}")

        resolved_weight_version = weight_version_id or default_weight_version_id(model_config)
        for bundle_path in sorted((self.tuned_config_root / family_id / resolved_weight_version).glob("*.yaml")):
            bundle = load_yaml_file(bundle_path)
            if not isinstance(bundle, dict):
                continue
            tuned_bundle = bundle.get("tuned_config_bundle")
            if not isinstance(tuned_bundle, dict):
                continue
            round_provenance = tuned_bundle.get("round_provenance")
            if isinstance(round_provenance, dict) and bool(round_provenance.get("dry_run")):
                raise RuntimeError("bootstrap-round preflight failed: dry_run_bundle_exists")

        if not workload.seed_trace_ref:
            raise RuntimeError("bootstrap-round preflight failed: seed trace missing")

    def _run_cli_probe(self, args: list[str]) -> str:
        env = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[1])
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_root if not existing_pythonpath else f"{src_root}:{existing_pythonpath}"
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "lumo_flywheel_serving.cli", *args],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            stdout = exc.stdout.strip()
            stderr = exc.stderr.strip()
            detail = stdout or stderr or f"exit {exc.returncode}"
            raise RuntimeError(f"bootstrap-round preflight failed: {detail}") from exc
        return completed.stdout.strip()

    @staticmethod
    def _assert_supported_codex_cli_version(version_output: str) -> None:
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_output)
        if match is None:
            raise RuntimeError("bootstrap-round preflight failed: codex cli missing or wrong version")
        version = tuple(int(part) for part in match.groups())
        if version < MIN_CODEX_CLI_VERSION:
            expected = ".".join(str(part) for part in MIN_CODEX_CLI_VERSION)
            found = ".".join(str(part) for part in version)
            raise RuntimeError(
                f"bootstrap-round preflight failed: codex cli missing or wrong version "
                f"(need >= {expected}, found {found})"
            )

    def _validate_measurement_trace(self, trace: dict[str, Any], *, status: str) -> None:
        self._normalize_trace(trace)
        candidate_uuid = trace.get("candidate_uuid")
        if not isinstance(candidate_uuid, str) or not candidate_uuid.strip():
            raise RuntimeError("commit_refused: malformed_trace")
        profile = trace.get("profile")
        if profile not in {"screen", "full"}:
            raise RuntimeError("commit_refused: malformed_trace")
        if status == "harness_fault":
            try:
                throughput = float(trace.get("eval_throughput"))
            except (TypeError, ValueError):
                throughput = -1.0
            if bool(trace.get("window_completed")) and throughput >= 0.0 and trace.get("per_request_latencies") is not None:
                raise RuntimeError("commit_refused: harness_fault requires invalid measurement state")
        if "eval_throughput" not in trace:
            raise RuntimeError("commit_refused: malformed_trace")
        if trace.get("window_completed") is not True and bool(trace.get("feasible")):
            raise RuntimeError("commit_refused: malformed_trace")
        if trace.get("no_oom_events") is not True and bool(trace.get("feasible")):
            raise RuntimeError("commit_refused: malformed_trace")
        if float(trace.get("reasoning_content_purity", 0.0)) != 1.0:
            raise RuntimeError("commit_refused: malformed_trace")

        cache_isolation = trace.get("cache_isolation")
        if not isinstance(cache_isolation, dict):
            raise RuntimeError("commit_refused: malformed_trace")
        if cache_isolation.get("cache_salt") != candidate_uuid:
            raise RuntimeError("commit_refused: malformed_trace")
        if cache_isolation.get("prefix_cache_reset_at_bootstrap") is not True:
            raise RuntimeError("commit_refused: malformed_trace")
        try:
            first_ten_hit_rate = float(cache_isolation["first_10_req_prefix_cache_hit_rate"])
            float(cache_isolation["last_10_req_prefix_cache_hit_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("commit_refused: malformed_trace") from exc
        if first_ten_hit_rate > 0.10:
            raise RuntimeError("commit_refused: malformed_trace")

    @staticmethod
    def _valid_latency_cross_checks(trace: dict[str, Any]) -> bool:
        for key in ("ttft_p95_ms", "tpot_p95_ms", "turn_latency_p95_ms"):
            payload = trace.get(key)
            if not isinstance(payload, dict):
                return False
            try:
                delta_pct = float(payload["delta_pct"])
            except (KeyError, TypeError, ValueError):
                return False
            if delta_pct > 10.0:
                return False
        return True

    @staticmethod
    def _normalize_trace(trace: dict[str, Any]) -> None:
        if "eval_throughput" not in trace and "sustained_concurrency" in trace:
            trace["eval_throughput"] = trace["sustained_concurrency"]
        trace.setdefault("window_completed", True)
        trace.setdefault("no_oom_events", True)
        if "diagnostics" not in trace:
            trace["diagnostics"] = {
                "ttft_p95_ms": trace.get("ttft_p95_ms", {"driver": 0.0, "promql": 0.0, "delta_pct": 0.0}),
                "tpot_p95_ms": trace.get("tpot_p95_ms", {"driver": 0.0, "promql": 0.0, "delta_pct": 0.0}),
                "turn_latency_p95_ms": trace.get(
                    "turn_latency_p95_ms",
                    {"driver": 0.0, "promql": 0.0, "delta_pct": 0.0},
                ),
                "rollout_throughput": trace.get("rollout_throughput", 0.0),
                "target_concurrency": trace.get("sustained_concurrency", 1),
            }

    def _trace_to_pending_row(self, trace: dict[str, Any]) -> ResultsRow:
        eval_throughput = ""
        if bool(trace.get("feasible")):
            eval_throughput = str(trace.get("eval_throughput", ""))
        return ResultsRow(
            candidate_uuid=str(trace["candidate_uuid"]),
            parent_candidate_uuid=str(trace.get("parent_candidate_uuid") or ""),
            iteration=str(trace["iteration"]),
            candidate_label=str(trace["candidate_label"]),
            feasible=bool(trace["feasible"]),
            eval_throughput=eval_throughput,
            objective_mean="",
            objective_ci_95="",
            measurement_count=1,
            window_completed=bool(trace.get("window_completed")),
            no_oom_events=bool(trace.get("no_oom_events")),
            reasoning_content_purity=str(trace["reasoning_content_purity"]),
            determinism_pass_rate=str(trace["determinism_pass_rate"]),
            status="",
            notes="",
        )

    def _candidate_commit_message(
        self,
        *,
        round_id: str,
        iteration: str,
        row: ResultsRow,
        trace_path: Path,
        extra_trailers: list[str] | None = None,
    ) -> str:
        objective = row.eval_throughput or f"infeasible:{row.notes or 'unscored'}"
        message_lines = [
            f"AR({round_id}) C{iteration}: {row.notes or row.candidate_label}",
            "",
            f"status={row.status} eval_throughput={objective} feasible={'true' if row.feasible else 'false'}",
            f"window_completed={'true' if row.window_completed else 'false'} no_oom={'true' if row.no_oom_events else 'false'} purity={row.reasoning_content_purity} determinism={row.determinism_pass_rate}",
            f"trace_ref={trace_path.as_posix()}",
            "",
            f"Candidate-UUID: {row.candidate_uuid}",
        ]
        if extra_trailers:
            message_lines.extend(extra_trailers)
        message_lines.append(f"Signed-off-by: {SIGNED_OFF_BY}")
        return "\n".join(message_lines) + "\n"

    def _commit_paths(
        self,
        paths: list[Path],
        message: str,
        skip_git: bool,
        *,
        branch: str,
        context: str,
    ) -> str:
        self._assert_only_allowed_staged_paths(paths, context=context)
        if skip_git:
            return f"synthetic-{uuid4()}"
        rel_paths = [str(path) for path in paths]
        parent_commit = self._branch_head(branch)
        branch_ref = self._branch_ref(branch)
        with tempfile.NamedTemporaryFile(prefix="lumo-auto-research-index-", delete=False) as handle:
            index_path = Path(handle.name)
        env = {"GIT_INDEX_FILE": str(index_path)}
        try:
            self._git(["read-tree", parent_commit], env=env)
            # Round artifacts intentionally live under output/, which is ignored
            # for normal development but must be committed to the experiment ledger.
            self._git(["add", "-A", "-f", "--", *rel_paths], capture_output=False, env=env)
            tree_sha = self._git(["write-tree"], env=env).stdout.strip()
            commit_sha = self._git(
                ["commit-tree", tree_sha, "-p", parent_commit, "-m", message],
                env=env,
            ).stdout.strip()
            self._git(["update-ref", branch_ref, commit_sha, parent_commit], capture_output=False)
            return commit_sha
        finally:
            index_path.unlink(missing_ok=True)

    def _round_dir(self, round_id: str) -> Path:
        direct = self.repo_root / "output" / "auto_research" / round_id
        if direct.is_dir():
            return direct
        for candidate in self.repo_root.glob(f"**/{round_id}/round_spec.yaml"):
            return candidate.parent
        raise FileNotFoundError(f"Unknown round_id: {round_id}")

    def _git(
        self,
        args: list[str],
        *,
        capture_output: bool = True,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command_env = os.environ.copy()
        if env:
            command_env.update(env)
        return subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=capture_output,
            text=True,
            check=True,
            env=command_env,
        )

    @staticmethod
    def _branch_ref(branch: str) -> str:
        return f"refs/heads/{branch}"

    def _branch_head(self, branch: str) -> str:
        return self._git(["rev-parse", self._branch_ref(branch)]).stdout.strip()

    def _create_branch_ref(self, *, round_branch: str, start_point: str) -> None:
        ref = self._branch_ref(round_branch)
        exists = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", ref],
            cwd=self.repo_root,
            check=False,
        ).returncode == 0
        if exists:
            raise RuntimeError(f"Round branch already exists: {round_branch}")
        self._git(["update-ref", ref, start_point], capture_output=False)

    def _git_status_short(self) -> str:
        return self._git(["status", "--short"]).stdout.strip()

    def _dirty_entries(self) -> list[tuple[str, str]]:
        lines = self._git(["status", "--short", "--untracked-files=all"]).stdout.splitlines()
        entries: list[tuple[str, str]] = []
        for line in lines:
            if len(line) < 4:
                continue
            status = line[:2]
            path = line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            entries.append((status, path))
        return entries

    def _assert_dirty_paths_match_expected(
        self,
        *,
        mutable_paths: list[Path],
        context: str,
        bootstrap_paths: list[Path] | None = None,
    ) -> None:
        mutable = [path.relative_to(self.repo_root).as_posix().rstrip("/") for path in mutable_paths]
        bootstrap = [
            path.relative_to(self.repo_root).as_posix().rstrip("/")
            for path in (bootstrap_paths or [])
            if path.exists()
        ]
        unexpected: set[str] = set()
        for status, path in self._dirty_entries():
            if path.endswith(".pyc") or path.startswith("__pycache__/") or "/__pycache__/" in path:
                continue
            if any(path == candidate or path.startswith(f"{candidate}/") for candidate in mutable):
                continue
            if any(path == candidate or path.startswith(f"{candidate}/") for candidate in bootstrap):
                if status == "??":
                    continue
            unexpected.add(path)
        if unexpected:
            rendered = ", ".join(sorted(unexpected))
            raise RuntimeError(f"{context}: dirty paths outside expected set: {rendered}")

    def _assert_immutable_round_artifacts(self, round_dir: Path, *, spec: RoundSpecRecord, context: str) -> None:
        if (round_dir / "impl_brief.md").read_text(encoding="utf-8") != IMPL_BRIEF_TEMPLATE:
            raise RuntimeError(f"{context}: immutable round artifact changed: impl_brief.md")
        if (round_dir / "iteration_brief.md").read_text(encoding="utf-8") != ITERATION_BRIEF_TEMPLATE:
            raise RuntimeError(f"{context}: immutable round artifact changed: iteration_brief.md")
        if (round_dir / "codex-home").exists():
            raise RuntimeError(f"{context}: immutable round artifact changed: codex-home")

        default_candidate = load_registry(self.registry_path)[spec.model_id].vllm_config()
        for suffix in ("a", "b"):
            baseline_path = round_dir / "candidates" / f"baseline_{suffix}" / "candidate.yaml"
            if load_yaml_file(baseline_path) != default_candidate:
                raise RuntimeError(
                    f"{context}: immutable round artifact changed: candidates/baseline_{suffix}/candidate.yaml"
                )

    def _bootstrap_round_artifact_paths(self, round_dir: Path) -> list[Path]:
        paths = [
            round_dir / "round_spec.yaml",
            round_dir / "impl_brief.md",
            round_dir / "iteration_brief.md",
            round_dir / ".round.lock",
            round_dir / "candidates" / "baseline_a",
            round_dir / "candidates" / "baseline_b",
        ]
        return [path for path in paths if path.exists()]

    def _staged_paths(self) -> list[str]:
        return [line.strip() for line in self._git(["diff", "--cached", "--name-only"]).stdout.splitlines() if line.strip()]

    def _assert_git_index_clean(self, *, context: str) -> None:
        staged = self._staged_paths()
        if staged:
            raise RuntimeError(f"{context}: git index not clean: {', '.join(staged)}")

    def _assert_only_allowed_staged_paths(self, allowed_paths: list[Path], *, context: str) -> None:
        allowed = [path.as_posix().rstrip("/") for path in allowed_paths]
        staged = set(self._staged_paths())
        unexpected = sorted(
            path
            for path in staged
            if not any(path == candidate or path.startswith(f"{candidate}/") for candidate in allowed)
        )
        if unexpected:
            raise RuntimeError(f"{context}: staged paths outside allow-list: {', '.join(unexpected)}")

    @staticmethod
    def _round_branch_name(*, model_id: str, family_id: str, sprint: str, timestamp: str) -> str:
        return f"autoresearch/{model_id}/{family_id}/{sprint}/{timestamp}"

    def _validate_iteration_id(self, iteration: str) -> None:
        if ITERATION_ID_RE.fullmatch(iteration) is None:
            raise RuntimeError(f"Invalid iteration id: {iteration}")

    def _validate_measure_request(self, *, spec: RoundSpecRecord, iteration_id: str, profile: str) -> None:
        if profile not in {"screen", "full"}:
            raise RuntimeError(f"measure_refused: unsupported profile {profile!r}")
        if iteration_id.isdigit() and int(iteration_id) > spec.iteration_cap:
            raise RuntimeError(
                f"measure_refused: iteration {iteration_id} exceeds iteration_cap {spec.iteration_cap}"
            )
        if iteration_id.startswith("rescreen_"):
            rescreen_index = int(iteration_id.split("_", 1)[1])
            if rescreen_index > spec.rescreen_top_k:
                raise RuntimeError(
                    f"measure_refused: iteration {iteration_id} exceeds rescreen_top_k {spec.rescreen_top_k}"
                )
            if profile != "full":
                raise RuntimeError("measure_refused: rescreen iterations require the full profile")

    @staticmethod
    def _metric_tie_break_value(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    def _validate_finalize_generator_consistency(self, round_dir: Path) -> None:
        generators: set[str] = set()
        for trace_path in sorted((round_dir / "candidates").glob("*/measurement_trace.json")):
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            generator = str(trace.get("generator") or "")
            if not generator:
                raise RuntimeError("finalize-round refuses because a measurement trace is missing generator")
            generators.add(generator)
        if len(generators) > 1:
            raise RuntimeError("finalize-round refuses because measurement generator changed mid-round")

    def _validate_finalize_harness_fault_rows(self, rows: list[ResultsRow]) -> None:
        for index, row in enumerate(rows):
            if row.status != "harness_fault":
                continue
            if any(later.feasible for later in rows[index + 1 :]):
                continue
            raise RuntimeError("finalize-round refuses because a harness_fault row has no successor feasible run")

    def _pick_winner_row(self, rows: list[ResultsRow], *, objective_field: str) -> ResultsRow:
        return min(
            rows,
            key=lambda row: (
                -self._metric_tie_break_value(getattr(row, objective_field)),
                row.iteration,
                row.candidate_uuid,
            ),
        )

    def _read_results(self, path: Path) -> list[ResultsRow]:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        rows: list[ResultsRow] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split("\t")
            payload = {key: values[index] if index < len(values) else "" for index, key in enumerate(header)}
            rows.append(ResultsRow.from_dict(payload))
        return rows

    def _write_results(self, path: Path, rows: list[ResultsRow]) -> None:
        rendered = ["\t".join(RESULTS_COLUMNS)]
        for row in rows:
            rendered.append("\t".join(row.as_dict().get(column, "") for column in RESULTS_COLUMNS))
        path.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    def _write_yaml(self, path: Path, payload: Any) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _updated_spec_with_noise_floor(
        self,
        *,
        spec: RoundSpecRecord,
        rows: list[ResultsRow],
    ) -> RoundSpecRecord:
        baseline_rows = sorted(
            (
                row
                for row in rows
                if row.iteration in {"baseline_a", "baseline_b"} and row.status == "baseline"
            ),
            key=lambda row: row.iteration,
        )
        if len(baseline_rows) != 2:
            return spec
        try:
            objectives = [float(row.eval_throughput) for row in baseline_rows]
        except (TypeError, ValueError):
            return spec
        noise_floor = 2.0 * abs(objectives[0] - objectives[1])
        if math.isclose(spec.noise_floor, noise_floor, rel_tol=0.0, abs_tol=1e-9):
            return spec
        return RoundSpecRecord(
            **{
                **spec.__dict__,
                "noise_floor": noise_floor,
            }
        )

    @staticmethod
    def _ci95(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return 1.96 * math.sqrt(variance) / math.sqrt(len(values))

    def _bundle_output_path(self, bundle: TunedConfigBundle) -> Path:
        return self.tuned_config_root / bundle.family_id / bundle.weight_version_id / f"{bundle.run_slug}.yaml"

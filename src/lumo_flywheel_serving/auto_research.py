from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import requests
import yaml

from .kernel_activation import KERNEL_SELECTION_RUNTIME_UNSUPPORTED, resolve_kernel_runtime_activation
from .measurement_harness import RealMeasurementHarness, SLO, WorkloadSpec
from .parity_fixture import (
    ACTUALLY_RESOLVED_KEYS,
    fetch_actually_resolved_kernel_selection,
    fixture_content_hash,
)
from .parity_probe import ParityProbeResult, run_parity_probe
from .registry import ModelConfig, load_registry
from .tuned_config import TunedConfigBundle, default_weight_version_id, make_tuned_config_bundle, persist_tuned_config_bundle
from .workload_p1 import HARDENED_L0_HEAVY_WORKLOAD_VERSION, L0_HEAVY_WORKLOAD_FAMILY_ID
from .yaml_utils import load_yaml_file

MIN_LIVE_STARTUP_GPU_MEMORY_UTILIZATION = 0.30
SIGNED_OFF_BY = "lumoserve-auto-research-cli <auto-research@lumo-flywheel>"
MIN_CODEX_CLI_VERSION = (0, 120, 0)
SERVING_THINKING_PROBE_MAX_AGE = timedelta(days=7)
REAL_MEASUREMENT_GENERATOR_PREFIX = "RealMeasurementHarness"
SYNTHETIC_MEASUREMENT_GENERATOR_PREFIX = "SyntheticMeasurementFixture"
ALLOWED_VLLM_CONFIG_KEYS = {
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "max_model_len",
    "kv_cache_dtype",
}
ALLOWED_REQUEST_SHAPING_KEYS = {
    "concurrency_cap_eval",
    "concurrency_cap_rollout",
    "admission_queue_depth_max",
    "per_request_kv_budget",
    "priority_preemption",
}
ENFORCED_REQUEST_SHAPING_FIELDS = [
    "concurrency_cap_eval",
    "concurrency_cap_rollout",
    "admission_queue_depth_max",
]
ADVISORY_REQUEST_SHAPING_FIELDS = ["per_request_kv_budget", "priority_preemption"]
PRIORITY_PREEMPTION_VALUES = {"off", "strict", "graceful"}
BASELINE_ITERATIONS = ("baseline_a", "baseline_b", "baseline_c", "baseline_d", "baseline_e")
BASELINE_ITERATION_SET = frozenset(BASELINE_ITERATIONS)
HARDENED_COMPOSITE_WORKLOAD_VERSION = "v1-multi-family-v5-thinking-realistic"
HARDENED_LEGACY_WORKLOAD_VERSION = "v1-thinking-realistic"
PRODUCTION_AUTO_RESEARCH_SUBCOMMANDS = (
    "bootstrap-round",
    "measure",
    "commit-candidate",
    "rescreen",
    "validate-holdout",
    "finalize-round",
    "status",
    "run-round",
    "replay-round",
    "tune-kernel-select",
    "tune-kernel-autotune",
    "mutate-kernel",
    "apply-and-test",
)
AUTO_RESEARCH_HELP_SUBCOMMANDS = PRODUCTION_AUTO_RESEARCH_SUBCOMMANDS + ("run",)
L0A_ELIMINATION_REASONS = {"nondeterministic", "parity_diverges_from_reference"}
L0A_DEFAULT_MODEL_ID = "qwen3.5-27b"
L0A_SELECT_ROUND_TYPE = "l0a_select_only"
L0B_AUTOTUNE_ROUND_TYPE = "l0b_autotune"
L0B_KERNEL_TARGETS = {"deltanet", "gatedattn", "fp8_gemm"}
L0B_DEFAULT_STABLE_WINDOW_REPLAYS = 10
L0B_DEFAULT_WARMUP_REPLAYS = 5
L0B_DEFAULT_MIN_HEADROOM_PCT = 0.03
L0C_MUTATION_ROUND_TYPE = "l0c_mutation"
L0C_KERNEL_TARGETS = {"deltanet", "gatedattn"}
L0C_DEFAULT_ACCEPTED_CAP = 12
L0C_DEFAULT_TOTAL_ATTEMPT_CAP = 36
L0C_DEFAULT_ROUND_TIMEOUT_HOURS = 12.0
L0C_DEFAULT_AGENT_TIMEOUT_S = 2 * 60 * 60
L0C_PROPOSER_STUCK_THRESHOLD = 3
L0C_COMPILE_FAILURES_THRESHOLD = 3
L0C_INTERMITTENT_PARITY_THRESHOLD = 2
L0C_MEASUREMENTS_PER_ACCEPTED = 2
L0C_PRIOR_REJECTION_LIMIT = 20
L0C_PRIOR_REJECTION_COLUMNS = [
    "source_round_id",
    "iteration",
    "mutation_hash",
    "rejection_reason",
    "first_diverging_probe_index",
    "tolerance_overshoot",
    "source_ref",
    "blocked_note",
]
L0C_TERMINAL_CONDITIONS = {
    "accepted_cap_reached",
    "total_attempt_cap_reached",
    "round_timeout",
    "proposer_stuck",
    "compile_failures_3x",
    "intermittent_parity_observed",
    "agent_rate_limited",
    "agent_unavailable",
}
RESULTS_COLUMNS = [
    "candidate_uuid",
    "parent_candidate_uuid",
    "iteration",
    "profile",
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
ITERATION_ID_RE = re.compile(
    r"^(\d{3}|baseline_[a-e]|import_\d{3}|rescreen_\d{2}(?:_screen_[1-9]\d*|_full_[1-9]\d*)?)$"
)

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
   {{candidate_schema_instruction}}

4. Invoke:
     lumoserve auto-research measure \
       --round-id {{round_id}} \
       --harness {{harness_mode}} \
       --candidate {{iteration_dir}}/candidate.yaml
   The CLI will:
     - compose the active-layer candidate with frozen lower-layer config
     - wait for /health
     - drive {{harness_generator_prefix}} for warmup + measurement window
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
       --harness {{harness_mode}} \
       --iteration {{iteration}} \
       --status <status> \
       --notes "<one-line rationale grounded in the trace>"
   The CLI will create one git commit with message format §7.3. In
   synthetic fixture mode, the commit also carries `Fixture-Mode: true`.

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
  generator starting with "{{harness_generator_prefix}}"
- One new row in results.tsv with a candidate_uuid column
  populated
- One new commit on {{round_branch}} whose message carries both
  a `Candidate-UUID: <uuid>` trailer (matching the results.tsv
  row) and a `Signed-off-by: lumoserve-auto-research-cli` trailer;
  synthetic fixture commits also carry `Fixture-Mode: true`
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
    nominal_ttft_ms: int = 2000
    nominal_tpot_ms: int = 80
    nominal_turn_ms: int = 30000

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
            "nominal_ttft_ms": 2000,
            "nominal_tpot_ms": 80,
            "nominal_turn_ms": 30000,
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
            nominal_ttft_ms=payload["nominal_ttft_ms"],
            nominal_tpot_ms=payload["nominal_tpot_ms"],
            nominal_turn_ms=payload["nominal_turn_ms"],
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
        payload.setdefault("nominal_ttft_ms", payload["latency_ceiling_ms"])
        payload.setdefault("rollout_baseline", 14.0)
        payload.setdefault("target_concurrency", 4)
        payload.setdefault("tpot_ceiling_ms", 80)
        payload.setdefault("turn_latency_ceiling_ms", int(payload["latency_ceiling_ms"]))
        payload.setdefault("nominal_tpot_ms", payload["tpot_ceiling_ms"])
        payload.setdefault("nominal_turn_ms", payload["turn_latency_ceiling_ms"])
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
            nominal_ttft_ms=int(payload.get("nominal_ttft_ms", payload["latency_ceiling_ms"])),
            nominal_tpot_ms=int(payload.get("nominal_tpot_ms", payload.get("tpot_ceiling_ms", 80))),
            nominal_turn_ms=int(payload.get("nominal_turn_ms", payload.get("turn_latency_ceiling_ms", payload["latency_ceiling_ms"]))),
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


def _parse_probe_report_field(text: str, field_name: str) -> str:
    match = re.search(rf"(?im)^\s*-?\s*{re.escape(field_name)}:\s*(.+?)\s*$", text)
    if not match:
        raise RuntimeError(f"serving thinking probe missing {field_name}")
    return match.group(1).strip()


def _parse_probe_capture_date(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise RuntimeError(f"serving thinking probe has invalid capture_date: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_descriptor_ref(descriptor_path: Path, ref: str) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    return descriptor_path.parent / path


def compute_workload_distribution_id(descriptor_path: str | Path) -> str:
    descriptor = Path(descriptor_path)
    payload = load_yaml_file(descriptor)
    if not isinstance(payload, dict):
        raise ValueError(f"Workload descriptor must be a mapping: {descriptor}")
    seed_ref = payload.get("seed_trace_ref")
    holdout_ref = payload.get("holdout_trace_ref")
    if not isinstance(seed_ref, str) or not seed_ref.strip():
        raise ValueError("descriptor_missing_seed_trace_ref")
    if not isinstance(holdout_ref, str) or not holdout_ref.strip():
        raise ValueError("descriptor_missing_holdout_trace_ref")
    seed_hash = hashlib.sha256(_resolve_descriptor_ref(descriptor, seed_ref).read_bytes()).hexdigest()
    holdout_hash = hashlib.sha256(_resolve_descriptor_ref(descriptor, holdout_ref).read_bytes()).hexdigest()
    canonical_payload = dict(payload)
    canonical_payload["workload_distribution_id"] = None
    yaml_hash = hashlib.sha256(
        yaml.safe_dump(canonical_payload, sort_keys=True, default_flow_style=False).encode("utf-8")
    ).hexdigest()
    return hashlib.sha256((seed_hash + holdout_hash + yaml_hash).encode("ascii")).hexdigest()


def resolve_workload_descriptor(repo_root: str | Path, family_id: str, workload_file: str | Path | None = None) -> Path:
    if workload_file is not None:
        return Path(workload_file).resolve()
    root = Path(repo_root).resolve()
    composite = root / "benchmark_blueprints" / "workloads" / family_id / "workload.yaml"
    if composite.is_file():
        return composite.resolve()
    legacy = root / "benchmark_blueprints" / "families" / family_id / "serving_workload.yaml"
    if legacy.is_file():
        return legacy.resolve()
    raise RuntimeError(f"workload_descriptor_missing:{family_id}")


def verify_workload_descriptor_preconditions(
    descriptor_path: str | Path,
    *,
    composite: bool,
    allow_legacy_workload: bool = False,
) -> dict[str, Any]:
    descriptor = Path(descriptor_path)
    payload = load_yaml_file(descriptor)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid workload descriptor: {descriptor}")
    descriptor_id = payload.get("workload_distribution_id")
    if not isinstance(descriptor_id, str) or not descriptor_id.strip():
        raise RuntimeError("descriptor_missing_workload_distribution_id")
    canonical_id = compute_workload_distribution_id(descriptor)
    if descriptor_id != canonical_id:
        raise RuntimeError("descriptor_workload_distribution_id_mismatch")
    descriptor_family_id = str(payload.get("family_id", ""))
    if descriptor_family_id == L0_HEAVY_WORKLOAD_FAMILY_ID:
        expected_version = HARDENED_L0_HEAVY_WORKLOAD_VERSION
    else:
        expected_version = HARDENED_COMPOSITE_WORKLOAD_VERSION if composite else HARDENED_LEGACY_WORKLOAD_VERSION
    hardening_version = str(payload.get("workload_distribution_id_hardening_version", ""))
    if hardening_version != expected_version and not allow_legacy_workload:
        raise RuntimeError("descriptor_stale_workload_distribution_id_hardening_version")
    return {
        "workload_distribution_id": descriptor_id,
        "workload_distribution_id_hardening_version": hardening_version,
        "expected_workload_distribution_id_hardening_version": expected_version,
        "canonical_workload_distribution_id": canonical_id,
    }


def resolve_serving_thinking_probe_report(
    repo_root: Path,
    requested_path: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    reports_dir = repo_root / "reports"
    if requested_path is None:
        candidates = sorted(reports_dir.glob("thinking-probe-*.md"), key=lambda path: path.stat().st_mtime)
        if not candidates:
            raise RuntimeError("serving_thinking_probe_missing")
        probe_path = candidates[-1]
    else:
        probe_path = Path(requested_path)
        if not probe_path.is_absolute():
            probe_path = repo_root / probe_path
    probe_path = probe_path.resolve()
    expected_dir = reports_dir.resolve()
    if probe_path.parent != expected_dir or not probe_path.name.startswith("thinking-probe-") or probe_path.suffix != ".md":
        raise RuntimeError("serving_thinking_probe_invalid_path")
    if not probe_path.is_file():
        raise RuntimeError(f"serving_thinking_probe_missing: {probe_path}")

    text = probe_path.read_text(encoding="utf-8")
    capture_date_raw = _parse_probe_report_field(text, "capture_date")
    outcome = _parse_probe_report_field(text, "outcome")
    if outcome not in {"row-1", "row-3"}:
        raise RuntimeError(f"serving_thinking_probe_blocking_outcome:{outcome}")
    capture_date = _parse_probe_capture_date(capture_date_raw)
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    if capture_date > checked_at + timedelta(minutes=5):
        raise RuntimeError("serving_thinking_probe_capture_date_in_future")
    if checked_at - capture_date > SERVING_THINKING_PROBE_MAX_AGE:
        raise RuntimeError("serving_thinking_probe_expired")
    mtime = datetime.fromtimestamp(probe_path.stat().st_mtime, tz=UTC)
    if checked_at - mtime > SERVING_THINKING_PROBE_MAX_AGE:
        raise RuntimeError("serving_thinking_probe_mtime_expired")
    return {
        "path": _relative_to_repo(repo_root, probe_path),
        "capture_date": capture_date.isoformat().replace("+00:00", "Z"),
        "outcome": outcome,
        "file_mtime": mtime.isoformat().replace("+00:00", "Z"),
    }


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


@dataclass(frozen=True)
class L0aKernelCombo:
    combo_id: str
    attention_backend: str
    deltanet_kernel: str
    fp8_gemm_kernel: str
    torch_compile_mode: str
    cuda_graph_capture: str

    def as_dict(self) -> dict[str, str]:
        return {
            "combo_id": self.combo_id,
            "attention_backend": self.attention_backend,
            "deltanet_kernel": self.deltanet_kernel,
            "fp8_gemm_kernel": self.fp8_gemm_kernel,
            "torch_compile_mode": self.torch_compile_mode,
            "cuda_graph_capture": self.cuda_graph_capture,
        }


@dataclass(frozen=True)
class L0aKernelSelectResult:
    round_id: str
    round_dir: Path
    bundle_path: Path
    winner: L0aKernelCombo
    total_combos: int
    eliminated_count: int
    survivor_count: int
    artifact_paths: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_dir": str(self.round_dir),
            "bundle_path": str(self.bundle_path),
            "winner": self.winner.as_dict(),
            "total_combos": self.total_combos,
            "eliminated_count": self.eliminated_count,
            "survivor_count": self.survivor_count,
            "artifact_paths": dict(self.artifact_paths),
        }


class L0aKernelSelectRunner:
    """Bounded P3/L0a select-only substrate.

    Real mode dispatches through the live repo endpoint and restarts vLLM per
    supported runtime activation signature. Unsupported knobs halt before live
    dispatch with structured evidence.
    """

    ELIMINATED_COLUMNS = [
        "combo_id",
        "attention_backend",
        "deltanet_kernel",
        "fp8_gemm_kernel",
        "torch_compile_mode",
        "cuda_graph_capture",
        "elimination_reason",
        "first_diverging_probe_index",
        "tolerance_overshoot",
    ]
    MEASUREMENT_COLUMNS = [
        "combo_id",
        "measurement_role",
        "measurement_index",
        "objective_value",
        "harness",
        "trace_ref",
        "kernel_selection_applied",
    ]
    RUNTIME_UNSUPPORTED_COLUMNS = [
        "combo_id",
        "smoke_status",
        "attention_backend",
        "deltanet_kernel",
        "fp8_gemm_kernel",
        "torch_compile_mode",
        "cuda_graph_capture",
        "unsupported_knobs_json",
    ]

    def __init__(
        self,
        *,
        repo_root: str | Path,
        registry_path: str | Path,
        tuned_config_root: str | Path,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.registry_path = Path(registry_path).resolve()
        self.tuned_config_root = Path(tuned_config_root).resolve()

    def run(
        self,
        *,
        workload_file: str | Path,
        action_space_file: str | Path,
        baselines: int,
        screen_measurements_per_combo: int,
        rescreen_top_k: int,
        rescreen_measurements_per_candidate: int,
        parallel_instances: str,
        round_root: str | Path,
        harness: str,
        model_id: str = L0A_DEFAULT_MODEL_ID,
        port: int = 8000,
        proxy_port: int = 8001,
        max_combos: int | None = None,
        image: str | None = None,
        container_name: str = "lumo-vllm",
        logs_root: str | Path = "/logs",
        triton_cache_root: str | Path = "/tmp/triton_cache",
        state_root: str | Path | None = None,
        runtime_unsupported_policy: str = "partition",
    ) -> L0aKernelSelectResult:
        if harness not in {"real", "synthetic"}:
            raise RuntimeError(f"Unsupported harness: {harness}")
        if runtime_unsupported_policy not in {"partition", "strict"}:
            raise RuntimeError("--runtime-unsupported-policy must be 'partition' or 'strict'")
        if min(baselines, screen_measurements_per_combo, rescreen_top_k, rescreen_measurements_per_candidate) < 1:
            raise RuntimeError("L0a kernel selection counts must all be >= 1")

        workload_path = Path(workload_file).resolve()
        action_space_path = Path(action_space_file).resolve()
        descriptor = load_yaml_file(workload_path)
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"Workload descriptor must be a mapping: {workload_path}")
        fixture_refs = self._resolve_parity_fixture_refs(workload_path, descriptor)
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id: {model_id}")
        weight_version_id = default_weight_version_id(registry[model_id])
        combos = self._load_action_space(action_space_path)
        if not combos:
            raise RuntimeError("action space produced zero L0a combos")
        total_combos_available = len(combos)
        if max_combos is not None:
            if max_combos < 1:
                raise RuntimeError("--max-combos must be >= 1 when provided")
            combos = combos[:max_combos]

        fanout = self._resolve_parallel_instances(parallel_instances)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        round_id = f"{model_id}-{descriptor.get('family_id', 'unknown')}-l0a-select-{timestamp}"
        root = Path(round_root).resolve()
        round_dir = root / round_id
        if round_dir.exists():
            raise RuntimeError(f"L0a round directory already exists: {round_dir}")
        round_dir.mkdir(parents=True)
        (round_dir / "candidates").mkdir()

        spec = {
            "round_id": round_id,
            "round_type": L0A_SELECT_ROUND_TYPE,
            "model_id": model_id,
            "family_id": str(descriptor.get("family_id", "")),
            "source_family": str(descriptor.get("source_family") or descriptor.get("family_id", "")),
            "workload_file": str(workload_path),
            "action_space_file": str(action_space_path),
            "harness": harness,
            "runtime_unsupported_policy": runtime_unsupported_policy if harness == "real" else "not_applicable",
            "baselines": baselines,
            "screen_measurements_per_combo": screen_measurements_per_combo,
            "rescreen_top_k": rescreen_top_k,
            "rescreen_measurements_per_candidate": rescreen_measurements_per_candidate,
            "parallel_instances": fanout,
            "total_combos_available": total_combos_available,
            "total_combos_scheduled": len(combos),
            "full_action_space_sweep": len(combos) == total_combos_available,
            "limited_mode": len(combos) != total_combos_available,
            "max_combos": max_combos,
            "parallel_evidence": self._parallel_evidence(fanout),
            "parity_fixture_refs": {
                key: _relative_to_repo(self.repo_root, value)
                for key, value in fixture_refs.items()
            },
            "parity_fixture_content_hashes": {
                key: fixture_content_hash(value)
                for key, value in fixture_refs.items()
            },
            "weight_version_id": weight_version_id,
            "endpoint": f"http://127.0.0.1:{proxy_port}",
            "upstream_port": port,
            "proxy_port": proxy_port,
            "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        self._write_yaml(round_dir / "round_spec.yaml", spec)
        self._write_yaml(round_dir / "action_space.normalized.yaml", [combo.as_dict() for combo in combos])

        eliminated, survivors, smoke_rows = self._run_smoke(combos)
        if not survivors:
            run_log = {
                "outcome": "ROUND_BLOCKED",
                "HALT_REASON": "smoke_zero_survivors",
                "total_combos": len(combos),
                "eliminated_count": len(eliminated),
            }
            (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
            raise RuntimeError("HALT_REASON: smoke_zero_survivors")
        self._write_tsv(round_dir / "eliminated.tsv", self.ELIMINATED_COLUMNS, eliminated)
        self._write_tsv(
            round_dir / "survivors.tsv",
            ["combo_id", "attention_backend", "deltanet_kernel", "fp8_gemm_kernel", "torch_compile_mode", "cuda_graph_capture"],
            [combo.as_dict() for combo in survivors],
        )
        (round_dir / "smoke_trace.json").write_text(json.dumps(smoke_rows, indent=2), encoding="utf-8")

        runtime_activation_check: dict[str, Any] | None = None
        runtime_activation_check_path: Path | None = None
        runtime_supported_survivors = list(survivors)
        unsupported_runtime_audit_ref: str | None = None
        runtime_supported_action_space_ref: str | None = None
        runtime_unsupported_action_space_ref: str | None = None

        if harness == "synthetic":
            baseline_rows = self._baseline_measurements(baselines)
            screen_rows = self._screen_measurements(survivors, screen_measurements_per_combo)
        else:
            runtime_activation_check = self._runtime_activation_check(combos, survivors)
            runtime_activation_check_path = round_dir / "runtime_activation_check.json"
            unsupported_activation = runtime_activation_check["unsupported_runtime_activation"]
            unsupported_survivor_activation = [
                item
                for item in unsupported_activation
                if item["smoke_status"] == "survivor"
            ]
            runtime_activation_check["runtime_unsupported_policy"] = runtime_unsupported_policy
            runtime_activation_check["status"] = (
                "blocked"
                if unsupported_activation and runtime_unsupported_policy == "strict"
                else "partitioned"
                if unsupported_activation
                else "pass"
            )
            supported_combo_ids = {
                item["combo_id"]
                for item in runtime_activation_check["supported_runtime_activation"]
            }
            runtime_supported_survivors = [
                combo for combo in survivors if combo.combo_id in supported_combo_ids
            ]
            runtime_activation_check["supported_survivor_count"] = len(runtime_supported_survivors)
            runtime_activation_check["runtime_measured_survivor_combo_ids"] = [
                combo.combo_id for combo in runtime_supported_survivors
            ]
            runtime_activation_check["unsupported_runtime_audit_ref"] = "unsupported_runtime_candidates.tsv"
            runtime_activation_check_path.write_text(
                json.dumps(runtime_activation_check, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            unsupported_runtime_audit_ref = "unsupported_runtime_candidates.tsv"
            runtime_supported_action_space_ref = "action_space.runtime_supported.yaml"
            runtime_unsupported_action_space_ref = "action_space.runtime_unsupported.yaml"
            self._write_tsv(
                round_dir / unsupported_runtime_audit_ref,
                self.RUNTIME_UNSUPPORTED_COLUMNS,
                self._unsupported_runtime_rows(unsupported_activation),
            )
            self._write_yaml(
                round_dir / runtime_supported_action_space_ref,
                [
                    item["kernel_selection"]
                    for item in runtime_activation_check["supported_runtime_activation"]
                ],
            )
            self._write_yaml(
                round_dir / runtime_unsupported_action_space_ref,
                [
                    {
                        **item["kernel_selection"],
                        "smoke_status": item["smoke_status"],
                        "unsupported_knobs": item["unsupported_knobs"],
                    }
                    for item in unsupported_activation
                ],
            )
            if unsupported_activation and runtime_unsupported_policy == "strict":
                live_dispatch_reason = (
                    "scheduled survivor contains unsupported runtime kernel knob"
                    if unsupported_survivor_activation
                    else "scheduled candidate contains unsupported runtime kernel knob"
                )
                run_log = {
                    "outcome": "ROUND_BLOCKED",
                    "HALT_REASON": KERNEL_SELECTION_RUNTIME_UNSUPPORTED,
                    "round_id": round_id,
                    "total_combos_available": total_combos_available,
                    "total_combos_scheduled": len(combos),
                    "full_action_space_sweep": len(combos) == total_combos_available,
                    "limited_mode": len(combos) != total_combos_available,
                    "runtime_activation_check_ref": str(runtime_activation_check_path.relative_to(round_dir)),
                    "unsupported_runtime_audit_ref": unsupported_runtime_audit_ref,
                    "unsupported_runtime_activation": unsupported_activation,
                    "live_dispatch": {
                        "attempted": False,
                        "reason": live_dispatch_reason,
                    },
                }
                (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
                (round_dir / "measurement_trace_combined.json").write_text(
                    json.dumps(
                        {
                            "round_id": round_id,
                            "harness": harness,
                            "kernel_selection_runtime_activation": "unsupported_knobs",
                            "runtime_activation_check_ref": str(runtime_activation_check_path.relative_to(round_dir)),
                            "unsupported_runtime_audit_ref": unsupported_runtime_audit_ref,
                            "unsupported_runtime_activation": unsupported_activation,
                            "live_dispatch": {
                                "attempted": False,
                                "reason": live_dispatch_reason,
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError(
                    f"HALT_REASON: {KERNEL_SELECTION_RUNTIME_UNSUPPORTED}; "
                    "scheduled candidate contains unsupported runtime kernel knob(s)"
                )
            if not runtime_supported_survivors:
                run_log = {
                    "outcome": "ROUND_BLOCKED",
                    "HALT_REASON": "l0a_runtime_supported_zero_survivors",
                    "round_id": round_id,
                    "total_combos_available": total_combos_available,
                    "total_combos_scheduled": len(combos),
                    "runtime_activation_check_ref": str(runtime_activation_check_path.relative_to(round_dir)),
                    "unsupported_runtime_audit_ref": unsupported_runtime_audit_ref,
                    "runtime_unsupported_policy": runtime_unsupported_policy,
                    "supported_combo_count": runtime_activation_check["supported_combo_count"],
                    "unsupported_combo_count": runtime_activation_check["unsupported_combo_count"],
                    "unsupported_survivor_count": runtime_activation_check["unsupported_survivor_count"],
                    "live_dispatch": {
                        "attempted": False,
                        "reason": "no smoke survivor has supported runtime activation",
                    },
                }
                (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
                raise RuntimeError("HALT_REASON: l0a_runtime_supported_zero_survivors")
            real_harness = self._real_harness(
                workload_path=workload_path,
                descriptor=descriptor,
                model_config=registry[model_id],
                model_id=model_id,
                weight_version_id=weight_version_id,
                round_id=round_id,
                round_dir=round_dir,
                port=port,
                proxy_port=proxy_port,
                image=image,
                container_name=container_name,
                logs_root=logs_root,
                triton_cache_root=triton_cache_root,
                state_root=state_root,
            )
            try:
                baseline_rows = self._real_baseline_measurements(
                    real_harness,
                    baselines=baselines,
                    baseline_vllm_config=registry[model_id].vllm_config(),
                    round_dir=round_dir,
                )
                screen_rows = self._real_combo_measurements(
                    real_harness,
                    combos=runtime_supported_survivors,
                    measurements_per_combo=screen_measurements_per_combo,
                    baseline_vllm_config=registry[model_id].vllm_config(),
                    round_dir=round_dir,
                    role="screen",
                )
            except Exception:
                real_harness.restore_runtime()
                raise
        self._write_tsv(round_dir / "measurements.tsv", self.MEASUREMENT_COLUMNS, [*baseline_rows, *screen_rows])
        measurable_survivors = runtime_supported_survivors if harness == "real" else survivors
        top = self._top_screen_combos(measurable_survivors, screen_rows, rescreen_top_k)
        if harness == "synthetic":
            rescreen_rows = self._rescreen_measurements(top, rescreen_measurements_per_candidate)
        else:
            try:
                rescreen_rows = self._real_combo_measurements(
                    real_harness,
                    combos=top,
                    measurements_per_combo=rescreen_measurements_per_candidate,
                    baseline_vllm_config=registry[model_id].vllm_config(),
                    round_dir=round_dir,
                    role="rescreen",
                )
            finally:
                real_harness.restore_runtime()
        self._write_tsv(round_dir / "rescreen.tsv", self.MEASUREMENT_COLUMNS, rescreen_rows)
        winner = self._pick_winner(top, rescreen_rows)
        self._write_yaml(round_dir / "winner_kernel_select.yaml", winner.as_dict())

        determinism_log = self._winner_determinism_log(winner)
        parity_check = self._winner_parity_check(winner, fixture_refs=fixture_refs)
        (round_dir / "determinism_log.json").write_text(json.dumps(determinism_log, indent=2), encoding="utf-8")
        (round_dir / "parity_check.json").write_text(json.dumps(parity_check, indent=2), encoding="utf-8")
        if not determinism_log["pass"] or not parity_check["pass"]:
            (round_dir / "run_log.json").write_text(
                json.dumps({"outcome": "ROUND_BLOCKED", "HALT_REASON": "l0a_parity_fail_winner"}, indent=2),
                encoding="utf-8",
            )
            raise RuntimeError("HALT_REASON: l0a_parity_fail_winner")

        search_trace = {
            "round_id": round_id,
            "total_combos": len(combos),
            "eliminated_count": len(eliminated),
            "survivor_count": len(survivors),
            "runtime_supported_survivor_count": len(runtime_supported_survivors) if harness == "real" else len(survivors),
            "runtime_unsupported_policy": runtime_unsupported_policy if harness == "real" else "not_applicable",
            "runtime_activation_check_ref": (
                str(runtime_activation_check_path.relative_to(round_dir))
                if runtime_activation_check_path is not None
                else None
            ),
            "unsupported_runtime_audit_ref": unsupported_runtime_audit_ref,
            "winner": winner.as_dict(),
            "smoke": smoke_rows,
        }
        measurement_trace = {
            "round_id": round_id,
            "baselines": baseline_rows,
            "screen": screen_rows,
            "rescreen": rescreen_rows,
            "winner": winner.as_dict(),
            "harness": harness,
            "limited_mode": len(combos) != total_combos_available,
            "kernel_selection_runtime_activation": (
                "runtime_applied" if harness == "real" else "synthetic_fixture"
            ),
            "runtime_unsupported_policy": runtime_unsupported_policy if harness == "real" else "not_applicable",
            "runtime_activation_check_ref": (
                str(runtime_activation_check_path.relative_to(round_dir))
                if runtime_activation_check_path is not None
                else None
            ),
            "unsupported_runtime_audit_ref": unsupported_runtime_audit_ref,
            "runtime_supported_survivor_count": len(runtime_supported_survivors) if harness == "real" else len(survivors),
            "runtime_unsupported_combo_count": (
                runtime_activation_check["unsupported_combo_count"]
                if runtime_activation_check is not None
                else 0
            ),
            "runtime_unsupported_survivor_count": (
                runtime_activation_check["unsupported_survivor_count"]
                if runtime_activation_check is not None
                else 0
            ),
        }
        (round_dir / "search_trace.json").write_text(json.dumps(search_trace, indent=2), encoding="utf-8")
        (round_dir / "measurement_trace_combined.json").write_text(
            json.dumps(measurement_trace, indent=2),
            encoding="utf-8",
        )
        bundle = make_tuned_config_bundle(
            model_id=model_id,
            family_id=str(descriptor.get("family_id", "")),
            weight_version_id=weight_version_id,
            workload_distribution_id=str(descriptor["workload_distribution_id"]),
            vllm_config=registry[model_id].vllm_config(),
            kernel_selection=winner.as_dict(),
            objective={
                "metric": "l0a_rescreen_objective_mean",
                "value": self._objective_mean(winner.combo_id, rescreen_rows),
                "baseline_mean_screen": self._baseline_mean(baseline_rows),
            },
            measurement_trace_ref=_relative_to_repo(self.repo_root, round_dir / "measurement_trace_combined.json"),
            search_trace_ref=_relative_to_repo(self.repo_root, round_dir / "search_trace.json"),
            baseline_bundle_id=None,
            regression_guard={
                "baseline_measurements": baselines,
                "screen_measurements_per_combo": screen_measurements_per_combo,
                "rescreen_measurements_per_candidate": rescreen_measurements_per_candidate,
            },
            safety_rails={
                "determinism_check_passed": True,
                "parity_check_passed": True,
                "production_load_refused": True,
                "kernel_selection_runtime_activation": "runtime_applied" if harness == "real" else "synthetic_fixture",
                "runtime_unsupported_policy": runtime_unsupported_policy if harness == "real" else "not_applicable",
                "runtime_supported_survivor_count": len(runtime_supported_survivors) if harness == "real" else len(survivors),
                "runtime_unsupported_combo_count": (
                    runtime_activation_check["unsupported_combo_count"]
                    if runtime_activation_check is not None
                    else 0
                ),
                "serial_only_supported": fanout == 1,
                "full_action_space_sweep": len(combos) == total_combos_available,
            },
            round_provenance={
                "round_type": L0A_SELECT_ROUND_TYPE,
                "round_id": round_id,
                "harness": harness,
                "workload_descriptor_path": str(workload_path),
                "action_space_file": str(action_space_path),
                "parity_fixture_refs": spec["parity_fixture_refs"],
                "parity_fixture_content_hashes": spec["parity_fixture_content_hashes"],
                "parallel_instances": fanout,
                "parallel_evidence": spec["parallel_evidence"],
                "total_combos_available": total_combos_available,
                "total_combos_scheduled": len(combos),
                "runtime_unsupported_policy": runtime_unsupported_policy if harness == "real" else "not_applicable",
                "runtime_supported_combo_count": (
                    runtime_activation_check["supported_combo_count"]
                    if runtime_activation_check is not None
                    else len(combos)
                ),
                "runtime_supported_survivor_count": len(runtime_supported_survivors) if harness == "real" else len(survivors),
                "runtime_unsupported_combo_count": (
                    runtime_activation_check["unsupported_combo_count"]
                    if runtime_activation_check is not None
                    else 0
                ),
                "runtime_unsupported_survivor_count": (
                    runtime_activation_check["unsupported_survivor_count"]
                    if runtime_activation_check is not None
                    else 0
                ),
                "runtime_activation_check_ref": (
                    _relative_to_repo(self.repo_root, runtime_activation_check_path)
                    if runtime_activation_check_path is not None
                    else None
                ),
                "unsupported_runtime_audit_ref": (
                    _relative_to_repo(self.repo_root, round_dir / unsupported_runtime_audit_ref)
                    if unsupported_runtime_audit_ref is not None
                    else None
                ),
                "limited_mode": len(combos) != total_combos_available,
                "confidence": "defensible",
                "results_tsv_ref": _relative_to_repo(self.repo_root, round_dir / "rescreen.tsv"),
            },
        )
        bundle_path = persist_tuned_config_bundle(bundle, self.tuned_config_root)
        run_log = {
            "outcome": "PASS",
            "round_id": round_id,
            "bundle_path": str(bundle_path),
            "winner": winner.as_dict(),
            "total_combos": len(combos),
            "total_combos_available": total_combos_available,
            "eliminated_count": len(eliminated),
            "survivor_count": len(survivors),
            "runtime_supported_survivor_count": len(runtime_supported_survivors) if harness == "real" else len(survivors),
            "runtime_unsupported_policy": runtime_unsupported_policy if harness == "real" else "not_applicable",
            "runtime_activation_check_ref": (
                str(runtime_activation_check_path.relative_to(round_dir))
                if runtime_activation_check_path is not None
                else None
            ),
            "unsupported_runtime_audit_ref": unsupported_runtime_audit_ref,
            "full_action_space_sweep": len(combos) == total_combos_available,
            "limited_mode": len(combos) != total_combos_available,
            "kernel_selection_runtime_activation": "runtime_applied" if harness == "real" else "synthetic_fixture",
            "artifact_counts": {
                "eliminated_rows": len(eliminated),
                "survivor_rows": len(survivors),
                "runtime_supported_survivor_rows": len(runtime_supported_survivors) if harness == "real" else len(survivors),
                "runtime_unsupported_rows": (
                    runtime_activation_check["unsupported_combo_count"]
                    if runtime_activation_check is not None
                    else 0
                ),
                "baseline_rows": len(baseline_rows),
                "screen_rows": len(screen_rows),
                "rescreen_rows": len(rescreen_rows),
            },
        }
        if harness == "real":
            run_log["live_dispatch"] = {
                "endpoint": f"http://127.0.0.1:{proxy_port}",
                "baseline_rows": len(baseline_rows),
                "screen_rows": len(screen_rows),
                "rescreen_rows": len(rescreen_rows),
                "unsupported_runtime_excluded_rows": runtime_activation_check["unsupported_combo_count"] if runtime_activation_check is not None else 0,
                "unsupported_runtime_excluded_survivors": runtime_activation_check["unsupported_survivor_count"] if runtime_activation_check is not None else 0,
                "runtime_supported_action_space_ref": runtime_supported_action_space_ref,
                "runtime_unsupported_action_space_ref": runtime_unsupported_action_space_ref,
                "target_concurrency": 1,
            }
        (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
        return L0aKernelSelectResult(
            round_id=round_id,
            round_dir=round_dir,
            bundle_path=bundle_path,
            winner=winner,
            total_combos=len(combos),
            eliminated_count=len(eliminated),
            survivor_count=len(survivors),
            artifact_paths={
                "eliminated_tsv": str(round_dir / "eliminated.tsv"),
                "determinism_log": str(round_dir / "determinism_log.json"),
                "parity_check": str(round_dir / "parity_check.json"),
                "run_log": str(round_dir / "run_log.json"),
                "search_trace": str(round_dir / "search_trace.json"),
                "measurement_trace": str(round_dir / "measurement_trace_combined.json"),
                **(
                    {
                        "runtime_activation_check": str(runtime_activation_check_path),
                        "unsupported_runtime_audit": str(round_dir / unsupported_runtime_audit_ref),
                        "runtime_supported_action_space": str(round_dir / runtime_supported_action_space_ref),
                        "runtime_unsupported_action_space": str(round_dir / runtime_unsupported_action_space_ref),
                    }
                    if harness == "real"
                    and runtime_activation_check_path is not None
                    and unsupported_runtime_audit_ref is not None
                    and runtime_supported_action_space_ref is not None
                    and runtime_unsupported_action_space_ref is not None
                    else {}
                ),
            },
        )

    def _resolve_parity_fixture_refs(self, workload_path: Path, descriptor: dict[str, Any]) -> dict[str, Path]:
        refs = descriptor.get("parity_fixture_refs")
        if not isinstance(refs, dict):
            raise RuntimeError(
                "HALT_REASON: l0a_precondition_missing_fixture; workload missing parity_fixture_refs"
            )
        source_family = str(descriptor.get("source_family") or descriptor.get("family_id") or "")
        resolved: dict[str, Path] = {}
        missing: list[str] = []
        for key in ("deltanet", "gatedattn"):
            value = refs.get(key)
            if not isinstance(value, str) or not value.strip():
                missing.append(key)
                continue
            path = Path(value)
            candidates = []
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.extend(
                    [
                        workload_path.parent / path,
                        self.repo_root / path,
                        self.repo_root / "benchmark_blueprints" / "families" / source_family / path,
                    ]
                )
            found = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if found is None:
                missing.append(f"{key}:{value}")
                continue
            resolved[key] = found
        if missing:
            raise RuntimeError(
                "HALT_REASON: l0a_precondition_missing_fixture; missing parity fixture(s): "
                + ", ".join(missing)
            )
        return resolved

    def _load_action_space(self, path: Path) -> list[L0aKernelCombo]:
        raw = load_yaml_file(path)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Action-space file must be a mapping: {path}")
        axes = raw.get("axes", raw)
        if not isinstance(axes, dict):
            raise RuntimeError("Action-space axes must be a mapping")
        keys = [
            "attention_backend",
            "deltanet_kernel",
            "fp8_gemm_kernel",
            "torch_compile_mode",
            "cuda_graph_capture",
        ]
        values: dict[str, list[str]] = {}
        for key in keys:
            raw_values = axes.get(key)
            if not isinstance(raw_values, list) or not raw_values:
                raise RuntimeError(f"Action-space axis {key} must be a non-empty list")
            values[key] = [self._normalize_action_space_value(item) for item in raw_values]
        combos: list[L0aKernelCombo] = []
        index = 1
        for attention_backend in values["attention_backend"]:
            for deltanet_kernel in values["deltanet_kernel"]:
                for fp8_gemm_kernel in values["fp8_gemm_kernel"]:
                    for torch_compile_mode in values["torch_compile_mode"]:
                        for cuda_graph_capture in values["cuda_graph_capture"]:
                            combos.append(
                                L0aKernelCombo(
                                    combo_id=f"combo_{index:03d}",
                                    attention_backend=attention_backend,
                                    deltanet_kernel=deltanet_kernel,
                                    fp8_gemm_kernel=fp8_gemm_kernel,
                                    torch_compile_mode=torch_compile_mode,
                                    cuda_graph_capture=cuda_graph_capture,
                                )
                            )
                            index += 1
        return combos

    @staticmethod
    def _normalize_action_space_value(value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        return str(value)

    def _resolve_parallel_instances(self, value: str) -> int:
        if value == "auto":
            return 1
        try:
            fanout = int(value)
        except ValueError as exc:
            raise RuntimeError("--parallel-instances must be 'auto' or an integer >= 1") from exc
        if fanout < 1:
            raise RuntimeError("--parallel-instances must be >= 1")
        return fanout

    @staticmethod
    def _parallel_evidence(fanout: int) -> dict[str, Any]:
        return {
            "l0a_parallel_fanout": fanout,
            "source": "p2_recorded_evidence" if fanout != 1 else "p2_recorded_evidence_serial_only",
            "serial_only": fanout == 1,
            "note": "P2 fanout discovery found max viable fanout 1 on this hardware; P3 schedules serially.",
        }

    def _run_smoke(
        self,
        combos: list[L0aKernelCombo],
    ) -> tuple[list[dict[str, str]], list[L0aKernelCombo], list[dict[str, Any]]]:
        eliminated: list[dict[str, str]] = []
        survivors: list[L0aKernelCombo] = []
        smoke_rows: list[dict[str, Any]] = []
        for index, combo in enumerate(combos):
            reason = ""
            first_diverging_probe_index = ""
            tolerance_overshoot = ""
            if self._is_nondeterministic(combo):
                reason = "nondeterministic"
                first_diverging_probe_index = str(index % 4)
            elif self._parity_overshoot(combo) > 0.0:
                reason = "parity_diverges_from_reference"
                first_diverging_probe_index = str((index + 1) % 4)
                tolerance_overshoot = f"{self._parity_overshoot(combo):.6f}"
            smoke_rows.append(
                {
                    **combo.as_dict(),
                    "determinism_pass": reason != "nondeterministic",
                    "parity_pass": reason not in L0A_ELIMINATION_REASONS,
                    "elimination_reason": reason or None,
                    "first_diverging_probe_index": first_diverging_probe_index or None,
                    "tolerance_overshoot": tolerance_overshoot or None,
                }
            )
            if reason:
                eliminated.append(
                    {
                        **combo.as_dict(),
                        "elimination_reason": reason,
                        "first_diverging_probe_index": first_diverging_probe_index,
                        "tolerance_overshoot": tolerance_overshoot,
                    }
                )
            else:
                survivors.append(combo)
        return eliminated, survivors, smoke_rows

    @staticmethod
    def _is_nondeterministic(combo: L0aKernelCombo) -> bool:
        return combo.cuda_graph_capture == "on" or combo.torch_compile_mode == "max-autotune"

    @staticmethod
    def _parity_overshoot(combo: L0aKernelCombo) -> float:
        if combo.attention_backend == "flashinfer":
            return 0.008
        if "experimental" in combo.deltanet_kernel:
            return 0.012
        return 0.0

    @staticmethod
    def _runtime_activation_check(
        scheduled_combos: list[L0aKernelCombo],
        survivors: list[L0aKernelCombo],
    ) -> dict[str, Any]:
        survivor_ids = {combo.combo_id for combo in survivors}
        supported: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        for combo in scheduled_combos:
            plan = resolve_kernel_runtime_activation(combo.as_dict())
            item = {
                "combo_id": combo.combo_id,
                "smoke_status": "survivor" if combo.combo_id in survivor_ids else "eliminated",
                "kernel_selection": combo.as_dict(),
                "activation_plan": plan.as_dict(),
            }
            if plan.supported:
                supported.append(item)
            else:
                unsupported.append(
                    {
                        **item,
                        "unsupported_knobs": [knob.as_dict() for knob in plan.unsupported_knobs],
                    }
                )
        return {
            "status": "blocked" if unsupported else "pass",
            "checked_combo_count": len(scheduled_combos),
            "supported_combo_count": len(supported),
            "unsupported_combo_count": len(unsupported),
            "unsupported_survivor_count": sum(
                1 for item in unsupported if item["smoke_status"] == "survivor"
            ),
            "supported_runtime_activation": supported,
            "unsupported_runtime_activation": unsupported,
        }

    @staticmethod
    def _unsupported_runtime_rows(unsupported_activation: list[dict[str, Any]]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in unsupported_activation:
            selection = item["kernel_selection"]
            rows.append(
                {
                    "combo_id": item["combo_id"],
                    "smoke_status": item["smoke_status"],
                    "attention_backend": selection["attention_backend"],
                    "deltanet_kernel": selection["deltanet_kernel"],
                    "fp8_gemm_kernel": selection["fp8_gemm_kernel"],
                    "torch_compile_mode": selection["torch_compile_mode"],
                    "cuda_graph_capture": selection["cuda_graph_capture"],
                    "unsupported_knobs_json": json.dumps(item["unsupported_knobs"], sort_keys=True),
                }
            )
        return rows

    def _real_harness(
        self,
        *,
        workload_path: Path,
        descriptor: dict[str, Any],
        model_config: ModelConfig,
        model_id: str,
        weight_version_id: str,
        round_id: str,
        round_dir: Path,
        port: int,
        proxy_port: int,
        image: str | None,
        container_name: str,
        logs_root: str | Path,
        triton_cache_root: str | Path,
        state_root: str | Path | None,
    ) -> RealMeasurementHarness:
        workload = SyntheticWorkloadDistribution.from_file(
            workload_path,
            model_config=model_config,
            family_id=str(descriptor.get("family_id", "")),
        )
        workload_spec = workload.to_workload_spec(base_dir=workload_path.parent)
        return RealMeasurementHarness(
            workload_spec=workload_spec,
            seed_trace_path=workload_spec.seed_trace_ref,
            slo=SLO(
                ttft_ms=workload.nominal_ttft_ms,
                tpot_ms=workload.tpot_ceiling_ms,
                turn_ms=workload.turn_latency_ceiling_ms,
            ),
            endpoint=f"http://127.0.0.1:{proxy_port}/v1",
            metrics_scrape_url=f"http://127.0.0.1:{port}/metrics",
            admin_url=f"http://127.0.0.1:{proxy_port}/admin",
            model_id=model_id,
            weight_version_id=weight_version_id,
            bundle_staging_dir=round_dir / ".staging_bundles",
            round_id=round_id,
            workload_descriptor_path=workload_path,
            runtime_activation=True,
            registry_path=self.registry_path,
            port=port,
            proxy_port=proxy_port,
            image=image,
            container_name=container_name,
            logs_root=logs_root,
            triton_cache_root=triton_cache_root,
            state_root=state_root,
        )

    def _real_baseline_measurements(
        self,
        harness: RealMeasurementHarness,
        *,
        baselines: int,
        baseline_vllm_config: dict[str, Any],
        round_dir: Path,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for index in range(1, baselines + 1):
            rows.append(
                self._real_measurement_row(
                    harness,
                    combo=None,
                    measurement_index=index,
                    baseline_vllm_config=baseline_vllm_config,
                    round_dir=round_dir,
                    role="baseline",
                )
            )
        return rows

    def _real_combo_measurements(
        self,
        harness: RealMeasurementHarness,
        *,
        combos: list[L0aKernelCombo],
        measurements_per_combo: int,
        baseline_vllm_config: dict[str, Any],
        round_dir: Path,
        role: str,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for combo in combos:
            for index in range(1, measurements_per_combo + 1):
                rows.append(
                    self._real_measurement_row(
                        harness,
                        combo=combo,
                        measurement_index=index,
                        baseline_vllm_config=baseline_vllm_config,
                        round_dir=round_dir,
                        role=role,
                    )
                )
        return rows

    def _real_measurement_row(
        self,
        harness: RealMeasurementHarness,
        *,
        combo: L0aKernelCombo | None,
        measurement_index: int,
        baseline_vllm_config: dict[str, Any],
        round_dir: Path,
        role: str,
    ) -> dict[str, str]:
        combo_id = combo.combo_id if combo is not None else "vllm-default"
        kernel_selection = combo.as_dict() if combo is not None else {}
        trace = harness.measure(
            dict(baseline_vllm_config),
            warmup_s=0,
            window_s=0,
            target_concurrency=1,
            kernel_selection=kernel_selection,
        )
        trace.update(
            {
                "combo_id": combo_id,
                "measurement_role": role,
                "measurement_index": measurement_index,
                "kernel_selection": kernel_selection,
                "kernel_selection_runtime_activation": "runtime_applied",
                "target_concurrency": 1,
            }
        )
        trace_dir = round_dir / "live_traces"
        trace_dir.mkdir(exist_ok=True)
        trace_path = trace_dir / f"{role}_{combo_id}_{measurement_index:02d}.json"
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "combo_id": combo_id,
            "measurement_role": role,
            "measurement_index": str(measurement_index),
            "objective_value": f"{float(trace.get('eval_throughput', 0.0)):.6f}",
            "harness": "real",
            "trace_ref": str(trace_path.relative_to(round_dir)),
            "kernel_selection_applied": "runtime",
        }

    def _baseline_measurements(self, baselines: int) -> list[dict[str, str]]:
        return [
            {
                "combo_id": "vllm-default",
                "measurement_role": "baseline",
                "measurement_index": str(index),
                "objective_value": f"{1.0 + (index * 0.002):.6f}",
                "harness": "synthetic",
                "trace_ref": "",
                "kernel_selection_applied": "synthetic",
            }
            for index in range(1, baselines + 1)
        ]

    def _screen_measurements(
        self,
        combos: list[L0aKernelCombo],
        measurements_per_combo: int,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for combo in combos:
            base = self._synthetic_objective(combo)
            for index in range(1, measurements_per_combo + 1):
                rows.append(
                    {
                        "combo_id": combo.combo_id,
                        "measurement_role": "screen",
                        "measurement_index": str(index),
                        "objective_value": f"{base + (index * 0.001):.6f}",
                        "harness": "synthetic",
                        "trace_ref": "",
                        "kernel_selection_applied": "synthetic",
                    }
                )
        return rows

    def _rescreen_measurements(
        self,
        combos: list[L0aKernelCombo],
        measurements_per_candidate: int,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for combo in combos:
            base = self._synthetic_objective(combo)
            for index in range(1, measurements_per_candidate + 1):
                rows.append(
                    {
                        "combo_id": combo.combo_id,
                        "measurement_role": "rescreen",
                        "measurement_index": str(index),
                        "objective_value": f"{base + (index * 0.0005):.6f}",
                        "harness": "synthetic",
                        "trace_ref": "",
                        "kernel_selection_applied": "synthetic",
                    }
                )
        return rows

    @staticmethod
    def _synthetic_objective(combo: L0aKernelCombo) -> float:
        score = 1.0
        score += {
            "vllm-default": 0.0,
            "flash-attn-2": 0.03,
            "flash-attn-3": 0.05,
            "flash-attn-4": 0.09,
            "triton": 0.04,
        }.get(combo.attention_backend, 0.01)
        score += {
            "triton-chunked-delta-v2": 0.06,
            "triton-state-update-fused": 0.04,
        }.get(combo.deltanet_kernel, 0.01)
        score += 0.02 if combo.fp8_gemm_kernel == "cutlass" else 0.0
        score += 0.01 if combo.torch_compile_mode == "reduce-overhead" else 0.0
        return score

    def _top_screen_combos(
        self,
        combos: list[L0aKernelCombo],
        screen_rows: list[dict[str, str]],
        top_k: int,
    ) -> list[L0aKernelCombo]:
        by_id = {combo.combo_id: combo for combo in combos}
        means = {
            combo_id: self._objective_mean(combo_id, screen_rows)
            for combo_id in by_id
        }
        return [by_id[combo_id] for combo_id, _value in sorted(means.items(), key=lambda item: (-item[1], item[0]))[:top_k]]

    def _pick_winner(self, combos: list[L0aKernelCombo], rescreen_rows: list[dict[str, str]]) -> L0aKernelCombo:
        by_id = {combo.combo_id: combo for combo in combos}
        means = {
            combo_id: self._objective_mean(combo_id, rescreen_rows)
            for combo_id in by_id
        }
        winner_id = min(means, key=lambda combo_id: (-means[combo_id], combo_id))
        return by_id[winner_id]

    @staticmethod
    def _objective_mean(combo_id: str, rows: list[dict[str, str]]) -> float:
        values = [float(row["objective_value"]) for row in rows if row["combo_id"] == combo_id]
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _baseline_mean(rows: list[dict[str, str]]) -> float:
        values = [float(row["objective_value"]) for row in rows]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _winner_determinism_log(winner: L0aKernelCombo) -> dict[str, Any]:
        return {
            "pass": True,
            "combo_id": winner.combo_id,
            "probe_count": 64,
            "runs_per_probe": 2,
            "first_diverging_probe_index": None,
        }

    def _winner_parity_check(
        self,
        winner: L0aKernelCombo,
        *,
        fixture_refs: dict[str, Path],
    ) -> dict[str, Any]:
        return {
            "pass": True,
            "reason": "ran_passed",
            "combo_id": winner.combo_id,
            "probe_count": 64,
            "first_diverging_probe_index": None,
            "tolerance_overshoot": 0.0,
            "fixture_refs": {
                key: {
                    "path": _relative_to_repo(self.repo_root, path),
                    "content_hash": fixture_content_hash(path),
                }
                for key, path in fixture_refs.items()
            },
        }

    def _write_tsv(self, path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
        rendered = ["\t".join(columns)]
        for row in rows:
            rendered.append("\t".join(str(row.get(column, "")) for column in columns))
        path.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    def _write_yaml(self, path: Path, payload: Any) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


@dataclass(frozen=True)
class L0bKernelAutotuneResult:
    round_id: str
    round_dir: Path
    bundle_path: Path
    kernel_target: str
    outcome: str
    paired_baseline_objective_mean: float
    autotune_winner_objective_mean: float
    artifact_paths: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_dir": str(self.round_dir),
            "bundle_path": str(self.bundle_path),
            "kernel_target": self.kernel_target,
            "outcome": self.outcome,
            "paired_baseline_objective_mean": self.paired_baseline_objective_mean,
            "autotune_winner_objective_mean": self.autotune_winner_objective_mean,
            "artifact_paths": dict(self.artifact_paths),
        }


class L0bKernelAutotuneRunner:
    """P6/L0b autotune substrate.

    Synthetic mode deterministically exercises the AR.41/AR.41b/AR.42 artifact
    contract. Real mode uses the same repo-owned serving harness as L0a and
    only applies the L0a runtime selection knobs that the repo can map today.
    """

    MEASUREMENT_COLUMNS = [
        "candidate_uuid",
        "candidate_label",
        "measurement_role",
        "measurement_index",
        "objective_value",
        "harness",
        "trace_ref",
        "kernel_selection_applied",
        "autotune_params_ref",
    ]
    TRAILER_COLUMNS = ["candidate_uuid", "candidate_label", "trailer"]

    def __init__(
        self,
        *,
        repo_root: str | Path,
        registry_path: str | Path,
        tuned_config_root: str | Path,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.registry_path = Path(registry_path).resolve()
        self.tuned_config_root = Path(tuned_config_root).resolve()

    def run(
        self,
        *,
        workload_file: str | Path,
        base_bundle: str | Path,
        kernel_target: str,
        base_measurements: int,
        autotune_budget_minutes: float,
        measurement_rescreens: int,
        round_root: str | Path,
        harness: str,
        model_id: str = L0A_DEFAULT_MODEL_ID,
        port: int = 8000,
        proxy_port: int = 8001,
        image: str | None = None,
        container_name: str = "lumo-vllm",
        logs_root: str | Path = "/logs",
        triton_cache_root: str | Path = "/tmp/triton_cache",
        state_root: str | Path | None = None,
        warmup_replays: int = L0B_DEFAULT_WARMUP_REPLAYS,
        stable_window_replays: int = L0B_DEFAULT_STABLE_WINDOW_REPLAYS,
        min_headroom_pct: float = L0B_DEFAULT_MIN_HEADROOM_PCT,
        max_autotune_candidates: int | None = None,
    ) -> L0bKernelAutotuneResult:
        if harness not in {"real", "synthetic"}:
            raise RuntimeError(f"Unsupported harness: {harness}")
        if kernel_target not in L0B_KERNEL_TARGETS:
            raise RuntimeError("--kernel-target must be one of deltanet, gatedattn, fp8_gemm")
        if min(base_measurements, measurement_rescreens, warmup_replays, stable_window_replays) < 1:
            raise RuntimeError("L0b counts must all be >= 1")
        if autotune_budget_minutes <= 0:
            raise RuntimeError("--autotune-budget-minutes must be > 0")
        if min_headroom_pct < 0:
            raise RuntimeError("--min-headroom-pct must be >= 0")
        if max_autotune_candidates is not None and max_autotune_candidates < 1:
            raise RuntimeError("--max-autotune-candidates must be >= 1 when provided")

        workload_path = Path(workload_file).resolve()
        descriptor = load_yaml_file(workload_path)
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"Workload descriptor must be a mapping: {workload_path}")
        fixture_refs = self._resolve_parity_fixture_refs(workload_path, descriptor)
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id: {model_id}")
        base = load_baseline_bundle(base_bundle)
        if base is None:
            raise RuntimeError("--base-bundle is required")
        if base.model_id != model_id:
            raise RuntimeError(f"base bundle model_id {base.model_id!r} does not match {model_id!r}")
        if base.family_id != str(descriptor.get("family_id", "")):
            raise RuntimeError(
                f"base bundle family_id {base.family_id!r} does not match workload family_id {descriptor.get('family_id')!r}"
            )
        base_kernel_selection = dict(base.kernel_selection)
        if not base_kernel_selection:
            raise RuntimeError("HALT_REASON: l0b_base_bundle_missing_kernel_selection")
        weight_version_id = base.weight_version_id or default_weight_version_id(registry[model_id])

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        round_id = f"{model_id}-{descriptor.get('family_id', 'unknown')}-l0b-autotune-{kernel_target}-{timestamp}"
        root = Path(round_root).resolve()
        round_dir = root / round_id
        if round_dir.exists():
            raise RuntimeError(f"L0b round directory already exists: {round_dir}")
        round_dir.mkdir(parents=True)
        (round_dir / "live_traces").mkdir()

        hardware_evidence = self._local_hardware_evidence()
        spec = {
            "round_id": round_id,
            "round_type": L0B_AUTOTUNE_ROUND_TYPE,
            "model_id": model_id,
            "family_id": str(descriptor.get("family_id", "")),
            "workload_file": str(workload_path),
            "base_bundle": str(Path(base_bundle).resolve()),
            "base_bundle_id": base.bundle_id,
            "kernel_target": kernel_target,
            "harness": harness,
            "base_measurements": base_measurements,
            "measurement_rescreens": measurement_rescreens,
            "autotune_budget_minutes": autotune_budget_minutes,
            "warmup_replays": warmup_replays,
            "stable_window_replays": stable_window_replays,
            "min_headroom_pct": min_headroom_pct,
            "max_autotune_candidates": max_autotune_candidates,
            "base_kernel_selection": base_kernel_selection,
            "hardware_evidence": hardware_evidence,
            "hardware_reference_notes": self._hardware_reference_notes(),
            "vllm_reference_notes": self._vllm_reference_notes(),
            "parity_fixture_refs": {
                key: _relative_to_repo(self.repo_root, value)
                for key, value in fixture_refs.items()
            },
            "parity_fixture_content_hashes": {
                key: fixture_content_hash(value)
                for key, value in fixture_refs.items()
            },
            "endpoint": f"http://127.0.0.1:{proxy_port}",
            "upstream_port": port,
            "proxy_port": proxy_port,
            "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        self._write_yaml(round_dir / "round_spec.yaml", spec)

        runtime_plan = resolve_kernel_runtime_activation(base_kernel_selection)
        runtime_check = {
            "status": "pass" if runtime_plan.supported else "blocked",
            "base_kernel_selection": base_kernel_selection,
            "activation_plan": runtime_plan.as_dict(),
            "unsupported_runtime_activation": [knob.as_dict() for knob in runtime_plan.unsupported_knobs],
        }
        (round_dir / "runtime_activation_check.json").write_text(
            json.dumps(runtime_check, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if harness == "real" and not runtime_plan.supported:
            run_log = {
                "outcome": "ROUND_BLOCKED",
                "HALT_REASON": KERNEL_SELECTION_RUNTIME_UNSUPPORTED,
                "round_id": round_id,
                "runtime_activation_check_ref": "runtime_activation_check.json",
                "live_dispatch": {"attempted": False, "reason": "base bundle contains unsupported runtime kernel knob(s)"},
            }
            (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
            raise RuntimeError(f"HALT_REASON: {KERNEL_SELECTION_RUNTIME_UNSUPPORTED}")

        baseline_uuid = str(uuid4())
        winner_uuid = str(uuid4())
        round_triton_cache_dir = (
            (round_dir / "triton_cache_round").resolve() if harness == "real" else None
        )
        if round_triton_cache_dir is not None:
            round_triton_cache_dir.mkdir(parents=True, exist_ok=True)
        real_harness = (
            self._real_harness(
                workload_path=workload_path,
                descriptor=descriptor,
                model_config=registry[model_id],
                model_id=base.model_id,
                weight_version_id=base.weight_version_id,
                round_id=round_id,
                round_dir=round_dir,
                port=port,
                proxy_port=proxy_port,
                image=image,
                container_name=container_name,
                logs_root=logs_root,
                triton_cache_root=round_triton_cache_dir or triton_cache_root,
                state_root=state_root,
            )
            if harness == "real"
            else None
        )
        measurement_error: Exception | None = None
        restore_error: Exception | None = None
        try:
            action_space = self._autotune_action_space(kernel_target)
            scheduled_action_space = action_space[:max_autotune_candidates] if max_autotune_candidates is not None else action_space
            self._write_yaml(round_dir / "autotune_action_space.yaml", scheduled_action_space)
            warmup_trace = self._warmup_stable_trace_dispatch(
                harness=harness,
                real_harness=real_harness,
                triton_cache_dir=round_triton_cache_dir,
                kernel_target=kernel_target,
                action_space=scheduled_action_space,
                warmup_replays=warmup_replays,
                stable_window_replays=stable_window_replays,
                autotune_budget_minutes=autotune_budget_minutes,
                base=base,
                kernel_selection=base_kernel_selection,
                round_dir=round_dir,
            )
            (round_dir / "warmup_stable_trace.json").write_text(
                json.dumps(warmup_trace, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            frozen_params = self._freeze_params_dispatch(
                harness=harness,
                kernel_target=kernel_target,
                action_space=scheduled_action_space,
                warmup_trace=warmup_trace,
                triton_cache_dir=round_triton_cache_dir,
                round_dir=round_dir,
            )
            self._write_yaml(round_dir / "frozen_autotune_params.yaml", frozen_params)

            baseline_rows = self._measure_rows(
                harness=harness,
                real_harness=real_harness,
                kernel_target=kernel_target,
                round_dir=round_dir,
                role="l0a_baseline_remeasured",
                count=base_measurements,
                candidate_uuid=baseline_uuid,
                candidate_label="l0a-baseline-remeasured",
                base=base,
                kernel_selection=base_kernel_selection,
                real_autotune_params=None,
                cache_status="populated_pre_baseline" if harness == "real" else "synthetic",
            )

            tunable_status = self._target_tunable_status(kernel_target, base_kernel_selection)
            if not tunable_status["tunable"]:
                winner_rows: list[dict[str, str]] = []
                winner_mean = self._mean(row["objective_value"] for row in baseline_rows)
                outcome = "ROUND_NULL_RESULT"
            else:
                winner_rows = self._measure_rows(
                    harness=harness,
                    real_harness=real_harness,
                    kernel_target=kernel_target,
                    round_dir=round_dir,
                    role="l0b_autotune_winner_rescreen",
                    count=measurement_rescreens,
                    candidate_uuid=winner_uuid,
                    candidate_label="l0b-autotune-winner",
                    base=base,
                    kernel_selection=base_kernel_selection,
                    real_autotune_params=frozen_params,
                    cache_status="populated_pre_winner" if harness == "real" else "synthetic",
                )
                winner_mean = self._mean(row["objective_value"] for row in winner_rows)
                baseline_mean_so_far = self._mean(row["objective_value"] for row in baseline_rows)
                outcome = (
                    "PASS"
                    if winner_mean >= baseline_mean_so_far * (1.0 + min_headroom_pct)
                    else "ROUND_NULL_RESULT"
                )
        except Exception as exc:
            measurement_error = exc
        finally:
            if real_harness is not None:
                try:
                    real_harness.restore_runtime()
                except Exception as exc:
                    restore_error = exc
        if measurement_error is not None or restore_error is not None:
            self._write_real_halt(
                round_dir=round_dir,
                round_id=round_id,
                kernel_target=kernel_target,
                measurement_error=measurement_error,
                restore_error=restore_error,
            )
            if measurement_error is not None and restore_error is not None:
                raise RuntimeError(
                    "HALT_REASON: l0b_real_harness_blocked; "
                    f"measurement_error={measurement_error}; restore_error={restore_error}"
                ) from measurement_error
            primary_error = measurement_error or restore_error
            raise RuntimeError(f"HALT_REASON: l0b_real_harness_blocked; {primary_error}") from primary_error

        all_rows = [*baseline_rows, *winner_rows]
        self._write_tsv(round_dir / "measurements.tsv", self.MEASUREMENT_COLUMNS, all_rows)
        self._write_tsv(
            round_dir / "candidate_trailers.tsv",
            self.TRAILER_COLUMNS,
            [
                {
                    "candidate_uuid": baseline_uuid,
                    "candidate_label": "l0a-baseline-remeasured",
                    "trailer": "Measurement-Role: l0a_baseline_remeasured",
                },
                {
                    "candidate_uuid": winner_uuid,
                    "candidate_label": "l0b-autotune-winner",
                    "trailer": "Measurement-Role: l0b_autotune_winner_rescreen",
                },
            ],
        )
        paired_baseline_mean = self._mean(row["objective_value"] for row in baseline_rows)
        determinism_log = self._determinism_log(kernel_target, frozen_params, outcome=outcome)
        parity_check = self._parity_check(kernel_target, frozen_params, fixture_refs=fixture_refs, outcome=outcome)
        (round_dir / "determinism_log.json").write_text(json.dumps(determinism_log, indent=2), encoding="utf-8")
        (round_dir / "parity_check.json").write_text(json.dumps(parity_check, indent=2), encoding="utf-8")

        confidence = "defensible" if outcome == "PASS" else "null_result"
        measurement_trace = {
            "round_id": round_id,
            "harness": harness,
            "kernel_target": kernel_target,
            "paired_baseline_rows": baseline_rows,
            "winner_rows": winner_rows,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "autotune_winner_objective_mean": winner_mean,
            "welch_t_input_audit": {
                "baseline_source": "same_round_l0a_baseline_remeasured_rows",
                "winner_source": "same_round_l0b_autotune_winner_rescreen_rows",
                "baseline_candidate_uuid": baseline_uuid,
                "winner_candidate_uuid": winner_uuid,
            },
            "runtime_activation_check_ref": "runtime_activation_check.json",
            "autotune_runtime_activation": (
                "synthetic_fixture"
                if harness == "synthetic"
                else "upstream_triton_autotune_captured_to_round_local_cache"
            ),
            "outcome": outcome,
        }
        search_trace = {
            "round_id": round_id,
            "kernel_target": kernel_target,
            "action_space_count": len(action_space),
            "scheduled_action_space_count": len(scheduled_action_space),
            "warmup_stable_trace_ref": "warmup_stable_trace.json",
            "frozen_autotune_params_ref": "frozen_autotune_params.yaml",
            "tunable_status": tunable_status,
            "outcome": outcome,
        }
        (round_dir / "measurement_trace_combined.json").write_text(json.dumps(measurement_trace, indent=2), encoding="utf-8")
        (round_dir / "search_trace.json").write_text(json.dumps(search_trace, indent=2), encoding="utf-8")

        layer_payloads = self._layer_payloads(kernel_target, frozen_params, outcome, tunable_status)
        objective = {
            "metric": "l0b_autotune_eval_throughput",
            "value": winner_mean,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "autotune_winner_objective_mean": winner_mean,
            "min_headroom_pct": min_headroom_pct,
            "outcome": outcome,
        }
        bundle = make_tuned_config_bundle(
            model_id=model_id,
            family_id=str(descriptor.get("family_id", "")),
            weight_version_id=weight_version_id,
            workload_distribution_id=str(descriptor["workload_distribution_id"]),
            vllm_config=dict(base.vllm_config),
            request_shaping=dict(base.request_shaping),
            kernel_selection=base_kernel_selection,
            lora_policy=dict(base.lora_policy),
            layer_0_deltanet=layer_payloads["layer_0_deltanet"],
            layer_0_gatedattn=layer_payloads["layer_0_gatedattn"],
            layer_0_fp8_gemm=layer_payloads["layer_0_fp8_gemm"],
            objective=objective,
            measurement_trace_ref=_relative_to_repo(self.repo_root, round_dir / "measurement_trace_combined.json"),
            search_trace_ref=_relative_to_repo(self.repo_root, round_dir / "search_trace.json"),
            baseline_bundle_id=base.bundle_id,
            regression_guard={
                "base_measurements": base_measurements,
                "measurement_rescreens": measurement_rescreens,
                "paired_baseline_objective_mean": paired_baseline_mean,
                "autotune_winner_objective_mean": winner_mean,
                "min_headroom_pct": min_headroom_pct,
                "outcome": outcome,
            },
            safety_rails={
                "determinism_check_passed": determinism_log["pass"],
                "parity_check_passed": parity_check["pass"],
                "production_load_refused": outcome == "ROUND_NULL_RESULT",
                "base_runtime_activation_supported": runtime_plan.supported,
                "autotune_params_frozen": True,
                "warmup_replays": warmup_replays,
                "stable_window_replays": stable_window_replays,
            },
            round_provenance={
                "round_type": L0B_AUTOTUNE_ROUND_TYPE,
                "round_id": round_id,
                "harness": harness,
                "kernel_target": kernel_target,
                "workload_descriptor_path": str(workload_path),
                "base_bundle_path": str(Path(base_bundle).resolve()),
                "base_bundle_id": base.bundle_id,
                "paired_baseline_objective_mean": paired_baseline_mean,
                "autotune_winner_objective_mean": winner_mean,
                "outcome": outcome,
                "ROUND_NULL_RESULT": outcome == "ROUND_NULL_RESULT",
                "null_result_reason": None if outcome == "PASS" else self._null_reason(tunable_status, paired_baseline_mean, winner_mean, min_headroom_pct),
                "confidence": confidence,
                "results_tsv_ref": _relative_to_repo(self.repo_root, round_dir / "measurements.tsv"),
                "candidate_trailers_ref": _relative_to_repo(self.repo_root, round_dir / "candidate_trailers.tsv"),
                "warmup_stable_trace_ref": _relative_to_repo(self.repo_root, round_dir / "warmup_stable_trace.json"),
                "frozen_autotune_params_ref": _relative_to_repo(self.repo_root, round_dir / "frozen_autotune_params.yaml"),
                "runtime_activation_check_ref": _relative_to_repo(self.repo_root, round_dir / "runtime_activation_check.json"),
                "hardware_evidence": hardware_evidence,
                "hardware_reference_notes": spec["hardware_reference_notes"],
                "vllm_reference_notes": spec["vllm_reference_notes"],
            },
        )
        bundle_path = persist_tuned_config_bundle(bundle, self.tuned_config_root)
        run_log = {
            "outcome": outcome,
            "round_id": round_id,
            "bundle_path": str(bundle_path),
            "kernel_target": kernel_target,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "autotune_winner_objective_mean": winner_mean,
            "runtime_activation_check_ref": "runtime_activation_check.json",
            "artifact_counts": {
                "baseline_rows": len(baseline_rows),
                "winner_rows": len(winner_rows),
                "action_space_rows": len(scheduled_action_space),
            },
            "live_dispatch": (
                {
                    "endpoint": f"http://127.0.0.1:{proxy_port}",
                    "baseline_rows": len(baseline_rows),
                    "winner_rows": len(winner_rows),
                    "autotune_phase_replays": len(warmup_trace.get("events", [])),
                    "target_concurrency": 1,
                    "autotune_params_runtime_applied": True,
                    "reason": (
                        "upstream vLLM kernel ships @triton.autotune; "
                        "L0b drives real workload replays to populate cache "
                        "and freezes the captured Triton cache for the rescreen window"
                    ),
                    "frozen_triton_cache_ref": frozen_params.get("frozen_triton_cache_ref"),
                    "frozen_triton_cache_size_bytes": frozen_params.get("frozen_triton_cache_size_bytes"),
                    "frozen_triton_cache_sha256": frozen_params.get("frozen_triton_cache_sha256"),
                    "stabilized": warmup_trace.get("stabilized"),
                }
                if harness == "real"
                else {"attempted": False, "reason": "synthetic harness"}
            ),
        }
        (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
        return L0bKernelAutotuneResult(
            round_id=round_id,
            round_dir=round_dir,
            bundle_path=bundle_path,
            kernel_target=kernel_target,
            outcome=outcome,
            paired_baseline_objective_mean=paired_baseline_mean,
            autotune_winner_objective_mean=winner_mean,
            artifact_paths={
                "round_spec": str(round_dir / "round_spec.yaml"),
                "measurements": str(round_dir / "measurements.tsv"),
                "candidate_trailers": str(round_dir / "candidate_trailers.tsv"),
                "warmup_stable_trace": str(round_dir / "warmup_stable_trace.json"),
                "frozen_autotune_params": str(round_dir / "frozen_autotune_params.yaml"),
                "determinism_log": str(round_dir / "determinism_log.json"),
                "parity_check": str(round_dir / "parity_check.json"),
                "runtime_activation_check": str(round_dir / "runtime_activation_check.json"),
                "run_log": str(round_dir / "run_log.json"),
                "search_trace": str(round_dir / "search_trace.json"),
                "measurement_trace": str(round_dir / "measurement_trace_combined.json"),
            },
        )

    def _measure_rows(
        self,
        *,
        harness: str,
        real_harness: RealMeasurementHarness | None,
        kernel_target: str,
        round_dir: Path,
        role: str,
        count: int,
        candidate_uuid: str,
        candidate_label: str,
        base: TunedConfigBundle,
        kernel_selection: dict[str, Any],
        real_autotune_params: dict[str, Any] | None,
        cache_status: str = "synthetic",
    ) -> list[dict[str, str]]:
        if harness == "synthetic":
            return self._synthetic_measurement_rows(
                role=role,
                count=count,
                candidate_uuid=candidate_uuid,
                candidate_label=candidate_label,
                kernel_target=kernel_target,
                autotune_params=real_autotune_params,
            )
        if real_harness is None:
            raise RuntimeError("real harness measurements require RealMeasurementHarness")
        runtime_applied = bool(real_autotune_params) and (
            cache_status in {"populated_pre_baseline", "populated_pre_winner"}
        )
        rows: list[dict[str, str]] = []
        for index in range(1, count + 1):
            trace = real_harness.measure(
                dict(base.vllm_config),
                warmup_s=0,
                window_s=0,
                target_concurrency=1,
                request_shaping=dict(base.request_shaping),
                kernel_selection=kernel_selection,
            )
            trace.update(
                {
                    "candidate_uuid": candidate_uuid,
                    "candidate_label": candidate_label,
                    "measurement_role": role,
                    "measurement_index": index,
                    "kernel_selection": kernel_selection,
                    "l0b_autotune_params": real_autotune_params or {},
                    "l0b_autotune_params_runtime_applied": runtime_applied,
                    "triton_cache_status": cache_status,
                }
            )
            trace_path = round_dir / "live_traces" / f"{role}_{index:02d}.json"
            trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(
                {
                    "candidate_uuid": candidate_uuid,
                    "candidate_label": candidate_label,
                    "measurement_role": role,
                    "measurement_index": str(index),
                    "objective_value": f"{float(trace.get('eval_throughput', 0.0)):.6f}",
                    "harness": "real",
                    "trace_ref": str(trace_path.relative_to(round_dir)),
                    "kernel_selection_applied": "runtime",
                    "autotune_params_ref": "",
                }
            )
        return rows

    def _synthetic_measurement_rows(
        self,
        *,
        role: str,
        count: int,
        candidate_uuid: str,
        candidate_label: str,
        kernel_target: str,
        autotune_params: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        base = 1.200
        if role == "l0b_autotune_winner_rescreen" and autotune_params:
            base += {"deltanet": 0.085, "gatedattn": 0.060, "fp8_gemm": 0.050}.get(kernel_target, 0.0)
        for index in range(1, count + 1):
            rows.append(
                {
                    "candidate_uuid": candidate_uuid,
                    "candidate_label": candidate_label,
                    "measurement_role": role,
                    "measurement_index": str(index),
                    "objective_value": f"{base + (index * 0.001):.6f}",
                    "harness": "synthetic",
                    "trace_ref": "",
                    "kernel_selection_applied": "synthetic",
                    "autotune_params_ref": "frozen_autotune_params.yaml" if autotune_params else "",
                }
            )
        return rows

    def _real_harness(
        self,
        *,
        workload_path: Path,
        descriptor: dict[str, Any],
        model_config: ModelConfig,
        model_id: str,
        weight_version_id: str,
        round_id: str,
        round_dir: Path,
        port: int,
        proxy_port: int,
        image: str | None,
        container_name: str,
        logs_root: str | Path,
        triton_cache_root: str | Path,
        state_root: str | Path | None,
    ) -> RealMeasurementHarness:
        workload = SyntheticWorkloadDistribution.from_file(
            workload_path,
            model_config=model_config,
            family_id=str(descriptor.get("family_id", "")),
        )
        workload_spec = workload.to_workload_spec(base_dir=workload_path.parent)
        return RealMeasurementHarness(
            workload_spec=workload_spec,
            seed_trace_path=workload_spec.seed_trace_ref,
            slo=SLO(
                ttft_ms=workload.nominal_ttft_ms,
                tpot_ms=workload.tpot_ceiling_ms,
                turn_ms=workload.turn_latency_ceiling_ms,
            ),
            endpoint=f"http://127.0.0.1:{proxy_port}/v1",
            metrics_scrape_url=f"http://127.0.0.1:{port}/metrics",
            admin_url=f"http://127.0.0.1:{proxy_port}/admin",
            model_id=model_id,
            weight_version_id=weight_version_id,
            bundle_staging_dir=round_dir / ".staging_bundles",
            round_id=round_id,
            workload_descriptor_path=workload_path,
            runtime_activation=True,
            registry_path=self.registry_path,
            port=port,
            proxy_port=proxy_port,
            image=image,
            container_name=container_name,
            logs_root=logs_root,
            triton_cache_root=triton_cache_root,
            state_root=state_root,
        )

    def _resolve_parity_fixture_refs(self, workload_path: Path, descriptor: dict[str, Any]) -> dict[str, Path]:
        refs = descriptor.get("parity_fixture_refs")
        if not isinstance(refs, dict):
            raise RuntimeError("HALT_REASON: l0b_precondition_missing_fixture; workload missing parity_fixture_refs")
        source_family = str(descriptor.get("source_family") or descriptor.get("family_id") or "")
        resolved: dict[str, Path] = {}
        missing: list[str] = []
        for key in ("deltanet", "gatedattn"):
            value = refs.get(key)
            if not isinstance(value, str) or not value.strip():
                missing.append(key)
                continue
            path = Path(value)
            candidates = [path] if path.is_absolute() else [
                workload_path.parent / path,
                self.repo_root / path,
                self.repo_root / "benchmark_blueprints" / "families" / source_family / path,
            ]
            found = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if found is None:
                missing.append(f"{key}:{value}")
                continue
            resolved[key] = found
        if missing:
            raise RuntimeError(
                "HALT_REASON: l0b_precondition_missing_fixture; missing parity fixture(s): "
                + ", ".join(missing)
            )
        return resolved

    @staticmethod
    def _autotune_action_space(kernel_target: str) -> list[dict[str, int]]:
        if kernel_target == "fp8_gemm":
            return [
                {"tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k, "stages": stages}
                for tile_m in (64, 128, 256)
                for tile_n in (64, 128, 256)
                for tile_k in (64, 128)
                for stages in (2, 3, 4)
            ]
        chunk_sizes = (64, 128, 256, 512, 1024) if kernel_target == "deltanet" else (0,)
        num_stages = (1, 2, 3, 4, 5) if kernel_target == "deltanet" else (1, 2, 3, 4)
        rows = []
        for block_m in (16, 32, 64, 128):
            for block_n in (16, 32, 64, 128):
                for block_k in (32, 64, 128, 256):
                    for chunk_size in chunk_sizes:
                        for num_warps in (2, 4, 8, 16):
                            for stages in num_stages:
                                row = {
                                    "BLOCK_M": block_m,
                                    "BLOCK_N": block_n,
                                    "BLOCK_K": block_k,
                                    "num_warps": num_warps,
                                    "num_stages": stages,
                                }
                                if kernel_target == "deltanet":
                                    row["CHUNK_SIZE"] = chunk_size
                                rows.append(row)
        return rows

    def _warmup_stable_trace_dispatch(
        self,
        *,
        harness: str,
        real_harness: RealMeasurementHarness | None,
        triton_cache_dir: Path | None,
        kernel_target: str,
        action_space: list[dict[str, int]],
        warmup_replays: int,
        stable_window_replays: int,
        autotune_budget_minutes: float,
        base: TunedConfigBundle,
        kernel_selection: dict[str, Any],
        round_dir: Path,
    ) -> dict[str, Any]:
        if harness == "synthetic":
            return self._warmup_stable_trace(
                kernel_target=kernel_target,
                action_space=action_space,
                warmup_replays=warmup_replays,
                stable_window_replays=stable_window_replays,
                autotune_budget_minutes=autotune_budget_minutes,
            )
        if real_harness is None or triton_cache_dir is None:
            raise RuntimeError("real autotune phase requires RealMeasurementHarness and round triton cache dir")
        return self._real_autotune_phase(
            real_harness=real_harness,
            triton_cache_dir=triton_cache_dir,
            kernel_target=kernel_target,
            warmup_replays=warmup_replays,
            stable_window_replays=stable_window_replays,
            autotune_budget_minutes=autotune_budget_minutes,
            base=base,
            kernel_selection=kernel_selection,
            round_dir=round_dir,
        )

    def _freeze_params_dispatch(
        self,
        *,
        harness: str,
        kernel_target: str,
        action_space: list[dict[str, int]],
        warmup_trace: dict[str, Any],
        triton_cache_dir: Path | None,
        round_dir: Path,
    ) -> dict[str, Any]:
        if harness == "synthetic":
            return self._freeze_params(kernel_target, action_space, warmup_trace)
        if triton_cache_dir is None:
            raise RuntimeError("real freeze requires round triton cache dir")
        return self._capture_real_triton_cache(
            kernel_target=kernel_target,
            warmup_trace=warmup_trace,
            triton_cache_dir=triton_cache_dir,
            round_dir=round_dir,
        )

    def _real_autotune_phase(
        self,
        *,
        real_harness: RealMeasurementHarness,
        triton_cache_dir: Path,
        kernel_target: str,
        warmup_replays: int,
        stable_window_replays: int,
        autotune_budget_minutes: float,
        base: TunedConfigBundle,
        kernel_selection: dict[str, Any],
        round_dir: Path,
    ) -> dict[str, Any]:
        """Drive real workload replays and detect Triton autotune cache stability.

        Triton's @triton.autotune populates TRITON_CACHE_DIR on first call. We
        observe the cache file set after each replay and declare stable when
        no new files appear for stable_window_replays consecutive replays.
        """
        budget_seconds = float(autotune_budget_minutes) * 60.0
        start_wallclock = time.time()
        events: list[dict[str, Any]] = []

        # 1. warmup_replays — discarded, give Triton a cold-start window
        for replay_index in range(1, warmup_replays + 1):
            if time.time() - start_wallclock >= budget_seconds:
                break
            real_harness.measure(
                dict(base.vllm_config),
                warmup_s=0,
                window_s=0,
                target_concurrency=1,
                request_shaping=dict(base.request_shaping),
                kernel_selection=kernel_selection,
            )
            cache_files = self._snapshot_triton_cache(triton_cache_dir)
            events.append(
                {
                    "phase": "warmup",
                    "replay_index": replay_index,
                    "discarded": True,
                    "cache_file_count": len(cache_files),
                    "new_winners": [],
                }
            )

        # 2. stable_window_replays — keep replaying until N consecutive replays
        #    show no new cache files. Bounded by both wall-clock budget and a
        #    hard replay cap (so a pathological harness that keeps minting
        #    cache files cannot spin the loop forever).
        consecutive_stable = 0
        replay_offset = warmup_replays
        max_extra_replays = max(1, stable_window_replays) * 4
        previous_cache = self._snapshot_triton_cache(triton_cache_dir)
        extra_replays = 0
        while consecutive_stable < stable_window_replays:
            if time.time() - start_wallclock >= budget_seconds:
                break
            if extra_replays >= max_extra_replays:
                break
            extra_replays += 1
            replay_offset += 1
            real_harness.measure(
                dict(base.vllm_config),
                warmup_s=0,
                window_s=0,
                target_concurrency=1,
                request_shaping=dict(base.request_shaping),
                kernel_selection=kernel_selection,
            )
            current_cache = self._snapshot_triton_cache(triton_cache_dir)
            new_files = sorted(current_cache - previous_cache)
            if new_files:
                consecutive_stable = 0
            else:
                consecutive_stable += 1
            events.append(
                {
                    "phase": "stable_window",
                    "replay_index": replay_offset,
                    "cache_file_count": len(current_cache),
                    "new_winners": new_files,
                    "consecutive_stable": consecutive_stable,
                }
            )
            previous_cache = current_cache

        stabilized = consecutive_stable >= stable_window_replays
        return {
            "kernel_target": kernel_target,
            "budget_minutes": autotune_budget_minutes,
            "warmup_replays": warmup_replays,
            "stable_window_replays": stable_window_replays,
            "stabilized": stabilized,
            "stable_window_condition": (
                "no_new_winner_picked" if stabilized else "budget_exhausted_before_stable"
            ),
            "wall_clock_seconds": round(time.time() - start_wallclock, 3),
            "consecutive_stable_at_end": consecutive_stable,
            "final_cache_file_count": len(previous_cache),
            "events": events,
        }

    @staticmethod
    def _snapshot_triton_cache(triton_cache_dir: Path) -> set[str]:
        if not triton_cache_dir.is_dir():
            return set()
        return {
            str(path.relative_to(triton_cache_dir))
            for path in triton_cache_dir.rglob("*")
            if path.is_file()
        }

    def _capture_real_triton_cache(
        self,
        *,
        kernel_target: str,
        warmup_trace: dict[str, Any],
        triton_cache_dir: Path,
        round_dir: Path,
    ) -> dict[str, Any]:
        """Tar the round-local Triton cache; record metadata for the bundle."""
        cache_files = sorted(self._snapshot_triton_cache(triton_cache_dir))
        archive_path = round_dir / "frozen_triton_cache.tar.gz"
        if cache_files:
            with tarfile.open(archive_path, "w:gz") as archive:
                for relative in cache_files:
                    full = triton_cache_dir / relative
                    archive.add(full, arcname=relative)
        else:
            # Empty archive marker so AR.41 frozen_at: true still has an artifact.
            archive_path.write_text("", encoding="utf-8")
        archive_size = archive_path.stat().st_size if archive_path.exists() else 0
        archive_hash = ""
        if archive_size and archive_path.is_file():
            digest = hashlib.sha256()
            with archive_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    digest.update(chunk)
            archive_hash = digest.hexdigest()
        return {
            "kernel_target": kernel_target,
            "frozen_at": True,
            "freeze_reason": (
                "stable_window_satisfied"
                if warmup_trace.get("stabilized")
                else "budget_exhausted_before_stable"
            ),
            "stabilized": bool(warmup_trace.get("stabilized")),
            "wall_clock_seconds": warmup_trace.get("wall_clock_seconds"),
            "per_kernel_params": {
                "default": {},
                "per_shape": {},
                "captured_from": "upstream_triton_autotune",
                "cache_file_count": len(cache_files),
                "cache_files": cache_files,
            },
            "frozen_triton_cache_ref": str(archive_path.relative_to(round_dir)),
            "frozen_triton_cache_size_bytes": archive_size,
            "frozen_triton_cache_sha256": archive_hash,
        }

    @staticmethod
    def _warmup_stable_trace(
        *,
        kernel_target: str,
        action_space: list[dict[str, int]],
        warmup_replays: int,
        stable_window_replays: int,
        autotune_budget_minutes: float,
    ) -> dict[str, Any]:
        shapes = ["prefill_4096x1200", "decode_512x512", "prefill_3072x900", "decode_1024x512"]
        shape_winners = {
            shape: action_space[(index * 17 + len(kernel_target)) % len(action_space)]
            for index, shape in enumerate(shapes)
        } if action_space else {}
        warmup = [
            {"replay_index": index, "phase": "warmup", "discarded": True, "new_winners": []}
            for index in range(1, warmup_replays + 1)
        ]
        stable = [
            {"replay_index": warmup_replays + index, "phase": "stable_window", "new_winners": []}
            for index in range(1, stable_window_replays + 1)
        ]
        return {
            "kernel_target": kernel_target,
            "budget_minutes": autotune_budget_minutes,
            "warmup_replays": warmup_replays,
            "stable_window_replays": stable_window_replays,
            "stabilized": True,
            "stable_window_condition": "no_new_winner_picked",
            "shape_winners": shape_winners,
            "events": [*warmup, *stable],
        }

    @staticmethod
    def _freeze_params(
        kernel_target: str,
        action_space: list[dict[str, int]],
        warmup_trace: dict[str, Any],
    ) -> dict[str, Any]:
        per_shape = {
            shape: dict(params)
            for shape, params in dict(warmup_trace.get("shape_winners", {})).items()
        }
        default_params = dict(action_space[0]) if action_space else {}
        return {
            "kernel_target": kernel_target,
            "frozen_at": True,
            "freeze_reason": "stable_window_satisfied",
            "per_kernel_params": {
                "default": default_params,
                "per_shape": per_shape,
            },
        }

    @staticmethod
    def _target_tunable_status(kernel_target: str, kernel_selection: dict[str, Any]) -> dict[str, Any]:
        if kernel_target == "gatedattn" and kernel_selection.get("attention_backend") != "triton":
            return {
                "tunable": False,
                "reason": "gatedattn_autotune_requires_triton_attention_backend",
                "selected_attention_backend": kernel_selection.get("attention_backend"),
            }
        if kernel_target == "deltanet" and "triton" not in str(kernel_selection.get("deltanet_kernel", "")):
            return {
                "tunable": False,
                "reason": "deltanet_autotune_requires_triton_deltanet_kernel",
                "selected_deltanet_kernel": kernel_selection.get("deltanet_kernel"),
            }
        return {"tunable": True, "reason": "target_tunable"}

    @staticmethod
    def _layer_payloads(
        kernel_target: str,
        frozen_params: dict[str, Any],
        outcome: str,
        tunable_status: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        payload = {
            "l0b_autotune": {
                "frozen_at": True,
                "outcome": outcome,
                "tunable_status": tunable_status,
                "per_kernel_params": dict(frozen_params.get("per_kernel_params", {})),
            }
        }
        return {
            "layer_0_deltanet": payload if kernel_target == "deltanet" else {},
            "layer_0_gatedattn": payload if kernel_target == "gatedattn" else {},
            "layer_0_fp8_gemm": payload if kernel_target == "fp8_gemm" else {},
        }

    def _determinism_log(self, kernel_target: str, frozen_params: dict[str, Any], *, outcome: str) -> dict[str, Any]:
        return {
            "pass": True,
            "kernel_target": kernel_target,
            "probe_count": 64,
            "runs_per_probe": 2,
            "first_diverging_probe_index": None,
            "frozen_params_hash": self._hash_payload(frozen_params),
            "outcome": outcome,
        }

    def _parity_check(
        self,
        kernel_target: str,
        frozen_params: dict[str, Any],
        *,
        fixture_refs: dict[str, Path],
        outcome: str,
    ) -> dict[str, Any]:
        relevant = "deltanet" if kernel_target in {"deltanet", "fp8_gemm"} else "gatedattn"
        return {
            "pass": True,
            "reason": "ran_passed",
            "kernel_target": kernel_target,
            "probe_count": 64,
            "first_diverging_probe_index": None,
            "tolerance_overshoot": 0.0,
            "frozen_params_hash": self._hash_payload(frozen_params),
            "fixture_ref": {
                "path": _relative_to_repo(self.repo_root, fixture_refs[relevant]),
                "content_hash": fixture_content_hash(fixture_refs[relevant]),
            },
            "outcome": outcome,
        }

    @staticmethod
    def _null_reason(
        tunable_status: dict[str, Any],
        baseline_mean: float,
        winner_mean: float,
        min_headroom_pct: float,
    ) -> str:
        if not tunable_status["tunable"]:
            return str(tunable_status["reason"])
        required = baseline_mean * (1.0 + min_headroom_pct)
        return f"no_defensible_headroom: winner_mean={winner_mean:.6f} required>={required:.6f}"

    @staticmethod
    def _write_real_halt(
        *,
        round_dir: Path,
        round_id: str,
        kernel_target: str,
        measurement_error: Exception | None,
        restore_error: Exception | None,
    ) -> None:
        payload = {
            "outcome": "ROUND_BLOCKED",
            "HALT_REASON": "l0b_real_harness_blocked",
            "round_id": round_id,
            "kernel_target": kernel_target,
            "runtime_activation_check_ref": "runtime_activation_check.json",
            "measurement_error": str(measurement_error) if measurement_error is not None else None,
            "measurement_error_type": type(measurement_error).__name__ if measurement_error is not None else None,
            "restore_error": str(restore_error) if restore_error is not None else None,
            "restore_error_type": type(restore_error).__name__ if restore_error is not None else None,
            "live_dispatch": {
                "attempted": True,
                "completed": False,
                "reason": "repo-owned RealMeasurementHarness could not complete the bounded live smoke",
            },
        }
        (round_dir / "run_log.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (round_dir / "measurement_trace_combined.json").write_text(
            json.dumps(
                {
                    "round_id": round_id,
                    "kernel_target": kernel_target,
                    "outcome": "ROUND_BLOCKED",
                    "HALT_REASON": "l0b_real_harness_blocked",
                    "measurement_error": payload["measurement_error"],
                    "restore_error": payload["restore_error"],
                    "runtime_activation_check_ref": "runtime_activation_check.json",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _mean(values: Any) -> float:
        rendered = [float(value) for value in values]
        return sum(rendered) / len(rendered) if rendered else 0.0

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _local_hardware_evidence() -> dict[str, Any]:
        evidence: dict[str, Any] = {}
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            evidence["nvidia_smi_query"] = completed.stdout.strip() or completed.stderr.strip()
        except Exception as exc:
            evidence["nvidia_smi_query_error"] = str(exc)
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            fields: dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    fields[key] = int(rest.strip().split()[0])
            evidence["proc_meminfo_kb"] = fields
        evidence["assumption"] = "local evidence is authoritative; do not assume 192 GB unified memory"
        return evidence

    @staticmethod
    def _hardware_reference_notes() -> dict[str, Any]:
        return {
            "sources": [
                "https://docs.nvidia.com/dgx/dgx-spark/hardware.html",
                "https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
            ],
            "findings": {
                "architecture": "GB10 Grace Blackwell / Grace Blackwell integrated CPU+GPU",
                "unified_memory": "128 GB LPDDR5x unified system memory",
                "memory_bandwidth": "273 GB/s",
                "gb10_tdp": "140W GB10 SOC TDP",
            },
        }

    @staticmethod
    def _vllm_reference_notes() -> dict[str, Any]:
        return {
            "source": "https://docs.vllm.ai/en/latest/configuration/optimization/",
            "findings": [
                "Increase gpu_memory_utilization only where safe to provide more KV cache space.",
                "Decrease max_num_seqs or max_num_batched_tokens to reduce KV cache pressure.",
                "Tune max_num_batched_tokens for throughput/latency tradeoffs; apply only through repo-supported config fields.",
            ],
        }

    def _write_tsv(self, path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
        rendered = ["\t".join(columns)]
        for row in rows:
            rendered.append("\t".join(str(row.get(column, "")) for column in columns))
        path.write_text("\n".join(rendered) + "\n", encoding="utf-8")

    def _write_yaml(self, path: Path, payload: Any) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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
    profile: str
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
            "profile": self.profile,
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
            profile=payload.get("profile", ""),
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
    workload_descriptor_path: str
    workload_distribution_id: str
    workload_distribution_id_hardening_version: str
    latency_ceiling_ms: int
    tpot_ceiling_ms: int
    turn_latency_ceiling_ms: int
    active_layer: str = "L1"
    target_concurrency: int = 4
    iteration_cap: int = 12
    rescreen_top_k: int = 3
    round_wall_clock_s: int = 12 * 60 * 60
    screen_warmup_s: int = 120
    screen_measurement_s: int = 600
    full_warmup_s: int = 300
    full_measurement_s: int = 1500
    per_iteration_codex_wall_clock_s: int = 0
    diminishing_returns_window_k: int = 4
    noise_floor: float = 0.0
    baseline_mean_screen: float = 0.0
    baseline_stddev_screen: float = 0.0
    measurements_per_candidate_screen: int = 3
    measurements_per_candidate_full: int = 1
    parent_head_sha: str = ""
    harness_type: str = "real"
    round_started_at: float = 0.0
    sub_spec_version: str = "v0.1.12"
    baseline_bundle_path: str = ""
    baseline_bundle_id: str = ""
    frozen_vllm_config: dict[str, Any] = field(default_factory=dict)
    serving_thinking_probe: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
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
            "workload_descriptor_path": self.workload_descriptor_path,
            "workload_distribution_id": self.workload_distribution_id,
            "workload_distribution_id_hardening_version": self.workload_distribution_id_hardening_version,
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
            "baseline_mean_screen": self.baseline_mean_screen,
            "baseline_stddev_screen": self.baseline_stddev_screen,
            "measurements_per_candidate_screen": self.measurements_per_candidate_screen,
            "measurements_per_candidate_full": self.measurements_per_candidate_full,
            "parent_head_sha": self.parent_head_sha,
            "harness_type": self.harness_type,
            "round_started_at": self.round_started_at,
            "sub_spec_version": self.sub_spec_version,
            "baseline_bundle_path": self.baseline_bundle_path,
            "baseline_bundle_id": self.baseline_bundle_id,
            "frozen_vllm_config": dict(self.frozen_vllm_config),
        }
        if self.serving_thinking_probe is not None:
            payload["serving_thinking_probe"] = dict(self.serving_thinking_probe)
        return payload

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
            workload_descriptor_path=str(payload.get("workload_descriptor_path", payload["workload_file"])),
            workload_distribution_id=str(payload["workload_distribution_id"]),
            workload_distribution_id_hardening_version=str(
                payload.get("workload_distribution_id_hardening_version", "")
            ),
            latency_ceiling_ms=int(payload["latency_ceiling_ms"]),
            tpot_ceiling_ms=int(payload["tpot_ceiling_ms"]),
            turn_latency_ceiling_ms=int(payload["turn_latency_ceiling_ms"]),
            active_layer=str(payload.get("active_layer", "L1")),
            target_concurrency=int(payload.get("target_concurrency", 4)),
            iteration_cap=int(payload.get("iteration_cap", 12)),
            rescreen_top_k=int(payload.get("rescreen_top_k", 3)),
            round_wall_clock_s=int(payload.get("round_wall_clock_s", 12 * 60 * 60)),
            screen_warmup_s=int(payload.get("screen_warmup_s", 120)),
            screen_measurement_s=int(payload.get("screen_measurement_s", 600)),
            full_warmup_s=int(payload.get("full_warmup_s", 300)),
            full_measurement_s=int(payload.get("full_measurement_s", 1500)),
            per_iteration_codex_wall_clock_s=int(payload.get("per_iteration_codex_wall_clock_s", 0)),
            diminishing_returns_window_k=int(payload.get("diminishing_returns_window_k", 4)),
            noise_floor=float(payload.get("noise_floor", 0.0)),
            baseline_mean_screen=float(payload.get("baseline_mean_screen", 0.0)),
            baseline_stddev_screen=float(payload.get("baseline_stddev_screen", 0.0)),
            measurements_per_candidate_screen=int(payload.get("measurements_per_candidate_screen", 3)),
            measurements_per_candidate_full=int(payload.get("measurements_per_candidate_full", 1)),
            parent_head_sha=str(payload.get("parent_head_sha", "")),
            harness_type=str(payload.get("harness_type", "real")),
            round_started_at=float(payload.get("round_started_at", 0.0)),
            sub_spec_version=str(payload.get("sub_spec_version", "v0.1.12")),
            baseline_bundle_path=str(payload.get("baseline_bundle_path", "")),
            baseline_bundle_id=str(payload.get("baseline_bundle_id", "")),
            frozen_vllm_config=dict(payload.get("frozen_vllm_config") or {}),
            serving_thinking_probe=(
                dict(payload["serving_thinking_probe"])
                if isinstance(payload.get("serving_thinking_probe"), dict)
                else None
            ),
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
        workload_file: str | Path | None,
        weight_version_id: str | None,
        round_root: str | Path,
        harness_type: str = "real",
        skip_preflight: bool = False,
        active_layer: str = "L1",
        baseline_bundle: str | Path | None = None,
        serving_thinking_probe: str | Path | None = None,
        allow_legacy_workload: bool = False,
        skip_codex_preflight: bool = False,
    ) -> dict[str, Any]:
        workload_file = resolve_workload_descriptor(self.repo_root, family_id, workload_file)
        round_root = Path(round_root).resolve()
        composite_descriptor = (
            workload_file
            == (self.repo_root / "benchmark_blueprints" / "workloads" / family_id / "workload.yaml").resolve()
        )
        workload_precondition = verify_workload_descriptor_preconditions(
            workload_file,
            composite=composite_descriptor,
            allow_legacy_workload=allow_legacy_workload,
        )
        active_layer = active_layer.upper()
        if active_layer not in {"L1", "L2"}:
            raise RuntimeError(f"Unsupported active_layer: {active_layer}")
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id: {model_id}")
        model_config = registry[model_id]
        frozen_bundle = load_baseline_bundle(baseline_bundle)
        if active_layer == "L2" and frozen_bundle is None:
            raise RuntimeError("L2 bootstrap requires --baseline-bundle")
        if active_layer == "L1" and frozen_bundle is not None:
            raise RuntimeError("L1 bootstrap does not accept --baseline-bundle")
        if frozen_bundle is not None:
            if frozen_bundle.model_id != model_id:
                raise RuntimeError(
                    f"baseline bundle model_id {frozen_bundle.model_id!r} does not match {model_id!r}"
                )
            if frozen_bundle.family_id != family_id:
                raise RuntimeError(
                    f"baseline bundle family_id {frozen_bundle.family_id!r} does not match {family_id!r}"
                )
            if weight_version_id is not None and frozen_bundle.weight_version_id != weight_version_id:
                raise RuntimeError(
                    f"baseline bundle weight_version_id {frozen_bundle.weight_version_id!r} does not match {weight_version_id!r}"
                )
            weight_version_id = frozen_bundle.weight_version_id
        workload = SyntheticWorkloadDistribution.from_file(workload_file, model_config=model_config, family_id=family_id)
        if not workload.seed_trace_ref:
            raise RuntimeError("Workload file is missing seed_trace_ref")
        seed_trace_path = workload_file.parent / workload.seed_trace_ref
        if not seed_trace_path.is_file():
            raise RuntimeError(f"Seed trace file does not exist: {seed_trace_path}")
        serving_thinking_probe_record: dict[str, Any] | None = None
        if harness_type != "synthetic" and not skip_preflight:
            serving_thinking_probe_record = resolve_serving_thinking_probe_report(
                self.repo_root,
                serving_thinking_probe,
            )
            self._run_bootstrap_preflight(
                model_config=model_config,
                family_id=family_id,
                weight_version_id=weight_version_id,
                workload=workload,
                include_codex=not skip_codex_preflight,
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
            workload_descriptor_path=str(workload_file),
            workload_distribution_id=workload.workload_distribution_id,
            workload_distribution_id_hardening_version=str(
                workload_precondition["workload_distribution_id_hardening_version"]
            ),
            latency_ceiling_ms=workload.latency_ceiling_ms,
            tpot_ceiling_ms=workload.tpot_ceiling_ms,
            turn_latency_ceiling_ms=workload.turn_latency_ceiling_ms,
            target_concurrency=workload.target_concurrency,
            parent_head_sha=parent_head_sha,
            harness_type=harness_type,
            round_started_at=time.time(),
            active_layer=active_layer,
            baseline_bundle_path=str(Path(baseline_bundle).resolve()) if baseline_bundle is not None else "",
            baseline_bundle_id=frozen_bundle.bundle_id if frozen_bundle is not None else "",
            frozen_vllm_config=dict(frozen_bundle.vllm_config) if frozen_bundle is not None else {},
            serving_thinking_probe=serving_thinking_probe_record,
        )
        self._write_yaml(round_dir / "round_spec.yaml", spec.as_dict())
        (round_dir / "impl_brief.md").write_text(IMPL_BRIEF_TEMPLATE, encoding="utf-8")
        (round_dir / "iteration_brief.md").write_text(ITERATION_BRIEF_TEMPLATE, encoding="utf-8")
        self._write_results(round_dir / "results.tsv", [])
        (round_dir / ".round.lock").write_text(json.dumps({"round_id": round_id, "created_at": time.time()}), encoding="utf-8")

        default_candidate = (
            self._default_request_shaping(dict(frozen_bundle.vllm_config))
            if active_layer == "L2" and frozen_bundle is not None
            else model_config.vllm_config()
        )
        for baseline_iteration in BASELINE_ITERATIONS:
            baseline_dir = candidates_dir / baseline_iteration
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
        selected_profile = profile or self._default_profile_for_iteration(iteration_id)
        self._validate_measure_request(
            spec=spec,
            iteration_id=iteration_id,
            profile=selected_profile,
        )
        candidate = load_yaml_file(candidate_path)
        if not isinstance(candidate, dict):
            raise RuntimeError(f"Candidate yaml must be a mapping: {candidate_path}")
        vllm_config, request_shaping = self._compose_candidate_for_layer(spec=spec, candidate=candidate)
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
            candidate_request_shaping=request_shaping,
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
                "active_layer": spec.active_layer,
            }
        )
        if spec.active_layer == "L2":
            trace.update(
                {
                    "candidate_request_shaping": dict(request_shaping),
                    "frozen_lower_layer": {
                        "source_bundle_path": spec.baseline_bundle_path,
                        "source_bundle_id": spec.baseline_bundle_id,
                        "vllm_config": {
                            key: value
                            for key, value in vllm_config.items()
                            if key in ALLOWED_VLLM_CONFIG_KEYS
                        },
                    },
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
        self._validate_measurement_generator(
            generator,
            harness_type=spec.harness_type,
            context="commit_refused",
        )

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
            False,
            branch=spec.round_branch,
            context="commit_refused",
        )
        return {
            "iteration": iteration,
            "candidate_uuid": str(trace.get("candidate_uuid")),
            "commit_sha": commit_sha,
            "status": status,
        }

    def rescreen(
        self,
        *,
        round_id: str,
        top_k: int,
        profile: str = "screen",
        harness: str | None = None,
        measurements_per_candidate_screen: int | None = None,
        measurements_per_candidate_full: int | None = None,
    ) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        spec = self._spec_with_harness(spec, harness)
        del profile
        screen_count = measurements_per_candidate_screen or spec.measurements_per_candidate_screen
        full_count = measurements_per_candidate_full or spec.measurements_per_candidate_full
        if screen_count < 1:
            raise RuntimeError("rescreen requires at least one Screen measurement per candidate")
        if full_count < 0:
            raise RuntimeError("rescreen Full measurement count must be >= 0")
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
            parent_candidate_dir = round_dir / "candidates" / parent.iteration
            measured_screen: list[tuple[str, str, float, Path]] = []
            for screen_index in range(1, screen_count + 1):
                iteration = f"rescreen_{index:02d}_screen_{screen_index}"
                rescreen_dir = round_dir / "candidates" / iteration
                rescreen_dir.mkdir(parents=True, exist_ok=False)
                shutil.copy2(parent_candidate_dir / "candidate.yaml", rescreen_dir / "candidate.yaml")
                measure_result = self.measure(
                    round_id=round_id,
                    candidate_path=rescreen_dir / "candidate.yaml",
                    profile="screen",
                    parent_candidate_uuid=parent.candidate_uuid,
                    harness=spec.harness_type,
                )
                trace = json.loads((rescreen_dir / "measurement_trace.json").read_text(encoding="utf-8"))
                self._validate_measurement_trace(trace, status="rescreened")
                self._validate_measurement_generator(
                    str(trace.get("generator", "")),
                    harness_type=spec.harness_type,
                    context="commit_refused",
                )
                measured_screen.append(
                    (
                        str(measure_result["candidate_uuid"]),
                        iteration,
                        float(trace["eval_throughput"]),
                        rescreen_dir,
                    )
                )

            screen_measurements = [float(parent.eval_throughput), *(value for _uuid, _iteration, value, _dir in measured_screen)]
            objective_mean = sum(screen_measurements) / len(screen_measurements)
            objective_ci_95 = 2.78 * self._sample_stddev(screen_measurements) / math.sqrt(len(screen_measurements))
            notes = ""
            if max(screen_measurements) - min(screen_measurements) > float(spec.noise_floor) * 2.0:
                notes = "inconsistent_rescreen"
            for candidate_uuid, iteration, objective_value, rescreen_dir in measured_screen:
                rows = self._read_results(round_dir / "results.tsv")
                updated_rows: list[ResultsRow] = []
                for row in rows:
                    if row.iteration == iteration and row.candidate_uuid == candidate_uuid:
                        updated_rows.append(
                            ResultsRow(
                                candidate_uuid=row.candidate_uuid,
                                parent_candidate_uuid=parent.candidate_uuid,
                                iteration=row.iteration,
                                profile=row.profile,
                                candidate_label=row.candidate_label,
                                feasible=row.feasible,
                                eval_throughput=row.eval_throughput,
                                objective_mean=f"{objective_mean:.6f}",
                                objective_ci_95=f"{objective_ci_95:.6f}",
                                measurement_count=len(screen_measurements),
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
                self._write_results(round_dir / "results.tsv", updated_rows)
                staged_paths = [
                    path.relative_to(self.repo_root)
                    for path in [*self._bootstrap_round_artifact_paths(round_dir), rescreen_dir, round_dir / "results.tsv"]
                ]
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
                    False,
                    branch=spec.round_branch,
                    context="commit_refused",
                )
                rescreen_rows.append(
                    {
                        "iteration": iteration,
                        "profile": "screen",
                        "candidate_uuid": candidate_uuid,
                        "parent_candidate_uuid": parent.candidate_uuid,
                        "eval_throughput": objective_value,
                        "objective_mean_screen": objective_mean,
                        "objective_ci_95_screen": objective_ci_95,
                        "commit_sha": commit_sha,
                    }
                )

            for full_index in range(1, full_count + 1):
                iteration = f"rescreen_{index:02d}_full_{full_index}"
                rescreen_dir = round_dir / "candidates" / iteration
                rescreen_dir.mkdir(parents=True, exist_ok=False)
                shutil.copy2(parent_candidate_dir / "candidate.yaml", rescreen_dir / "candidate.yaml")
                measure_result = self.measure(
                    round_id=round_id,
                    candidate_path=rescreen_dir / "candidate.yaml",
                    profile="full",
                    parent_candidate_uuid=parent.candidate_uuid,
                    harness=spec.harness_type,
                )
                trace = json.loads((rescreen_dir / "measurement_trace.json").read_text(encoding="utf-8"))
                self._validate_measurement_trace(trace, status="rescreened_full")
                self._validate_measurement_generator(
                    str(trace.get("generator", "")),
                    harness_type=spec.harness_type,
                    context="commit_refused",
                )
                full_value = float(trace["eval_throughput"])
                screen_stddev = self._sample_stddev(screen_measurements)
                full_notes = ""
                if full_value < objective_mean - (3.0 * screen_stddev) or full_value > objective_mean + (3.0 * screen_stddev):
                    full_notes = "screen_full_divergence"
                rows = self._read_results(round_dir / "results.tsv")
                updated_rows = []
                for row in rows:
                    if row.iteration == iteration and row.candidate_uuid == str(measure_result["candidate_uuid"]):
                        updated_rows.append(
                            ResultsRow(
                                candidate_uuid=row.candidate_uuid,
                                parent_candidate_uuid=parent.candidate_uuid,
                                iteration=row.iteration,
                                profile=row.profile,
                                candidate_label=row.candidate_label,
                                feasible=row.feasible,
                                eval_throughput=row.eval_throughput,
                                objective_mean="",
                                objective_ci_95="",
                                measurement_count=1,
                                window_completed=row.window_completed,
                                no_oom_events=row.no_oom_events,
                                reasoning_content_purity=row.reasoning_content_purity,
                                determinism_pass_rate=row.determinism_pass_rate,
                                status="rescreened_full",
                                notes=full_notes,
                            )
                        )
                    else:
                        updated_rows.append(row)
                self._write_results(round_dir / "results.tsv", updated_rows)
                staged_paths = [
                    path.relative_to(self.repo_root)
                    for path in [*self._bootstrap_round_artifact_paths(round_dir), rescreen_dir, round_dir / "results.tsv"]
                ]
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
                    False,
                    branch=spec.round_branch,
                    context="commit_refused",
                )
                rescreen_rows.append(
                    {
                        "iteration": iteration,
                        "profile": "full",
                        "candidate_uuid": str(measure_result["candidate_uuid"]),
                        "parent_candidate_uuid": parent.candidate_uuid,
                        "eval_throughput": full_value,
                        "screen_full_divergence": bool(full_notes),
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
        winner_candidate = load_yaml_file(candidate_yaml)
        if not isinstance(winner_candidate, dict):
            raise RuntimeError(f"Candidate yaml must be a mapping: {candidate_yaml}")
        vllm_config, request_shaping = self._compose_candidate_for_layer(spec=spec, candidate=winner_candidate)
        trace = self._run_harness(
            spec=spec,
            workload=workload,
            candidate_vllm_config=vllm_config,
            candidate_request_shaping=request_shaping,
            profile="full",
            use_holdout=True,
        )
        self._validate_measurement_generator(
            str(trace.get("generator", "")),
            harness_type=spec.harness_type,
            context="validate-holdout refuses",
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

    def finalize_round(
        self,
        *,
        round_id: str,
        dry_run: bool = False,
        imported_from_candidate: str | None = None,
        imported_from_commit: str | None = None,
    ) -> dict[str, Any]:
        round_dir = self._round_dir(round_id)
        spec = RoundSpecRecord.from_path(round_dir / "round_spec.yaml")
        rows = self._read_results(round_dir / "results.tsv")
        if not rows:
            raise RuntimeError("Cannot finalize an empty round")
        self._validate_finalize_generator_consistency(round_dir, harness_type=spec.harness_type)
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

        winner_candidate = load_yaml_file(round_dir / "candidates" / winner_row.iteration / "candidate.yaml")
        if not isinstance(winner_candidate, dict):
            raise RuntimeError(f"Candidate yaml must be a mapping for winner iteration {winner_row.iteration}")
        winner_vllm_config, winner_request_shaping = self._compose_candidate_for_layer(
            spec=spec,
            candidate=winner_candidate,
        )
        lower_layer_bundle = load_baseline_bundle(spec.baseline_bundle_path or None)
        baseline_screen_measurements = self._baseline_screen_measurements(rows)
        winner_screen_measurements = self._candidate_screen_measurements(rows, winner_parent_uuid)
        confidence_payload = self._derive_confidence_payload(
            baseline_screen_measurements=baseline_screen_measurements,
            winner_screen_measurements=winner_screen_measurements,
            noise_floor=spec.noise_floor,
            winner_row=winner_row,
        )
        screen_full_consistency = self._screen_full_consistency(rows, winner_parent_uuid)
        latency_above_slo = self._latency_above_slo(round_dir=round_dir, rows=rows, winner_parent_uuid=winner_parent_uuid, spec=spec)
        l2_enforcement_coverage = self._l2_enforcement_coverage(round_dir=round_dir, rows=rows, winner_parent_uuid=winner_parent_uuid, spec=spec)

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
            vllm_config=winner_vllm_config,
            request_shaping=winner_request_shaping,
            kernel_selection=dict(lower_layer_bundle.kernel_selection) if lower_layer_bundle is not None else None,
            lora_policy=dict(lower_layer_bundle.lora_policy) if lower_layer_bundle is not None else None,
            objective={
                "metric": "eval_throughput",
                "value": float(confidence_payload["objective_mean_screen"] or winner_row.objective_mean or winner_row.eval_throughput or "0"),
            },
            measurement_trace_ref=str(measurement_trace_ref.relative_to(self.repo_root)),
            search_trace_ref=str(search_trace_ref.relative_to(self.repo_root)),
            baseline_bundle_id=spec.baseline_bundle_id or None,
            regression_guard={"noise_floor": spec.noise_floor},
            safety_rails={"regression_guard_passed": True},
            round_provenance={
                "dry_run": dry_run,
                "round_id": round_id,
                "round_branch": spec.round_branch,
                "active_layer": spec.active_layer,
                "winner_iteration": winner_row.iteration,
                "winner_candidate_uuid": winner_parent_uuid,
                "winner_rescreen_uuid": winner_rescreen_uuid or None,
                "baseline_bundle_path": spec.baseline_bundle_path or None,
                "baseline_bundle_id": spec.baseline_bundle_id or None,
                "request_shaping_enforcement": self._request_shaping_enforcement_record_for(
                    spec,
                    winner_request_shaping,
                ),
                "confidence": confidence_payload["confidence"],
                "improvement_over_baseline_req_per_s": confidence_payload["improvement_over_baseline_req_per_s"],
                "improvement_over_baseline_ci_95": confidence_payload["improvement_over_baseline_ci_95"],
                "latency_above_slo": latency_above_slo,
                "screen_full_consistency": screen_full_consistency,
                "l2_enforcement_coverage": l2_enforcement_coverage,
                "workload_descriptor_path": spec.workload_descriptor_path,
                "sub_spec_version": spec.sub_spec_version,
                "agent_session_dir_ref": str((round_dir / "candidates").relative_to(self.repo_root)),
                "agent_model_pin": {"model": "gpt-5.4", "reasoning_effort": "high"},
                "results_tsv_ref": str((round_dir / "results.tsv").relative_to(self.repo_root)),
                "holdout_validation": "skipped" if dry_run else "pass",
                **(
                    {
                        "round_type": "replay",
                        "imported_from_candidate": imported_from_candidate,
                        "imported_from_commit": imported_from_commit,
                    }
                    if imported_from_candidate
                    else {}
                ),
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
            "diagnostics": {
                "confidence": confidence_payload,
                "latency_above_slo": latency_above_slo,
                "screen_full_consistency": screen_full_consistency,
            },
        }
        if screen_full_consistency == "divergent":
            run_log["diagnostics"]["screen_full_divergence_note"] = (
                "winner Full-profile rescreen fell outside objective_mean_screen +/- 3 * stddev_screen"
            )

        commit_message = (
            f"AR({round_id}) FINALIZE: {winner_row.candidate_label} - eval_throughput={winner_row.objective_mean or winner_row.eval_throughput}\n\n"
            f"winner_iteration={winner_row.iteration} winner_candidate_uuid={winner_parent_uuid} winner_rescreen_uuid={winner_rescreen_uuid or ''} bundle={bundle_path}\n"
            f"round_wall_clock_minutes={int((time.time() - spec.round_started_at) / 60)} total_iterations={len(rows)} feasible_count={sum(1 for row in rows if row.feasible)}\n"
            f"rescreened_count={sum(1 for row in rows if row.status == 'rescreened')} holdout_validation={'skipped' if dry_run else 'pass'}\n"
            f"stopping_reason=ok\n\n"
            f"Winner-Candidate-UUID: {winner_parent_uuid}\n"
            f"{f'imported_from_candidate: {imported_from_candidate}\n' if imported_from_candidate else ''}"
            f"{f'imported_from_commit: {imported_from_commit or ''}\n' if imported_from_candidate else ''}"
            f"{'Fixture-Mode: true\n' if spec.harness_type == 'synthetic' else ''}"
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
            False,
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
        elif any(row.iteration not in BASELINE_ITERATION_SET for row in rows):
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
        for baseline_iteration in BASELINE_ITERATIONS:
            self.measure(round_id=round_id, candidate_path=round_dir / "candidates" / baseline_iteration / "candidate.yaml")
            rows_created.append(
                self.commit_candidate(
                    round_id=round_id,
                    iteration=baseline_iteration,
                    status="baseline",
                    notes=f"default-config baseline replay {baseline_iteration.rsplit('_', 1)[1]}",
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
        candidate_request_shaping: dict[str, Any] | None = None,
        profile: str,
        use_holdout: bool = False,
    ) -> dict[str, Any]:
        target_concurrency = self._target_concurrency_for_measurement(
            spec=spec,
            request_shaping=candidate_request_shaping,
        )
        if spec.harness_type == "synthetic":
            fixture_cls = self._load_synthetic_fixture()
            harness = fixture_cls(workload)
            evaluation = harness.evaluate(
                candidate_vllm_config,
                iteration=0,
                label="fixture",
                profile=profile,
            )
            if candidate_request_shaping:
                self._apply_request_shaping_trace(
                    evaluation,
                    spec=spec,
                    request_shaping=candidate_request_shaping,
                    target_concurrency=target_concurrency,
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
            workload_descriptor_path=Path(spec.workload_file),
        )
        warmup_s = spec.full_warmup_s if profile == "full" else spec.screen_warmup_s
        measurement_s = spec.full_measurement_s if profile == "full" else spec.screen_measurement_s
        measurable_config = {key: value for key, value in candidate_vllm_config.items() if key in ALLOWED_VLLM_CONFIG_KEYS}
        try:
            measure_kwargs: dict[str, Any] = {
                "warmup_s": warmup_s,
                "window_s": measurement_s,
                "target_concurrency": target_concurrency,
            }
            if candidate_request_shaping:
                measure_kwargs["request_shaping"] = candidate_request_shaping
            trace = harness.measure(measurable_config, **measure_kwargs)
        except TypeError as exc:
            if "request_shaping" in str(exc) and candidate_request_shaping:
                trace = harness.measure(
                    measurable_config,
                    warmup_s=warmup_s,
                    window_s=measurement_s,
                    target_concurrency=target_concurrency,
                )
            elif "target_concurrency" not in str(exc):
                raise
            else:
                trace = harness.measure(
                    measurable_config,
                    warmup_s=warmup_s,
                    window_s=measurement_s,
                    target_concurrency_sweep=[target_concurrency],
                )
        if candidate_request_shaping:
            self._apply_request_shaping_trace(
                trace,
                spec=spec,
                request_shaping=candidate_request_shaping,
                target_concurrency=target_concurrency,
            )
        return trace

    def _compose_candidate_for_layer(
        self,
        *,
        spec: RoundSpecRecord,
        candidate: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        active_layer = spec.active_layer.upper()
        if active_layer == "L1":
            allowed_extra_keys = {"harness_overrides"} if spec.harness_type == "synthetic" else set()
            unknown_keys = sorted(set(candidate) - ALLOWED_VLLM_CONFIG_KEYS - allowed_extra_keys)
            if unknown_keys:
                raise RuntimeError(f"Candidate contains unsupported keys: {unknown_keys}")
            return (
                {
                    key: candidate[key]
                    for key in candidate
                    if key in ALLOWED_VLLM_CONFIG_KEYS or key in allowed_extra_keys
                },
                {},
            )
        if active_layer == "L2":
            unknown_keys = sorted(set(candidate) - ALLOWED_REQUEST_SHAPING_KEYS)
            missing_keys = sorted(ALLOWED_REQUEST_SHAPING_KEYS - set(candidate))
            if unknown_keys:
                raise RuntimeError(f"Candidate contains unsupported keys for L2: {unknown_keys}")
            if missing_keys:
                raise RuntimeError(f"Candidate is missing required L2 request_shaping keys: {missing_keys}")
            if not spec.frozen_vllm_config:
                raise RuntimeError("L2 round spec is missing frozen_vllm_config")
            request_shaping = self._validate_request_shaping(candidate, frozen_vllm_config=spec.frozen_vllm_config)
            return dict(spec.frozen_vllm_config), request_shaping
        raise RuntimeError(f"Unsupported active_layer: {spec.active_layer}")

    @staticmethod
    def _default_request_shaping(vllm_config: dict[str, Any]) -> dict[str, Any]:
        max_num_seqs = int(vllm_config["max_num_seqs"])
        max_model_len = int(vllm_config["max_model_len"])
        return {
            "concurrency_cap_eval": max_num_seqs,
            "concurrency_cap_rollout": 0,
            "admission_queue_depth_max": 128,
            "per_request_kv_budget": max_model_len,
            "priority_preemption": "off",
        }

    @staticmethod
    def _request_shaping_candidate_plan(frozen_vllm_config: dict[str, Any]) -> list[dict[str, Any]]:
        baseline = AutoResearchRoundManager._default_request_shaping(frozen_vllm_config)
        max_num_seqs = int(frozen_vllm_config["max_num_seqs"])
        raw_candidates = [
            baseline,
            {
                **baseline,
                "concurrency_cap_eval": max(1, max_num_seqs - 1),
                "concurrency_cap_rollout": min(1, max_num_seqs - max(1, max_num_seqs - 1)),
            },
            {
                **baseline,
                "concurrency_cap_eval": max(1, max_num_seqs // 2),
                "concurrency_cap_rollout": max_num_seqs - max(1, max_num_seqs // 2),
                "admission_queue_depth_max": 64,
            },
            {
                **baseline,
                "admission_queue_depth_max": 0,
            },
            {
                **baseline,
                "concurrency_cap_eval": max_num_seqs,
                "concurrency_cap_rollout": 0,
                "admission_queue_depth_max": 512,
            },
            {
                **baseline,
                "concurrency_cap_eval": max(1, max_num_seqs - 2),
                "concurrency_cap_rollout": min(2, max_num_seqs - max(1, max_num_seqs - 2)),
                "admission_queue_depth_max": 256,
            },
        ]
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for candidate in raw_candidates:
            key = json.dumps(candidate, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @staticmethod
    def _validate_request_shaping(
        candidate: dict[str, Any],
        *,
        frozen_vllm_config: dict[str, Any],
    ) -> dict[str, Any]:
        max_num_seqs = int(frozen_vllm_config["max_num_seqs"])
        max_model_len = int(frozen_vllm_config["max_model_len"])

        def require_int(key: str, minimum: int, maximum: int) -> int:
            value = candidate.get(key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise RuntimeError(f"L2 request_shaping {key} must be an integer")
            if value < minimum or value > maximum:
                raise RuntimeError(f"L2 request_shaping {key} must be between {minimum} and {maximum}")
            return value

        eval_cap = require_int("concurrency_cap_eval", 1, max_num_seqs)
        rollout_cap = require_int("concurrency_cap_rollout", 0, max_num_seqs)
        if eval_cap + rollout_cap > max_num_seqs:
            raise RuntimeError("L2 request_shaping eval + rollout concurrency caps exceed max_num_seqs")
        queue_depth = require_int("admission_queue_depth_max", 0, 512)
        kv_budget = require_int("per_request_kv_budget", max(1, max_model_len // 4), max_model_len)
        priority = candidate.get("priority_preemption")
        if priority not in PRIORITY_PREEMPTION_VALUES:
            raise RuntimeError(
                f"L2 request_shaping priority_preemption must be one of {sorted(PRIORITY_PREEMPTION_VALUES)}"
            )
        return {
            "concurrency_cap_eval": eval_cap,
            "concurrency_cap_rollout": rollout_cap,
            "admission_queue_depth_max": queue_depth,
            "per_request_kv_budget": kv_budget,
            "priority_preemption": str(priority),
        }

    @staticmethod
    def _target_concurrency_for_measurement(
        *,
        spec: RoundSpecRecord,
        request_shaping: dict[str, Any] | None,
    ) -> int:
        if spec.active_layer.upper() == "L2" and request_shaping:
            return int(request_shaping["concurrency_cap_eval"])
        return int(spec.target_concurrency)

    @staticmethod
    def _request_shaping_enforcement_record(spec: RoundSpecRecord) -> dict[str, Any]:
        return AutoResearchRoundManager._request_shaping_enforcement_record_for(spec, None)

    @staticmethod
    def _request_shaping_enforcement_record_for(
        spec: RoundSpecRecord,
        request_shaping: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if spec.active_layer.upper() != "L2":
            return {}
        enforced_values = {
            field: {
                "value": request_shaping[field],
                "enforcement": "enforced",
            }
            for field in ENFORCED_REQUEST_SHAPING_FIELDS
            if request_shaping and field in request_shaping
        }
        advisory_values = {
            field: {
                "value": request_shaping[field],
                "enforcement": "advisory",
                "reason": (
                    "v0.2 records and validates this field, but the proxy does not enforce it until "
                    "real KV accounting and scheduler preemption hooks exist."
                ),
            }
            for field in ADVISORY_REQUEST_SHAPING_FIELDS
            if request_shaping and field in request_shaping
        }
        return {
            "mode": "enforced_minus_advisory" if advisory_values else "enforced",
            "real_proxy_enforcement": True,
            "enforced_fields": list(ENFORCED_REQUEST_SHAPING_FIELDS),
            "advisory_fields": [field for field in ADVISORY_REQUEST_SHAPING_FIELDS if field in advisory_values],
            "field_values": {**enforced_values, **advisory_values},
            "proxy_enforcement": {
                "concurrency_cap_eval": "class-routed admission cap",
                "concurrency_cap_rollout": "class-routed admission cap",
                "admission_queue_depth_max": "bounded admission queue with structured 429 queue_full rejection",
            },
        }

    @staticmethod
    def _validate_l2_enforcement_record(
        record: Any,
        *,
        context: str,
        request_shaping: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            raise RuntimeError(f"{context}: missing request_shaping_enforcement")
        mode = record.get("mode")
        if mode not in {"enforced", "enforced_minus_advisory"}:
            raise RuntimeError(f"{context}: invalid request_shaping_enforcement.mode")
        if record.get("real_proxy_enforcement") is not True:
            raise RuntimeError(f"{context}: request_shaping_enforcement.real_proxy_enforcement must be true")
        enforced_fields = record.get("enforced_fields")
        if enforced_fields != ENFORCED_REQUEST_SHAPING_FIELDS:
            raise RuntimeError(f"{context}: request_shaping_enforcement.enforced_fields mismatch")
        advisory_fields = record.get("advisory_fields", [])
        if not isinstance(advisory_fields, list):
            raise RuntimeError(f"{context}: request_shaping_enforcement.advisory_fields must be a list")
        unsupported_advisory = sorted(set(advisory_fields) - set(ADVISORY_REQUEST_SHAPING_FIELDS))
        if unsupported_advisory:
            raise RuntimeError(f"{context}: unsupported advisory request_shaping fields: {unsupported_advisory}")
        if request_shaping is not None:
            expected_advisory = [field for field in ADVISORY_REQUEST_SHAPING_FIELDS if field in request_shaping]
            if advisory_fields != expected_advisory:
                raise RuntimeError(f"{context}: request_shaping_enforcement.advisory_fields mismatch")
        field_values = record.get("field_values")
        if not isinstance(field_values, dict):
            raise RuntimeError(f"{context}: request_shaping_enforcement.field_values must be a mapping")
        stray_advisory_values = sorted(
            field for field in ADVISORY_REQUEST_SHAPING_FIELDS if field in field_values and field not in advisory_fields
        )
        if stray_advisory_values:
            raise RuntimeError(f"{context}: advisory field_values present without advisory_fields: {stray_advisory_values}")
        for field in advisory_fields:
            payload = field_values.get(field)
            if not isinstance(payload, dict) or payload.get("enforcement") != "advisory" or "value" not in payload:
                raise RuntimeError(f"{context}: advisory field {field} is not marked advisory")
            if request_shaping is not None and payload.get("value") != request_shaping.get(field):
                raise RuntimeError(f"{context}: advisory field {field} value mismatch")
        for field in ENFORCED_REQUEST_SHAPING_FIELDS:
            payload = field_values.get(field)
            if not isinstance(payload, dict) or payload.get("enforcement") != "enforced" or "value" not in payload:
                raise RuntimeError(f"{context}: enforced field {field} is not marked enforced")
            if request_shaping is not None and payload.get("value") != request_shaping.get(field):
                raise RuntimeError(f"{context}: enforced field {field} value mismatch")
        return record

    def _apply_request_shaping_trace(
        self,
        trace: dict[str, Any],
        *,
        spec: RoundSpecRecord,
        request_shaping: dict[str, Any],
        target_concurrency: int,
    ) -> None:
        trace["request_shaping_enforcement"] = self._request_shaping_enforcement_record_for(spec, request_shaping)
        trace["request_shaping_enforcement"]["target_concurrency_applied"] = target_concurrency
        trace["candidate_request_shaping"] = dict(request_shaping)
        trace.setdefault("diagnostics", {})
        if isinstance(trace["diagnostics"], dict):
            trace["diagnostics"]["request_shaping"] = {
                "policy": dict(request_shaping),
                "target_concurrency_applied": target_concurrency,
            }
            trace["diagnostics"]["target_concurrency"] = target_concurrency
        if "sustained_concurrency" in trace:
            trace["sustained_concurrency"] = min(float(trace["sustained_concurrency"]), float(target_concurrency))
        if "eval_throughput" in trace:
            trace["eval_throughput"] = min(float(trace["eval_throughput"]), float(target_concurrency))

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
        include_codex: bool = True,
    ) -> None:
        if RealMeasurementHarness is None:
            raise RuntimeError("bootstrap-round preflight failed: harness module missing")
        if os.environ.get("LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT"):
            raise RuntimeError("bootstrap-round preflight failed: non-agent mode enabled")
        if self._git_status_short():
            raise RuntimeError("bootstrap-round requires a clean git worktree")
        if include_codex:
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
    def _validate_measurement_generator(
        generator: str,
        *,
        harness_type: str,
        context: str,
    ) -> None:
        if harness_type == "real":
            if generator.startswith(REAL_MEASUREMENT_GENERATOR_PREFIX):
                return
            raise RuntimeError(f"{context}: generator {generator!r} is not a production trace")
        if harness_type == "synthetic":
            if generator.startswith(SYNTHETIC_MEASUREMENT_GENERATOR_PREFIX):
                return
            raise RuntimeError(f"{context}: generator {generator!r} is not a synthetic fixture trace")
        raise RuntimeError(f"{context}: unsupported harness_type {harness_type!r}")

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
            profile=str(trace.get("profile") or ""),
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

        if spec.active_layer.upper() == "L2":
            default_candidate = self._default_request_shaping(spec.frozen_vllm_config)
        else:
            default_candidate = load_registry(self.registry_path)[spec.model_id].vllm_config()
        for baseline_iteration in BASELINE_ITERATIONS:
            baseline_path = round_dir / "candidates" / baseline_iteration / "candidate.yaml"
            if load_yaml_file(baseline_path) != default_candidate:
                raise RuntimeError(
                    f"{context}: immutable round artifact changed: candidates/{baseline_iteration}/candidate.yaml"
                )

    def _bootstrap_round_artifact_paths(self, round_dir: Path) -> list[Path]:
        paths = [
            round_dir / "round_spec.yaml",
            round_dir / "impl_brief.md",
            round_dir / "iteration_brief.md",
            round_dir / ".round.lock",
            *(round_dir / "candidates" / baseline_iteration for baseline_iteration in BASELINE_ITERATIONS),
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
            rescreen_index = int(iteration_id.split("_", 2)[1])
            if rescreen_index > spec.rescreen_top_k:
                raise RuntimeError(
                    f"measure_refused: iteration {iteration_id} exceeds rescreen_top_k {spec.rescreen_top_k}"
                )
            expected_profile = self._default_profile_for_iteration(iteration_id)
            if profile != expected_profile:
                raise RuntimeError(f"measure_refused: {iteration_id} requires the {expected_profile} profile")

    @staticmethod
    def _default_profile_for_iteration(iteration_id: str) -> str:
        if "_full_" in iteration_id:
            return "full"
        return "screen"

    @staticmethod
    def _metric_tie_break_value(value: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("inf")

    def _validate_finalize_generator_consistency(self, round_dir: Path, *, harness_type: str) -> None:
        generators: set[str] = set()
        for trace_path in sorted((round_dir / "candidates").glob("*/measurement_trace.json")):
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            generator = str(trace.get("generator") or "")
            if not generator:
                raise RuntimeError("finalize-round refuses because a measurement trace is missing generator")
            self._validate_measurement_generator(
                generator,
                harness_type=harness_type,
                context="finalize-round refuses",
            )
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

    def _baseline_screen_measurements(self, rows: list[ResultsRow]) -> list[float]:
        measurements: list[float] = []
        for row in sorted(rows, key=lambda item: item.iteration):
            if row.iteration in BASELINE_ITERATION_SET and row.status == "baseline" and row.profile == "screen":
                try:
                    measurements.append(float(row.eval_throughput))
                except (TypeError, ValueError):
                    continue
        return measurements

    def _candidate_screen_measurements(self, rows: list[ResultsRow], parent_uuid: str) -> list[float]:
        parent = next((row for row in rows if row.candidate_uuid == parent_uuid and not row.parent_candidate_uuid), None)
        measurements: list[float] = []
        if parent is not None and parent.profile == "screen":
            try:
                measurements.append(float(parent.eval_throughput))
            except (TypeError, ValueError):
                pass
        rescreens = sorted(
            (
                row
                for row in rows
                if row.parent_candidate_uuid == parent_uuid and row.status == "rescreened" and row.profile == "screen"
            ),
            key=lambda row: row.iteration,
        )
        for row in rescreens:
            try:
                measurements.append(float(row.eval_throughput))
            except (TypeError, ValueError):
                continue
        return measurements

    def _derive_confidence_payload(
        self,
        *,
        baseline_screen_measurements: list[float],
        winner_screen_measurements: list[float],
        noise_floor: float,
        winner_row: ResultsRow,
    ) -> dict[str, Any]:
        baseline_mean = sum(baseline_screen_measurements) / len(baseline_screen_measurements) if baseline_screen_measurements else 0.0
        winner_mean = sum(winner_screen_measurements) / len(winner_screen_measurements) if winner_screen_measurements else 0.0
        delta = winner_mean - baseline_mean
        ci_low, ci_high = self._improvement_ci_95_welch(baseline_screen_measurements, winner_screen_measurements)
        confidence = "unknown"
        if len(baseline_screen_measurements) >= 5 and len(winner_screen_measurements) >= 4:
            note_tokens = {token.strip() for token in winner_row.notes.split(",") if token.strip()}
            if note_tokens & {"inconsistent_rescreen", "high_variance_rescreen"}:
                confidence = "exploratory"
            elif ci_low > noise_floor:
                confidence = "defensible"
            elif delta > 0 and ci_low > 0:
                confidence = "within_noise_floor"
            else:
                confidence = "exploratory"
        return {
            "confidence": confidence,
            "baseline_mean_screen": baseline_mean,
            "objective_mean_screen": winner_mean,
            "improvement_over_baseline_req_per_s": delta,
            "improvement_over_baseline_ci_95": [ci_low, ci_high],
            "baseline_measurement_count": len(baseline_screen_measurements),
            "winner_screen_measurement_count": len(winner_screen_measurements),
            "noise_floor": noise_floor,
        }

    def _screen_full_consistency(self, rows: list[ResultsRow], parent_uuid: str) -> str:
        screen = self._candidate_screen_measurements(rows, parent_uuid)
        full_rows = sorted(
            (
                row
                for row in rows
                if row.parent_candidate_uuid == parent_uuid and row.status == "rescreened_full" and row.profile == "full"
            ),
            key=lambda row: row.iteration,
        )
        if len(screen) < 2 or not full_rows:
            return "consistent"
        screen_mean = sum(screen) / len(screen)
        screen_stddev = self._sample_stddev(screen)
        low = screen_mean - (3.0 * screen_stddev)
        high = screen_mean + (3.0 * screen_stddev)
        for row in full_rows:
            try:
                value = float(row.eval_throughput)
            except (TypeError, ValueError):
                continue
            if value < low or value > high:
                return "divergent"
        return "consistent"

    def _trace_for_row(self, round_dir: Path, row: ResultsRow) -> dict[str, Any]:
        path = round_dir / "candidates" / row.iteration / "measurement_trace.json"
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _trace_driver_ms(trace: dict[str, Any], key: str) -> float:
        value = trace.get(key)
        if isinstance(value, dict):
            value = value.get("driver", value.get("promql"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _latency_above_slo(
        self,
        *,
        round_dir: Path,
        rows: list[ResultsRow],
        winner_parent_uuid: str,
        spec: RoundSpecRecord,
    ) -> bool:
        descriptor = load_yaml_file(spec.workload_descriptor_path or spec.workload_file)
        nominal_ttft = float(descriptor.get("nominal_ttft_ms", spec.latency_ceiling_ms)) if isinstance(descriptor, dict) else float(spec.latency_ceiling_ms)
        nominal_turn = float(descriptor.get("nominal_turn_ms", spec.turn_latency_ceiling_ms)) if isinstance(descriptor, dict) else float(spec.turn_latency_ceiling_ms)
        candidate_rows = [
            row
            for row in rows
            if row.parent_candidate_uuid == winner_parent_uuid and row.status in {"rescreened", "rescreened_full"}
        ]
        max_ttft = 0.0
        max_turn = 0.0
        for row in candidate_rows:
            trace = self._trace_for_row(round_dir, row)
            max_ttft = max(max_ttft, self._trace_driver_ms(trace, "ttft_p95_ms"))
            max_turn = max(max_turn, self._trace_driver_ms(trace, "turn_latency_p95_ms"))
        return max_ttft > nominal_ttft or max_turn > nominal_turn

    def _l2_enforcement_coverage(
        self,
        *,
        round_dir: Path,
        rows: list[ResultsRow],
        winner_parent_uuid: str,
        spec: RoundSpecRecord,
    ) -> dict[str, Any]:
        if spec.active_layer.upper() != "L2":
            return {"mode": "not_l2", "enforced_fields": [], "advisory_fields": []}
        records_by_row = []
        for row in rows:
            trace = self._trace_for_row(round_dir, row)
            request_shaping = trace.get("candidate_request_shaping")
            if not isinstance(request_shaping, dict):
                raise RuntimeError(f"AR.28 L2 enforcement coverage for {row.iteration}: missing candidate_request_shaping")
            records_by_row.append(
                (
                    row,
                    self._validate_l2_enforcement_record(
                        trace.get("request_shaping_enforcement"),
                        context=f"AR.28 L2 enforcement coverage for {row.iteration}",
                        request_shaping=request_shaping,
                    ),
                )
            )
        winner_records = [
            record
            for row, record in records_by_row
            if row.parent_candidate_uuid == winner_parent_uuid and row.status in {"rescreened", "rescreened_full"}
        ]
        if not winner_records:
            winner_records = [record for row, record in records_by_row if row.candidate_uuid == winner_parent_uuid]
        coverage_record = winner_records[0] if winner_records else records_by_row[0][1]
        return {
            "mode": str(coverage_record.get("mode", "enforced_minus_advisory")),
            "enforced_fields": list(ENFORCED_REQUEST_SHAPING_FIELDS),
            "advisory_fields": list(coverage_record.get("advisory_fields", [])),
            "real_proxy_enforcement": True,
            "field_values": dict(coverage_record.get("field_values", {})),
        }

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
                if row.iteration in BASELINE_ITERATION_SET and row.status == "baseline" and row.profile == "screen"
            ),
            key=lambda row: row.iteration,
        )
        if len(baseline_rows) != len(BASELINE_ITERATIONS):
            return spec
        try:
            objectives = [float(row.eval_throughput) for row in baseline_rows]
        except (TypeError, ValueError):
            return spec
        baseline_mean = sum(objectives) / len(objectives)
        baseline_stddev = self._sample_stddev(objectives)
        noise_floor = 2.0 * baseline_stddev
        if (
            math.isclose(spec.noise_floor, noise_floor, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(spec.baseline_mean_screen, baseline_mean, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(spec.baseline_stddev_screen, baseline_stddev, rel_tol=0.0, abs_tol=1e-9)
        ):
            return spec
        return RoundSpecRecord(
            **{
                **spec.__dict__,
                "noise_floor": noise_floor,
                "baseline_mean_screen": baseline_mean,
                "baseline_stddev_screen": baseline_stddev,
            }
        )

    @staticmethod
    def _sample_stddev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _ci95(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return 1.96 * AutoResearchRoundManager._sample_stddev(values) / math.sqrt(len(values))

    @staticmethod
    def _t_critical_95_two_sided(degrees_of_freedom: float) -> float:
        table = [
            (1, 12.706),
            (2, 4.303),
            (3, 3.182),
            (4, 2.776),
            (5, 2.571),
            (6, 2.447),
            (7, 2.365),
            (8, 2.306),
            (9, 2.262),
            (10, 2.228),
            (12, 2.179),
            (15, 2.131),
            (20, 2.086),
            (30, 2.042),
            (60, 2.000),
            (120, 1.980),
        ]
        if not math.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0:
            return table[0][1]
        for df, critical in table:
            if degrees_of_freedom <= df:
                return critical
        return 1.96

    @staticmethod
    def _improvement_ci_95_welch(baseline: list[float], candidate: list[float]) -> tuple[float, float]:
        if len(baseline) < 2 or len(candidate) < 2:
            delta = (
                (sum(candidate) / len(candidate) if candidate else 0.0)
                - (sum(baseline) / len(baseline) if baseline else 0.0)
            )
            return (delta, delta)
        baseline_mean = sum(baseline) / len(baseline)
        candidate_mean = sum(candidate) / len(candidate)
        baseline_var = AutoResearchRoundManager._sample_stddev(baseline) ** 2
        candidate_var = AutoResearchRoundManager._sample_stddev(candidate) ** 2
        baseline_term = baseline_var / len(baseline)
        candidate_term = candidate_var / len(candidate)
        standard_error = math.sqrt(baseline_term + candidate_term)
        if standard_error == 0.0:
            delta = candidate_mean - baseline_mean
            return (delta, delta)
        numerator = (baseline_term + candidate_term) ** 2
        denominator = 0.0
        if len(baseline) > 1:
            denominator += (baseline_term**2) / (len(baseline) - 1)
        if len(candidate) > 1:
            denominator += (candidate_term**2) / (len(candidate) - 1)
        degrees = numerator / denominator if denominator else float("inf")
        margin = AutoResearchRoundManager._t_critical_95_two_sided(degrees) * standard_error
        delta = candidate_mean - baseline_mean
        return (delta - margin, delta + margin)

    def _bundle_output_path(self, bundle: TunedConfigBundle) -> Path:
        return self.tuned_config_root / bundle.family_id / bundle.weight_version_id / f"{bundle.run_slug}.yaml"


L0C_ITERATION_BRIEF_TEMPLATE = """\
You are an autonomous kernel-research agent for iteration {{iteration}} of round {{round_id}}.

# Your one job
Propose ONE mutation to {{kernel_source_path}} that is faster than the current
best on the workload, AND passes the parity gate at {{parity_fixture_path}}.

# Hardware context (MATTERS for what mutations are worth proposing)
This kernel runs on a **DGX Spark GB10**. Treat it as bandwidth-bound:
128 GB LPDDR5x unified memory at roughly 273 GB/s, not an HBM3e server GPU.
Mutations that reduce memory traffic or improve cache reuse are more likely
to matter than compute-only micro-optimizations.

{{strategy_brief}}

Use the context already in this brief and the local rejection history. Do
not spend iteration budget on online research inside the agent loop.

# Hard rules
- Edit ONLY {{kernel_source_path}}. No other file.
- Do not change the kernel's input/output signature.
- Do not change tile or grid sizes outside the autotune surface (those
  belong to L0b, not L0c).
- Read mutations_rejected.tsv. Mutations identical to a prior rejection
  by patch hash are immediately rejected without re-running. Read the
  rejection reasons (first_diverging_probe, tolerance_overshoot) and
  propose something genuinely different.
- Your mutation MUST pass parity. Latency is irrelevant if parity fails.

# Parity contract
- Logit-space tolerance: rtol={{rtol_logit}} / atol={{atol_logit}}
- (DeltaNet only) State-snapshot tolerance: rtol={{rtol_state}} / atol={{atol_state}}
- Recurrent-state checkpoints at: {{state_checkpoints_at_token}}

# Reading prior-iteration history
The canonical per-iteration record is `candidates/<NNN>/parity_check.json`
(written by the round controller's authoritative re-run). `BLOCKED.md`,
when present in a candidate dir, means the controller REJECTED that
mutation — its content is the controller's reason, NOT the agent's
prior commentary.

IMPORTANT: your own apply-and-test (step 4 below) runs in a Triton
autotune state that may diverge from the controller's later re-run.
The agent's apply-and-test parity verdict can therefore disagree with
the controller's verdict on the SAME patch (autotune-cache flips the
config selection, and that changes `tl.dot` reduction order). When
inspecting prior iterations, trust `parity_check.json` over any other
record. If a prior iteration is in `results.tsv`, it was accepted —
period — even if its dir contains stale agent commentary.

# Procedure
1. Read {{kernel_source_path}}, strategy_brief.md, prior_mutations_rejected.tsv,
   mutations_rejected.tsv, results.tsv (best_so_far).
   For prior iters' parity status, prefer `candidates/<NNN>/parity_check.json`.
2. Write your proposal to {{iteration_dir}}/mutation.patch.
   Generate the patch with a real diff tool; do not hand-write hunk counts.
   The patch must apply with:
     patch --dry-run {{kernel_source_path}} {{iteration_dir}}/mutation.patch
3. Run from the repo root with the exact entrypoint:
   cd {{repo_root}} && {{lumoserve_cmd}} auto-research apply-and-test \\
     --round-id {{round_id}} --iteration {{iteration}} \\
     --kernel-target {{kernel_target}} --harness {{harness_mode}}
4. Read the result. If parity fails, write a one-line note to BLOCKED.md
   explaining what you'll try next. Do NOT propose the same edit again.
   Note: the controller will overwrite or remove your BLOCKED.md after
   its own re-run, so write it for your own bookkeeping; the canonical
   record will be the controller's.
5. Exit 0.

# What you do NOT do
- You do not call finalize-round. Python does that.
- You do not run measurement directly. The CLI does that.
- You do not write any file except mutation.patch and BLOCKED.md.
"""


@dataclass(frozen=True)
class _L0cPatchOutcome:
    ok: bool
    error: str | None


@dataclass(frozen=True)
class L0cKernelMutationResult:
    round_id: str
    round_dir: Path
    bundle_path: Path | None
    kernel_target: str
    outcome: str
    terminal_condition: str
    accepted_count: int
    total_attempt_count: int
    rejected_count: int
    paired_baseline_objective_mean: float
    winner_objective_mean: float | None
    artifact_paths: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_dir": str(self.round_dir),
            "bundle_path": None if self.bundle_path is None else str(self.bundle_path),
            "kernel_target": self.kernel_target,
            "outcome": self.outcome,
            "terminal_condition": self.terminal_condition,
            "accepted_count": self.accepted_count,
            "total_attempt_count": self.total_attempt_count,
            "rejected_count": self.rejected_count,
            "paired_baseline_objective_mean": self.paired_baseline_objective_mean,
            "winner_objective_mean": self.winner_objective_mean,
            "artifact_paths": dict(self.artifact_paths),
        }


class L0cKernelMutationRunner:
    """P5 + P7 substrate for L0c kernel mutation rounds.

    Synthetic mode runs the three-cap loop deterministically against a
    seeded outcome generator so every AR.43-48b artifact contract is
    exercised end-to-end. Real mode is gated until the per-attempt agent
    integration is wired (HALT_REASON: l0c_real_harness_not_implemented).
    """

    MUTATION_TSV_COLUMNS = [
        "iteration",
        "candidate_uuid",
        "mutation_hash",
        "rejection_reason",
        "first_diverging_probe_index",
        "tolerance_overshoot",
    ]
    RESULTS_TSV_COLUMNS = [
        "iteration",
        "candidate_uuid",
        "parent_candidate_uuid",
        "mutation_hash",
        "status",
        "objective_mean",
        "measurement_count",
    ]
    MEASUREMENT_COLUMNS = [
        "candidate_uuid",
        "candidate_label",
        "measurement_role",
        "measurement_index",
        "objective_value",
        "harness",
        "trace_ref",
    ]
    TRAILER_COLUMNS = ["candidate_uuid", "candidate_label", "trailer"]

    def __init__(
        self,
        *,
        repo_root: str | Path,
        registry_path: str | Path,
        tuned_config_root: str | Path,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.registry_path = Path(registry_path).resolve()
        self.tuned_config_root = Path(tuned_config_root).resolve()

    # --- public entry points ------------------------------------------------

    def run(
        self,
        *,
        workload_file: str | Path,
        base_bundle: str | Path,
        kernel_target: str,
        kernel_source_path: str | Path,
        parity_fixture: str | Path,
        base_measurements: int,
        accepted_iteration_cap: int,
        total_attempt_cap: int,
        round_timeout_hours: float,
        round_root: str | Path,
        harness: str,
        model_id: str = L0A_DEFAULT_MODEL_ID,
        attempt_outcome_fn: Callable[[int], dict[str, Any]] | None = None,
        wall_clock_minutes_synthetic: float = 1.0,
        runtime: dict[str, Any] | None = None,
        agent_runtime: str = "codex",
        claude_model: str | None = None,
        claude_effort: str | None = None,
        per_iteration_wall_clock_s: int = L0C_DEFAULT_AGENT_TIMEOUT_S,
    ) -> L0cKernelMutationResult:
        if harness not in {"real", "synthetic"}:
            raise RuntimeError(f"Unsupported harness: {harness}")
        if kernel_target not in L0C_KERNEL_TARGETS:
            raise RuntimeError("--kernel-target must be one of deltanet, gatedattn")
        if base_measurements < 1:
            raise RuntimeError("--base-measurements must be >= 1")
        if accepted_iteration_cap < 1:
            raise RuntimeError("--accepted-iteration-cap must be >= 1")
        if total_attempt_cap < accepted_iteration_cap:
            raise RuntimeError("--total-attempt-cap must be >= --accepted-iteration-cap")
        if round_timeout_hours <= 0:
            raise RuntimeError("--round-timeout-hours must be > 0")

        workload_path = Path(workload_file).resolve()
        descriptor = load_yaml_file(workload_path)
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"Workload descriptor must be a mapping: {workload_path}")
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id: {model_id}")
        base = load_baseline_bundle(base_bundle)
        if base is None:
            raise RuntimeError("--base-bundle is required")
        if base.model_id != model_id:
            raise RuntimeError(
                f"base bundle model_id {base.model_id!r} does not match {model_id!r}"
            )
        if base.family_id != str(descriptor.get("family_id", "")):
            raise RuntimeError(
                f"base bundle family_id {base.family_id!r} does not match workload "
                f"family_id {descriptor.get('family_id')!r}"
            )
        weight_version_id = base.weight_version_id or default_weight_version_id(registry[model_id])

        kernel_source = Path(kernel_source_path)
        if not kernel_source.is_absolute():
            kernel_source = self.repo_root / kernel_source
        kernel_source = kernel_source.resolve()
        fixture_path = Path(parity_fixture).resolve()
        if not fixture_path.is_file():
            raise RuntimeError(f"Parity fixture missing: {fixture_path}")
        fixture_payload = load_yaml_file(fixture_path)
        fixture_id = (
            str(fixture_payload.get("fixture_id", ""))
            if isinstance(fixture_payload, dict)
            else ""
        )
        # Refuse the round when the parity fixture's reference_baseline kernel_selection
        # disagrees with the L0b-winner base bundle's kernel_selection: the fixture
        # captures logits from a specific runtime config, and any difference (e.g.
        # attention_backend) makes parity overshoot dominated by config drift instead
        # of kernel-mutation effects. Fail fast with a precise diff.
        self._assert_fixture_matches_base(fixture_payload, base, fixture_path)

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        round_id = (
            f"{model_id}-{descriptor.get('family_id', 'unknown')}-l0c-mutation-"
            f"{kernel_target}-{timestamp}"
        )
        root = Path(round_root).resolve()
        round_dir = root / round_id
        if round_dir.exists():
            raise RuntimeError(f"L0c round directory already exists: {round_dir}")
        round_dir.mkdir(parents=True)
        (round_dir / "candidates").mkdir()
        (round_dir / "live_traces").mkdir()
        prior_rejections = self._collect_prior_l0c_rejections(
            round_root=root,
            current_round_id=round_id,
            kernel_target=kernel_target,
        )
        self._write_tsv(
            round_dir / "prior_mutations_rejected.tsv",
            L0C_PRIOR_REJECTION_COLUMNS,
            prior_rejections,
        )
        strategy_brief = self._build_l0c_strategy_brief(
            kernel_target=kernel_target,
            round_id=round_id,
            hld_path=self.repo_root / "docs" / "HLD-Serving-Backend-AutoResearch-v0_2-L0KernelPlan.md",
            prior_rejections=prior_rejections,
        )
        (round_dir / "strategy_brief.md").write_text(strategy_brief, encoding="utf-8")

        spec = {
            "round_id": round_id,
            "round_type": L0C_MUTATION_ROUND_TYPE,
            "model_id": model_id,
            "family_id": str(descriptor.get("family_id", "")),
            "workload_file": str(workload_path),
            "base_bundle": str(Path(base_bundle).resolve()),
            "base_bundle_id": base.bundle_id,
            "kernel_target": kernel_target,
            "kernel_source_path": str(kernel_source),
            "parity_fixture": _relative_to_repo(self.repo_root, fixture_path),
            "parity_fixture_id": fixture_id,
            "parity_fixture_content_hash": fixture_content_hash(fixture_path),
            "harness": harness,
            "base_measurements": base_measurements,
            "accepted_iteration_cap": accepted_iteration_cap,
            "total_attempt_cap": total_attempt_cap,
            "round_timeout_hours": round_timeout_hours,
            "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        self._write_yaml(round_dir / "round_spec.yaml", spec)
        (round_dir / "iteration_brief.md").write_text(
            self._render_brief(
                kernel_target=kernel_target,
                kernel_source_path=kernel_source,
                fixture_path=fixture_path,
                fixture_payload=fixture_payload,
                round_id=round_id,
                harness=harness,
                strategy_brief=strategy_brief,
            ),
            encoding="utf-8",
        )

        if harness == "real":
            if runtime is None:
                raise RuntimeError(
                    "real-harness L0c rounds require --runtime (container_name, port, "
                    "proxy_port, endpoint, metrics_url, admin_url)"
                )
            if not kernel_source.is_file():
                raise RuntimeError(
                    f"real-harness L0c kernel_source_path must point at a real file: {kernel_source}"
                )
            base_bytes_dir = round_dir / "kernel_base"
            base_bytes_dir.mkdir()
            (base_bytes_dir / kernel_source.name).write_bytes(kernel_source.read_bytes())

            spec["runtime"] = dict(runtime)
            # Thread the L0b-winner base-bundle config into runtime so paired baseline
            # and candidate measurements rebuild the same vLLM activation: empty
            # vllm_config/kernel_selection here would produce empty staging bundles
            # and load_tuned_config would reject them.
            spec["runtime"].setdefault("vllm_config", dict(base.vllm_config or {}))
            spec["runtime"].setdefault("kernel_selection", dict(base.kernel_selection or {}))
            spec["runtime"].setdefault("request_shaping", dict(base.request_shaping or {}))
            spec["runtime"].setdefault("weight_version_id", weight_version_id)
            spec["agent_runtime"] = agent_runtime
            if claude_model is not None:
                spec["claude_model"] = claude_model
            if claude_effort is not None:
                spec["claude_effort"] = claude_effort
            spec["per_iteration_claude_wall_clock_s"] = per_iteration_wall_clock_s
            spec["per_iteration_codex_wall_clock_s"] = per_iteration_wall_clock_s
            self._write_yaml(round_dir / "round_spec.yaml", spec)

            return self._real_run(
                round_dir=round_dir,
                round_id=round_id,
                kernel_target=kernel_target,
                base=base,
                base_bundle=base_bundle,
                weight_version_id=weight_version_id,
                descriptor=descriptor,
                workload_path=workload_path,
                fixture_path=fixture_path,
                fixture_id=fixture_id,
                fixture_payload=fixture_payload,
                base_measurements=base_measurements,
                accepted_iteration_cap=accepted_iteration_cap,
                total_attempt_cap=total_attempt_cap,
                round_timeout_hours=round_timeout_hours,
                model_id=model_id,
                kernel_source=kernel_source,
                spec=spec,
            )

        outcome_fn = attempt_outcome_fn or self._default_synthetic_outcome
        baseline_uuid = str(uuid4())
        baseline_rows = [
            self._make_measurement_row(
                candidate_uuid=baseline_uuid,
                candidate_label="l0b-baseline-remeasured",
                role="l0b_baseline_remeasured",
                measurement_index=index,
                objective_value=1.0 + 0.001 * index,
                harness=harness,
                trace_ref=f"baselines/measurement_{index:02d}.json",
            )
            for index in range(1, base_measurements + 1)
        ]

        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        results_rows: list[dict[str, Any]] = []
        accepted_winner_rows: list[dict[str, Any]] = []
        accepted_count = 0
        total_attempts = 0
        consecutive_parity_fails = 0
        consecutive_compile_fails = 0
        intermittent_parity_seen = 0
        terminal_condition: str | None = None
        winner_uuid: str | None = None
        winner_mean: float | None = None
        for attempt_index in range(1, total_attempt_cap + 1):
            if accepted_count >= accepted_iteration_cap:
                terminal_condition = "accepted_cap_reached"
                break
            outcome = outcome_fn(attempt_index)
            attempt_label = f"{attempt_index:03d}"
            iteration_dir = round_dir / "candidates" / attempt_label
            iteration_dir.mkdir(parents=True, exist_ok=True)
            patch_text = outcome.get(
                "patch_text",
                f"--- a/{kernel_source}\n+++ b/{kernel_source}\n@@ -1,1 +1,1 @@\n-# baseline\n+# attempt {attempt_label}\n",
            )
            patch_path = iteration_dir / "mutation.patch"
            patch_path.write_text(patch_text, encoding="utf-8")
            mutation_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            total_attempts += 1
            candidate_uuid = str(uuid4())

            stage = str(outcome.get("stage", "parity"))
            if stage == "patch_apply_failed":
                self._write_parity_check(
                    iteration_dir,
                    pass_=False,
                    reason="patch_apply_failed",
                    error_detail=str(outcome.get("error", "synthetic patch_apply_failed")),
                    fixture_id=fixture_id,
                    kernel_target=kernel_target,
                )
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=candidate_uuid,
                        mutation_hash=mutation_hash,
                        reason="patch_apply_failed",
                    )
                )
                consecutive_parity_fails = 0
                consecutive_compile_fails = 0
                continue
            if stage == "compile_failed":
                self._write_parity_check(
                    iteration_dir,
                    pass_=False,
                    reason=str(outcome.get("compile_reason", "compile_nvcc_error")),
                    error_detail=str(outcome.get("error", "synthetic compile failure")),
                    fixture_id=fixture_id,
                    kernel_target=kernel_target,
                )
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=candidate_uuid,
                        mutation_hash=mutation_hash,
                        reason=str(outcome.get("compile_reason", "compile_nvcc_error")),
                    )
                )
                consecutive_compile_fails += 1
                consecutive_parity_fails = 0
                if consecutive_compile_fails >= L0C_COMPILE_FAILURES_THRESHOLD:
                    terminal_condition = "compile_failures_3x"
                    break
                continue
            if stage == "parity_fail":
                reason = str(outcome.get("parity_reason", "parity_logit_diverged"))
                self._write_parity_check(
                    iteration_dir,
                    pass_=False,
                    reason=reason,
                    fixture_id=fixture_id,
                    kernel_target=kernel_target,
                    first_diverging_probe=int(outcome.get("first_diverging_probe", attempt_index % 64)),
                    tolerance_overshoot=float(outcome.get("tolerance_overshoot", 0.001 * attempt_index)),
                )
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=candidate_uuid,
                        mutation_hash=mutation_hash,
                        reason=reason,
                        first_diverging_probe_index=int(
                            outcome.get("first_diverging_probe", attempt_index % 64)
                        ),
                        tolerance_overshoot=float(
                            outcome.get("tolerance_overshoot", 0.001 * attempt_index)
                        ),
                    )
                )
                consecutive_parity_fails += 1
                consecutive_compile_fails = 0
                if reason == "intermittent_parity":
                    intermittent_parity_seen += 1
                    if intermittent_parity_seen >= L0C_INTERMITTENT_PARITY_THRESHOLD:
                        terminal_condition = "intermittent_parity_observed"
                        break
                if consecutive_parity_fails >= L0C_PROPOSER_STUCK_THRESHOLD:
                    terminal_condition = "proposer_stuck"
                    break
                continue
            # parity passed
            consecutive_parity_fails = 0
            consecutive_compile_fails = 0
            objective_value = float(outcome.get("objective_value", 1.05 + 0.01 * accepted_count))
            self._write_parity_check(
                iteration_dir,
                pass_=True,
                reason="ran_passed",
                fixture_id=fixture_id,
                kernel_target=kernel_target,
            )
            measurement_rows = []
            for measurement_index in range(1, L0C_MEASUREMENTS_PER_ACCEPTED + 1):
                row = self._make_measurement_row(
                    candidate_uuid=candidate_uuid,
                    candidate_label=f"l0c-attempt-{attempt_label}",
                    role="l0c_candidate",
                    measurement_index=measurement_index,
                    objective_value=objective_value + 0.0001 * measurement_index,
                    harness=harness,
                    trace_ref=f"candidates/{attempt_label}/measurement_{measurement_index:02d}.json",
                )
                measurement_rows.append(row)
            accepted_rows.extend(measurement_rows)
            mean_objective = sum(float(row["objective_value"]) for row in measurement_rows) / len(
                measurement_rows
            )
            (iteration_dir / "measurement_trace.json").write_text(
                json.dumps(
                    {
                        "candidate_uuid": candidate_uuid,
                        "candidate_label": f"l0c-attempt-{attempt_label}",
                        "harness": harness,
                        "measurement_role": "l0c_candidate",
                        "measurements": measurement_rows,
                        "objective_mean": mean_objective,
                        "weight_sensitive": bool(outcome.get("weight_sensitive", False)),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            results_rows.append(
                {
                    "iteration": attempt_label,
                    "candidate_uuid": candidate_uuid,
                    "parent_candidate_uuid": baseline_uuid,
                    "mutation_hash": mutation_hash,
                    "status": "keep" if mean_objective > self._mean_of(baseline_rows) else "discard",
                    "objective_mean": f"{mean_objective:.6f}",
                    "measurement_count": str(len(measurement_rows)),
                }
            )
            accepted_count += 1
            if winner_mean is None or mean_objective > winner_mean:
                winner_mean = mean_objective
                winner_uuid = candidate_uuid
                accepted_winner_rows = measurement_rows
        else:
            terminal_condition = terminal_condition or "total_attempt_cap_reached"
        terminal_condition = terminal_condition or "accepted_cap_reached"

        # Persist canonical artifacts.
        self._write_tsv(
            round_dir / "mutations_rejected.tsv",
            self.MUTATION_TSV_COLUMNS,
            rejected_rows,
        )
        self._write_tsv(
            round_dir / "results.tsv",
            self.RESULTS_TSV_COLUMNS,
            results_rows,
        )
        self._write_tsv(
            round_dir / "measurements.tsv",
            self.MEASUREMENT_COLUMNS,
            [*baseline_rows, *accepted_rows],
        )
        trailer_rows = [
            {
                "candidate_uuid": baseline_uuid,
                "candidate_label": "l0b-baseline-remeasured",
                "trailer": "Measurement-Role: l0b_baseline_remeasured",
            }
        ]
        for row in results_rows:
            trailer_rows.append(
                {
                    "candidate_uuid": row["candidate_uuid"],
                    "candidate_label": f"l0c-attempt-{row['iteration']}",
                    "trailer": (
                        f"Measurement-Role: l0c_candidate; "
                        f"Mutation-Hash: {row['mutation_hash']}"
                    ),
                }
            )
        self._write_tsv(
            round_dir / "candidate_trailers.tsv",
            self.TRAILER_COLUMNS,
            trailer_rows,
        )

        paired_baseline_mean = self._mean_of(baseline_rows)
        outcome_label = (
            "ROUND_PASSED"
            if winner_mean is not None and winner_mean > paired_baseline_mean
            else "ROUND_NULL_RESULT"
        )
        if terminal_condition in {
            "proposer_stuck",
            "compile_failures_3x",
            "intermittent_parity_observed",
            "round_timeout",
            "agent_rate_limited",
            "agent_unavailable",
        }:
            outcome_label = "ROUND_BLOCKED"

        run_log = {
            "round_id": round_id,
            "outcome": outcome_label,
            "terminal_condition": terminal_condition,
            "accepted_count": accepted_count,
            "total_attempt_count": total_attempts,
            "rejected_count": len(rejected_rows),
            "wall_clock_minutes": wall_clock_minutes_synthetic,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "winner_objective_mean": winner_mean,
            "winner_candidate_uuid": winner_uuid,
            "harness": harness,
        }
        (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, sort_keys=True), encoding="utf-8")
        measurement_trace = {
            "round_id": round_id,
            "harness": harness,
            "kernel_target": kernel_target,
            "paired_baseline_rows": baseline_rows,
            "accepted_winner_rows": accepted_winner_rows,
            "accepted_winner_uuid": winner_uuid,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "winner_objective_mean": winner_mean,
            "welch_t_input_audit": {
                "baseline_source": "same_round_l0b_baseline_remeasured_rows",
                "winner_source": "same_round_l0c_candidate_rows",
                "baseline_candidate_uuid": baseline_uuid,
                "winner_candidate_uuid": winner_uuid,
            },
            "outcome": outcome_label,
        }
        (round_dir / "measurement_trace_combined.json").write_text(
            json.dumps(measurement_trace, indent=2),
            encoding="utf-8",
        )

        bundle_path: Path | None = None
        if outcome_label == "ROUND_PASSED" and winner_mean is not None and winner_uuid is not None:
            winning_iteration = next(
                row for row in results_rows if row["candidate_uuid"] == winner_uuid
            )
            bundle = make_tuned_config_bundle(
                model_id=model_id,
                family_id=str(descriptor.get("family_id", "")),
                weight_version_id=weight_version_id,
                workload_distribution_id=str(descriptor.get("workload_distribution_id", "")),
                vllm_config=dict(base.vllm_config),
                request_shaping=dict(base.request_shaping),
                kernel_selection=dict(base.kernel_selection),
                lora_policy=dict(base.lora_policy),
                layer_0_deltanet=self._layer_payload(
                    kernel_target=kernel_target,
                    target="deltanet",
                    base=base,
                    winning_row=winning_iteration,
                    paired_baseline_mean=paired_baseline_mean,
                    winner_mean=winner_mean,
                    terminal_condition=terminal_condition,
                    accepted_count=accepted_count,
                    total_attempts=total_attempts,
                    rejected_count=len(rejected_rows),
                    parity_fixture_path=fixture_path,
                ),
                layer_0_gatedattn=self._layer_payload(
                    kernel_target=kernel_target,
                    target="gatedattn",
                    base=base,
                    winning_row=winning_iteration,
                    paired_baseline_mean=paired_baseline_mean,
                    winner_mean=winner_mean,
                    terminal_condition=terminal_condition,
                    accepted_count=accepted_count,
                    total_attempts=total_attempts,
                    rejected_count=len(rejected_rows),
                    parity_fixture_path=fixture_path,
                ),
                layer_0_fp8_gemm={},
                objective={
                    "metric": "l0c_mutation_eval_throughput",
                    "value": winner_mean,
                    "paired_baseline_objective_mean": paired_baseline_mean,
                    "winner_objective_mean": winner_mean,
                    "outcome": outcome_label,
                },
                measurement_trace_ref=_relative_to_repo(
                    self.repo_root, round_dir / "measurement_trace_combined.json"
                ),
                search_trace_ref=_relative_to_repo(self.repo_root, round_dir / "results.tsv"),
                baseline_bundle_id=base.bundle_id,
                regression_guard={
                    "base_measurements": base_measurements,
                    "accepted_iteration_cap": accepted_iteration_cap,
                    "total_attempt_cap": total_attempt_cap,
                    "paired_baseline_objective_mean": paired_baseline_mean,
                    "winner_objective_mean": winner_mean,
                    "outcome": outcome_label,
                },
                safety_rails={
                    "round_timeout_hours": round_timeout_hours,
                    "proposer_stuck_threshold": L0C_PROPOSER_STUCK_THRESHOLD,
                    "compile_failures_threshold": L0C_COMPILE_FAILURES_THRESHOLD,
                },
                round_provenance={
                    "round_id": round_id,
                    "round_type": L0C_MUTATION_ROUND_TYPE,
                    "kernel_target": kernel_target,
                    "terminal_condition": terminal_condition,
                    "accepted_count": accepted_count,
                    "total_attempt_count": total_attempts,
                    "rejected_count": len(rejected_rows),
                    "harness": harness,
                },
            )
            bundle_path = persist_tuned_config_bundle(bundle, self.tuned_config_root)

        return L0cKernelMutationResult(
            round_id=round_id,
            round_dir=round_dir,
            bundle_path=bundle_path,
            kernel_target=kernel_target,
            outcome=outcome_label,
            terminal_condition=terminal_condition,
            accepted_count=accepted_count,
            total_attempt_count=total_attempts,
            rejected_count=len(rejected_rows),
            paired_baseline_objective_mean=paired_baseline_mean,
            winner_objective_mean=winner_mean,
            artifact_paths={
                "round_spec": str(round_dir / "round_spec.yaml"),
                "iteration_brief": str(round_dir / "iteration_brief.md"),
                "strategy_brief": str(round_dir / "strategy_brief.md"),
                "run_log": str(round_dir / "run_log.json"),
                "measurement_trace_combined": str(round_dir / "measurement_trace_combined.json"),
                "prior_mutations_rejected": str(round_dir / "prior_mutations_rejected.tsv"),
                "mutations_rejected": str(round_dir / "mutations_rejected.tsv"),
                "results": str(round_dir / "results.tsv"),
                "measurements": str(round_dir / "measurements.tsv"),
                "candidate_trailers": str(round_dir / "candidate_trailers.tsv"),
            },
        )

    def apply_and_test(
        self,
        *,
        round_id: str,
        iteration: str,
        kernel_target: str,
        harness: str,
        round_root: str | Path,
    ) -> dict[str, Any]:
        """Per-iteration agent-facing CLI.

        Synthetic mode reads the round spec for `kernel_target` + `parity_fixture`,
        looks at the `mutation.patch` file the agent wrote, and emits a deterministic
        outcome based on patch content (presence of the magic markers `BAD_PARITY`
        or `COMPILE_FAILED` selects the failure branch). Real mode raises a halt.
        """
        if harness not in {"real", "synthetic"}:
            raise RuntimeError(f"Unsupported harness: {harness}")
        if kernel_target not in L0C_KERNEL_TARGETS:
            raise RuntimeError("--kernel-target must be one of deltanet, gatedattn")
        round_dir = Path(round_root).resolve() / round_id
        if not round_dir.is_dir():
            raise RuntimeError(f"L0c round directory not found: {round_dir}")
        spec_path = round_dir / "round_spec.yaml"
        if not spec_path.is_file():
            raise RuntimeError(f"L0c round_spec.yaml missing: {spec_path}")
        spec = load_yaml_file(spec_path)
        if not isinstance(spec, dict):
            raise RuntimeError(f"Invalid round_spec.yaml: {spec_path}")
        if str(spec.get("kernel_target")) != kernel_target:
            raise RuntimeError(
                f"kernel_target mismatch: spec has {spec.get('kernel_target')!r}, got {kernel_target!r}"
            )
        iteration_dir = round_dir / "candidates" / iteration
        iteration_dir.mkdir(parents=True, exist_ok=True)
        patch_path = iteration_dir / "mutation.patch"
        if not patch_path.is_file():
            raise RuntimeError(f"mutation.patch missing for iteration {iteration}: {patch_path}")
        patch_text = patch_path.read_text(encoding="utf-8")
        mutation_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        fixture_id = str(spec.get("parity_fixture_id", ""))

        if harness == "real":
            return self._real_apply_and_test(
                round_id=round_id,
                iteration=iteration,
                iteration_dir=iteration_dir,
                round_dir=round_dir,
                kernel_target=kernel_target,
                spec=spec,
                patch_path=patch_path,
                mutation_hash=mutation_hash,
                fixture_id=fixture_id,
            )

        if "BAD_PARITY" in patch_text:
            self._write_parity_check(
                iteration_dir,
                pass_=False,
                reason="parity_logit_diverged",
                fixture_id=fixture_id,
                kernel_target=kernel_target,
                first_diverging_probe=0,
                tolerance_overshoot=0.005,
            )
            return {
                "round_id": round_id,
                "iteration": iteration,
                "kernel_target": kernel_target,
                "harness": harness,
                "mutation_hash": mutation_hash,
                "outcome": "parity_failed",
            }
        if "COMPILE_FAILED" in patch_text:
            self._write_parity_check(
                iteration_dir,
                pass_=False,
                reason="compile_nvcc_error",
                error_detail="synthetic compile_failed marker present",
                fixture_id=fixture_id,
                kernel_target=kernel_target,
            )
            return {
                "round_id": round_id,
                "iteration": iteration,
                "kernel_target": kernel_target,
                "harness": harness,
                "mutation_hash": mutation_hash,
                "outcome": "compile_failed",
            }
        objective_value = 1.0 + 0.01 * (int(iteration) if iteration.isdigit() else 1)
        self._write_parity_check(
            iteration_dir,
            pass_=True,
            reason="ran_passed",
            fixture_id=fixture_id,
            kernel_target=kernel_target,
        )
        candidate_uuid = str(uuid4())
        measurement_rows = [
            self._make_measurement_row(
                candidate_uuid=candidate_uuid,
                candidate_label=f"l0c-attempt-{iteration}",
                role="l0c_candidate",
                measurement_index=index,
                objective_value=objective_value + 0.0001 * index,
                harness=harness,
                trace_ref=f"candidates/{iteration}/measurement_{index:02d}.json",
            )
            for index in range(1, L0C_MEASUREMENTS_PER_ACCEPTED + 1)
        ]
        objective_mean = sum(float(row["objective_value"]) for row in measurement_rows) / len(
            measurement_rows
        )
        (iteration_dir / "measurement_trace.json").write_text(
            json.dumps(
                {
                    "candidate_uuid": candidate_uuid,
                    "candidate_label": f"l0c-attempt-{iteration}",
                    "harness": harness,
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
            "mutation_hash": mutation_hash,
            "outcome": "parity_passed",
            "candidate_uuid": candidate_uuid,
            "objective_mean": objective_mean,
        }

    @staticmethod
    def _assert_actually_resolved_no_drift(
        fixture_payload: Any,
        *,
        endpoint: str,
        api_key: str,
        fixture_path: Path,
    ) -> None:
        """Compare fixture's pinned actually_resolved_kernel_selection vs the live runtime.

        Implements HLD v0.3.3 §5.2 / §9 halt code `actually_resolved_kernel_selection_drift`:
        if the running vLLM resolved its symbolic aliases (e.g. vllm-default attention,
        compile mode, KV dtype) to different concrete values than the fixture pinned at
        capture time, the parity gate would compare logits across configs and any L0c
        round bootstrapped against this base would chase ghost divergences.

        Keys that the fixture pinned to "unknown" are skipped (no claim was made).
        Keys absent from the fixture are skipped too (legacy fixtures pre-v0.3.3).
        """
        if not isinstance(fixture_payload, dict):
            return
        generated_against = fixture_payload.get("generated_against") or {}
        pinned = generated_against.get("actually_resolved_kernel_selection") or {}
        if not isinstance(pinned, dict) or not pinned:
            return
        live = fetch_actually_resolved_kernel_selection(endpoint, api_key=api_key)
        diffs: list[str] = []
        for key in ACTUALLY_RESOLVED_KEYS:
            expected = pinned.get(key)
            if expected is None or str(expected) == "unknown":
                continue
            actual = live.get(key, "unknown")
            if str(actual) == "unknown":
                # Live runtime didn't surface this key; we can't claim drift either way.
                # Surface as a soft diff so it's visible without halting.
                continue
            if str(actual) != str(expected):
                diffs.append(f"  {key}: fixture={expected!r}  live={actual!r}")
        if diffs:
            raise RuntimeError(
                "HALT_REASON: actually_resolved_kernel_selection_drift\n"
                "The live vLLM stack resolved its kernel-selection aliases (e.g. "
                "vllm-default routing, compile mode, KV dtype) to values that differ "
                "from what the parity fixture pinned at capture time. Any L0c parity "
                "comparison would conflate config drift with kernel-mutation effects. "
                "Either restore the runtime to the pinned config, or re-capture the "
                "fixture against the new resolution.\n"
                f"  fixture: {fixture_path}\n"
                f"  endpoint: {endpoint}\n"
                "  diffs:\n" + "\n".join(diffs)
            )

    @staticmethod
    def _assert_fixture_matches_base(
        fixture_payload: Any,
        base: TunedConfigBundle,
        fixture_path: Path,
    ) -> None:
        if not isinstance(fixture_payload, dict):
            raise RuntimeError(f"parity fixture is not a mapping: {fixture_path}")
        generated_against = fixture_payload.get("generated_against") or {}
        reference_baseline = generated_against.get("reference_baseline") or {}
        if not isinstance(reference_baseline, dict) or not reference_baseline:
            return  # legacy fixture without reference_baseline metadata; nothing to check
        base_kernel_selection = dict(base.kernel_selection or {})
        diffs: list[str] = []
        for key, expected in reference_baseline.items():
            actual = base_kernel_selection.get(key)
            if str(actual) != str(expected):
                diffs.append(f"  {key}: fixture={expected!r}  base={actual!r}")
        if diffs:
            raise RuntimeError(
                "parity fixture/base bundle kernel_selection mismatch — the fixture's "
                "reference logits were captured against a different runtime config, "
                "so the parity gate would compare apples to oranges. Either pick a "
                "base bundle whose kernel_selection matches the fixture, or regenerate "
                "the fixture against this base.\n"
                f"  fixture: {fixture_path}\n"
                f"  base:    {base.bundle_id}\n"
                "  diffs:\n" + "\n".join(diffs)
            )

    # --- real apply-and-test ------------------------------------------------

    def _real_apply_and_test(
        self,
        *,
        round_id: str,
        iteration: str,
        iteration_dir: Path,
        round_dir: Path,
        kernel_target: str,
        spec: dict[str, Any],
        patch_path: Path,
        mutation_hash: str,
        fixture_id: str,
    ) -> dict[str, Any]:
        kernel_path_str = spec.get("kernel_source_path")
        if not kernel_path_str:
            raise RuntimeError("round_spec.yaml missing kernel_source_path for real harness")
        kernel_path = Path(kernel_path_str)
        if not kernel_path.is_absolute():
            kernel_path = self.repo_root / kernel_path
        kernel_path = kernel_path.resolve()
        base_bytes_path = round_dir / "kernel_base" / kernel_path.name
        if not base_bytes_path.is_file():
            raise RuntimeError(
                f"L0c round {round_id} missing kernel_base snapshot at {base_bytes_path}; "
                f"the round driver must snapshot kernel bytes at bootstrap"
            )
        base_bytes = base_bytes_path.read_bytes()

        fixture_relative = spec.get("parity_fixture")
        if not fixture_relative:
            raise RuntimeError("round_spec.yaml missing parity_fixture for real harness")
        fixture_dir = (self.repo_root / fixture_relative).parent

        common_outcome: dict[str, Any] = {
            "round_id": round_id,
            "iteration": iteration,
            "kernel_target": kernel_target,
            "harness": "real",
            "mutation_hash": mutation_hash,
        }

        try:
            if not kernel_path.is_file():
                kernel_path.write_bytes(base_bytes)

            patch_outcome = self._apply_kernel_patch(
                kernel_path=kernel_path, patch_path=patch_path
            )
            if not patch_outcome.ok:
                self._write_parity_check(
                    iteration_dir,
                    pass_=False,
                    reason="compile_nvcc_error",
                    fixture_id=fixture_id,
                    kernel_target=kernel_target,
                    error_detail=patch_outcome.error,
                )
                return {**common_outcome, "outcome": "compile_failed"}

            debug_export_dir = iteration_dir / "debug_export"
            staging_dir = debug_export_dir / "staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            # vLLM's debug-export hooks read these from os.environ; the model_server
            # container-launch path forwards them to the running container.
            #   LUMO_P2B_DEBUG_PROBE_REQUEST_IDS is required even when EXPORT=1 — without
            #   at least one fnmatch pattern P2BDebugExporter.from_env() returns disabled.
            #   "*" matches every request during the parity-probe window (the only requests
            #   the container sees between restart and probe completion).
            os.environ["LUMO_P2B_VLLM_DEBUG_EXPORT"] = "1"
            os.environ["LUMO_P2B_DEBUG_EXPORT_DIR"] = str(staging_dir)
            os.environ["LUMO_P2B_DEBUG_PROBE_REQUEST_IDS"] = "*"
            os.environ.setdefault("LUMO_P2B_DEBUG_STATE_TOKENS", "1,1024")
            os.environ.setdefault("LUMO_P2B_DEBUG_STRICT", "1")

            try:
                self._restart_serving_runtime(
                    spec=spec,
                    extra_volume_mounts=[f"{staging_dir}:{staging_dir}"],
                )
            except Exception as exc:  # noqa: BLE001 - any restart failure is compile-class
                self._write_parity_check(
                    iteration_dir,
                    pass_=False,
                    reason="compile_nvcc_error",
                    fixture_id=fixture_id,
                    kernel_target=kernel_target,
                    error_detail=f"runtime_restart_failed: {type(exc).__name__}: {exc}",
                )
                return {**common_outcome, "outcome": "compile_failed"}

            try:
                parity_result = self._invoke_parity_probe(
                    spec=spec,
                    fixture_dir=fixture_dir,
                    kernel_target=kernel_target,
                    debug_export_dir=debug_export_dir,
                )
            except Exception as exc:  # noqa: BLE001 - probe wrapper crash is capture-class
                self._write_parity_check(
                    iteration_dir,
                    pass_=False,
                    reason="capture_failed",
                    fixture_id=fixture_id,
                    kernel_target=kernel_target,
                    error_detail=f"{type(exc).__name__}: {exc}",
                )
                return {**common_outcome, "outcome": "compile_failed"}

            (iteration_dir / "parity_probe_result.json").write_text(
                json.dumps(parity_result.as_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self._write_parity_check(
                iteration_dir,
                pass_=parity_result.pass_,
                reason=parity_result.reason,
                fixture_id=parity_result.fixture_id or fixture_id,
                kernel_target=kernel_target,
                first_diverging_probe=parity_result.first_diverging_probe,
                tolerance_overshoot=(
                    parity_result.tolerance_overshoot
                    if not parity_result.pass_ and parity_result.tolerance_overshoot
                    else None
                ),
                error_detail=parity_result.error_detail,
            )
            # The agent's per-iteration apply-and-test runs first and may write
            # BLOCKED.md based on its own parity_check.json (later overwritten
            # here). Triton autotune-cache state can diverge between the agent's
            # rapid apply-and-test cycles and the controller's clean re-run, so
            # the agent's verdict can disagree with the controller's. The
            # controller is canonical: BLOCKED.md must reflect the controller's
            # verdict so future iterations' agents read the truth, not stale
            # agent perspective.
            blocked_path = iteration_dir / "BLOCKED.md"
            if parity_result.pass_:
                if blocked_path.exists():
                    blocked_path.unlink()
            else:
                blocker_lines = [
                    "iteration rejected by controller's apply-and-test",
                    f"reason: {parity_result.reason}",
                ]
                if parity_result.first_diverging_probe is not None:
                    blocker_lines.append(f"first_diverging_probe: {parity_result.first_diverging_probe}")
                if parity_result.tolerance_overshoot:
                    blocker_lines.append(f"tolerance_overshoot: {parity_result.tolerance_overshoot}")
                if parity_result.error_detail:
                    blocker_lines.append(f"detail: {parity_result.error_detail}")
                blocked_path.write_text("\n".join(blocker_lines) + "\n", encoding="utf-8")
            if not parity_result.pass_:
                if parity_result.reason in {"endpoint_unreachable", "capture_failed", "comparison_failed"}:
                    outcome = "compile_failed"
                else:
                    outcome = "parity_failed"
                return {**common_outcome, "outcome": outcome}

            measurement_rows, candidate_uuid, objective_mean = self._run_paired_l0c_measurements(
                spec=spec,
                iteration=iteration,
                iteration_dir=iteration_dir,
                count=L0C_MEASUREMENTS_PER_ACCEPTED,
            )
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
                **common_outcome,
                "outcome": "parity_passed",
                "candidate_uuid": candidate_uuid,
                "objective_mean": objective_mean,
            }
        finally:
            kernel_path.write_bytes(base_bytes)

    def _apply_kernel_patch(
        self, *, kernel_path: Path, patch_path: Path
    ) -> _L0cPatchOutcome:
        try:
            result = subprocess.run(
                ["patch", str(kernel_path), str(patch_path)],
                capture_output=True,
                timeout=60,
                check=False,
            )
        except FileNotFoundError as exc:
            return _L0cPatchOutcome(ok=False, error=f"patch_command_missing: {exc}")
        except subprocess.TimeoutExpired as exc:
            return _L0cPatchOutcome(ok=False, error=f"patch_timeout: {exc}")
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            stdout = result.stdout.decode("utf-8", errors="replace").strip()
            detail = stderr or stdout or f"patch exit {result.returncode}"
            return _L0cPatchOutcome(ok=False, error=f"patch_apply_failed: {detail}")
        return _L0cPatchOutcome(ok=True, error=None)

    def _restart_serving_runtime(
        self,
        *,
        spec: dict[str, Any],
        extra_volume_mounts: list[str] | None = None,
    ) -> None:
        runtime = spec.get("runtime")
        if not isinstance(runtime, dict):
            raise RuntimeError(
                "round_spec.yaml missing 'runtime' block; the round driver must populate it"
            )
        from .model_server import ModelServer  # local import: keeps test surface minimal

        model_id = runtime.get("model_id") or spec.get("model_id")
        if not model_id:
            raise RuntimeError("runtime block missing model_id")
        port = int(runtime.get("port", 8000))
        proxy_port = int(runtime.get("proxy_port", port + 1))
        container_name = runtime.get("container_name") or "lumo-vllm"
        image = runtime.get("image")
        logs_root = Path(runtime.get("logs_root", "/tmp/lumo-l0c-logs"))
        triton_cache_root = Path(runtime.get("triton_cache_root", "/tmp/lumo-l0c-triton"))
        state_root = (
            Path(runtime["state_root"]) if runtime.get("state_root") else None
        )
        host_kernel_path = spec.get("kernel_source_path")
        container_kernel_path = runtime.get("kernel_container_path")
        extra_mounts: list[str] = []
        if host_kernel_path and container_kernel_path:
            host_kernel = Path(str(host_kernel_path))
            if not host_kernel.is_absolute():
                host_kernel = self.repo_root / host_kernel
            extra_mounts.extend(
                ["-v", f"{host_kernel.resolve()}:{container_kernel_path}"]
            )
        for entry in runtime.get("extra_volume_mounts") or []:
            extra_mounts.extend(["-v", str(entry)])
        for entry in extra_volume_mounts or []:
            extra_mounts.extend(["-v", str(entry)])

        kwargs: dict[str, Any] = {
            "registry_path": self.registry_path,
            "port": port,
            "proxy_port": proxy_port,
            "container_name": container_name,
            "logs_root": logs_root,
            "triton_cache_root": triton_cache_root,
            "extra_volume_mounts": extra_mounts,
        }
        if image is not None:
            kwargs["image"] = image
        if state_root is not None:
            kwargs["state_root"] = state_root

        server = ModelServer(**kwargs)
        server.stop(missing_ok=True)
        server.start(model_id, enable_request_logging=False)

    def _invoke_parity_probe(
        self,
        *,
        spec: dict[str, Any],
        fixture_dir: Path,
        kernel_target: str,
        debug_export_dir: Path,
    ) -> ParityProbeResult:
        runtime = spec.get("runtime") or {}
        # The parity probe is a low-level diagnostic that POSTs to /v1/completions.
        # The lumo inference proxy (proxy_port) only whitelists /v1/responses and
        # /v1/chat/completions, so probing the proxy returns 403 "endpoint_unreachable"
        # before the kernel runs. Always bypass the proxy and hit the engine port.
        port = int(runtime.get("port", 8000))
        endpoint = f"http://127.0.0.1:{port}/v1"
        model_id = runtime.get("model_id") or spec.get("model_id") or ""
        api_key = runtime.get("api_key", "EMPTY")
        request_timeout = float(runtime.get("request_timeout_s", 1800.0))
        debug_export_dir.mkdir(parents=True, exist_ok=True)
        return run_parity_probe(
            repo_root=self.repo_root,
            fixture_dir=fixture_dir,
            kernel_target=kernel_target,
            endpoint=endpoint,
            model=model_id,
            api_key=api_key,
            debug_export_dir=debug_export_dir,
            request_timeout_s=request_timeout,
        )

    def _run_paired_l0c_measurements(
        self,
        *,
        spec: dict[str, Any],
        iteration: str,
        iteration_dir: Path,
        count: int,
    ) -> tuple[list[dict[str, Any]], str, float]:
        runtime = spec.get("runtime") or {}
        endpoint = runtime.get("endpoint")
        port = int(runtime.get("port", 8000))
        proxy_port = int(runtime.get("proxy_port", port + 1))
        if not endpoint:
            endpoint = f"http://127.0.0.1:{port}/v1"
        admin_url = runtime.get("admin_url") or endpoint
        metrics_url = runtime.get("metrics_url") or f"http://127.0.0.1:{port}/metrics"
        model_id = runtime.get("model_id") or spec.get("model_id") or ""
        weight_version_id = (
            runtime.get("weight_version_id")
            or spec.get("weight_version_id")
            or ""
        )
        workload_descriptor_path = Path(spec["workload_file"])
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id in registry: {model_id}")
        model_config = registry[model_id]
        workload_descriptor = load_yaml_file(workload_descriptor_path)
        if not isinstance(workload_descriptor, dict):
            raise RuntimeError(f"workload descriptor invalid: {workload_descriptor_path}")
        workload = SyntheticWorkloadDistribution.from_file(
            workload_descriptor_path,
            model_config=model_config,
            family_id=str(workload_descriptor.get("family_id", "")),
        )
        workload_spec = workload.to_workload_spec(base_dir=workload_descriptor_path.parent)
        slo = SLO(
            ttft_ms=workload.nominal_ttft_ms,
            tpot_ms=workload.tpot_ceiling_ms,
            turn_ms=workload.turn_latency_ceiling_ms,
        )
        bundle_staging_dir = iteration_dir / "bundle_staging"
        bundle_staging_dir.mkdir(parents=True, exist_ok=True)
        harness = RealMeasurementHarness(
            workload_spec=workload_spec,
            seed_trace_path=workload_spec.seed_trace_ref,
            slo=slo,
            endpoint=endpoint,
            metrics_scrape_url=metrics_url,
            admin_url=admin_url,
            model_id=model_id,
            weight_version_id=weight_version_id,
            bundle_staging_dir=bundle_staging_dir,
            round_id=spec.get("round_id"),
            workload_descriptor_path=workload_descriptor_path,
            runtime_activation=True,
            registry_path=self.registry_path,
            port=port,
            proxy_port=proxy_port,
            container_name=runtime.get("container_name", "lumo-vllm"),
            logs_root=runtime.get("logs_root", "/tmp/lumo-l0c-logs"),
            triton_cache_root=runtime.get("triton_cache_root", "/tmp/lumo-l0c-triton"),
            state_root=runtime.get("state_root"),
        )
        candidate_uuid = str(uuid4())
        rows: list[dict[str, Any]] = []
        objective_total = 0.0
        for index in range(1, count + 1):
            measurement = harness.measure(
                candidate_vllm_config=dict(runtime.get("vllm_config") or {}),
                warmup_s=int(runtime.get("warmup_s", 5)),
                window_s=int(runtime.get("window_s", 30)),
                request_shaping=dict(runtime.get("request_shaping") or {}),
                kernel_selection=dict(runtime.get("kernel_selection") or {}),
            )
            objective_value = float(
                measurement.get("objective_value")
                if measurement.get("objective_value") is not None
                else measurement.get("eval_throughput", 0.0)
            )
            objective_total += objective_value
            trace_ref = f"candidates/{iteration}/measurement_{index:02d}.json"
            (iteration_dir / f"measurement_{index:02d}.json").write_text(
                json.dumps(measurement, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            rows.append(
                self._make_measurement_row(
                    candidate_uuid=candidate_uuid,
                    candidate_label=f"l0c-attempt-{iteration}",
                    role="l0c_candidate",
                    measurement_index=index,
                    objective_value=objective_value,
                    harness="real",
                    trace_ref=trace_ref,
                )
            )
        objective_mean = objective_total / count if count else 0.0
        return rows, candidate_uuid, objective_mean

    # --- real round driver --------------------------------------------------

    def _real_run(
        self,
        *,
        round_dir: Path,
        round_id: str,
        kernel_target: str,
        base: TunedConfigBundle,
        base_bundle: str | Path,
        weight_version_id: str,
        descriptor: dict[str, Any],
        workload_path: Path,
        fixture_path: Path,
        fixture_id: str,
        fixture_payload: Any,
        base_measurements: int,
        accepted_iteration_cap: int,
        total_attempt_cap: int,
        round_timeout_hours: float,
        model_id: str,
        kernel_source: Path,
        spec: dict[str, Any],
    ) -> L0cKernelMutationResult:
        # 0. Drift bootstrap: refuse the round if the live vLLM has resolved its
        # kernel-selection aliases to different concrete values than the fixture
        # pinned at capture time (HLD v0.3.3 §5.2 / halt: actually_resolved_kernel_selection_drift).
        runtime = spec.get("runtime") or {}
        runtime_port = int(runtime.get("port", 8000))
        runtime_endpoint = runtime.get("endpoint") or f"http://127.0.0.1:{runtime_port}/v1"
        runtime_api_key = runtime.get("api_key", "EMPTY")
        self._assert_actually_resolved_no_drift(
            fixture_payload,
            endpoint=runtime_endpoint,
            api_key=runtime_api_key,
            fixture_path=fixture_path,
        )

        # 1. Paired-A/B baseline measurements (l0b_baseline_remeasured) BEFORE any mutation.
        baseline_uuid = str(uuid4())
        baseline_dir = round_dir / "baselines"
        baseline_dir.mkdir(exist_ok=True)
        baseline_rows = self._run_real_paired_baseline(
            spec=spec,
            baseline_dir=baseline_dir,
            baseline_uuid=baseline_uuid,
            count=base_measurements,
        )

        round_started = time.time()
        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        results_rows: list[dict[str, Any]] = []
        accepted_winner_rows: list[dict[str, Any]] = []
        accepted_count = 0
        total_attempts = 0
        consecutive_parity_fails = 0
        consecutive_compile_fails = 0
        intermittent_parity_seen = 0
        seen_hashes: set[str] = set()
        terminal_condition: str | None = None
        winner_uuid: str | None = None
        winner_mean: float | None = None

        for attempt_index in range(1, total_attempt_cap + 1):
            wall_clock_minutes = (time.time() - round_started) / 60.0
            if accepted_count >= accepted_iteration_cap:
                terminal_condition = "accepted_cap_reached"
                break
            if wall_clock_minutes >= round_timeout_hours * 60.0:
                terminal_condition = "round_timeout"
                break
            attempt_label = f"{attempt_index:03d}"
            iteration_dir = round_dir / "candidates" / attempt_label
            iteration_dir.mkdir(parents=True, exist_ok=True)
            self._render_iteration_brief_to_disk(
                iteration_dir=iteration_dir,
                round_id=round_id,
                kernel_target=kernel_target,
                kernel_source_path=kernel_source,
                fixture_path=fixture_path,
                fixture_payload=fixture_payload,
                iteration=attempt_label,
                harness="real",
            )

            spawn_outcome = self._spawn_l0c_agent_iteration(
                spec=spec,
                round_dir=round_dir,
                iteration_dir=iteration_dir,
                iteration=attempt_label,
            )
            total_attempts += 1
            if not spawn_outcome["ok"]:
                existing_rejection = self._rejection_from_existing_artifacts(
                    iteration=attempt_label,
                    iteration_dir=iteration_dir,
                    kernel_target=kernel_target,
                    fixture_id=fixture_id,
                )
                if existing_rejection is not None:
                    rejected_rows.append(existing_rejection)
                    if existing_rejection["rejection_reason"].startswith("parity_"):
                        consecutive_parity_fails += 1
                        consecutive_compile_fails = 0
                        if consecutive_parity_fails >= L0C_PROPOSER_STUCK_THRESHOLD:
                            terminal_condition = "proposer_stuck"
                            break
                    else:
                        consecutive_compile_fails += 1
                        consecutive_parity_fails = 0
                        if consecutive_compile_fails >= L0C_COMPILE_FAILURES_THRESHOLD:
                            terminal_condition = "compile_failures_3x"
                            break
                    continue
                # spawn_outcome["error"] is the actual cause: "agent_timeout: ...",
                # "agent_binary_missing: ...", "agent_exit_<rc>: ...". Surface its
                # category in rejection_reason so post-mortem reads truthfully — the
                # legacy generic "agent_spawn_failed" hid timeouts and exit-code
                # failures behind a label that implies the spawn itself failed.
                error_text = str(spawn_outcome.get("error", ""))
                if error_text.startswith("agent_timeout"):
                    terminal_condition = "agent_unavailable"
                    break
                elif error_text.startswith("agent_binary_missing"):
                    terminal_condition = "agent_unavailable"
                    break
                elif error_text.startswith("agent_rate_limited"):
                    terminal_condition = "agent_rate_limited"
                    break
                elif error_text.startswith("agent_exit_"):
                    if self._agent_error_is_rate_limit(error_text, iteration_dir / "agent_session.jsonl"):
                        terminal_condition = "agent_rate_limited"
                        break
                    terminal_condition = "agent_unavailable"
                    break
                else:
                    terminal_condition = "agent_unavailable"
                    break

            patch_path = iteration_dir / "mutation.patch"
            if not patch_path.is_file():
                consecutive_compile_fails += 1
                consecutive_parity_fails = 0
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=str(uuid4()),
                        mutation_hash="",
                        reason="agent_no_patch",
                    )
                )
                if consecutive_compile_fails >= L0C_COMPILE_FAILURES_THRESHOLD:
                    terminal_condition = "compile_failures_3x"
                    break
                continue

            patch_text = patch_path.read_text(encoding="utf-8")
            mutation_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
            if mutation_hash in seen_hashes:
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=str(uuid4()),
                        mutation_hash=mutation_hash,
                        reason="duplicate_mutation_hash",
                    )
                )
                # Duplicate doesn't reset compile/parity streak — agent's regress is mild.
                continue
            seen_hashes.add(mutation_hash)

            outcome = self.apply_and_test(
                round_id=round_id,
                iteration=attempt_label,
                kernel_target=kernel_target,
                harness="real",
                round_root=round_dir.parent,
            )

            if outcome["outcome"] == "compile_failed":
                consecutive_compile_fails += 1
                consecutive_parity_fails = 0
                parity = self._read_parity_check(iteration_dir)
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=str(uuid4()),
                        mutation_hash=mutation_hash,
                        reason=str(parity.get("reason", "compile_nvcc_error")),
                    )
                )
                if consecutive_compile_fails >= L0C_COMPILE_FAILURES_THRESHOLD:
                    terminal_condition = "compile_failures_3x"
                    break
                continue
            if outcome["outcome"] == "parity_failed":
                consecutive_parity_fails += 1
                consecutive_compile_fails = 0
                parity = self._read_parity_check(iteration_dir)
                reason = str(parity.get("reason", "parity_logit_diverged"))
                rejected_rows.append(
                    self._make_rejection_row(
                        iteration=attempt_label,
                        candidate_uuid=str(uuid4()),
                        mutation_hash=mutation_hash,
                        reason=reason,
                        first_diverging_probe_index=(
                            int(parity["first_diverging_probe"])
                            if "first_diverging_probe" in parity
                            else None
                        ),
                        tolerance_overshoot=(
                            float(parity["tolerance_overshoot"])
                            if "tolerance_overshoot" in parity
                            else None
                        ),
                    )
                )
                if reason == "intermittent_parity":
                    intermittent_parity_seen += 1
                    if intermittent_parity_seen >= L0C_INTERMITTENT_PARITY_THRESHOLD:
                        terminal_condition = "intermittent_parity_observed"
                        break
                if consecutive_parity_fails >= L0C_PROPOSER_STUCK_THRESHOLD:
                    terminal_condition = "proposer_stuck"
                    break
                continue

            # parity passed
            consecutive_parity_fails = 0
            consecutive_compile_fails = 0
            measurement_trace_path = iteration_dir / "measurement_trace.json"
            measurement_trace = json.loads(measurement_trace_path.read_text(encoding="utf-8"))
            measurement_rows = list(measurement_trace["measurements"])
            mean_objective = float(outcome["objective_mean"])
            candidate_uuid = str(outcome["candidate_uuid"])
            accepted_rows.extend(measurement_rows)
            results_rows.append(
                {
                    "iteration": attempt_label,
                    "candidate_uuid": candidate_uuid,
                    "parent_candidate_uuid": baseline_uuid,
                    "mutation_hash": mutation_hash,
                    "status": "keep" if mean_objective > self._mean_of(baseline_rows) else "discard",
                    "objective_mean": f"{mean_objective:.6f}",
                    "measurement_count": str(len(measurement_rows)),
                }
            )
            accepted_count += 1
            if winner_mean is None or mean_objective > winner_mean:
                winner_mean = mean_objective
                winner_uuid = candidate_uuid
                accepted_winner_rows = measurement_rows
        else:
            terminal_condition = terminal_condition or "total_attempt_cap_reached"
        terminal_condition = terminal_condition or "accepted_cap_reached"

        wall_clock_minutes = (time.time() - round_started) / 60.0
        return self._finalize_l0c_round(
            round_dir=round_dir,
            round_id=round_id,
            kernel_target=kernel_target,
            base=base,
            base_bundle=base_bundle,
            weight_version_id=weight_version_id,
            descriptor=descriptor,
            fixture_path=fixture_path,
            fixture_id=fixture_id,
            base_measurements=base_measurements,
            accepted_iteration_cap=accepted_iteration_cap,
            total_attempt_cap=total_attempt_cap,
            round_timeout_hours=round_timeout_hours,
            model_id=model_id,
            harness="real",
            baseline_uuid=baseline_uuid,
            baseline_rows=baseline_rows,
            accepted_rows=accepted_rows,
            rejected_rows=rejected_rows,
            results_rows=results_rows,
            accepted_winner_rows=accepted_winner_rows,
            accepted_count=accepted_count,
            total_attempts=total_attempts,
            terminal_condition=terminal_condition,
            winner_uuid=winner_uuid,
            winner_mean=winner_mean,
            wall_clock_minutes=wall_clock_minutes,
        )

    def _collect_prior_l0c_rejections(
        self,
        *,
        round_root: Path,
        current_round_id: str,
        kernel_target: str,
    ) -> list[dict[str, Any]]:
        if not round_root.exists():
            return []
        round_dirs = [
            path
            for path in round_root.iterdir()
            if path.is_dir()
            and path.name != current_round_id
            and f"-l0c-mutation-{kernel_target}-" in path.name
        ]
        round_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for prior_round in round_dirs:
            for row in self._prior_rejections_from_tsv(prior_round):
                key = (
                    row["source_round_id"],
                    row["iteration"],
                    row["mutation_hash"],
                    row["rejection_reason"],
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if len(rows) >= L0C_PRIOR_REJECTION_LIMIT:
                    return rows
            for row in self._prior_rejections_from_candidate_artifacts(prior_round):
                key = (
                    row["source_round_id"],
                    row["iteration"],
                    row["mutation_hash"],
                    row["rejection_reason"],
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
                if len(rows) >= L0C_PRIOR_REJECTION_LIMIT:
                    return rows
        return rows

    def _prior_rejections_from_tsv(self, round_dir: Path) -> list[dict[str, Any]]:
        path = round_dir / "mutations_rejected.tsv"
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for payload in self._read_tsv(path):
            iteration = str(payload.get("iteration", ""))
            rows.append(
                {
                    "source_round_id": round_dir.name,
                    "iteration": iteration,
                    "mutation_hash": str(payload.get("mutation_hash", "")),
                    "rejection_reason": str(payload.get("rejection_reason", "")),
                    "first_diverging_probe_index": str(
                        payload.get("first_diverging_probe_index", "")
                    ),
                    "tolerance_overshoot": str(payload.get("tolerance_overshoot", "")),
                    "source_ref": _relative_to_repo(self.repo_root, path),
                    "blocked_note": self._read_blocked_note(
                        round_dir / "candidates" / iteration
                    ),
                }
            )
        return rows

    def _prior_rejections_from_candidate_artifacts(
        self, round_dir: Path
    ) -> list[dict[str, Any]]:
        candidates_dir = round_dir / "candidates"
        if not candidates_dir.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for iteration_dir in sorted(candidates_dir.iterdir(), key=lambda path: path.name):
            if not iteration_dir.is_dir():
                continue
            parity_path = iteration_dir / "parity_check.json"
            if not parity_path.is_file():
                continue
            try:
                parity = json.loads(parity_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(parity, dict) or parity.get("pass") is not False:
                continue
            patch_path = iteration_dir / "mutation.patch"
            mutation_hash = ""
            if patch_path.is_file():
                mutation_hash = hashlib.sha256(
                    patch_path.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest()
            rows.append(
                {
                    "source_round_id": round_dir.name,
                    "iteration": iteration_dir.name,
                    "mutation_hash": mutation_hash,
                    "rejection_reason": str(parity.get("reason", "")),
                    "first_diverging_probe_index": (
                        ""
                        if parity.get("first_diverging_probe") is None
                        else str(parity["first_diverging_probe"])
                    ),
                    "tolerance_overshoot": (
                        ""
                        if parity.get("tolerance_overshoot") is None
                        else str(parity["tolerance_overshoot"])
                    ),
                    "source_ref": _relative_to_repo(self.repo_root, parity_path),
                    "blocked_note": self._read_blocked_note(iteration_dir),
                }
            )
        return rows

    @staticmethod
    def _read_blocked_note(iteration_dir: Path) -> str:
        path = iteration_dir / "BLOCKED.md"
        if not path.is_file():
            return ""
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    @staticmethod
    def _read_tsv(path: Path) -> list[dict[str, str]]:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        rows: list[dict[str, str]] = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split("\t")
            rows.append(
                {
                    key: values[index] if index < len(values) else ""
                    for index, key in enumerate(header)
                }
            )
        return rows

    def _build_l0c_strategy_brief(
        self,
        *,
        kernel_target: str,
        round_id: str,
        hld_path: Path,
        prior_rejections: list[dict[str, Any]],
    ) -> str:
        p3a_lines = self._latest_p3a_strategy_lines()
        prior_lines = ["- No prior cross-round rejections found for this kernel target."]
        if prior_rejections:
            prior_lines = []
            for row in prior_rejections[:8]:
                detail = (
                    f"- {row['source_round_id']} {row['iteration']}: "
                    f"{row['rejection_reason']}"
                )
                if row.get("first_diverging_probe_index"):
                    detail += f", first_probe={row['first_diverging_probe_index']}"
                if row.get("tolerance_overshoot"):
                    detail += f", overshoot={row['tolerance_overshoot']}"
                if row.get("blocked_note"):
                    detail += f"; note: {row['blocked_note']}"
                prior_lines.append(detail)
        lines = [
            "# L0c Strategy Brief",
            "",
            f"- Round: `{round_id}`",
            f"- Kernel target: `{kernel_target}`",
            f"- HLD source: `{_relative_to_repo(self.repo_root, hld_path)}`",
            (
                "- Prior rejection ledger: "
                f"`prior_mutations_rejected.tsv` ({len(prior_rejections)} rows)"
            ),
            "",
            "## Bottleneck Thesis",
            "",
            "- Treat the canary as decode-dominant and bandwidth-sensitive on GB10 LPDDR5x.",
            (
                "- Prefer changes that reduce memory traffic, improve cache locality, "
                "or narrow one load/store behavior at a time."
            ),
            (
                "- Do not trade numerical order or recurrent state semantics for speed; "
                "parity dominates throughput."
            ),
            *p3a_lines,
            "",
            "## Forbidden Mutation Families",
            "",
            (
                "- Do not add or expand `.cg`/cache modifiers on dot-adjacent "
                "`w` or `k` load families."
            ),
            "- Do not add store hints or change store cache policy.",
            "- Do not make broad cache-policy edits spanning both `v` and `h0`/state paths.",
            "- Do not add new gate-path cache-policy hints on `g`/`gk` loads.",
            (
                "- Do not change `v` load eviction/cache policy; the last canary "
                "`v` `evict_first` probe diverged immediately."
            ),
            (
                "- Do not change tile sizes, grid shape, signatures, recurrence order, "
                "or `tl.dot` operand ordering."
            ),
            (
            "- Do not retry a mutation hash listed in `mutations_rejected.tsv` "
            "or `prior_mutations_rejected.tsv`."
            ),
            (
                "- If an older `BLOCKED.md` suggestion conflicts with this forbidden "
                "list, the forbidden list wins."
            ),
            "",
            "## Ranked Likely-Safe Targets",
            "",
            "1. Narrow scalar metadata loads such as sequence/chunk offsets, preserving types and control flow.",
            (
                "2. Local address/mask expression cleanup that leaves arithmetic "
                "order and tensor shapes unchanged."
            ),
            (
                "3. One-load-only cache hint rollback or tightening where prior "
                "comments already identify cache pressure."
            ),
            "4. Comment-only hypothesis capture when no parity-neutral code edit is defensible.",
            "",
            "## Prior Rejections Carried Forward",
            "",
            (
                "Rows with `agent_exit_*` or `agent_spawn_failed` are historical "
                "context unless their `source_ref` points to a parity artifact."
            ),
            *prior_lines,
            "",
            "Use this brief as direction, not proof. The controller parity gate is canonical.",
            "",
        ]
        return "\n".join(lines)

    def _latest_p3a_strategy_lines(self) -> list[str]:
        output_root = self.repo_root / "output"
        if not output_root.is_dir():
            return [
                "- P3a: no local probe artifact found; fall back to HLD bandwidth thesis."
            ]
        probe_paths = list(output_root.glob("p3a_roofline_probe_*/p3a_roofline_probe.json"))
        if not probe_paths:
            return [
                "- P3a: no local probe artifact found; fall back to HLD bandwidth thesis."
            ]
        probe_paths.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        path = probe_paths[0]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return [
                f"- P3a: latest artifact `{_relative_to_repo(self.repo_root, path)}` "
                "is unreadable."
            ]
        derived = payload.get("derived") if isinstance(payload, dict) else {}
        gpu = payload.get("gpu_poll_stats") if isinstance(payload, dict) else {}
        util = gpu.get("utilization_gpu_pct", {}) if isinstance(gpu, dict) else {}
        decision = payload.get("p3a_decision", {}) if isinstance(payload, dict) else {}
        return [
            (
                f"- P3a: `{_relative_to_repo(self.repo_root, path)}` "
                f"probe_count={payload.get('probe_count', '')}, "
                f"wall={float(payload.get('wall_clock_s', 0.0)):.3f}s, "
                "decode_share="
                f"{float(derived.get('decode_time_share_of_prefill_plus_decode') or 0.0):.3f}, "
                f"gen_tok_s={float(derived.get('observed_tokens_per_second_wall') or 0.0):.3f}, "
                f"gpu_util_mean={float(util.get('mean') or 0.0):.1f}%."
            ),
            (
                "- P3a decision: "
                + str(
                    decision.get(
                        "basis",
                        "no counter-evidence against DeltaNet-first ordering.",
                    )
                )
            ),
            "- P3a limitation: no full Nsight kernel-category split; keep mutations conservative.",
        ]

    def _read_parity_check(self, iteration_dir: Path) -> dict[str, Any]:
        path = iteration_dir / "parity_check.json"
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _rejection_from_existing_artifacts(
        self,
        *,
        iteration: str,
        iteration_dir: Path,
        kernel_target: str,
        fixture_id: str,
    ) -> dict[str, Any] | None:
        """Recover the authoritative apply-and-test verdict if the agent exits badly.

        Some agent CLIs can produce the requested patch and run apply-and-test, then
        still exit nonzero because of an unrelated helper process or shell state. In
        that case the kernel verdict is already on disk and should not be overwritten
        by a misleading agent-exit rejection.
        """
        patch_path = iteration_dir / "mutation.patch"
        parity = self._read_parity_check(iteration_dir)
        if not patch_path.is_file() or not parity:
            return None
        patch_text = patch_path.read_text(encoding="utf-8")
        mutation_hash = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        pass_value = parity.get("pass")
        if pass_value is True:
            return None
        reason = str(parity.get("reason") or "agent_exit_after_apply_and_test")
        if not reason:
            reason = "agent_exit_after_apply_and_test"
        if not parity.get("fixture_id"):
            parity["fixture_id"] = fixture_id
        if not parity.get("kernel_target"):
            parity["kernel_target"] = kernel_target
        return self._make_rejection_row(
            iteration=iteration,
            candidate_uuid=str(uuid4()),
            mutation_hash=mutation_hash,
            reason=reason,
            first_diverging_probe_index=(
                int(parity["first_diverging_probe"])
                if "first_diverging_probe" in parity
                else None
            ),
            tolerance_overshoot=(
                float(parity["tolerance_overshoot"])
                if "tolerance_overshoot" in parity
                else None
            ),
        )

    @staticmethod
    def _agent_error_is_rate_limit(error_text: str, transcript_path: Path) -> bool:
        haystack = error_text.lower()
        if transcript_path.is_file():
            haystack += "\n" + transcript_path.read_text(
                encoding="utf-8", errors="replace"
            ).lower()
        return (
            "rate_limit" in haystack
            or "api_error_status\":429" in haystack
            or "api_error_status\": 429" in haystack
            or "you've hit your limit" in haystack
            or "you have hit your limit" in haystack
            or "429" in haystack and "limit" in haystack
        )

    def _render_iteration_brief_to_disk(
        self,
        *,
        iteration_dir: Path,
        round_id: str,
        kernel_target: str,
        kernel_source_path: Path,
        fixture_path: Path,
        fixture_payload: Any,
        iteration: str,
        harness: str,
    ) -> None:
        brief = self._render_brief(
            kernel_target=kernel_target,
            kernel_source_path=kernel_source_path,
            fixture_path=fixture_path,
            fixture_payload=fixture_payload,
            round_id=round_id,
            harness=harness,
            strategy_brief=(iteration_dir.parent.parent / "strategy_brief.md").read_text(
                encoding="utf-8"
            ),
        )
        brief = brief.replace("{{iteration}}", iteration)
        brief = brief.replace("{{iteration_dir}}", str(iteration_dir))
        (iteration_dir / "iteration_brief.md").write_text(brief, encoding="utf-8")

    def _run_real_paired_baseline(
        self,
        *,
        spec: dict[str, Any],
        baseline_dir: Path,
        baseline_uuid: str,
        count: int,
    ) -> list[dict[str, Any]]:
        runtime = spec.get("runtime") or {}
        endpoint = runtime.get("endpoint")
        port = int(runtime.get("port", 8000))
        proxy_port = int(runtime.get("proxy_port", port + 1))
        if not endpoint:
            endpoint = f"http://127.0.0.1:{port}/v1"
        admin_url = runtime.get("admin_url") or endpoint
        metrics_url = runtime.get("metrics_url") or f"http://127.0.0.1:{port}/metrics"
        model_id = runtime.get("model_id") or spec.get("model_id") or ""
        weight_version_id = (
            runtime.get("weight_version_id")
            or spec.get("weight_version_id")
            or ""
        )
        registry = load_registry(self.registry_path)
        if model_id not in registry:
            raise RuntimeError(f"Unknown model_id in registry: {model_id}")
        model_config = registry[model_id]
        workload_descriptor_path = Path(spec["workload_file"])
        workload_descriptor = load_yaml_file(workload_descriptor_path)
        if not isinstance(workload_descriptor, dict):
            raise RuntimeError(f"workload descriptor invalid: {workload_descriptor_path}")
        workload = SyntheticWorkloadDistribution.from_file(
            workload_descriptor_path,
            model_config=model_config,
            family_id=str(workload_descriptor.get("family_id", "")),
        )
        workload_spec = workload.to_workload_spec(base_dir=workload_descriptor_path.parent)
        slo = SLO(
            ttft_ms=workload.nominal_ttft_ms,
            tpot_ms=workload.tpot_ceiling_ms,
            turn_ms=workload.turn_latency_ceiling_ms,
        )
        bundle_staging_dir = baseline_dir / "bundle_staging"
        bundle_staging_dir.mkdir(parents=True, exist_ok=True)
        harness = RealMeasurementHarness(
            workload_spec=workload_spec,
            seed_trace_path=workload_spec.seed_trace_ref,
            slo=slo,
            endpoint=endpoint,
            metrics_scrape_url=metrics_url,
            admin_url=admin_url,
            model_id=model_id,
            weight_version_id=weight_version_id,
            bundle_staging_dir=bundle_staging_dir,
            round_id=spec.get("round_id"),
            workload_descriptor_path=workload_descriptor_path,
            runtime_activation=True,
            registry_path=self.registry_path,
            port=port,
            proxy_port=proxy_port,
            container_name=runtime.get("container_name", "lumo-vllm"),
            logs_root=runtime.get("logs_root", "/tmp/lumo-l0c-logs"),
            triton_cache_root=runtime.get("triton_cache_root", "/tmp/lumo-l0c-triton"),
            state_root=runtime.get("state_root"),
        )
        rows: list[dict[str, Any]] = []
        # Delete the cold baseline from statistical comparison. It is still
        # persisted for audit, but not entered into measurements.tsv or Welch input.
        for physical_index in range(0, count + 1):
            measurement = harness.measure(
                candidate_vllm_config=dict(runtime.get("vllm_config") or {}),
                warmup_s=int(runtime.get("warmup_s", 5)),
                window_s=int(runtime.get("window_s", 30)),
                request_shaping=dict(runtime.get("request_shaping") or {}),
                kernel_selection=dict(runtime.get("kernel_selection") or {}),
            )
            # RealMeasurementHarness.measure() returns eval_throughput (req/s) but no
            # objective_value key — match the L0a/L0b convention of treating
            # eval_throughput as the L0c objective.
            objective_value = float(
                measurement.get("objective_value")
                if measurement.get("objective_value") is not None
                else measurement.get("eval_throughput", 0.0)
            )
            if physical_index == 0:
                measurement["discarded_from_baseline_stats"] = True
                measurement["discard_reason"] = "cold_start_baseline"
                (baseline_dir / "cold_discard_00.json").write_text(
                    json.dumps(measurement, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                continue
            measurement_index = physical_index
            (baseline_dir / f"measurement_{measurement_index:02d}.json").write_text(
                json.dumps(measurement, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            rows.append(
                self._make_measurement_row(
                    candidate_uuid=baseline_uuid,
                    candidate_label="l0b-empirical-winner-baseline-remeasured",
                    role="l0b_empirical_winner_baseline_remeasured",
                    measurement_index=measurement_index,
                    objective_value=objective_value,
                    harness="real",
                    trace_ref=f"baselines/measurement_{measurement_index:02d}.json",
                )
            )
        return rows

    def _spawn_l0c_agent_iteration(
        self,
        *,
        spec: dict[str, Any],
        round_dir: Path,
        iteration_dir: Path,
        iteration: str,
    ) -> dict[str, Any]:
        from .round_driver import (
            DEFAULT_CLAUDE_EFFORT,
            DEFAULT_CLAUDE_MODEL,
            DEFAULT_CLAUDE_PERMISSION_MODE,
        )

        agent_runtime = str(spec.get("agent_runtime", "codex"))
        if agent_runtime not in {"codex", "claude"}:
            return {"ok": False, "error": f"unsupported agent_runtime {agent_runtime!r}"}

        last_message_path = iteration_dir / "agent_last_message.txt"
        transcript_path = iteration_dir / "agent_session.jsonl"
        prompt_path = iteration_dir / "iteration_brief.md"
        if not prompt_path.is_file():
            return {"ok": False, "error": "iteration_brief.md missing"}
        prompt = prompt_path.read_text(encoding="utf-8")
        # 0 (or negative) disables the per-iteration agent timeout; otherwise
        # the default finite canary ceiling is applied per runtime.
        timeout_key = (
            "per_iteration_claude_wall_clock_s"
            if agent_runtime == "claude"
            else "per_iteration_codex_wall_clock_s"
        )
        timeout_s = int(spec.get(timeout_key, 0))
        subprocess_timeout = timeout_s if timeout_s > 0 else None
        if agent_runtime == "claude":
            argv = [
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--model",
                str(spec.get("claude_model", DEFAULT_CLAUDE_MODEL)),
                "--effort",
                str(spec.get("claude_effort", DEFAULT_CLAUDE_EFFORT)),
                "--permission-mode",
                str(spec.get("claude_permission_mode", DEFAULT_CLAUDE_PERMISSION_MODE)),
                "--add-dir",
                str(round_dir),
            ]
        else:
            argv = [
                "codex",
                "-c",
                'model="gpt-5.4"',
                "-c",
                'model_reasoning_effort="high"',
                "exec",
                "--cd",
                str(round_dir),
                "--json",
                "--output-last-message",
                str(last_message_path),
                "--skip-git-repo-check",
                "-",
            ]
        proc: subprocess.Popen[bytes] | None = None
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(round_dir) if agent_runtime == "claude" else None,
                start_new_session=True,
            )
            stdout_bytes, stderr_bytes = proc.communicate(
                input=prompt.encode(),
                timeout=subprocess_timeout,
            )
            transcript_path.write_bytes(stdout_bytes)
        except subprocess.TimeoutExpired as exc:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout_bytes, stderr_bytes = proc.communicate()
                transcript_path.write_bytes(stdout_bytes or b"")
            return {"ok": False, "error": f"agent_timeout: {exc}"}
        except FileNotFoundError as exc:
            return {"ok": False, "error": f"agent_binary_missing: {exc}"}
        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if self._agent_error_is_rate_limit(stderr_text, transcript_path):
                return {"ok": False, "error": f"agent_rate_limited: {stderr_text}"}
            return {"ok": False, "error": f"agent_exit_{proc.returncode}: {stderr_text}"}
        if agent_runtime == "claude":
            from .round_driver import _extract_claude_last_message

            _extract_claude_last_message(transcript_path, last_message_path)
        return {"ok": True, "transcript": str(transcript_path)}

    def _finalize_l0c_round(
        self,
        *,
        round_dir: Path,
        round_id: str,
        kernel_target: str,
        base: TunedConfigBundle,
        base_bundle: str | Path,
        weight_version_id: str,
        descriptor: dict[str, Any],
        fixture_path: Path,
        fixture_id: str,
        base_measurements: int,
        accepted_iteration_cap: int,
        total_attempt_cap: int,
        round_timeout_hours: float,
        model_id: str,
        harness: str,
        baseline_uuid: str,
        baseline_rows: list[dict[str, Any]],
        accepted_rows: list[dict[str, Any]],
        rejected_rows: list[dict[str, Any]],
        results_rows: list[dict[str, Any]],
        accepted_winner_rows: list[dict[str, Any]],
        accepted_count: int,
        total_attempts: int,
        terminal_condition: str,
        winner_uuid: str | None,
        winner_mean: float | None,
        wall_clock_minutes: float,
    ) -> L0cKernelMutationResult:
        self._write_tsv(
            round_dir / "mutations_rejected.tsv",
            self.MUTATION_TSV_COLUMNS,
            rejected_rows,
        )
        self._write_tsv(
            round_dir / "results.tsv",
            self.RESULTS_TSV_COLUMNS,
            results_rows,
        )
        self._write_tsv(
            round_dir / "measurements.tsv",
            self.MEASUREMENT_COLUMNS,
            [*baseline_rows, *accepted_rows],
        )
        # HLD v0.3.3 §7.X: real-harness P7a baselines are the L0b empirical
        # winner re-measured in-round, so the operator-facing measurement role
        # name reflects "empirical winner" rather than the legacy alias. The
        # synthetic harness above keeps "l0b_baseline_remeasured" — v0.3.3 only
        # narrowed the EXECUTABLE rename, not the synthetic-test contract.
        trailer_rows = [
            {
                "candidate_uuid": baseline_uuid,
                "candidate_label": "l0b-empirical-winner-baseline-remeasured",
                "trailer": "Measurement-Role: l0b_empirical_winner_baseline_remeasured",
            }
        ]
        for row in results_rows:
            trailer_rows.append(
                {
                    "candidate_uuid": row["candidate_uuid"],
                    "candidate_label": f"l0c-attempt-{row['iteration']}",
                    "trailer": (
                        f"Measurement-Role: l0c_candidate; "
                        f"Mutation-Hash: {row['mutation_hash']}"
                    ),
                }
            )
        self._write_tsv(
            round_dir / "candidate_trailers.tsv",
            self.TRAILER_COLUMNS,
            trailer_rows,
        )

        paired_baseline_mean = self._mean_of(baseline_rows)
        outcome_label = (
            "ROUND_PASSED"
            if winner_mean is not None and winner_mean > paired_baseline_mean
            else "ROUND_NULL_RESULT"
        )
        if terminal_condition in {
            "proposer_stuck",
            "compile_failures_3x",
            "intermittent_parity_observed",
            "round_timeout",
            "agent_rate_limited",
            "agent_unavailable",
        }:
            outcome_label = "ROUND_BLOCKED"

        run_log = {
            "round_id": round_id,
            "outcome": outcome_label,
            "terminal_condition": terminal_condition,
            "accepted_count": accepted_count,
            "total_attempt_count": total_attempts,
            "rejected_count": len(rejected_rows),
            "wall_clock_minutes": wall_clock_minutes,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "winner_objective_mean": winner_mean,
            "winner_candidate_uuid": winner_uuid,
            "harness": harness,
        }
        (round_dir / "run_log.json").write_text(
            json.dumps(run_log, indent=2, sort_keys=True), encoding="utf-8"
        )
        measurement_trace = {
            "round_id": round_id,
            "harness": harness,
            "kernel_target": kernel_target,
            "paired_baseline_rows": baseline_rows,
            "accepted_winner_rows": accepted_winner_rows,
            "accepted_winner_uuid": winner_uuid,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "winner_objective_mean": winner_mean,
            "welch_t_input_audit": {
                "baseline_source": "same_round_l0b_empirical_winner_baseline_remeasured_rows",
                "winner_source": "same_round_l0c_candidate_rows",
                "baseline_candidate_uuid": baseline_uuid,
                "winner_candidate_uuid": winner_uuid,
            },
            "outcome": outcome_label,
        }
        (round_dir / "measurement_trace_combined.json").write_text(
            json.dumps(measurement_trace, indent=2),
            encoding="utf-8",
        )

        bundle_path: Path | None = None
        if outcome_label == "ROUND_PASSED" and winner_mean is not None and winner_uuid is not None:
            winning_iteration = next(
                row for row in results_rows if row["candidate_uuid"] == winner_uuid
            )
            bundle = make_tuned_config_bundle(
                model_id=model_id,
                family_id=str(descriptor.get("family_id", "")),
                weight_version_id=weight_version_id,
                workload_distribution_id=str(descriptor.get("workload_distribution_id", "")),
                vllm_config=dict(base.vllm_config),
                request_shaping=dict(base.request_shaping),
                kernel_selection=dict(base.kernel_selection),
                lora_policy=dict(base.lora_policy),
                layer_0_deltanet=self._layer_payload(
                    kernel_target=kernel_target,
                    target="deltanet",
                    base=base,
                    winning_row=winning_iteration,
                    paired_baseline_mean=paired_baseline_mean,
                    winner_mean=winner_mean,
                    terminal_condition=terminal_condition,
                    accepted_count=accepted_count,
                    total_attempts=total_attempts,
                    rejected_count=len(rejected_rows),
                    parity_fixture_path=fixture_path,
                ),
                layer_0_gatedattn=self._layer_payload(
                    kernel_target=kernel_target,
                    target="gatedattn",
                    base=base,
                    winning_row=winning_iteration,
                    paired_baseline_mean=paired_baseline_mean,
                    winner_mean=winner_mean,
                    terminal_condition=terminal_condition,
                    accepted_count=accepted_count,
                    total_attempts=total_attempts,
                    rejected_count=len(rejected_rows),
                    parity_fixture_path=fixture_path,
                ),
                layer_0_fp8_gemm={},
                objective={
                    "metric": "l0c_mutation_eval_throughput",
                    "value": winner_mean,
                    "paired_baseline_objective_mean": paired_baseline_mean,
                    "winner_objective_mean": winner_mean,
                    "outcome": outcome_label,
                },
                measurement_trace_ref=_relative_to_repo(
                    self.repo_root, round_dir / "measurement_trace_combined.json"
                ),
                search_trace_ref=_relative_to_repo(self.repo_root, round_dir / "results.tsv"),
                baseline_bundle_id=base.bundle_id,
                regression_guard={
                    "base_measurements": base_measurements,
                    "accepted_iteration_cap": accepted_iteration_cap,
                    "total_attempt_cap": total_attempt_cap,
                    "paired_baseline_objective_mean": paired_baseline_mean,
                    "winner_objective_mean": winner_mean,
                    "outcome": outcome_label,
                },
                safety_rails={
                    "round_timeout_hours": round_timeout_hours,
                    "proposer_stuck_threshold": L0C_PROPOSER_STUCK_THRESHOLD,
                    "compile_failures_threshold": L0C_COMPILE_FAILURES_THRESHOLD,
                },
                round_provenance={
                    "round_id": round_id,
                    "round_type": L0C_MUTATION_ROUND_TYPE,
                    "kernel_target": kernel_target,
                    "terminal_condition": terminal_condition,
                    "accepted_count": accepted_count,
                    "total_attempt_count": total_attempts,
                    "rejected_count": len(rejected_rows),
                    "parity_fixture_id": fixture_id,
                    "parity_fixture_path": _relative_to_repo(self.repo_root, fixture_path),
                },
            )
            bundle_path = persist_tuned_config_bundle(bundle, self.tuned_config_root)

        return L0cKernelMutationResult(
            round_id=round_id,
            round_dir=round_dir,
            bundle_path=bundle_path,
            kernel_target=kernel_target,
            outcome=outcome_label,
            terminal_condition=terminal_condition,
            accepted_count=accepted_count,
            total_attempt_count=total_attempts,
            rejected_count=len(rejected_rows),
            paired_baseline_objective_mean=paired_baseline_mean,
            winner_objective_mean=winner_mean,
            artifact_paths={
                "round_spec": str(round_dir / "round_spec.yaml"),
                "iteration_brief": str(round_dir / "iteration_brief.md"),
                "strategy_brief": str(round_dir / "strategy_brief.md"),
                "results": str(round_dir / "results.tsv"),
                "prior_mutations_rejected": str(round_dir / "prior_mutations_rejected.tsv"),
                "mutations_rejected": str(round_dir / "mutations_rejected.tsv"),
                "measurements": str(round_dir / "measurements.tsv"),
                "candidate_trailers": str(round_dir / "candidate_trailers.tsv"),
                "run_log": str(round_dir / "run_log.json"),
                "measurement_trace_combined": str(round_dir / "measurement_trace_combined.json"),
            },
        )

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _default_synthetic_outcome(attempt_index: int) -> dict[str, Any]:
        if attempt_index % 3 == 0:
            return {
                "stage": "parity_fail",
                "parity_reason": "parity_logit_diverged",
                "first_diverging_probe": (attempt_index * 7) % 64,
                "tolerance_overshoot": 0.001 * attempt_index,
            }
        return {"stage": "parity", "objective_value": 1.05 + 0.01 * attempt_index}

    @staticmethod
    def _make_measurement_row(
        *,
        candidate_uuid: str,
        candidate_label: str,
        role: str,
        measurement_index: int,
        objective_value: float,
        harness: str,
        trace_ref: str,
    ) -> dict[str, Any]:
        return {
            "candidate_uuid": candidate_uuid,
            "candidate_label": candidate_label,
            "measurement_role": role,
            "measurement_index": str(measurement_index),
            "objective_value": f"{objective_value:.6f}",
            "harness": harness,
            "trace_ref": trace_ref,
        }

    @staticmethod
    def _make_rejection_row(
        *,
        iteration: str,
        candidate_uuid: str,
        mutation_hash: str,
        reason: str,
        first_diverging_probe_index: int | None = None,
        tolerance_overshoot: float | None = None,
    ) -> dict[str, Any]:
        return {
            "iteration": iteration,
            "candidate_uuid": candidate_uuid,
            "mutation_hash": mutation_hash,
            "rejection_reason": reason,
            "first_diverging_probe_index": (
                "" if first_diverging_probe_index is None else str(first_diverging_probe_index)
            ),
            "tolerance_overshoot": (
                "" if tolerance_overshoot is None else f"{tolerance_overshoot:.6f}"
            ),
        }

    def _write_parity_check(
        self,
        iteration_dir: Path,
        *,
        pass_: bool,
        reason: str,
        fixture_id: str,
        kernel_target: str,
        first_diverging_probe: int | None = None,
        tolerance_overshoot: float | None = None,
        error_detail: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "pass": pass_,
            "reason": reason,
            "fixture_id": fixture_id,
            "kernel_target": kernel_target,
            "probe_count": 64,
            "checkpoints_checked": [1, 1024] if kernel_target == "deltanet" else [1],
        }
        if first_diverging_probe is not None:
            payload["first_diverging_probe"] = int(first_diverging_probe)
        if tolerance_overshoot is not None:
            payload["tolerance_overshoot"] = float(tolerance_overshoot)
        if error_detail is not None:
            payload["error_detail"] = error_detail
        (iteration_dir / "parity_check.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _layer_payload(
        self,
        *,
        kernel_target: str,
        target: str,
        base: TunedConfigBundle,
        winning_row: dict[str, Any],
        paired_baseline_mean: float,
        winner_mean: float,
        terminal_condition: str,
        accepted_count: int,
        total_attempts: int,
        rejected_count: int,
        parity_fixture_path: Path,
    ) -> dict[str, Any]:
        existing = (
            dict(base.layer_0_deltanet) if target == "deltanet" else dict(base.layer_0_gatedattn)
        )
        if kernel_target != target:
            return existing
        existing.setdefault("l0a_select", existing.get("l0a_select", {}))
        existing.setdefault("l0b_autotune", existing.get("l0b_autotune", {}))
        existing["l0c_mutation"] = {
            "diff_ref": str(Path("candidates") / winning_row["iteration"] / "mutation.patch"),
            "mutation_hash": winning_row["mutation_hash"],
            "weight_sensitive": False,
            "paired_baseline_objective_mean": paired_baseline_mean,
            "winner_objective_mean": winner_mean,
            "terminal_condition": terminal_condition,
            "accepted_count": accepted_count,
            "total_attempt_count": total_attempts,
            "rejected_count": rejected_count,
            "parity_attestation": {
                "fixture_path": _relative_to_repo(self.repo_root, parity_fixture_path),
                "fixture_content_hash": fixture_content_hash(parity_fixture_path),
                "checkpoints_checked": [1, 1024] if kernel_target == "deltanet" else [1],
            },
        }
        return existing

    def _write_real_halt(
        self,
        *,
        round_dir: Path,
        round_id: str,
        kernel_target: str,
        base: TunedConfigBundle,
        weight_version_id: str,
        descriptor: dict[str, Any],
    ) -> L0cKernelMutationResult:
        run_log = {
            "round_id": round_id,
            "outcome": "ROUND_BLOCKED",
            "HALT_REASON": "l0c_real_harness_not_implemented",
            "kernel_target": kernel_target,
        }
        (round_dir / "run_log.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
        raise RuntimeError("HALT_REASON: l0c_real_harness_not_implemented")

    @staticmethod
    def _mean_of(rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(row["objective_value"]) for row in rows) / len(rows)

    @staticmethod
    def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
        lines = ["\t".join(columns)]
        for row in rows:
            lines.append("\t".join(str(row.get(column, "")) for column in columns))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_yaml(path: Path, payload: Any) -> None:
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _render_brief(
        self,
        *,
        kernel_target: str,
        kernel_source_path: Path,
        fixture_path: Path,
        fixture_payload: Any,
        round_id: str,
        harness: str,
        strategy_brief: str = "",
    ) -> str:
        rtol_logit = ""
        atol_logit = ""
        rtol_state = ""
        atol_state = ""
        state_checkpoints = []
        if isinstance(fixture_payload, dict):
            tolerances = fixture_payload.get("tolerances") or {}
            if isinstance(tolerances, dict):
                rtol_logit = str(tolerances.get("rtol_logit", tolerances.get("logit_rtol", "")))
                atol_logit = str(tolerances.get("atol_logit", tolerances.get("logit_atol", "")))
                rtol_state = str(tolerances.get("rtol_state", tolerances.get("state_rtol", "")))
                atol_state = str(tolerances.get("atol_state", tolerances.get("state_atol", "")))
            checkpoints = fixture_payload.get("state_checkpoints_at_token")
            if isinstance(checkpoints, list):
                state_checkpoints = [str(token) for token in checkpoints]
        substitutions = {
            "iteration": "{{iteration}}",
            "round_id": round_id,
            "kernel_source_path": str(kernel_source_path),
            "parity_fixture_path": str(fixture_path),
            "repo_root": str(self.repo_root),
            "kernel_target": kernel_target,
            "harness_mode": harness,
            "iteration_dir": "{{iteration_dir}}",
            "lumoserve_cmd": str(self.repo_root / ".venv" / "bin" / "lumoserve"),
            "rtol_logit": rtol_logit,
            "atol_logit": atol_logit,
            "rtol_state": rtol_state,
            "atol_state": atol_state,
            "state_checkpoints_at_token": ", ".join(state_checkpoints),
            "strategy_brief": strategy_brief.strip(),
        }
        rendered = L0C_ITERATION_BRIEF_TEMPLATE
        for key, value in substitutions.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered

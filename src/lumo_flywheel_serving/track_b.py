from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


TRACK_B_BASELINE_TPS = 7.5
TRACK_B_TARGET_MULTIPLIER = 2.0
TRACK_B_TARGET_TPS = TRACK_B_BASELINE_TPS * TRACK_B_TARGET_MULTIPLIER
TRACK_B_SOURCE_REPORT = "docs/reports/auto_research/l0-warm-decode-quality-bounded-track-20260505.md"
TRACK_B_CUTLASS_CLOSEOUT_REPORTS = [
    "docs/reports/auto_research/l0c-fp8-cutlass-loop-20260505.md",
    "docs/reports/auto_research/l0c-fp8-cutlass-round-20260505-closeout.md",
    "docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md",
]

B1_THRESHOLDS = {
    "mean_kl": 0.05,
    "p95_kl": 0.25,
    "top1_agreement": 0.98,
    "entropy_delta_abs": 0.05,
}
B2_THRESHOLDS = {
    "avg_quality_delta_pp": -1.0,
    "single_benchmark_delta_pp": -1.5,
    "workload_behavioral_judge_score_delta": -0.1,
    "needle_recall_ratio": 0.95,
}
B3_THRESHOLDS = {
    "aggregate_quality_score_delta_pp": -0.5,
    "single_benchmark_delta_pp": -1.5,
    "mauve": 0.95,
    "perplexity_ratio_delta": 0.005,
}


@dataclass(frozen=True)
class TrackBLaunchResult:
    round_id: str
    round_dir: Path
    status: str
    target_decode_tps: float
    current_best_decode_tps: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_dir": str(self.round_dir),
            "status": self.status,
            "target_decode_tps": self.target_decode_tps,
            "current_best_decode_tps": self.current_best_decode_tps,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "unknown"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_tsv(path: Path, columns: list[str], rows: list[dict[str, Any]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows or []:
            writer.writerow(row)


def _read_jsonl_sample(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _read_tsv_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows.append({str(key): str(value) for key, value in row.items() if key is not None})
            if len(rows) >= limit:
                break
    return rows


def _trace_stats(path: Path) -> dict[str, Any]:
    rows = _read_jsonl_sample(path, 10_000)
    prompt_tokens = [int(row.get("prompt_tokens", 0) or 0) for row in rows]
    output_tokens = [int(row.get("output_tokens", 0) or 0) for row in rows]
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "row_count_sampled": len(rows),
        "prompt_tokens_max": max(prompt_tokens) if prompt_tokens else None,
        "output_tokens_max": max(output_tokens) if output_tokens else None,
    }


def collect_cutlass_round_memory(repo_root: Path, round_root: Path, limit: int = 8) -> dict[str, Any]:
    """Summarize the prior L0c FP8 CUTLASS rounds for Track B prompts.

    Track B is deliberately wider than the CUTLASS-only loop, but it should
    inherit that loop's negative evidence so agents do not rediscover the same
    schedule/tile/stage failures.
    """
    if not round_root.is_absolute():
        round_root = repo_root / round_root
    round_dirs = sorted(
        [
            path
            for path in round_root.glob("*l0c-mutation-fp8_gemm-*")
            if path.is_dir()
        ],
        key=lambda path: path.name,
        reverse=True,
    )
    rounds: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    research_rows: list[dict[str, Any]] = []
    for round_dir in round_dirs[:limit]:
        run_log_path = round_dir / "run_log.json"
        run_log: dict[str, Any] = {}
        if run_log_path.is_file():
            try:
                loaded = json.loads(run_log_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    run_log = loaded
            except json.JSONDecodeError:
                run_log = {"parse_error": True}
        rounds.append(
            {
                "round_id": round_dir.name,
                "path": str(round_dir),
                "run_log_ref": str(run_log_path) if run_log_path.is_file() else None,
                "outcome": run_log.get("outcome") or run_log.get("status"),
                "terminal_condition": run_log.get("terminal_condition"),
                "halt_reason": run_log.get("HALT_REASON"),
            }
        )
        for row in _read_tsv_rows(round_dir / "mutations_rejected.tsv", 6):
            row["source_round_id"] = round_dir.name
            rejected_rows.append(row)
        for row in _read_tsv_rows(round_dir / "research_memory.tsv", 6):
            row["source_round_id"] = round_dir.name
            research_rows.append(row)
    report_refs = [
        report
        for report in TRACK_B_CUTLASS_CLOSEOUT_REPORTS
        if (repo_root / report).is_file()
    ]
    return {
        "schema": "lumo.track_b.cutlass_prior_memory.v1",
        "round_count_indexed": len(round_dirs),
        "recent_rounds": rounds,
        "rejected_rows_sample": rejected_rows[:limit],
        "research_rows_sample": research_rows[:limit],
        "closeout_report_refs": report_refs,
        "summary": {
            "warm_decode_observed_tps": "7.36-7.39 tok/s in May 5 CUTLASS diagnostics",
            "track_a_surface_status": "exhausted_for_2x_target",
            "negative_memory": [
                "CUTLASS schedule/tile/stage/caller edits left B-weight bytes unchanged.",
                "Warm speed-gate failures were below 0.25% lift, far below the 2x target.",
                "MX/NV block-scaled OpClassBlockScaledTensorOp is not a semantics-preserving direct swap for vLLM's FP32-scale path.",
                "Further CUTLASS-only work needs a new low-level timing lever before full vLLM validation.",
            ],
        },
    }


def render_qwen36_availability_audit(
    *,
    preferred_repo: str = "Qwen/Qwen3.6-27B-FP8",
    fallback_repo: str = "Qwen/Qwen3.5-27B-FP8",
    generated_at: str | None = None,
) -> str:
    generated_at = generated_at or _now_iso()
    return "\n".join(
        [
            "# Qwen 3.6 Availability Audit",
            "",
            f"Generated: {generated_at}",
            "",
            "## Decision",
            "",
            f"- preferred_target: `{preferred_repo}`",
            f"- fallback_target: `{fallback_repo}`",
            "- status: `preferred_available_pending_local_download`",
            "",
            "## Evidence Captured For This Launch",
            "",
            "- Hugging Face has a `Qwen/Qwen3.6-27B-FP8` repository with FP8 safetensor shards, Apache-2.0 license metadata, and Qwen-owned namespace.",
            "- The matching dense `Qwen/Qwen3.6-27B` model card lists 27B parameters, hidden size 5120, 64 layers, and the same 16 x (3 DeltaNet + 1 Gated Attention) hybrid layout family used by the Qwen3.5 target.",
            "- The Qwen3.6 model card recommends vLLM >= 0.19.0 and documents a vLLM MTP speculative config path.",
            "",
            "## Local Follow-Up",
            "",
            "- Download the preferred FP8 checkpoint before live runs.",
            "- Verify tokenizer compatibility against `benchmark_blueprints/families/responses-sdk-adapter-cutover/seed_trace_v5.jsonl`.",
            "- Re-run the P3a in-process timing pass if local config inspection shows layer/head/layout drift from the model card.",
        ]
    ) + "\n"


class TrackBRoundManager:
    def __init__(self, *, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()

    def launch(
        self,
        *,
        round_root: str | Path,
        workload_trace: str | Path,
        model_id: str = "qwen3.6-27b-fp8",
        fallback_model_id: str = "qwen3.5-27b",
        baseline_decode_tps: float = TRACK_B_BASELINE_TPS,
        target_multiplier: float = TRACK_B_TARGET_MULTIPLIER,
        mode: str = "round0_prefix_cache",
        dry_run: bool = False,
        inherit_cutlass_memory: bool = True,
        round_id: str | None = None,
    ) -> TrackBLaunchResult:
        if baseline_decode_tps <= 0:
            raise RuntimeError("--baseline-decode-tps must be > 0")
        if target_multiplier <= 1.0:
            raise RuntimeError("--target-multiplier must be > 1.0")
        if mode not in {"round0_prefix_cache", "round1_spec_decode"}:
            raise RuntimeError("--mode must be round0_prefix_cache or round1_spec_decode")

        trace_path = Path(workload_trace)
        if not trace_path.is_absolute():
            trace_path = self.repo_root / trace_path
        trace_path = trace_path.resolve()
        if not trace_path.is_file():
            raise RuntimeError(f"Track B workload trace missing: {trace_path}")

        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        round_id = round_id or f"{_safe_id(model_id)}-track-b-{mode}-{timestamp}"
        root = Path(round_root)
        if not root.is_absolute():
            root = self.repo_root / root
        round_dir = root.resolve() / "track_b" / round_id
        if round_dir.exists():
            raise RuntimeError(f"Track B round already exists: {round_dir}")

        target_tps = baseline_decode_tps * target_multiplier
        round_dir.mkdir(parents=True)
        for child in ("candidates", "quality_fixture", "logs"):
            (round_dir / child).mkdir()

        trace_stats = _trace_stats(trace_path)
        cutlass_memory = (
            collect_cutlass_round_memory(self.repo_root, Path(round_root))
            if inherit_cutlass_memory
            else {
                "schema": "lumo.track_b.cutlass_prior_memory.v1",
                "round_count_indexed": 0,
                "recent_rounds": [],
                "rejected_rows_sample": [],
                "research_rows_sample": [],
                "closeout_report_refs": [],
                "summary": {},
            }
        )
        self._write_quality_fixture_metadata(round_dir, trace_path, trace_stats)
        self._write_round_ledgers(round_dir, cutlass_memory)

        spec = {
            "schema": "lumo.track_b.round_spec.v1",
            "round_id": round_id,
            "round_type": "track_b_quality_bounded_mutation",
            "source_report": TRACK_B_SOURCE_REPORT,
            "mode": mode,
            "extends_round_type": "l0c_mutation",
            "extends_surface": "fp8_gemm_cutlass_quality_bounded_successor",
            "dry_run": dry_run,
            "model_id": model_id,
            "fallback_model_id": fallback_model_id,
            "workload_trace": str(trace_path),
            "workload_trace_sha256": trace_stats["sha256"],
            "baseline_decode_tps": baseline_decode_tps,
            "target_multiplier": target_multiplier,
            "target_decode_tps": target_tps,
            "success_criteria": {
                "decode_speed_at_least_tps": target_tps,
                "decode_speedup_at_least": target_multiplier,
                "b1_distributional_pass": True,
                "b2_behavioral_pass": True,
                "b3_full_pass_before_promotion": True,
                "human_review_required": True,
            },
            "quality_fixtures": {
                "b1_distributional": "quality_fixture/b1_distributional/v1.yaml",
                "b2_behavioral": "quality_fixture/b2_behavioral/v1.yaml",
                "b3_full": "quality_fixture/b3_full/v1.yaml",
            },
            "prior_cutlass_memory": {
                "enabled": inherit_cutlass_memory,
                "artifact_ref": "prior_cutlass_memory.json",
                "round_count_indexed": cutlass_memory.get("round_count_indexed", 0),
                "closeout_report_refs": cutlass_memory.get("closeout_report_refs", []),
            },
            "started_at": _now_iso(),
        }
        _write_yaml(round_dir / "round_spec.yaml", spec)
        (round_dir / "prior_cutlass_memory.json").write_text(
            json.dumps(cutlass_memory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (round_dir / "prior_cutlass_memory.md").write_text(
            self._render_cutlass_memory_doc(cutlass_memory),
            encoding="utf-8",
        )

        (round_dir / "qwen36_availability_audit.md").write_text(
            render_qwen36_availability_audit(), encoding="utf-8"
        )
        (round_dir / "strategy_brief.md").write_text(
            self._render_strategy_brief(spec, trace_stats, cutlass_memory), encoding="utf-8"
        )
        (round_dir / "iteration_brief.md").write_text(
            self._render_iteration_brief(spec), encoding="utf-8"
        )
        self._seed_candidate(round_dir, spec)
        self._write_run_log(round_dir, spec, status="initialized_dry_run" if dry_run else "initialized")

        return TrackBLaunchResult(
            round_id=round_id,
            round_dir=round_dir,
            status="initialized_dry_run" if dry_run else "initialized",
            target_decode_tps=target_tps,
            current_best_decode_tps=None,
        )

    def _write_quality_fixture_metadata(
        self,
        round_dir: Path,
        trace_path: Path,
        trace_stats: dict[str, Any],
    ) -> None:
        b1 = {
            "schema": "lumo.track_b.b1_distributional_fixture.v1",
            "source_trace": str(trace_path),
            "source_trace_sha256": trace_stats["sha256"],
            "probe_policy": {
                "held_out_from_proposer": True,
                "prompt_count": 64,
                "tokens_per_prompt": 256,
                "stratification": ["length_bucket", "task_type_bucket"],
            },
            "thresholds": B1_THRESHOLDS,
            "runner": "scripts/run_b1_distributional.py",
        }
        b2 = {
            "schema": "lumo.track_b.b2_behavioral_fixture.v1",
            "source_trace": str(trace_path),
            "source_trace_sha256": trace_stats["sha256"],
            "benchmarks": ["mmlu_mini", "gsm8k_mini", "humaneval", "truthfulqa_mini", "workload_behavioral", "needle"],
            "thresholds": B2_THRESHOLDS,
            "runner": "scripts/run_b2_behavioral.py",
        }
        b3 = {
            "schema": "lumo.track_b.b3_full_fixture.v1",
            "source_trace": str(trace_path),
            "source_trace_sha256": trace_stats["sha256"],
            "benchmarks": ["mmlu", "gsm8k", "humaneval_plus", "math_500", "ifeval", "ruler", "mauve", "held_out_perplexity"],
            "thresholds": B3_THRESHOLDS,
            "runner": "scripts/run_b3_full.py",
        }
        _write_yaml(round_dir / "quality_fixture" / "b1_distributional" / "v1.yaml", b1)
        _write_yaml(round_dir / "quality_fixture" / "b2_behavioral" / "v1.yaml", b2)
        _write_yaml(round_dir / "quality_fixture" / "b3_full" / "v1.yaml", b3)

    def _write_round_ledgers(self, round_dir: Path, cutlass_memory: dict[str, Any]) -> None:
        (round_dir / "branch_log.json").write_text("[]\n", encoding="utf-8")
        (round_dir / "winning_diffs.md").write_text("# Winning Diffs\n\nNo accepted Track B candidates yet.\n", encoding="utf-8")
        _write_tsv(
            round_dir / "mutations_rejected.tsv",
            ["candidate_id", "tier", "cost_bucket", "reason", "first_failing_metric", "recorded_at"],
        )
        prior_rejections = []
        for index, row in enumerate(cutlass_memory.get("rejected_rows_sample", []), start=1):
            prior_rejections.append(
                {
                    "candidate_id": f"cutlass_prior_{index:03d}",
                    "tier": "prior_cutlass_l0c",
                    "cost_bucket": row.get("rejection_reason") or row.get("failure_class") or "prior_negative_memory",
                    "reason": row.get("rejection_reason") or row.get("next_implication") or json.dumps(row, sort_keys=True),
                    "first_failing_metric": row.get("first_diverging_probe_index") or "",
                    "recorded_at": _now_iso(),
                }
            )
        _write_tsv(
            round_dir / "prior_cutlass_rejections.tsv",
            ["candidate_id", "tier", "cost_bucket", "reason", "first_failing_metric", "recorded_at"],
            prior_rejections,
        )
        _write_tsv(
            round_dir / "quality_gate_history.tsv",
            ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
        )

    def _seed_candidate(self, round_dir: Path, spec: dict[str, Any]) -> None:
        candidate_dir = round_dir / "candidates" / "000"
        candidate_dir.mkdir(parents=True)
        mode = str(spec["mode"])
        if mode == "round0_prefix_cache":
            serve_config = {
                "enable_prefix_caching": True,
                "block_size": 32,
                "enable_chunked_prefill": True,
                "num_gpu_blocks_override": "auto_high_for_gb10_unified_memory",
                "lmcache": {
                    "enabled": True,
                    "cpu_tier": True,
                    "disk_tier": True,
                    "status": "requires_runtime_validation",
                },
            }
            thesis = "Round 0 enables aggressive prefix reuse so cache-hit agent turns skip repeated long prefill."
            counter = "vllm prefix-cache hit/query counters should rise and per-turn wall time should drop on turns 2-N."
        else:
            serve_config = {
                "speculative_config": {
                    "pld": {"enabled": True, "ngram_prompt_lookup_max": 8},
                    "eagle3_or_mtp": {"enabled": True, "status": "requires_draft_model_audit"},
                }
            }
            thesis = "Round 1 amortizes target-model weight reads by accepting multiple verified tokens per target forward pass."
            counter = "speculative acceptance rate should exceed 0.5 on edit/echo or reasoning turns and decode tok/s should rise."
        _write_yaml(candidate_dir / "serve_config.yaml", serve_config)
        (candidate_dir / "candidate_analysis.md").write_text(
            "\n".join(
                [
                    "# Candidate 000 Analysis",
                    "",
                    f"- speed_thesis: {thesis}",
                    f"- expected_affected_counter: {counter}",
                    "- quality_risk: strong-equivalence path should have near-zero B-1 KL; any drift indicates cache/state or rejection-sampling correctness bugs.",
                    "- why_not_prior_failure: this is a Track B config/decoding path, not another Track A tile/schedule mutation that leaves bytes-per-token unchanged.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_run_log(self, round_dir: Path, spec: dict[str, Any], *, status: str) -> None:
        payload = {
            "schema": "lumo.track_b.run_log.v1",
            "round_id": spec["round_id"],
            "status": status,
            "target_decode_tps": spec["target_decode_tps"],
            "baseline_decode_tps": spec["baseline_decode_tps"],
            "next_command": (
                "scripts/run_b1_distributional.py --candidate-metrics <metrics.json> "
                f"--fixture {round_dir / 'quality_fixture' / 'b1_distributional' / 'v1.yaml'}"
            ),
            "written_at": _now_iso(),
        }
        (round_dir / "run_log.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _render_cutlass_memory_doc(self, cutlass_memory: dict[str, Any]) -> str:
        summary = cutlass_memory.get("summary", {})
        lines = [
            "# Prior CUTLASS Auto-Research Memory",
            "",
            f"- indexed_round_count: {cutlass_memory.get('round_count_indexed', 0)}",
            f"- warm_decode_observed_tps: {summary.get('warm_decode_observed_tps', 'unknown')}",
            f"- track_a_surface_status: {summary.get('track_a_surface_status', 'unknown')}",
            "",
            "## Closeout Reports",
            "",
        ]
        for ref in cutlass_memory.get("closeout_report_refs", []):
            lines.append(f"- `{ref}`")
        if not cutlass_memory.get("closeout_report_refs"):
            lines.append("- none found")
        lines.extend(["", "## Negative Memory", ""])
        for item in summary.get("negative_memory", []):
            lines.append(f"- {item}")
        lines.extend(["", "## Recent Rounds", ""])
        for round_row in cutlass_memory.get("recent_rounds", [])[:8]:
            lines.append(
                "- `{round_id}` outcome={outcome} terminal={terminal}".format(
                    round_id=round_row.get("round_id"),
                    outcome=round_row.get("outcome"),
                    terminal=round_row.get("terminal_condition") or round_row.get("halt_reason"),
                )
            )
        return "\n".join(lines) + "\n"

    def _render_strategy_brief(
        self,
        spec: dict[str, Any],
        trace_stats: dict[str, Any],
        cutlass_memory: dict[str, Any],
    ) -> str:
        memory_summary = cutlass_memory.get("summary", {})
        return "\n".join(
            [
                "# Track B Strategy Brief",
                "",
                f"- source_report: `{TRACK_B_SOURCE_REPORT}`",
                f"- extends: `L0c FP8 CUTLASS auto-research`; prior memory at `prior_cutlass_memory.md`",
                f"- baseline_decode_tps: {spec['baseline_decode_tps']}",
                f"- target_decode_tps: {spec['target_decode_tps']}",
                f"- mode: `{spec['mode']}`",
                f"- workload_trace_sha256: `{trace_stats['sha256']}`",
                "",
                "## Bottleneck",
                "",
                "- Warm-cache decode is anchored on the FP8 GEMM family: ffn_linear, deltanet_projection_linear, and gatedattn_projection_linear.",
                "- The prior Track A tile/schedule surface is bandwidth bounded and exhausted for the 2x target.",
                "- Track B changes serving behavior or runtime bytes-per-token while preserving shipped FP8 target weights.",
                "",
                "## Prior CUTLASS Round Memory",
                "",
                f"- indexed_round_count: {cutlass_memory.get('round_count_indexed', 0)}",
                f"- observed_warm_decode: {memory_summary.get('warm_decode_observed_tps', 'unknown')}",
                f"- prior_surface_status: {memory_summary.get('track_a_surface_status', 'unknown')}",
                "- Do not retry schedule/tile/stage/caller mutations unless a new low-level timing lever proves a material per-kernel win.",
                "- The May 5 speed-gate failures improved only around 0.18-0.24%, so they are explicit negative memory for this 2x objective.",
                "",
                "## Required Gates",
                "",
                "- B-1 distributional gate before ranking any candidate.",
                "- B-2 behavioral gate for top candidates.",
                "- B-3 full benchmark plus human review before promotion.",
            ]
        ) + "\n"

    def _render_iteration_brief(self, spec: dict[str, Any]) -> str:
        if spec["mode"] == "round0_prefix_cache":
            objective = "Ship prefix caching + LMCache CPU/disk-tier validation for cache-hit Codex turns."
            self_verify = "Measure prefix-cache hit rate, B-1 KL near zero, and latency reduction on turns 2-N."
        else:
            objective = "Bring up PLD + Eagle/MTP-style speculative decoding while preserving target-model output distribution."
            self_verify = "Measure greedy token equality, B-1 KL near zero, acceptance rate, and decode tok/s."
        return "\n".join(
            [
                "# Track B Iteration Brief",
                "",
                f"Round: `{spec['round_id']}`",
                f"Objective: {objective}",
                "",
                "## Hard Success Target",
                "",
                f"- Decode throughput must reach at least {spec['target_decode_tps']:.2f} tok/s.",
                f"- Baseline is {spec['baseline_decode_tps']:.2f} tok/s, so this is a {spec['target_multiplier']:.2f}x gate.",
                "",
                "## Self Verify",
                "",
                f"- {self_verify}",
                "- Do not modify quality fixtures, gate runners, Track B ledgers, or controller files from a candidate checkout.",
            ]
        ) + "\n"


def evaluate_b1_metrics(metrics: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = thresholds or B1_THRESHOLDS
    checks = {
        "mean_kl": float(metrics.get("mean_kl", float("inf"))) <= float(thresholds["mean_kl"]),
        "p95_kl": float(metrics.get("p95_kl", float("inf"))) <= float(thresholds["p95_kl"]),
        "top1_agreement": float(metrics.get("top1_agreement", -float("inf"))) >= float(thresholds["top1_agreement"]),
        "entropy_delta_abs": abs(float(metrics.get("entropy_delta", float("inf")))) <= float(thresholds["entropy_delta_abs"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "metrics": metrics, "thresholds": thresholds}


def evaluate_b2_metrics(metrics: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = thresholds or B2_THRESHOLDS
    benchmark_deltas = [float(value) for value in metrics.get("benchmark_deltas_pp", {}).values()]
    avg_delta = float(metrics.get("avg_quality_delta_pp", sum(benchmark_deltas) / len(benchmark_deltas) if benchmark_deltas else -999.0))
    worst_delta = min(benchmark_deltas) if benchmark_deltas else float(metrics.get("single_benchmark_worst_delta_pp", -999.0))
    checks = {
        "avg_quality_delta_pp": avg_delta >= float(thresholds["avg_quality_delta_pp"]),
        "single_benchmark_delta_pp": worst_delta >= float(thresholds["single_benchmark_delta_pp"]),
        "workload_behavioral_judge_score_delta": float(metrics.get("workload_behavioral_judge_score_delta", -999.0)) >= float(thresholds["workload_behavioral_judge_score_delta"]),
        "needle_recall_ratio": float(metrics.get("needle_recall_ratio", -float("inf"))) >= float(thresholds["needle_recall_ratio"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "metrics": metrics, "thresholds": thresholds}


def evaluate_b3_metrics(metrics: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = thresholds or B3_THRESHOLDS
    benchmark_deltas = [float(value) for value in metrics.get("benchmark_deltas_pp", {}).values()]
    worst_delta = min(benchmark_deltas) if benchmark_deltas else float(metrics.get("single_benchmark_worst_delta_pp", -999.0))
    checks = {
        "aggregate_quality_score_delta_pp": float(metrics.get("aggregate_quality_score_delta_pp", -999.0)) >= float(thresholds["aggregate_quality_score_delta_pp"]),
        "single_benchmark_delta_pp": worst_delta >= float(thresholds["single_benchmark_delta_pp"]),
        "mauve": float(metrics.get("mauve", -float("inf"))) >= float(thresholds["mauve"]),
        "perplexity_ratio_delta": float(metrics.get("perplexity_ratio_delta", float("inf"))) <= float(thresholds["perplexity_ratio_delta"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "metrics": metrics, "thresholds": thresholds}

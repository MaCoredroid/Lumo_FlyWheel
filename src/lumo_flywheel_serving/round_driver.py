from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .auto_research import AutoResearchRoundManager, BASELINE_ITERATIONS, OfflineAutoResearchRunner, load_baseline_bundle
from .registry import load_registry

ROUND_PASSED = "ROUND_PASSED"
ROUND_INFEASIBLE = "ROUND_INFEASIBLE"
ROUND_BLOCKED = "ROUND_BLOCKED"
ROUND_BUNDLE_REJECTED = "ROUND_BUNDLE_REJECTED"
ROUND_BUNDLE_READY = "ROUND_BUNDLE_READY"

TERMINAL_OUTCOMES = frozenset(
    {
        ROUND_PASSED,
        ROUND_INFEASIBLE,
        ROUND_BLOCKED,
        ROUND_BUNDLE_REJECTED,
        ROUND_BUNDLE_READY,
    }
)
HONEST_TERMINAL_EXIT_ZERO = frozenset({ROUND_PASSED, ROUND_INFEASIBLE})


@dataclass(frozen=True)
class RoundContext:
    round_id: str
    round_dir: Path
    round_branch: str
    worktree: Path
    round_spec_path: Path
    round_spec: dict[str, Any]
    harness_mode: str
    registry_path: Path
    tuned_config_root: Path
    port: int = 8000
    proxy_port: int = 8001
    iteration_cap: int | None = None

    @classmethod
    def from_bootstrap_json(
        cls,
        payload: dict[str, Any],
        *,
        harness_mode: str,
        registry_path: Path,
        tuned_config_root: Path,
        port: int = 8000,
        proxy_port: int = 8001,
        iteration_cap: int | None = None,
    ) -> "RoundContext":
        round_spec_path = Path(str(payload["round_spec_path"])).resolve()
        raw = yaml.safe_load(round_spec_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid round spec: {round_spec_path}")
        return cls(
            round_id=str(payload["round_id"]),
            round_dir=Path(str(payload["round_dir"])).resolve(),
            round_branch=str(payload["round_branch"]),
            worktree=Path(str(payload.get("worktree_path", payload["round_dir"]))).resolve(),
            round_spec_path=round_spec_path,
            round_spec=raw,
            harness_mode=harness_mode,
            registry_path=Path(registry_path).resolve(),
            tuned_config_root=Path(tuned_config_root).resolve(),
            port=port,
            proxy_port=proxy_port,
            iteration_cap=iteration_cap,
        )


@dataclass(frozen=True)
class RoundResult:
    round_id: str
    round_branch: str
    outcome: str
    stopping_reason: str
    bundle_path: str | None
    iterations_total: int
    feasible_count: int
    rescreened_count: int
    holdout_validation: str
    live_gate: str
    blocker: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_branch": self.round_branch,
            "outcome": self.outcome,
            "stopping_reason": self.stopping_reason,
            "bundle_path": self.bundle_path,
            "iterations_total": self.iterations_total,
            "feasible_count": self.feasible_count,
            "rescreened_count": self.rescreened_count,
            "holdout_validation": self.holdout_validation,
            "live_gate": self.live_gate,
            "blocker": self.blocker,
        }

    def write_report(self, path: Path) -> None:
        payload = {
            "schema_version": "lumo.auto_research.round_result.v1",
            "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            **self.as_dict(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def restore_worktree_head(worktree: Path, branch: str) -> None:
    subprocess.run(["git", "checkout", "--force", branch], cwd=worktree, check=True, capture_output=True, text=True)
    subprocess.run(["git", "reset", "--hard", branch], cwd=worktree, check=True, capture_output=True, text=True)


def run_round(ctx: RoundContext) -> RoundResult:
    repo_root = _git(["rev-parse", "--show-toplevel"], cwd=ctx.worktree).strip()
    manager = AutoResearchRoundManager(
        registry_path=ctx.registry_path,
        repo_root=repo_root,
        tuned_config_root=ctx.tuned_config_root,
        port=ctx.port,
        proxy_port=ctx.proxy_port,
    )

    try:
        _run_baselines(manager, ctx)
        if ctx.harness_mode == "synthetic":
            _run_synthetic_main_loop(manager, ctx)
        else:
            _run_codex_main_loop(manager, ctx)

        if not (ctx.round_dir / "rescreen_trace.json").is_file():
            manager.rescreen(
                round_id=ctx.round_id,
                top_k=int(ctx.round_spec.get("rescreen_top_k", 3)),
                measurements_per_candidate_screen=int(ctx.round_spec.get("measurements_per_candidate_screen", 3)),
                measurements_per_candidate_full=int(ctx.round_spec.get("measurements_per_candidate_full", 1)),
                harness=ctx.harness_mode,
            )
        try:
            winner_uuid = _winner_parent_uuid(ctx.round_dir)
        except RuntimeError as exc:
            if str(exc) != "no_feasible_rescreen_winner":
                raise
            return _result_from_status(
                manager,
                ctx,
                outcome=ROUND_INFEASIBLE,
                stopping_reason="no_feasible_rescreen_winner",
                bundle_path=None,
                holdout_validation="not_run",
                live_gate="skipped_no_bundle",
                blocker=None,
            )
        holdout_path = ctx.round_dir / "holdout_trace.json"
        if holdout_path.is_file():
            holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        else:
            holdout = manager.validate_holdout(
                round_id=ctx.round_id,
                candidate_uuid=winner_uuid,
                harness=ctx.harness_mode,
            )
        if not bool(holdout.get("pass")):
            return _result_from_status(
                manager,
                ctx,
                outcome=ROUND_INFEASIBLE,
                stopping_reason="holdout_rejected",
                bundle_path=None,
                holdout_validation="fail",
                live_gate="skipped_no_bundle",
                blocker=",".join(str(reason) for reason in holdout.get("reasons_failed", [])),
            )

        finalized = manager.finalize_round(round_id=ctx.round_id, dry_run=ctx.harness_mode == "synthetic")
        live_gate = "skipped_fixture_mode" if ctx.harness_mode == "synthetic" else "not_run"
        return _result_from_status(
            manager,
            ctx,
            outcome=ROUND_BUNDLE_READY,
            stopping_reason="ok",
            bundle_path=str(finalized["bundle_path"]),
            holdout_validation="pass",
            live_gate=live_gate,
        )
    except Exception as exc:
        return _result_from_status(
            manager,
            ctx,
            outcome=ROUND_BLOCKED,
            stopping_reason="exception",
            bundle_path=None,
            holdout_validation="not_run",
            live_gate="skipped_no_bundle",
            blocker=str(exc),
        )


def _run_baselines(manager: AutoResearchRoundManager, ctx: RoundContext) -> None:
    finalized = _finalized_iterations(ctx.round_dir)
    for iteration in BASELINE_ITERATIONS:
        if iteration in finalized:
            continue
        candidate_path = ctx.round_dir / "candidates" / iteration / "candidate.yaml"
        manager.measure(round_id=ctx.round_id, candidate_path=candidate_path, harness=ctx.harness_mode)
        manager.commit_candidate(
            round_id=ctx.round_id,
            iteration=iteration,
            status="baseline",
            notes=f"default-config baseline replay {iteration.rsplit('_', 1)[1]}",
            harness=ctx.harness_mode,
        )


def _run_synthetic_main_loop(manager: AutoResearchRoundManager, ctx: RoundContext) -> None:
    finalized = _finalized_iterations(ctx.round_dir)
    if str(ctx.round_spec.get("active_layer", "L1")).upper() == "L2":
        for index, candidate in enumerate(
            AutoResearchRoundManager._request_shaping_candidate_plan(
                dict(ctx.round_spec.get("frozen_vllm_config") or {})
            ),
            start=1,
        ):
            if index > int(ctx.iteration_cap or ctx.round_spec.get("iteration_cap", 12)):
                break
            iteration = f"{index:03d}"
            if iteration in finalized:
                continue
            candidate_dir = ctx.round_dir / "candidates" / iteration
            candidate_dir.mkdir(parents=True, exist_ok=True)
            (candidate_dir / "candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
            manager.measure(round_id=ctx.round_id, candidate_path=candidate_dir / "candidate.yaml", harness=ctx.harness_mode)
            trace = json.loads((candidate_dir / "measurement_trace.json").read_text(encoding="utf-8"))
            status = "keep" if bool(trace.get("feasible")) else ("crash" if not trace.get("no_oom_events") else "discard")
            manager.commit_candidate(
                round_id=ctx.round_id,
                iteration=iteration,
                status=status,
                notes="synthetic run-round L2 request-shaping candidate",
                harness=ctx.harness_mode,
            )
        return

    registry = load_registry(ctx.registry_path)
    model_config = registry[str(ctx.round_spec["model_id"])]
    workload_file = Path(str(ctx.round_spec["workload_file"]))
    from .auto_research import SyntheticWorkloadDistribution

    workload_distribution = SyntheticWorkloadDistribution.from_file(
        workload_file,
        model_config=model_config,
        family_id=str(ctx.round_spec["family_id"]),
    )
    runner = OfflineAutoResearchRunner(
        model_config=model_config,
        family_id=str(ctx.round_spec["family_id"]),
        output_root=ctx.round_dir / "_synthetic_plan",
        workload=workload_distribution,
        baseline_bundle=load_baseline_bundle(None),
        weight_version_id=str(ctx.round_spec["weight_version_id"]),
        iteration_cap=int(ctx.iteration_cap or ctx.round_spec.get("iteration_cap", 12)),
    )
    for index, candidate in enumerate(runner._candidate_plan(), start=1):
        if index > int(ctx.iteration_cap or ctx.round_spec.get("iteration_cap", 12)):
            break
        iteration = f"{index:03d}"
        if iteration in finalized:
            continue
        candidate_dir = ctx.round_dir / "candidates" / iteration
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        manager.measure(round_id=ctx.round_id, candidate_path=candidate_dir / "candidate.yaml", harness=ctx.harness_mode)
        trace = json.loads((candidate_dir / "measurement_trace.json").read_text(encoding="utf-8"))
        status = "keep" if bool(trace.get("feasible")) else ("crash" if not trace.get("no_oom_events") else "discard")
        manager.commit_candidate(
            round_id=ctx.round_id,
            iteration=iteration,
            status=status,
            notes="synthetic run-round candidate",
            harness=ctx.harness_mode,
        )


def _run_codex_main_loop(manager: AutoResearchRoundManager, ctx: RoundContext) -> None:
    for index in range(1, int(ctx.iteration_cap or ctx.round_spec.get("iteration_cap", 12)) + 1):
        iteration = f"{index:03d}"
        if iteration in _finalized_iterations(ctx.round_dir):
            continue
        iteration_dir = ctx.round_dir / "candidates" / iteration
        iteration_dir.mkdir(parents=True, exist_ok=True)
        prompt = _iteration_prompt(ctx, iteration=iteration, next_iteration=f"{index + 1:03d}")
        transcript = iteration_dir / "agent_session.jsonl"
        with transcript.open("wb") as transcript_handle:
            result = subprocess.run(
                [
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
                    str(iteration_dir / "agent_last_message.txt"),
                    "--skip-git-repo-check",
                    "-",
                ],
                input=prompt.encode(),
                stdout=transcript_handle,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                timeout=int(ctx.round_spec.get("per_iteration_codex_wall_clock_s", 45 * 60)),
            )
        if result.returncode == 2:
            raise RuntimeError(f"iteration {iteration} blocked")
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip() or f"codex exited {result.returncode}")
        status = manager.status(round_id=ctx.round_id)
        if status["iterations_total"] >= index + 2:
            continue


def _iteration_prompt(ctx: RoundContext, *, iteration: str, next_iteration: str) -> str:
    template = (ctx.round_dir / "iteration_brief.md").read_text(encoding="utf-8")
    repo_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=ctx.worktree).strip())
    lumoserve_cmd = (
        f"{repo_root / '.venv' / 'bin' / 'lumoserve'} "
        f"--registry {ctx.registry_path} "
        f"--port {ctx.port} "
        f"--proxy-port {ctx.proxy_port}"
    )
    template = template.replace("lumoserve auto-research", f"{lumoserve_cmd} auto-research")
    active_layer = str(ctx.round_spec.get("active_layer", "L1")).upper()
    if active_layer == "L2":
        candidate_schema_instruction = (
            "Schema: parent HLD §5.3.3 L2 request_shaping keys only: "
            "concurrency_cap_eval, concurrency_cap_rollout, admission_queue_depth_max, "
            "per_request_kv_budget, priority_preemption. No L0, no L1, no L3 keys. "
            "No extra keys. The lower-layer vllm_config is frozen from baseline_bundle_path."
        )
    else:
        candidate_schema_instruction = (
            "Schema: parent HLD §5.3.2 L1 action space keys only. No L0, no L2, "
            "no L3 keys. No extra keys. The baseline case (iteration=000) is "
            "the default-config dict from the model registry."
        )
    values = {
        "round_id": ctx.round_id,
        "iteration": iteration,
        "next_iteration": next_iteration,
        "round_dir": str(ctx.worktree),
        "harness_mode": ctx.harness_mode,
        "harness_generator_prefix": (
            "SyntheticMeasurementFixture" if ctx.harness_mode == "synthetic" else "RealMeasurementHarness"
        ),
        "lumoserve_cmd": lumoserve_cmd,
        "model_id": str(ctx.round_spec["model_id"]),
        "family_id": str(ctx.round_spec["family_id"]),
        "active_layer": active_layer,
        "round_branch": ctx.round_branch,
        "per_candidate_wall_clock_minutes": str(int(ctx.round_spec.get("screen_profile_s", 900)) // 60),
        "workload_file": str(ctx.round_spec["workload_file"]),
        "candidate_schema_instruction": candidate_schema_instruction,
    }
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def _finalized_iterations(round_dir: Path) -> set[str]:
    results_path = round_dir / "results.tsv"
    if not results_path.is_file():
        return set()
    return {
        row["iteration"]
        for row in _read_results_dicts(results_path)
        if row.get("iteration") and row.get("status")
    }


def _winner_parent_uuid(round_dir: Path) -> str:
    rows = _read_results_dicts(round_dir / "results.tsv")
    rescreened = [row for row in rows if row.get("status") == "rescreened" and row.get("objective_mean") and row.get("notes") != "inconsistent_rescreen"]
    if rescreened:
        winner = min(
            rescreened,
            key=lambda row: (-float(row["objective_mean"]), row["iteration"], row["candidate_uuid"]),
        )
        return str(winner["parent_candidate_uuid"])
    raise RuntimeError("no_feasible_rescreen_winner")


def _read_results_dicts(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [
        {key: values[index] if index < len(values) else "" for index, key in enumerate(header)}
        for values in (line.split("\t") for line in lines[1:] if line.strip())
    ]


def _result_from_status(
    manager: AutoResearchRoundManager,
    ctx: RoundContext,
    *,
    outcome: str,
    stopping_reason: str,
    bundle_path: str | None,
    holdout_validation: str,
    live_gate: str,
    blocker: str | None = None,
) -> RoundResult:
    status = manager.status(round_id=ctx.round_id)
    result = RoundResult(
        round_id=ctx.round_id,
        round_branch=ctx.round_branch,
        outcome=outcome,
        stopping_reason=stopping_reason,
        bundle_path=bundle_path,
        iterations_total=int(status.get("iterations_total", 0)),
        feasible_count=int(status.get("feasible_count", 0)),
        rescreened_count=int(status.get("rescreened_count", 0)),
        holdout_validation=holdout_validation,
        live_gate=live_gate,
        blocker=blocker,
    )
    result.write_report(ctx.round_dir / "round_result.json")
    return result


def run_round_exit_code(result: RoundResult) -> int:
    """CLI policy: JSON outcome is authoritative; exit code reports lifecycle health."""
    if result.outcome in HONEST_TERMINAL_EXIT_ZERO:
        return 0
    if result.outcome == ROUND_BUNDLE_READY and not _bundle_ready_needs_live_gate(result):
        return 0
    return 1


def _bundle_ready_needs_live_gate(result: RoundResult) -> bool:
    return result.live_gate not in {"pass", "skipped_fixture_mode"}


def _git(args: list[str], *, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

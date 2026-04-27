from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from .auto_research import AutoResearchRoundManager, BASELINE_ITERATIONS, OfflineAutoResearchRunner, ResultsRow, load_baseline_bundle
from .registry import load_registry

ROUND_PASSED = "ROUND_PASSED"
ROUND_INFEASIBLE = "ROUND_INFEASIBLE"
ROUND_BLOCKED = "ROUND_BLOCKED"
ROUND_BUNDLE_REJECTED = "ROUND_BUNDLE_REJECTED"
ROUND_BUNDLE_READY = "ROUND_BUNDLE_READY"

AGENT_RUNTIMES = frozenset({"codex", "claude"})
DEFAULT_AGENT_RUNTIME = "codex"
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_CLAUDE_EFFORT = "xhigh"
DEFAULT_CLAUDE_PERMISSION_MODE = "bypassPermissions"

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
    agent_runtime: str = DEFAULT_AGENT_RUNTIME

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
        agent_runtime: str = DEFAULT_AGENT_RUNTIME,
    ) -> "RoundContext":
        round_spec_path = Path(str(payload["round_spec_path"])).resolve()
        raw = yaml.safe_load(round_spec_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid round spec: {round_spec_path}")
        if agent_runtime not in AGENT_RUNTIMES:
            raise RuntimeError(
                f"Invalid agent_runtime {agent_runtime!r}; expected one of {sorted(AGENT_RUNTIMES)}"
            )
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
            agent_runtime=agent_runtime,
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
            _run_agent_main_loop(manager, ctx)

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


def run_replay_round(
    *,
    registry_path: Path,
    tuned_config_root: Path,
    port: int,
    proxy_port: int,
    workload_file: Path,
    baselines: int,
    import_candidate: Path,
    rescreens_screen: int,
    rescreens_full: int,
    holdout_rows: int,
    round_root: Path,
    harness_mode: str,
    model_id: str,
    family_id: str | None = None,
    sprint: str = "sprint-0",
) -> dict[str, Any]:
    if baselines < 1 or baselines > len(BASELINE_ITERATIONS):
        raise RuntimeError(f"replay-round requires --baselines between 1 and {len(BASELINE_ITERATIONS)}")
    if rescreens_screen < 0 or rescreens_full < 0:
        raise RuntimeError("replay-round rescreen counts must be >= 0")
    if holdout_rows < 1:
        raise RuntimeError("replay-round requires --holdout-rows >= 1")
    import_candidate = import_candidate.resolve()
    if not import_candidate.is_file():
        raise RuntimeError(f"Imported candidate file does not exist: {import_candidate}")
    workload_file = workload_file.resolve()
    if not workload_file.is_file():
        raise RuntimeError(f"Workload descriptor does not exist: {workload_file}")
    workload_payload = yaml.safe_load(workload_file.read_text(encoding="utf-8"))
    if not isinstance(workload_payload, dict):
        raise RuntimeError(f"Workload descriptor must be a mapping: {workload_file}")
    resolved_family_id = family_id or str(workload_payload.get("family_id") or "").strip()
    if not resolved_family_id:
        raise RuntimeError("replay-round could not infer family_id from workload descriptor")
    pool_size = int(workload_payload.get("pool_size") or len(workload_payload.get("pool_families") or []) or 0)
    split = workload_payload.get("split_per_family") if isinstance(workload_payload.get("split_per_family"), dict) else {}
    min_holdout_rows = pool_size * int(split.get("holdout_rows", 1)) if pool_size else 1
    if holdout_rows < min_holdout_rows:
        raise RuntimeError(f"replay-round --holdout-rows {holdout_rows} is below required minimum {min_holdout_rows}")

    repo_root = Path(_git(["rev-parse", "--show-toplevel"], cwd=workload_file.parent).strip())
    manager = AutoResearchRoundManager(
        registry_path=registry_path,
        repo_root=repo_root,
        tuned_config_root=tuned_config_root,
        port=port,
        proxy_port=proxy_port,
    )
    bootstrap = manager.bootstrap_round(
        model_id=model_id,
        family_id=resolved_family_id,
        sprint=sprint,
        workload_file=workload_file,
        weight_version_id=None,
        round_root=round_root,
        harness_type=harness_mode,
        skip_codex_preflight=True,
    )
    round_id = str(bootstrap["round_id"])
    round_dir = Path(str(bootstrap["round_dir"])).resolve()

    for iteration in BASELINE_ITERATIONS[:baselines]:
        manager.measure(round_id=round_id, candidate_path=round_dir / "candidates" / iteration / "candidate.yaml", harness=harness_mode)
        manager.commit_candidate(
            round_id=round_id,
            iteration=iteration,
            status="baseline",
            notes=f"default-config baseline replay {iteration.rsplit('_', 1)[1]}",
            harness=harness_mode,
        )

    import_dir = round_dir / "candidates" / "import_001"
    import_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(import_candidate, import_dir / "candidate.yaml")
    imported = manager.measure(
        round_id=round_id,
        candidate_path=import_dir / "candidate.yaml",
        profile="screen",
        harness=harness_mode,
    )
    imported_uuid = str(imported["candidate_uuid"])
    manager.commit_candidate(
        round_id=round_id,
        iteration="import_001",
        status="keep",
        notes="imported candidate replay",
        harness=harness_mode,
    )

    rescreened = _replay_import_rescreens(
        manager=manager,
        round_id=round_id,
        round_dir=round_dir,
        parent_iteration="import_001",
        parent_uuid=imported_uuid,
        screen_count=rescreens_screen,
        full_count=rescreens_full,
        harness_mode=harness_mode,
    )
    holdout = manager.validate_holdout(round_id=round_id, candidate_uuid=imported_uuid, harness=harness_mode)
    if not bool(holdout.get("pass")):
        return {
            "round_id": round_id,
            "round_dir": str(round_dir),
            "round_branch": bootstrap["round_branch"],
            "outcome": ROUND_INFEASIBLE,
            "stopping_reason": "holdout_rejected",
            "imported_candidate_uuid": imported_uuid,
            "holdout_validation": "fail",
            "rescreened": rescreened,
            "blocker": ",".join(str(reason) for reason in holdout.get("reasons_failed", [])),
        }

    finalized = manager.finalize_round(
        round_id=round_id,
        dry_run=harness_mode == "synthetic",
        imported_from_candidate=str(import_candidate),
        imported_from_commit=_source_commit_for(import_candidate, cwd=repo_root),
    )
    return {
        "round_id": round_id,
        "round_dir": str(round_dir),
        "round_branch": bootstrap["round_branch"],
        "outcome": ROUND_BUNDLE_READY,
        "stopping_reason": "ok",
        "imported_candidate_uuid": imported_uuid,
        "holdout_validation": "pass",
        "rescreened": rescreened,
        **finalized,
    }


def _replay_import_rescreens(
    *,
    manager: AutoResearchRoundManager,
    round_id: str,
    round_dir: Path,
    parent_iteration: str,
    parent_uuid: str,
    screen_count: int,
    full_count: int,
    harness_mode: str,
) -> list[dict[str, Any]]:
    parent_dir = round_dir / "candidates" / parent_iteration
    parent_row = next(row for row in manager._read_results(round_dir / "results.tsv") if row.candidate_uuid == parent_uuid)
    parent_value = float(parent_row.eval_throughput or 0.0)
    measured_screen: list[tuple[str, str, float, Path]] = []
    rescreen_rows: list[dict[str, Any]] = []
    for screen_index in range(1, screen_count + 1):
        iteration = f"rescreen_01_screen_{screen_index}"
        rescreen_dir = round_dir / "candidates" / iteration
        rescreen_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(parent_dir / "candidate.yaml", rescreen_dir / "candidate.yaml")
        measured = manager.measure(
            round_id=round_id,
            candidate_path=rescreen_dir / "candidate.yaml",
            profile="screen",
            parent_candidate_uuid=parent_uuid,
            harness=harness_mode,
        )
        trace = json.loads((rescreen_dir / "measurement_trace.json").read_text(encoding="utf-8"))
        measured_screen.append((str(measured["candidate_uuid"]), iteration, float(trace["eval_throughput"]), rescreen_dir))

    screen_measurements = [parent_value, *(value for _uuid, _iteration, value, _dir in measured_screen)]
    objective_mean = sum(screen_measurements) / len(screen_measurements)
    objective_ci_95 = 2.78 * manager._sample_stddev(screen_measurements) / (len(screen_measurements) ** 0.5)
    for candidate_uuid, iteration, objective_value, rescreen_dir in measured_screen:
        rows = manager._read_results(round_dir / "results.tsv")
        updated_rows: list[ResultsRow] = []
        for row in rows:
            if row.iteration == iteration and row.candidate_uuid == candidate_uuid:
                updated_rows.append(
                    ResultsRow(
                        candidate_uuid=row.candidate_uuid,
                        parent_candidate_uuid=parent_uuid,
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
                        notes="",
                    )
                )
            else:
                updated_rows.append(row)
        manager._write_results(round_dir / "results.tsv", updated_rows)
        commit_sha = _commit_replay_rescreen(
            manager=manager,
            round_id=round_id,
            round_dir=round_dir,
            rescreen_dir=rescreen_dir,
            row=next(row for row in updated_rows if row.iteration == iteration),
            parent_uuid=parent_uuid,
            harness_mode=harness_mode,
        )
        rescreen_rows.append(
            {
                "iteration": iteration,
                "profile": "screen",
                "candidate_uuid": candidate_uuid,
                "parent_candidate_uuid": parent_uuid,
                "eval_throughput": objective_value,
                "objective_mean_screen": objective_mean,
                "objective_ci_95_screen": objective_ci_95,
                "commit_sha": commit_sha,
            }
        )

    for full_index in range(1, full_count + 1):
        iteration = f"rescreen_01_full_{full_index}"
        rescreen_dir = round_dir / "candidates" / iteration
        rescreen_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(parent_dir / "candidate.yaml", rescreen_dir / "candidate.yaml")
        measured = manager.measure(
            round_id=round_id,
            candidate_path=rescreen_dir / "candidate.yaml",
            profile="full",
            parent_candidate_uuid=parent_uuid,
            harness=harness_mode,
        )
        trace = json.loads((rescreen_dir / "measurement_trace.json").read_text(encoding="utf-8"))
        full_value = float(trace["eval_throughput"])
        screen_stddev = manager._sample_stddev(screen_measurements)
        full_notes = ""
        if full_value < objective_mean - (3.0 * screen_stddev) or full_value > objective_mean + (3.0 * screen_stddev):
            full_notes = "screen_full_divergence"
        rows = manager._read_results(round_dir / "results.tsv")
        updated_rows = []
        for row in rows:
            if row.iteration == iteration and row.candidate_uuid == str(measured["candidate_uuid"]):
                updated_rows.append(
                    ResultsRow(
                        candidate_uuid=row.candidate_uuid,
                        parent_candidate_uuid=parent_uuid,
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
        manager._write_results(round_dir / "results.tsv", updated_rows)
        commit_sha = _commit_replay_rescreen(
            manager=manager,
            round_id=round_id,
            round_dir=round_dir,
            rescreen_dir=rescreen_dir,
            row=next(row for row in updated_rows if row.iteration == iteration),
            parent_uuid=parent_uuid,
            harness_mode=harness_mode,
        )
        rescreen_rows.append(
            {
                "iteration": iteration,
                "profile": "full",
                "candidate_uuid": str(measured["candidate_uuid"]),
                "parent_candidate_uuid": parent_uuid,
                "eval_throughput": full_value,
                "screen_full_divergence": bool(full_notes),
                "commit_sha": commit_sha,
            }
        )

    (round_dir / "rescreen_trace.json").write_text(json.dumps(rescreen_rows, indent=2), encoding="utf-8")
    return rescreen_rows


def _commit_replay_rescreen(
    *,
    manager: AutoResearchRoundManager,
    round_id: str,
    round_dir: Path,
    rescreen_dir: Path,
    row: ResultsRow,
    parent_uuid: str,
    harness_mode: str,
) -> str:
    round_spec = yaml.safe_load((round_dir / "round_spec.yaml").read_text(encoding="utf-8"))
    branch = str(round_spec["round_branch"])
    staged_paths = [
        path.relative_to(manager.repo_root)
        for path in [*manager._bootstrap_round_artifact_paths(round_dir), rescreen_dir, round_dir / "results.tsv"]
    ]
    message = manager._candidate_commit_message(
        round_id=round_id,
        iteration=row.iteration,
        row=row,
        trace_path=(rescreen_dir / "measurement_trace.json").relative_to(manager.repo_root),
        extra_trailers=[
            f"Rescreen-Of-UUID: {parent_uuid}",
            *(["Fixture-Mode: true"] if harness_mode == "synthetic" else []),
        ],
    )
    return manager._commit_paths(staged_paths, message, False, branch=branch, context="commit_refused")


def _source_commit_for(path: Path, *, cwd: Path) -> str:
    try:
        return _git(["log", "-n", "1", "--format=%H", "--", str(path)], cwd=cwd).strip()
    except subprocess.CalledProcessError:
        return ""


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


def _run_agent_main_loop(manager: AutoResearchRoundManager, ctx: RoundContext) -> None:
    if ctx.agent_runtime not in AGENT_RUNTIMES:
        raise RuntimeError(
            f"Invalid agent_runtime {ctx.agent_runtime!r}; expected one of {sorted(AGENT_RUNTIMES)}"
        )
    for index in range(1, int(ctx.iteration_cap or ctx.round_spec.get("iteration_cap", 12)) + 1):
        iteration = f"{index:03d}"
        if iteration in _finalized_iterations(ctx.round_dir):
            continue
        iteration_dir = ctx.round_dir / "candidates" / iteration
        iteration_dir.mkdir(parents=True, exist_ok=True)
        prompt = _iteration_prompt(ctx, iteration=iteration, next_iteration=f"{index + 1:03d}")
        transcript = iteration_dir / "agent_session.jsonl"
        last_message_path = iteration_dir / "agent_last_message.txt"
        argv, timeout_seconds = _agent_invocation(
            ctx,
            iteration_dir=iteration_dir,
            last_message_path=last_message_path,
        )
        with transcript.open("wb") as transcript_handle:
            result = subprocess.run(
                argv,
                input=prompt.encode(),
                stdout=transcript_handle,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                cwd=str(ctx.worktree) if ctx.agent_runtime == "claude" else None,
                timeout=timeout_seconds,
            )
        if ctx.agent_runtime == "claude":
            _extract_claude_last_message(transcript, last_message_path)
        if result.returncode == 2 and ctx.agent_runtime == "codex":
            raise RuntimeError(f"iteration {iteration} blocked")
        if result.returncode != 0:
            stderr_text = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(stderr_text or f"{ctx.agent_runtime} exited {result.returncode}")
        status = manager.status(round_id=ctx.round_id)
        if status["iterations_total"] >= index + 2:
            continue


# Backwards-compat alias for callers/tests that import the old name.
_run_codex_main_loop = _run_agent_main_loop


def _agent_invocation(
    ctx: RoundContext,
    *,
    iteration_dir: Path,
    last_message_path: Path,
) -> tuple[list[str], int]:
    if ctx.agent_runtime == "codex":
        argv = [
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
        timeout = int(ctx.round_spec.get("per_iteration_codex_wall_clock_s", 45 * 60))
        return argv, timeout
    if ctx.agent_runtime == "claude":
        model = str(ctx.round_spec.get("claude_model", DEFAULT_CLAUDE_MODEL))
        effort = str(ctx.round_spec.get("claude_effort", DEFAULT_CLAUDE_EFFORT))
        permission_mode = str(
            ctx.round_spec.get("claude_permission_mode", DEFAULT_CLAUDE_PERMISSION_MODE)
        )
        argv = [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--effort",
            effort,
            "--permission-mode",
            permission_mode,
            "--add-dir",
            str(ctx.worktree),
        ]
        timeout = int(
            ctx.round_spec.get(
                "per_iteration_claude_wall_clock_s",
                ctx.round_spec.get("per_iteration_codex_wall_clock_s", 45 * 60),
            )
        )
        return argv, timeout
    raise RuntimeError(f"Unsupported agent_runtime {ctx.agent_runtime!r}")


def _extract_claude_last_message(transcript: Path, last_message_path: Path) -> None:
    """Mirror codex's --output-last-message: write the final assistant text from the stream."""
    try:
        raw = transcript.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        last_message_path.write_text("", encoding="utf-8")
        return
    final_text = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result_text = event.get("result")
            if isinstance(result_text, str) and result_text:
                final_text = result_text
                continue
        if event.get("type") == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                            and block["text"]
                        ):
                            final_text = block["text"]
    last_message_path.write_text(final_text, encoding="utf-8")


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
        frozen_vllm_config = ctx.round_spec.get("frozen_vllm_config") or {}
        advisory_kv_budget = frozen_vllm_config.get("max_model_len", "<frozen max_model_len>")
        candidate_schema_instruction = (
            "Schema: parent HLD §5.3.3 L2 request_shaping keys only: "
            "concurrency_cap_eval, concurrency_cap_rollout, admission_queue_depth_max, "
            "per_request_kv_budget, priority_preemption. Vary only the three enforced fields: "
            "concurrency_cap_eval, concurrency_cap_rollout, admission_queue_depth_max. "
            f"Keep advisory fields fixed as metadata: per_request_kv_budget={advisory_kv_budget}, "
            "priority_preemption=off. No L0, no L1, no L3 keys. No extra keys. "
            "The lower-layer vllm_config is frozen from baseline_bundle_path."
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

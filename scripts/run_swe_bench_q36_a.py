#!/usr/bin/env python3
"""SWE-Bench per-instance orchestrator for Q36-A + Codex CLI 0.128.0.

Spec ref: docs/reports/auto_research/swe-bench-bounded-time-spec-20260520.md
          §9 (artifact layout), §11 (per-task protocol), §7 (concurrency).

For each instance from a pre-registered subset (built by
scripts/build_swe_bench_subset.py):
  1. Hydrate the workspace at the SWE-Bench base_commit, drop AGENTS.md
     with the problem_statement.
  2. Launch codex-runner:v1 Docker against the codex-bench proxy at
     :8022 with a wall budget.
  3. Diff workspace vs base_commit -> patch.diff (per-attempt artifact).
  4. Invoke codex-bench-eval-swe on the patch.
  5. Write per-task artifacts under
     output/swe_bench_q36_a_temp06/<dataset>/per_task/<instance_id>/.
  6. Aggregate predictions.jsonl + campaign_summary.json.

Defaults follow the spec:
  - Concurrency: 1 (LLD-05 §4.6 default; bump after Sprint-1 validation).
  - Codex wall budget: 25 min (spec §6) + 5 min eval buffer.
  - Proxy: http://127.0.0.1:8022/v1
  - Reasoning effort: high (carried over from launch_qwen36_ablation_point.py).
  - Temperature is governed by the vLLM relaunch bundle (Q36-A: temp=0.6).
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "swe_bench_q36_a_temp06"
DEFAULT_REPO_CACHE = REPO_ROOT / ".cache" / "swe_bench_repos"
DEFAULT_HF_HOME = REPO_ROOT / ".cache" / "huggingface"
DEFAULT_ENDPOINT = "http://127.0.0.1:8022/v1"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_AGENT_WALL_S = 25 * 60
DEFAULT_EVAL_TIMEOUT_S = 30 * 60
DEFAULT_MODEL_NAME_TAG = "qwen3.6-27b-fp8::codex-cli-0.128.0::q36-a"
# Same capture path used by launch_qwen36_ablation_point.py / Track B benches.
DEFAULT_PROXY_CAPTURE = Path("/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl")

CODEX_TEMPLATE = (
    "docker run --rm --name {container_name} --network=host -u 1000:1000 "
    "-v {workspace}:/workspace:rw "
    "-e OPENAI_API_KEY=EMPTY -e OPENAI_BASE_URL={endpoint} -e HOME=/tmp "
    "-w /workspace codex-runner:v1 "
    "codex exec --json --skip-git-repo-check "
    "--dangerously-bypass-approvals-and-sandbox -C /workspace "
    "-c 'model_provider=\"local-proxy\"' "
    "-c 'model_providers.local-proxy={{name=\"local-proxy\","
    "base_url=\"{endpoint}\",env_key=\"OPENAI_API_KEY\","
    "wire_api=\"responses\",stream_idle_timeout_ms=600000}}' "
    "-c 'model_reasoning_effort=\"high\"' "
    "-c 'model_supports_reasoning_summaries=true' "
    "-c 'model_reasoning_summary=\"auto\"' "
    "--model {model} "
    "\"Read the task prompt at /workspace/AGENTS.md and complete it in this workspace. "
    "Edit the source files directly to implement the fix. Do not write a diff file -- "
    "modify the files in place so that running pytest passes the tests described in the prompt.\""
)


def _iso_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_subset(subset_json: Path) -> tuple[str, list[str]]:
    payload = json.loads(subset_json.read_text())
    dataset_name = payload["dataset_name"]
    instance_ids = list(payload["instance_ids"])
    return dataset_name, instance_ids


def _load_dataset(dataset_name: str) -> dict[str, dict]:
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="test")
    out: dict[str, dict] = {}
    for ex in ds:
        out[ex["instance_id"]] = dict(ex)
    return out


def _repo_clone_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def _ensure_repo_cache(repo: str, cache_root: Path) -> Path:
    safe = repo.replace("/", "__")
    cache_path = cache_root / safe
    if not cache_path.is_dir():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", _repo_clone_url(repo), str(cache_path)],
            check=True,
        )
    return cache_path


def _fetch_commit(cache_path: Path, base_commit: str) -> None:
    rc = subprocess.run(
        ["git", "-C", str(cache_path), "cat-file", "-e", base_commit],
    ).returncode
    if rc != 0:
        subprocess.run(
            ["git", "-C", str(cache_path), "fetch", "origin", base_commit],
            check=False,
        )


def _hydrate_workspace(
    *,
    cache_path: Path,
    base_commit: str,
    workspace_path: Path,
) -> None:
    if workspace_path.exists():
        shutil.rmtree(workspace_path)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    # Absolutize: `git -C <cache> worktree add <relpath>` resolves the
    # destination against <cache>, not against the script CWD.
    abs_workspace = workspace_path.resolve()
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "add", "--detach",
         str(abs_workspace), base_commit],
        check=True,
    )


def _remove_workspace(cache_path: Path, workspace_path: Path) -> None:
    abs_workspace = workspace_path.resolve() if workspace_path.exists() else workspace_path
    if not abs_workspace.exists():
        return
    subprocess.run(
        ["git", "-C", str(cache_path), "worktree", "remove", "--force",
         str(abs_workspace)],
        check=False,
    )
    if abs_workspace.exists():
        shutil.rmtree(abs_workspace, ignore_errors=True)


def _write_agents_md(workspace: Path, instance: dict) -> None:
    body = []
    body.append(f"# SWE-Bench task: {instance['instance_id']}")
    body.append("")
    body.append(f"**Repo:** `{instance['repo']}`  ")
    body.append(f"**Base commit:** `{instance['base_commit']}`  ")
    if instance.get("version"):
        body.append(f"**Version:** `{instance['version']}`  ")
    body.append("")
    body.append("## Problem statement")
    body.append("")
    body.append(instance.get("problem_statement") or "(empty problem statement)")
    body.append("")
    body.append("## Required behavior")
    body.append("")
    body.append(
        "Implement the fix described in the problem statement by editing the "
        "source files in this workspace. Do NOT modify any test files. The "
        "hidden grader will apply its own test patch and run the test suite; "
        "your code must make those tests pass without breaking existing ones."
    )
    body.append("")
    (workspace / "AGENTS.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def _extract_patch(cache_path: Path, workspace: Path, base_commit: str) -> str:
    # Stage tracked-file diffs and untracked file additions against base_commit.
    proc = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--no-color", "--binary", base_commit],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


def _run_codex(
    *,
    workspace: Path,
    endpoint: str,
    model: str,
    timeout_s: int,
    instance_id: str,
    stdout_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    container_name = f"swe-codex-{instance_id.replace('/', '_')[:48]}-{int(time.time())}"
    cmd = CODEX_TEMPLATE.format(
        container_name=container_name,
        workspace=str(workspace),
        endpoint=endpoint,
        model=model,
    )
    started = time.monotonic()
    rc: int | None = None
    timed_out = False
    # Send stdout/stderr straight to files so subprocess.run can enforce
    # the wallclock with timeout=... and not block on a pipe. The trace
    # is the agent's JSON event stream; stdout file gets stderr noise.
    with stdout_path.open("w", encoding="utf-8") as stdout_f, \
         trace_path.open("w", encoding="utf-8") as trace_f:
        try:
            completed = subprocess.run(
                shlex.split(cmd),
                stdout=trace_f,
                stderr=stdout_f,
                timeout=max(timeout_s, 30),
                check=False,
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            # Best-effort container stop (kills the cgroup; codex CLI inside
            # the container goes with it).
            subprocess.run(
                ["docker", "kill", container_name], check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            # Reap the docker client; --rm cleans up the container.
            try:
                subprocess.run(
                    ["docker", "wait", container_name], check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except subprocess.TimeoutExpired:
                pass
            rc = -1
    elapsed = time.monotonic() - started
    return {
        "elapsed_s": round(elapsed, 3),
        "exit_code": rc if rc is not None else -1,
        "timed_out": timed_out,
        "container_name": container_name,
    }


def _run_eval(
    *,
    instance_id: str,
    patch_path: Path,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
    timeout_s: int,
    eval_log_path: Path,
) -> dict[str, Any]:
    cbe_exe = shutil.which("codex-bench-eval-swe")
    if cbe_exe is None:
        cbe_exe = str(REPO_ROOT / ".venv" / "bin" / "codex-bench-eval-swe")
    cmd = [
        cbe_exe,
        "--instance-id", instance_id,
        "--patch-path", str(patch_path),
        "--output-dir", str(output_dir),
        "--dataset-name", dataset_name,
        "--model-name", model_name,
        "--timeout-s", str(timeout_s),
        "--cache-level", "env",
    ]
    started = time.monotonic()
    with eval_log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    return {
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 3),
    }


def _process_one(
    *,
    instance_id: str,
    instance: dict,
    dataset_name: str,
    per_task_root: Path,
    repo_cache_root: Path,
    endpoint: str,
    model: str,
    model_name: str,
    agent_wall_s: int,
    eval_timeout_s: int,
    skip_existing: bool,
) -> dict[str, Any]:
    # Use absolute paths everywhere so docker volume mounts and git
    # worktree add (which resolves relative to -C cache) both work.
    task_dir = (per_task_root / instance_id).resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    runner_meta_path = task_dir / "runner_metadata.json"
    if skip_existing and runner_meta_path.is_file():
        return {"instance_id": instance_id, "status": "skipped_existing"}

    workspace_path = task_dir / "workspace"
    patch_path = task_dir / "patch.diff"
    codex_stdout = task_dir / "codex_stdout.log"
    codex_trace = task_dir / "codex_trace.jsonl"
    eval_log = task_dir / "eval_invocation.log"
    eval_output = task_dir / "eval"
    eval_output.mkdir(parents=True, exist_ok=True)

    started_iso = _iso_now()
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "dataset_name": dataset_name,
        "started_at": started_iso,
        "repo": instance.get("repo"),
        "base_commit": instance.get("base_commit"),
    }

    cache_path = None
    try:
        cache_path = _ensure_repo_cache(instance["repo"], repo_cache_root)
        _fetch_commit(cache_path, instance["base_commit"])
        _hydrate_workspace(
            cache_path=cache_path,
            base_commit=instance["base_commit"],
            workspace_path=workspace_path,
        )
        _write_agents_md(workspace_path, instance)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "hydration_failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        runner_meta_path.write_text(json.dumps(summary, indent=2))
        if cache_path is not None:
            _remove_workspace(cache_path, workspace_path)
        return summary

    # Snapshot proxy capture byte offset before invoking Codex so we can
    # slice this task's vllm_request_metrics.jsonl after the run.
    proxy_capture = DEFAULT_PROXY_CAPTURE
    proxy_offset_before = (
        proxy_capture.stat().st_size if proxy_capture.is_file() else 0
    )

    codex_meta = _run_codex(
        workspace=workspace_path,
        endpoint=endpoint,
        model=model,
        timeout_s=agent_wall_s,
        instance_id=instance_id,
        stdout_path=codex_stdout,
        trace_path=codex_trace,
    )
    summary["codex"] = codex_meta

    # Slice the new proxy rows into a per-task file (matches Track B layout).
    per_task_metrics = task_dir / "vllm_request_metrics.jsonl"
    try:
        if proxy_capture.is_file():
            with proxy_capture.open("rb") as src:
                src.seek(proxy_offset_before)
                payload = src.read()
            per_task_metrics.write_bytes(payload)
            summary["vllm_request_metrics_bytes"] = len(payload)
        else:
            per_task_metrics.write_bytes(b"")
            summary["vllm_request_metrics_bytes"] = 0
            summary["vllm_request_metrics_warning"] = (
                "proxy capture file not present; verbose request metrics unavailable"
            )
    except Exception as exc:  # noqa: BLE001
        summary["vllm_request_metrics_error"] = f"{type(exc).__name__}: {exc}"

    patch_text = ""
    try:
        patch_text = _extract_patch(cache_path, workspace_path, instance["base_commit"])
    except Exception as exc:  # noqa: BLE001
        summary["patch_extract_error"] = f"{type(exc).__name__}: {exc}"
    patch_path.write_text(patch_text, encoding="utf-8")
    summary["patch_bytes"] = len(patch_text)

    # Always run the evaluator -- the CLI handles empty-patch -> exit 1.
    eval_meta = _run_eval(
        instance_id=instance_id,
        patch_path=patch_path,
        output_dir=eval_output,
        dataset_name=dataset_name,
        model_name=model_name,
        timeout_s=eval_timeout_s,
        eval_log_path=eval_log,
    )
    summary["eval"] = eval_meta

    eval_report_path = eval_output / "eval_report.json"
    if eval_report_path.is_file():
        try:
            summary["eval_report"] = json.loads(eval_report_path.read_text())
        except Exception:  # noqa: BLE001
            pass

    # Tear down the worktree to free disk; preserve patch and artifacts.
    _remove_workspace(cache_path, workspace_path)

    summary["ended_at"] = _iso_now()
    runner_meta_path.write_text(json.dumps(summary, indent=2))
    return summary


def _aggregate(per_task_root: Path, summary_path: Path, predictions_path: Path,
               started_at: str, ended_at: str, model_name: str) -> dict[str, Any]:
    instance_summaries: list[dict[str, Any]] = []
    verdict_counter: Counter = Counter()
    failure_mode_counter: Counter = Counter()
    repo_counter: Counter = Counter()
    repo_pass_counter: Counter = Counter()
    eval_wall: list[float] = []
    codex_wall: list[float] = []
    predictions_lines: list[str] = []
    for task_dir in sorted(p for p in per_task_root.iterdir() if p.is_dir()):
        meta_path = task_dir / "runner_metadata.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text())
        instance_summaries.append(meta)
        verdict = (meta.get("eval_report") or {}).get("verdict", "missing")
        failure_mode = (meta.get("eval_report") or {}).get("failure_mode", "missing")
        verdict_counter[verdict] += 1
        failure_mode_counter[failure_mode] += 1
        repo = meta.get("repo") or "unknown"
        repo_counter[repo] += 1
        if verdict == "resolved":
            repo_pass_counter[repo] += 1
        if (meta.get("eval_report") or {}).get("eval_wall_clock_seconds") is not None:
            eval_wall.append(float(meta["eval_report"]["eval_wall_clock_seconds"]))
        if (meta.get("codex") or {}).get("elapsed_s") is not None:
            codex_wall.append(float(meta["codex"]["elapsed_s"]))
        pred_file = task_dir / "eval" / "predictions.jsonl"
        if pred_file.is_file():
            predictions_lines.extend(
                line for line in pred_file.read_text().splitlines() if line.strip()
            )

    def _percentiles(xs: list[float]) -> dict[str, float]:
        if not xs:
            return {}
        xs = sorted(xs)
        def _pct(p: float) -> float:
            i = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
            return round(xs[i], 3)
        return {"p50": _pct(0.5), "p90": _pct(0.9), "p99": _pct(0.99),
                "min": round(min(xs), 3), "max": round(max(xs), 3)}

    summary = {
        "model_name_or_path": model_name,
        "started_at": started_at,
        "ended_at": ended_at,
        "instances_total": len(instance_summaries),
        "verdict_counts": dict(verdict_counter),
        "failure_mode_counts": dict(failure_mode_counter),
        "per_repo_total": dict(repo_counter),
        "per_repo_resolved": dict(repo_pass_counter),
        "eval_wall_seconds": _percentiles(eval_wall),
        "codex_wall_seconds": _percentiles(codex_wall),
        "resolved_rate": (
            round(verdict_counter["resolved"] / len(instance_summaries), 4)
            if instance_summaries else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    predictions_path.write_text("\n".join(predictions_lines) + ("\n" if predictions_lines else ""))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True,
                        help="JSON subset emitted by build_swe_bench_subset.py")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dataset-tag", default=None,
                        help="Override the per-dataset subdirectory name "
                             "(default: 'verified' or 'pro' inferred from subset).")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME_TAG)
    parser.add_argument("--agent-wall-s", type=int, default=DEFAULT_AGENT_WALL_S)
    parser.add_argument("--eval-timeout-s", type=int, default=DEFAULT_EVAL_TIMEOUT_S)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Agent concurrency. LLD-05 §4.6 default is 1; raise after Sprint-1 validation.")
    parser.add_argument("--repo-cache", type=Path, default=DEFAULT_REPO_CACHE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N instances (for smoke runs).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip instances whose runner_metadata.json is already on disk.")
    args = parser.parse_args(argv)

    dataset_name, instance_ids = _load_subset(args.subset)
    if args.limit is not None:
        instance_ids = instance_ids[: args.limit]
    dataset_tag = args.dataset_tag or ("pro" if "Pro" in dataset_name else "verified")
    dataset_out = args.out_root / dataset_tag
    per_task_root = dataset_out / "per_task"
    per_task_root.mkdir(parents=True, exist_ok=True)

    print(f"=== [{_iso_now()}] dataset={dataset_name} tag={dataset_tag} n={len(instance_ids)} "
          f"concurrency={args.concurrency} ===", flush=True)
    dataset_records = _load_dataset(dataset_name)
    missing = [i for i in instance_ids if i not in dataset_records]
    if missing:
        print(f"WARNING: {len(missing)} subset instances missing from dataset: {missing[:5]}",
              flush=True)
        instance_ids = [i for i in instance_ids if i in dataset_records]

    args.repo_cache.mkdir(parents=True, exist_ok=True)

    started_at = _iso_now()
    summaries: list[dict[str, Any]] = []

    def _job(iid: str) -> dict[str, Any]:
        t0 = time.time()
        print(f"[{_iso_now()}] -> {iid}", flush=True)
        try:
            res = _process_one(
                instance_id=iid,
                instance=dataset_records[iid],
                dataset_name=dataset_name,
                per_task_root=per_task_root,
                repo_cache_root=args.repo_cache,
                endpoint=args.endpoint,
                model=args.model,
                model_name=args.model_name,
                agent_wall_s=args.agent_wall_s,
                eval_timeout_s=args.eval_timeout_s,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:  # noqa: BLE001
            res = {"instance_id": iid, "status": "orchestrator_crash",
                   "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        verdict = (res.get("eval_report") or {}).get("verdict", res.get("status", "?"))
        print(f"[{_iso_now()}] <- {iid} verdict={verdict} elapsed_total={time.time()-t0:.1f}s",
              flush=True)
        return res

    if args.concurrency <= 1:
        for iid in instance_ids:
            summaries.append(_job(iid))
    else:
        with _cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for res in ex.map(_job, instance_ids):
                summaries.append(res)

    ended_at = _iso_now()
    summary = _aggregate(
        per_task_root=per_task_root,
        summary_path=dataset_out / "campaign_summary.json",
        predictions_path=dataset_out / "predictions.jsonl",
        started_at=started_at,
        ended_at=ended_at,
        model_name=args.model_name,
    )
    print(f"=== [{ended_at}] DONE n={summary['instances_total']} "
          f"resolved_rate={summary.get('resolved_rate')} "
          f"verdicts={summary['verdict_counts']} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

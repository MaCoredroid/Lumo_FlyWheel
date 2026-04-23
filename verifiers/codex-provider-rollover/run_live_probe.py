#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FAMILY = "codex-provider-rollover"
FAMILY_ROOT = REPO / "benchmark_blueprints" / "families" / FAMILY
WORKSPACE_BUNDLE = FAMILY_ROOT / "workspace_bundle"
VERIFIER_DATA = REPO / "verifier_data" / FAMILY
SCORER = REPO / "verifiers" / FAMILY / "score_provider_rollover.py"
REPORT_ROOT = FAMILY_ROOT / "report"
DEFAULT_VARIANTS = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

PROMPT = (
    "Read AGENTS.md in this directory and solve the task completely in this workspace only. "
    "Repair the maintenance profile, strengthen the smoke so it verifies exact "
    "previous_response_id follow-up continuity, preserve the local tuning block, "
    "align the docs, and run bin/run-visible-tests before finishing. "
    "Do not modify read-only files."
)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def overlay_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", default="attempt_02")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--probe-root", default="/tmp/codex_provider_rollover_probe")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    args = parser.parse_args()

    probe_run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    attempt_root = REPORT_ROOT / args.attempt
    runs_jsonl = attempt_root / "probe_runs.jsonl"
    logs_root = attempt_root / "probe_run_logs"
    ensure_clean_dir(attempt_root)
    logs_root.mkdir(parents=True, exist_ok=True)
    Path(args.probe_root).mkdir(parents=True, exist_ok=True)

    print(
        f"probe_run_id={probe_run_id} family={FAMILY} n={args.n} "
        f"model={args.model} reasoning_effort={args.reasoning_effort}"
    )
    print(f"attempt_root={attempt_root}")
    print(f"probe_root={args.probe_root}")

    for variant in args.variants:
        src = WORKSPACE_BUNDLE / variant
        if not src.exists():
            raise SystemExit(f"missing workspace bundle: {src}")
        for run_index in range(1, args.n + 1):
            run_tag = f"{probe_run_id}-{variant}-run{run_index:02d}"
            tmp_root = Path(args.probe_root) / run_tag
            workspace = tmp_root / "workspace"
            artifact_root = attempt_root / variant / f"run_{run_index:02d}"
            ensure_clean_dir(tmp_root)
            ensure_clean_dir(artifact_root)
            shutil.copytree(src, workspace)

            codex_log = logs_root / f"{run_tag}.log"
            codex_cmd = [
                "timeout",
                str(args.timeout),
                "codex",
                "exec",
                "-m",
                args.model,
                "-c",
                f'reasoning_effort="{args.reasoning_effort}"',
                "-c",
                f'model_reasoning_effort="{args.reasoning_effort}"',
                "--cd",
                str(workspace),
                "--skip-git-repo-check",
                "--sandbox",
                "workspace-write",
                "--color",
                "never",
                "--ephemeral",
                PROMPT,
            ]
            print(f"=== {variant} run {run_index}/{args.n} ({run_tag}) ===")
            start = time.time()
            codex_env = os.environ.copy()
            codex_env["PYTHONDONTWRITEBYTECODE"] = "1"
            codex_env["PYTEST_ADDOPTS"] = "-p no:cacheprovider"
            proc = run_cmd(codex_cmd, env=codex_env)
            codex_seconds = round(time.time() - start, 2)
            codex_log.write_text(proc.stdout)
            print(f"  codex exit={proc.returncode} seconds={codex_seconds} log={codex_log}")

            result_file = artifact_root / "verify_result.json"
            score_env = os.environ.copy()
            score_env.update(
                {
                    "AGENT_WS": str(workspace),
                    "VERIFIER_DATA": str(VERIFIER_DATA),
                    "VARIANT_ID": variant,
                    "RESULT_FILE": str(result_file),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTEST_ADDOPTS": "-p no:cacheprovider",
                }
            )
            score_proc = run_cmd([sys.executable, str(SCORER)], env=score_env)
            if score_proc.returncode != 0:
                raise SystemExit(
                    f"scorer failed for {variant} run {run_index}: {score_proc.stdout}"
                )

            diff_proc = run_cmd(["git", "diff", "--no-index", "--", str(src), str(workspace)])
            (artifact_root / "workspace.diff").write_text(diff_proc.stdout)
            shutil.copy2(codex_log, artifact_root / "codex.log")
            overlay_tree(workspace, artifact_root / "workspace")

            result = load_json(result_file)
            record = {
                "probe_run_id": probe_run_id,
                "attempt": args.attempt,
                "variant": variant,
                "run_index": run_index,
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "codex_exit": proc.returncode,
                "codex_seconds": codex_seconds,
                "workspace_path": str((artifact_root / "workspace").resolve()),
                "score": int(result.get("score", 0)),
                "P_benchmark": int(result.get("P_benchmark", 0)),
                "M_training": float(result.get("M_training", 0.0)),
                "pass": bool(result.get("pass", False)),
                "shortcut_detected": bool(result.get("shortcut_detected", False)),
                "ceilings_applied": list(result.get("ceilings_applied", [])),
                "integrity_flag": int(result.get("integrity_flag", 0)),
                "milestones": dict(result.get("milestones", {})),
                "breakdown": dict(result.get("breakdown", {})),
                "errors": list(result.get("errors", [])),
                "result_path": str(result_file.resolve()),
                "codex_log_path": str((artifact_root / "codex.log").resolve()),
                "diff_path": str((artifact_root / "workspace.diff").resolve()),
            }
            with runs_jsonl.open("a") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"  score={record['score']} pass={record['pass']} "
                f"ceilings={record['ceilings_applied']}"
            )

    meta = {
        "probe_run_id": probe_run_id,
        "attempt": args.attempt,
        "family": FAMILY,
        "n": args.n,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "variants": args.variants,
        "runs_jsonl": str(runs_jsonl.resolve()),
    }
    (attempt_root / "probe_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"done. runs_jsonl={runs_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

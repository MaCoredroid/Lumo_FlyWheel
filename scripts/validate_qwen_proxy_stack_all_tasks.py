#!/usr/bin/env python3
"""Validate the §13-17 proxy patch stack across all 13 v4a tasks.

For each task in the v4a corpus:
 1. Copy the workspace_bundle/v1-clean-baseline → fresh temp dir
 2. Build prompt.md from AGENTS.md + .scenario_variant (matches v4a runner)
 3. Run codex exec through the bench proxy (assumes
    LUMO_PROXY_NONSTREAM_BYPASS=1 and LUMO_PROXY_AUTO_CONTINUE=1)
 4. Capture: tool call count, apply_patch keyword count, agent-messages
    with text, workspace files written, output token total
 5. Emit a summary JSON + a printed table

Acceptance signal per task: at least one workspace file written
(loosened from "apply_patch" per §10.7).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Excluded: plugin-scaffold-alignment + skill-router-contract-upgrade lack
# AGENTS.md in their v1-clean-baseline workspace_bundle (verified
# 2026-05-12). The Track B prompt builder falls back to task_spec.md but
# that yields a less directive prompt, and qwen3.5-27b doesn't produce
# real artifacts on this fallback. Treat these as benchmark-fixture
# defects, not model/harness failures. Re-include only once their
# workspace_bundles get a proper AGENTS.md added.
EXCLUDED_TASKS_NO_AGENTS_MD = {
    "plugin-scaffold-alignment",
    "skill-router-contract-upgrade",
}
TASKS = [
    "dead-flag-reachability-audit",
    "fanout-fullstack-release-blocker",
    "incident-evidence-synthesis",
    "multi-tool-transaction-repair",
    "policy-aware-request-resolution",
    "release-note-to-plan-translation",
    "responses-sdk-adapter-cutover",
    "responsive-checkout-visual-regression",
    "security-audit-hotfix-remediation",
    "sqlalchemy-2-session-modernization",
    "transcript-merge-regression",
]
# Verify the exclusion list matches the assertion in EXCLUDED_TASKS_NO_AGENTS_MD.
_ALL_V4A = TASKS + sorted(EXCLUDED_TASKS_NO_AGENTS_MD)
assert sorted(_ALL_V4A) == sorted(set(_ALL_V4A)), "duplicate task in TASKS+EXCLUDED"
VARIANT = "v1-clean-baseline"


def build_prompt(workspace: Path) -> str:
    pieces: list[str] = []
    for rel in ["AGENTS.md", ".scenario_variant"]:
        path = workspace / rel
        if path.is_file():
            pieces.append(
                f"## {rel}\n\n{path.read_text(encoding='utf-8', errors='replace').strip()}\n"
            )
    if not pieces:
        # Fallback: try task_spec.md
        spec_path = REPO_ROOT / "benchmark_blueprints" / "families" / workspace.parent.parent.name / "task_spec.md"
        if spec_path.is_file():
            pieces.append(f"## task_spec.md\n\n{spec_path.read_text(encoding='utf-8', errors='replace').strip()}\n")
    return "\n".join(pieces).strip() + "\n"


def parse_codex_stdout(path: Path) -> dict:
    turns = 0
    tool_calls = 0
    apply_patch = 0
    msgs_with_text = 0
    cmds: list[str] = []
    total_in = 0
    total_out = 0
    if not path.is_file():
        return {
            "turns": 0,
            "tool_calls": 0,
            "apply_patch": 0,
            "msgs_with_text": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "commands": [],
        }
    with path.open() as fh:
        for line in fh:
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = evt.get("type")
            if t == "turn.started":
                turns += 1
            elif t == "turn.completed":
                u = evt.get("usage") or {}
                total_in += u.get("input_tokens", 0)
                total_out += u.get("output_tokens", 0)
            elif t == "item.completed":
                item = evt.get("item", {}) or {}
                it = item.get("type")
                if it == "command_execution":
                    tool_calls += 1
                    cmd = item.get("command", "")
                    if "apply_patch" in cmd:
                        apply_patch += 1
                    cmds.append(cmd[:120])
                elif it == "agent_message":
                    if (item.get("text") or "").strip():
                        msgs_with_text += 1
    return {
        "turns": turns,
        "tool_calls": tool_calls,
        "apply_patch": apply_patch,
        "msgs_with_text": msgs_with_text,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "commands": cmds,
    }


def run_task(task: str, out_root: Path, endpoint: str, model: str, timeout_s: int) -> dict:
    src = REPO_ROOT / "benchmark_blueprints" / "families" / task / "workspace_bundle" / VARIANT
    if not src.is_dir():
        return {"task": task, "status": "MISSING_BUNDLE"}
    task_dir = out_root / task
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    workspace = task_dir / "workspace"
    shutil.copytree(src, workspace)
    prompt = build_prompt(workspace)
    prompt_path = task_dir / "prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    user_msg = f"Read the task prompt at {prompt_path} and complete it in this workspace."
    cmd = [
        "codex", "exec", "--json", "--skip-git-repo-check",
        "-C", str(workspace),
        "-c", 'model_provider="local-proxy"',
        "-c", f'model_providers.local-proxy={{name="local-proxy",base_url="{endpoint}",env_key="OPENAI_API_KEY",wire_api="responses",stream_idle_timeout_ms=600000}}',
        "-c", 'model_reasoning_effort="high"',
        "-c", 'model_supports_reasoning_summaries=true',
        "-c", 'model_reasoning_summary="auto"',
        "--model", model,
        user_msg,
    ]
    stdout_path = task_dir / "codex_stdout.log"
    stderr_path = task_dir / "codex_stderr.log"
    env = {
        "OPENAI_API_KEY": "EMPTY",
        "OPENAI_BASE_URL": endpoint,
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    # inherit HOME etc.
    import os as _os
    full_env = {**_os.environ, **env}
    t0 = time.time()
    try:
        with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
            rc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=so,
                stderr=se,
                env=full_env,
                timeout=timeout_s,
            ).returncode
    except subprocess.TimeoutExpired:
        rc = 124
    elapsed = time.time() - t0
    # Workspace mutations
    new_files = []
    try:
        for f in workspace.rglob("*"):
            if not f.is_file():
                continue
            try:
                if f.stat().st_mtime > prompt_path.stat().st_mtime:
                    new_files.append(str(f.relative_to(workspace)))
            except OSError:
                pass
    except Exception:
        pass
    parsed = parse_codex_stdout(stdout_path)
    return {
        "task": task,
        "rc": rc,
        "elapsed_s": round(elapsed, 1),
        "tool_calls": parsed["tool_calls"],
        "apply_patch": parsed["apply_patch"],
        "msgs_with_text": parsed["msgs_with_text"],
        "input_tokens": parsed["input_tokens"],
        "output_tokens": parsed["output_tokens"],
        "files_written": len(new_files),
        "new_files": new_files[:12],
        "status": "WROTE_FILES" if new_files else ("NO_FILES_NO_RC" if rc == 0 else f"RC={rc}_NO_FILES"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="/tmp/qwen_proxy_validation")
    p.add_argument("--endpoint", default="http://127.0.0.1:8022/v1")
    p.add_argument("--model", default="qwen3.5-27b")
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--only", default=None, help="comma-separated subset of task names")
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = TASKS if not args.only else [t for t in TASKS if t in args.only.split(",")]
    results: list[dict] = []
    print(f"Running {len(tasks)} tasks ...", flush=True)
    for i, task in enumerate(tasks, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(tasks)}] {task} starting at {time.strftime('%H:%M:%S')}", flush=True)
        result = run_task(task, out_root, args.endpoint, args.model, args.timeout_s)
        results.append(result)
        elapsed = time.time() - t0
        print(
            f"  -> tools={result.get('tool_calls')} apply_patch={result.get('apply_patch')} "
            f"msgs={result.get('msgs_with_text')} files={result.get('files_written')} "
            f"out_tok={result.get('output_tokens')} rc={result.get('rc')} "
            f"status={result.get('status')} ({elapsed:.0f}s)",
            flush=True,
        )

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps({"tasks": results}, indent=2) + "\n", encoding="utf-8")
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"{'task':<42} {'tools':>5} {'patch':>5} {'msgs':>4} {'files':>5} {'out_tok':>7} {'status':<20}")
    for r in results:
        print(
            f"{r['task']:<42} {r.get('tool_calls',0):>5} {r.get('apply_patch',0):>5} "
            f"{r.get('msgs_with_text',0):>4} {r.get('files_written',0):>5} "
            f"{r.get('output_tokens',0):>7} {r.get('status','?'):<20}"
        )
    wins = sum(1 for r in results if r.get("files_written", 0) > 0)
    print(f"\nTasks that wrote >=1 file: {wins} / {len(results)}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

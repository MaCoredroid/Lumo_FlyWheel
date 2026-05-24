#!/usr/bin/env python3
"""CNB-55 v4a active-set runner for codex-on-x86 (alienware).

Mirrors scripts/validate_qwen_proxy_stack_all_tasks.py but drives codex via the
codex-runner:v1 docker image (--network=host) so it can run on alienware while
inference stays on the DGX vLLM, reached through the reverse SSH tunnel
(alienware:8022 -> DGX proxy:8022 -> vLLM:9950). Strictly serial (B=1).

Per family: copy workspace_bundle/v1-clean-baseline -> <out-root>/<task>/workspace,
build prompt.md from AGENTS.md (+ .scenario_variant), run codex exec, capture
stdout/stderr + runner_metadata.json (started/ended for validity), parse metrics.
Grading is separate (run_v4a_graders_on_validation.py on the DGX afterward).
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time, datetime as dt
from pathlib import Path

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
VARIANT = "v1-clean-baseline"


def build_prompt(workspace: Path, families_root: Path, task: str) -> str:
    pieces: list[str] = []
    for rel in ["AGENTS.md", ".scenario_variant"]:
        p = workspace / rel
        if p.is_file():
            pieces.append(f"## {rel}\n\n{p.read_text(encoding='utf-8', errors='replace').strip()}\n")
    if not pieces:
        spec = families_root / task / "task_spec.md"
        if spec.is_file():
            pieces.append(f"## task_spec.md\n\n{spec.read_text(encoding='utf-8', errors='replace').strip()}\n")
    return "\n".join(pieces).strip() + "\n"


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()


def run_task(task: str, families_root: Path, out_root: Path, endpoint: str, model: str, timeout_s: int) -> dict:
    src = families_root / task / "workspace_bundle" / VARIANT
    if not src.is_dir():
        return {"task": task, "status": "MISSING_BUNDLE"}
    task_dir = out_root / task
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)
    workspace = (task_dir / "workspace").resolve()
    shutil.copytree(src, workspace)
    prompt = build_prompt(workspace, families_root, task)
    (task_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    # prompt.md also inside workspace so it is visible at /workspace/prompt.md
    (workspace / "prompt.md").write_text(prompt, encoding="utf-8")

    name = f"cnb-{task}-{int(time.time())}"
    cmd = [
        "docker", "run", "--rm", "--name", name, "--network=host", "-u", "1000:1000",
        "-v", f"{workspace}:/workspace:rw",
        "-e", "OPENAI_API_KEY=EMPTY", "-e", f"OPENAI_BASE_URL={endpoint}", "-e", "HOME=/tmp",
        "-w", "/workspace", "codex-runner:v1",
        "codex", "exec", "--json", "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox", "-C", "/workspace",
        "-c", 'model_provider="local-proxy"',
        "-c", f'model_providers.local-proxy={{name="local-proxy",base_url="{endpoint}",env_key="OPENAI_API_KEY",wire_api="responses",stream_idle_timeout_ms=600000}}',
        "-c", 'model_reasoning_effort="high"',
        "-c", 'model_supports_reasoning_summaries=true',
        "-c", 'model_reasoning_summary="auto"',
        "--model", model,
        "Read the task prompt at /workspace/prompt.md and complete it in this workspace. Edit files directly to implement it.",
    ]
    stdout_path = task_dir / "codex_stdout.log"
    stderr_path = task_dir / "codex_stderr.log"
    started = time.time()
    try:
        with stdout_path.open("wb") as so, stderr_path.open("wb") as se:
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=so, stderr=se, timeout=timeout_s).returncode
    except subprocess.TimeoutExpired:
        rc = 124
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    ended = time.time()
    elapsed = ended - started
    (task_dir / "runner_metadata.json").write_text(
        json.dumps({"task": task, "started_at": iso(started), "ended_at": iso(ended),
                    "elapsed_s": round(elapsed, 1), "rc": rc, "model": model, "endpoint": endpoint,
                    "variant": VARIANT}, indent=2) + "\n", encoding="utf-8")
    new_files = []
    pm = (task_dir / "prompt.md").stat().st_mtime
    try:
        for f in workspace.rglob("*"):
            if f.is_file():
                try:
                    if f.stat().st_mtime > pm and f.name != "prompt.md":
                        new_files.append(str(f.relative_to(workspace)))
                except OSError:
                    pass
    except Exception:
        pass
    return {"task": task, "rc": rc, "elapsed_s": round(elapsed, 1),
            "files_written": len(new_files), "new_files": new_files[:12],
            "status": "WROTE_FILES" if new_files else (f"RC={rc}_NO_FILES")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families-root", required=True, help="dir holding <task>/workspace_bundle/v1-clean-baseline")
    ap.add_argument("--out-root", default=os.path.expanduser("~/cnb_v4a/output/run1"))
    ap.add_argument("--endpoint", default="http://127.0.0.1:8022/v1")
    ap.add_argument("--model", default="qwen3.6-27b")
    ap.add_argument("--timeout-s", type=int, default=1800)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    families_root = Path(args.families_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if not args.only else [t for t in TASKS if t in args.only.split(",")]
    log = out_root / "driver.log"
    def dlog(m):
        with log.open("a") as fh: fh.write(f"[cnb] {dt.datetime.now(dt.timezone.utc).isoformat()} {m}\n")
    dlog(f"START cnb-v4a config=D B=1 tasks={len(tasks)} model={args.model} endpoint={args.endpoint}")
    results = []
    for i, task in enumerate(tasks, 1):
        dlog(f"[{i}/{len(tasks)}] {task} START")
        r = run_task(task, families_root, out_root, args.endpoint, args.model, args.timeout_s)
        results.append(r)
        dlog(f"[{i}/{len(tasks)}] {task} END rc={r.get('rc')} files={r.get('files_written')} elapsed={r.get('elapsed_s')} status={r.get('status')}")
    (out_root / "summary.json").write_text(json.dumps({"tasks": results}, indent=2) + "\n", encoding="utf-8")
    dlog(f"ALL CNB-V4A DONE tasks={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

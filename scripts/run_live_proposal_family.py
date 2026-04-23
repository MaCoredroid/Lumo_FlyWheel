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
from urllib.parse import urlparse

import requests


DEFAULT_PROMPT = (
    "Read AGENTS.md in this directory and follow it exactly. "
    "Author brief_input.json at the workspace root and run "
    "./bin/cnb55-brief submit brief_input.json to produce brief/manager_brief.json. "
    "Do not hand-write brief/manager_brief.md. Do not modify files outside brief/."
)


def _run_output_dir(repo_root: Path, family: str, variant: str) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = repo_root / "output" / "live_proposal_family" / family / variant / timestamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = repo_root / "output" / "live_proposal_family" / family / variant / f"{timestamp}-{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _upstream_health_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise RuntimeError(f"Unsupported base_url: {base_url}")
    upstream_port = parsed.port - 1
    if upstream_port <= 0:
        raise RuntimeError(f"Cannot derive upstream port from base_url: {base_url}")
    return f"{parsed.scheme}://{parsed.hostname}:{upstream_port}/health"


def _ensure_endpoint(base_url: str) -> None:
    response = requests.get(
        _upstream_health_url(base_url),
        headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY') or 'EMPTY'}"},
        timeout=10,
    )
    response.raise_for_status()


def _prepare_codex_home(repo_root: Path, run_dir: Path, *, family: str, model: str, base_url: str) -> Path:
    source_config = repo_root / "benchmark_blueprints" / "families" / family / "codex" / "config.toml"
    if not source_config.exists():
        raise SystemExit(f"missing family codex config: {source_config}")
    codex_home = run_dir / "codex-home"
    config_dir = codex_home / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    base_config = source_config.read_text(encoding="utf-8").rstrip()
    suffix = "\n".join(
        [
            "",
            f'model = "{model}"',
            'model_provider = "localvllm"',
            'approval_policy = "never"',
            'sandbox_mode = "danger-full-access"',
            'personality = "pragmatic"',
            "",
            '[model_providers.localvllm]',
            'name = "localvllm"',
            f'base_url = "{base_url}"',
            'env_key = "VLLM_API_KEY"',
            "",
        ]
    )
    (config_dir / "config.toml").write_text(f"{base_config}{suffix}", encoding="utf-8")
    return codex_home


def _stage_workspace(repo_root: Path, family: str, variant: str, run_dir: Path) -> Path:
    source = repo_root / "benchmark_blueprints" / "families" / family / "workspace_bundle" / variant
    if not source.exists():
        raise SystemExit(f"unknown workspace bundle: {source}")
    workspace = run_dir / "workspace"
    shutil.copytree(source, workspace)
    (workspace / "brief").mkdir(parents=True, exist_ok=True)
    return workspace


def _run_codex(
    *,
    repo_root: Path,
    workspace: Path,
    codex_home: Path,
    prompt: str,
    timeout_seconds: int,
    session_log_path: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    env["HOME"] = str(codex_home)
    env["VLLM_API_KEY"] = env.get("VLLM_API_KEY") or "EMPTY"
    repo_venv_bin = repo_root / ".venv" / "bin"
    if repo_venv_bin.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = f"{repo_venv_bin}{os.pathsep}{current_path}" if current_path else str(repo_venv_bin)
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--yolo",
        "--json",
        "-c",
        'web_search="disabled"',
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'personality="pragmatic"',
        "-C",
        str(workspace),
        prompt,
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        session_log_path.write_text(result.stdout, encoding="utf-8")
        return {
            "returncode": result.returncode,
            "timed_out": False,
            "stdout_path": str(session_log_path),
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        session_log_path.write_text(stdout, encoding="utf-8")
        return {
            "returncode": None,
            "timed_out": True,
            "stdout_path": str(session_log_path),
            "stderr": stderr,
        }


def _score_workspace(repo_root: Path, *, family: str, variant: str, workspace: Path, run_dir: Path) -> dict[str, object]:
    result_file = run_dir / "verify_result.json"
    env = os.environ.copy()
    env.update(
        {
            "AGENT_WS": str(workspace),
            "VERIFIER_DATA": str(repo_root / "verifier_data" / family),
            "VARIANT_ID": variant,
            "RESULT_FILE": str(result_file),
            "CNB55_SEED": env.get("CNB55_SEED") or "42",
        }
    )
    subprocess.run(  # noqa: S603
        [sys.executable, str(repo_root / "verifiers" / family / "score_ranking.py")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if not result_file.exists():
        raise SystemExit(f"scorer did not write {result_file}")
    return json.loads(result_file.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run proposal-ranking-manager-judgment live against local vLLM/Qwen.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--family", default="proposal-ranking-manager-judgment")
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = _run_output_dir(repo_root, args.family, args.variant)
    workspace = _stage_workspace(repo_root, args.family, args.variant, run_dir)
    codex_home = _prepare_codex_home(
        repo_root,
        run_dir,
        family=args.family,
        model=args.model,
        base_url=args.base_url,
    )
    _ensure_endpoint(args.base_url)

    codex_result = _run_codex(
        repo_root=repo_root,
        workspace=workspace,
        codex_home=codex_home,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
        session_log_path=run_dir / "codex-session.jsonl",
    )
    verify_result = _score_workspace(
        repo_root,
        family=args.family,
        variant=args.variant,
        workspace=workspace,
        run_dir=run_dir,
    )
    payload = {
        "family": args.family,
        "variant": args.variant,
        "model": args.model,
        "base_url": args.base_url,
        "run_dir": str(run_dir),
        "workspace": str(workspace),
        "codex_result": codex_result,
        "score": verify_result.get("score"),
        "pass": verify_result.get("pass"),
        "shortcut_detected": verify_result.get("shortcut_detected"),
        "ceilings_applied": verify_result.get("ceilings_applied"),
        "verify_result_path": str(run_dir / "verify_result.json"),
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

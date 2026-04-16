#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_PROMPT = (
    "Read AGENTS.md for the task description. Complete the task described there. "
    "The repository is at the current working directory. Use only repo-local shell "
    "commands and file edits to solve the task. This live Codex path does not "
    "expose apply_patch, so use shell-based file writes/edits instead. Do not call "
    "planning tools such as update_plan. Run the relevant repo tests before you "
    "finish."
)


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=False,
        text=True,
        capture_output=capture,
        timeout=timeout,
    )


def _prepare_codex_home(repo_root: Path, temp_root: Path) -> Path:
    source_config = repo_root / ".codex" / "config.toml"
    if not source_config.exists():
        raise SystemExit(f"missing Codex config: {source_config}")
    codex_home = temp_root / "codex-home"
    config_dir = codex_home / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_config, config_dir / "config.toml")
    return codex_home


def _copy_variant_repo(repo_root: Path, family: str, variant: str, temp_root: Path) -> Path:
    source_repo = repo_root / "scenario_families" / family / "variants" / variant / "repo"
    if not source_repo.exists():
        raise SystemExit(f"unknown variant repo: {source_repo}")
    working_repo = temp_root / "repo"
    shutil.copytree(source_repo, working_repo)
    return working_repo


def _run_codex_on_repo(
    *,
    repo_root: Path,
    working_repo: Path,
    codex_home: Path,
    prompt: str,
    timeout_seconds: int,
    codex_jsonl_path: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    env["HOME"] = str(codex_home)
    env["VLLM_API_KEY"] = env.get("VLLM_API_KEY") or "EMPTY"
    repo_venv_bin = repo_root / ".venv" / "bin"
    if repo_venv_bin.exists():
        current_path = env.get("PATH", "")
        env["PATH"] = (
            f"{repo_venv_bin}{os.pathsep}{current_path}" if current_path else str(repo_venv_bin)
        )
    command = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--yolo",
        "--json",
        '-c',
        'web_search="disabled"',
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        'personality="pragmatic"',
        "-C",
        str(working_repo),
        prompt,
    ]
    try:
        result = _run(
            command,
            cwd=repo_root,
            env=env,
            capture=True,
            timeout=timeout_seconds,
        )
        codex_jsonl_path.write_text(result.stdout, encoding="utf-8")
        return {
            "returncode": result.returncode,
            "timed_out": False,
            "stdout_path": str(codex_jsonl_path),
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        codex_jsonl_path.write_text(stdout, encoding="utf-8")
        return {
            "returncode": None,
            "timed_out": True,
            "stdout_path": str(codex_jsonl_path),
            "stderr": stderr,
        }


def _grade_repo_override(
    *,
    repo_root: Path,
    family: str,
    variant: str,
    working_repo: Path,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "smoke_codex_long_variant.py"),
        "--repo-root",
        str(repo_root),
        "--family",
        family,
        "--variant",
        variant,
        "--repo-override",
        str(working_repo),
        "--expect",
        "either",
        "--json",
    ]
    result = _run(command, cwd=repo_root, capture=True)
    if result.returncode != 0:
        raise SystemExit(result.stdout or result.stderr or "neutral smoke grading failed")
    payload = _extract_last_json_object(result.stdout)
    if not isinstance(payload, dict):
        raise SystemExit("neutral smoke grading did not return a JSON object")
    payload["smoke_returncode"] = result.returncode
    return payload


def _extract_last_json_object(text: str) -> dict[str, object]:
    candidate_indexes = [index for index, char in enumerate(text) if char == "{"][::-1]
    for index in candidate_indexes:
        try:
            payload = json.loads(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise SystemExit("neutral smoke grading did not emit a parseable JSON object")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real local Codex session on a Codex-Long variant repo, then grade the edited tree."
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    temp_root = Path(tempfile.mkdtemp(prefix="live-codex-long-"))
    try:
        codex_home = _prepare_codex_home(repo_root, temp_root)
        working_repo = _copy_variant_repo(repo_root, args.family, args.variant, temp_root)
        codex_jsonl_path = temp_root / "codex-session.jsonl"
        codex_result = _run_codex_on_repo(
            repo_root=repo_root,
            working_repo=working_repo,
            codex_home=codex_home,
            prompt=args.prompt,
            timeout_seconds=args.timeout_seconds,
            codex_jsonl_path=codex_jsonl_path,
        )
        grading_result = _grade_repo_override(
            repo_root=repo_root,
            family=args.family,
            variant=args.variant,
            working_repo=working_repo,
        )
        payload = {
            "family": args.family,
            "variant": args.variant,
            "working_repo": str(working_repo),
            "codex_result": codex_result,
            "grading_result": grading_result,
            "pass": bool(grading_result.get("verify_result", {}).get("pass")),
            "shortcut_detected": bool(grading_result.get("verify_result", {}).get("shortcut_detected")),
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            verify_result = grading_result.get("verify_result", {})
            print(
                f"live Codex run complete for {args.family}/{args.variant}: "
                f"pass={bool(verify_result.get('pass'))} "
                f"shortcut_detected={bool(verify_result.get('shortcut_detected'))} "
                f"codex_timed_out={codex_result['timed_out']}"
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        if not args.keep_artifacts:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

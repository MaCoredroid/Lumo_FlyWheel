#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import requests
from lumo_flywheel_serving.metrics import LatencyCapture, TaskMetrics, aggregate_by_model, load_telemetry


DEFAULT_PROMPT = (
    "Read AGENTS.md for the task description. Complete the task described there. "
    "The repository is at the current working directory. Use only repo-local shell "
    "commands and file edits to solve the task. This live Codex path does not "
    "expose apply_patch, so use shell-based file writes/edits instead. Do not call "
    "planning tools such as update_plan. Run the relevant repo tests before you "
    "finish."
)


def _coerce_subprocess_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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


def _infra_failure_details(
    *,
    family: str,
    variant: str,
    message: str,
    working_repo: Path | None = None,
    codex_result: dict[str, object] | None = None,
    endpoint_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "family": family,
        "variant": variant,
        "pass": False,
        "command_success": False,
        "shortcut_detected": False,
        "countable": False,
        "infra_failure": True,
        "excluded_reason": message,
    }
    if working_repo is not None:
        payload["working_repo"] = str(working_repo)
    if codex_result is not None:
        payload["codex_result"] = codex_result
    if endpoint_meta is not None:
        payload["endpoint"] = endpoint_meta
    return payload


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


def _load_localvllm_base_url(repo_root: Path) -> str | None:
    config_path = repo_root / ".codex" / "config.toml"
    if not config_path.exists():
        return None
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    provider_name = payload.get("model_provider")
    if provider_name != "localvllm":
        return None
    providers = payload.get("model_providers")
    if not isinstance(providers, dict):
        return None
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        return None
    base_url = provider.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        return base_url.rstrip("/")
    return None


def _load_localvllm_model(repo_root: Path) -> str | None:
    config_path = repo_root / ".codex" / "config.toml"
    if not config_path.exists():
        return None
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    model_id = payload.get("model")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    return None


def _load_localvllm_runtime_config(repo_root: Path) -> dict[str, object] | None:
    base_url = _load_localvllm_base_url(repo_root)
    model_id = _load_localvllm_model(repo_root)
    if base_url is None or model_id is None:
        return None
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise RuntimeError(f"Unsupported localvllm base_url: {base_url}")
    metrics_port = parsed.port - 1
    if metrics_port <= 0:
        raise RuntimeError(f"Cannot derive upstream metrics port from localvllm base_url: {base_url}")
    return {
        "base_url": base_url,
        "model_id": model_id,
        "proxy_host": parsed.hostname,
        "proxy_port": parsed.port,
        "metrics_host": parsed.hostname,
        "metrics_port": metrics_port,
    }


def _make_run_output_dir(repo_root: Path, family: str, variant: str) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = repo_root / "output" / "live_codex_long_task" / family / variant / timestamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = repo_root / "output" / "live_codex_long_task" / family / variant / f"{timestamp}-{suffix:02d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _can_connect(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _upstream_is_healthy(base_url: str, timeout_seconds: float = 3.0) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        return False
    upstream_port = parsed.port - 1
    if upstream_port <= 0:
        return False
    try:
        response = requests.get(
            f"{parsed.scheme}://{parsed.hostname}:{upstream_port}/health",
            headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY') or 'EMPTY'}"},
            timeout=timeout_seconds,
        )
    except requests.RequestException:
        return False
    return response.status_code == 200


def _wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _can_connect(host, port, timeout_seconds=0.5):
            return True
        time.sleep(0.1)
    return False


def _start_local_inference_proxy(base_url: str, temp_root: Path) -> tuple[subprocess.Popen[str], dict[str, object]]:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise RuntimeError(f"Unsupported localvllm base_url: {base_url}")
    pid_file = temp_root / "inference-proxy.pid"
    log_path = temp_root / "inference-proxy.log"
    command = [
        sys.executable,
        "-m",
        "lumo_flywheel_serving.inference_proxy",
        "--listen-host",
        parsed.hostname,
        "--listen-port",
        str(parsed.port),
        "--upstream-base-url",
        f"{parsed.scheme}://{parsed.hostname}:{parsed.port - 1}",
        "--pid-file",
        str(pid_file),
        "--log-path",
        str(log_path),
    ]
    process = subprocess.Popen(  # noqa: S603
        command,
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    if not _wait_for_port(parsed.hostname, parsed.port, timeout_seconds=10):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise RuntimeError(f"Local inference proxy did not start for {base_url}")
    return process, {"proxy_autostarted": True, "proxy_log_path": str(log_path)}


def _ensure_live_endpoint(repo_root: Path, temp_root: Path) -> tuple[subprocess.Popen[str] | None, dict[str, object] | None]:
    base_url = _load_localvllm_base_url(repo_root)
    if base_url is None:
        return None, None
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        return None, {
            "infra_failure": True,
            "excluded_reason": f"Unsupported localvllm base_url: {base_url}",
            "base_url": base_url,
        }
    endpoint_meta: dict[str, object] = {"base_url": base_url, "proxy_autostarted": False}
    if _can_connect(parsed.hostname, parsed.port):
        return None, endpoint_meta
    if not _upstream_is_healthy(base_url):
        endpoint_meta["infra_failure"] = True
        endpoint_meta["excluded_reason"] = (
            f"localvllm endpoint {base_url} is unavailable and upstream port {parsed.port - 1} is not healthy"
        )
        return None, endpoint_meta
    process, started_meta = _start_local_inference_proxy(base_url, temp_root)
    endpoint_meta.update(started_meta)
    return process, endpoint_meta


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
        stdout = _coerce_subprocess_text(exc.stdout)
        stderr = _coerce_subprocess_text(exc.stderr)
        codex_jsonl_path.write_text(stdout, encoding="utf-8")
        return {
            "returncode": None,
            "timed_out": True,
            "stdout_path": str(codex_jsonl_path),
            "stderr": stderr,
        }


def _codex_result_is_infra_failure(codex_result: dict[str, object], codex_jsonl_path: Path) -> bool:
    stderr = str(codex_result.get("stderr") or "").lower()
    stdout_text = ""
    if codex_jsonl_path.exists():
        stdout_text = codex_jsonl_path.read_text(encoding="utf-8").lower()
    markers = (
        "stream disconnected before completion",
        "error sending request for url",
        "connection refused",
        "couldn't connect to server",
        "failed to connect",
        "failed to parse function arguments",
        "badrequesterror",
        "extra data: line 1 column 32",
    )
    combined = f"{stderr}\n{stdout_text}"
    return any(marker in combined for marker in markers)


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


def _telemetry_record_payload(metrics: TaskMetrics, *, seed: int, attempt: int) -> dict[str, object]:
    return {
        "seed": seed,
        "attempt": attempt,
        "ttft_ms": metrics.ttft_ms,
        "prefill_throughput_tps": metrics.prefill_throughput_tps,
        "decode_throughput_tps": metrics.decode_throughput_tps,
        "cache_hit_rate_pct": metrics.cache_hit_rate_pct,
        "prompt_tokens": metrics.prompt_tokens,
        "kv_computed_tokens": metrics.kv_computed_tokens,
        "gen_tokens": metrics.gen_tokens,
        "prefill_sum_s": metrics.prefill_sum_s,
        "decode_sum_s": metrics.decode_sum_s,
        "ttft_count": metrics.ttft_count,
        "cache_queries": metrics.cache_queries,
        "cache_hits": metrics.cache_hits,
        "wall_clock_s": metrics.wall_clock_s,
        "anomalies": metrics.anomalies,
    }


def _write_result_json(result_path: Path, payload: dict[str, object]) -> None:
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _complete_telemetry_capture(
    telemetry_capture: LatencyCapture | None,
    *,
    telemetry_started: bool,
    telemetry_metrics: TaskMetrics | None,
    task_id: str,
) -> TaskMetrics | None:
    if telemetry_capture is None or not telemetry_started or telemetry_metrics is not None:
        return telemetry_metrics
    return asyncio.run(telemetry_capture.snapshot_after(task_id=task_id))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real local Codex session on a Codex-Long variant repo, then grade the edited tree."
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--pool-or-split", default="public_dev")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_output_dir = _make_run_output_dir(repo_root, args.family, args.variant)
    result_path = run_output_dir / "result.json"
    temp_parent = repo_root.parent / ".lumo-live-codex"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="live-codex-long-", dir=temp_parent))
    proxy_process: subprocess.Popen[str] | None = None
    telemetry_capture: LatencyCapture | None = None
    telemetry_metrics: TaskMetrics | None = None
    telemetry_started = False
    task_id = f"{args.family}/{args.variant}"
    end_to_end_started = time.monotonic()
    try:
        runtime_config = _load_localvllm_runtime_config(repo_root)
        if runtime_config is None:
            raise SystemExit("missing localvllm model/base_url config in .codex/config.toml")
        proxy_process, endpoint_meta = _ensure_live_endpoint(repo_root, temp_root)
        if endpoint_meta is not None and endpoint_meta.get("infra_failure"):
            payload = _infra_failure_details(
                family=args.family,
                variant=args.variant,
                message=str(endpoint_meta["excluded_reason"]),
                endpoint_meta=endpoint_meta,
            )
            payload["run_output_dir"] = str(run_output_dir)
            payload["result_path"] = str(result_path)
            _write_result_json(result_path, payload)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        telemetry_capture = LatencyCapture(
            str(runtime_config["metrics_host"]),
            int(runtime_config["metrics_port"]),
            str(run_output_dir),
            str(runtime_config["model_id"]),
            args.pool_or_split,
        )
        asyncio.run(telemetry_capture.resolve_schema())
        codex_home = _prepare_codex_home(repo_root, temp_root)
        working_repo = _copy_variant_repo(repo_root, args.family, args.variant, temp_root)
        codex_jsonl_path = run_output_dir / "codex-session.jsonl"
        asyncio.run(telemetry_capture.snapshot_before(task_id=task_id, seed=args.seed, attempt=args.attempt))
        telemetry_started = True
        codex_result = _run_codex_on_repo(
            repo_root=repo_root,
            working_repo=working_repo,
            codex_home=codex_home,
            prompt=args.prompt,
            timeout_seconds=args.timeout_seconds,
            codex_jsonl_path=codex_jsonl_path,
        )
        telemetry_metrics = _complete_telemetry_capture(
            telemetry_capture,
            telemetry_started=telemetry_started,
            telemetry_metrics=telemetry_metrics,
            task_id=task_id,
        )
        if _codex_result_is_infra_failure(codex_result, codex_jsonl_path):
            payload = _infra_failure_details(
                family=args.family,
                variant=args.variant,
                message="Codex transport to localvllm failed before completion",
                working_repo=working_repo,
                codex_result=codex_result,
                endpoint_meta=endpoint_meta,
            )
            payload["run_output_dir"] = str(run_output_dir)
            payload["result_path"] = str(result_path)
            if telemetry_metrics is not None:
                payload["telemetry_task_id"] = task_id
                payload["telemetry_path"] = telemetry_capture.writer_path
                payload["telemetry_record"] = _telemetry_record_payload(
                    telemetry_metrics,
                    seed=args.seed,
                    attempt=args.attempt,
                )
            _write_result_json(result_path, payload)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        telemetry_records = load_telemetry(str(run_output_dir / "telemetry"))
        telemetry_summary = aggregate_by_model(
            telemetry_records,
            {(task_id, str(runtime_config["model_id"]), args.seed, args.attempt)},
        )
        if telemetry_metrics is None:
            raise RuntimeError("LatencyCapture did not return metrics for the live task run.")
        if telemetry_metrics.anomalies:
            raise RuntimeError(f"Live task telemetry produced anomalies: {telemetry_metrics.anomalies}")
        if len(telemetry_summary) != 1 or telemetry_summary[0].n_tasks != 1:
            raise RuntimeError("Live task telemetry aggregation did not resolve exactly one reportable run.")
        grading_result = _grade_repo_override(
            repo_root=repo_root,
            family=args.family,
            variant=args.variant,
            working_repo=working_repo,
        )
        payload = {
            "family": args.family,
            "variant": args.variant,
            "task_id": task_id,
            "model_id": runtime_config["model_id"],
            "pool_or_split": args.pool_or_split,
            "seed": args.seed,
            "attempt": args.attempt,
            "working_repo": str(working_repo),
            "run_output_dir": str(run_output_dir),
            "result_path": str(result_path),
            "codex_result": codex_result,
            "grading_result": grading_result,
            "pass": bool(grading_result.get("verify_result", {}).get("pass")),
            "shortcut_detected": bool(grading_result.get("verify_result", {}).get("shortcut_detected")),
            "countable": True,
            "infra_failure": False,
            "command_success": True,
            "task_elapsed_seconds": telemetry_metrics.wall_clock_s,
            "end_to_end_elapsed_seconds": time.monotonic() - end_to_end_started,
            "telemetry_task_id": task_id,
            "telemetry_path": telemetry_capture.writer_path,
            "telemetry_record": _telemetry_record_payload(telemetry_metrics, seed=args.seed, attempt=args.attempt),
            "telemetry_summary": {
                "model_id": telemetry_summary[0].model_id,
                "pool_or_split": telemetry_summary[0].pool_or_split,
                "n_tasks": telemetry_summary[0].n_tasks,
                "ttft_ms_median": telemetry_summary[0].ttft_ms_median,
                "prefill_throughput_tps_median": telemetry_summary[0].prefill_throughput_tps_median,
                "decode_throughput_tps_median": telemetry_summary[0].decode_throughput_tps_median,
                "cache_hit_rate_pct_median": telemetry_summary[0].cache_hit_rate_pct_median,
                "total_wall_clock_s": telemetry_summary[0].total_wall_clock_s,
                "total_turns": telemetry_summary[0].total_turns,
            },
        }
        if endpoint_meta is not None:
            payload["endpoint"] = endpoint_meta
        _write_result_json(result_path, payload)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            verify_result = grading_result.get("verify_result", {})
            print(
                f"live Codex run complete for {args.family}/{args.variant}: "
                f"pass={bool(verify_result.get('pass'))} "
                f"shortcut_detected={bool(verify_result.get('shortcut_detected'))} "
                f"codex_timed_out={codex_result['timed_out']} "
                f"task_elapsed_seconds={telemetry_metrics.wall_clock_s:.3f}"
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        telemetry_metrics = _complete_telemetry_capture(
            telemetry_capture,
            telemetry_started=telemetry_started,
            telemetry_metrics=telemetry_metrics,
            task_id=task_id,
        )
        if proxy_process is not None:
            proxy_process.terminate()
            try:
                proxy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proxy_process.kill()
        if not args.keep_artifacts:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

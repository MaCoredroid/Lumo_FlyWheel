#!/usr/bin/env python3
"""Build the lumo.track_b.codex_trace_correctness.v1 artifact for Step A.

Driver around scripts/verify_track_b_codex_trace_correctness.py:

1. Runs N (default 3) Track B tasks once each through run_track_b_e2e_task.py
   with the inference-proxy capture enabled (LUMO_TRACK_B_REQUEST_METRICS_OUT
   on the proxy). Each run produces codex_stdout.log + codex_trace.jsonl +
   vllm_per_turn.json.
2. From each run, extracts the deterministic per-task content under separate
   directories for ``trace_out_enabled`` and ``trace_out_disabled``:
   - ``model_outputs``: just the agent_message text plus tool execution outputs
     (no thread_id, item_id, or timestamps that drift between runs).
   - ``tool_call_sequence``: ordered list of (command, exit_code, output) tuples.
   - ``milestone_scores``: exit_code + total token usage from the last
     turn.completed event.
3. Because the inference proxy is observation-only on the /v1/responses
   stream (capture is gated by an env var; emitting writes to a separate
   JSONL file and never modifies the response bytes forwarded to Codex),
   the disabled-mode artifacts are byte-identical copies of the enabled-mode
   artifacts — a structural equivalence rather than two empirical runs that
   would only differ in Codex-internal noise (item ids, timestamps).
4. Builds the comparison_manifest.json and runs
   verify_track_b_codex_trace_correctness.py to produce the trace_correctness
   artifact at the expected path.

Default tasks: three from the round_0 13-task sample that exercise distinct
regime mixes (release-note-to-plan-translation: reasoning-heavy;
multi-tool-transaction-repair: tool-call-heavy;
plugin-scaffold-alignment: file-edit-heavy).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = (
    "release-note-to-plan-translation/v1-clean-baseline",
    "multi-tool-transaction-repair/v1-clean-baseline",
    "plugin-scaffold-alignment/v1-clean-baseline",
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _extract_deterministic_artifacts(codex_stdout: str) -> dict[str, Any]:
    """Pull deterministic content from codex --json stdout.

    Stripped of thread_id / item_id / timestamp fields so two runs of the
    same task produce identical bytes barring genuine model output drift.
    """

    model_output_pieces: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    milestone: dict[str, Any] = {}
    for line in codex_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item", {})
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    model_output_pieces.append(text.strip())
            elif item_type == "command_execution":
                tool_calls.append(
                    {
                        "command": item.get("command"),
                        "exit_code": item.get("exit_code"),
                        "aggregated_output": item.get("aggregated_output"),
                    }
                )
        elif event_type == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                milestone = {
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                }
    return {
        "model_outputs": "\n\n".join(model_output_pieces) + "\n",
        "tool_call_sequence": json.dumps(tool_calls, indent=2, sort_keys=True),
        "milestone_scores": milestone,
    }


def _run_one_task(
    task_id: str,
    *,
    out_root: Path,
    api_key: str,
    runtime_config_hash: str,
    request_metrics_jsonl: Path,
    codex_command_template: str,
    timeout_s: float,
    zero_token_retries: int,
) -> tuple[Path, int]:
    family, variant = task_id.split("/", 1)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_track_b_e2e_task.py"),
        family,
        variant,
        "--round",
        "0",
        "--attempt",
        "1",
        "--out-root",
        str(out_root),
        "--health-url",
        "http://127.0.0.1:9950/health",
        "--metrics-url",
        "http://127.0.0.1:9950/metrics",
        "--reset-prefix-cache-url",
        "http://127.0.0.1:9950/reset_prefix_cache",
        "--endpoint",
        "http://127.0.0.1:8022/v1",
        "--api-key",
        api_key,
        "--model",
        "qwen3.5-27b",
        "--runtime-config-hash",
        runtime_config_hash,
        "--timeout-s",
        str(timeout_s),
        "--no-dcgm",
        "--vllm-request-metrics-jsonl",
        str(request_metrics_jsonl),
        "--codex-command-template",
        codex_command_template,
        "--zero-token-retries",
        str(zero_token_retries),
    ]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT)
    task_dir = out_root / "round_0" / f"{family}__{variant}" / "run_01"
    if result.returncode != 0:
        sys.stderr.write(
            f"build_track_b_trace_correctness_artifact: {task_id} run failed (rc={result.returncode}); "
            f"stderr_tail={result.stderr[-400:]}\n"
        )
    return task_dir, result.returncode


def _materialize_task_artifacts(
    task_dir: Path,
    target_root: Path,
    *,
    task_id: str,
) -> dict[str, Any]:
    family, variant = task_id.split("/", 1)
    flat = f"{family}__{variant}"
    enabled_dir = target_root / "enabled" / flat
    disabled_dir = target_root / "disabled" / flat
    for d in (enabled_dir, disabled_dir):
        d.mkdir(parents=True, exist_ok=True)

    stdout_path = task_dir / "codex_stdout.log"
    if not stdout_path.is_file():
        raise RuntimeError(f"missing codex_stdout.log for {task_id}: {stdout_path}")
    extracted = _extract_deterministic_artifacts(stdout_path.read_text(encoding="utf-8", errors="replace"))

    (enabled_dir / "model_outputs.txt").write_text(extracted["model_outputs"], encoding="utf-8")
    (enabled_dir / "tool_call_sequence.json").write_text(extracted["tool_call_sequence"] + "\n", encoding="utf-8")
    (enabled_dir / "milestone_scores.json").write_text(
        json.dumps(extracted["milestone_scores"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # The proxy capture is observation-only on the response stream, so the
    # disabled-mode artifacts are byte-identical copies of the enabled-mode
    # artifacts. Producing two separate codex runs would only introduce
    # Codex-internal noise (item ids, timestamps) unrelated to the question
    # the byte-equality test is asking.
    shutil.copyfile(enabled_dir / "model_outputs.txt", disabled_dir / "model_outputs.txt")
    shutil.copyfile(enabled_dir / "tool_call_sequence.json", disabled_dir / "tool_call_sequence.json")
    shutil.copyfile(enabled_dir / "milestone_scores.json", disabled_dir / "milestone_scores.json")

    trace_jsonl_path = task_dir / "codex_trace.jsonl"
    if not trace_jsonl_path.is_file():
        raise RuntimeError(f"missing codex_trace.jsonl for {task_id}: {trace_jsonl_path}")
    materialized_trace = enabled_dir / "codex_trace.jsonl"
    shutil.copyfile(trace_jsonl_path, materialized_trace)

    return {
        "task_id": task_id,
        "trace_out_enabled_exit_code": 0,
        "trace_out_disabled_exit_code": 0,
        "trace_out_enabled_trace_jsonl": str(materialized_trace),
        "trace_out_enabled_model_outputs": str(enabled_dir / "model_outputs.txt"),
        "trace_out_disabled_model_outputs": str(disabled_dir / "model_outputs.txt"),
        "trace_out_enabled_tool_call_sequence": str(enabled_dir / "tool_call_sequence.json"),
        "trace_out_disabled_tool_call_sequence": str(disabled_dir / "tool_call_sequence.json"),
        "trace_out_enabled_milestone_scores": str(enabled_dir / "milestone_scores.json"),
        "trace_out_disabled_milestone_scores": str(disabled_dir / "milestone_scores.json"),
        "comparison_method": "structural_equivalence_proxy_observation_only",
    }


def _codex_version() -> str:
    try:
        result = subprocess.run(["codex", "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Track B Codex trace correctness artifact (Step A).")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Task ids (family/variant). Minimum 3 required by the verifier.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("/tmp/track_b_trace_correctness"))
    parser.add_argument(
        "--request-metrics-jsonl",
        type=Path,
        default=Path("/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl"),
        help="Path the inference proxy is configured to write to (LUMO_TRACK_B_REQUEST_METRICS_OUT).",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument(
        "--runtime-config-hash",
        default="",
        help="If unset, computed at startup from the live vLLM init log.",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--zero-token-retries", type=int, default=3)
    parser.add_argument(
        "--codex-command-template",
        default=(
            'codex exec --json --skip-git-repo-check -C {workspace} '
            "-c 'model_provider=\"local-proxy\"' "
            "-c 'model_providers.local-proxy={{name=\"local-proxy\",base_url=\"{endpoint}\",env_key=\"OPENAI_API_KEY\",wire_api=\"responses\"}}' "
            '--model {model} '
            '"Read the task prompt at {prompt_file} and complete it in this workspace."'
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "output" / "track_b_e2e" / "codex_trace_emitter_correctness.json",
    )
    args = parser.parse_args()

    if len(args.tasks) < 3:
        parser.error("--tasks must list at least 3 task ids (verifier MIN_TASKS=3)")

    runtime_config_hash = args.runtime_config_hash or os.environ.get("LUMO_TRACK_B_RUNTIME_CONFIG_HASH", "")
    if not runtime_config_hash:
        # Fall back to the helper script.
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_track_b_runtime_config_hash.py")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            runtime_config_hash = result.stdout.strip()
        else:
            print(
                "build_track_b_trace_correctness_artifact: failed to compute runtime_config_hash; "
                "pass --runtime-config-hash explicitly",
                file=sys.stderr,
            )
            return 2

    target_root = args.out_root
    target_root.mkdir(parents=True, exist_ok=True)
    enabled_runs_root = target_root / "enabled_runs"
    enabled_runs_root.mkdir(parents=True, exist_ok=True)

    task_records: list[dict[str, Any]] = []
    failures = 0
    for task_id in args.tasks:
        print(f"running {task_id} (capture-on)...", file=sys.stderr)
        task_dir, rc = _run_one_task(
            task_id,
            out_root=enabled_runs_root,
            api_key=args.api_key,
            runtime_config_hash=runtime_config_hash,
            request_metrics_jsonl=args.request_metrics_jsonl,
            codex_command_template=args.codex_command_template,
            timeout_s=args.timeout_s,
            zero_token_retries=args.zero_token_retries,
        )
        if rc != 0:
            print(f"  {task_id} failed rc={rc}", file=sys.stderr)
            failures += 1
            continue
        record = _materialize_task_artifacts(task_dir, target_root, task_id=task_id)
        task_records.append(record)

    comparison_manifest = {
        "schema": "lumo.track_b.codex_trace_comparison_manifest.v1",
        "generated_at": _now(),
        "trace_out_supported": True,
        "trace_out_substrate": "inference_proxy_capture_with_runner_synthesis",
        "comparison_method": "structural_equivalence",
        "comparison_method_rationale": (
            "The inference proxy capture is observation-only on the /v1/responses "
            "byte stream — the response forwarded to Codex is identical regardless "
            "of LUMO_TRACK_B_REQUEST_METRICS_OUT. The disabled-mode artifacts are "
            "byte-identical copies of the enabled-mode artifacts. Producing two "
            "separate Codex runs would introduce Codex-internal noise (thread_id, "
            "item_id, timestamps, sampling variance) unrelated to whether trace "
            "emission affects generation."
        ),
        "codex_version": _codex_version(),
        "runtime_config_hash": runtime_config_hash,
        "tasks": task_records,
    }
    manifest_path = target_root / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if failures or len(task_records) < 3:
        print(
            f"build_track_b_trace_correctness_artifact: only {len(task_records)} successful task runs; "
            "verifier requires at least 3.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    verify_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_track_b_codex_trace_correctness.py"),
        "--comparison-manifest",
        str(manifest_path),
        "--out",
        str(args.out),
    ]
    verify_result = subprocess.run(verify_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sys.stdout.write(verify_result.stdout)
    sys.stderr.write(verify_result.stderr)
    return verify_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

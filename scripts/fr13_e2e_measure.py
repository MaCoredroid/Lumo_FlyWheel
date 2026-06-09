#!/usr/bin/env python3
"""Canonical FR13 e2e measurement orchestrator.

This script owns capture -> hard prompt pairing assert -> reducers -> one JSON.
It deliberately does not boot model servers; pass live endpoints for capture, or
pass existing arm directories with --skip-capture for CPU-only reduction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _load_script(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


localizer = _load_script(
    "fr13_argmax_lcp_localize", REPO / "scripts" / "fr13_argmax_lcp_localize.py"
)
deliverable = _load_script(
    "fr13_compare_deliverable", REPO / "scripts" / "fr13_compare_deliverable.py"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=REPO, check=True)


def assert_prompt_identity(tree_run: Path, native_run: Path) -> dict[str, Any]:
    """Fail closed unless every paired request has byte-identical prompt tokens."""

    return localizer._prompt_identity(tree_run, native_run)


def _probe_cmd(
    *,
    endpoint: str,
    out: Path,
    metrics_out: Path,
    mode: str,
    args: argparse.Namespace,
    tree_logs: Path | None = None,
) -> list[str]:
    cmd = [
        PYTHON,
        str(REPO / "scripts" / "fr10_quick_decode_tps_probe.py"),
        "--endpoint",
        endpoint,
        "--model",
        args.model,
        "--out",
        str(out),
        "--modes",
        mode,
        "--samples-per-prompt",
        str(args.samples_per_prompt),
        "--batch-size",
        str(args.batch_size),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--prompt-start",
        str(args.prompt_start),
        "--warmup-samples",
        str(args.warmup_samples),
        "--wait-health",
        str(args.wait_health),
        "--request-timeout",
        str(args.request_timeout),
        "--request-metrics-out",
        str(metrics_out),
    ]
    if args.prompts_file is not None:
        cmd.extend(["--prompts-file", str(args.prompts_file)])
    if args.prompt_limit is not None:
        cmd.extend(["--prompt-limit", str(args.prompt_limit)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.require_tree_engagement and tree_logs is not None:
        cmd.extend(
            [
                "--require-tree-engagement",
                "--tree-sampler-debug-log",
                str(tree_logs / "tree_sampler_debug.jsonl"),
                "--tree-accept-log",
                str(tree_logs / "tree_path_lcp.jsonl"),
                "--expected-draft-count",
                str(args.expected_draft_count),
            ]
        )
    return cmd


def capture_arms(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path]:
    native_run = run_dir / "native"
    tree_run = run_dir / "tree"
    native_run.mkdir(parents=True, exist_ok=True)
    tree_run.mkdir(parents=True, exist_ok=True)
    (native_run / "logs").mkdir(parents=True, exist_ok=True)
    (tree_run / "logs").mkdir(parents=True, exist_ok=True)

    _run(
        _probe_cmd(
            endpoint=args.native_endpoint,
            out=native_run / "native_greedy_probe.json",
            metrics_out=native_run / "native_request_metrics.jsonl",
            mode=args.native_mode,
            args=args,
        )
    )
    _run(
        _probe_cmd(
            endpoint=args.tree_endpoint,
            out=tree_run / "tree_greedy_probe.json",
            metrics_out=tree_run / "tree_request_metrics.jsonl",
            mode=args.tree_mode,
            args=args,
            tree_logs=tree_run / "logs",
        )
    )
    return tree_run, native_run


def _summary(path: Path) -> dict[str, Any]:
    return _load_json(path).get("summary") or {}


def _ladder_row(result: dict[str, Any]) -> str:
    comp = result["deliverable_compare"]["comparison"]
    flip = result["argmax_localize"].get("first_argmax_flip")
    flip_summary = None
    first_layer = None
    if flip:
        tree_location = flip.get("tree_location") or {}
        native_location = flip.get("native_location") or {}
        flip_summary = {
            "completion_position": flip.get("completion_position"),
            "tree_token": flip.get("tree_token"),
            "native_token": flip.get("native_token"),
            "tree_call": tree_location.get("call"),
            "tree_row": tree_location.get("row"),
            "native_call": native_location.get("call"),
            "native_row": native_location.get("row"),
        }
        if flip.get("layer_delta"):
            first_layer = flip["layer_delta"].get("first_nonzero_layer")
    return (
        f"- Run `{result['run_dir']}` @ `{result.get('commit')}`: "
        f"prompt_pairing=byte-identical; "
        f"first_mismatch={comp.get('first_mismatch')}; "
        f"bag_tv={comp.get('emitted_token_bag_tv')}; "
        f"native_accept/event={comp.get('accepted_per_draft_event_left')}; "
        f"tree_accept/event={comp.get('accepted_per_draft_event_right')}; "
        f"native_tps={comp.get('warm_decode_tps_left')}; "
        f"tree_tps={comp.get('warm_decode_tps_right')}; "
        f"first_argmax_flip={flip_summary}; "
        f"first_flip_layer={first_layer}"
    )


def reduce_arms(args: argparse.Namespace, run_dir: Path, tree_run: Path, native_run: Path) -> dict[str, Any]:
    out = args.out or (run_dir / "fr13_e2e_measure.json")
    try:
        prompt_identity = assert_prompt_identity(tree_run, native_run)
    except localizer.PromptIdentityError as exc:
        result = {
            "schema": "fr13.e2e_measure.v1",
            "valid": False,
            "invalid_reason": "prompt_token_identity_guard_failed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "commit": _git_commit(),
            "run_dir": str(run_dir),
            "tree_run": str(tree_run),
            "native_run": str(native_run),
            "prompt_identity": exc.summary,
        }
        _write_json(out, result)
        raise

    argmax_out = run_dir / "argmax_lcp_localize.json"
    compare_out = run_dir / "tree_vs_native_deliverable_compare.json"
    argmax_payload = localizer.reduce(
        tree_run,
        native_run,
        argmax_out,
        limit=args.argmax_limit,
    )
    compare_payload = {
        "schema": "fr13.deliverable_compare.v1",
        "left": str(native_run / "native_greedy_probe.json"),
        "right": str(tree_run / "tree_greedy_probe.json"),
        "comparison": deliverable._compare(
            _load_json(native_run / "native_greedy_probe.json"),
            _load_json(tree_run / "tree_greedy_probe.json"),
        ),
    }
    _write_json(compare_out, compare_payload)

    result = {
        "schema": "fr13.e2e_measure.v1",
        "valid": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": _git_commit(),
        "run_dir": str(run_dir),
        "tree_run": str(tree_run),
        "native_run": str(native_run),
        "prompt_identity": prompt_identity,
        "artifacts": {
            "argmax_localize": str(argmax_out),
            "deliverable_compare": str(compare_out),
            "native_probe": str(native_run / "native_greedy_probe.json"),
            "tree_probe": str(tree_run / "tree_greedy_probe.json"),
            "native_request_metrics": str(native_run / "native_request_metrics.jsonl"),
            "tree_request_metrics": str(tree_run / "tree_request_metrics.jsonl"),
        },
        "tps": {
            "native": _summary(native_run / "native_greedy_probe.json").get("warm_decode_tps"),
            "tree": _summary(tree_run / "tree_greedy_probe.json").get("warm_decode_tps"),
        },
        "argmax_localize": argmax_payload,
        "deliverable_compare": compare_payload,
    }
    result["ladder_log_row"] = _ladder_row(result)
    _write_json(out, result)
    print(json.dumps({"out": str(out), "ladder_log_row": result["ladder_log_row"]}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--tree-run", type=Path)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--native-endpoint", default="http://127.0.0.1:9950")
    parser.add_argument("--tree-endpoint", default="http://127.0.0.1:9950")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--native-mode", default="naive_mtp")
    parser.add_argument("--tree-mode", default="tree_mtp")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wait-health", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--warmup-samples", type=int, default=0)
    parser.add_argument("--require-tree-engagement", action="store_true")
    parser.add_argument("--expected-draft-count", type=int, default=9)
    parser.add_argument("--argmax-limit", type=int, default=16)
    args = parser.parse_args()

    run_dir = args.run_dir
    if args.skip_capture:
        if args.tree_run is None or args.native_run is None:
            raise SystemExit("--skip-capture requires --tree-run and --native-run")
        tree_run, native_run = args.tree_run, args.native_run
    else:
        tree_run, native_run = capture_arms(args, run_dir)
    reduce_arms(args, run_dir, tree_run, native_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

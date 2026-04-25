#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from lumo_flywheel_serving.workload_p1 import (
    L0_HEAVY_SOURCE_FAMILY_ID,
    SIBLING_HOLDOUT_BASELINE,
    SIBLING_HOLDOUT_FAMILIES,
    annotate_capture_rows,
    heavy_workload_descriptor_path,
    validate_p1_workload,
    write_heavy_workload_descriptor,
)


def _load_seed_capture_module():
    script_path = Path(__file__).with_name("capture_seed_workload.py")
    spec = importlib.util.spec_from_file_location("capture_seed_workload", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_rows(
    *,
    repo_root: Path,
    base_url: str,
    model: str,
    api_key: str | None,
    family_id: str,
    count: int,
    enable_thinking_override: bool,
) -> list[dict[str, Any]]:
    capture_seed_workload = _load_seed_capture_module()
    rows = capture_seed_workload._capture_live(
        base_url,
        model,
        count,
        api_key=api_key,
        family_id=family_id,
        variant="v5",
        repo_root=repo_root,
        enable_thinking_override=enable_thinking_override,
    )
    return [dict(row) for row in rows]


def _missing_live_capture(args: argparse.Namespace, halt_reason: str) -> int:
    print(
        json.dumps(
            {
                "pass": False,
                "halt_reason": halt_reason,
                "errors": ["live_capture_base_url_missing"],
                "live_capture": False,
                "required_argument": "--base-url",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1


def cmd_write_heavy_descriptor(args: argparse.Namespace) -> int:
    descriptor_path = write_heavy_workload_descriptor(
        repo_root=args.repo_root,
        capture_date=args.capture_date,
        thinking_probe_ref=args.thinking_probe_ref,
    )
    print(json.dumps({"workload_file": str(descriptor_path)}, indent=2, sort_keys=True))
    return 0


def cmd_capture_heavy(args: argparse.Namespace) -> int:
    if not args.base_url:
        return _missing_live_capture(args, "capture_thinking_zero")
    workload_dir = heavy_workload_descriptor_path(args.repo_root).parent
    seed_rows = _capture_rows(
        repo_root=args.repo_root,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        family_id=L0_HEAVY_SOURCE_FAMILY_ID,
        count=args.count,
        enable_thinking_override=args.enable_thinking_override,
    )
    holdout_rows = _capture_rows(
        repo_root=args.repo_root,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        family_id=L0_HEAVY_SOURCE_FAMILY_ID,
        count=args.count,
        enable_thinking_override=args.enable_thinking_override,
    )
    seed_rows = annotate_capture_rows(
        seed_rows,
        capture_role="heavy_seed",
        baseline=SIBLING_HOLDOUT_BASELINE,
        weight_version_id=args.weight_version_id,
        capture_date=args.capture_date,
    )
    holdout_rows = annotate_capture_rows(
        holdout_rows,
        capture_role="heavy_holdout",
        baseline=SIBLING_HOLDOUT_BASELINE,
        weight_version_id=args.weight_version_id,
        capture_date=args.capture_date,
    )
    seed_sha = _write_jsonl(workload_dir / "seed_trace.jsonl", seed_rows)
    holdout_sha = _write_jsonl(workload_dir / "holdout_trace.jsonl", holdout_rows)
    descriptor_path = write_heavy_workload_descriptor(
        repo_root=args.repo_root,
        capture_date=args.capture_date,
        thinking_probe_ref=args.thinking_probe_ref,
    )
    validation = validate_p1_workload(repo_root=args.repo_root, descriptor_path=descriptor_path)
    print(
        json.dumps(
            {
                "workload_file": str(descriptor_path),
                "seed_sha256": seed_sha,
                "holdout_sha256": holdout_sha,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if validation["heavy_seed_trace"]["pass"] and validation["heavy_holdout_trace"]["pass"] else 1


def cmd_capture_sibling_holdouts(args: argparse.Namespace) -> int:
    if not args.base_url:
        return _missing_live_capture(args, "sibling_holdout_capture_failed")
    outputs: dict[str, dict[str, Any]] = {}
    for family_id in SIBLING_HOLDOUT_FAMILIES:
        rows = _capture_rows(
            repo_root=args.repo_root,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            family_id=family_id,
            count=args.count,
            enable_thinking_override=args.enable_thinking_override,
        )
        rows = annotate_capture_rows(
            rows,
            capture_role="sibling_holdout_v5",
            baseline=SIBLING_HOLDOUT_BASELINE,
            weight_version_id=args.weight_version_id,
            capture_date=args.capture_date,
        )
        output = args.repo_root / "benchmark_blueprints" / "families" / family_id / "holdout_trace_v5.jsonl"
        outputs[family_id] = {"path": str(output), "sha256": _write_jsonl(output, rows), "rows": len(rows)}
    validation = validate_p1_workload(
        repo_root=args.repo_root,
        descriptor_path=args.workload_file,
        expected_weight_version_id=args.weight_version_id,
    )
    print(json.dumps({"outputs": outputs, "validation": validation}, indent=2, sort_keys=True))
    return 0 if not validation["missing_sibling_holdouts"] and not validation["invalid_sibling_holdouts"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_p1_workload(
        repo_root=args.repo_root,
        descriptor_path=args.workload_file,
        expected_weight_version_id=args.expected_weight_version_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture and validate L0-kernel P1 workload artifacts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_descriptor = subparsers.add_parser("write-heavy-descriptor")
    write_descriptor.add_argument("--thinking-probe-ref", default="reports/thinking-probe-20260424.md")
    write_descriptor.add_argument("--capture-date")
    write_descriptor.set_defaults(func=cmd_write_heavy_descriptor)

    capture_heavy = subparsers.add_parser("capture-heavy")
    capture_heavy.add_argument("--base-url")
    capture_heavy.add_argument("--api-key")
    capture_heavy.add_argument("--model", default="qwen3.5-27b")
    capture_heavy.add_argument("--count", type=int, default=4)
    capture_heavy.add_argument("--weight-version-id", required=True)
    capture_heavy.add_argument("--thinking-probe-ref", default="reports/thinking-probe-20260424.md")
    capture_heavy.add_argument("--capture-date")
    capture_heavy.add_argument("--enable-thinking-override", action="store_true")
    capture_heavy.set_defaults(func=cmd_capture_heavy)

    capture_siblings = subparsers.add_parser("capture-sibling-holdouts")
    capture_siblings.add_argument("--base-url")
    capture_siblings.add_argument("--api-key")
    capture_siblings.add_argument("--model", default="qwen3.5-27b")
    capture_siblings.add_argument("--count", type=int, default=4)
    capture_siblings.add_argument("--weight-version-id", required=True)
    capture_siblings.add_argument("--workload-file", type=Path)
    capture_siblings.add_argument("--capture-date")
    capture_siblings.add_argument("--enable-thinking-override", action="store_true")
    capture_siblings.set_defaults(func=cmd_capture_sibling_holdouts)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--workload-file", type=Path)
    validate.add_argument("--expected-weight-version-id")
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if getattr(args, "workload_file", None) is not None and args.workload_file is not None:
        args.workload_file = args.workload_file if args.workload_file.is_absolute() else args.repo_root / args.workload_file
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

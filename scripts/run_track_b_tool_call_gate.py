#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.metrics import compute_task_metrics, parse_prometheus_text, resolve_metric_schema  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_text_if_present(path: Path, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit]


def _benchmark_context(repo_root: Path, family: str, variant: str) -> dict[str, Any]:
    family_dir = repo_root / "benchmark_blueprints" / "families" / family
    variant_dir = family_dir / "workspace_bundle" / variant
    return {
        "family": family,
        "variant": variant,
        "task_spec_ref": str(family_dir / "task_spec.md"),
        "variant_ref": str(variant_dir),
        "task_spec_excerpt": _read_text_if_present(family_dir / "task_spec.md", limit=2400),
        "agents_excerpt": _read_text_if_present(variant_dir / "AGENTS.md", limit=1200),
    }


def _tool_cases(context: dict[str, Any], count: int) -> list[dict[str, Any]]:
    family = str(context["family"])
    variant = str(context["variant"])
    base = [
        {
            "label": "read_file_agents",
            "tool": {
                "type": "function",
                "name": "read_file",
                "description": "Read a UTF-8 file from the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            "expected_name": "read_file",
            "expected_arguments": {"path": "AGENTS.md"},
            "required_contains": {},
            "instruction": "Call read_file for AGENTS.md.",
        },
        {
            "label": "shell_visible_tests",
            "tool": {
                "type": "function",
                "name": "exec_command",
                "description": "Run a shell command in the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "workdir": {"type": "string"},
                    },
                    "required": ["cmd", "workdir"],
                    "additionalProperties": False,
                },
            },
            "expected_name": "exec_command",
            "expected_arguments": {"cmd": "pytest -q", "workdir": "."},
            "required_contains": {},
            "instruction": "Call exec_command to run pytest -q in workdir .",
        },
        {
            "label": "apply_patch_note",
            "tool": {
                "type": "function",
                "name": "apply_patch",
                "description": "Apply a unified patch to the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"patch": {"type": "string"}},
                    "required": ["patch"],
                    "additionalProperties": False,
                },
            },
            "expected_name": "apply_patch",
            "expected_arguments": {
                "patch": "*** Begin Patch\n*** Add File: artifacts/tool_gate_marker.txt\n+tool gate\n*** End Patch\n"
            },
            "required_contains": {"patch": ["tool_gate_marker", "tool gate"]},
            "instruction": "Call apply_patch with a patch that adds artifacts/tool_gate_marker.txt containing tool gate.",
        },
        {
            "label": "write_report_json",
            "tool": {
                "type": "function",
                "name": "write_file",
                "description": "Write a UTF-8 file in the benchmark workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
            "expected_name": "write_file",
            "expected_arguments": {
                "path": "artifacts/tool_gate.json",
                "content": json.dumps({"family": family, "variant": variant, "status": "checked"}, sort_keys=True),
            },
            "required_contains": {"content": [family, variant, "checked"]},
            "instruction": "Call write_file to write artifacts/tool_gate.json with JSON content containing the family, variant, and checked status.",
        },
    ]
    if count <= len(base):
        return base[:count]
    expanded: list[dict[str, Any]] = []
    while len(expanded) < count:
        for case in base:
            copy = dict(case)
            copy["label"] = f"{case['label']}_{len(expanded) + 1:02d}"
            expanded.append(copy)
            if len(expanded) >= count:
                break
    return expanded


def _arguments_valid(case: dict[str, Any], arguments: Any, *, exact: bool) -> bool:
    if not isinstance(arguments, dict):
        return False
    expected = case["expected_arguments"]
    if exact:
        return arguments == expected
    for key, value in expected.items():
        if key not in arguments:
            return False
        if key not in case.get("required_contains", {}) and arguments.get(key) != value:
            return False
    for key, needles in case.get("required_contains", {}).items():
        value = arguments.get(key)
        if not isinstance(value, str):
            return False
        if not all(needle in value for needle in needles):
            return False
    return True


def _extract_function_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    output = payload.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if isinstance(item, dict) and item.get("type") == "function_call":
            arguments = item.get("arguments")
            parsed_args: Any = None
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_args = None
            return {
                "name": item.get("name"),
                "arguments_raw": arguments,
                "arguments": parsed_args,
                "call_id": item.get("call_id"),
                "status": item.get("status"),
            }
    return None


def _normalized_call(call: Any) -> dict[str, Any] | None:
    if not isinstance(call, dict):
        return None
    return {
        "name": call.get("name"),
        "arguments": call.get("arguments"),
        "status": call.get("status"),
    }


def _calls_match(serial_call: Any, concurrent_call: Any, *, exact_arguments: bool) -> bool:
    if exact_arguments:
        return _normalized_call(serial_call) == _normalized_call(concurrent_call)
    if not isinstance(serial_call, dict) or not isinstance(concurrent_call, dict):
        return False
    return serial_call.get("name") == concurrent_call.get("name") and serial_call.get("status") == concurrent_call.get("status")


def _metrics(metrics_url: str) -> dict[str, float]:
    response = requests.get(metrics_url, timeout=20)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def _metric_delta(before: dict[str, float], after: dict[str, float], *, elapsed_s: float, request_count: int) -> dict[str, Any]:
    schema = resolve_metric_schema(after)
    metrics = compute_task_metrics(before=before, after=after, schema=schema)
    gen_tokens = float(metrics.get("gen_tokens") or 0.0)
    decode_sum_s = float(metrics.get("decode_sum_s") or 0.0)
    prefill_sum_s = float(metrics.get("prefill_sum_s") or 0.0)
    prompt_tokens = float(metrics.get("prompt_tokens") or 0.0)
    return {
        "generation_tokens": gen_tokens,
        "decode_sum_s": decode_sum_s,
        "prefill_sum_s": prefill_sum_s,
        "prompt_tokens": prompt_tokens,
        "decode_tokens_per_s": round(gen_tokens / decode_sum_s, 6) if decode_sum_s > 0 else None,
        "wall_output_tokens_per_s": round(gen_tokens / elapsed_s, 6) if elapsed_s > 0 else None,
        "request_count": request_count,
        "generation_tokens_per_request": round(gen_tokens / max(request_count, 1), 3),
        "prompt_tokens_per_request": round(prompt_tokens / max(request_count, 1), 3),
    }


def _post_case(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    context: dict[str, Any],
    case: dict[str, Any],
    max_output_tokens: int,
    tool_choice_mode: str,
    exact_arguments: bool,
) -> dict[str, Any]:
    prompt = (
        "You are solving an authored Codex benchmark variant. Use exactly the requested tool; "
        "do not answer in prose.\n\n"
        f"Family: {context['family']}\n"
        f"Variant: {context['variant']}\n\n"
        "Task spec excerpt:\n"
        f"{context.get('task_spec_excerpt', '')}\n\n"
        f"Instruction: {case['instruction']}"
    )
    started = time.perf_counter()
    request_payload: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "temperature": 0,
        "tools": [case["tool"]],
    }
    if tool_choice_mode == "forced":
        request_payload["tool_choice"] = {"type": "function", "name": case["expected_name"]}
    elif tool_choice_mode == "auto":
        request_payload["tool_choice"] = "auto"
    else:
        raise RuntimeError(f"unsupported tool_choice_mode: {tool_choice_mode}")
    response = requests.post(
        f"{endpoint.rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {api_key}"},
        json=request_payload,
        timeout=max(120, max_output_tokens * 2),
    )
    wall_s = time.perf_counter() - started
    if response.status_code >= 400:
        body = response.text.strip()
        return {
            "label": case["label"],
            "wall_s": round(wall_s, 6),
            "valid": False,
            "expected_name": case["expected_name"],
            "expected_arguments": case["expected_arguments"],
            "function_call": None,
            "usage": None,
            "response_status": "http_error",
            "http_status": response.status_code,
            "response_body": body[:2000],
        }
    payload = response.json()
    call = _extract_function_call(payload if isinstance(payload, dict) else {})
    valid = (
        isinstance(call, dict)
        and call.get("name") == case["expected_name"]
        and _arguments_valid(case, call.get("arguments"), exact=exact_arguments)
    )
    return {
        "label": case["label"],
        "wall_s": round(wall_s, 6),
        "valid": bool(valid),
        "expected_name": case["expected_name"],
        "expected_arguments": case["expected_arguments"],
        "function_call": call,
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
        "response_status": payload.get("status") if isinstance(payload, dict) else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    requests.get(args.health_url, timeout=20).raise_for_status()
    if args.reset_prefix_cache:
        requests.post(args.reset_prefix_cache_url, headers={"Authorization": f"Bearer {args.api_key}"}, timeout=30).raise_for_status()
    context = _benchmark_context(REPO_ROOT, args.benchmark_family, args.variant)
    cases = _tool_cases(context, args.probe_count)
    before_metrics = _metrics(args.metrics_url) if args.measure_throughput else None
    started = time.monotonic()
    serial = [
        _post_case(
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            context=context,
            case=case,
            max_output_tokens=args.max_output_tokens,
            tool_choice_mode=args.tool_choice_mode,
            exact_arguments=args.exact_arguments,
        )
        for case in cases
    ]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrent_requests)) as executor:
        futures = {
            executor.submit(
                _post_case,
                endpoint=args.endpoint,
                api_key=args.api_key,
                model=args.model,
                context=context,
                case=case,
                max_output_tokens=args.max_output_tokens,
                tool_choice_mode=args.tool_choice_mode,
                exact_arguments=args.exact_arguments,
            ): index
            for index, case in enumerate(cases)
        }
        concurrent: list[dict[str, Any] | None] = [None] * len(cases)
        for future in as_completed(futures):
            concurrent[futures[future]] = future.result()
    elapsed_s = time.monotonic() - started
    after_metrics = _metrics(args.metrics_url) if args.measure_throughput else None
    comparisons = []
    for index, (case, serial_row, concurrent_row) in enumerate(zip(cases, serial, concurrent, strict=True)):
        assert concurrent_row is not None
        serial_call = serial_row.get("function_call")
        concurrent_call = concurrent_row.get("function_call")
        normalized_serial = _normalized_call(serial_call)
        normalized_concurrent = _normalized_call(concurrent_call)
        comparisons.append(
            {
                "index": index,
                "label": case["label"],
                "serial_valid": serial_row["valid"],
                "concurrent_valid": concurrent_row["valid"],
                "match": _calls_match(serial_call, concurrent_call, exact_arguments=args.exact_arguments),
                "normalized_serial": normalized_serial,
                "normalized_concurrent": normalized_concurrent,
                "serial": serial_row,
                "concurrent": concurrent_row,
            }
        )
    pass_count = sum(
        1
        for row in comparisons
        if row["serial_valid"] and row["concurrent_valid"] and row["match"]
    )
    pass_rate = pass_count / len(comparisons) if comparisons else 0.0
    result = {
        "schema": "lumo.track_b.tool_call_correctness_gate.v1",
        "measured_at": _now(),
        "suite": args.suite,
        "model": args.model,
        "endpoint": args.endpoint,
        "benchmark_family": args.benchmark_family,
        "variant": args.variant,
        "tool_choice_mode": args.tool_choice_mode,
        "exact_arguments": args.exact_arguments,
        "probe_count": len(comparisons),
        "concurrent_requests": args.concurrent_requests,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "min_pass_rate": args.min_pass_rate,
        "pass": bool(comparisons) and pass_rate >= args.min_pass_rate,
        "comparisons": comparisons,
    }
    if before_metrics is not None and after_metrics is not None:
        metrics_delta = _metric_delta(before_metrics, after_metrics, elapsed_s=elapsed_s, request_count=len(cases) * 2)
        decode_tps = metrics_delta["decode_tokens_per_s"]
        result["metrics_delta"] = metrics_delta
        result["target_decode_tps"] = args.target_decode_tps
        result["pass_target_gate"] = bool(result["pass"] and decode_tps is not None and decode_tps >= args.target_decode_tps)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tool-call-inclusive correctness gate for Track B spec decode candidates.")
    parser.add_argument("--suite", choices=["b1", "b2", "b3"], default="b2")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--benchmark-family", default="policy-aware-request-resolution")
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--concurrent-requests", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--tool-choice-mode", choices=["forced", "auto"], default="forced")
    parser.add_argument("--exact-arguments", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--measure-throughput", action="store_true")
    parser.add_argument("--target-decode-tps", type=float, default=30.0)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.probe_count < 1:
        raise RuntimeError("--probe-count must be >= 1")
    if args.concurrent_requests < 1:
        raise RuntimeError("--concurrent-requests must be >= 1")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.measure_throughput:
        return 0 if result.get("pass_target_gate") else 2
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

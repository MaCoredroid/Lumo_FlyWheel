#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


CASE_A_INPUT = "Summarize: AI is useful."
CASE_B_INPUT = "Prove that sqrt(2) is irrational. Show every step."


def _responses_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/responses"


def _case_a_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": CASE_A_INPUT,
        "max_output_tokens": 2048,
    }


def _case_b_payload(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": CASE_B_INPUT,
        "max_output_tokens": 8192,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def _int_usage(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _int_nested_usage(usage: dict[str, Any], parent_key: str, key: str) -> int:
    nested = usage.get(parent_key, {})
    if not isinstance(nested, dict):
        return 0
    return _int_usage(nested, key)


def _has_reasoning_output(payload: dict[str, Any]) -> bool:
    output = payload.get("output")
    if not isinstance(output, list):
        return False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "reasoning":
            return True
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if str(part.get("type", "")).startswith("reasoning") and (part.get("text") or part.get("reasoning_text")):
                return True
    return False


def _normalize_usage(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    output_tokens = _int_usage(usage, "output_tokens")
    reasoning_tokens = _int_usage(usage, "reasoning_tokens") or _int_nested_usage(
        usage,
        "output_tokens_details",
        "reasoning_tokens",
    )
    if reasoning_tokens <= 0 and output_tokens > 0 and _has_reasoning_output(payload):
        reasoning_tokens = output_tokens
    return {
        "input_tokens": _int_usage(usage, "input_tokens"),
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": _int_usage(usage, "total_tokens"),
    }


def classify_outcome(case_a_reasoning_tokens: int, case_b_reasoning_tokens: int) -> tuple[str, str]:
    a_positive = case_a_reasoning_tokens > 0
    b_positive = case_b_reasoning_tokens > 0
    if not a_positive and b_positive:
        return (
            "row-1",
            "Pipeline does not propagate thinking by default; explicit thinking override works.",
        )
    if not a_positive and not b_positive:
        return (
            "row-2",
            "Blocker: thinking is off at the serve layer even with explicit override.",
        )
    if a_positive and b_positive:
        return (
            "row-3",
            "Thinking fires by default and with explicit override.",
        )
    return (
        "bug",
        "Unexpected: explicit thinking override produced fewer reasoning tokens than the default call.",
    )


def _post_probe(
    *,
    transport: Callable[..., Any],
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    response = transport(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Responses probe returned a non-object JSON payload")
    return data


def _write_report(
    *,
    report_path: Path,
    capture_date: datetime,
    base_url: str,
    model: str,
    case_a_payload: dict[str, Any],
    case_b_payload: dict[str, Any],
    case_a_usage: dict[str, int],
    case_b_usage: dict[str, int],
    outcome: str,
    diagnosis: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Serving Thinking Probe",
                "",
                f"- capture_date: {capture_date.isoformat().replace('+00:00', 'Z')}",
                f"- base_url: {base_url}",
                f"- model: {model}",
                f"- outcome: {outcome}",
                f"- diagnosis: {diagnosis}",
                "",
                "## Inputs",
                "",
                "### Case A",
                "",
                "```json",
                json.dumps(case_a_payload, indent=2, sort_keys=True),
                "```",
                "",
                "### Case B",
                "",
                "```json",
                json.dumps(case_b_payload, indent=2, sort_keys=True),
                "```",
                "",
                "## Usage Summary",
                "",
                "| case | input_tokens | output_tokens | reasoning_tokens | total_tokens |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| A | {case_a_usage['input_tokens']} | {case_a_usage['output_tokens']} | "
                    f"{case_a_usage['reasoning_tokens']} | {case_a_usage['total_tokens']} |"
                ),
                (
                    f"| B | {case_b_usage['input_tokens']} | {case_b_usage['output_tokens']} | "
                    f"{case_b_usage['reasoning_tokens']} | {case_b_usage['total_tokens']} |"
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_probe(
    *,
    base_url: str,
    api_key: str,
    model: str,
    reports_dir: Path,
    capture_date: datetime | None = None,
    timeout: int = 120,
    transport: Callable[..., Any] = requests.post,
) -> dict[str, Any]:
    captured_at = capture_date or datetime.now(UTC)
    url = _responses_url(base_url)
    case_a_request = _case_a_payload(model)
    case_b_request = _case_b_payload(model)
    case_a_response = _post_probe(
        transport=transport,
        url=url,
        api_key=api_key,
        payload=case_a_request,
        timeout=timeout,
    )
    case_b_response = _post_probe(
        transport=transport,
        url=url,
        api_key=api_key,
        payload=case_b_request,
        timeout=timeout,
    )
    case_a_usage = _normalize_usage(case_a_response)
    case_b_usage = _normalize_usage(case_b_response)
    outcome, diagnosis = classify_outcome(
        case_a_usage["reasoning_tokens"],
        case_b_usage["reasoning_tokens"],
    )
    report_path = reports_dir / f"thinking-probe-{captured_at.strftime('%Y%m%d')}.md"
    _write_report(
        report_path=report_path,
        capture_date=captured_at,
        base_url=base_url,
        model=model,
        case_a_payload=case_a_request,
        case_b_payload=case_b_request,
        case_a_usage=case_a_usage,
        case_b_usage=case_b_usage,
        outcome=outcome,
        diagnosis=diagnosis,
    )
    return {
        "report_path": str(report_path),
        "capture_date": captured_at.isoformat().replace("+00:00", "Z"),
        "outcome": outcome,
        "diagnosis": diagnosis,
        "usage": {"A": case_a_usage, "B": case_b_usage},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe serving-layer thinking behavior via /v1/responses.")
    parser.add_argument("--base-url", required=True, help="Responses API base URL, normally ending in /v1.")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("--api-key or VLLM_API_KEY is required")
    result = run_probe(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        reports_dir=args.reports_dir,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

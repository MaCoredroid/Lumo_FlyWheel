#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_BALANCED_MATRIX: tuple[tuple[str, str], ...] = (
    ("report-cli-markdown-evolution", "inventory-ops"),
    ("report-cli-markdown-evolution", "incident-triage"),
    ("normalizer-api-migration", "alert-routing"),
    ("normalizer-api-migration", "billing-ledger"),
    ("ci-config-coverage-drift", "inventory-gate"),
    ("ci-config-coverage-drift", "payments-gate"),
    ("alert-dedupe-investigation", "payments-oncall"),
    ("alert-dedupe-investigation", "search-oncall"),
    ("owner-field-cross-layer", "project-board"),
    ("owner-field-cross-layer", "warehouse-queue"),
)


def parse_variant_ref(value: str) -> tuple[str, str]:
    family, slash, variant = value.partition("/")
    if not slash or not family or not variant:
        raise argparse.ArgumentTypeError(
            f"variant references must use '<family>/<variant>' form, got: {value}"
        )
    return family, variant


def _run_live_variant(
    *,
    repo_root: Path,
    family: str,
    variant: str,
    timeout_seconds: int,
    keep_artifacts: bool,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_live_codex_long_task.py"),
        "--repo-root",
        str(repo_root),
        "--family",
        family,
        "--variant",
        variant,
        "--timeout-seconds",
        str(timeout_seconds),
        "--json",
    ]
    if keep_artifacts:
        command.append("--keep-artifacts")
    result = subprocess.run(  # noqa: S603
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stdout or result.stderr or f"live run failed for {family}/{variant}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise SystemExit(f"live run did not return a JSON object for {family}/{variant}")
    return payload


def summarize_results(
    matrix_name: str,
    results: list[dict[str, object]],
) -> dict[str, object]:
    countable = [result for result in results if bool(result.get("countable"))]
    infra_failures = [result for result in results if bool(result.get("infra_failure"))]
    passes = [result for result in countable if bool(result.get("pass"))]
    adjusted_pass_rate = (
        len(passes) / len(countable)
        if countable
        else None
    )
    return {
        "matrix_name": matrix_name,
        "total_runs": len(results),
        "countable_runs": len(countable),
        "infra_failures": len(infra_failures),
        "passes": len(passes),
        "adjusted_pass_rate": adjusted_pass_rate,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a balanced live Codex-Long matrix and report exclusion-aware pass rate."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        type=parse_variant_ref,
        help="Optional '<family>/<variant>' entry. Defaults to the balanced two-per-family matrix.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    matrix = tuple(args.variants or DEFAULT_BALANCED_MATRIX)
    results: list[dict[str, object]] = []
    for family, variant in matrix:
        results.append(
            _run_live_variant(
                repo_root=repo_root,
                family=family,
                variant=variant,
                timeout_seconds=args.timeout_seconds,
                keep_artifacts=args.keep_artifacts,
            )
        )

    summary = summarize_results(
        "balanced-two-per-family" if matrix == DEFAULT_BALANCED_MATRIX else "custom",
        results,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        rate = summary["adjusted_pass_rate"]
        rate_text = "n/a" if rate is None else f"{summary['passes']}/{summary['countable_runs']} = {rate:.0%}"
        print(
            f"matrix={summary['matrix_name']} countable={summary['countable_runs']} "
            f"infra_failures={summary['infra_failures']} adjusted_pass_rate={rate_text}"
        )
        for result in results:
            print(
                f"{result['family']}/{result['variant']}: "
                f"countable={result['countable']} pass={result['pass']} infra_failure={result['infra_failure']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

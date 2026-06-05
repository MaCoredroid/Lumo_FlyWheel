#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_superset_gate import (  # noqa: E402
    AcceptanceEvent,
    GateReport,
    evaluate_path0_sequence_gate,
    evaluate_strict_win_gate,
    evaluate_superset_hard_gate,
    evaluate_total_acceptance_gate,
    load_spec_trace,
    load_tree_accept_trace,
)


DEFAULT_CATERPILLAR_TREE = (
    "[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), "
    "(0, 0, 0, 0, 0), (0, 1), (0, 0, 1), "
    "(0, 0, 0, 1), (0, 0, 0, 0, 1)]"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, GateReport):
        return {
            "passed": value.passed,
            "violations": value.violations,
            "metrics": _jsonable(value.metrics),
        }
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _path0_proxy_native(tree_events: list[AcceptanceEvent]) -> list[AcceptanceEvent]:
    """Create a native-like sequence equal to the tree's own path0.

    ``evaluate_superset_hard_gate`` enforces both ``tree >= path0`` and
    ``path0 >= native`` because it also serves paired external diagnostics. For
    the internal structural gate requested here, native is intentionally the
    tree path0 itself so only ``tree >= path0`` can fail.
    """
    proxy: list[AcceptanceEvent] = []
    for event in tree_events:
        if event.path0_len is None:
            raise ValueError(f"tree event missing path0_len: {event}")
        proxy.append(
            AcceptanceEvent(
                request_id=event.request_id,
                step_index=event.step_index,
                accepted_len=int(event.path0_len),
                path0_len=int(event.path0_len),
                timestamp=event.timestamp,
                source="tree_path0_proxy",
            )
        )
    return proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FR10 superset/lossless acceptance gates on tree/native traces."
    )
    parser.add_argument("--tree-path-lcp", required=True, help="tree_path_lcp_max.jsonl")
    parser.add_argument(
        "--native-spec-trace",
        help="paired native per_req_spec_trace.jsonl; enables gates 2-4",
    )
    parser.add_argument(
        "--speculative-token-tree",
        default=DEFAULT_CATERPILLAR_TREE,
        help="SPEC_CONFIG speculative_token_tree string",
    )
    parser.add_argument("--out", required=True, help="output JSON report")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tree_events = load_tree_accept_trace(
        args.tree_path_lcp,
        speculative_token_tree=args.speculative_token_tree,
        source="tree",
    )
    if not tree_events:
        raise SystemExit(f"no tree acceptance events loaded from {args.tree_path_lcp}")

    reports: dict[str, GateReport] = {}
    reports["superset_hard_internal"] = evaluate_superset_hard_gate(
        native_events=_path0_proxy_native(tree_events),
        tree_events=tree_events,
    )

    native_events: list[AcceptanceEvent] = []
    if args.native_spec_trace:
        native_events = load_spec_trace(args.native_spec_trace, source="native")
        if not native_events:
            raise SystemExit(f"no native acceptance events loaded from {args.native_spec_trace}")
        reports["path0_sequence"] = evaluate_path0_sequence_gate(
            native_events=native_events,
            tree_events=tree_events,
        )
        reports["total_acceptance"] = evaluate_total_acceptance_gate(
            native_events=native_events,
            tree_events=tree_events,
        )
        reports["strict_win"] = evaluate_strict_win_gate(
            native_events=native_events,
            tree_events=tree_events,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )

    payload = {
        "schema": "fr10.superset_gate_report.v1",
        "tree_path_lcp": str(args.tree_path_lcp),
        "native_spec_trace": str(args.native_spec_trace) if args.native_spec_trace else None,
        "speculative_token_tree": args.speculative_token_tree,
        "tree_events": len(tree_events),
        "native_events": len(native_events),
        "gates": _jsonable(reports),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for name, report in reports.items():
        status = "PASS" if report.passed else "FAIL"
        print(f"{name}: {status} {json.dumps(report.metrics, sort_keys=True)}")
        if report.violations:
            print(f"  first_violation: {report.violations[0]}")
    return 0 if all(report.passed for report in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

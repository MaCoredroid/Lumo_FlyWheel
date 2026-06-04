"""FR10 external acceptance-superset gate.

The existing tree LCP checker proves an internal property: the tree winner is
at least as long as the tree verifier's own path0. FR10 also needs an external
gate against native MTP because a GDN-hybrid tree can be internally consistent
while degrading the native top-1 chain. This module compares tree path0 against
native MTP and then compares total accepted tokens for the sampled speed gate.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class AcceptanceEvent:
    request_id: str
    step_index: int
    accepted_len: int
    path0_len: int | None = None
    accepted_node_ids: tuple[int, ...] = ()
    emitted_tokens: tuple[int, ...] = ()
    draft_token_ids: tuple[int, ...] = ()
    timestamp: float | None = None
    source: str = ""


@dataclass(frozen=True)
class GateReport:
    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def runtime_tree_choices(speculative_token_tree: str | None) -> list[tuple[int, ...]]:
    if not speculative_token_tree:
        return []
    choices = ast.literal_eval(speculative_token_tree)
    return sorted((tuple(int(part) for part in choice) for choice in choices),
                  key=lambda item: (len(item), item))


def runtime_node_map(speculative_token_tree: str | None) -> dict[int, tuple[int, ...]]:
    return {idx: path for idx, path in enumerate(runtime_tree_choices(speculative_token_tree))}


def path0_runtime_nodes(speculative_token_tree: str | None) -> tuple[int, ...]:
    node_map = runtime_node_map(speculative_token_tree)
    return tuple(
        idx
        for idx, path in sorted(node_map.items())
        if path and all(part == 0 for part in path)
    )


def common_prefix_len(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for a, b in zip(left, right):
        if int(a) != int(b):
            break
        count += 1
    return count


def infer_path0_len(row: dict[str, Any], path0_nodes: Sequence[int]) -> int | None:
    if "path0_lcp" in row:
        return int(row["path0_lcp"])
    accepted_nodes = tuple(int(node) for node in row.get("accepted_node_ids") or ())
    if not path0_nodes:
        return None
    return common_prefix_len(accepted_nodes, path0_nodes)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def load_spec_trace(path: str | Path, *, source: str = "native") -> list[AcceptanceEvent]:
    """Load per_req_spec_trace.jsonl-style rows.

    The trace has one row per spec event and records the accepted count as
    ``acc``. It does not carry tree topology, so ``path0_len`` is the accepted
    count for native/linear MTP.
    """
    events: list[AcceptanceEvent] = []
    request_steps: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        rid = str(row.get("rid") or row.get("request_id") or row.get("req_index") or "unknown")
        step = request_steps[rid]
        request_steps[rid] += 1
        acc = int(row.get("acc", row.get("accepted_len", 0)) or 0)
        events.append(
            AcceptanceEvent(
                request_id=rid,
                step_index=step,
                accepted_len=acc,
                path0_len=acc,
                timestamp=float(row["ts"]) if row.get("ts") is not None else None,
                source=source,
            )
        )
    return events


def load_tree_accept_trace(
    path: str | Path,
    *,
    speculative_token_tree: str | None,
    source: str = "tree",
) -> list[AcceptanceEvent]:
    """Load tree_path_lcp_max/tree_sample_accept rows.

    ``path0_len`` is read from ``path0_lcp`` when available. Sampled FR10 rows
    currently emit accepted node ids; for those rows we infer path0 length as
    the common prefix between the accepted path and the runtime path0 chain.
    """
    path0_nodes = path0_runtime_nodes(speculative_token_tree)
    events: list[AcceptanceEvent] = []
    request_steps: Counter[str] = Counter()
    for row in _iter_jsonl(path):
        event = str(row.get("event") or "")
        if event not in {"tree_sample_accept", "tree_path_lcp_max"}:
            continue
        rid = str(row.get("rid") or row.get("request_id") or row.get("req_index") or "unknown")
        step = request_steps[rid]
        request_steps[rid] += 1
        accepted_nodes = tuple(int(node) for node in row.get("accepted_node_ids") or ())
        events.append(
            AcceptanceEvent(
                request_id=rid,
                step_index=step,
                accepted_len=int(row.get("accepted_len", 0) or 0),
                path0_len=infer_path0_len(row, path0_nodes),
                accepted_node_ids=accepted_nodes,
                emitted_tokens=tuple(int(x) for x in row.get("emitted_tokens") or ()),
                draft_token_ids=tuple(int(x) for x in row.get("draft_token_ids") or ()),
                timestamp=float(row["ts"]) if row.get("ts") is not None else None,
                source=source,
            )
        )
    return events


def _sequence(events: Sequence[AcceptanceEvent], *, field: str) -> list[int]:
    values: list[int] = []
    for event in sorted(events, key=lambda item: (item.request_id, item.step_index)):
        value = getattr(event, field)
        if value is None:
            raise ValueError(f"{field} is missing for {event}")
        values.append(int(value))
    return values


def _first_diff(left: Sequence[int], right: Sequence[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def evaluate_path0_sequence_gate(
    *,
    native_events: Sequence[AcceptanceEvent],
    tree_events: Sequence[AcceptanceEvent],
) -> GateReport:
    """Tier 1: deterministic native MTP chain must equal tree path0."""
    native = _sequence(native_events, field="accepted_len")
    tree_path0 = _sequence(tree_events, field="path0_len")
    violations: list[str] = []
    first_diff = _first_diff(native, tree_path0)
    if first_diff is not None:
        n_val = native[first_diff] if first_diff < len(native) else None
        t_val = tree_path0[first_diff] if first_diff < len(tree_path0) else None
        violations.append(
            "path0 accepted sequence differs from native MTP at "
            f"event_index={first_diff}: native={n_val} tree_path0={t_val}"
        )
    return GateReport(
        passed=not violations,
        violations=violations,
        metrics={
            "native_events": len(native),
            "tree_events": len(tree_path0),
            "native_total": sum(native),
            "tree_path0_total": sum(tree_path0),
            "native_avg": sum(native) / len(native) if native else 0.0,
            "tree_path0_avg": sum(tree_path0) / len(tree_path0) if tree_path0 else 0.0,
            "first_diff": first_diff,
        },
    )


def evaluate_total_acceptance_gate(
    *,
    native_events: Sequence[AcceptanceEvent],
    tree_events: Sequence[AcceptanceEvent],
    min_delta: float = -0.05,
) -> GateReport:
    """Tier 2: sampled tree total acceptance must not underperform native MTP."""
    native = _sequence(native_events, field="accepted_len")
    tree = _sequence(tree_events, field="accepted_len")
    native_avg = sum(native) / len(native) if native else 0.0
    tree_avg = sum(tree) / len(tree) if tree else 0.0
    delta = tree_avg - native_avg
    violations: list[str] = []
    if not native or not tree:
        violations.append("missing native or tree acceptance events")
    elif delta < min_delta:
        violations.append(
            f"tree accepted/event degraded vs native: delta={delta:.6f} "
            f"below allowed {min_delta:.6f}"
        )
    return GateReport(
        passed=not violations,
        violations=violations,
        metrics={
            "native_events": len(native),
            "tree_events": len(tree),
            "native_avg": native_avg,
            "tree_avg": tree_avg,
            "tree_minus_native_avg": delta,
            "min_delta": min_delta,
        },
    )


def acceptance_by_depth(events: Sequence[AcceptanceEvent], *, field: str) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for event in events:
        value = getattr(event, field)
        if value is None:
            continue
        for depth in range(int(value)):
            counts[depth] += 1
    return dict(sorted(counts.items()))


def diagnose_spine_degradation(
    *,
    native_events: Sequence[AcceptanceEvent],
    tree_events: Sequence[AcceptanceEvent],
) -> GateReport:
    """Classify a path0 degradation when draft-token diagnostics are present.

    If tree path0 drafts differ from native drafts at the first degraded depth,
    the likely source is draft-side recurrent-state contamination. If drafts
    match but accepted length falls, the likely source is commit/position
    bookkeeping above the verifier.
    """
    native_seq = _sequence(native_events, field="accepted_len")
    tree_seq = _sequence(tree_events, field="path0_len")
    first_degrade = None
    for idx, (native_len, tree_len) in enumerate(zip(native_seq, tree_seq)):
        if tree_len < native_len:
            first_degrade = idx
            break
    if first_degrade is None:
        return GateReport(
            passed=True,
            metrics={
                "classification": "no_degradation",
                "native_by_depth": acceptance_by_depth(native_events, field="accepted_len"),
                "tree_path0_by_depth": acceptance_by_depth(tree_events, field="path0_len"),
            },
        )

    native_event = sorted(native_events, key=lambda item: (item.request_id, item.step_index))[
        first_degrade
    ]
    tree_event = sorted(tree_events, key=lambda item: (item.request_id, item.step_index))[
        first_degrade
    ]
    classification = "unknown_missing_draft_tokens"
    native_drafts = native_event.draft_token_ids
    tree_drafts = tree_event.draft_token_ids
    if native_drafts and tree_drafts:
        depth = int(tree_seq[first_degrade])
        native_token = native_drafts[depth] if depth < len(native_drafts) else None
        tree_token = tree_drafts[depth] if depth < len(tree_drafts) else None
        classification = (
            "draft_side_recurrent_state_corruption"
            if native_token != tree_token
            else "commit_or_position_bookkeeping_bug"
        )
    return GateReport(
        passed=False,
        violations=[
            f"path0 degradation at event_index={first_degrade}: "
            f"native={native_seq[first_degrade]} tree_path0={tree_seq[first_degrade]} "
            f"classification={classification}"
        ],
        metrics={
            "classification": classification,
            "first_degrade_event_index": first_degrade,
            "native_len": native_seq[first_degrade],
            "tree_path0_len": tree_seq[first_degrade],
            "native_draft_tokens": list(native_drafts),
            "tree_draft_tokens": list(tree_drafts),
            "native_by_depth": acceptance_by_depth(native_events, field="accepted_len"),
            "tree_path0_by_depth": acceptance_by_depth(tree_events, field="path0_len"),
        },
    )


def write_report(path: str | Path, report: GateReport) -> None:
    payload = {
        "passed": report.passed,
        "violations": report.violations,
        "metrics": report.metrics,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


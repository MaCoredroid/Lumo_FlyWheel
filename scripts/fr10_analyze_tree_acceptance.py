#!/usr/bin/env python3
"""Analyze FR10 tree branch acceptance from committed runtime traces."""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield line_no, json.loads(raw)
            except json.JSONDecodeError as exc:
                yield line_no, {"_parse_error": str(exc)}


def _parse_spec_metrics(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "drafts": 0.0,
        "accepted_total": 0.0,
        "accepted_per_pos": {},
    }
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total"):
            out["drafts"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            out["accepted_total"] = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            match = re.search(r'position="(\d+)"', line)
            if match:
                out["accepted_per_pos"][int(match.group(1))] = float(line.rsplit(" ", 1)[1])
    return out


def _metric_delta(before: Path, after: Path) -> dict[str, Any]:
    pre = _parse_spec_metrics(before.read_text(encoding="utf-8"))
    post = _parse_spec_metrics(after.read_text(encoding="utf-8"))
    drafts = post["drafts"] - pre["drafts"]
    accepted_total = post["accepted_total"] - pre["accepted_total"]
    per_pos_counts = {}
    per_pos_rates = {}
    max_pos = max([*pre["accepted_per_pos"].keys(), *post["accepted_per_pos"].keys(), 0])
    for pos in range(max_pos + 1):
        count = post["accepted_per_pos"].get(pos, 0.0) - pre["accepted_per_pos"].get(pos, 0.0)
        per_pos_counts[str(pos)] = count
        per_pos_rates[str(pos)] = count / drafts if drafts else None
    return {
        "draft_events": drafts,
        "accepted_total": accepted_total,
        "accepted_per_draft_event": accepted_total / drafts if drafts else None,
        "per_position_counts": per_pos_counts,
        "per_position_rates": per_pos_rates,
    }


def _root_from_path(row: dict[str, Any]) -> int | None:
    accepted = row.get("accepted_node_ids") or []
    if accepted:
        return int(accepted[0])
    root = row.get("accepted_root")
    return int(root) if root is not None else None


def _runtime_tree_choices(speculative_token_tree: str | None) -> list[tuple[int, ...]]:
    if not speculative_token_tree:
        return []
    choices = ast.literal_eval(speculative_token_tree)
    return sorted((tuple(choice) for choice in choices), key=lambda t: (len(t), t))


def _node_map(speculative_token_tree: str | None) -> dict[int, tuple[int, ...]]:
    return {idx: path for idx, path in enumerate(_runtime_tree_choices(speculative_token_tree))}


def _is_branch_path(path: tuple[int, ...]) -> bool:
    return any(int(part) != 0 for part in path)


def analyze_trace(path: Path, speculative_token_tree: str | None = None) -> dict[str, Any]:
    node_map = _node_map(speculative_token_tree)
    branch_nodes = {idx for idx, tree_path in node_map.items() if _is_branch_path(tree_path)}
    rows = 0
    parse_errors = 0
    accepted_events = 0
    branch_event_count = 0
    branch_token_count = 0
    accepted_token_total = 0
    root_events: Counter[str] = Counter()
    root_tokens: Counter[str] = Counter()
    node_usage: Counter[str] = Counter()
    branch_node_usage: Counter[str] = Counter()
    path_counts: Counter[str] = Counter()
    accepted_len_counts: Counter[str] = Counter()
    events_by_type: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for line_no, row in _iter_jsonl(path) or []:
        rows += 1
        if "_parse_error" in row:
            parse_errors += 1
            if len(examples) < 5:
                examples.append({"line": line_no, "error": row["_parse_error"]})
            continue
        event = str(row.get("event") or "unknown")
        events_by_type[event] += 1
        if event not in {"tree_sample_accept", "tree_path_lcp_max"}:
            continue
        accepted_len = int(row.get("accepted_len", 0) or 0)
        accepted_nodes = [int(x) for x in (row.get("accepted_node_ids") or [])]
        path_counts[str(accepted_nodes)] += 1
        for node in accepted_nodes:
            node_key = str(node)
            node_usage[node_key] += 1
            if node in branch_nodes:
                branch_node_usage[node_key] += 1
        accepted_branch_nodes = [node for node in accepted_nodes if node in branch_nodes]
        if accepted_branch_nodes:
            branch_event_count += 1
            branch_token_count += len(accepted_branch_nodes)
        accepted_len_counts[str(accepted_len)] += 1
        accepted_token_total += accepted_len
        if accepted_len <= 0:
            root_events["none"] += 1
            continue
        accepted_events += 1
        root = _root_from_path(row)
        key = str(root) if root is not None else "unknown"
        root_events[key] += 1
        root_tokens[key] += accepted_len
        if root not in (0, None) and len(examples) < 5:
            examples.append({
                "line": line_no,
                "event": event,
                "accepted_len": accepted_len,
                "accepted_node_ids": accepted_nodes,
                "emitted_tokens": row.get("emitted_tokens"),
            })

    runtime_tree_node_map = {
        str(idx): list(tree_path) for idx, tree_path in sorted(node_map.items())
    }
    branch_node_labels = {
        str(idx): list(node_map[idx]) for idx in sorted(branch_nodes)
    }
    return {
        "trace": str(path),
        "rows": rows,
        "parse_errors": parse_errors,
        "events_by_type": dict(sorted(events_by_type.items())),
        "runtime_tree_node_map": runtime_tree_node_map,
        "branch_node_labels": branch_node_labels,
        "accepted_events": accepted_events,
        "branch_event_count": branch_event_count,
        "branch_token_count": branch_token_count,
        "accepted_token_total": accepted_token_total,
        "accepted_len_counts": dict(sorted(accepted_len_counts.items(), key=lambda kv: int(kv[0]))),
        "accepted_path_counts": dict(path_counts.most_common()),
        "node_usage": dict(sorted(node_usage.items(), key=lambda kv: int(kv[0]))),
        "branch_node_usage": dict(sorted(branch_node_usage.items(), key=lambda kv: int(kv[0]))),
        "branch_node_usage_labeled": {
            str(node): {
                "tree_path": list(node_map[int(node)]),
                "count": count,
            }
            for node, count in sorted(branch_node_usage.items(), key=lambda kv: int(kv[0]))
        },
        "branch_event_fraction_of_all_rows": (
            branch_event_count / rows if rows else None
        ),
        "branch_event_fraction_of_accepted_events": (
            branch_event_count / accepted_events if accepted_events else None
        ),
        "branch_token_fraction_of_accepted_tokens": (
            branch_token_count / accepted_token_total if accepted_token_total else None
        ),
        "root_event_counts": dict(sorted(root_events.items())),
        "root_token_counts": dict(sorted(root_tokens.items())),
        "branch1_event_fraction_of_accepted": (
            root_events.get("1", 0) / accepted_events if accepted_events else None
        ),
        "branch1_token_fraction_of_accepted": (
            root_tokens.get("1", 0) / accepted_token_total if accepted_token_total else None
        ),
        "path0_event_fraction_of_accepted": (
            root_events.get("0", 0) / accepted_events if accepted_events else None
        ),
        "path0_token_fraction_of_accepted": (
            root_tokens.get("0", 0) / accepted_token_total if accepted_token_total else None
        ),
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metrics-before", type=Path, required=True)
    parser.add_argument("--metrics-after", type=Path, required=True)
    parser.add_argument("--speculative-token-tree", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = {
        "schema": "fr10.tree_acceptance_cost_gate.v1",
        "trace_summary": analyze_trace(
            args.trace,
            speculative_token_tree=args.speculative_token_tree,
        ),
        "spec_counter_delta": _metric_delta(args.metrics_before, args.metrics_after),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

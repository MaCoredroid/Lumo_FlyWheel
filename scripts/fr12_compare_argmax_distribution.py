#!/usr/bin/env python3
"""Compare many tree/native spine logit captures.

The lossless gate is token-level and distributional.  This script matches
tree/native draft events by the path0 draft-token tuple, then reports argmax
mismatch, the explicit one-depth lag pattern, and probability-distribution
distance on the matched rows.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class EventRecord:
    capture_path: Path
    capture_index: int
    request_index: int
    mode: str
    has_tree_parent_indices: bool
    draft_tokens: tuple[int, ...]
    entries: tuple[int, ...]
    target_rows: tuple[int, ...]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _paths(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches:
            p = Path(pattern)
            if p.exists():
                matches = [str(p)]
        for raw in sorted(matches):
            p = Path(raw)
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def _counts(capture: dict[str, Any]) -> list[int]:
    return [int(x) for x in capture["num_draft_tokens"]]


def _request_start(counts: list[int], req: int) -> int:
    return int(sum(counts[:req]))


def _tree_path_entries(capture: dict[str, Any], req: int, depth: int) -> list[int]:
    counts = _counts(capture)
    start = _request_start(counts, req)
    count = counts[req]
    if count <= 0:
        return []
    parents = capture.get("tree_parent_indices")
    if parents is None:
        raise RuntimeError("tree capture is missing tree_parent_indices")
    local_parents = [int(x) for x in parents[start : start + count].tolist()]
    local_entries: list[int] = []
    current = 0
    local_entries.append(current)
    while len(local_entries) < min(depth, count):
        children = [idx for idx, parent in enumerate(local_parents) if parent == current]
        if not children:
            break
        current = children[0]
        local_entries.append(current)
    return [start + idx for idx in local_entries[:depth]]


def _native_path_entries(capture: dict[str, Any], req: int, depth: int) -> list[int]:
    counts = _counts(capture)
    start = _request_start(counts, req)
    count = counts[req]
    return list(range(start, start + min(depth, count)))


def _records_for_capture(path: Path, *, tree: bool, depth: int) -> list[EventRecord]:
    capture = _load(path)
    counts = _counts(capture)
    target_rows_all = capture["target_logits_indices"]
    draft_all = capture["draft_token_ids"]
    records: list[EventRecord] = []
    for req, count in enumerate(counts):
        if count <= 0:
            continue
        entries = (
            _tree_path_entries(capture, req, depth)
            if tree
            else _native_path_entries(capture, req, depth)
        )
        if len(entries) < depth:
            continue
        draft_tokens = tuple(int(draft_all[e].item()) for e in entries[:depth])
        target_rows = tuple(int(target_rows_all[e].item()) for e in entries[:depth])
        records.append(
            EventRecord(
                capture_path=path,
                capture_index=int(capture.get("capture_call_index", -1)),
                request_index=int(req),
                mode=str(capture.get("mode")),
                has_tree_parent_indices=bool(capture.get("has_tree_parent_indices")),
                draft_tokens=draft_tokens,
                entries=tuple(int(e) for e in entries[:depth]),
                target_rows=target_rows,
            )
        )
    return records


def _records(patterns: list[str], *, tree: bool, depth: int) -> tuple[list[EventRecord], dict[str, Any]]:
    paths = _paths(patterns)
    records: list[EventRecord] = []
    for path in paths:
        records.extend(_records_for_capture(path, tree=tree, depth=depth))
    deduped: list[EventRecord] = []
    seen: set[tuple[str, int, int, tuple[int, ...]]] = set()
    for rec in records:
        key = (rec.capture_path.name, rec.capture_index, rec.request_index, rec.draft_tokens)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)
    return deduped, {
        "paths": [str(p) for p in paths],
        "captures": len(paths),
        "events": len(deduped),
    }


def _match_records(
    tree_records: list[EventRecord], native_records: list[EventRecord]
) -> tuple[list[tuple[EventRecord, EventRecord]], list[EventRecord], list[EventRecord]]:
    native_by_tokens: dict[tuple[int, ...], collections.deque[EventRecord]] = {}
    for rec in native_records:
        native_by_tokens.setdefault(rec.draft_tokens, collections.deque()).append(rec)
    matches: list[tuple[EventRecord, EventRecord]] = []
    unmatched_tree: list[EventRecord] = []
    for tree_rec in tree_records:
        queue = native_by_tokens.get(tree_rec.draft_tokens)
        if queue:
            matches.append((tree_rec, queue.popleft()))
        else:
            unmatched_tree.append(tree_rec)
    unmatched_native = [rec for queue in native_by_tokens.values() for rec in queue]
    return matches, unmatched_tree, unmatched_native


def _row_metrics(tree_logits: torch.Tensor, native_logits: torch.Tensor) -> dict[str, float | int | bool]:
    tree_logits = tree_logits.to(torch.float32)
    native_logits = native_logits.to(torch.float32)
    tree_argmax = int(torch.argmax(tree_logits).item())
    native_argmax = int(torch.argmax(native_logits).item())
    tree_logp = torch.log_softmax(tree_logits, dim=-1)
    native_logp = torch.log_softmax(native_logits, dim=-1)
    tree_p = tree_logp.exp()
    native_p = native_logp.exp()
    tv = 0.5 * torch.sum(torch.abs(tree_p - native_p))
    m = 0.5 * (tree_p + native_p)
    log_m = torch.log(m)
    tree_mask = tree_p > 0
    native_mask = native_p > 0
    js = 0.5 * (
        torch.sum(tree_p[tree_mask] * (tree_logp[tree_mask] - log_m[tree_mask]))
        + torch.sum(native_p[native_mask] * (native_logp[native_mask] - log_m[native_mask]))
    )
    return {
        "tree_argmax": tree_argmax,
        "native_argmax": native_argmax,
        "argmax_match": bool(tree_argmax == native_argmax),
        "tv": float(tv.item()),
        "js_nats": float(js.item()),
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    vals = sorted(float(v) for v in values)

    def q(frac: float) -> float:
        idx = min(len(vals) - 1, max(0, int(math.ceil(frac * len(vals)) - 1)))
        return vals[idx]

    return {
        "mean": float(sum(vals) / len(vals)),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": vals[-1],
    }


def _summarize_depth(rows: list[dict[str, Any]], depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for d in range(depth):
        drows = [r for r in rows if int(r["depth"]) == d]
        out[str(d)] = {
            "rows": len(drows),
            "argmax_mismatches": sum(1 for r in drows if not r["argmax_match"]),
            "argmax_mismatch_rate": (
                sum(1 for r in drows if not r["argmax_match"]) / len(drows)
                if drows
                else None
            ),
            "lag_matches": sum(1 for r in drows if r["tree_argmax_equals_native_prev_depth"]),
            "lag_match_rate": (
                sum(1 for r in drows if r["tree_argmax_equals_native_prev_depth"]) / len(drows)
                if drows
                else None
            ),
            "draft_prob_abs_delta": _quantiles(
                [float(r["prob_draft_abs_delta"]) for r in drows]
            ),
            "tv": _quantiles([float(r["tv"]) for r in drows]),
            "js_nats": _quantiles([float(r["js_nats"]) for r in drows]),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-logits", action="append", required=True)
    parser.add_argument("--native-logits", action="append", required=True)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    tree_records, tree_meta = _records(args.tree_logits, tree=True, depth=args.depth)
    native_records, native_meta = _records(args.native_logits, tree=False, depth=args.depth)
    matches, unmatched_tree, unmatched_native = _match_records(tree_records, native_records)
    if args.max_pairs is not None:
        matches = matches[: args.max_pairs]
    capture_cache: dict[Path, dict[str, Any]] = {}

    def capture(path: Path) -> dict[str, Any]:
        if path not in capture_cache:
            capture_cache[path] = _load(path)
        return capture_cache[path]

    per_depth: list[dict[str, Any]] = []
    per_event: list[dict[str, Any]] = []
    for pair_index, (tree_rec, native_rec) in enumerate(matches):
        tree_cap = capture(tree_rec.capture_path)
        native_cap = capture(native_rec.capture_path)
        event_mismatches = 0
        event_tvs: list[float] = []
        event_js: list[float] = []
        lag_depths: list[int] = []
        for d, (tree_entry, native_entry) in enumerate(zip(tree_rec.entries, native_rec.entries)):
            metrics = _row_metrics(
                tree_cap["target_logits"][tree_entry],
                native_cap["target_logits"][native_entry],
            )
            tree_probs = torch.softmax(
                tree_cap["target_logits"][tree_entry].to(torch.float32), dim=-1
            )
            native_probs = torch.softmax(
                native_cap["target_logits"][native_entry].to(torch.float32), dim=-1
            )
            draft = int(tree_rec.draft_tokens[d])
            lag_match = False
            if d > 0:
                prev_native_argmax = int(
                    torch.argmax(native_cap["target_logits"][native_rec.entries[d - 1]]).item()
                )
                lag_match = int(metrics["tree_argmax"]) == prev_native_argmax
            if not metrics["argmax_match"]:
                event_mismatches += 1
            if lag_match:
                lag_depths.append(d)
            event_tvs.append(float(metrics["tv"]))
            event_js.append(float(metrics["js_nats"]))
            per_depth.append(
                {
                    "pair_index": pair_index,
                    "depth": d,
                    "draft_token": draft,
                    "tree": {
                        "capture": str(tree_rec.capture_path),
                        "capture_call_index": tree_rec.capture_index,
                        "request_index": tree_rec.request_index,
                        "entry": int(tree_entry),
                        "target_row": int(tree_rec.target_rows[d]),
                        "argmax": int(metrics["tree_argmax"]),
                        "prob_draft": float(tree_probs[draft].item()),
                    },
                    "native": {
                        "capture": str(native_rec.capture_path),
                        "capture_call_index": native_rec.capture_index,
                        "request_index": native_rec.request_index,
                        "entry": int(native_entry),
                        "target_row": int(native_rec.target_rows[d]),
                        "argmax": int(metrics["native_argmax"]),
                        "prob_draft": float(native_probs[draft].item()),
                    },
                    "draft_match": True,
                    "argmax_match": bool(metrics["argmax_match"]),
                    "tree_argmax_equals_native_prev_depth": bool(lag_match),
                    "prob_draft_abs_delta": abs(
                        float(tree_probs[draft].item()) - float(native_probs[draft].item())
                    ),
                    "tv": float(metrics["tv"]),
                    "js_nats": float(metrics["js_nats"]),
                }
            )
        per_event.append(
            {
                "pair_index": pair_index,
                "draft_tokens": list(tree_rec.draft_tokens),
                "tree_capture": str(tree_rec.capture_path),
                "tree_capture_call_index": tree_rec.capture_index,
                "tree_request_index": tree_rec.request_index,
                "native_capture": str(native_rec.capture_path),
                "native_capture_call_index": native_rec.capture_index,
                "native_request_index": native_rec.request_index,
                "argmax_mismatch_depths": [
                    row["depth"]
                    for row in per_depth
                    if row["pair_index"] == pair_index and not row["argmax_match"]
                ],
                "lag_match_depths": lag_depths,
                "lag_persists_from_first_branch_gap": lag_depths[:3] == [2, 3, 4],
                "mean_tv": float(sum(event_tvs) / len(event_tvs)) if event_tvs else None,
                "max_tv": max(event_tvs) if event_tvs else None,
                "mean_js_nats": float(sum(event_js) / len(event_js)) if event_js else None,
                "max_js_nats": max(event_js) if event_js else None,
            }
        )

    row_count = len(per_depth)
    mismatch_count = sum(1 for row in per_depth if not row["argmax_match"])
    lag_count = sum(1 for row in per_depth if row["tree_argmax_equals_native_prev_depth"])
    result = {
        "schema": "fr12.argmax_distribution_compare.v1",
        "depth": args.depth,
        "tree_input": tree_meta,
        "native_input": native_meta,
        "matched_events": len(matches),
        "unmatched_tree_events": len(unmatched_tree),
        "unmatched_native_events": len(unmatched_native),
        "rows_compared": row_count,
        "argmax_mismatches": mismatch_count,
        "argmax_mismatch_rate": mismatch_count / row_count if row_count else None,
        "lag_matches": lag_count,
        "lag_match_rate": lag_count / row_count if row_count else None,
        "events_with_any_argmax_mismatch": sum(
            1 for row in per_event if row["argmax_mismatch_depths"]
        ),
        "events_with_lag_from_first_branch_gap": sum(
            1 for row in per_event if row["lag_persists_from_first_branch_gap"]
        ),
        "overall": {
            "draft_prob_abs_delta": _quantiles(
                [float(row["prob_draft_abs_delta"]) for row in per_depth]
            ),
            "tv": _quantiles([float(row["tv"]) for row in per_depth]),
            "js_nats": _quantiles([float(row["js_nats"]) for row in per_depth]),
        },
        "by_depth": _summarize_depth(per_depth, args.depth),
        "per_event": per_event,
        "per_depth": per_depth,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "matched_events": result["matched_events"],
                "rows_compared": result["rows_compared"],
                "argmax_mismatch_rate": result["argmax_mismatch_rate"],
                "lag_match_rate": result["lag_match_rate"],
                "overall": result["overall"],
                "by_depth": result["by_depth"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

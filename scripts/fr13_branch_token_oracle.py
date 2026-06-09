#!/usr/bin/env python3
"""Live native token oracle for FR13 off-spine branch paths.

The tree log does not contain prompt IDs, so this reducer aligns
`tree_path_lcp_max` rows to the served tree token stream. Rows are expected to be
an ordered subsequence of the served output: vLLM can emit unlogged regular
tokens before a tree event, and max_tokens can truncate the final tree event.
The reducer fails closed if an emitted tree event cannot be aligned in order.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EventContext:
    record_key: tuple[int, int]
    event_index: int
    served_prefix: list[int]
    event_start: int
    event_end: int
    alignment_gap: list[int]
    row: dict[str, Any]


class AlignmentError(SystemExit):
    pass


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _tree_records(tree_run: Path) -> list[dict[str, Any]]:
    payload = _load_json(tree_run / "tree_greedy_probe.json")
    records = list(payload.get("records") or [])
    records.sort(key=lambda row: (int(row["prompt_id"]), int(row["sample_index"])))
    return records


def _request_rows(tree_run: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(tree_run / "tree_request_metrics.jsonl")
    rows.sort(key=lambda row: (int(row["prompt_id"]), int(row.get("sample_index") or 0)))
    return rows


def _lcp_rows(tree_run: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(tree_run / "logs" / "tree_path_lcp.jsonl")
    return [row for row in rows if row.get("event") == "tree_path_lcp_max"]


def _token_ids(row: dict[str, Any], key: str) -> list[int]:
    return [int(x) for x in (row.get(key) or [])]


def _find_subsequence(haystack: list[int], needle: list[int], *, start: int) -> int | None:
    if not needle:
        return start
    last = len(haystack) - len(needle)
    for idx in range(start, last + 1):
        if haystack[idx : idx + len(needle)] == needle:
            return idx
    return None


def align_events(tree_run: Path) -> list[EventContext]:
    """Map tree_path_lcp_max rows to tree probe records.

    This intentionally supports the clean greedy gate shape: one completion per
    HTTP request (`batch_size=1`, `samples_per_prompt=1`). That avoids ambiguous
    request-boundary inference in vLLM's per-batch req_index logs.
    """

    records = _tree_records(tree_run)
    request_rows = _request_rows(tree_run)
    if len(records) != len(request_rows):
        raise AlignmentError(
            f"tree record/request row count mismatch: records={len(records)} "
            f"request_rows={len(request_rows)}"
        )
    bad_batches = [
        idx
        for idx, row in enumerate(request_rows)
        if int(row.get("batch_size") or 0) != 1
    ]
    if bad_batches:
        raise AlignmentError(
            "branch oracle requires batch_size=1 tree capture; "
            f"first_bad_request_row={bad_batches[0]}"
        )

    rows = _lcp_rows(tree_run)
    contexts: list[EventContext] = []
    cursor = 0
    skipped_overrun_rows: list[dict[str, Any]] = []
    for record in records:
        key = (int(record["prompt_id"]), int(record["sample_index"]))
        target = _token_ids(record, "token_ids")
        served_cursor = 0
        event_index = 0
        while served_cursor < len(target):
            if cursor >= len(rows):
                break
            row = rows[cursor]
            cursor += 1
            if int(row.get("req_index") or 0) != 0:
                raise AlignmentError(
                    "branch oracle requires req_index=0 rows from batch_size=1 "
                    f"capture; got req_index={row.get('req_index')} record={key}"
                )
            emitted = _token_ids(row, "emitted_tokens")
            start = _find_subsequence(target, emitted, start=served_cursor)
            if start is None:
                remaining = target[served_cursor:]
                if served_cursor == 0 and contexts:
                    skipped_overrun_rows.append(
                        {
                            "record": list(key),
                            "row_index": cursor - 1,
                            "emitted_tokens": emitted,
                        }
                    )
                    continue
                if remaining and emitted[: len(remaining)] == remaining:
                    # The request stopped mid-tree-event because max_tokens was
                    # reached. Do not create an oracle context for a partial row.
                    break
                raise AlignmentError(
                    "tree emitted-token log cannot be aligned to served output "
                    f"for record={key} search_start={served_cursor} emitted={emitted} "
                    f"remaining={target[served_cursor:]}"
                )
            gap = target[served_cursor:start]
            contexts.append(
                EventContext(
                    record_key=key,
                    event_index=event_index,
                    served_prefix=list(target[:start]),
                    event_start=int(start),
                    event_end=int(start + len(emitted)),
                    alignment_gap=gap,
                    row=row,
                )
            )
            served_cursor = start + len(emitted)
            event_index += 1
    align_events.skipped_overrun_rows = skipped_overrun_rows  # type: ignore[attr-defined]
    return contexts


def _native_next_token(
    *,
    endpoint: str,
    model: str,
    prompt_token_ids: list[int],
    mode: str,
    timeout: float,
) -> int:
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": mode},
    }
    data = _post_json(endpoint, "/v1/completions", payload, timeout)
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise RuntimeError(f"unexpected native completion response: {data!r}")
    token_ids = choices[0].get("token_ids") or []
    if not token_ids:
        raise RuntimeError(f"native completion returned no token_ids: {data!r}")
    return int(token_ids[0])


def _parse_path(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not value.strip():
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_targets(value: str | None) -> set[tuple[int, int]]:
    if value is None or not value.strip():
        return set()
    out: set[tuple[int, int]] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        prompt, pos = item.split(":", 1)
        out.add((int(prompt), int(pos)))
    return out


def _path_checks_for_context(
    *,
    ctx: EventContext,
    path_idx: int,
    path: list[int],
    prompt_tokens: list[int],
    endpoint: str,
    model: str,
    native_mode: str,
    timeout: float,
) -> list[dict[str, Any]]:
    row = ctx.row
    drafts = _token_ids(row, "draft_token_ids")
    parent_targets = _token_ids(row, "parent_target_ids")
    self_targets = _token_ids(row, "self_target_ids")
    prefix_tokens: list[int] = []
    checks: list[dict[str, Any]] = []
    for depth, node in enumerate(path):
        native_parent = _native_next_token(
            endpoint=endpoint,
            model=model,
            prompt_token_ids=prompt_tokens + ctx.served_prefix + prefix_tokens,
            mode=native_mode,
            timeout=timeout,
        )
        tree_parent = int(parent_targets[node])
        draft = int(drafts[node])
        checks.append(
            {
                "kind": "parent_target",
                "event_index": ctx.event_index,
                "path_index": int(path_idx),
                "depth": int(depth),
                "node": int(node),
                "path_prefix_token_ids": list(prefix_tokens),
                "draft_token": draft,
                "tree_parent_target_token": tree_parent,
                "native_next_token": native_parent,
                "tree_matches_native": tree_parent == native_parent,
                "tree_accepts_draft": tree_parent == draft,
                "native_accepts_draft": native_parent == draft,
            }
        )
        prefix_tokens.append(draft)

    leaf = int(path[-1])
    native_self = _native_next_token(
        endpoint=endpoint,
        model=model,
        prompt_token_ids=prompt_tokens + ctx.served_prefix + prefix_tokens,
        mode=native_mode,
        timeout=timeout,
    )
    tree_self = int(self_targets[leaf])
    checks.append(
        {
            "kind": "leaf_self_target",
            "event_index": ctx.event_index,
            "path_index": int(path_idx),
            "depth": int(len(path)),
            "node": leaf,
            "path_prefix_token_ids": list(prefix_tokens),
            "tree_self_target_token": tree_self,
            "native_next_token": native_self,
            "tree_matches_native": tree_self == native_self,
        }
    )
    return checks


def run_branch_oracle(args: argparse.Namespace) -> dict[str, Any]:
    contexts = align_events(args.tree_run)
    request_by_key = {
        (int(row["prompt_id"]), int(row.get("sample_index") or 0)): row
        for row in _tree_records(args.tree_run)
    }
    checks = []
    path_filter = _parse_path(args.path)
    started = time.time()
    selected_contexts = contexts
    if args.event_index is not None:
        selected_contexts = [ctx for ctx in contexts if ctx.event_index == args.event_index]
    target_positions = _parse_targets(args.targets)
    if target_positions:
        selected_contexts = [
            ctx
            for ctx in selected_contexts
            if any(
                ctx.record_key[0] == prompt_id
                and ctx.event_start <= position < ctx.event_end
                for prompt_id, position in target_positions
            )
        ]
    if args.max_events:
        selected_contexts = selected_contexts[: args.max_events]
    for ctx in selected_contexts:
        record = request_by_key[ctx.record_key]
        prompt_tokens = _token_ids(record, "prompt_token_ids")
        row = ctx.row
        winner_path = [int(x) for x in (row.get("winner_path") or [])]
        for path_idx, score in enumerate(row.get("path_scores") or []):
            if path_idx == 0 and not args.include_path0:
                continue
            path = [int(x) for x in (score.get("path") or [])]
            if not path:
                continue
            if args.winner_only and path != winner_path:
                continue
            if path_filter is not None and path != path_filter:
                continue
            path_checks = _path_checks_for_context(
                ctx=ctx,
                path_idx=int(path_idx),
                path=path,
                prompt_tokens=prompt_tokens,
                endpoint=args.endpoint,
                model=args.model,
                native_mode=args.native_mode,
                timeout=args.request_timeout,
            )
            for check in path_checks:
                check.update(
                    {
                        "prompt_id": ctx.record_key[0],
                        "sample_index": ctx.record_key[1],
                        "path": path,
                        "served_prefix_len": len(ctx.served_prefix),
                        "event_start": ctx.event_start,
                        "event_end": ctx.event_end,
                        "alignment_gap": ctx.alignment_gap,
                    }
                )
                checks.append(check)
    exact = sum(1 for row in checks if row.get("tree_matches_native"))
    total = len(checks)
    first_mismatch = next((row for row in checks if not row.get("tree_matches_native")), None)
    result = {
        "schema": "fr13.branch_token_oracle.v1",
        "tree_run": str(args.tree_run),
        "endpoint": args.endpoint,
        "model": args.model,
        "native_mode": args.native_mode,
        "events_aligned": len(contexts),
        "skipped_overrun_rows": getattr(align_events, "skipped_overrun_rows", []),
        "events_checked": len(selected_contexts),
        "event_index_filter": args.event_index,
        "target_positions": sorted([list(item) for item in target_positions]),
        "winner_only": bool(args.winner_only),
        "path_filter": path_filter,
        "checks_total": total,
        "tree_native_matches": exact,
        "tree_native_match_rate": exact / total if total else None,
        "first_mismatch": first_mismatch,
        "elapsed_s": time.time() - started,
        "checks": checks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-run", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--native-mode", default="naive_mtp")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--event-index", type=int)
    parser.add_argument(
        "--targets",
        help="Comma-separated prompt_id:position filters, e.g. 0:46,1:28",
    )
    parser.add_argument("--winner-only", action="store_true")
    parser.add_argument("--path", help="Comma-separated node path, e.g. 0,1,3,5,7")
    parser.add_argument("--include-path0", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run_branch_oracle(args)
    return 0 if result["first_mismatch"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

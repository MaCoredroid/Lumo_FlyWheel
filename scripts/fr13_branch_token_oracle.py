#!/usr/bin/env python3
"""Live native token oracle for FR13 off-spine branch paths.

The tree log does not contain prompt IDs, so this reducer reconstructs request
boundaries by replaying logged `emitted_tokens` against the tree probe records.
It fails closed if the reconstruction does not exactly match the served tree
token stream.
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
    rows.sort(key=lambda row: (int(row["prompt_id"]), int(row.get("batch_size") or 1)))
    return rows


def _lcp_rows(tree_run: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(tree_run / "logs" / "tree_path_lcp.jsonl")
    return [row for row in rows if row.get("event") == "tree_path_lcp_max"]


def _token_ids(row: dict[str, Any], key: str) -> list[int]:
    return [int(x) for x in (row.get(key) or [])]


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
    for record in records:
        key = (int(record["prompt_id"]), int(record["sample_index"]))
        target = _token_ids(record, "token_ids")
        served: list[int] = []
        event_index = 0
        while len(served) < len(target):
            if cursor >= len(rows):
                raise AlignmentError(
                    f"ran out of tree_path_lcp_max rows while aligning record={key}"
                )
            row = rows[cursor]
            cursor += 1
            if int(row.get("req_index") or 0) != 0:
                raise AlignmentError(
                    "branch oracle requires req_index=0 rows from batch_size=1 "
                    f"capture; got req_index={row.get('req_index')} record={key}"
                )
            emitted = _token_ids(row, "emitted_tokens")
            if target[len(served) : len(served) + len(emitted)] != emitted:
                raise AlignmentError(
                    "tree emitted-token log does not reconstruct served output "
                    f"for record={key} position={len(served)} emitted={emitted} "
                    f"target_slice={target[len(served):len(served) + len(emitted)]}"
                )
            contexts.append(
                EventContext(
                    record_key=key,
                    event_index=event_index,
                    served_prefix=list(served),
                    row=row,
                )
            )
            served.extend(emitted)
            event_index += 1
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


def run_branch_oracle(args: argparse.Namespace) -> dict[str, Any]:
    contexts = align_events(args.tree_run)
    request_by_key = {
        (int(row["prompt_id"]), int(row.get("sample_index") or 0)): row
        for row in _tree_records(args.tree_run)
    }
    checks = []
    exact = 0
    total = 0
    started = time.time()
    for ctx in contexts[: args.max_events if args.max_events else None]:
        record = request_by_key[ctx.record_key]
        prompt_tokens = _token_ids(record, "prompt_token_ids")
        row = ctx.row
        drafts = _token_ids(row, "draft_token_ids")
        self_targets = _token_ids(row, "self_target_ids")
        for path_idx, score in enumerate(row.get("path_scores") or []):
            if path_idx == 0:
                continue
            path = [int(x) for x in (score.get("path") or [])]
            if not path:
                continue
            leaf = int(path[-1])
            if leaf >= len(self_targets):
                raise RuntimeError(
                    f"self_target_ids missing leaf={leaf} in event={ctx.event_index}"
                )
            path_tokens = [int(drafts[node]) for node in path]
            native_token = _native_next_token(
                endpoint=args.endpoint,
                model=args.model,
                prompt_token_ids=prompt_tokens + ctx.served_prefix + path_tokens,
                mode=args.native_mode,
                timeout=args.request_timeout,
            )
            tree_token = int(self_targets[leaf])
            match = native_token == tree_token
            exact += int(match)
            total += 1
            checks.append(
                {
                    "prompt_id": ctx.record_key[0],
                    "sample_index": ctx.record_key[1],
                    "event_index": ctx.event_index,
                    "path_index": int(path_idx),
                    "leaf": leaf,
                    "path": path,
                    "served_prefix_len": len(ctx.served_prefix),
                    "path_token_ids": path_tokens,
                    "tree_self_target_token": tree_token,
                    "native_next_token": native_token,
                    "match": match,
                }
            )
    first_mismatch = next((row for row in checks if not row["match"]), None)
    result = {
        "schema": "fr13.branch_token_oracle.v1",
        "tree_run": str(args.tree_run),
        "endpoint": args.endpoint,
        "model": args.model,
        "native_mode": args.native_mode,
        "events_aligned": len(contexts),
        "events_checked": (
            min(len(contexts), int(args.max_events)) if args.max_events else len(contexts)
        ),
        "branch_checks": total,
        "branch_token_matches": exact,
        "branch_token_match_rate": exact / total if total else None,
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
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run_branch_oracle(args)
    return 0 if result["first_mismatch"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

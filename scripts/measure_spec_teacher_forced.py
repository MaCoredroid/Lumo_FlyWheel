#!/usr/bin/env python3
"""Teacher-forced E5/tree speculative decode comparison.

This diagnostic removes generated-text divergence by forcing both servers through
the same fixed token sequence. For each forced prefix it records the MTP draft
top-one chain and the one-token greedy target argmax.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from measure_spec_per_position import (  # noqa: E402
    GuardError,
    detect_batch_invariant,
    validate_live_server,
    validate_sampling,
)

DEFAULT_REFERENCE = Path("tests/fixtures/spec_teacher_force_reference.txt")
DEFAULT_TRACE = Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs/teacherforce_mtp.jsonl")
DEFAULT_TREE_LCP = Path("/tmp/lumo-l0c-fp8-cutlass-run30-logs/tree_path_lcp_max.jsonl")


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
    headers = {"Authorization": "Bearer EMPTY"}
    data = None
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tokenize(endpoint: str, model: str, prompt: str) -> list[int]:
    data = _http_json(
        endpoint.rstrip("/") + "/tokenize",
        {"model": model, "prompt": prompt},
        timeout=120,
    )
    tokens = data.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise RuntimeError(f"unexpected /tokenize response: {data!r}")
    return tokens


def detokenize(endpoint: str, model: str, tokens: list[int]) -> str:
    data = _http_json(
        endpoint.rstrip("/") + "/detokenize",
        {"model": model, "tokens": tokens},
        timeout=120,
    )
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError(f"unexpected /detokenize response: {data!r}")
    return prompt


def trace_records_after(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return 0, []
    size = path.stat().st_size
    if offset > size:
        offset = 0
    records: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    for raw in chunk.decode("utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "mtp_draft":
            records.append(row)
    return size, records


def wait_for_one_trace_record(path: Path, offset: int, timeout_s: float = 30.0) -> tuple[int, dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        new_offset, records = trace_records_after(path, offset)
        if records:
            return new_offset, records[0]
        offset = new_offset
        time.sleep(0.05)
    raise RuntimeError(f"no mtp_draft record appended to {path} within {timeout_s}s")


def tree_lcp_records_after(path: Path, offset: int) -> tuple[int, list[dict[str, Any]]]:
    if not path.exists():
        return 0, []
    size = path.stat().st_size
    if offset > size:
        offset = 0
    records: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        fh.seek(offset)
        chunk = fh.read()
    for raw in chunk.decode("utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "tree_path_lcp_max":
            records.append(row)
    return size, records


def wait_for_one_tree_lcp_record(path: Path, offset: int, timeout_s: float = 30.0) -> tuple[int, dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        new_offset, records = tree_lcp_records_after(path, offset)
        if records:
            return new_offset, records[0]
        offset = new_offset
        time.sleep(0.05)
    raise RuntimeError(f"no tree_path_lcp_max record appended to {path} within {timeout_s}s")


def tree_path0_indices(depth: int, spines: int) -> list[int]:
    return [level * spines for level in range(depth)]


def draft_top1_chain(
    record: dict[str, Any], mode: str, depth: int = 5, spines: int = 2
) -> list[int]:
    draft = record.get("draft")
    if not isinstance(draft, list) or not draft or not isinstance(draft[0], list):
        raise ValueError(f"bad mtp_draft row: {record!r}")
    row = [int(x) for x in draft[0]]
    if mode == "e5":
        return row[:depth]
    if mode == "tree":
        return [row[i] for i in tree_path0_indices(depth, spines) if i < len(row)]
    raise ValueError(f"unknown mode {mode!r}")


def forced_acceptance_lcps(rows: list[dict[str, Any]], depth: int) -> list[int]:
    """Per-row LCP computed by the live verifier during measurement."""
    lcps: list[int] = []
    for row in rows:
        value = row.get("teacher_forced_accept_lcp")
        if value is None:
            continue
        lcps.append(min(int(value), depth))
    return lcps


def summarize_lcps(lcps: list[int], depth: int) -> dict[str, Any]:
    n = len(lcps)
    return {
        "avg": sum(lcps) / n if n else None,
        "acc0": sum(v == 0 for v in lcps) / n if n else None,
        "full": sum(v >= depth for v in lcps) / n if n else None,
        "per_position": [
            sum(v >= k for v in lcps) / n if n else None for k in range(1, depth + 1)
        ],
        "counts": dict(sorted(Counter(lcps).items())),
    }


def forced_acceptance_summary(measurement: dict[str, Any]) -> dict[str, Any]:
    rows = measurement.get("rows") or []
    depth = int(measurement.get("depth") or 5)
    lcps = forced_acceptance_lcps(rows, depth)
    summary = summarize_lcps(lcps, depth)
    summary["available"] = len(lcps) == len(rows)
    summary["n_rows"] = len(lcps)
    summary["source"] = (
        "engine tree_path_lcp_max for tree mode; live target queries for e5 mode"
    )
    return summary


def compute_teacher_forced_acceptance(
    endpoint: str,
    model: str,
    prefix_tokens: list[int],
    first_target_token_id: int | None,
    draft_top1: list[int],
) -> tuple[int | None, list[int | None], str | None]:
    if first_target_token_id is None:
        return None, [], "missing first target argmax token id"
    accepted: list[int] = []
    target_chain: list[int | None] = []
    for draft_token in draft_top1:
        verifier_prefix_tokens = prefix_tokens + [int(first_target_token_id)] + accepted
        verifier_prefix_text = detokenize(endpoint, model, verifier_prefix_tokens)
        response = _completion_one(endpoint, model, verifier_prefix_text)
        completion_text = response["choices"][0].get("text") or ""
        target_token_id, warning = recover_argmax_token_id(
            endpoint,
            model,
            verifier_prefix_text,
            verifier_prefix_tokens,
            completion_text,
        )
        target_chain.append(target_token_id)
        if warning is not None:
            return len(accepted), target_chain, warning
        if target_token_id != int(draft_token):
            break
        accepted.append(int(draft_token))
    return len(accepted), target_chain, None


def _completion_one(endpoint: str, model: str, prompt: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
        "logprobs": 5,
    }
    return _http_json(endpoint.rstrip("/") + "/v1/completions", payload, timeout=900)


def recover_argmax_token_id(
    endpoint: str,
    model: str,
    prefix_text: str,
    prefix_tokens: list[int],
    completion_text: str,
) -> tuple[int | None, str | None]:
    combined = tokenize(endpoint, model, prefix_text + completion_text)
    if combined[: len(prefix_tokens)] == prefix_tokens and len(combined) > len(prefix_tokens):
        return int(combined[len(prefix_tokens)]), None
    return None, "could not recover argmax token id from prefix+completion tokenization"


def _logprob_token(response: dict[str, Any]) -> str | None:
    try:
        tokens = response["choices"][0]["logprobs"]["tokens"]
    except (KeyError, TypeError, IndexError):
        return None
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], str):
        return tokens[0]
    return None


def measure(args: argparse.Namespace) -> dict[str, Any]:
    validate_sampling(0.0, 1.0)
    guard = detect_batch_invariant(args.endpoint)
    if not args.no_live_guards:
        validate_live_server(args.endpoint, guard)

    reference_path = Path(args.reference)
    reference_text = reference_path.read_text(encoding="utf-8")
    reference_tokens = tokenize(args.endpoint, args.model, reference_text)
    if len(reference_tokens) < args.start_token + args.limit + 1:
        raise SystemExit(
            f"reference has {len(reference_tokens)} tokens; need at least "
            f"{args.start_token + args.limit + 1}"
        )

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.touch()
    offset = trace_path.stat().st_size
    tree_lcp_path = Path(args.tree_lcp_log)
    if args.mode == "tree" and not tree_lcp_path.exists():
        tree_lcp_path.parent.mkdir(parents=True, exist_ok=True)
        tree_lcp_path.touch()

    rows: list[dict[str, Any]] = []
    start_ts = time.time()
    for row_index, position in enumerate(range(args.start_token, args.start_token + args.limit)):
        prefix_tokens = reference_tokens[:position]
        prefix_text = detokenize(args.endpoint, args.model, prefix_tokens)
        before_offset = trace_path.stat().st_size
        before_tree_lcp_offset = tree_lcp_path.stat().st_size if args.mode == "tree" else 0
        response = _completion_one(args.endpoint, args.model, prefix_text)
        offset, trace_row = wait_for_one_trace_record(trace_path, before_offset)
        tree_lcp_row = None
        if args.mode == "tree":
            _, tree_lcp_row = wait_for_one_tree_lcp_record(
                tree_lcp_path, before_tree_lcp_offset
            )
        choice = response["choices"][0]
        completion_text = choice.get("text") or ""
        argmax_token_id, recovery_warning = recover_argmax_token_id(
            args.endpoint, args.model, prefix_text, prefix_tokens, completion_text
        )
        draft_top1 = draft_top1_chain(
            trace_row, args.mode, depth=args.depth, spines=args.spines
        )
        accept_lcp = None
        accept_target_chain: list[int | None] = []
        accept_warning = None
        if not args.skip_acceptance and args.mode == "tree":
            accept_lcp = min(int((tree_lcp_row or {}).get("path0_lcp", 0)), args.depth)
        elif not args.skip_acceptance:
            accept_lcp, accept_target_chain, accept_warning = compute_teacher_forced_acceptance(
                args.endpoint,
                args.model,
                prefix_tokens,
                argmax_token_id,
                draft_top1,
            )
        row = {
            "row_index": row_index,
            "forced_position": position,
            "next_reference_token_id": int(reference_tokens[position]),
            "draft_top1": draft_top1,
            "target_argmax_token_id": argmax_token_id,
            "target_argmax_text": completion_text,
            "target_argmax_logprob_token": _logprob_token(response),
            "teacher_forced_accept_lcp": accept_lcp,
            "teacher_forced_accept_target_chain": accept_target_chain,
            "finish_reason": choice.get("finish_reason"),
            "trace_idx": trace_row.get("idx"),
            "trace_ts": trace_row.get("ts"),
            "recovery_warning": recovery_warning,
            "acceptance_recovery_warning": accept_warning,
            "engine_tree_path_lcp": tree_lcp_row,
        }
        rows.append(row)
        if args.progress_every and (row_index + 1) % args.progress_every == 0:
            print(f"{args.mode}: forced {row_index + 1}/{args.limit}", flush=True)

    end_ts = time.time()
    measurement = {
        "schema": "spec_teacher_forced.v1",
        "mode": args.mode,
        "endpoint": args.endpoint,
        "model": args.model,
        "reference": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "reference_token_count": len(reference_tokens),
        "start_token": args.start_token,
        "limit": args.limit,
        "depth": args.depth,
        "spines": args.spines,
        "trace_file": str(trace_path),
        "tree_lcp_log": str(tree_lcp_path),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "guard": guard,
        "rows": rows,
    }
    measurement["forced_acceptance"] = forced_acceptance_summary(measurement)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
    print(f"{args.mode}: wrote {len(rows)} teacher-forced rows to {out}")
    return measurement


def compare_measurements(e5: dict[str, Any], tree: dict[str, Any]) -> dict[str, Any]:
    if e5.get("reference_sha256") != tree.get("reference_sha256"):
        raise ValueError("reference_sha256 differs")
    if e5.get("start_token") != tree.get("start_token"):
        raise ValueError("start_token differs")
    e5_rows = e5.get("rows") or []
    tree_rows = tree.get("rows") or []
    limit = min(len(e5_rows), len(tree_rows))
    table: list[dict[str, Any]] = []
    draft_mismatches = 0
    target_mismatches = 0
    for idx in range(limit):
        left = e5_rows[idx]
        right = tree_rows[idx]
        if left.get("forced_position") != right.get("forced_position"):
            raise ValueError(f"forced_position mismatch at row {idx}")
        draft_match = left.get("draft_top1") == right.get("draft_top1")
        left_target = left.get("target_argmax_token_id")
        right_target = right.get("target_argmax_token_id")
        target_match = left_target == right_target
        if not draft_match:
            draft_mismatches += 1
        if not target_match:
            target_mismatches += 1
        table.append(
            {
                "forced_position": left["forced_position"],
                "e5_draft_top1": left.get("draft_top1"),
                "tree_spine_a_draft_top1": right.get("draft_top1"),
                "draft_match": draft_match,
                "e5_target_argmax_token_id": left_target,
                "tree_target_argmax_token_id": right_target,
                "target_match": target_match,
            }
        )
    if target_mismatches:
        verdict = "target-fundamental"
    elif draft_mismatches:
        verdict = "proposer-bug"
    else:
        verdict = "divergence-only"
    e5_for_acceptance = dict(e5)
    tree_for_acceptance = dict(tree)
    e5_for_acceptance["rows"] = e5_rows[:limit]
    tree_for_acceptance["rows"] = tree_rows[:limit]
    e5_acceptance = forced_acceptance_summary(e5_for_acceptance)
    tree_acceptance = forced_acceptance_summary(tree_for_acceptance)
    e5_avg = e5_acceptance.get("avg")
    tree_avg = tree_acceptance.get("avg")
    return {
        "schema": "spec_teacher_forced_compare.v1",
        "verdict": verdict,
        "n_rows": limit,
        "draft_mismatches": draft_mismatches,
        "target_mismatches": target_mismatches,
        "draft_match_rate": (limit - draft_mismatches) / limit if limit else None,
        "target_match_rate": (limit - target_mismatches) / limit if limit else None,
        "e5_forced_acceptance": e5_acceptance,
        "tree_forced_acceptance": tree_acceptance,
        "forced_acceptance_delta_avg": (
            tree_avg - e5_avg
            if isinstance(e5_avg, (int, float)) and isinstance(tree_avg, (int, float))
            else None
        ),
        "table": table,
    }


def print_compare(summary: dict[str, Any], max_rows: int) -> None:
    print(
        f"verdict={summary['verdict']} rows={summary['n_rows']} "
        f"draft_match={100 * summary['draft_match_rate']:.1f}% "
        f"target_match={100 * summary['target_match_rate']:.1f}%"
    )
    e5_acc = summary["e5_forced_acceptance"]
    tree_acc = summary["tree_forced_acceptance"]
    if e5_acc.get("available") and tree_acc.get("available"):
        print(
            "forced_acceptance_avg "
            f"e5={e5_acc['avg']:.3f} tree={tree_acc['avg']:.3f} "
            f"delta={summary['forced_acceptance_delta_avg']:.3f}"
        )
        print(
            "forced_acceptance_per_pos "
            f"e5={[round(x, 4) for x in e5_acc['per_position']]} "
            f"tree={[round(x, 4) for x in tree_acc['per_position']]}"
        )
    else:
        print("forced_acceptance unavailable: rerun measure without --skip-acceptance")
    print("pos  draft  target  e5_draft0  tree_draft0  e5_argmax  tree_argmax")
    for row in summary["table"][:max_rows]:
        print(
            f"{row['forced_position']:>3}  "
            f"{'yes' if row['draft_match'] else 'NO ':>5}  "
            f"{'yes' if row['target_match'] else 'NO ':>6}  "
            f"{str(row['e5_draft_top1'][:5]):>24}  "
            f"{str(row['tree_spine_a_draft_top1'][:5]):>24}  "
            f"{str(row['e5_target_argmax_token_id']):>9}  "
            f"{str(row['tree_target_argmax_token_id']):>11}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure_parser = sub.add_parser("measure")
    measure_parser.add_argument("--mode", choices=["e5", "tree"], required=True)
    measure_parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    measure_parser.add_argument("--out", required=True)
    measure_parser.add_argument("--endpoint", default="http://127.0.0.1:9950")
    measure_parser.add_argument("--model", default="qwen3.6-27b")
    measure_parser.add_argument("--trace-file", default=str(DEFAULT_TRACE))
    measure_parser.add_argument("--tree-lcp-log", default=str(DEFAULT_TREE_LCP))
    measure_parser.add_argument("--start-token", type=int, default=16)
    measure_parser.add_argument("--limit", type=int, default=384)
    measure_parser.add_argument("--depth", type=int, default=5)
    measure_parser.add_argument("--spines", type=int, default=2)
    measure_parser.add_argument("--progress-every", type=int, default=25)
    measure_parser.add_argument("--skip-acceptance", action="store_true")
    measure_parser.add_argument("--no-live-guards", action="store_true", help=argparse.SUPPRESS)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--e5", required=True)
    compare_parser.add_argument("--tree", required=True)
    compare_parser.add_argument("--out")
    compare_parser.add_argument("--max-rows", type=int, default=40)

    args = parser.parse_args(argv)
    try:
        if args.command == "measure":
            measure(args)
        else:
            summary = compare_measurements(
                json.loads(Path(args.e5).read_text(encoding="utf-8")),
                json.loads(Path(args.tree).read_text(encoding="utf-8")),
            )
            if args.out:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print_compare(summary, args.max_rows)
        return 0
    except GuardError as exc:
        print(f"guard failed: {exc}", file=sys.stderr)
        return 2
    except (RuntimeError, ValueError, KeyError, urllib.error.URLError) as exc:
        print(f"teacher-force failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""FR10 native-spine vs tree path0 draft-token parity probe.

This is a draft-side diagnostic only. It forces fixed prefixes from a reference
text, captures the final tensor returned by ``EagleProposer.propose`` via
``LUMO_MTP_DRAFT_TRACE_FILE``, and compares native/spine top-1 draft tokens
against the branched tree runtime path0 slots.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE = Path("tests/fixtures/spec_teacher_force_reference.txt")


def _http_json(url: str, payload: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    headers = {"Authorization": "Bearer EMPTY", "Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tokenize(endpoint: str, model: str, prompt: str) -> list[int]:
    data = _http_json(endpoint.rstrip("/") + "/tokenize", {"model": model, "prompt": prompt})
    tokens = data.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"bad tokenize response: {data!r}")
    return [int(x) for x in tokens]


def detokenize(endpoint: str, model: str, tokens: list[int]) -> str:
    data = _http_json(endpoint.rstrip("/") + "/detokenize", {"model": model, "tokens": tokens})
    text = data.get("prompt")
    if not isinstance(text, str):
        raise RuntimeError(f"bad detokenize response: {data!r}")
    return text


def completion(endpoint: str, model: str, prompt: str, mode: str | None) -> None:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "top_p": 1,
    }
    if mode:
        payload["vllm_xargs"] = {"fr10_decode_mode": mode}
    _http_json(endpoint.rstrip("/") + "/v1/completions", payload, timeout=900)


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


def wait_for_trace(path: Path, offset: int, timeout_s: float = 60.0) -> tuple[int, dict[str, Any]]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        new_offset, rows = trace_records_after(path, offset)
        if rows:
            return new_offset, rows[0]
        offset = new_offset
        time.sleep(0.05)
    raise RuntimeError(f"no mtp_draft row appended to {path}")


def path0_runtime_nodes(speculative_token_tree: str, depth: int) -> list[int]:
    choices = sorted(ast.literal_eval(speculative_token_tree), key=lambda item: (len(item), item))
    nodes = [
        idx
        for idx, choice in enumerate(choices)
        if all(int(part) == 0 for part in choice)
    ]
    return nodes[:depth]


def extract_chain(
    trace_row: dict[str, Any],
    *,
    extract: str,
    depth: int,
    tree: str | None,
) -> list[int]:
    draft = trace_row.get("draft")
    if not isinstance(draft, list) or not draft or not isinstance(draft[0], list):
        raise RuntimeError(f"bad mtp_draft row: {trace_row!r}")
    row = [int(x) for x in draft[0]]
    if extract == "native":
        return row[:depth]
    if extract == "tree_path0":
        if tree is None:
            raise RuntimeError("--tree is required for tree_path0 extraction")
        return [row[idx] for idx in path0_runtime_nodes(tree, depth) if idx < len(row)]
    raise RuntimeError(f"unknown extract mode {extract!r}")


def measure(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = Path(args.reference)
    reference_text = reference_path.read_text(encoding="utf-8")
    reference_tokens = tokenize(args.endpoint, args.model, reference_text)
    if len(reference_tokens) < args.start_token + args.limit + 1:
        raise SystemExit(
            f"reference has {len(reference_tokens)} tokens; need "
            f"{args.start_token + args.limit + 1}"
        )
    trace_path = Path(args.trace_file)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.touch(exist_ok=True)
    offset = trace_path.stat().st_size

    rows: list[dict[str, Any]] = []
    for row_index, pos in enumerate(range(args.start_token, args.start_token + args.limit)):
        prefix_tokens = reference_tokens[:pos]
        prefix = detokenize(args.endpoint, args.model, prefix_tokens)
        before = trace_path.stat().st_size
        completion(args.endpoint, args.model, prefix, args.request_mode)
        offset, trace_row = wait_for_trace(trace_path, before)
        rows.append(
            {
                "row_index": row_index,
                "forced_position": pos,
                "trace_idx": trace_row.get("idx"),
                "trace_mode": trace_row.get("mode"),
                "draft_chain": extract_chain(
                    trace_row, extract=args.extract, depth=args.depth, tree=args.tree
                ),
                "full_draft": trace_row.get("draft", [[]])[0],
            }
        )
        if args.progress_every and (row_index + 1) % args.progress_every == 0:
            print(f"{args.extract}: forced {row_index + 1}/{args.limit}", flush=True)

    out = {
        "schema": "fr10.draft_token_parity_measurement.v1",
        "endpoint": args.endpoint,
        "model": args.model,
        "reference": str(reference_path),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "start_token": args.start_token,
        "limit": args.limit,
        "depth": args.depth,
        "extract": args.extract,
        "request_mode": args.request_mode,
        "tree": args.tree,
        "path0_runtime_nodes": path0_runtime_nodes(args.tree, args.depth)
        if args.tree and args.extract == "tree_path0"
        else None,
        "trace_file": str(trace_path),
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{args.extract}: wrote {len(rows)} rows to {args.out}")
    return out


def compare(args: argparse.Namespace) -> dict[str, Any]:
    native = json.loads(Path(args.native).read_text(encoding="utf-8"))
    tree = json.loads(Path(args.tree_measurement).read_text(encoding="utf-8"))
    if native["reference_sha256"] != tree["reference_sha256"]:
        raise RuntimeError("reference hashes differ")
    native_rows = native.get("rows") or []
    tree_rows = tree.get("rows") or []
    n = min(len(native_rows), len(tree_rows))
    by_depth = [
        {"depth": depth, "matches": 0, "compared": 0, "mismatches": 0}
        for depth in range(int(args.depth))
    ]
    table: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    for idx in range(n):
        left = native_rows[idx]
        right = tree_rows[idx]
        if left["forced_position"] != right["forced_position"]:
            raise RuntimeError(f"forced_position mismatch at row {idx}")
        l_chain = [int(x) for x in left.get("draft_chain") or []]
        r_chain = [int(x) for x in right.get("draft_chain") or []]
        mismatch_depth = None
        for depth in range(min(args.depth, len(l_chain), len(r_chain))):
            by_depth[depth]["compared"] += 1
            if l_chain[depth] == r_chain[depth]:
                by_depth[depth]["matches"] += 1
            else:
                by_depth[depth]["mismatches"] += 1
                if mismatch_depth is None:
                    mismatch_depth = depth
        row = {
            "row_index": idx,
            "forced_position": left["forced_position"],
            "native": l_chain,
            "tree_path0": r_chain,
            "match": l_chain == r_chain,
            "first_mismatch_depth": mismatch_depth,
        }
        table.append(row)
        if first_mismatch is None and mismatch_depth is not None:
            first_mismatch = row
    for item in by_depth:
        compared = item["compared"]
        item["match_rate"] = item["matches"] / compared if compared else None
    verdict = "match" if first_mismatch is None and n > 0 else "draft_divergence"
    out = {
        "schema": "fr10.draft_token_parity_compare.v1",
        "verdict": verdict,
        "rows_compared": n,
        "depth": args.depth,
        "by_depth": by_depth,
        "first_mismatch": first_mismatch,
        "table": table,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(
        f"verdict={verdict} rows={n} first_mismatch_depth="
        f"{None if first_mismatch is None else first_mismatch['first_mismatch_depth']}"
    )
    print("depth compared matches mismatches match_rate")
    for item in by_depth:
        rate = item["match_rate"]
        print(
            f"{item['depth']:>5} {item['compared']:>8} {item['matches']:>7} "
            f"{item['mismatches']:>10} {rate if rate is not None else 'NA'}"
        )
    if first_mismatch is not None:
        print(json.dumps(first_mismatch, indent=2))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    measure_p = sub.add_parser("measure")
    measure_p.add_argument("--endpoint", default="http://127.0.0.1:9950")
    measure_p.add_argument("--model", default="qwen3.6-27b")
    measure_p.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    measure_p.add_argument("--trace-file", required=True)
    measure_p.add_argument("--out", required=True)
    measure_p.add_argument("--extract", choices=["native", "tree_path0"], required=True)
    measure_p.add_argument("--tree")
    measure_p.add_argument("--request-mode")
    measure_p.add_argument("--start-token", type=int, default=16)
    measure_p.add_argument("--limit", type=int, default=32)
    measure_p.add_argument("--depth", type=int, default=5)
    measure_p.add_argument("--progress-every", type=int, default=8)

    compare_p = sub.add_parser("compare")
    compare_p.add_argument("--native", required=True)
    compare_p.add_argument("--tree-measurement", required=True)
    compare_p.add_argument("--out")
    compare_p.add_argument("--depth", type=int, default=5)

    args = parser.parse_args()
    if args.command == "measure":
        measure(args)
    else:
        compare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

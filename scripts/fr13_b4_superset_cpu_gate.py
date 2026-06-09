#!/usr/bin/env python3
"""CPU-only FR13 B4 superset/topology diagnostic from saved probe artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PATH0_NODES = [0, 1, 3, 5, 7]
EXPECTED_TREE = [
    (0,),
    (0, 0),
    (0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0, 0),
    (0, 1),
    (0, 0, 1),
    (0, 0, 0, 1),
    (0, 0, 0, 0, 1),
]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_path(nodes: list[int]) -> str:
    if not nodes:
        return "reject0"
    return "spine" if nodes == PATH0_NODES[: len(nodes)] else "branch"


def _tree_spec(draft_rows: list[dict[str, Any]]) -> list[tuple[int, ...]] | None:
    for row in draft_rows:
        spec = row.get("speculative_token_tree")
        if not spec:
            continue
        try:
            payload = json.loads(spec)
            tree = payload.get("speculative_token_tree")
            if tree:
                return [tuple(x) for x in ast.literal_eval(tree)]
        except Exception:
            continue
    return None


def _summarize_tree_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter()
    accepted_len = Counter()
    selected_sources = Counter()
    reject_steps = Counter()
    branch_total = 0
    branch_top1_zero = 0
    branch_top1_positive = 0
    branch_both_positive = 0
    reject_total = 0
    reject_top1_positive = 0
    reject_both_zero = 0
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for idx, row in enumerate(rows):
        nodes = [int(x) for x in row.get("accepted_node_ids") or []]
        cls = _classify_path(nodes)
        classes[cls] += 1
        accepted_len[int(row.get("accepted_len") or 0)] += 1
        if cls == "branch" and len(examples["branch"]) < 5:
            examples["branch"].append(
                {
                    "row_index": idx,
                    "req_index": row.get("req_index"),
                    "accepted_len": row.get("accepted_len"),
                    "accepted_node_ids": nodes,
                    "emitted_tokens": row.get("emitted_tokens"),
                    "draft_token_ids": row.get("draft_token_ids"),
                }
            )
        if cls == "reject0" and len(examples["reject0"]) < 5:
            examples["reject0"].append(
                {
                    "row_index": idx,
                    "req_index": row.get("req_index"),
                    "emitted_tokens": row.get("emitted_tokens"),
                    "draft_token_ids": row.get("draft_token_ids"),
                    "first_step": (row.get("committer_step_trace") or [None])[0],
                }
            )

        for step in row.get("committer_step_trace") or []:
            probs = list(step.get("target_prob_at_draft_token_ids") or [])
            if step.get("accepted"):
                source = int(step.get("selected_source_index", -1))
                selected_sources[source] += 1
                if source == 1:
                    branch_total += 1
                    top1 = float(probs[0]) if len(probs) >= 1 else 0.0
                    top2 = float(probs[1]) if len(probs) >= 2 else 0.0
                    if top1 > 0:
                        branch_top1_positive += 1
                    else:
                        branch_top1_zero += 1
                    if top1 > 0 and top2 > 0:
                        branch_both_positive += 1
            else:
                reject_total += 1
                reject_steps[int(step.get("step", -1))] += 1
                top1 = float(probs[0]) if len(probs) >= 1 else 0.0
                top2 = float(probs[1]) if len(probs) >= 2 else 0.0
                if top1 > 0:
                    reject_top1_positive += 1
                if top1 == 0.0 and (len(probs) < 2 or top2 == 0.0):
                    reject_both_zero += 1

    return {
        "rows": len(rows),
        "accepted_len_counts": dict(sorted(accepted_len.items())),
        "path_class_counts": dict(classes),
        "selected_accepted_source_counts": dict(selected_sources),
        "reject_step_counts": dict(sorted(reject_steps.items())),
        "branch_source1_total": branch_total,
        "branch_source1_top1_zero": branch_top1_zero,
        "branch_source1_top1_positive": branch_top1_positive,
        "branch_source1_both_positive": branch_both_positive,
        "reject_steps_total": reject_total,
        "reject_steps_top1_positive": reject_top1_positive,
        "reject_steps_both_zero": reject_both_zero,
        "accepted_len_sum": sum(
            int(k) * int(v) for k, v in accepted_len.items()
        ),
        "examples": examples,
    }


def _first_token_mismatches(tree_probe: dict[str, Any], native_probe: dict[str, Any]) -> list[dict[str, Any]]:
    native = {
        (int(row["prompt_id"]), int(row["sample_index"])): row
        for row in native_probe.get("records", [])
    }
    out = []
    for row in tree_probe.get("records", []):
        key = (int(row["prompt_id"]), int(row["sample_index"]))
        other = native.get(key)
        if other is None:
            continue
        t = list(row.get("token_ids") or [])
        n = list(other.get("token_ids") or [])
        first = None
        for idx, (a, b) in enumerate(zip(t, n)):
            if int(a) != int(b):
                first = idx
                break
        if first is None and len(t) != len(n):
            first = min(len(t), len(n))
        out.append(
            {
                "prompt_id": key[0],
                "sample_index": key[1],
                "first_diff": first,
                "tree_token": None if first is None or first >= len(t) else int(t[first]),
                "native_token": None if first is None or first >= len(n) else int(n[first]),
                "tree_len": len(t),
                "native_len": len(n),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = args.run
    tree_probe = _load(run / "tree_b4_swe4" / "tree_b4_swe4_probe.json")
    native_probe = _load(run / "native_b4_swe4" / "native_b4_swe4_probe.json")
    lcp_rows = _jsonl(run / "tree_b4_swe4" / "logs" / "tree_path_lcp.jsonl")
    sampler_rows = _jsonl(run / "tree_b4_swe4" / "logs" / "tree_sampler_debug.jsonl")
    draft_rows = _jsonl(run / "tree_b4_swe4" / "logs" / "fr10_mtp_draft_trace.jsonl")

    measured_drafts = int(round(tree_probe["summary"]["spec_drafts"]))
    measured_tail = lcp_rows[-measured_drafts:] if measured_drafts <= len(lcp_rows) else lcp_rows
    tree_spec = _tree_spec(draft_rows)
    metadata_ok = [
        row
        for row in sampler_rows
        if row.get("event") == "gpu_tree_metadata"
        and row.get("mode") == "tree_mtp"
        and row.get("reason") == "ok"
        and row.get("tree_len") == 9
        and row.get("has_tree_parent_indices") is True
        and row.get("has_draft_token_indices") is True
    ]
    topology = {
        "speculative_token_tree": [list(x) for x in tree_spec] if tree_spec else None,
        "expected_top1_top2_tree": [list(x) for x in EXPECTED_TREE],
        "matches_expected_top1_top2_tree": tree_spec == EXPECTED_TREE,
        "path0_nodes": PATH0_NODES,
        "branch_sibling_nodes": [2, 4, 6, 8],
        "gpu_tree_metadata_ok_rows": len(metadata_ok),
        "gpu_tree_metadata_total_rows": sum(
            1 for row in sampler_rows if row.get("event") == "gpu_tree_metadata"
        ),
    }
    payload = {
        "schema": "fr13.b4_superset_cpu_gate.v1",
        "run": str(run),
        "tree_summary": tree_probe["summary"],
        "native_summary": native_probe["summary"],
        "topology": topology,
        "tree_lcp_rows_total": len(lcp_rows),
        "tree_measured_drafts_from_summary": measured_drafts,
        "tree_all_rows": _summarize_tree_rows(lcp_rows),
        "tree_measured_tail_rows": _summarize_tree_rows(measured_tail),
        "served_token_mismatches": {
            "records": _first_token_mismatches(tree_probe, native_probe),
            "exact_sequence_matches": sum(
                1
                for trow in tree_probe.get("records", [])
                for nrow in native_probe.get("records", [])
                if int(trow["prompt_id"]) == int(nrow["prompt_id"])
                and int(trow["sample_index"]) == int(nrow["sample_index"])
                and list(trow.get("token_ids") or []) == list(nrow.get("token_ids") or [])
            ),
        },
        "native_trace_limitation": (
            "The native B4 artifact has aggregate spec counters and served token IDs, "
            "but no per-event MTP draft/accept trace. Exact native per-event accepted "
            "depth cannot be reconstructed CPU-only from this run."
        ),
        "classification": {
            "topology_break": False if topology["matches_expected_top1_top2_tree"] else True,
            "verify_or_target_row_break": True,
            "committer_break": False,
            "reason": (
                "The 9-node top1+top2 tree is present. In tree verifier rows, many "
                "events reject at step0 or assign zero probability to the top1/spine "
                "child and accept source_index=1. The committer follows those verifier "
                "scores; it is not missing the tree topology."
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "topology": topology,
                "tree_measured_tail_rows": payload["tree_measured_tail_rows"],
                "classification": payload["classification"],
                "native_trace_limitation": payload["native_trace_limitation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

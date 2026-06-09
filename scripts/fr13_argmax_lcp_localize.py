#!/usr/bin/env python3
"""Reduce FR13 paired greedy tree_path_lcp_max/native captures.

This reducer fails closed on prompt-pairing bugs. A tree/native argmax
comparison is only valid when the captured prompt token IDs are byte-identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


TREE_SPINE_PATH = [0, 1, 3, 5, 7]


class PromptIdentityError(SystemExit):
    """Raised when tree/native request prompt tokens are not byte-identical."""

    def __init__(self, summary: dict[str, Any]):
        self.summary = summary
        super().__init__(_prompt_identity_error_message(summary))


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _common_prefix(left: list[int], right: list[int]) -> int:
    prefix = 0
    for lhs, rhs in zip(left, right):
        if int(lhs) != int(rhs):
            break
        prefix += 1
    return prefix


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _records(path: Path) -> list[dict[str, Any]]:
    data = _load_json(path)
    if "records" in data:
        return data["records"]
    modes = data.get("modes") or {}
    for value in modes.values():
        if isinstance(value, dict) and "records" in value:
            return value["records"]
    return []


def _token_alignment(left: list[int], right: list[int]) -> dict[str, Any]:
    prefix = _common_prefix(left, right)
    first_mismatch = None
    if prefix < len(left) or prefix < len(right):
        first_mismatch = {
            "position": int(prefix),
            "left_token": int(left[prefix]) if prefix < len(left) else None,
            "right_token": int(right[prefix]) if prefix < len(right) else None,
        }
    return {
        "left_len": len(left),
        "right_len": len(right),
        "common_prefix": int(prefix),
        "exact": first_mismatch is None,
        "first_mismatch": first_mismatch,
    }


def _request_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_jsonl(path)


def _prompt_identity_error_message(summary: dict[str, Any]) -> str:
    if summary.get("missing_prompt_token_rows"):
        first = summary["missing_prompt_token_rows"][0]
        return (
            "prompt token identity guard failed: missing prompt_token_ids at "
            f"pair_index={first.get('pair_index')} "
            f"tree_prompt_id={first.get('tree_prompt_id')} "
            f"native_prompt_id={first.get('native_prompt_id')}"
        )
    if summary.get("mismatches"):
        first = summary["mismatches"][0]
        return (
            "prompt token identity guard failed: prompt_token_ids differ at "
            f"pair_index={first.get('pair_index')} "
            f"common_prefix_tokens={first.get('common_prefix_tokens')} "
            f"tree_first_diff={first.get('tree_first_diff')} "
            f"native_first_diff={first.get('native_first_diff')}"
        )
    if summary.get("tree_request_rows") != summary.get("native_request_rows"):
        return (
            "prompt token identity guard failed: request row count mismatch "
            f"tree={summary.get('tree_request_rows')} "
            f"native={summary.get('native_request_rows')}"
        )
    return "prompt token identity guard failed"


def _compute_prompt_identity(tree_run: Path, native_run: Path) -> dict[str, Any]:
    tree_rows = _request_rows(tree_run / "tree_request_metrics.jsonl")
    native_rows = _request_rows(native_run / "native_request_metrics.jsonl")
    compared = []
    missing = []
    mismatches = []
    for idx, (tree_row, native_row) in enumerate(zip(tree_rows, native_rows)):
        tree_tokens = tree_row.get("prompt_token_ids")
        native_tokens = native_row.get("prompt_token_ids")
        if tree_tokens is None or native_tokens is None:
            missing.append(
                {
                    "pair_index": idx,
                    "tree_prompt_id": tree_row.get("prompt_id"),
                    "native_prompt_id": native_row.get("prompt_id"),
                    "tree_has_prompt_token_ids": tree_tokens is not None,
                    "native_has_prompt_token_ids": native_tokens is not None,
                }
            )
            continue
        same = [int(x) for x in tree_tokens] == [int(x) for x in native_tokens]
        item = {
            "pair_index": idx,
            "tree_prompt_id": tree_row.get("prompt_id"),
            "native_prompt_id": native_row.get("prompt_id"),
            "tree_prompt_token_count": len(tree_tokens),
            "native_prompt_token_count": len(native_tokens),
            "byte_identical": same,
        }
        compared.append(item)
        if not same:
            prefix = _common_prefix(
                [int(x) for x in tree_tokens],
                [int(x) for x in native_tokens],
            )
            item["common_prefix_tokens"] = prefix
            item["tree_first_diff"] = (
                int(tree_tokens[prefix]) if prefix < len(tree_tokens) else None
            )
            item["native_first_diff"] = (
                int(native_tokens[prefix]) if prefix < len(native_tokens) else None
            )
            mismatches.append(item)
    return {
        "tree_request_rows": len(tree_rows),
        "native_request_rows": len(native_rows),
        "row_count_mismatch": len(tree_rows) != len(native_rows),
        "compared_rows": len(compared),
        "missing_prompt_token_rows": missing,
        "mismatches": mismatches,
        "byte_identical": bool(compared)
        and not missing
        and not mismatches
        and len(tree_rows) == len(native_rows),
        "rows": compared,
    }


def _prompt_identity(tree_run: Path, native_run: Path) -> dict[str, Any]:
    summary = _compute_prompt_identity(tree_run, native_run)
    if not summary["byte_identical"]:
        raise PromptIdentityError(summary)
    return summary


def _first_layer_delta(tree_hidden: Path, native_hidden: Path, *, tree_row: int, native_row: int) -> dict[str, Any]:
    tree = _load_pt(tree_hidden)
    native = _load_pt(native_hidden)
    layers = []
    first_nonzero = None
    for idx, (tree_layer, native_layer) in enumerate(zip(tree["layers"], native["layers"])):
        delta = _max_abs(tree_layer["hidden"][tree_row], native_layer["hidden"][native_row])
        item = {
            "layer_idx": int(idx),
            "tree_layer_type": tree_layer.get("layer_type"),
            "native_layer_type": native_layer.get("layer_type"),
            "hidden_max_abs": delta,
        }
        layers.append(item)
        if first_nonzero is None and delta != 0.0:
            first_nonzero = item
    return {
        "tree_num_tokens": int(tree["num_tokens"]),
        "native_num_tokens": int(native["num_tokens"]),
        "tree_rows": [int(x) for x in tree["rows"]],
        "native_rows": [int(x) for x in native["rows"]],
        "input_max_abs": _max_abs(tree["input_hidden"][tree_row], native["input_hidden"][native_row]),
        "final_norm_max_abs": _max_abs(
            tree["final_norm_hidden"][tree_row],
            native["final_norm_hidden"][native_row],
        ),
        "first_nonzero_layer": first_nonzero or {"where": "none"},
        "layers": layers,
    }


def _exact_row0_pairs(tree_run: Path, native_run: Path, *, limit: int) -> list[dict[str, Any]]:
    pairs = []
    for tree_call in range(limit):
        tree_path = tree_run / "logs" / f"tree_layer_hidden.call{tree_call}.pt"
        if not tree_path.exists():
            continue
        tree = _load_pt(tree_path)
        for native_call in range(limit):
            native_path = native_run / "logs" / f"native_layer_hidden.call{native_call}.pt"
            if not native_path.exists():
                continue
            native = _load_pt(native_path)
            input_delta = _max_abs(tree["input_hidden"][0], native["input_hidden"][0])
            if input_delta != 0.0:
                continue
            layer_deltas = [
                _max_abs(tree_layer["hidden"][0], native_layer["hidden"][0])
                for tree_layer, native_layer in zip(tree["layers"], native["layers"])
            ]
            final_delta = _max_abs(tree["final_norm_hidden"][0], native["final_norm_hidden"][0])
            pairs.append(
                {
                    "tree_call": int(tree_call),
                    "native_call": int(native_call),
                    "input_max_abs": input_delta,
                    "max_layer_max_abs": float(max(layer_deltas) if layer_deltas else 0.0),
                    "final_norm_max_abs": final_delta,
                    "all_layers_bit_exact": all(delta == 0.0 for delta in layer_deltas)
                    and final_delta == 0.0,
                }
            )
    return pairs


def _tree_lcp_rows(tree_run: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in _load_jsonl(tree_run / "logs" / "tree_path_lcp.jsonl")
        if row.get("event") == "tree_path_lcp_max"
    ]


def _native_drafts(native_run: Path) -> list[list[int]]:
    path = native_run / "logs" / "fr10_mtp_draft_trace.jsonl"
    if not path.exists():
        return []
    drafts = []
    for row in _load_jsonl(path):
        draft = row.get("draft")
        if (
            row.get("event") == "mtp_draft"
            and isinstance(draft, list)
            and draft
            and isinstance(draft[0], list)
        ):
            drafts.append([int(x) for x in draft[0]])
    return drafts


def _native_argmax_rows(native_run: Path, *, limit: int) -> list[list[int]]:
    rows = []
    for call in range(limit):
        path = native_run / "logs" / f"native_final_logits.call{call}.pt"
        if not path.exists():
            continue
        logits = _load_pt(path)["logits"].float()
        rows.append([int(logits[idx].argmax().item()) for idx in range(logits.shape[0])])
    return rows


def _native_emitted_events(native_run: Path, *, limit: int) -> list[dict[str, Any]]:
    drafts = _native_drafts(native_run)
    argmax_rows = _native_argmax_rows(native_run, limit=limit)
    events = []
    for idx, argmax in enumerate(argmax_rows):
        draft = drafts[idx] if idx < len(drafts) else []
        accepted = 0
        for draft_token, target_token in zip(draft, argmax):
            if int(draft_token) != int(target_token):
                break
            accepted += 1
        emitted = list(draft[:accepted])
        if accepted < len(argmax):
            emitted.append(int(argmax[accepted]))
        events.append(
            {
                "call": int(idx),
                "draft_token_ids": draft,
                "target_argmax_ids": argmax,
                "accepted_len": int(accepted),
                "emitted_tokens": emitted,
            }
        )
    return events


def _tree_emitted_events(tree_lcp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for idx, row in enumerate(tree_lcp):
        events.append(
            {
                "call": int(idx),
                "winner_path": row.get("winner_path"),
                "accepted_len": int(row.get("accepted_len") or 0),
                "draft_token_ids": [int(x) for x in row.get("draft_token_ids") or []],
                "parent_target_ids": [int(x) for x in row.get("parent_target_ids") or []],
                "self_target_ids": [int(x) for x in row.get("self_target_ids") or []],
                "emitted_tokens": [int(x) for x in row.get("emitted_tokens") or []],
            }
        )
    return events


def _stream(events: list[dict[str, Any]]) -> list[int]:
    return [
        int(token)
        for event in events
        for token in event.get("emitted_tokens", [])
    ]


def _locate_tree_stream_pos(events: list[dict[str, Any]], stream_pos: int) -> dict[str, Any] | None:
    start = 0
    for event in events:
        emitted = event.get("emitted_tokens", [])
        end = start + len(emitted)
        if start <= stream_pos < end:
            local = stream_pos - start
            winner_path = event.get("winner_path") or []
            accepted_len = int(event.get("accepted_len") or 0)
            if accepted_len > 0 and local < accepted_len and local < len(winner_path):
                row = int(winner_path[local])
            else:
                row = 0
            return {
                "call": int(event["call"]),
                "local_emitted_index": int(local),
                "row": int(row),
                "stream_start": int(start),
                "stream_end": int(end),
                "event": event,
            }
        start = end
    return None


def _locate_native_stream_pos(events: list[dict[str, Any]], stream_pos: int) -> dict[str, Any] | None:
    start = 0
    for event in events:
        emitted = event.get("emitted_tokens", [])
        end = start + len(emitted)
        if start <= stream_pos < end:
            local = stream_pos - start
            return {
                "call": int(event["call"]),
                "local_emitted_index": int(local),
                "row": int(local),
                "stream_start": int(start),
                "stream_end": int(end),
                "event": event,
            }
        start = end
    return None


def _argmax_stream_alignment(
    tree_lcp: list[dict[str, Any]],
    native_run: Path,
    *,
    served_tokens: list[int],
    limit: int,
) -> dict[str, Any]:
    tree_events = _tree_emitted_events(tree_lcp)
    native_events = _native_emitted_events(native_run, limit=limit)
    tree_stream = _stream(tree_events)
    native_stream = _stream(native_events)

    served_prefix_offset = None
    max_offset = min(len(served_tokens), 8)
    for offset in range(max_offset + 1):
        if served_tokens[offset : offset + min(len(tree_stream), len(served_tokens) - offset)] == tree_stream[
            : min(len(tree_stream), len(served_tokens) - offset)
        ]:
            served_prefix_offset = offset
            break
    served_stream_len = (
        max(0, len(served_tokens) - served_prefix_offset)
        if served_prefix_offset is not None
        else min(len(tree_stream), len(native_stream))
    )
    observed_tree_stream = tree_stream[:served_stream_len]
    observed_native_stream = native_stream[:served_stream_len]
    alignment = _token_alignment(observed_tree_stream, observed_native_stream)
    uncensored_alignment = _token_alignment(tree_stream, native_stream)

    first_flip = None
    if alignment["first_mismatch"] is not None:
        mismatch = alignment["first_mismatch"]
        tree_location = _locate_tree_stream_pos(tree_events, int(mismatch["position"]))
        native_location = _locate_native_stream_pos(native_events, int(mismatch["position"]))
        first_flip = {
            "stream_position": mismatch["position"],
            "completion_position": (
                int(mismatch["position"] + served_prefix_offset)
                if served_prefix_offset is not None
                else None
            ),
            "tree_token": mismatch["left_token"],
            "native_token": mismatch["right_token"],
            "tree_location": tree_location,
            "native_location": native_location,
        }

    return {
        "tree_events": tree_events,
        "native_events": native_events,
        "tree_emitted_stream": tree_stream,
        "native_emitted_stream": native_stream,
        "served_compared_stream_tokens": int(served_stream_len),
        "stream_alignment": alignment,
        "uncensored_stream_alignment": uncensored_alignment,
        "served_prefix_offset_for_tree_stream": served_prefix_offset,
        "first_argmax_flip": first_flip,
    }


def reduce(tree_run: Path, native_run: Path, out: Path, *, limit: int) -> dict[str, Any]:
    try:
        prompt_identity = _prompt_identity(tree_run, native_run)
    except PromptIdentityError as exc:
        result = {
            "schema": "fr13.argmax_lcp_localize.v1",
            "tree_run": str(tree_run),
            "native_run": str(native_run),
            "valid": False,
            "invalid_reason": "prompt_token_identity_guard_failed",
            "prompt_identity": exc.summary,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise

    tree_lcp = _tree_lcp_rows(tree_run)
    native_final0 = _load_pt(native_run / "logs" / "native_final_logits.call0.pt")
    native_argmax0 = int(native_final0["logits"][0].float().argmax().item())
    tree_argmax0 = int(tree_lcp[0]["parent_target_ids"][0])
    first_delta = _first_layer_delta(
        tree_run / "logs" / "tree_layer_hidden.call0.pt",
        native_run / "logs" / "native_layer_hidden.call0.pt",
        tree_row=0,
        native_row=0,
    )
    exact_pairs = _exact_row0_pairs(tree_run, native_run, limit=limit)

    tree_records = _records(tree_run / "tree_greedy_probe.json")
    native_records = _records(native_run / "native_greedy_probe.json")
    tree_tokens0 = [int(x) for x in (tree_records[0].get("token_ids") if tree_records else [])]
    native_tokens0 = [int(x) for x in (native_records[0].get("token_ids") if native_records else [])]
    served_alignment = _token_alignment(tree_tokens0, native_tokens0)
    argmax_alignment = _argmax_stream_alignment(
        tree_lcp,
        native_run,
        served_tokens=tree_tokens0,
        limit=limit,
    )
    first_flip = argmax_alignment["first_argmax_flip"]
    if first_flip is not None:
        tree_location = first_flip.get("tree_location") or {}
        native_location = first_flip.get("native_location") or {}
        tree_call = tree_location.get("call")
        native_call = native_location.get("call")
        tree_row = tree_location.get("row")
        native_row = native_location.get("row")
        if (
            tree_call is not None
            and native_call is not None
            and tree_row is not None
            and native_row is not None
        ):
            tree_path = tree_run / "logs" / f"tree_layer_hidden.call{tree_call}.pt"
            native_path = native_run / "logs" / f"native_layer_hidden.call{native_call}.pt"
            if tree_path.exists() and native_path.exists():
                first_flip["layer_delta"] = _first_layer_delta(
                    tree_path,
                    native_path,
                    tree_row=int(tree_row),
                    native_row=int(native_row),
                )

    result = {
        "schema": "fr13.argmax_lcp_localize.v1",
        "tree_run": str(tree_run),
        "native_run": str(native_run),
        "method": (
            "B1 eager paired greedy/temp0; tree target argmax from "
            "tree_path_lcp_max.parent_target_ids; native target argmax from "
            "native_final_logits.call*.pt"
        ),
        "valid": True,
        "prompt_identity": prompt_identity,
        "capture_filters": {
            "tree_hidden_num_tokens": 10,
            "native_hidden_num_tokens": 6,
            "tree_rows": list(range(10)),
            "native_rows": list(range(6)),
        },
        "depth0_call0_argmax": {
            "tree_call": 0,
            "native_call": 0,
            "depth": 0,
            "tree_argmax": tree_argmax0,
            "native_argmax": native_argmax0,
            "argmax_match": tree_argmax0 == native_argmax0,
            "tree_lcp_row": {
                key: tree_lcp[0].get(key)
                for key in (
                    "accepted_len",
                    "path0_lcp",
                    "winner_path",
                    "draft_token_ids",
                    "parent_target_ids",
                    "self_target_ids",
                    "emitted_tokens",
                    "native_bonus_token",
                )
            },
            "layer_delta": first_delta,
            "verdict": (
                "argmax_flip"
                if tree_argmax0 != native_argmax0
                else "argmax_match_hidden_drift_observed"
                if first_delta["first_nonzero_layer"].get("where") != "none"
                else "argmax_match_bit_exact"
            ),
        },
        "served_output_alignment": {
            "tree_prompt0_tokens": tree_tokens0,
            "native_prompt0_tokens": native_tokens0,
            **served_alignment,
        },
        "argmax_stream_alignment": argmax_alignment,
        "first_argmax_flip": first_flip,
        "exact_row0_verify_pairs": exact_pairs,
        "exact_row0_verify_pair_count": len(exact_pairs),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-run", required=True, type=Path)
    parser.add_argument("--native-run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    result = reduce(args.tree_run, args.native_run, args.out, limit=args.limit)
    print(
        json.dumps(
            {
                "prompt_identity": result["prompt_identity"],
                "served_output_alignment": result["served_output_alignment"],
                "first_argmax_flip": result["first_argmax_flip"],
                "depth0_call0_argmax": result["depth0_call0_argmax"],
                "exact_row0_verify_pair_count": result["exact_row0_verify_pair_count"],
                "exact_row0_verify_pairs_head": result["exact_row0_verify_pairs"][:5],
                "out": str(args.out),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

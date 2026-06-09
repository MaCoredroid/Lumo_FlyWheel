#!/usr/bin/env python3
"""CPU-only characterization for the guarded FR13 prompt-0 pos-8 flip."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.fr13_argmax_lcp_localize import (  # noqa: E402
    _first_layer_delta,
    _load_jsonl,
    _max_abs,
    _native_emitted_events,
    _tree_emitted_events,
)


SUBSTATE_KEY_FRAGMENTS = (
    "h_recurrent",
    "h0",
    "conv",
    "conv_state",
    "recurrent_state",
)


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _tree_lcp_rows(run: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in _load_jsonl(run / "tree/logs/tree_path_lcp.jsonl")
        if row.get("event") == "tree_path_lcp_max"
    ]


def _native_argmax(run: Path, call: int) -> list[int]:
    logits = _load_pt(run / f"native/logs/native_final_logits.call{call}.pt")["logits"].float()
    return [int(logits[row].argmax().item()) for row in range(logits.shape[0])]


def _first_raw_row_mismatch(run: Path) -> dict[str, Any] | None:
    tree_lcp = _tree_lcp_rows(run)
    for call, row in enumerate(tree_lcp):
        native_arg = _native_argmax(run, call)
        tree_parent = [int(x) for x in row.get("parent_target_ids", [])]
        for idx, (tree_token, native_token) in enumerate(zip(tree_parent, native_arg)):
            if int(tree_token) != int(native_token):
                return {
                    "call": int(call),
                    "row": int(idx),
                    "tree_parent_target": int(tree_token),
                    "native_argmax": int(native_token),
                    "caveat": (
                        "raw same-index tree branch rows are not necessarily "
                        "native sequential-path comparable"
                    ),
                }
    return None


def _stream_first_flip(run: Path) -> dict[str, Any]:
    tree_events = _tree_emitted_events(_tree_lcp_rows(run))
    native_events = _native_emitted_events(run / "native", limit=16)

    def _tree_stream() -> list[dict[str, Any]]:
        out = []
        for event in tree_events:
            winner_path = event.get("winner_path") or []
            accepted_len = int(event.get("accepted_len") or 0)
            for local, token in enumerate(event["emitted_tokens"]):
                row = (
                    int(winner_path[local])
                    if accepted_len > 0 and local < accepted_len and local < len(winner_path)
                    else 0
                )
                out.append(
                    {
                        "token": int(token),
                        "call": int(event["call"]),
                        "row": row,
                        "local_emitted_index": int(local),
                        "event": event,
                    }
                )
        return out

    def _native_stream() -> list[dict[str, Any]]:
        out = []
        for event in native_events:
            for local, token in enumerate(event["emitted_tokens"]):
                out.append(
                    {
                        "token": int(token),
                        "call": int(event["call"]),
                        "row": int(local),
                        "local_emitted_index": int(local),
                        "event": event,
                    }
                )
        return out

    for pos, (tree_item, native_item) in enumerate(zip(_tree_stream(), _native_stream())):
        if tree_item["token"] != native_item["token"]:
            return {
                "stream_position": int(pos),
                "completion_position": int(pos + 1),
                "tree_token": int(tree_item["token"]),
                "native_token": int(native_item["token"]),
                "tree_call": int(tree_item["call"]),
                "tree_row": int(tree_item["row"]),
                "native_call": int(native_item["call"]),
                "native_row": int(native_item["row"]),
                "tree_event": tree_item["event"],
                "native_event": native_item["event"],
            }
    raise RuntimeError("no stream flip found")


def _row_layer_summary(run: Path, call: int, row: int) -> dict[str, Any]:
    return _first_layer_delta(
        run / f"tree/logs/tree_layer_hidden.call{call}.pt",
        run / f"native/logs/native_layer_hidden.call{call}.pt",
        tree_row=row,
        native_row=row,
    )


def _call_row_inputs(run: Path, call: int, rows: int = 6) -> list[dict[str, Any]]:
    tree = _load_pt(run / f"tree/logs/tree_layer_hidden.call{call}.pt")
    native = _load_pt(run / f"native/logs/native_layer_hidden.call{call}.pt")
    out = []
    n = min(rows, tree["input_hidden"].shape[0], native["input_hidden"].shape[0])
    for row in range(n):
        first = None
        for idx, (tree_layer, native_layer) in enumerate(zip(tree["layers"], native["layers"])):
            delta = _max_abs(tree_layer["hidden"][row], native_layer["hidden"][row])
            if delta != 0.0:
                first = {
                    "layer_idx": int(idx),
                    "layer_type": tree_layer.get("layer_type"),
                    "hidden_max_abs": delta,
                }
                break
        out.append(
            {
                "row": int(row),
                "input_max_abs": _max_abs(tree["input_hidden"][row], native["input_hidden"][row]),
                "first_nonzero_layer": first,
            }
        )
    return out


def _native_margin(run: Path, call: int, row: int, tree_token: int, native_token: int) -> dict[str, Any]:
    logits = _load_pt(run / f"native/logs/native_final_logits.call{call}.pt")["logits"].float()[row]
    values, indices = torch.topk(logits, 10)
    return {
        "native_call": int(call),
        "native_row": int(row),
        "native_argmax": int(native_token),
        "tree_token": int(tree_token),
        "native_argmax_logit": float(logits[native_token].item()),
        "tree_token_logit_in_native": float(logits[tree_token].item()),
        "native_argmax_minus_tree_token": float((logits[native_token] - logits[tree_token]).item()),
        "native_top1_minus_top2": float((values[0] - values[1]).item()),
        "native_top10": [
            {"token": int(token), "logit": float(value)}
            for token, value in zip(indices.tolist(), values.tolist())
        ],
    }


def _capture_inventory(run: Path) -> dict[str, Any]:
    files = sorted((run / "tree/logs").glob("*.pt")) + sorted((run / "native/logs").glob("*.pt"))
    hits = []
    for path in files:
        obj = _load_pt(path)
        keys = sorted(str(key) for key in obj.keys())
        matching = [
            key
            for key in keys
            if any(fragment in key.lower() for fragment in SUBSTATE_KEY_FRAGMENTS)
        ]
        if matching:
            hits.append({"path": str(path), "keys": matching})
    return {
        "pt_files_checked": len(files),
        "gdn_substate_keys_present": bool(hits),
        "matching_keys": hits,
        "required_substates": ["h_recurrent", "conv/conv_state"],
    }


def characterize(run: Path) -> dict[str, Any]:
    stream_flip = _stream_first_flip(run)
    layer_delta = _row_layer_summary(
        run,
        call=int(stream_flip["tree_call"]),
        row=int(stream_flip["tree_row"]),
    )
    call2_rows = _call_row_inputs(run, 2)
    row0_first_by_call = []
    for call in range(5):
        summary = _row_layer_summary(run, call=call, row=0)
        row0_first_by_call.append(
            {
                "call": int(call),
                "input_max_abs": summary["input_max_abs"],
                "first_nonzero_layer": summary["first_nonzero_layer"],
                "final_norm_max_abs": summary["final_norm_max_abs"],
            }
        )
    return {
        "schema": "fr13.pos8_flip_cpu_characterize.v1",
        "run": str(run),
        "raw_same_index_first_mismatch": _first_raw_row_mismatch(run),
        "authoritative_stream_first_flip": stream_flip,
        "first_flip_layer_delta": {
            "input_max_abs": layer_delta["input_max_abs"],
            "first_nonzero_layer": layer_delta["first_nonzero_layer"],
            "final_norm_max_abs": layer_delta["final_norm_max_abs"],
        },
        "native_margin_at_flip": _native_margin(
            run,
            call=int(stream_flip["native_call"]),
            row=int(stream_flip["native_row"]),
            tree_token=int(stream_flip["tree_token"]),
            native_token=int(stream_flip["native_token"]),
        ),
        "call2_same_index_row_inputs": call2_rows,
        "call2_input_divergent_rows": [
            row["row"] for row in call2_rows if float(row["input_max_abs"]) != 0.0
        ],
        "row0_first_nonzero_by_call": row0_first_by_call,
        "capture_inventory": _capture_inventory(run),
        "verdict": {
            "flip_valid": True,
            "cause_determined": False,
            "not_supported": ["GDN recurrent-state writeback root cause"],
            "classification_from_existing_captures": (
                "above-floor model-path divergence at/through layer0 for the "
                "first comparable stream flip; exact GDN substate cause is not "
                "captured"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = characterize(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["verdict"], indent=2, sort_keys=True))
    print(json.dumps(result["authoritative_stream_first_flip"], indent=2, sort_keys=True))
    print(json.dumps(result["first_flip_layer_delta"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

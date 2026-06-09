import json
import importlib.util
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


branch_oracle = _load_script(
    "fr13_branch_token_oracle", REPO / "scripts" / "fr13_branch_token_oracle.py"
)
AlignmentError = branch_oracle.AlignmentError
align_events = branch_oracle.align_events


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_align_events_reconstructs_tree_output(tmp_path):
    tree_run = tmp_path / "tree"
    _write_json(
        tree_run / "tree_greedy_probe.json",
        {
            "records": [
                {
                    "prompt_id": 0,
                    "sample_index": 0,
                    "prompt_token_ids": [101, 102],
                    "token_ids": [11, 12, 13],
                }
            ]
        },
    )
    _write_jsonl(
        tree_run / "tree_request_metrics.jsonl",
        [{"prompt_id": 0, "sample_index": 0, "batch_size": 1}],
    )
    _write_jsonl(
        tree_run / "logs" / "tree_path_lcp.jsonl",
        [
            {
                "event": "tree_path_lcp_max",
                "req_index": 0,
                "emitted_tokens": [11, 12],
                "draft_token_ids": [1, 2],
                "self_target_ids": [3, 4],
                "path_scores": [{"path": [0]}, {"path": [0, 1]}],
            },
            {
                "event": "tree_path_lcp_max",
                "req_index": 0,
                "emitted_tokens": [13],
                "draft_token_ids": [5],
                "self_target_ids": [6],
                "path_scores": [{"path": [0]}],
            },
        ],
    )

    events = align_events(tree_run)

    assert len(events) == 2
    assert events[0].served_prefix == []
    assert events[1].served_prefix == [11, 12]


def test_align_events_allows_leading_gap_and_truncated_final_event(tmp_path):
    tree_run = tmp_path / "tree"
    _write_json(
        tree_run / "tree_greedy_probe.json",
        {
            "records": [
                {
                    "prompt_id": 0,
                    "sample_index": 0,
                    "prompt_token_ids": [101, 102],
                    "token_ids": [7, 11, 12, 13],
                }
            ]
        },
    )
    _write_jsonl(
        tree_run / "tree_request_metrics.jsonl",
        [{"prompt_id": 0, "sample_index": 0, "batch_size": 1}],
    )
    _write_jsonl(
        tree_run / "logs" / "tree_path_lcp.jsonl",
        [
            {
                "event": "tree_path_lcp_max",
                "req_index": 0,
                "emitted_tokens": [11, 12],
                "draft_token_ids": [1, 2],
                "self_target_ids": [3, 4],
                "path_scores": [{"path": [0]}, {"path": [0, 1]}],
            },
            {
                "event": "tree_path_lcp_max",
                "req_index": 0,
                "emitted_tokens": [13, 14, 15],
                "draft_token_ids": [5],
                "self_target_ids": [6],
                "path_scores": [{"path": [0]}],
            },
        ],
    )

    events = align_events(tree_run)

    assert len(events) == 1
    assert events[0].alignment_gap == [7]
    assert events[0].served_prefix == [7]
    assert events[0].event_start == 1
    assert events[0].event_end == 3


def test_align_events_raises_on_emitted_token_mismatch(tmp_path):
    tree_run = tmp_path / "tree"
    _write_json(
        tree_run / "tree_greedy_probe.json",
        {
            "records": [
                {
                    "prompt_id": 0,
                    "sample_index": 0,
                    "prompt_token_ids": [101, 102],
                    "token_ids": [11],
                }
            ]
        },
    )
    _write_jsonl(
        tree_run / "tree_request_metrics.jsonl",
        [{"prompt_id": 0, "sample_index": 0, "batch_size": 1}],
    )
    _write_jsonl(
        tree_run / "logs" / "tree_path_lcp.jsonl",
        [{"event": "tree_path_lcp_max", "req_index": 0, "emitted_tokens": [99]}],
    )

    with pytest.raises(AlignmentError, match="cannot be aligned"):
        align_events(tree_run)

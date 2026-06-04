from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fr10_quick_decode_tps_probe.py"
spec = importlib.util.spec_from_file_location("fr10_quick_decode_tps_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(probe)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_tree_engagement_assertion_accepts_ok_metadata_and_tree_rows(tmp_path: Path) -> None:
    debug = tmp_path / "tree_sampler_debug.jsonl"
    accept = tmp_path / "tree_path_lcp_max.jsonl"
    _write_jsonl(
        debug,
        [
            {
                "event": "gpu_tree_metadata",
                "reason": "ok",
                "has_tree_parent_indices": True,
                "num_draft_tokens": [9],
            }
        ],
    )
    _write_jsonl(
        accept,
        [{"event": "tree_sample_accept", "accepted_len": 1, "accepted_node_ids": [0]}],
    )

    summary = probe._assert_tree_engagement(
        sampler_debug_path=debug,
        tree_accept_path=accept,
        expected_draft_count=9,
    )

    assert summary["engaged"] is True
    assert summary["gpu_tree_metadata_ok_rows"] == 1
    assert summary["tree_accept_rows"] == 1


def test_tree_engagement_assertion_accepts_padded_batch_draft_counts(tmp_path: Path) -> None:
    debug = tmp_path / "tree_sampler_debug.jsonl"
    accept = tmp_path / "tree_path_lcp_max.jsonl"
    _write_jsonl(
        debug,
        [
            {
                "event": "gpu_tree_metadata",
                "reason": "ok",
                "has_tree_parent_indices": True,
                "num_draft_tokens": [9, 9, 0, 0],
            }
        ],
    )
    _write_jsonl(accept, [{"event": "tree_sample_accept", "accepted_len": 1}])

    summary = probe._assert_tree_engagement(
        sampler_debug_path=debug,
        tree_accept_path=accept,
        expected_draft_count=9,
    )

    assert summary["engaged"] is True
    assert summary["gpu_tree_metadata_ok_rows"] == 1


def test_tree_engagement_assertion_rejects_draft_count_mismatch(tmp_path: Path) -> None:
    debug = tmp_path / "tree_sampler_debug.jsonl"
    accept = tmp_path / "tree_path_lcp_max.jsonl"
    _write_jsonl(
        debug,
        [
            {
                "event": "gpu_tree_metadata",
                "reason": "draft_count_mismatch:10!=9",
                "has_tree_parent_indices": False,
                "num_draft_tokens": [10],
            }
        ],
    )
    _write_jsonl(accept, [])

    with pytest.raises(RuntimeError, match="tree_mtp engagement assertion failed"):
        probe._assert_tree_engagement(
            sampler_debug_path=debug,
            tree_accept_path=accept,
            expected_draft_count=9,
        )

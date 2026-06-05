from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_spec_per_position.py"
spec = importlib.util.spec_from_file_location("measure_spec_per_position", SCRIPT)
measure = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(measure)


def test_spine_a_lcp_prefers_path_scores() -> None:
    record = {
        "accepted_node_ids": [1, 3, 5],
        "path_scores": [
            {"path": [0, 2, 4, 6, 8], "lcp": 3},
            {"path": [1, 3, 5, 7, 9], "lcp": 3},
        ],
    }

    assert measure.lcp_from_tree_record(record) == 3


def test_spine_a_lcp_falls_back_to_leading_even_nodes() -> None:
    assert measure.lcp_from_tree_record({"accepted_node_ids": [0, 2, 9]}) == 2
    assert measure.lcp_from_tree_record({"accepted_node_ids": [1, 3, 5]}) == 0
    assert measure.lcp_from_tree_record({"accepted_node_ids": [0, 2, 4, 6, 8]}) == 5


def test_per_position_summary() -> None:
    summary = measure.summarize_lcps([0, 1, 3, 5])

    assert summary["n_events"] == 4
    assert summary["avg"] == pytest.approx(2.25)
    assert summary["acc0"] == pytest.approx(0.25)
    assert summary["full5"] == pytest.approx(0.25)
    assert summary["per_position"] == pytest.approx([0.75, 0.5, 0.5, 0.25, 0.25])


def test_sampling_guard_rejects_non_greedy() -> None:
    with pytest.raises(measure.GuardError, match="temperature"):
        measure.validate_sampling(0.6, 1.0)
    with pytest.raises(measure.GuardError, match="top_p"):
        measure.validate_sampling(0.0, 0.95)


def test_batch_invariance_guard_requires_env_and_supported_attention_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(measure, "_http_json", lambda *args, **kwargs: {})
    with pytest.raises(measure.GuardError, match="VLLM_BATCH_INVARIANT=1"):
        measure.validate_live_server(
            "http://127.0.0.1:9950",
            {"env_has_vllm_batch_invariant": False, "cmdline_has_flash_attn": True},
        )
    with pytest.raises(measure.GuardError, match="FLASH_ATTN or --attention-backend TREE_ATTN"):
        measure.validate_live_server(
            "http://127.0.0.1:9950",
            {"env_has_vllm_batch_invariant": True, "cmdline_has_flash_attn": False},
        )
    measure.validate_live_server(
        "http://127.0.0.1:9950",
        {
            "env_has_vllm_batch_invariant": True,
            "cmdline_has_flash_attn": False,
            "cmdline_has_tree_attn": True,
        },
    )

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


localizer = _load_script(
    "fr13_argmax_lcp_localize", REPO / "scripts" / "fr13_argmax_lcp_localize.py"
)
measure = _load_script("fr13_e2e_measure", REPO / "scripts" / "fr13_e2e_measure.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_request_metrics(tree_run: Path, native_run: Path) -> None:
    _write_jsonl(
        tree_run / "tree_request_metrics.jsonl",
        [
            {
                "prompt_id": 0,
                "prompt_token_ids": [1, 2, 3, 4],
                "prompt_token_count": 4,
            }
        ],
    )
    _write_jsonl(
        native_run / "native_request_metrics.jsonl",
        [
            {
                "prompt_id": 0,
                "prompt_token_ids": [1, 2, 9, 4],
                "prompt_token_count": 4,
            }
        ],
    )


def test_prompt_identity_raises_on_mismatched_prompt_tokens(tmp_path: Path) -> None:
    tree_run = tmp_path / "tree"
    native_run = tmp_path / "native"
    _write_request_metrics(tree_run, native_run)

    with pytest.raises(SystemExit, match="prompt_token_ids differ") as exc:
        localizer._prompt_identity(tree_run, native_run)

    assert exc.value.summary["byte_identical"] is False
    assert exc.value.summary["mismatches"][0]["common_prefix_tokens"] == 2
    assert exc.value.summary["mismatches"][0]["tree_first_diff"] == 3
    assert exc.value.summary["mismatches"][0]["native_first_diff"] == 9


def test_orchestrator_fails_closed_before_reducers_on_prompt_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tree_run = run_dir / "tree"
    native_run = run_dir / "native"
    out = run_dir / "fr13_e2e_measure.json"
    _write_request_metrics(tree_run, native_run)
    args = argparse.Namespace(out=out, argmax_limit=16)

    with pytest.raises(SystemExit, match="prompt_token_ids differ"):
        measure.reduce_arms(args, run_dir, tree_run, native_run)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["valid"] is False
    assert payload["invalid_reason"] == "prompt_token_identity_guard_failed"
    assert payload["prompt_identity"]["mismatches"][0]["pair_index"] == 0
    assert "argmax_localize" not in payload
    assert "deliverable_compare" not in payload

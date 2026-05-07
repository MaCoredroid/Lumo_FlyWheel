from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_track_b_e2e_task as runner  # noqa: E402


def test_runner_normalizes_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(
        json.dumps(
            {
                "request_id": "req-1",
                "prompt_tokens": 50,
                "generation_tokens": 12,
                "prefill_s": 0.2,
                "decode_s": 1.0,
                "spec_decode_num_accepted_tokens": 3,
                "spec_decode_num_draft_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    runner._write_vllm_per_turn_from_jsonl(task_dir, source)

    payload = json.loads((task_dir / "vllm_per_turn.json").read_text(encoding="utf-8"))
    metrics = payload["requests"]["req-1"]
    assert metrics["completion_tokens"] == 12
    assert metrics["prefill_sum_s"] == 0.2
    assert metrics["decode_sum_s"] == 1.0
    assert metrics["decode_tps"] == 12.0
    assert metrics["accepted_per_draft_token"] == 0.25
    assert (task_dir / "vllm_request_metrics.jsonl").is_file()


def test_runner_rejects_incomplete_vllm_request_metrics_jsonl(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    source = tmp_path / "vllm_requests.jsonl"
    source.write_text(json.dumps({"request_id": "req-1", "prompt_tokens": 50}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing numeric fields"):
        runner._write_vllm_per_turn_from_jsonl(task_dir, source)

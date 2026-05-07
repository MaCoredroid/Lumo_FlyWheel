from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_track_b_codex_trace_correctness as verifier  # noqa: E402


def _write_task_evidence(root: Path, name: str, *, model: str = "ok", tools: str = "[]", score: float = 1.0) -> dict[str, object]:
    enabled_model = root / f"{name}_enabled_model.txt"
    disabled_model = root / f"{name}_disabled_model.txt"
    enabled_tools = root / f"{name}_enabled_tools.json"
    disabled_tools = root / f"{name}_disabled_tools.json"
    enabled_scores = root / f"{name}_enabled_scores.json"
    disabled_scores = root / f"{name}_disabled_scores.json"
    enabled_trace = root / f"{name}_enabled_trace.jsonl"
    enabled_model.write_text(model, encoding="utf-8")
    disabled_model.write_text(model, encoding="utf-8")
    enabled_tools.write_text(tools, encoding="utf-8")
    disabled_tools.write_text(tools, encoding="utf-8")
    enabled_scores.write_text(json.dumps({"M1": score}) + "\n", encoding="utf-8")
    disabled_scores.write_text(json.dumps({"M1": score}) + "\n", encoding="utf-8")
    task_id = f"{name}/v1-clean-baseline"
    enabled_trace.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "event": "task_start",
                    "task_id": task_id,
                    "runtime_config_hash": "sha256:test",
                    "ts": "2026-05-07T21:30:00.000Z",
                },
                {
                    "event": "turn_start",
                    "turn": 0,
                    "regime": "plan",
                    "vllm_request_id": "req-1",
                    "ts": "2026-05-07T21:30:00.100Z",
                },
                {
                    "event": "task_end",
                    "ts": "2026-05-07T21:30:01.000Z",
                    "exit_code": 0,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "task_id": task_id,
        "trace_out_enabled_exit_code": 0,
        "trace_out_disabled_exit_code": 0,
        "trace_out_enabled_trace_jsonl": enabled_trace.name,
        "trace_out_enabled_model_outputs": enabled_model.name,
        "trace_out_disabled_model_outputs": disabled_model.name,
        "trace_out_enabled_tool_call_sequence": enabled_tools.name,
        "trace_out_disabled_tool_call_sequence": disabled_tools.name,
        "trace_out_enabled_milestone_scores": enabled_scores.name,
        "trace_out_disabled_milestone_scores": disabled_scores.name,
    }


def test_trace_correctness_verifier_writes_passing_artifact(tmp_path: Path) -> None:
    tasks = [_write_task_evidence(tmp_path, f"task-{index}") for index in range(3)]
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps({"codex_version": "codex patched", "trace_out_supported": True, "tasks": tasks}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "codex_trace_emitter_correctness.json"

    payload = verifier.verify(Namespace(comparison_manifest=str(manifest), base_dir="", out=str(out)))

    assert payload["ok"] is True
    assert payload["schema"] == "lumo.track_b.codex_trace_correctness.v1"
    assert len(payload["tasks"]) == 3
    assert payload["tasks"][0]["trace_schema_valid"] is True
    assert out.is_file()


def test_trace_correctness_verifier_rejects_mismatched_evidence(tmp_path: Path) -> None:
    tasks = [_write_task_evidence(tmp_path, f"task-{index}") for index in range(3)]
    (tmp_path / "task-0_disabled_model.txt").write_text("different", encoding="utf-8")
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps({"codex_version": "codex patched", "trace_out_supported": True, "tasks": tasks}) + "\n",
        encoding="utf-8",
    )

    payload = verifier.verify(
        Namespace(
            comparison_manifest=str(manifest),
            base_dir="",
            out=str(tmp_path / "codex_trace_emitter_correctness.json"),
        )
    )

    assert payload["ok"] is False
    assert payload["tasks"][0]["model_outputs_byte_identical"] is False


def test_trace_correctness_verifier_rejects_trace_schema_gap(tmp_path: Path) -> None:
    tasks = [_write_task_evidence(tmp_path, f"task-{index}") for index in range(3)]
    (tmp_path / "task-0_enabled_trace.jsonl").write_text(
        json.dumps(
            {
                "event": "task_start",
                "task_id": "task-0/v1-clean-baseline",
                "ts": "2026-05-07T21:30:00.000Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "comparisons.json"
    manifest.write_text(
        json.dumps({"codex_version": "codex patched", "trace_out_supported": True, "tasks": tasks}) + "\n",
        encoding="utf-8",
    )

    payload = verifier.verify(
        Namespace(
            comparison_manifest=str(manifest),
            base_dir="",
            out=str(tmp_path / "codex_trace_emitter_correctness.json"),
        )
    )

    assert payload["ok"] is False
    assert payload["tasks"][0]["trace_schema_valid"] is False
    assert "task_start_runtime_config_hash_missing" in payload["tasks"][0]["trace_schema_reasons"]

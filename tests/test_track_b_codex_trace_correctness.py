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
    enabled_model.write_text(model, encoding="utf-8")
    disabled_model.write_text(model, encoding="utf-8")
    enabled_tools.write_text(tools, encoding="utf-8")
    disabled_tools.write_text(tools, encoding="utf-8")
    enabled_scores.write_text(json.dumps({"M1": score}) + "\n", encoding="utf-8")
    disabled_scores.write_text(json.dumps({"M1": score}) + "\n", encoding="utf-8")
    return {
        "task_id": f"{name}/v1-clean-baseline",
        "trace_out_enabled_exit_code": 0,
        "trace_out_disabled_exit_code": 0,
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

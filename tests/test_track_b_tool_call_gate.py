from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_gate_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_track_b_tool_call_gate.py"
    spec = importlib.util.spec_from_file_location("run_track_b_tool_call_gate", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_structural_tool_call_match_allows_argument_variants() -> None:
    gate = _load_gate_module()

    serial = {
        "name": "apply_patch",
        "status": "completed",
        "arguments": {"patch": "diff --git a/tool_gate_marker.txt b/tool_gate_marker.txt\n+tool gate"},
    }
    concurrent = {
        "name": "apply_patch",
        "status": "completed",
        "arguments": {"patch": "diff --git a/artifact/tool_gate_marker.txt b/artifact/tool_gate_marker.txt\n+tool gate"},
    }

    assert gate._calls_match(serial, concurrent, exact_arguments=False)
    assert not gate._calls_match(serial, concurrent, exact_arguments=True)


def test_structural_argument_validation_keeps_required_content() -> None:
    gate = _load_gate_module()
    case = {
        "expected_arguments": {"patch": "unused exact patch"},
        "required_contains": {"patch": ["tool_gate_marker", "tool gate"]},
    }

    assert gate._arguments_valid(
        case,
        {"patch": "diff --git a/tool_gate_marker.txt b/tool_gate_marker.txt\n+tool gate"},
        exact=False,
    )
    assert not gate._arguments_valid(case, {"patch": "diff --git a/other.txt b/other.txt\n+tool gate"}, exact=False)

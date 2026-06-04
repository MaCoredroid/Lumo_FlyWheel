from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fr10_serving_wiring_gate.py"
spec = importlib.util.spec_from_file_location("fr10_serving_wiring_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["fr10_serving_wiring_gate"] = gate
spec.loader.exec_module(gate)


def _write_counter(path: Path, diag: list[float]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "shape": "n10_pad16",
                    "has_sibling": True,
                    "count": 12,
                    "conv_diag": diag,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_serving_wiring_gate_passes_captured_conv_replay_and_fails_missing_full_commit(tmp_path: Path) -> None:
    counters = tmp_path / "counters.json"
    commit = tmp_path / "commit.jsonl"
    diag = [0.0] * 32
    diag[1] = 5.0
    diag[4] = 3.0
    diag[5] = 10.0
    diag[9] = 7.0
    _write_counter(counters, diag)
    commit.write_text(
        json.dumps({"event": "mamba_commit_copy_bias", "tree_accepted_row": 4}) + "\n",
        encoding="utf-8",
    )

    matrix, summary = gate.evaluate_wiring(counters_path=counters, commit_log_path=commit)
    by_gate = {row.gate: row for row in matrix}

    assert by_gate["conv_gather"].passed
    assert by_gate["conv_gather"].case == "wiring_op"
    assert by_gate["conv_gather"].regime == "deterministic_captured_replay_byte_exact"
    assert by_gate["conv_commit_state"].passed
    assert by_gate["sibling_leaf_no_trunk_mutation"].passed
    assert by_gate["scan_state_staging"].passed
    assert by_gate["postprocess_commit_redirect"].passed
    assert not by_gate["scan_output_serial_parity"].passed
    assert not by_gate["committed_state_native_replay"].passed
    assert not summary["passed"]


def test_serving_wiring_gate_requires_flat_negative_control_power(tmp_path: Path) -> None:
    counters = tmp_path / "counters.json"
    diag = [0.0] * 32
    diag[4] = 1.0
    _write_counter(counters, diag)

    matrix, _ = gate.evaluate_wiring(counters_path=counters)
    by_gate = {row.gate: row for row in matrix}

    assert not by_gate["conv_gather"].passed
    assert not by_gate["sibling_leaf_no_trunk_mutation"].passed

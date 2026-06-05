from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fr10_superset_gate_report.py"
spec = importlib.util.spec_from_file_location("fr10_superset_gate_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["fr10_superset_gate_report"] = report
spec.loader.exec_module(report)


def _write_tree(path: Path, accepted: list[tuple[int, list[int]]]) -> None:
    rows = []
    for idx, (accepted_len, nodes) in enumerate(accepted):
        rows.append(
            {
                "event": "tree_sample_accept",
                "req_index": 0,
                "accepted_len": accepted_len,
                "accepted_node_ids": nodes,
                "ts": float(idx),
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_path0_proxy_internal_gate_passes_when_tree_contains_path0(tmp_path: Path) -> None:
    tree_path = tmp_path / "tree.jsonl"
    _write_tree(tree_path, [(3, [0, 1, 3]), (1, [0])])
    events = report.load_tree_accept_trace(
        tree_path,
        speculative_token_tree=report.DEFAULT_CATERPILLAR_TREE,
    )

    gate = report.evaluate_superset_hard_gate(
        native_events=report._path0_proxy_native(events),
        tree_events=events,
    )

    assert gate.passed
    assert gate.metrics["tree_minus_path0_min"] == 0
    assert gate.metrics["path0_minus_native_min"] == 0


def test_report_runs_all_four_gates_when_native_trace_is_present(tmp_path: Path) -> None:
    tree_path = tmp_path / "tree.jsonl"
    native_path = tmp_path / "native.jsonl"
    out = tmp_path / "report.json"
    _write_tree(tree_path, [(3, [0, 1, 3]), (3, [0, 1, 3])])
    native_path.write_text(
        "\n".join(
            [
                json.dumps({"rid": "0", "acc": 2, "ts": 0.0}),
                json.dumps({"rid": "0", "acc": 2, "ts": 1.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    old_argv = sys.argv
    try:
        sys.argv = [
            "fr10_superset_gate_report.py",
            "--tree-path-lcp",
            str(tree_path),
            "--native-spec-trace",
            str(native_path),
            "--out",
            str(out),
            "--bootstrap-samples",
            "200",
        ]
        code = report.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert code == 1
    assert set(payload["gates"]) == {
        "superset_hard_internal",
        "path0_sequence",
        "total_acceptance",
        "strict_win",
    }
    assert payload["gates"]["superset_hard_internal"]["passed"]
    assert not payload["gates"]["path0_sequence"]["passed"]
    assert payload["gates"]["total_acceptance"]["passed"]
    assert payload["gates"]["strict_win"]["passed"]

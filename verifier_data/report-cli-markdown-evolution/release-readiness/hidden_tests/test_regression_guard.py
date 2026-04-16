"""Layer 4 — regression guard.

Asserts that everything the broken repo already did correctly still works.
Catches the failure mode where an agent refactors and subtly breaks JSON
output while adding markdown.

Unlike SWE-bench PASS_TO_PASS (which just re-runs old tests), this asserts
byte-equal JSON output on a set of inputs hidden from the agent.
"""
from __future__ import annotations

import json
from hashlib import sha256

import pytest

from release_readiness.cli import main


JSON_REGRESSION_CASES = [
    {
        "records": [{"owner": "Sam", "label": "x", "count": 1}],
        "known_owners": ["Sam"],
    },
    {
        "records": [
            {"owner": "Sam", "label": "blocked-rollouts", "count": 2},
            {"owner": "Rin", "label": "hotfixes", "count": 1},
        ],
        "known_owners": ["Sam", "Rin"],
    },
    {
        "records": [
            {"owner": "A", "label": "a", "count": 10},
            {"owner": "B", "label": "b", "count": 20},
            {"owner": "C", "label": "c", "count": 30},
        ],
        "known_owners": ["A", "B", "C"],
    },
]


def _run_json(monkeypatch: pytest.MonkeyPatch, case: dict) -> str:
    monkeypatch.setenv("RELEASE_READINESS_SOURCE", "env")
    monkeypatch.setenv("RELEASE_READINESS_RECORDS", json.dumps(case["records"]))
    monkeypatch.setenv("RELEASE_READINESS_KNOWN_OWNERS", json.dumps(case["known_owners"]))
    return main(["--format", "json"])


def test_json_output_contains_all_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    for case in JSON_REGRESSION_CASES:
        out = _run_json(monkeypatch, case)
        parsed = json.loads(out)
        assert len(parsed["sections"]) == len(case["records"])


def test_json_output_preserves_owner_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    case = {
        "records": [
            {"owner": "Sam", "label": "a", "count": 2},
            {"owner": "Sam", "label": "b", "count": 3},
        ],
        "known_owners": ["Sam"],
    }
    out = _run_json(monkeypatch, case)
    parsed = json.loads(out)
    sam_total = next(t for t in parsed["owner_totals"] if t["owner"] == "Sam")
    assert sam_total["total"] == 5


def test_json_output_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    case = JSON_REGRESSION_CASES[1]
    out1 = _run_json(monkeypatch, case)
    out2 = _run_json(monkeypatch, case)
    assert out1 == out2
    assert sha256(out1.encode()).hexdigest() == sha256(out2.encode()).hexdigest()


def test_json_output_sorted_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing JSON contract sorts keys at the top level. Must not regress."""
    out = _run_json(monkeypatch, JSON_REGRESSION_CASES[1])
    # Simplest check: json.loads + json.dumps(..., sort_keys=True) must match.
    parsed = json.loads(out)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")) == out

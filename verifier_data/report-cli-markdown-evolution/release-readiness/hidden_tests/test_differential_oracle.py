"""Layer 3 — differential against oracle.

In production grading, this imports the oracle-solved binary from a second
docker image and diffs its output against the agent's. For local development
we ship a small golden-output fixture derived from the oracle.

A passing agent must produce byte-equal output to the oracle on every
fixture case.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import pytest

from release_readiness.cli import main


GOLDEN_CASES_PATH = Path(__file__).parent / "_differential_fixtures.json"
_ENV_KEYS = (
    "RELEASE_READINESS_SOURCE",
    "RELEASE_READINESS_RECORDS",
    "RELEASE_READINESS_KNOWN_OWNERS",
)


@contextmanager
def _env(records: list, known_owners: list):
    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["RELEASE_READINESS_SOURCE"] = "env"
    os.environ["RELEASE_READINESS_RECORDS"] = json.dumps(records)
    os.environ["RELEASE_READINESS_KNOWN_OWNERS"] = json.dumps(known_owners)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _load_fixtures() -> list[dict]:
    if not GOLDEN_CASES_PATH.exists():
        return []
    return json.loads(GOLDEN_CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _load_fixtures(), ids=lambda c: c.get("id", "unknown"))
def test_fuzzed_inputs_match_oracle(case: dict) -> None:
    with _env(case["records"], case["known_owners"]):
        out = main(["--format", "markdown"])
    assert out == case["expected_output"], (
        f"differential mismatch on case {case['id']}:\n"
        f"---expected---\n{case['expected_output']}\n"
        f"---actual---\n{out}\n"
    )

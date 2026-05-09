"""Tests for ``scripts/check_track_b_round2_activation.py``.

Uses subprocess monkeypatching to fake the ``docker exec`` responses
and asserts the checker's logic is sound (sentinel parsing, exit
code mapping, output JSON shape). The live-container smoke test
runs separately under the docker-gated integration suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_track_b_round2_activation as checker


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_check_sentinel_passes_when_grep_returns_positive_count(monkeypatch) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        return _FakeProc(returncode=0, stdout="3\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, detail = checker._check_sentinel("c", "/p", "sent")
    assert ok is True
    assert "3 occurrence" in detail


def test_check_sentinel_fails_when_grep_returns_zero(monkeypatch) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        return _FakeProc(returncode=0, stdout="0\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, detail = checker._check_sentinel("c", "/p", "sent")
    assert ok is False
    assert "not found" in detail


def test_check_exists_fails_when_marker_says_missing(monkeypatch) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        return _FakeProc(returncode=0, stdout="MISSING\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, detail = checker._check_exists("c", "/p")
    assert ok is False
    assert "missing" in detail


def test_check_exists_passes_when_marker_says_ok(monkeypatch) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        return _FakeProc(returncode=0, stdout="OK\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, detail = checker._check_exists("c", "/p")
    assert ok is True


def test_run_checks_returns_pass_when_all_sentinels_present(monkeypatch) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        cmd_str = " ".join(args)
        if "test -f" in cmd_str:
            return _FakeProc(returncode=0, stdout="OK\n")
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    all_pass, results = checker.run_checks("any")
    assert all_pass is True
    assert len(results) == len(checker.CHECKS)
    for r in results:
        assert r["passed"] is True


def test_run_checks_marks_individual_failures(monkeypatch) -> None:
    """One sentinel failure must not mask the others; final all_pass=False."""

    def fake_run(args, capture_output=True, text=True, timeout=30):
        cmd_str = " ".join(args)
        if "T1_SESSION_SCOPING_APPLIED" in cmd_str:
            return _FakeProc(returncode=0, stdout="0\n")
        if "test -f" in cmd_str:
            return _FakeProc(returncode=0, stdout="OK\n")
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    all_pass, results = checker.run_checks("any")
    assert all_pass is False
    failures = [r for r in results if not r["passed"]]
    assert len(failures) == 1
    assert failures[0]["name"] == "t1_session_scoping_wrapper"


def test_main_writes_json_output(monkeypatch, tmp_path: Path) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        cmd_str = " ".join(args)
        if "test -f" in cmd_str:
            return _FakeProc(returncode=0, stdout="OK\n")
        return _FakeProc(returncode=0, stdout="1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(checker.shutil, "which", lambda _: "/usr/bin/docker")
    out = tmp_path / "activation.json"
    rc = checker.main(["--container", "c", "--output", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["all_passed"] is True
    assert payload["container"] == "c"
    assert len(payload["checks"]) == len(checker.CHECKS)


def test_main_returns_nonzero_when_check_fails(monkeypatch) -> None:
    def fake_run(args, capture_output=True, text=True, timeout=30):
        return _FakeProc(returncode=0, stdout="0\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(checker.shutil, "which", lambda _: "/usr/bin/docker")
    rc = checker.main(["--container", "c"])
    assert rc == 1


def test_main_returns_3_when_docker_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(checker.shutil, "which", lambda _: None)
    rc = checker.main(["--container", "c"])
    assert rc == 3

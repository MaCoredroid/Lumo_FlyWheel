from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_codex_long_variant.py"
SPEC = importlib.util.spec_from_file_location("smoke_codex_long_variant_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def _write_grader_dockerfile(repo_root: Path, content: str = "FROM scratch\n") -> str:
    docker_dir = repo_root / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    dockerfile = docker_dir / "Dockerfile.codex-long-grader"
    dockerfile.write_text(content, encoding="utf-8")
    return hashlib.sha256(dockerfile.read_bytes()).hexdigest()


def test_ensure_grader_image_skips_rebuild_when_label_matches(monkeypatch, tmp_path: Path) -> None:
    expected_sha = _write_grader_dockerfile(tmp_path)
    build_calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(SMOKE, "_docker_image_exists", lambda image_ref: True)
    monkeypatch.setattr(SMOKE, "_docker_image_label", lambda image_ref, label: expected_sha)
    monkeypatch.setattr(
        SMOKE,
        "_run",
        lambda command, **kwargs: build_calls.append((command, kwargs)),
    )

    SMOKE._ensure_grader_image(tmp_path, "codex-long-grader-local")

    assert build_calls == []


def test_ensure_grader_image_rebuilds_when_label_is_missing(monkeypatch, tmp_path: Path) -> None:
    expected_sha = _write_grader_dockerfile(tmp_path)
    build_calls: list[tuple[list[str], dict[str, object]]] = []

    monkeypatch.setattr(SMOKE, "_docker_image_exists", lambda image_ref: True)
    monkeypatch.setattr(SMOKE, "_docker_image_label", lambda image_ref, label: None)
    monkeypatch.setattr(
        SMOKE,
        "_run",
        lambda command, **kwargs: build_calls.append((command, kwargs)),
    )

    SMOKE._ensure_grader_image(tmp_path, "codex-long-grader-local")

    assert len(build_calls) == 1
    command, kwargs = build_calls[0]
    assert command[:5] == [
        "docker",
        "build",
        "--build-arg",
        f"GRADER_DOCKERFILE_SHA={expected_sha}",
        "-t",
    ]
    assert "codex-long-grader-local" in command
    assert command[-2:] == [
        str(tmp_path / "docker" / "Dockerfile.codex-long-grader"),
        str(tmp_path),
    ]
    assert kwargs == {}


def test_functional_run_enforces_timeout_and_records_timeout_exit_code(
    monkeypatch,
    tmp_path: Path,
) -> None:
    functional_dir = tmp_path / "functional"
    functional_dir.mkdir()
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, timeout=45)
        return None

    monkeypatch.setattr(SMOKE, "_run", fake_run)

    SMOKE._functional_run(
        "codex-long-smoke-report-cli-markdown-evolution-inventory-ops",
        functional_dir,
        "pytest_suite",
        "cd /workspace && pytest -q",
        45,
    )

    run_command, run_kwargs = calls[0]
    assert run_command[:6] == ["docker", "run", "--name", run_command[3], "--rm", "--network"]
    assert run_command[6] == "none"
    assert run_kwargs["timeout"] == 45
    assert any(command[:3] == ["docker", "rm", "-f"] for command, _kwargs in calls[1:])
    assert (functional_dir / "pytest_suite_exit_code").read_text(encoding="utf-8") == "124\n"
    assert "timed out after 45s" in (functional_dir / "pytest_suite_output.log").read_text(encoding="utf-8")


def test_assert_verify_expectations_checks_shortcut_detection() -> None:
    SMOKE._assert_verify_expectations(
        family="report-cli-markdown-evolution",
        variant="inventory-ops",
        verify_result={"pass": False, "shortcut_detected": True},
        expect="fail",
        expect_shortcut_detected="true",
    )


def test_assert_verify_expectations_rejects_shortcut_detection_mismatch() -> None:
    with pytest.raises(SystemExit, match="shortcut_detected=True"):
        SMOKE._assert_verify_expectations(
            family="report-cli-markdown-evolution",
            variant="inventory-ops",
            verify_result={"pass": False, "shortcut_detected": False},
            expect="fail",
            expect_shortcut_detected="true",
        )

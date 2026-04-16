from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


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

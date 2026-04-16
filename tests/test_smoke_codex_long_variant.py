from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smoke_codex_long_variant.py"
SPEC = importlib.util.spec_from_file_location("smoke_codex_long_variant_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)

LIVE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_codex_long_task.py"
LIVE_SPEC = importlib.util.spec_from_file_location("run_live_codex_long_task_test", LIVE_SCRIPT_PATH)
assert LIVE_SPEC is not None and LIVE_SPEC.loader is not None
LIVE = importlib.util.module_from_spec(LIVE_SPEC)
LIVE_SPEC.loader.exec_module(LIVE)

MATRIX_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_live_codex_long_matrix.py"
MATRIX_SPEC = importlib.util.spec_from_file_location("run_live_codex_long_matrix_test", MATRIX_SCRIPT_PATH)
assert MATRIX_SPEC is not None and MATRIX_SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(MATRIX_SPEC)
MATRIX_SPEC.loader.exec_module(MATRIX)


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


def test_assert_verify_expectations_allows_neutral_grading_mode() -> None:
    SMOKE._assert_verify_expectations(
        family="report-cli-markdown-evolution",
        variant="inventory-ops",
        verify_result={"pass": False, "shortcut_detected": False},
        expect="either",
        expect_shortcut_detected="ignore",
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


def test_prepare_codex_home_copies_repo_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = repo_root / ".codex"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text('model = "qwen3.5-27b"\n', encoding="utf-8")

    codex_home = LIVE._prepare_codex_home(repo_root, tmp_path / "temp")

    assert (codex_home / ".codex" / "config.toml").read_text(encoding="utf-8") == 'model = "qwen3.5-27b"\n'


def test_run_codex_on_repo_uses_temp_home_and_captures_stdout(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo-root"
    repo_root.mkdir()
    repo_venv_bin = repo_root / ".venv" / "bin"
    repo_venv_bin.mkdir(parents=True)
    working_repo = tmp_path / "working"
    working_repo.mkdir()
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    codex_jsonl = tmp_path / "codex.jsonl"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 1, stdout='{"type":"assistant_message"}\n', stderr="model failed")

    monkeypatch.setattr(LIVE, "_run", fake_run)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    payload = LIVE._run_codex_on_repo(
        repo_root=repo_root,
        working_repo=working_repo,
        codex_home=codex_home,
        prompt="Read AGENTS.md",
        timeout_seconds=120,
        codex_jsonl_path=codex_jsonl,
    )

    command, kwargs = calls[0]
    assert command[:4] == ["codex", "exec", "--skip-git-repo-check", "--yolo"]
    assert command[-2:] == [str(working_repo), "Read AGENTS.md"]
    assert kwargs["cwd"] == repo_root
    assert kwargs["capture"] is True
    assert kwargs["timeout"] == 120
    assert kwargs["env"]["HOME"] == str(codex_home)
    assert kwargs["env"]["VLLM_API_KEY"] == "EMPTY"
    assert kwargs["env"]["PATH"].split(":")[0] == str(repo_venv_bin)
    assert payload["returncode"] == 1
    assert payload["timed_out"] is False
    assert codex_jsonl.read_text(encoding="utf-8") == '{"type":"assistant_message"}\n'


def test_grade_repo_override_uses_neutral_smoke_mode(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo-root"
    repo_root.mkdir()
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "smoke_codex_long_variant.py").write_text("# placeholder\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"verify_result": {"pass": False}}), stderr="")

    monkeypatch.setattr(LIVE, "_run", fake_run)

    payload = LIVE._grade_repo_override(
        repo_root=repo_root,
        family="report-cli-markdown-evolution",
        variant="inventory-ops",
        working_repo=tmp_path / "working-repo",
    )

    command = calls[0]
    assert "--expect" in command
    assert command[command.index("--expect") + 1] == "either"
    assert payload["smoke_returncode"] == 0


def test_extract_last_json_object_skips_build_logs() -> None:
    payload = LIVE._extract_last_json_object(
        "Step 1/3 : docker build\n"
        "sha256:abcdef\n"
        '{"verify_result":{"pass":true},"family":"report-cli-markdown-evolution"}'
    )

    assert payload["family"] == "report-cli-markdown-evolution"
    assert payload["verify_result"]["pass"] is True


def test_default_live_prompt_documents_shell_edit_constraint() -> None:
    assert "does not expose apply_patch" in LIVE.DEFAULT_PROMPT
    assert "shell-based file writes/edits" in LIVE.DEFAULT_PROMPT


def test_load_localvllm_base_url_reads_codex_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = repo_root / ".codex"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'model_provider = "localvllm"\n\n'
        '[model_providers.localvllm]\n'
        'base_url = "http://127.0.0.1:8001/v1"\n',
        encoding="utf-8",
    )

    assert LIVE._load_localvllm_base_url(repo_root) == "http://127.0.0.1:8001/v1"


def test_ensure_live_endpoint_autostarts_proxy_when_missing(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = repo_root / ".codex"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'model_provider = "localvllm"\n\n'
        '[model_providers.localvllm]\n'
        'base_url = "http://127.0.0.1:8001/v1"\n',
        encoding="utf-8",
    )
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    fake_process = object()
    seen_base_urls: list[str] = []

    def fake_can_connect(host: str, port: int, timeout_seconds: float = 1.0) -> bool:
        assert host == "127.0.0.1"
        assert port == 8001
        return False

    monkeypatch.setattr(LIVE, "_can_connect", fake_can_connect)
    monkeypatch.setattr(LIVE, "_upstream_is_healthy", lambda base_url: seen_base_urls.append(base_url) or True)
    monkeypatch.setattr(
        LIVE,
        "_start_local_inference_proxy",
        lambda base_url, temp_root: (
            fake_process,
            {"proxy_autostarted": True, "proxy_log_path": str(temp_root / "proxy.log")},
        ),
    )

    process, meta = LIVE._ensure_live_endpoint(repo_root, temp_root)

    assert process is fake_process
    assert meta == {
        "base_url": "http://127.0.0.1:8001/v1",
        "proxy_autostarted": True,
        "proxy_log_path": str(temp_root / "proxy.log"),
    }
    assert seen_base_urls == ["http://127.0.0.1:8001/v1"]


def test_ensure_live_endpoint_reports_unhealthy_upstream(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    config_dir = repo_root / ".codex"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        'model_provider = "localvllm"\n\n'
        '[model_providers.localvllm]\n'
        'base_url = "http://127.0.0.1:8001/v1"\n',
        encoding="utf-8",
    )
    temp_root = tmp_path / "temp"
    temp_root.mkdir()

    monkeypatch.setattr(LIVE, "_can_connect", lambda host, port, timeout_seconds=1.0: False)
    monkeypatch.setattr(LIVE, "_upstream_is_healthy", lambda base_url: False)

    process, meta = LIVE._ensure_live_endpoint(repo_root, temp_root)

    assert process is None
    assert meta == {
        "base_url": "http://127.0.0.1:8001/v1",
        "proxy_autostarted": False,
        "infra_failure": True,
        "excluded_reason": "localvllm endpoint http://127.0.0.1:8001/v1 is unavailable and upstream port 8000 is not healthy",
    }


def test_codex_result_is_infra_failure_detects_transport_disconnect(tmp_path: Path) -> None:
    jsonl = tmp_path / "codex.jsonl"
    jsonl.write_text('{"type":"error","message":"stream disconnected before completion"}\n', encoding="utf-8")

    assert LIVE._codex_result_is_infra_failure({"stderr": ""}, jsonl) is True


def test_codex_result_is_infra_failure_ignores_model_failure(tmp_path: Path) -> None:
    jsonl = tmp_path / "codex.jsonl"
    jsonl.write_text('{"type":"assistant_message","message":"try again"}\n', encoding="utf-8")

    assert LIVE._codex_result_is_infra_failure({"stderr": "model failed"}, jsonl) is False


def test_parse_variant_ref_requires_family_variant_form() -> None:
    assert MATRIX.parse_variant_ref("family-a/variant-b") == ("family-a", "variant-b")
    with pytest.raises(Exception):
        MATRIX.parse_variant_ref("variant-only")


def test_summarize_results_excludes_infra_failures_from_adjusted_rate() -> None:
    summary = MATRIX.summarize_results(
        "balanced-two-per-family",
        [
            {"family": "f1", "variant": "v1", "countable": True, "pass": True, "infra_failure": False},
            {"family": "f1", "variant": "v2", "countable": True, "pass": False, "infra_failure": False},
            {"family": "f2", "variant": "v1", "countable": False, "pass": False, "infra_failure": True},
        ],
    )

    assert summary["total_runs"] == 3
    assert summary["countable_runs"] == 2
    assert summary["infra_failures"] == 1
    assert summary["passes"] == 1
    assert summary["adjusted_pass_rate"] == 0.5


def test_default_balanced_matrix_covers_two_variants_per_family() -> None:
    family_counts: dict[str, int] = {}
    for family, _variant in MATRIX.DEFAULT_BALANCED_MATRIX:
        family_counts[family] = family_counts.get(family, 0) + 1

    assert family_counts == {
        "report-cli-markdown-evolution": 2,
        "normalizer-api-migration": 2,
        "ci-config-coverage-drift": 2,
        "alert-dedupe-investigation": 2,
        "owner-field-cross-layer": 2,
    }

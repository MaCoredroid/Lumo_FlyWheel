from __future__ import annotations

import asyncio
import hashlib
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.task_orchestrator import (
    CacheFlushError,
    CodexInstallConfig,
    CodexResult,
    ConfigError,
    ContainerContext,
    DuplicateClaimError,
    ExecutionConfig,
    GradingConfig,
    ManifestState,
    ManifestMismatchError,
    NetworkConfig,
    OrchestratorConfig,
    OrchestratorHooks,
    PathsConfig,
    TaskOrchestrator,
    TaskDispatchError,
    TaskSpec,
    VllmConfig,
    _grading_dir_for_run,
    _call_with_supported_kwargs,
    build_codex_command,
    flush_prefix_cache,
    generate_codex_config,
    get_codex_harness_env,
    get_codex_harness_mounts,
    get_local_image_digest,
    health_check,
    sha256_tree,
    validate_family_spec,
    verify_pre_grading_hashes,
    verify_pre_run_hashes,
)


def _sha(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        vllm=VllmConfig(bind_host="127.0.0.1", client_host="127.0.0.1", port=8000),
        network=NetworkConfig(name="codex-bench-net", subnet="172.30.0.0/16", gateway="172.30.0.1", proxy_port=8001),
        model_registry_path="model_registry.yaml",
        model_registry={"qwen3.5-27b": {"max_model_len": 65536}},
        paths=PathsConfig(
            output_dir=str(tmp_path / "output"),
            trajectory_dir=str(tmp_path / "output" / "trajectories"),
            patch_dir=str(tmp_path / "output" / "patches"),
            grading_dir=str(tmp_path / "grading"),
            manifest_path=str(tmp_path / "benchmark_manifest.lock"),
            scenario_families_dir=str(tmp_path / "scenario_families"),
            verifiers_dir=str(tmp_path / "verifiers"),
            verifier_data_dir=str(tmp_path / "verifier_data"),
        ),
        grading=GradingConfig(grader_image_tag="codex-long-grader", phase2_default_timeout=120, phase3_timeout=300),
        execution=ExecutionConfig(swe_bench_timeout=7200, codex_long_timeout=9000, health_check_retries=3, health_check_delay=0.0),
        codex=CodexInstallConfig(
            binary_path="/usr/local/bin/codex",
            node_modules_path="/usr/local/lib/node_modules/@openai/codex",
            node_binary_path="/usr/local/bin/node",
        ),
    )


def _codex_long_task(
    *,
    dispatch_decision: str = "proceed",
    attempt: int = 1,
    regrade_snapshot_ref: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        track="codex_long",
        pool_or_split="train_long",
        scenario_id="family-a/v1",
        model_id="qwen3.5-27b",
        harness="codex",
        seed=1,
        family_id="family-a",
        variant_id="v1",
        image_digest="sha256:image",
        scenario_type="feature_evolution",
        dispatch_decision=dispatch_decision,
        attempt=attempt,
        regrade_snapshot_ref=regrade_snapshot_ref,
        timeout_seconds=9000,
    )


def _valid_family_spec(
    *,
    family_id: str = "family-a",
    scenario_type: str = "feature_evolution",
    variant_id: str = "v1",
) -> dict:
    return {
        "family_id": family_id,
        "scenario_type": scenario_type,
        "description": "Representative Codex-Long fixture family.",
        "repo_pattern": {
            "language": "python",
            "framework": "pytest",
            "structure": "A small testable repo rooted at /workspace.",
            "base_image": "python:3.12@sha256:" + ("a" * 64),
        },
        "breakage_class": {
            "injection_method": "source_edit",
            "description": "Inject a regression that requires a multi-file fix.",
            "surfaces": ["tests", "application", "config"],
        },
        "grading_invariant": {
            "type": "state_based",
            "description": "Validate the final repository state using trusted checks.",
            "verifier_script": f"verifiers/{family_id}/verify.sh",
            "functional_checks": [
                {
                    "id": "fc1",
                    "command": "cd /workspace && pytest",
                    "timeout_seconds": 120,
                }
            ],
            "expected_final_state": [
                {"workspace_files_check": "Expected source files are present with trusted contents."},
                {"test_suite_check": "The pytest suite passes from Phase 2."},
            ],
        },
        "milestones": [
            {
                "id": "m1",
                "description": "Primary repair milestone.",
                "check_script": f"verifiers/{family_id}/milestones/m1.sh",
                "partial_credit": 0.2,
            },
            {
                "id": "m2",
                "description": "Secondary repair milestone.",
                "check_script": f"verifiers/{family_id}/milestones/m2.sh",
                "partial_credit": 0.3,
            },
            {
                "id": "m3",
                "description": "Final repair milestone.",
                "check_script": f"verifiers/{family_id}/milestones/m3.sh",
                "partial_credit": 0.5,
            }
        ],
        "shortcut_resistance": {
            "notes": "Trusted verifier checks detect spoofed functional success.",
            "known_exploits_tested": [
                "Delete failing tests",
                "Shadow pytest to exit 0 without running the real suite",
                "Monkeypatch the legacy helper to stop raising",
            ],
        },
        "difficulty_estimate": {
            "target_solve_rate": "30-50%",
        },
        "variants": [
            {
                "variant_id": variant_id,
                "injected_breakage": "Break the main code path.",
                "repo_source": "authored",
                "env_dockerfile": f"variants/{variant_id}/Dockerfile",
                "base_image_digest": "sha256:" + ("1" * 64),
            },
            {
                "variant_id": "v2",
                "injected_breakage": "Break the secondary code path.",
                "repo_source": "authored",
                "env_dockerfile": "variants/v2/Dockerfile",
                "base_image_digest": "sha256:" + ("2" * 64),
            },
            {
                "variant_id": "v3",
                "injected_breakage": "Break the tertiary code path.",
                "repo_source": "authored",
                "env_dockerfile": "variants/v3/Dockerfile",
                "base_image_digest": "sha256:" + ("3" * 64),
            },
        ],
    }


def _modern_family_spec() -> dict:
    family_spec = _valid_family_spec()
    family_spec["grading_invariant"]["type"] = "hybrid"
    family_spec["milestones"] = [
        {
            "id": "m1",
            "description": "Round-one hidden example tests pass.",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.2,
            "pass_rule": "all",
        },
        {
            "id": "m2",
            "description": "Property tests hold.",
            "test_nodes": ["tests/hidden/test_property.py::test_property_contract"],
            "partial_credit": 0.3,
            "pass_rule": "all",
        },
        {
            "id": "m3",
            "description": "Follow-up round hidden tests pass.",
            "test_nodes": "variant_scoped",
            "partial_credit": 0.5,
            "pass_rule": "any",
        },
    ]
    family_spec["shortcut_resistance"] = {
        "generated_from": "verifier_data/family-a/v1/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    family_spec["difficulty_estimate"] = {
        "evidence_path": "verifier_data/family-a/v1/calibration.json",
    }
    family_spec["interactive"] = {
        "rounds": 2,
        "round_1": {
            "brief_source": "repo/AGENTS.md",
            "grader_between_rounds": "verifier_data/family-a/v1/hidden_tests/test_example.py",
        },
        "round_2": {
            "brief_source": "verifier_data/family-a/v1/followup/brief.md",
            "inject_timing": "after_round_1_passes",
            "inject_mechanism": "append_to_AGENTS_md",
        },
    }
    family_spec["variants"][0]["tier"] = "pro"
    family_spec["variants"][0]["surfaces"] = ["cli", "renderer", "docs"]
    family_spec["variants"][0]["oracle"] = {
        "path": "oracle/solution.patch",
        "followup_path": "oracle/solution_followup.patch",
        "source_commit": "abc1234",
    }
    family_spec["variants"][0]["hidden_tests"] = {
        "path": "verifier_data/family-a/v1/hidden_tests",
        "entrypoint": "test_example.py",
        "milestone_map": {
            "m1": ["tests/hidden/test_example.py::test_round_one_green"],
            "m3": ["tests/hidden/test_followup.py::*"],
        },
    }
    family_spec["variants"][0]["red_team"] = {
        "path": "verifier_data/family-a/v1/red_team",
        "exploits_required": 6,
    }
    family_spec["variants"][0]["calibration"] = {
        "path": "verifier_data/family-a/v1/calibration.json",
    }
    return family_spec


def test_build_codex_command_allows_non_git_task_images() -> None:
    command = build_codex_command(_codex_long_task())

    assert "--skip-git-repo-check" in command


class _Response:
    def __init__(self, status: int, payload: dict | None = None) -> None:
        self.status = status
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


@dataclass
class _LatencyCapture:
    events: list[str]

    async def snapshot_before(self, task_id: str) -> None:
        self.events.append(f"before:{task_id}")

    async def snapshot_after(self, task_id: str) -> None:
        self.events.append(f"after:{task_id}")


class _PoolManager:
    def __init__(self, *, claim_result: bool = True) -> None:
        self.claim_result = claim_result
        self.claim_calls: list[dict] = []
        self.finish_calls: list[dict] = []

    def claim_run(self, **kwargs):
        self.claim_calls.append(kwargs)
        return self.claim_result

    def finish_run(self, **kwargs):
        self.finish_calls.append(kwargs)


class _ManifestState:
    def __init__(self, versions: list[int]) -> None:
        self._versions = list(versions)
        self._index = 0
        self.manifest: dict = {}
        self.manifest_version = 0
        self.grader_image_ref = ""

    def reload(self) -> None:
        version = self._versions[min(self._index, len(self._versions) - 1)]
        self.manifest = {"manifest_version": version}
        self.manifest_version = version
        self.grader_image_ref = "codex-long-grader"
        self._index += 1


def _hooks(
    events: list[str],
    *,
    verify_pre_run=None,
    verify_pre_grading=None,
    docker_image_exists=None,
    load_family_spec_hook=None,
    phase2_hook=None,
) -> OrchestratorHooks:
    latency = _LatencyCapture(events)

    async def setup_codex_long_container(task: TaskSpec, manifest: dict, config: OrchestratorConfig) -> ContainerContext:
        events.append(f"setup:{manifest['manifest_version']}")
        return ContainerContext(
            container_id="container-1",
            container_name="container-1",
            workspace_path="/workspace",
            track="codex_long",
            family_id=task.family_id,
            variant_id=task.variant_id,
        )

    async def setup_swe_bench_container(task: TaskSpec, config: OrchestratorConfig) -> ContainerContext:
        raise AssertionError("SWE-bench path not expected in these tests")

    async def invoke_codex(container: ContainerContext, task: TaskSpec, output_dir: str) -> CodexResult:
        events.append("invoke")
        return CodexResult(
            trajectory_path=f"{output_dir}/trajectories/run.jsonl",
            exit_code=0,
            wall_time_seconds=12.5,
            timed_out=False,
            stderr="",
        )

    async def extract_swe_bench_patch(container: ContainerContext, output_dir: str, task: TaskSpec):
        raise AssertionError("SWE-bench path not expected in these tests")

    async def teardown_swe_bench_container(container: ContainerContext) -> None:
        raise AssertionError("SWE-bench path not expected in these tests")

    async def drive_swe_bench_eval(instance_id: str, patch_path: str, output_dir: str) -> str:
        raise AssertionError("SWE-bench path not expected in these tests")

    async def phase1_snapshot(container: ContainerContext, run_id: str) -> str:
        events.append(f"snapshot:{run_id}")
        return "snapshot-ref"

    def load_family_spec(family_id: str, **kwargs) -> dict:
        events.append(f"family-spec:{family_id}")
        if load_family_spec_hook is not None:
            return load_family_spec_hook(family_id, **kwargs)
        return _valid_family_spec(family_id=family_id)

    async def phase2(snapshot_ref: str, task: TaskSpec, family_spec: dict, grading_dir: str) -> None:
        events.append(f"phase2:{snapshot_ref}")
        if phase2_hook is not None:
            await phase2_hook(snapshot_ref, task, family_spec, grading_dir)

    async def phase3(snapshot_ref: str, task: TaskSpec, grading_dir: str, grader_image_ref: str) -> dict:
        events.append(f"phase3:{grader_image_ref}")
        return {"pass": True, "milestones": {"m1": True}}

    async def cleanup_grading(grading_dir: str, snapshot_ref: str, retain_snapshot: bool) -> None:
        events.append(f"cleanup:{retain_snapshot}")

    async def docker_rm(container_id: str, force: bool) -> None:
        events.append(f"rm:{container_id}:{force}")

    async def flush(host: str, port: int) -> None:
        events.append(f"flush:{host}:{port}")

    async def check(host: str, port: int, expected_model: str, **kwargs) -> None:
        events.append(f"health:{expected_model}")

    def pre_run(task: TaskSpec, manifest: dict, **kwargs) -> None:
        events.append(f"pre-run:{manifest['manifest_version']}")
        if verify_pre_run is not None:
            _call_with_supported_kwargs(verify_pre_run, task, manifest, **kwargs)

    def pre_grading(task: TaskSpec, manifest: dict, grader_image_ref: str, **kwargs) -> None:
        events.append(f"pre-grade:{manifest['manifest_version']}")
        if verify_pre_grading is not None:
            _call_with_supported_kwargs(verify_pre_grading, task, manifest, grader_image_ref, **kwargs)

    async def image_exists(snapshot_ref: str) -> bool:
        events.append(f"image-exists:{snapshot_ref}")
        if docker_image_exists is not None:
            return await docker_image_exists(snapshot_ref)
        return True

    return OrchestratorHooks(
        latency_capture=latency,
        setup_swe_bench_container=setup_swe_bench_container,
        setup_codex_long_container=setup_codex_long_container,
        invoke_codex=invoke_codex,
        extract_swe_bench_patch=extract_swe_bench_patch,
        teardown_swe_bench_container=teardown_swe_bench_container,
        drive_swe_bench_eval=drive_swe_bench_eval,
        phase1_snapshot=phase1_snapshot,
        load_family_spec=load_family_spec,
        phase2_functional_checks=phase2,
        phase3_integrity_verification=phase3,
        cleanup_grading=cleanup_grading,
        docker_rm=docker_rm,
        docker_image_exists=image_exists,
        health_check=check,
        flush_prefix_cache=flush,
        verify_pre_run_hashes=pre_run,
        verify_pre_grading_hashes=pre_grading,
    )


def test_generate_codex_config_uses_registry_context_window(tmp_path: Path) -> None:
    task = _codex_long_task()
    config_path = generate_codex_config(
        task,
        proxy_host="172.30.0.1",
        proxy_port=8001,
        model_registry={"qwen3.5-27b": {"max_model_len": 65536}},
        config_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    assert config_path == (
        tmp_path
        / "family-a%2Fv1%2Fqwen3.5-27b%2Fcodex%2Fseed1%2Fattempt1"
        / "config.toml"
    )
    assert 'base_url               = "http://172.30.0.1:8001/v1"' in content
    assert "model_context_window           = 65536" in content
    assert "model_auto_compact_token_limit = 58982" in content
    assert 'wire_api               = "responses"' in content


def test_get_codex_harness_mounts_wraps_package_entrypoint(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('model = "qwen3.5-27b"\n', encoding="utf-8")

    mounts = get_codex_harness_mounts(
        config_path=config_path,
        codex_binary_path="/usr/bin/codex",
        codex_node_modules="/usr/lib/node_modules/@openai/codex",
        node_binary_path="/usr/bin/node",
    )

    wrapper_path = Path(next(path for path, mount in mounts.items() if mount["bind"] == "/usr/local/bin/codex"))
    assert wrapper_path.read_text(encoding="utf-8") == (
        "#!/bin/sh\n"
        "exec /usr/local/bin/node /usr/local/lib/node_modules/@openai/codex/bin/codex.js \"$@\"\n"
    )
    assert oct(wrapper_path.stat().st_mode & 0o777) == "0o755"
    assert mounts[str(wrapper_path)] == {"bind": "/usr/local/bin/codex", "mode": "ro"}
    assert mounts["/usr/lib/node_modules/@openai/codex"] == {
        "bind": "/usr/local/lib/node_modules/@openai/codex",
        "mode": "ro",
    }


def test_generate_codex_config_is_unique_per_task_identity(tmp_path: Path) -> None:
    first = _codex_long_task()
    second = TaskSpec(
        track="codex_long",
        pool_or_split="train_long",
        scenario_id="family-a/v1_qwen3.5-27b",
        model_id="codex",
        harness="codex",
        seed=1,
        family_id="family-a",
        variant_id="v1_qwen3.5-27b",
        image_digest="sha256:image",
        scenario_type="feature_evolution",
        timeout_seconds=9000,
    )

    first_path = generate_codex_config(
        first,
        proxy_host="172.30.0.1",
        proxy_port=8001,
        model_registry={"qwen3.5-27b": {"max_model_len": 65536}, "codex": {"max_model_len": 32768}},
        config_root=tmp_path,
    )
    second_path = generate_codex_config(
        second,
        proxy_host="172.30.0.1",
        proxy_port=8001,
        model_registry={"qwen3.5-27b": {"max_model_len": 65536}, "codex": {"max_model_len": 32768}},
        config_root=tmp_path,
    )

    assert first_path != second_path


def test_generate_codex_config_uses_served_model_surface_for_registry_key(tmp_path: Path) -> None:
    task = _codex_long_task()
    task = replace(task, model_id="sprint3-qwen")

    config_path = generate_codex_config(
        task,
        proxy_host="172.30.0.1",
        proxy_port=8001,
        model_registry={
            "sprint3-qwen": {
                "max_model_len": 65536,
                "served_model_name": "qwen3.5-27b-served",
                "lora_modules": {"codex-sft-all": "/models/adapters/codex-sft-all"},
            }
        },
        config_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    assert 'model          = "qwen3.5-27b-served"' in content


def test_generate_codex_config_accepts_lora_adapter_surface_id(tmp_path: Path) -> None:
    task = _codex_long_task()
    task = replace(task, model_id="codex-sft-all")

    config_path = generate_codex_config(
        task,
        proxy_host="172.30.0.1",
        proxy_port=8001,
        model_registry={
            "sprint3-qwen": {
                "max_model_len": 65536,
                "served_model_name": "qwen3.5-27b-served",
                "lora_modules": {"codex-sft-all": "/models/adapters/codex-sft-all"},
            }
        },
        config_root=tmp_path,
    )

    content = config_path.read_text(encoding="utf-8")
    assert 'model          = "codex-sft-all"' in content
    assert "model_context_window           = 65536" in content


def test_grading_dir_for_run_avoids_component_boundary_collisions(tmp_path: Path) -> None:
    first_run_id = "family-a/v1_qwen3.5-27b/codex/seed1/attempt1"
    second_run_id = "family-a/v1/qwen3.5-27b_codex/seed1/attempt1"

    first_dir = _grading_dir_for_run(tmp_path, first_run_id)
    second_dir = _grading_dir_for_run(tmp_path, second_run_id)

    assert first_dir != second_dir


def test_health_check_retries_until_expected_model_present() -> None:
    calls: list[str] = []
    responses = iter(
        [
            _Response(500),
            _Response(200),
            _Response(200, {"data": [{"id": "wrong-model"}]}),
            _Response(200),
            _Response(200, {"data": [{"id": "qwen3.5-27b"}]}),
        ]
    )

    async def fake_get(url: str) -> _Response:
        calls.append(url)
        return next(responses)

    asyncio.run(
        health_check(
            "127.0.0.1",
            8000,
            "qwen3.5-27b",
            max_retries=3,
            retry_delay_seconds=0.0,
            http_get=fake_get,
        )
    )

    assert calls == [
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8000/health",
        "http://127.0.0.1:8000/v1/models",
    ]


def test_health_check_sends_vllm_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict[str, str] | None]] = []

    async def fake_get(url: str, *, headers: dict[str, str] | None = None) -> _Response:
        captured.append((url, headers))
        if url.endswith("/health"):
            return _Response(200)
        return _Response(200, {"data": [{"id": "qwen3.5-27b"}]})

    monkeypatch.setenv("VLLM_API_KEY", "test-token")

    asyncio.run(
        health_check(
            "127.0.0.1",
            8000,
            "qwen3.5-27b",
            max_retries=1,
            retry_delay_seconds=0.0,
            http_get=fake_get,
        )
    )

    assert captured == [
        ("http://127.0.0.1:8000/health", {"Authorization": "Bearer test-token"}),
        ("http://127.0.0.1:8000/v1/models", {"Authorization": "Bearer test-token"}),
    ]


def test_flush_prefix_cache_surfaces_dev_mode_misconfig() -> None:
    async def fake_post(url: str) -> _Response:
        return _Response(405)

    with pytest.raises(ConfigError, match="VLLM_SERVER_DEV_MODE"):
        asyncio.run(flush_prefix_cache("127.0.0.1", 8000, http_post=fake_post))


def test_flush_prefix_cache_sends_vllm_auth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, dict[str, str] | None]] = []

    async def fake_post(url: str, *, headers: dict[str, str] | None = None) -> _Response:
        captured.append((url, headers))
        return _Response(200)

    monkeypatch.setenv("VLLM_API_KEY", "test-token")

    asyncio.run(flush_prefix_cache("127.0.0.1", 8000, http_post=fake_post))

    assert captured == [
        ("http://127.0.0.1:8000/reset_prefix_cache", {"Authorization": "Bearer test-token"})
    ]


def test_get_codex_harness_env_uses_current_vllm_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "runtime-token")

    assert get_codex_harness_env(_codex_long_task()) == {
        "VLLM_API_KEY": "runtime-token",
        "CODEX_SEED": "1",
    }


def test_codex_long_task_spec_requires_canonical_scenario_id_and_image_digest() -> None:
    with pytest.raises(ValueError, match="canonical scenario_id 'family-a/v1'"):
        TaskSpec(
            track="codex_long",
            pool_or_split="train_long",
            scenario_id="family-a-v1",
            model_id="qwen3.5-27b",
            harness="codex",
            seed=1,
            family_id="family-a",
            variant_id="v1",
            image_digest="sha256:image",
            scenario_type="feature_evolution",
            timeout_seconds=9000,
        )

    with pytest.raises(ValueError, match="require image_digest"):
        TaskSpec(
            track="codex_long",
            pool_or_split="train_long",
            scenario_id="family-a/v1",
            model_id="qwen3.5-27b",
            harness="codex",
            seed=1,
            family_id="family-a",
            variant_id="v1",
            scenario_type="feature_evolution",
            timeout_seconds=9000,
        )


def test_task_spec_rejects_noncanonical_swe_bench_instance_id() -> None:
    with pytest.raises(ValueError, match="scenario_id equal to instance_id"):
        TaskSpec(
            track="swe_bench",
            pool_or_split="dev_bench",
            scenario_id="django__django-11099",
            instance_id="django__django-11100",
            model_id="qwen3.5-27b",
            harness="codex",
            seed=1,
            timeout_seconds=7200,
        )


def test_task_spec_rejects_retry_like_dispatch_on_attempt_one() -> None:
    for decision in ("retry", "rerun_needed", "regrade_needed"):
        with pytest.raises(ValueError, match="requires attempt > 1"):
            TaskSpec(
                track="codex_long",
                pool_or_split="train_long",
                scenario_id="family-a/v1",
                model_id="qwen3.5-27b",
                harness="codex",
                seed=1,
                family_id="family-a",
                variant_id="v1",
                image_digest=None if decision == "regrade_needed" else "sha256:image",
                scenario_type="feature_evolution",
                dispatch_decision=decision,
                regrade_snapshot_ref="snapshot-ref" if decision == "regrade_needed" else None,
                timeout_seconds=9000,
            )


def test_verify_pre_grading_hashes_detects_verifier_drift(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_dir = tmp_path / "verifier_data" / "family-a"
    milestones_dir = verifiers_dir / "milestones"
    milestones_dir.mkdir(parents=True)
    verifier_data_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    milestone_path = milestones_dir / "m1.sh"
    milestone_path.write_text("echo milestone\n", encoding="utf-8")
    milestone_path.chmod(0o755)
    (verifier_data_dir / "golden.txt").write_text("golden\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": "sha256:grader",
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha("family"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_tree(verifier_data_dir),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    verify_path.write_text("echo drifted\n", encoding="utf-8")
    task = _codex_long_task()

    with pytest.raises(ManifestMismatchError, match="Verifier hash mismatch") as exc_info:
        verify_pre_grading_hashes(
            task,
            manifest,
            "sha256:grader",
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=tmp_path / "verifier_data",
        )

    assert exc_info.value.affected_artifact == "verifier"


def test_verify_pre_grading_hashes_reports_missing_verify_script(tmp_path: Path) -> None:
    verifier_data_dir = tmp_path / "verifier_data" / "family-a"
    verifier_data_dir.mkdir(parents=True)
    (verifier_data_dir / "golden.txt").write_text("golden\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha("family"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_tree(verifier_data_dir),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Verifier script missing") as exc_info:
        verify_pre_grading_hashes(
            _codex_long_task(),
            manifest,
            _sha("grader"),
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=tmp_path / "verifier_data",
        )

    assert exc_info.value.affected_artifact == "verifier"


def test_verify_pre_grading_hashes_reports_missing_verifier_data_dir(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    milestones_dir = verifiers_dir / "milestones"
    milestones_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (milestones_dir / "m1.sh").write_text("echo milestone\n", encoding="utf-8")
    (milestones_dir / "m1.sh").chmod(0o755)

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha("family"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": _sha("data"),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Verifier data missing") as exc_info:
        verify_pre_grading_hashes(
            _codex_long_task(),
            manifest,
            _sha("grader"),
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=tmp_path / "verifier_data",
        )

    assert exc_info.value.affected_artifact == "verifier_data"


def test_verify_pre_grading_hashes_rejects_untracked_verifier_tree_files(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_dir = tmp_path / "verifier_data" / "family-a"
    milestones_dir = verifiers_dir / "milestones"
    milestones_dir.mkdir(parents=True)
    verifier_data_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (milestones_dir / "m1.sh").write_text("echo milestone\n", encoding="utf-8")
    (milestones_dir / "m1.sh").chmod(0o755)
    (verifier_data_dir / "golden.txt").write_text("golden\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": "sha256:grader",
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha("family"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_tree(verifier_data_dir),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    (verifiers_dir / "helpers.sh").write_text("echo helper\n", encoding="utf-8")
    task = _codex_long_task()

    with pytest.raises(ManifestMismatchError, match="untracked files") as exc_info:
        verify_pre_grading_hashes(
            task,
            manifest,
            "sha256:grader",
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=tmp_path / "verifier_data",
        )

    assert exc_info.value.affected_artifact == "verifier"


def test_verify_pre_grading_hashes_detects_family_spec_drift_when_configured(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_dir = tmp_path / "verifier_data" / "family-a"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    milestones_dir = verifiers_dir / "milestones"
    milestones_dir.mkdir(parents=True)
    verifier_data_dir.mkdir(parents=True)
    scenario_families_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (milestones_dir / "m1.sh").write_text("echo milestone\n", encoding="utf-8")
    (milestones_dir / "m1.sh").chmod(0o755)
    (verifier_data_dir / "golden.txt").write_text("golden\n", encoding="utf-8")
    family_spec_path = scenario_families_dir / "family.yaml"
    family_spec_path.write_text("grading_invariant:\n  functional_checks: []\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": "sha256:grader",
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha("grading_invariant:\n  functional_checks: []\n"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_tree(verifier_data_dir),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    family_spec_path.write_text("grading_invariant:\n  functional_checks:\n    - command: pytest\n", encoding="utf-8")
    task = _codex_long_task()

    with pytest.raises(ManifestMismatchError, match="Family spec hash mismatch") as exc_info:
        verify_pre_grading_hashes(
            task,
            manifest,
            "sha256:grader",
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=tmp_path / "verifier_data",
            scenario_families_dir=tmp_path / "scenario_families",
        )

    assert exc_info.value.affected_artifact == "family_spec"


def test_verify_pre_grading_hashes_accepts_test_node_milestone_hashes(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_root = tmp_path / "verifier_data"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    verifier_data_dir = verifier_data_root / "family-a" / "v1"
    variant_repo_dir = scenario_families_dir / "variants" / "v1" / "repo"
    scenario_families_dir.mkdir(parents=True)
    variant_repo_dir.mkdir(parents=True)
    (verifier_data_dir / "hidden_tests").mkdir(parents=True)
    (verifier_data_dir / "red_team").mkdir(parents=True)
    (verifier_data_dir / "followup").mkdir(parents=True)
    verifiers_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (variant_repo_dir / "AGENTS.md").write_text("agent brief\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_example.py").write_text("def test_round_one_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_property.py").write_text("def test_property_contract():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_followup.py").write_text("def test_round_two_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "red_team" / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (verifier_data_dir / "followup" / "brief.md").write_text("follow up\n", encoding="utf-8")
    (verifier_data_dir / "calibration.json").write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
    oracle_dir = scenario_families_dir / "variants" / "v1" / "oracle"
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (oracle_dir / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_spec = _modern_family_spec()
    family_spec_path = scenario_families_dir / "family.yaml"
    family_spec_text = yaml.safe_dump(family_spec, sort_keys=False)
    family_spec_path.write_text(family_spec_text, encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": _sha(
                    "verifier_data/family-a/v1/calibration.json"
                    "verifier_data/family-a/v1/followup/brief.md"
                    "verifier_data/family-a/v1/hidden_tests/test_example.py"
                    "verifier_data/family-a/v1/hidden_tests/test_followup.py"
                    "verifier_data/family-a/v1/red_team/run_all.sh"
                ),
                "milestone_hashes": {
                    "m1": _sha('{"id":"m1","pass_rule":"all","test_nodes":["tests/hidden/test_example.py::test_round_one_green"]}'),
                    "m2": _sha('{"id":"m2","pass_rule":"all","test_nodes":["tests/hidden/test_property.py::test_property_contract"]}'),
                    "m3": _sha('{"id":"m3","pass_rule":"any","test_nodes":["tests/hidden/test_followup.py::*"]}'),
                },
            }
        ],
    }

    from lumo_flywheel_serving.task_orchestrator import milestone_contract_hash, sha256_path_set

    manifest["variants"][0]["milestone_hashes"] = {
        milestone_id: milestone_contract_hash(
            family_spec,
            family_id="family-a",
            variant_id="v1",
            milestone_id=milestone_id,
        )
        for milestone_id in ("m1", "m2", "m3")
    }
    manifest["variants"][0]["verifier_data_hash"] = sha256_path_set(
        [
            "verifier_data/family-a/v1/hidden_tests",
            "verifier_data/family-a/v1/red_team",
            "verifier_data/family-a/v1/calibration.json",
            "verifier_data/family-a/v1/followup/brief.md",
        ],
        repo_root=tmp_path,
    )

    verify_pre_grading_hashes(
        _codex_long_task(),
        manifest,
        _sha("grader"),
        image_digest_resolver=lambda image_ref: image_ref,
        verifiers_dir=tmp_path / "verifiers",
        verifier_data_dir=verifier_data_root,
        scenario_families_dir=tmp_path / "scenario_families",
    )


def test_verify_pre_grading_hashes_hashes_declared_variant_asset_subset(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_root = tmp_path / "verifier_data"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    verifier_data_dir = verifier_data_root / "family-a"
    variant_data_dir = verifier_data_dir / "v1"
    other_variant_dir = verifier_data_dir / "v2"
    variant_repo_dir = scenario_families_dir / "variants" / "v1" / "repo"
    scenario_families_dir.mkdir(parents=True)
    variant_repo_dir.mkdir(parents=True)
    (variant_data_dir / "hidden_tests").mkdir(parents=True)
    other_variant_dir.mkdir(parents=True)
    verifiers_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (variant_repo_dir / "AGENTS.md").write_text("agent brief\n", encoding="utf-8")
    (variant_data_dir / "hidden_tests" / "test_example.py").write_text("def test_round_one_green():\n    pass\n", encoding="utf-8")
    (variant_data_dir / "hidden_tests" / "test_property.py").write_text("def test_property_contract():\n    pass\n", encoding="utf-8")
    (variant_data_dir / "hidden_tests" / "test_followup.py").write_text("def test_round_two_green():\n    pass\n", encoding="utf-8")
    (variant_data_dir / "red_team").mkdir(parents=True)
    (variant_data_dir / "red_team" / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (variant_data_dir / "followup").mkdir(parents=True)
    (variant_data_dir / "followup" / "brief.md").write_text("follow up\n", encoding="utf-8")
    (variant_data_dir / "calibration.json").write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")
    (other_variant_dir / "noise.txt").write_text("ignore me\n", encoding="utf-8")
    oracle_dir = scenario_families_dir / "variants" / "v1" / "oracle"
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (oracle_dir / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_spec = _modern_family_spec()
    family_spec_text = yaml.safe_dump(family_spec, sort_keys=False)
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    from lumo_flywheel_serving.task_orchestrator import milestone_contract_hash, sha256_path_set

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_path_set(
                    [
                        "verifier_data/family-a/v1/hidden_tests",
                        "verifier_data/family-a/v1/red_team",
                        "verifier_data/family-a/v1/calibration.json",
                        "verifier_data/family-a/v1/followup/brief.md",
                    ],
                    repo_root=tmp_path,
                ),
                "milestone_hashes": {
                    milestone_id: milestone_contract_hash(
                        family_spec,
                        family_id="family-a",
                        variant_id="v1",
                        milestone_id=milestone_id,
                    )
                    for milestone_id in ("m1", "m2", "m3")
                },
            }
        ],
    }

    verify_pre_grading_hashes(
        _codex_long_task(),
        manifest,
        _sha("grader"),
        image_digest_resolver=lambda image_ref: image_ref,
        verifiers_dir=tmp_path / "verifiers",
        verifier_data_dir=verifier_data_root,
        scenario_families_dir=tmp_path / "scenario_families",
    )


def test_verify_pre_grading_hashes_accepts_family_level_template_asset_paths(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_root = tmp_path / "verifier_data"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    verifier_data_dir = verifier_data_root / "family-a" / "v1"
    variant_repo_dir = scenario_families_dir / "variants" / "v1" / "repo"
    scenario_families_dir.mkdir(parents=True)
    variant_repo_dir.mkdir(parents=True)
    (verifier_data_dir / "hidden_tests").mkdir(parents=True)
    (verifier_data_dir / "red_team").mkdir(parents=True)
    verifiers_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (variant_repo_dir / "AGENTS.md").write_text("agent brief\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_example.py").write_text("def test_round_one_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_property.py").write_text("def test_property_contract():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_followup.py").write_text("def test_round_two_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "red_team" / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (verifier_data_dir / "followup").mkdir(parents=True)
    (verifier_data_dir / "followup" / "brief.md").write_text("follow up\n", encoding="utf-8")
    oracle_dir = scenario_families_dir / "variants" / "v1" / "oracle"
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (oracle_dir / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")
    (verifier_data_dir / "calibration.json").write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")

    family_spec = _modern_family_spec()
    family_spec["hidden_tests"] = {
        "path": "verifier_data/family-a/<variant_id>/hidden_tests",
        "entrypoint": "test_example.py",
    }
    family_spec["red_team"] = {
        "path": "verifier_data/family-a/<variant_id>/red_team",
        "exploits_required": 6,
    }
    family_spec["calibration"] = {
        "path": "verifier_data/family-a/<variant_id>/calibration.json",
    }
    family_spec["shortcut_resistance"] = {
        "generated_from": "verifier_data/family-a/<variant_id>/red_team/",
        "min_exploits": 5,
        "mutation_score_floor": 0.85,
    }
    family_spec["difficulty_estimate"] = {
        "evidence_path": "verifier_data/family-a/<variant_id>/calibration.json",
    }
    family_spec["variants"][0]["hidden_tests"] = {
        "milestone_map": {
            "m1": ["tests/hidden/test_example.py::test_round_one_green"],
            "m3": ["tests/hidden/test_followup.py::*"],
        },
    }
    family_spec["variants"][0].pop("red_team")
    family_spec["variants"][0].pop("calibration")

    family_spec_text = yaml.safe_dump(family_spec, sort_keys=False)
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    from lumo_flywheel_serving.task_orchestrator import milestone_contract_hash, sha256_path_set

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_path_set(
                    [
                        "verifier_data/family-a/v1/hidden_tests",
                        "verifier_data/family-a/v1/red_team",
                        "verifier_data/family-a/v1/calibration.json",
                        "verifier_data/family-a/v1/followup/brief.md",
                    ],
                    repo_root=tmp_path,
                ),
                "milestone_hashes": {
                    milestone_id: milestone_contract_hash(
                        family_spec,
                        family_id="family-a",
                        variant_id="v1",
                        milestone_id=milestone_id,
                    )
                    for milestone_id in ("m1", "m2", "m3")
                },
            }
        ],
    }

    verify_pre_grading_hashes(
        _codex_long_task(),
        manifest,
        _sha("grader"),
        image_digest_resolver=lambda image_ref: image_ref,
        verifiers_dir=tmp_path / "verifiers",
        verifier_data_dir=verifier_data_root,
        scenario_families_dir=tmp_path / "scenario_families",
    )


def test_verify_pre_grading_hashes_rejects_missing_interactive_repo_brief_source(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_root = tmp_path / "verifier_data"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    verifier_data_dir = verifier_data_root / "family-a" / "v1"
    variant_repo_dir = scenario_families_dir / "variants" / "v1" / "repo"
    scenario_families_dir.mkdir(parents=True)
    variant_repo_dir.mkdir(parents=True)
    (verifier_data_dir / "hidden_tests").mkdir(parents=True)
    (verifier_data_dir / "red_team").mkdir(parents=True)
    verifiers_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (variant_repo_dir / "AGENTS.md").write_text("agent brief\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_example.py").write_text("def test_round_one_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_property.py").write_text("def test_property_contract():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_followup.py").write_text("def test_round_two_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "red_team" / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (verifier_data_dir / "followup").mkdir(parents=True)
    (verifier_data_dir / "followup" / "brief.md").write_text("follow up\n", encoding="utf-8")
    oracle_dir = scenario_families_dir / "variants" / "v1" / "oracle"
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (oracle_dir / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")
    (verifier_data_dir / "calibration.json").write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")

    family_spec = _modern_family_spec()
    family_spec["interactive"]["round_1"]["brief_source"] = "repo/followup.md"
    family_spec_text = yaml.safe_dump(family_spec, sort_keys=False)
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    from lumo_flywheel_serving.task_orchestrator import milestone_contract_hash, sha256_path_set

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_path_set(
                    [
                        "verifier_data/family-a/v1/hidden_tests",
                        "verifier_data/family-a/v1/red_team",
                        "verifier_data/family-a/v1/calibration.json",
                        "verifier_data/family-a/v1/followup/brief.md",
                    ],
                    repo_root=tmp_path,
                ),
                "milestone_hashes": {
                    milestone_id: milestone_contract_hash(
                        family_spec,
                        family_id="family-a",
                        variant_id="v1",
                        milestone_id=milestone_id,
                    )
                    for milestone_id in ("m1", "m2", "m3")
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Interactive asset 'brief_source' must resolve to a file"):
        verify_pre_grading_hashes(
            _codex_long_task(),
            manifest,
            _sha("grader"),
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=verifier_data_root,
            scenario_families_dir=tmp_path / "scenario_families",
        )


def test_verify_pre_grading_hashes_rejects_directory_calibration_asset(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_root = tmp_path / "verifier_data"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    verifier_data_dir = verifier_data_root / "family-a" / "v1"
    scenario_families_dir.mkdir(parents=True)
    (verifier_data_dir / "hidden_tests").mkdir(parents=True)
    (verifier_data_dir / "red_team").mkdir(parents=True)
    (verifier_data_dir / "calibration.json").mkdir(parents=True)
    verifiers_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (verifier_data_dir / "hidden_tests" / "test_example.py").write_text("def test_round_one_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_property.py").write_text("def test_property_contract():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_followup.py").write_text("def test_round_two_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "red_team" / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (verifier_data_dir / "followup").mkdir(parents=True)
    (verifier_data_dir / "followup" / "brief.md").write_text("follow up\n", encoding="utf-8")
    oracle_dir = scenario_families_dir / "variants" / "v1" / "oracle"
    oracle_dir.mkdir(parents=True)
    (oracle_dir / "solution.patch").write_text("diff --git a/x b/x\n", encoding="utf-8")
    (oracle_dir / "solution_followup.patch").write_text("diff --git a/y b/y\n", encoding="utf-8")

    family_spec = _modern_family_spec()
    family_spec_text = yaml.safe_dump(family_spec, sort_keys=False)
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    from lumo_flywheel_serving.task_orchestrator import milestone_contract_hash, sha256_path_set

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_path_set(
                    [
                        "verifier_data/family-a/v1/hidden_tests",
                        "verifier_data/family-a/v1/red_team",
                        "verifier_data/family-a/v1/calibration.json",
                        "verifier_data/family-a/v1/followup/brief.md",
                    ],
                    repo_root=tmp_path,
                ),
                "milestone_hashes": {
                    milestone_id: milestone_contract_hash(
                        family_spec,
                        family_id="family-a",
                        variant_id="v1",
                        milestone_id=milestone_id,
                    )
                    for milestone_id in ("m1", "m2", "m3")
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Calibration asset must resolve to a file"):
        verify_pre_grading_hashes(
            _codex_long_task(),
            manifest,
            _sha("grader"),
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=verifier_data_root,
            scenario_families_dir=tmp_path / "scenario_families",
        )


def test_verify_pre_grading_hashes_rejects_missing_template_oracle_asset(tmp_path: Path) -> None:
    verifiers_dir = tmp_path / "verifiers" / "family-a"
    verifier_data_root = tmp_path / "verifier_data"
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    verifier_data_dir = verifier_data_root / "family-a" / "v1"
    scenario_families_dir.mkdir(parents=True)
    (verifier_data_dir / "hidden_tests").mkdir(parents=True)
    (verifier_data_dir / "red_team").mkdir(parents=True)
    verifiers_dir.mkdir(parents=True)

    verify_path = verifiers_dir / "verify.sh"
    verify_path.write_text("echo verify\n", encoding="utf-8")
    verify_path.chmod(0o755)
    (verifier_data_dir / "hidden_tests" / "test_example.py").write_text("def test_round_one_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_property.py").write_text("def test_property_contract():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "hidden_tests" / "test_followup.py").write_text("def test_round_two_green():\n    pass\n", encoding="utf-8")
    (verifier_data_dir / "red_team" / "run_all.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (verifier_data_dir / "followup").mkdir(parents=True)
    (verifier_data_dir / "followup" / "brief.md").write_text("follow up\n", encoding="utf-8")
    (verifier_data_dir / "calibration.json").write_text('{"eligible_for_freeze": false}\n', encoding="utf-8")

    family_spec = _modern_family_spec()
    family_spec["oracle"] = {
        "path": "oracle/<variant_id>/solution.patch",
        "followup_path": "oracle/<variant_id>/solution_followup.patch",
    }
    family_spec["variants"][0].pop("oracle")
    family_spec_text = yaml.safe_dump(family_spec, sort_keys=False)
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    from lumo_flywheel_serving.task_orchestrator import milestone_contract_hash, sha256_path_set

    manifest = {
        "manifest_version": 4,
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("echo verify\n"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_path_set(
                    [
                        "verifier_data/family-a/v1/hidden_tests",
                        "verifier_data/family-a/v1/red_team",
                        "verifier_data/family-a/v1/calibration.json",
                        "verifier_data/family-a/v1/followup/brief.md",
                    ],
                    repo_root=tmp_path,
                ),
                "milestone_hashes": {
                    milestone_id: milestone_contract_hash(
                        family_spec,
                        family_id="family-a",
                        variant_id="v1",
                        milestone_id=milestone_id,
                    )
                    for milestone_id in ("m1", "m2", "m3")
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Oracle asset 'path' must resolve to a file"):
        verify_pre_grading_hashes(
            _codex_long_task(),
            manifest,
            _sha("grader"),
            image_digest_resolver=lambda image_ref: image_ref,
            verifiers_dir=tmp_path / "verifiers",
            verifier_data_dir=verifier_data_root,
            scenario_families_dir=tmp_path / "scenario_families",
        )


def test_verify_pre_run_hashes_reports_missing_agents_md(tmp_path: Path) -> None:
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    scenario_families_dir.mkdir(parents=True)
    family_spec_path = scenario_families_dir / "family.yaml"
    family_spec_path.write_text("grading_invariant:\n  functional_checks: []\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("verify"),
                "family_spec_hash": _sha("grading_invariant:\n  functional_checks: []\n"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": _sha("data"),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="AGENTS.md missing") as exc_info:
        verify_pre_run_hashes(
            replace(_codex_long_task(), image_digest=_sha("image")),
            manifest,
            image_digest_resolver=lambda image_ref: image_ref,
            agents_md_resolver=lambda image_ref: tmp_path / "missing" / "AGENTS.md",
            scenario_families_dir=tmp_path / "scenario_families",
        )

    assert exc_info.value.affected_artifact == "agents_md"


def test_verify_pre_run_hashes_reports_missing_family_spec(tmp_path: Path) -> None:
    agents_md_path = tmp_path / "AGENTS.md"
    agents_md_path.write_text("task description\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("verify"),
                "family_spec_hash": _sha("family"),
                "agents_md_hash": _sha("task description\n"),
                "verifier_data_hash": _sha("data"),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Family spec missing") as exc_info:
        verify_pre_run_hashes(
            replace(_codex_long_task(), image_digest=_sha("image")),
            manifest,
            image_digest_resolver=lambda image_ref: image_ref,
            agents_md_resolver=lambda image_ref: agents_md_path,
            scenario_families_dir=tmp_path / "scenario_families",
        )

    assert exc_info.value.affected_artifact == "family_spec"


def test_verify_pre_run_hashes_cleans_up_transient_agents_md_extract(tmp_path: Path) -> None:
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    scenario_families_dir.mkdir(parents=True)
    family_spec_text = "grading_invariant:\n  functional_checks: []\n"
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    transient_extract = tmp_path / "codex-bench-agents-temp"
    agents_md_path = transient_extract / "AGENTS.md"
    transient_extract.mkdir()
    agents_md_path.write_text("task description\n", encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("verify"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("task description\n"),
                "verifier_data_hash": _sha("data"),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    verify_pre_run_hashes(
        replace(_codex_long_task(), image_digest=_sha("image")),
        manifest,
        image_digest_resolver=lambda image_ref: image_ref,
        agents_md_resolver=lambda image_ref: agents_md_path,
        scenario_families_dir=tmp_path / "scenario_families",
    )

    assert not transient_extract.exists()


def test_verify_pre_run_hashes_reports_missing_locked_image(tmp_path: Path) -> None:
    scenario_families_dir = tmp_path / "scenario_families" / "family-a"
    scenario_families_dir.mkdir(parents=True)
    family_spec_text = "grading_invariant:\n  functional_checks: []\n"
    (scenario_families_dir / "family.yaml").write_text(family_spec_text, encoding="utf-8")

    manifest = {
        "manifest_version": 4,
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha("verify"),
                "family_spec_hash": _sha(family_spec_text),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": _sha("data"),
                "milestone_hashes": {
                    "m1": _sha("echo milestone\n"),
                },
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Locked image missing") as exc_info:
        verify_pre_run_hashes(
            replace(_codex_long_task(), image_digest=_sha("image")),
            manifest,
            image_digest_resolver=lambda image_ref: (_ for _ in ()).throw(
                subprocess.CalledProcessError(1, ["docker", "image", "inspect", image_ref])
            ),
            agents_md_resolver=lambda image_ref: tmp_path / "AGENTS.md",
            scenario_families_dir=tmp_path / "scenario_families",
        )

    assert exc_info.value.affected_artifact == "image"


def test_validate_family_spec_rejects_missing_functional_checks() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["grading_invariant"]["functional_checks"] = []

    with pytest.raises(ManifestMismatchError, match="functional_checks"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_variant_mismatch() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec(variant_id="other-variant")

    with pytest.raises(ManifestMismatchError, match="does not declare variant_id 'v1'"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_missing_family_id() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec.pop("family_id")

    with pytest.raises(ManifestMismatchError, match="non-empty family_id"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_unpinned_repo_base_image() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["repo_pattern"]["base_image"] = "python:3.12"

    with pytest.raises(ManifestMismatchError, match="repo_pattern.base_image"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_malformed_repo_base_image_digest() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["repo_pattern"]["base_image"] = "python:3.12@sha256:not-a-real-digest"

    with pytest.raises(ManifestMismatchError, match="repo_pattern.base_image"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_missing_state_based_grading_contract() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["grading_invariant"]["type"] = "patch_based"

    with pytest.raises(ManifestMismatchError, match="grading_invariant.type"):
        validate_family_spec(task, family_spec)

    family_spec = _valid_family_spec()
    family_spec["grading_invariant"].pop("description")

    with pytest.raises(ManifestMismatchError, match="grading_invariant.description"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_too_few_variants() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["variants"] = family_spec["variants"][:2]

    with pytest.raises(ManifestMismatchError, match="at least 3 variants"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_invalid_target_solve_rate() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["difficulty_estimate"]["target_solve_rate"] = "10-90%"

    with pytest.raises(ManifestMismatchError, match="20-80% band"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_accepts_modern_quality_contract() -> None:
    validate_family_spec(_codex_long_task(), _modern_family_spec())


def test_validate_family_spec_accepts_family_level_template_oracle_paths() -> None:
    family_spec = _modern_family_spec()
    family_spec["oracle"] = {
        "path": "oracle/<variant_id>/solution.patch",
        "followup_path": "oracle/<variant_id>/solution_followup.patch",
    }
    family_spec["variants"][0].pop("oracle")

    validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_accepts_templated_interactive_asset_paths() -> None:
    family_spec = _modern_family_spec()
    family_spec["interactive"]["round_2"]["brief_source"] = "verifier_data/family-a/<variant_id>/followup/brief.md"
    family_spec["interactive"]["round_1"]["grader_between_rounds"] = (
        "verifier_data/family-a/<variant_id>/hidden_tests/test_example.py"
    )

    validate_family_spec(_codex_long_task(), family_spec)


@pytest.mark.parametrize("section_name", ["oracle", "hidden_tests", "red_team", "calibration"])
def test_validate_family_spec_rejects_non_mapping_family_quality_section(section_name: str) -> None:
    family_spec = _modern_family_spec()
    family_spec[section_name] = []
    family_spec["variants"][0].pop(section_name, None)

    with pytest.raises(ManifestMismatchError, match=rf"family\.{section_name} must be a mapping"):
        validate_family_spec(_codex_long_task(), family_spec)


@pytest.mark.parametrize("section_name", ["oracle", "hidden_tests", "red_team", "calibration"])
def test_validate_family_spec_rejects_non_mapping_variant_quality_override(section_name: str) -> None:
    family_spec = _modern_family_spec()
    family_spec[section_name] = family_spec["variants"][0].pop(section_name)
    family_spec["variants"][0][section_name] = []

    with pytest.raises(ManifestMismatchError, match=rf"variants\[0\].*{section_name} must be a mapping"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_declared_verifier_data_path_traversal() -> None:
    family_spec = _modern_family_spec()
    family_spec["variants"][0]["hidden_tests"]["path"] = "verifier_data/family-a/<variant_id>/../v2/hidden_tests"

    with pytest.raises(ManifestMismatchError, match="must define variants\\[0\\]\\.hidden_tests\\.path"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_oracle_path_traversal() -> None:
    family_spec = _modern_family_spec()
    family_spec["variants"][0]["oracle"]["path"] = "oracle/<variant_id>/../other/solution.patch"

    with pytest.raises(ManifestMismatchError, match="must define variants\\[0\\]\\.oracle\\.path"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_hidden_tests_entrypoint_traversal() -> None:
    family_spec = _modern_family_spec()
    family_spec["variants"][0]["hidden_tests"]["entrypoint"] = "../red_team/run_all.sh"

    with pytest.raises(ManifestMismatchError, match="must define variants\\[0\\]\\.hidden_tests\\.entrypoint"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_hidden_test_node_path_traversal() -> None:
    family_spec = _modern_family_spec()
    family_spec["variants"][0]["hidden_tests"]["milestone_map"]["m1"] = [
        "tests/hidden/../../red_team/run_all.sh::test_round_one_green"
    ]

    with pytest.raises(
        ManifestMismatchError,
        match="must define variants\\[0\\]\\.hidden_tests\\.milestone_map\\[m1\\]\\[0\\]",
    ):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_interactive_asset_path_traversal() -> None:
    family_spec = _modern_family_spec()
    family_spec["interactive"]["round_2"]["brief_source"] = "../outside.md"

    with pytest.raises(ManifestMismatchError, match="interactive\\.round_2\\.brief_source"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_non_variant_scoped_milestone_map_entries() -> None:
    family_spec = _modern_family_spec()
    family_spec["variants"][0]["hidden_tests"]["milestone_map"]["m2"] = [
        "tests/hidden/test_property.py::test_property_contract"
    ]

    with pytest.raises(ManifestMismatchError, match="must only define hidden_tests\\.milestone_map entries"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_missing_variant_scoped_hidden_tests() -> None:
    family_spec = _modern_family_spec()
    family_spec["variants"][0]["hidden_tests"]["milestone_map"].pop("m3")

    with pytest.raises(ManifestMismatchError, match="milestone test-node wiring"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_requires_spoofed_functional_success_exploit() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["shortcut_resistance"]["known_exploits_tested"] = [
        "Delete failing tests",
        "Monkeypatch the legacy helper to stop raising",
        "Downgrade back to the original version",
    ]

    with pytest.raises(ManifestMismatchError, match="spoofed functional-check success"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_mismatched_generated_red_team_path() -> None:
    family_spec = _modern_family_spec()
    family_spec["shortcut_resistance"]["generated_from"] = "verifier_data/family-a/v1/other_red_team/"

    with pytest.raises(ManifestMismatchError, match="shortcut_resistance.generated_from"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_mismatched_calibration_evidence_path() -> None:
    family_spec = _modern_family_spec()
    family_spec["difficulty_estimate"]["evidence_path"] = "verifier_data/family-a/v1/other_calibration.json"

    with pytest.raises(ManifestMismatchError, match="difficulty_estimate.evidence_path"):
        validate_family_spec(_codex_long_task(), family_spec)


def test_validate_family_spec_rejects_unknown_repo_source_mode() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["variants"][0]["repo_source"] = "mirrored"

    with pytest.raises(ManifestMismatchError, match="repo_source"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_derived_variant_without_provenance() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["variants"][0]["repo_source"] = "derived:github.com/example/repo"

    with pytest.raises(ManifestMismatchError, match="must define provenance"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_public_dev_variant_without_redistribution_rights() -> None:
    task = replace(_codex_long_task(), pool_or_split="public_dev")
    family_spec = _valid_family_spec()
    family_spec["variants"][0]["repo_source"] = "derived:github.com/example/repo"
    family_spec["variants"][0]["provenance"] = {
        "source_repo": "https://github.com/example/repo",
        "license": "MIT",
        "redistribution_ok": False,
        "modification_notice": "Derived fixture",
    }

    with pytest.raises(ManifestMismatchError, match="Public-Dev"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_invalid_expected_final_state_shape() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["grading_invariant"]["expected_final_state"] = [{"a": "x", "b": "y"}]

    with pytest.raises(ManifestMismatchError, match="single-entry mappings"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_requires_trusted_phase3_invariant() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["grading_invariant"]["expected_final_state"] = [
        {"test_suite_check": "The pytest suite passes from Phase 2."},
        {"functional_check": "Functional checks exit 0."},
    ]

    with pytest.raises(ManifestMismatchError, match="trusted Phase 3 expected_final_state invariant"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_non_monotonic_milestone_partial_credit() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["milestones"] = [
        {
            "id": "m1",
            "description": "First milestone",
            "check_script": "verifiers/family-a/milestones/m1.sh",
            "partial_credit": 0.3,
        },
        {
            "id": "m2",
            "description": "Second milestone",
            "check_script": "verifiers/family-a/milestones/m2.sh",
            "partial_credit": 0.2,
        },
        {
            "id": "m3",
            "description": "Third milestone",
            "check_script": "verifiers/family-a/milestones/m3.sh",
            "partial_credit": 0.5,
        },
    ]

    with pytest.raises(ManifestMismatchError, match="monotonically non-decreasing"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_too_few_milestones() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["milestones"] = family_spec["milestones"][:2]

    with pytest.raises(ManifestMismatchError, match="at least 3 milestones"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_missing_breakage_class_fields() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["breakage_class"]["surfaces"] = []

    with pytest.raises(ManifestMismatchError, match="breakage_class.surfaces"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_too_few_breakage_surfaces() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["breakage_class"]["surfaces"] = ["tests", "application"]

    with pytest.raises(ManifestMismatchError, match="at least 3 breakage_class.surfaces"):
        validate_family_spec(task, family_spec)


def test_execute_task_records_codex_long_manifest_versions(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)
    hooks = _hooks(events)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([7, 9])

    result = asyncio.run(orchestrator.execute_task(_codex_long_task(), pool_manager, manifest_state, config))

    assert result.outcome == "resolved"
    assert pool_manager.claim_calls[0]["launch_manifest_ver"] == 7
    assert pool_manager.finish_calls[0]["grading_manifest_ver"] == 9
    assert pool_manager.finish_calls[0]["snapshot_image_ref"] == "snapshot-ref"
    assert pool_manager.finish_calls[0]["codex_long_pass"] is True
    assert pool_manager.finish_calls[0]["milestone_results"] == {"m1": True}
    assert events == [
        "flush:127.0.0.1:8000",
        "health:qwen3.5-27b",
        "pre-run:7",
        "family-spec:family-a",
        "before:family-a/v1",
        "setup:7",
        "invoke",
        "snapshot:family-a/v1/qwen3.5-27b/codex/seed1/attempt1",
        "pre-grade:9",
        "family-spec:family-a",
        "phase2:snapshot-ref",
        "phase3:codex-long-grader",
        "cleanup:True",
        "after:family-a/v1",
        "rm:container-1:True",
    ]


def test_execute_task_health_check_uses_served_model_surface_for_registry_key(tmp_path: Path) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    base_config = _config(tmp_path)
    config = replace(
        base_config,
        model_registry={
            "sprint3-qwen": {
                "max_model_len": 65536,
                "served_model_name": "qwen3.5-27b-served",
                "lora_modules": {"codex-sft-all": "/models/adapters/codex-sft-all"},
            }
        },
    )
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([7, 9])

    task = replace(_codex_long_task(), model_id="sprint3-qwen")
    result = asyncio.run(orchestrator.execute_task(task, pool_manager, manifest_state, config))

    assert result.outcome == "resolved"
    assert events[:2] == ["flush:127.0.0.1:8000", "health:qwen3.5-27b-served"]


def test_execute_task_health_check_accepts_lora_adapter_surface_id(tmp_path: Path) -> None:
    events: list[str] = []
    hooks = _hooks(events)
    base_config = _config(tmp_path)
    config = replace(
        base_config,
        model_registry={
            "sprint3-qwen": {
                "max_model_len": 65536,
                "served_model_name": "qwen3.5-27b-served",
                "lora_modules": {"codex-sft-all": "/models/adapters/codex-sft-all"},
            }
        },
    )
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([7, 9])

    task = replace(_codex_long_task(), model_id="codex-sft-all")
    result = asyncio.run(orchestrator.execute_task(task, pool_manager, manifest_state, config))

    assert result.outcome == "resolved"
    assert events[:2] == ["flush:127.0.0.1:8000", "health:codex-sft-all"]


def test_execute_task_raises_duplicate_claim_before_running(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)
    hooks = _hooks(events)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager(claim_result=False)
    manifest_state = _ManifestState([3])

    with pytest.raises(DuplicateClaimError):
        asyncio.run(orchestrator.execute_task(_codex_long_task(), pool_manager, manifest_state, config))

    assert pool_manager.finish_calls == []
    assert events == [
        "flush:127.0.0.1:8000",
        "health:qwen3.5-27b",
        "pre-run:3",
        "family-spec:family-a",
    ]


def test_execute_task_preserves_snapshot_on_manifest_mismatch(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)

    def raise_mismatch(task: TaskSpec, manifest: dict, grader_image_ref: str) -> None:
        raise ManifestMismatchError("verifier drift", affected_artifact="verifier")

    hooks = _hooks(events, verify_pre_grading=raise_mismatch)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([3, 5])

    with pytest.raises(ManifestMismatchError):
        asyncio.run(orchestrator.execute_task(_codex_long_task(), pool_manager, manifest_state, config))

    assert pool_manager.finish_calls[0]["outcome"] == "crash"
    assert pool_manager.finish_calls[0]["grading_manifest_ver"] == 3
    assert pool_manager.finish_calls[0]["snapshot_image_ref"] == "snapshot-ref"
    assert events[-2:] == ["after:family-a/v1", "rm:container-1:True"]


def test_execute_task_rejects_invalid_codex_long_family_spec_before_claim(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)

    def invalid_family_spec(family_id: str, **kwargs) -> dict:
        spec = _valid_family_spec(family_id=family_id)
        spec.pop("breakage_class")
        return spec

    hooks = _hooks(events, load_family_spec_hook=invalid_family_spec)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([3])

    with pytest.raises(ManifestMismatchError, match="breakage_class"):
        asyncio.run(orchestrator.execute_task(_codex_long_task(), pool_manager, manifest_state, config))

    assert pool_manager.claim_calls == []
    assert pool_manager.finish_calls == []
    assert events == [
        "flush:127.0.0.1:8000",
        "health:qwen3.5-27b",
        "pre-run:3",
        "family-spec:family-a",
    ]


def test_execute_task_regrade_path_uses_retained_snapshot(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)
    hooks = _hooks(events)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([11])

    result = asyncio.run(
        orchestrator.execute_task(
            _codex_long_task(dispatch_decision="regrade_needed", attempt=2, regrade_snapshot_ref="snapshot-ref"),
            pool_manager,
            manifest_state,
            config,
        )
    )

    assert result.outcome == "resolved"
    assert pool_manager.claim_calls[0]["launch_manifest_ver"] is None
    assert pool_manager.finish_calls[0]["grading_manifest_ver"] == 11
    assert pool_manager.finish_calls[0]["snapshot_image_ref"] == "snapshot-ref"
    assert events == [
        "image-exists:snapshot-ref",
        "pre-grade:11",
        "family-spec:family-a",
        "phase2:snapshot-ref",
        "phase3:codex-long-grader",
        "cleanup:True",
    ]


def test_execute_task_regrade_path_checks_snapshot_before_claiming(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)

    async def image_missing(snapshot_ref: str) -> bool:
        return False

    hooks = _hooks(events, docker_image_exists=image_missing)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([11])

    with pytest.raises(TaskDispatchError, match="Snapshot may have been pruned|Full rerun required"):
        asyncio.run(
            orchestrator.execute_task(
                _codex_long_task(dispatch_decision="regrade_needed", attempt=2, regrade_snapshot_ref="snapshot-ref"),
                pool_manager,
                manifest_state,
                config,
            )
        )

    assert pool_manager.claim_calls == []
    assert pool_manager.finish_calls == []
    assert events == ["image-exists:snapshot-ref"]


def test_execute_task_passes_configured_codex_long_artifact_paths_to_hooks(tmp_path: Path) -> None:
    events: list[str] = []
    config = _config(tmp_path)
    orchestrator = TaskOrchestrator(
        _hooks(
            events,
            verify_pre_run=(
                lambda task, manifest, *, scenario_families_dir: events.append(
                    f"pre-run-dir:{scenario_families_dir}"
                )
            ),
            verify_pre_grading=(
                lambda task, manifest, grader_image_ref, *, verifiers_dir, verifier_data_dir, scenario_families_dir: events.append(
                    f"pre-grade-dirs:{verifiers_dir}:{verifier_data_dir}:{scenario_families_dir}"
                )
            ),
            load_family_spec_hook=(
                lambda family_id, *, scenario_families_dir: (
                    events.append(f"family-spec-dir:{scenario_families_dir}")
                    or _valid_family_spec(family_id=family_id)
                )
            ),
        )
    )
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([7, 9])

    result = asyncio.run(orchestrator.execute_task(_codex_long_task(), pool_manager, manifest_state, config))

    assert result.outcome == "resolved"
    assert f"pre-run-dir:{config.paths.scenario_families_dir}" in events
    assert (
        f"pre-grade-dirs:{config.paths.verifiers_dir}:{config.paths.verifier_data_dir}:{config.paths.scenario_families_dir}" in events
    )
    assert f"family-spec-dir:{config.paths.scenario_families_dir}" in events


def test_execute_task_regrade_path_uses_configured_unique_grading_dir(tmp_path: Path) -> None:
    events: list[str] = []
    grading_dirs: list[str] = []
    config = _config(tmp_path)

    async def record_phase2(snapshot_ref: str, task: TaskSpec, family_spec: dict, grading_dir: str) -> None:
        grading_dirs.append(grading_dir)

    hooks = _hooks(events, phase2_hook=record_phase2)
    orchestrator = TaskOrchestrator(hooks)
    pool_manager = _PoolManager()
    manifest_state = _ManifestState([11])
    task = _codex_long_task(dispatch_decision="regrade_needed", attempt=2, regrade_snapshot_ref="snapshot-ref")

    result = asyncio.run(orchestrator.execute_task(task, pool_manager, manifest_state, config))

    assert result.outcome == "resolved"
    assert grading_dirs == [
        str(
            Path(config.paths.grading_dir)
            / "family-a%2Fv1%2Fqwen3.5-27b%2Fcodex%2Fseed1%2Fattempt2"
        )
    ]


def test_manifest_state_uses_local_grader_tag_as_runtime_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "benchmark_manifest.lock"
    manifest_path.write_text("manifest_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        "lumo_flywheel_serving.task_orchestrator.load_codex_long_manifest",
        lambda path: {"manifest_version": 4, "grader_image_digest": "sha256:grader"},
    )

    state = ManifestState(manifest_path, grader_image_tag="codex-long-grader")

    assert state.manifest_version == 4
    assert state.grader_image_ref == "codex-long-grader"


def test_get_local_image_digest_uses_local_image_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lumo_flywheel_serving.task_orchestrator._cmd_output",
        lambda command: "sha256:local-image-id",
    )

    assert get_local_image_digest("codex-long-grader") == "sha256:local-image-id"


def test_get_local_image_digest_inspects_sha256_refs_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_cmd_output(command: list[str]) -> str:
        commands.append(command)
        return "sha256:local-image-id"

    monkeypatch.setattr("lumo_flywheel_serving.task_orchestrator._cmd_output", fake_cmd_output)

    assert get_local_image_digest("sha256:expected-image-id") == "sha256:local-image-id"
    assert commands == [
        [
            "docker",
            "image",
            "inspect",
            "sha256:expected-image-id",
            "--format",
            "{{.Id}}",
        ]
    ]


def test_verify_pre_grading_hashes_rejects_non_executable_verify_script(tmp_path: Path) -> None:
    task = _codex_long_task()
    verifiers_dir = tmp_path / "verifiers"
    verifier_data_dir = tmp_path / "verifier_data"
    family_dir = verifiers_dir / "family-a"
    milestones_dir = family_dir / "milestones"
    milestones_dir.mkdir(parents=True)
    verify_path = family_dir / "verify.sh"
    verify_path.write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    verify_path.chmod(0o644)
    milestone_path = milestones_dir / "m1.sh"
    milestone_path.write_text("#!/bin/bash\necho milestone\n", encoding="utf-8")
    milestone_path.chmod(0o755)
    data_dir = verifier_data_dir / "family-a"
    data_dir.mkdir(parents=True)
    (data_dir / "variant_expectations.json").write_text('{"variants": {}}', encoding="utf-8")

    manifest = {
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha(verify_path.read_text(encoding="utf-8")),
                "family_spec_hash": _sha("family-spec"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_tree(data_dir),
                "milestone_hashes": {"m1": _sha(milestone_path.read_text(encoding="utf-8"))},
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Verifier script is not executable"):
        verify_pre_grading_hashes(
            task,
            manifest,
            "grader-image",
            image_digest_resolver=lambda _ref: _sha("grader"),
            verifiers_dir=verifiers_dir,
            verifier_data_dir=verifier_data_dir,
        )


def test_verify_pre_grading_hashes_rejects_non_executable_milestone_helper(tmp_path: Path) -> None:
    task = _codex_long_task()
    verifiers_dir = tmp_path / "verifiers"
    verifier_data_dir = tmp_path / "verifier_data"
    family_dir = verifiers_dir / "family-a"
    milestones_dir = family_dir / "milestones"
    milestones_dir.mkdir(parents=True)
    verify_path = family_dir / "verify.sh"
    verify_path.write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    verify_path.chmod(0o755)
    milestone_path = milestones_dir / "m1.sh"
    milestone_path.write_text("#!/bin/bash\necho milestone\n", encoding="utf-8")
    milestone_path.chmod(0o644)
    data_dir = verifier_data_dir / "family-a"
    data_dir.mkdir(parents=True)
    (data_dir / "variant_expectations.json").write_text('{"variants": {}}', encoding="utf-8")

    manifest = {
        "grader_image_digest": _sha("grader"),
        "variants": [
            {
                "family_id": "family-a",
                "variant_id": "v1",
                "split": "train_long",
                "scenario_type": "feature_evolution",
                "image_digest": _sha("image"),
                "verifier_hash": _sha(verify_path.read_text(encoding="utf-8")),
                "family_spec_hash": _sha("family-spec"),
                "agents_md_hash": _sha("agents"),
                "verifier_data_hash": sha256_tree(data_dir),
                "milestone_hashes": {"m1": _sha(milestone_path.read_text(encoding="utf-8"))},
            }
        ],
    }

    with pytest.raises(ManifestMismatchError, match="Milestone helper is not executable"):
        verify_pre_grading_hashes(
            task,
            manifest,
            "grader-image",
            image_digest_resolver=lambda _ref: _sha("grader"),
            verifiers_dir=verifiers_dir,
            verifier_data_dir=verifier_data_dir,
        )

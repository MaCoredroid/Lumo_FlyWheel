from __future__ import annotations

import asyncio
import hashlib
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

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
    flush_prefix_cache,
    generate_codex_config,
    get_codex_harness_env,
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
        "repo_pattern": {
            "base_image": "python:3.12@sha256:" + ("a" * 64),
        },
        "grading_invariant": {
            "verifier_script": f"verifiers/{family_id}/verify.sh",
            "functional_checks": [
                {
                    "id": "fc1",
                    "command": "cd /workspace && pytest",
                    "timeout_seconds": 120,
                }
            ],
            "expected_final_state": ["workspace state validated"],
        },
        "milestones": [
            {
                "id": "m1",
                "check_script": f"verifiers/{family_id}/milestones/m1.sh",
                "partial_credit": 1.0,
            }
        ],
        "shortcut_resistance": {
            "known_exploits_tested": ["exploit-a", "exploit-b", "exploit-c"],
        },
        "variants": [
            {
                "variant_id": variant_id,
                "repo_source": "authored",
                "env_dockerfile": f"variants/{variant_id}/Dockerfile",
                "base_image_digest": "sha256:" + ("1" * 64),
            },
            {
                "variant_id": "v2",
                "repo_source": "authored",
                "env_dockerfile": "variants/v2/Dockerfile",
                "base_image_digest": "sha256:" + ("2" * 64),
            },
            {
                "variant_id": "v3",
                "repo_source": "authored",
                "env_dockerfile": "variants/v3/Dockerfile",
                "base_image_digest": "sha256:" + ("3" * 64),
            },
        ],
    }


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
    milestone_path = milestones_dir / "m1.sh"
    milestone_path.write_text("echo milestone\n", encoding="utf-8")
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
    (milestones_dir / "m1.sh").write_text("echo milestone\n", encoding="utf-8")

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
    (milestones_dir / "m1.sh").write_text("echo milestone\n", encoding="utf-8")
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
    (milestones_dir / "m1.sh").write_text("echo milestone\n", encoding="utf-8")
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


def test_validate_family_spec_rejects_too_few_variants() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["variants"] = family_spec["variants"][:2]

    with pytest.raises(ManifestMismatchError, match="at least 3 variants"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_derived_variant_without_provenance() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["variants"][0]["repo_source"] = "derived:github.com/example/repo"

    with pytest.raises(ManifestMismatchError, match="must define provenance"):
        validate_family_spec(task, family_spec)


def test_validate_family_spec_rejects_invalid_expected_final_state_shape() -> None:
    task = _codex_long_task()
    family_spec = _valid_family_spec()
    family_spec["grading_invariant"]["expected_final_state"] = [{"a": "x", "b": "y"}]

    with pytest.raises(ManifestMismatchError, match="single-entry mappings"):
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
        "health:qwen3.5-27b",
        "flush:127.0.0.1:8000",
        "pre-run:7",
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
    assert events[:2] == ["health:qwen3.5-27b-served", "flush:127.0.0.1:8000"]


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
    assert events[:2] == ["health:codex-sft-all", "flush:127.0.0.1:8000"]


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
        "health:qwen3.5-27b",
        "flush:127.0.0.1:8000",
        "pre-run:3",
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
    assert events[-1] == "after:family-a/v1"


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

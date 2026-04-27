"""Unit tests for L0cKernelMutationRunner._real_apply_and_test.

The real path's external boundaries (vLLM restart, parity probe, paired measurement
harness) are monkeypatched. The patch-apply step uses the real /usr/bin/patch so the
shell-out path itself is exercised.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from lumo_flywheel_serving import auto_research
from lumo_flywheel_serving.parity_probe import ParityProbeResult


def _write_registry(repo: Path) -> None:
    payload = {
        "qwen3.5-27b": {
            "weight_version_id": "wv-test",
            "huggingface": {"repo_id": "qwen/test-27b"},
            "served_model_id": "qwen3.5-27b",
        }
    }
    (repo / "model_registry.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_workload(repo: Path) -> Path:
    family_dir = repo / "benchmark_blueprints" / "families" / "test-family"
    family_dir.mkdir(parents=True)
    (family_dir / "seed_trace.jsonl").write_text(
        json.dumps({"turn_index": 0, "prompt_tokens": 64, "request_max_output_tokens": 16, "capture_prompt_label": "x"}) + "\n",
        encoding="utf-8",
    )
    workload_path = family_dir / "workload.yaml"
    workload_path.write_text(
        yaml.safe_dump(
            {
                "family_id": "test-family",
                "workload_distribution_id": "wd-test",
                "seed_trace_ref": "seed_trace.jsonl",
                "latency_ceiling_ms": 60000,
                "tpot_ceiling_ms": 80,
                "turn_latency_ceiling_ms": 60000,
                "avg_prompt_tokens": 64,
                "avg_output_tokens": 16,
                "measurement_window_minutes": 1,
                "rollout_baseline": 1.0,
                "target_concurrency": 1,
                "nominal_ttft_ms": 1500,
                "nominal_tpot_ms": 80,
                "nominal_turn_ms": 4200,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workload_path


def _write_parity_fixture(repo: Path) -> Path:
    family_dir = repo / "benchmark_blueprints" / "families" / "test-family" / "parity_fixture"
    family_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = family_dir / "deltanet_v1.yaml"
    fixture_path.write_text(
        yaml.safe_dump(
            {
                "fixture_id": "test-family-deltanet-v1",
                "probe_count": 2,
                "tolerances": {
                    "rtol_logit": 0.001,
                    "atol_logit": 0.001,
                    "rtol_state": 0.005,
                    "atol_state": 0.005,
                },
                "state_checkpoints_at_token": [1, 1024],
                "parity_check_method": "logit_plus_state_compare",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return fixture_path


def _seed_round(
    tmp_path: Path,
    *,
    iteration: str,
    patch_text: str,
    kernel_source_text: str = "alpha\nbeta\ngamma\n",
) -> tuple[auto_research.L0cKernelMutationRunner, Path, Path, Path, str]:
    """Lay down enough state for real apply_and_test to run end-to-end.

    Returns: (runner, round_dir, kernel_path, iteration_dir, round_id)
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_registry(repo)
    _write_workload(repo)
    fixture_path = _write_parity_fixture(repo)

    kernel_path = tmp_path / "kernels" / "chunk_delta_h.py"
    kernel_path.parent.mkdir(parents=True)
    kernel_path.write_text(kernel_source_text, encoding="utf-8")

    round_id = "qwen3.5-27b-test-family-l0c-mutation-deltanet-20260427T000000Z"
    round_dir = repo / "output" / "auto_research" / round_id
    round_dir.mkdir(parents=True)
    (round_dir / "candidates").mkdir()

    base_dir = round_dir / "kernel_base"
    base_dir.mkdir()
    (base_dir / kernel_path.name).write_text(kernel_source_text, encoding="utf-8")

    spec = {
        "round_id": round_id,
        "round_type": auto_research.L0C_MUTATION_ROUND_TYPE,
        "model_id": "qwen3.5-27b",
        "family_id": "test-family",
        "workload_file": str(repo / "benchmark_blueprints" / "families" / "test-family" / "workload.yaml"),
        "base_bundle": str(repo / "fake_base_bundle.yaml"),
        "kernel_target": "deltanet",
        "kernel_source_path": str(kernel_path),
        "parity_fixture": str(fixture_path.relative_to(repo)),
        "parity_fixture_id": "test-family-deltanet-v1",
        "harness": "real",
        "weight_version_id": "wv-test",
        "runtime": {
            "container_name": "lumo-vllm-test",
            "model_id": "qwen3.5-27b",
            "port": 18100,
            "proxy_port": 18101,
            "endpoint": "http://127.0.0.1:18101/v1",
            "metrics_url": "http://127.0.0.1:18100/metrics",
            "admin_url": "http://127.0.0.1:18101/admin",
        },
    }
    (round_dir / "round_spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    iteration_dir = round_dir / "candidates" / iteration
    iteration_dir.mkdir()
    (iteration_dir / "mutation.patch").write_text(patch_text, encoding="utf-8")

    runner = auto_research.L0cKernelMutationRunner(
        repo_root=repo,
        registry_path=repo / "model_registry.yaml",
        tuned_config_root=repo / "output" / "tuned_configs",
    )
    return runner, round_dir, kernel_path, iteration_dir, round_id


def _good_patch(kernel_path: Path) -> str:
    return (
        f"--- {kernel_path}\n+++ {kernel_path}\n"
        "@@ -1,3 +1,3 @@\n"
        "-alpha\n"
        "+ALPHA\n"
        " beta\n"
        " gamma\n"
    )


def _bad_patch(kernel_path: Path) -> str:
    return (
        f"--- {kernel_path}\n+++ {kernel_path}\n"
        "@@ -1,3 +1,3 @@\n"
        "-this_line_does_not_exist\n"
        "+REPLACEMENT\n"
        " beta\n"
        " gamma\n"
    )


def _stub_passing_helpers(monkeypatch, runner: auto_research.L0cKernelMutationRunner, *, objective: float = 1.05) -> dict[str, Any]:
    calls: dict[str, Any] = {"restart": 0, "probe": 0, "measure": 0}

    def _stub_restart(self, *, spec: dict[str, Any]) -> None:
        calls["restart"] += 1

    def _stub_probe(self, *, spec, fixture_dir, kernel_target, debug_export_dir) -> ParityProbeResult:
        calls["probe"] += 1
        return ParityProbeResult(
            pass_=True,
            fixture_id=str(spec.get("parity_fixture_id", "")),
            kernel_target=kernel_target,
            probes_total=2,
            probes_passed=2,
            first_diverging_probe=None,
            tolerance_overshoot=0.0,
            reason="ran_passed",
            checkpoints_checked=(1, 1024),
        )

    def _stub_measure(self, *, spec, iteration, iteration_dir, count):
        calls["measure"] += 1
        rows = [
            self._make_measurement_row(
                candidate_uuid="cand-uuid-test",
                candidate_label=f"l0c-attempt-{iteration}",
                role="l0c_candidate",
                measurement_index=i + 1,
                objective_value=objective + 0.0001 * i,
                harness="real",
                trace_ref=f"candidates/{iteration}/measurement_{i + 1:02d}.json",
            )
            for i in range(count)
        ]
        mean = sum(float(r["objective_value"]) for r in rows) / count
        return rows, "cand-uuid-test", mean

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_restart_serving_runtime", _stub_restart, raising=True
    )
    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_invoke_parity_probe", _stub_probe, raising=True
    )
    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_run_paired_l0c_measurements", _stub_measure, raising=True
    )
    return calls


@pytest.fixture(autouse=True)
def _ensure_patch_available() -> None:
    if shutil.which("patch") is None:
        pytest.skip("/usr/bin/patch not available")


def test_real_apply_and_test_passes_when_patch_runtime_probe_and_measure_succeed(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="001", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_good_patch(kernel_path), encoding="utf-8")
    calls = _stub_passing_helpers(monkeypatch, runner)

    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="001",
        kernel_target="deltanet",
        harness="real",
        round_root=round_dir.parent,
    )

    assert payload["outcome"] == "parity_passed", payload
    assert payload["candidate_uuid"] == "cand-uuid-test"
    assert payload["objective_mean"] == pytest.approx(1.05005, rel=1e-3)
    assert calls == {"restart": 1, "probe": 1, "measure": 1}

    parity = json.loads((iteration_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert parity["pass"] is True
    assert parity["reason"] == "ran_passed"
    assert (iteration_dir / "measurement_trace.json").is_file()
    assert (iteration_dir / "parity_probe_result.json").is_file()
    # base bytes restored (kernel_path back to baseline contents)
    assert kernel_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_real_apply_and_test_returns_compile_failed_when_patch_does_not_apply(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="002", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_bad_patch(kernel_path), encoding="utf-8")
    _stub_passing_helpers(monkeypatch, runner)

    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="002",
        kernel_target="deltanet",
        harness="real",
        round_root=round_dir.parent,
    )

    assert payload["outcome"] == "compile_failed", payload
    parity = json.loads((iteration_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert parity["pass"] is False
    assert parity["reason"] == "compile_nvcc_error"
    assert "patch_apply_failed" in (parity.get("error_detail") or "")
    assert not (iteration_dir / "measurement_trace.json").exists()
    assert kernel_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_real_apply_and_test_returns_compile_failed_when_runtime_restart_raises(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="003", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_good_patch(kernel_path), encoding="utf-8")
    _stub_passing_helpers(monkeypatch, runner)

    def _restart_explodes(self, *, spec):
        raise RuntimeError("vLLM container failed to come up")

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_restart_serving_runtime", _restart_explodes, raising=True
    )

    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="003",
        kernel_target="deltanet",
        harness="real",
        round_root=round_dir.parent,
    )
    assert payload["outcome"] == "compile_failed"
    parity = json.loads((iteration_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert "runtime_restart_failed" in (parity.get("error_detail") or "")
    assert kernel_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_real_apply_and_test_returns_parity_failed_when_probe_diverges(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="004", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_good_patch(kernel_path), encoding="utf-8")
    _stub_passing_helpers(monkeypatch, runner)

    def _probe_diverges(self, *, spec, fixture_dir, kernel_target, debug_export_dir):
        return ParityProbeResult(
            pass_=False,
            fixture_id=str(spec.get("parity_fixture_id", "")),
            kernel_target=kernel_target,
            probes_total=2,
            probes_passed=1,
            first_diverging_probe=1,
            tolerance_overshoot=0.0042,
            reason="parity_logit_diverged",
            error_detail="overshoot=4.200000e-03",
            checkpoints_checked=(1, 1024),
        )

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_invoke_parity_probe", _probe_diverges, raising=True
    )

    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="004",
        kernel_target="deltanet",
        harness="real",
        round_root=round_dir.parent,
    )
    assert payload["outcome"] == "parity_failed"
    parity = json.loads((iteration_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert parity["pass"] is False
    assert parity["reason"] == "parity_logit_diverged"
    assert parity["first_diverging_probe"] == 1
    assert parity["tolerance_overshoot"] == pytest.approx(0.0042)
    assert not (iteration_dir / "measurement_trace.json").exists()
    assert kernel_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_real_apply_and_test_treats_capture_failure_as_compile_class(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="005", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_good_patch(kernel_path), encoding="utf-8")
    _stub_passing_helpers(monkeypatch, runner)

    def _probe_capture_failed(self, *, spec, fixture_dir, kernel_target, debug_export_dir):
        return ParityProbeResult(
            pass_=False,
            fixture_id=str(spec.get("parity_fixture_id", "")),
            kernel_target=kernel_target,
            probes_total=2,
            probes_passed=0,
            first_diverging_probe=0,
            tolerance_overshoot=0.0,
            reason="capture_failed",
            error_detail="no debug .pt exports produced",
            checkpoints_checked=(1, 1024),
        )

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_invoke_parity_probe", _probe_capture_failed, raising=True
    )

    payload = runner.apply_and_test(
        round_id=round_id,
        iteration="005",
        kernel_target="deltanet",
        harness="real",
        round_root=round_dir.parent,
    )
    # capture_failed routes to compile_failed (not parity_failed) so the proposer doesn't
    # think a kernel that never ran is a parity divergence to be avoided.
    assert payload["outcome"] == "compile_failed"
    parity = json.loads((iteration_dir / "parity_check.json").read_text(encoding="utf-8"))
    assert parity["reason"] == "capture_failed"
    assert kernel_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_real_apply_and_test_raises_when_kernel_base_snapshot_missing(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="006", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_good_patch(kernel_path), encoding="utf-8")
    # remove the snapshot the round driver was supposed to write
    shutil.rmtree(round_dir / "kernel_base")

    with pytest.raises(RuntimeError, match="kernel_base"):
        runner.apply_and_test(
            round_id=round_id,
            iteration="006",
            kernel_target="deltanet",
            harness="real",
            round_root=round_dir.parent,
        )


def test_restart_serving_runtime_threads_kernel_bindmount_through_model_server(
    tmp_path: Path, monkeypatch
) -> None:
    """The runner constructs extra_volume_mounts so the host kernel file is
    bind-mounted into the vLLM container. Without this the agent's patches on
    host disk would be invisible to the running vLLM process after restart."""
    runner, round_dir, kernel_path, _iteration_dir, _round_id = _seed_round(
        tmp_path, iteration="010", patch_text="placeholder"
    )
    spec_path = round_dir / "round_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec["runtime"]["kernel_container_path"] = (
        "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py"
    )
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    captured: dict[str, Any] = {}

    class _StubServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def stop(self, missing_ok: bool = False) -> None:  # noqa: ARG002
            captured["stop_called"] = True

        def start(self, model_id: str, enable_request_logging: bool = False) -> None:  # noqa: ARG002
            captured["start_model_id"] = model_id

    import lumo_flywheel_serving.model_server as model_server_module

    monkeypatch.setattr(model_server_module, "ModelServer", _StubServer, raising=True)
    runner._restart_serving_runtime(spec=spec)

    mounts = captured.get("extra_volume_mounts") or []
    assert "-v" in mounts
    bindmount_pair_index = mounts.index("-v")
    pair = mounts[bindmount_pair_index + 1]
    assert pair.endswith(
        ":/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_delta_h.py"
    )
    assert pair.startswith(str(kernel_path))


def test_real_apply_and_test_restores_base_bytes_even_on_exception(
    tmp_path: Path, monkeypatch
) -> None:
    runner, round_dir, kernel_path, iteration_dir, round_id = _seed_round(
        tmp_path, iteration="007", patch_text="placeholder"
    )
    (iteration_dir / "mutation.patch").write_text(_good_patch(kernel_path), encoding="utf-8")
    _stub_passing_helpers(monkeypatch, runner)

    def _measure_raises(self, *, spec, iteration, iteration_dir, count):
        # patch already applied at this point — kernel bytes are mutated. Verify
        # the finally-block restores them after this raise.
        assert "ALPHA" in kernel_path.read_text(encoding="utf-8")
        raise RuntimeError("harness blew up mid-measurement")

    monkeypatch.setattr(
        auto_research.L0cKernelMutationRunner, "_run_paired_l0c_measurements", _measure_raises, raising=True
    )

    with pytest.raises(RuntimeError, match="harness blew up"):
        runner.apply_and_test(
            round_id=round_id,
            iteration="007",
            kernel_target="deltanet",
            harness="real",
            round_root=round_dir.parent,
        )

    assert kernel_path.read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"

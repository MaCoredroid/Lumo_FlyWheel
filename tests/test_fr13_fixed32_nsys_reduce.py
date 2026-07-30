from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from lumo_flywheel_serving.inference_proxy import (
    Fixed32DigestLedger,
    fixed32_canonical_task_set_sha256,
    fixed32_task_key_id,
)


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_nsys_reduce",
    REPO / "scripts" / "fr13_fixed32_nsys_reduce.py",
)
assert SPEC is not None
assert SPEC.loader is not None
reducer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reducer)


PROJECTION_CSV = """\
NOTICE: Existing SQLite export found.
Range,Style,Total Proj Time (ns),Range Instances,Total GPU Ops
unrelated.range,Push/Pop,9999,1,1
fr13.fixed32.step,Push/Pop,1700,3,24
fr13.fixed32.sfwd,Push/Pop,1200,3,12
fr13.fixed32.postprocess,Push/Pop,80,3,3
fr13.fixed32.dfwd,Push/Pop,300,3,6
fr13.fixed32.cfwd,Push/Pop,100,3,3
"""

PHASE_KERNEL_CSV = """\
NOTICE: Exporting report data...
NVTX Range,NVTX Inst,Kern Inst,Total Time (ns),Kernel Name
,,29484,145361156305,unscoped_kernel
fr13.fixed32.step,3,15,1190,kernel_a
fr13.fixed32.sfwd,2,4,500,kernel_a
fr13.fixed32.sfwd,1,2,250,kernel_a
fr13.fixed32.sfwd,3,3,400,kernel_b
fr13.fixed32.postprocess,3,3,70,kernel_logits
fr13.fixed32.dfwd,3,6,200,kernel_c
fr13.fixed32.cfwd,3,3,90,kernel_d
"""

OVERALL_KERNEL_CSV = """\
Generating CUDA GPU Kernel Summary...
Time (%),Total Time (ns),Instances,Name
60.0,900,9,kernel_a
26.7,400,3,kernel_b
13.3,200,6,"kernel_c<int, 4>"
"""


def _stats_csv() -> dict[str, str]:
    return {
        "nvtx_gpu_proj_sum": PROJECTION_CSV,
        "nvtx_kern_sum": PHASE_KERNEL_CSV,
        "cuda_gpu_kern_sum": OVERALL_KERNEL_CSV,
    }


def test_parser_skips_notice_prelude_and_preserves_quoted_kernel_name() -> None:
    rows = reducer._parse_stats_csv(
        OVERALL_KERNEL_CSV,
        report_name="cuda_gpu_kern_sum",
        required_columns=("Total Time", "Instances", "Name"),
    )

    assert rows[-1]["name"] == "kernel_c<int, 4>"
    assert rows[-1]["total time"] == "200"


def test_build_summary_is_complete_deterministic_and_path_free() -> None:
    summary = reducer._build_summary(
        report_sha256="a" * 64,
        report_bytes=12345,
        stats_csv=_stats_csv(),
        top=1,
    )

    assert summary["schema"] == "fr13.fixed32.nsys_attribution.v2"
    assert summary["attribution_only"] is True
    assert summary["acceptance_valid"] is False
    assert summary["curated_publishable"] is False
    assert summary["provenance_bound"] is False
    assert summary["raw_profiler_artifacts_publishable"] is False
    assert summary["report"] == {"bytes": 12345, "sha256": "a" * 64}
    assert summary["step_envelope"]["step_projected_gpu_time_ns"] == 1700
    assert summary["step_envelope"]["child_projected_gpu_time_ns"] == 1680
    assert summary["step_envelope"]["residual_projected_gpu_time_ns"] == 20
    assert summary["step_envelope"]["child_instance_delta_from_step"] == {
        "cfwd": 0,
        "dfwd": 0,
        "postprocess": 0,
        "sfwd": 0,
    }
    assert summary["phases"]["sfwd"] == {
        "gpu_ops": 12,
        "nvtx_range": "fr13.fixed32.sfwd",
        "projected_gpu_time_ns": 1200,
        "range_instances": 3,
        "top_kernels": [
            {
                "instances": 6,
                "name": "kernel_a",
                "nvtx_instances": 3,
                "total_time_ns": 750,
            }
        ],
    }
    assert summary["overall_top_kernels"] == [
        {"instances": 9, "name": "kernel_a", "total_time_ns": 900}
    ]
    assert all(
        kernel["name"] != "unscoped_kernel"
        for kernel in summary["step_envelope"]["top_kernels"]
    )

    rendered = json.dumps(summary, sort_keys=True)
    assert "/home/" not in rendered
    assert ".nsys-rep" not in rendered
    assert "astropy__astropy" not in rendered


def test_build_summary_rejects_missing_or_unexpected_fixed32_range() -> None:
    bad_stats = _stats_csv()
    bad_stats["nvtx_gpu_proj_sum"] = PROJECTION_CSV.replace(
        "fr13.fixed32.cfwd", "fr13.fixed32.commit"
    )

    with pytest.raises(
        reducer.ReductionError,
        match="fixed32 NVTX ranges do not match exactly",
    ):
        reducer._build_summary(
            report_sha256="b" * 64,
            report_bytes=1,
            stats_csv=bad_stats,
            top=20,
        )


def test_build_summary_rejects_materially_negative_step_residual() -> None:
    bad_stats = _stats_csv()
    bad_stats["nvtx_gpu_proj_sum"] = PROJECTION_CSV.replace(
        "fr13.fixed32.step,Push/Pop,1700",
        "fr13.fixed32.step,Push/Pop,500",
    )

    with pytest.raises(
        reducer.ReductionError,
        match="child projections exceed the step envelope",
    ):
        reducer._build_summary(
            report_sha256="c" * 64,
            report_bytes=1,
            stats_csv=bad_stats,
            top=20,
        )


def _write_runtime_attestation(logs: Path) -> Path:
    contract = reducer.fixed32_contract
    arctic_files = [
        {
            "path": "arctic_inference/suffix_decoding/cache.py",
            "size": 32,
            "sha256": hashlib.sha256(b"fixed32-nsys-fixture-arctic").hexdigest(),
        }
    ]
    payload = {
        "schema": contract.RUNTIME_SCHEMA,
        "canonical_format": contract.CANONICAL_FORMAT,
        "python": {
            "version": "3.12.3",
            "implementation": "CPython",
        },
        "vllm": {
            "version": contract.VLLM_VERSION,
            "module_path": "/usr/local/lib/python3.12/dist-packages/vllm/__init__.py",
        },
        "forked_fa2": {
            "source": {
                "path": str(contract.CONTAINER_FA2_SOURCE),
                "size": contract.FA2_SIZE,
                "sha256": contract.FA2_SHA256,
            },
            "destination": {
                "path": str(contract.CONTAINER_FA2_DESTINATION),
                "size": contract.FA2_SIZE,
                "sha256": contract.FA2_SHA256,
            },
            "byte_identical": True,
        },
        "arctic": {
            "name": "arctic-inference",
            "version": contract.ARCTIC_VERSION,
            "files": arctic_files,
            "canonical_sha256": hashlib.sha256(
                contract.canonical_bytes(arctic_files)
            ).hexdigest(),
            "cache_class_module": "arctic_inference.suffix_decoding.cache",
            "cache_class_qualname": "SuffixDecodingCache",
            "pinned_source_url": contract.ARCTIC_SDIST_URL,
            "pinned_source_sha256": contract.ARCTIC_SDIST_SHA256,
        },
    }
    payload["overall_canonical_sha256"] = hashlib.sha256(
        contract.canonical_bytes(payload)
    ).hexdigest()
    contract.validate_runtime_attestation(payload)
    path = logs / "fr13_fixed32_runtime_attestation.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _provenance_fixture(tmp_path: Path) -> dict[str, object]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"schema":"fixture"}\n', encoding="utf-8")

    arm_dir = tmp_path / "arm"
    logs = arm_dir / "logs"
    logs.mkdir(parents=True)
    metrics = arm_dir / "metrics_before_swe.txt"
    metrics.write_text("pretask fixture\n", encoding="utf-8")
    ready = arm_dir / "fixed32_ready_ack.json"
    ready.write_text('{"generation":0}\n', encoding="utf-8")
    census = logs / "fr13_fixed32_work_census.jsonl"
    marker = arm_dir / "fixed32_pretask_zero_traffic.json"
    marker.write_text(
        json.dumps(
            {
                "generation_probe_commands_executed": 0,
                "metrics": {
                    "path": str(metrics.resolve()),
                    "sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
                    "spec_drafts": 0,
                    "spec_tokens": 0,
                },
                "mode": "tail6_fixed32",
                "no_positive_probe": True,
                "ready_ack": {
                    "generation": 0,
                    "path": str(ready.resolve()),
                    "sha256": hashlib.sha256(ready.read_bytes()).hexdigest(),
                },
                "schema": "fr13-fixed32-pretask-zero-traffic-v1",
                "work_census": {
                    "bytes": 0,
                    "exists": False,
                    "path": str(census.resolve()),
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = reducer.fixed32_contract
    profile_environment = [
        "FR13_FIXED32_ATTRIBUTION_ONLY=1",
        "FR13_FIXED32_NVTX_PROFILE=1",
        "LUMO_NSYS_WRAP_VLLM=1",
        "PRIVATE_FIXTURE_VALUE=must-not-appear-in-reduced-output",
    ]
    process_identity = arm_dir / "fixed32_process_identity.json"
    process_identity.write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-process-identity-v1",
                "pid1": {
                    "pid": 1,
                    "argv": contract.expected_process_pid1_argv(
                        1,
                        attribution_only=True,
                    ),
                    "environ": profile_environment,
                    "forked_fa2_maps": [],
                },
                "engine_core": {
                    "pid": 321,
                    "argv": ["VLLM::EngineCore"],
                    "environ": profile_environment,
                    "forked_fa2_maps": [
                        "7f000000-7f100000 r-xp 00000000 00:00 0 "
                        f"{contract.CONTAINER_FA2_DESTINATION}"
                    ],
                },
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    container_identity = arm_dir / "fixed32_container_identity.json"
    container_identity.write_text(
        json.dumps(
            {
                "schema": "fr13-fixed32-container-identity-v1",
                "name": f"/fr13-bigdenom-{arm_dir.name}",
                "image_id": contract.IMAGE_ID,
                "configured_image": contract.IMAGE_REFERENCE,
                "platform": contract.IMAGE_OS,
                "running": True,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_attestation = _write_runtime_attestation(logs)

    task_key = fixed32_task_key_id(reducer.EXACT4_TASK_IDS[0])
    task_set = fixed32_canonical_task_set_sha256(reducer.EXACT4_TASK_IDS)
    logical = "1" * 64
    wire = "2" * 64
    engine_request = "3" * 64
    evidence = "4" * 64

    proxy_path = logs / "fr13_fixed32_proxy_ingress.jsonl"
    proxy = Fixed32DigestLedger(proxy_path, role="proxy")
    proxy.append(
        phase="preflight",
        event="campaign_begin",
        outcome="begun",
        evidence_sha256=task_set,
    )
    proxy.append(
        phase="campaign",
        event="logical_begin",
        route="chat",
        task_key_id=task_key,
        logical_id_sha256=logical,
        outcome="accepted",
    )
    proxy.append(
        phase="campaign",
        event="attempt_begin",
        route="chat",
        task_key_id=task_key,
        logical_id_sha256=logical,
        wire_id_sha256=wire,
        engine_request_id_sha256=engine_request,
        outcome="dispatched",
        evidence_sha256=evidence,
    )
    proxy.append(
        phase="campaign",
        event="attempt_result",
        route="chat",
        task_key_id=task_key,
        logical_id_sha256=logical,
        wire_id_sha256=wire,
        engine_request_id_sha256=engine_request,
        status_code=200,
        outcome="response",
        evidence_sha256=evidence,
    )
    proxy.append(
        phase="campaign",
        event="logical_complete",
        route="chat",
        task_key_id=task_key,
        logical_id_sha256=logical,
        outcome="completed",
    )
    proxy.close()

    engine_path = logs / "fr13_fixed32_engine_ingress.jsonl"
    engine = Fixed32DigestLedger(engine_path, role="engine")
    engine.append(
        phase="preflight",
        event="campaign_begin",
        outcome="begun",
        evidence_sha256=task_set,
    )
    engine.append(
        phase="campaign",
        event="request_accepted",
        route="chat",
        task_key_id=task_key,
        wire_id_sha256=wire,
        engine_request_id_sha256=engine_request,
        outcome="accepted",
        evidence_sha256=evidence,
    )
    engine.append(
        phase="campaign",
        event="request_complete",
        route="chat",
        task_key_id=task_key,
        wire_id_sha256=wire,
        engine_request_id_sha256=engine_request,
        outcome="completed",
        evidence_sha256=evidence,
    )
    engine.close()

    nsys = tmp_path / "nsys"
    nsys.write_text(
        "#!/bin/sh\nprintf '%s\\n' 'NVIDIA Nsight Systems version fixture'\n",
        encoding="utf-8",
    )
    nsys.chmod(0o755)
    return {
        "batch_size": 1,
        "concurrency": 1,
        "container_identity": container_identity,
        "driver_rc": 86,
        "engine_ledger": engine_path,
        "external_manifest_end": manifest,
        "external_manifest_launch": manifest,
        "mode": "tail6_fixed32",
        "nsys_bin": nsys,
        "nsys_config_directives": "CuptiUseRawGpuTimestamps=false",
        "nsys_delay_s": 1200,
        "nsys_discard_environment": True,
        "nsys_duration_s": 300,
        "nsys_flush_ms": 100,
        "nsys_trace": "cuda,cuda-sw,nvtx",
        "pretask_zero_traffic": marker,
        "process_identity": process_identity,
        "proxy_ledger": proxy_path,
        "runtime_attestation": runtime_attestation,
        "runtime_manifest_end": manifest,
        "runtime_manifest_launch": manifest,
        "subset": REPO / "config/fr13_fixed32/subset_b4_four.json",
    }


def test_provenance_binds_exact4_b1_and_cross_ledger_completion(
    tmp_path: Path,
) -> None:
    provenance = reducer._build_attribution_provenance(**_provenance_fixture(tmp_path))

    assert provenance["real_swe_verified"] is True
    assert provenance["batch_size"] == 1
    assert provenance["exact4_subset"]["task_count"] == 4
    assert provenance["ingress"]["matched_completed_attempts"] == 1
    assert provenance["nsight"]["discard_environment"] is True
    assert provenance["process_identity"]["pid1_attribution_contract_exact"] is True
    assert (
        provenance["container_identity"]["pinned_container_contract_exact"] is True
    )
    assert (
        provenance["container_identity"]["running_at_identity_capture"] is True
    )
    assert provenance["runtime_attestation"]["pinned_runtime_contract_exact"] is True
    rendered = json.dumps(provenance, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "astropy__astropy" not in rendered
    assert '"argv"' not in rendered
    assert '"environ"' not in rendered
    assert "VLLM::EngineCore" not in rendered
    assert "must-not-appear-in-reduced-output" not in rendered
    assert "/fr13-bigdenom-arm" not in rendered


def test_provenance_rejects_tampered_process_identity(tmp_path: Path) -> None:
    fixture = _provenance_fixture(tmp_path)
    path = Path(fixture["process_identity"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pid1"]["argv"][3] = "1201"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        reducer.ReductionError,
        match="PID1 argv is not the exact attribution contract",
    ):
        reducer._build_attribution_provenance(**fixture)


def test_provenance_rejects_tampered_profile_environment(tmp_path: Path) -> None:
    fixture = _provenance_fixture(tmp_path)
    path = Path(fixture["process_identity"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pid1"]["environ"] = [
        entry
        for entry in payload["pid1"]["environ"]
        if entry != "LUMO_NSYS_WRAP_VLLM=1"
    ]
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        reducer.ReductionError,
        match="PID1 environment is not attribution-only",
    ):
        reducer._build_attribution_provenance(**fixture)


def test_provenance_rejects_tampered_container_identity(tmp_path: Path) -> None:
    fixture = _provenance_fixture(tmp_path)
    path = Path(fixture["container_identity"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["image_id"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        reducer.ReductionError,
        match="container identity does not match the pinned contract",
    ):
        reducer._build_attribution_provenance(**fixture)


def test_provenance_rejects_tampered_runtime_attestation(tmp_path: Path) -> None:
    fixture = _provenance_fixture(tmp_path)
    path = Path(fixture["runtime_attestation"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["vllm"]["version"] = "tampered"
    digest_payload = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    payload["overall_canonical_sha256"] = hashlib.sha256(
        reducer.fixed32_contract.canonical_bytes(digest_payload)
    ).hexdigest()
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        reducer.ReductionError,
        match="runtime attestation does not match the pinned contract",
    ):
        reducer._build_attribution_provenance(**fixture)


@pytest.mark.parametrize(
    "artifact_key",
    ("process_identity", "container_identity", "runtime_attestation"),
)
def test_provenance_requires_runtime_identity_artifacts(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    fixture = _provenance_fixture(tmp_path)
    Path(fixture[artifact_key]).unlink()

    with pytest.raises(reducer.ReductionError, match="is unavailable"):
        reducer._build_attribution_provenance(**fixture)


@pytest.mark.parametrize(
    ("field", "tampered"),
    (("nsys_delay_s", 1201), ("nsys_duration_s", 301)),
)
def test_provenance_rejects_larger_noncanonical_capture_timing(
    tmp_path: Path,
    field: str,
    tampered: int,
) -> None:
    fixture = _provenance_fixture(tmp_path)
    fixture[field] = tampered

    with pytest.raises(
        reducer.ReductionError,
        match="canonical 1200s delay/300s duration",
    ):
        reducer._build_attribution_provenance(**fixture)


def test_main_runs_three_reports_separately_and_writes_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report_content = b"privacy-safe fake report identity"
    report.write_bytes(report_content)
    nsys_bin = tmp_path / "nsys"
    nsys_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    nsys_bin.chmod(0o755)
    output = tmp_path / "attribution.json"
    calls: list[str] = []
    fixtures = _stats_csv()

    def fake_run_stats(
        *,
        nsys_bin: Path,
        report_path: Path,
        report_name: str,
    ) -> str:
        assert nsys_bin == tmp_path / "nsys"
        assert report_path == report
        calls.append(report_name)
        return fixtures[report_name]

    monkeypatch.setattr(reducer, "_run_stats", fake_run_stats)

    assert (
        reducer.main(
            [
                str(report),
                "--output",
                str(output),
                "--nsys-bin",
                str(nsys_bin),
            ]
        )
        == 2
    )
    assert not output.exists()
    calls.clear()

    def fake_provenance(**kwargs: object) -> dict[str, object]:
        assert kwargs["mode"] == "tail6_fixed32"
        assert kwargs["batch_size"] == 1
        assert kwargs["concurrency"] == 1
        assert kwargs["nsys_discard_environment"] is True
        return {
            "real_swe_verified": True,
            "schema": "fr13.fixed32.nsys_attribution_provenance.fixture",
        }

    monkeypatch.setattr(reducer, "_build_attribution_provenance", fake_provenance)
    evidence = str(tmp_path / "evidence")
    assert (
        reducer.main(
            [
                str(report),
                "--output",
                str(output),
                "--nsys-bin",
                str(nsys_bin),
                "--top",
                "2",
                "--subset",
                evidence,
                "--runtime-manifest-launch",
                evidence,
                "--runtime-manifest-end",
                evidence,
                "--external-manifest-launch",
                evidence,
                "--external-manifest-end",
                evidence,
                "--process-identity",
                evidence,
                "--container-identity",
                evidence,
                "--runtime-attestation",
                evidence,
                "--pretask-zero-traffic",
                evidence,
                "--proxy-ledger",
                evidence,
                "--engine-ledger",
                evidence,
                "--mode",
                "tail6_fixed32",
                "--batch-size",
                "1",
                "--concurrency",
                "1",
                "--driver-rc",
                "86",
                "--nsys-delay-s",
                "1200",
                "--nsys-duration-s",
                "300",
                "--nsys-flush-ms",
                "100",
                "--nsys-trace",
                "cuda,cuda-sw,nvtx",
                "--nsys-config-directives",
                "CuptiUseRawGpuTimestamps=false",
                "--nsys-discard-environment",
                "true",
            ]
        )
        == 0
    )
    assert calls == list(reducer.REPORT_NAMES)

    rendered = output.read_text(encoding="utf-8")
    assert rendered.endswith("\n")
    assert rendered == (
        json.dumps(
            json.loads(rendered),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert json.loads(rendered)["report"] == {
        "bytes": len(report_content),
        "sha256": hashlib.sha256(report_content).hexdigest(),
    }
    assert json.loads(rendered)["provenance"]["real_swe_verified"] is True
    assert json.loads(rendered)["provenance_bound"] is True
    assert json.loads(rendered)["curated_publishable"] is True

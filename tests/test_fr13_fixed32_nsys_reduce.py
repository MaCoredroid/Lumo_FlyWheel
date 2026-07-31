from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
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


def test_build_summary_accepts_nsys_default_domain_range_prefix() -> None:
    stats = _stats_csv()
    stats["nvtx_gpu_proj_sum"] = stats["nvtx_gpu_proj_sum"].replace(
        "\nfr13.fixed32.", "\n:fr13.fixed32."
    )
    stats["nvtx_kern_sum"] = stats["nvtx_kern_sum"].replace(
        "\nfr13.fixed32.", "\n:fr13.fixed32."
    )

    summary = reducer._build_summary(
        report_sha256="d" * 64,
        report_bytes=12345,
        stats_csv=stats,
        top=1,
    )

    assert summary["step_envelope"]["range_instances"] == 3
    assert summary["phases"]["cfwd"]["nvtx_range"] == "fr13.fixed32.cfwd"


def test_build_summary_allows_both_partial_capture_boundary_steps() -> None:
    stats = _stats_csv()
    stats["nvtx_gpu_proj_sum"] = stats["nvtx_gpu_proj_sum"].replace(
        "fr13.fixed32.dfwd,Push/Pop,300,3,6",
        "fr13.fixed32.dfwd,Push/Pop,300,5,6",
    )

    summary = reducer._build_summary(
        report_sha256="e" * 64,
        report_bytes=12345,
        stats_csv=stats,
        top=1,
    )

    assert summary["step_envelope"]["child_instance_delta_from_step"]["dfwd"] == 2
    assert summary["step_envelope"]["capture_boundary_allowance_ns"] == 120


def test_build_summary_rejects_more_than_two_boundary_range_instances() -> None:
    stats = _stats_csv()
    stats["nvtx_gpu_proj_sum"] = stats["nvtx_gpu_proj_sum"].replace(
        "fr13.fixed32.dfwd,Push/Pop,300,3,6",
        "fr13.fixed32.dfwd,Push/Pop,300,6,6",
    )

    with pytest.raises(
        reducer.ReductionError,
        match="exceed the two capture boundaries",
    ):
        reducer._build_summary(
            report_sha256="f" * 64,
            report_bytes=12345,
            stats_csv=stats,
            top=1,
        )


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
        "LUMO_NSYS_SESSION_NAME=fr13-fixed32-20260730T120000Z-p1234",
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
    report = logs / "fr13_fixed32_b1_real_swe.nsys-rep"
    report.write_bytes(b"fixture report")

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
        "report": report,
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


def test_provenance_rejects_report_outside_attested_arm(tmp_path: Path) -> None:
    fixture = _provenance_fixture(tmp_path)
    foreign_report = tmp_path / "other-run" / "logs" / (
        "fr13_fixed32_b1_real_swe.nsys-rep"
    )
    foreign_report.parent.mkdir(parents=True)
    foreign_report.write_bytes(b"foreign report")
    fixture["report"] = foreign_report

    with pytest.raises(
        reducer.ReductionError,
        match="not the canonical report inside the provenance arm",
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


@pytest.mark.parametrize(
    "session_name",
    (
        None,
        "profile-1234",
        "fr13-fixed32-20260730T120000-p1234",
        "fr13-fixed32-20260730T120000Z-p0",
    ),
)
def test_provenance_rejects_missing_or_bad_pinned_session_name(
    tmp_path: Path,
    session_name: str | None,
) -> None:
    fixture = _provenance_fixture(tmp_path)
    path = Path(fixture["process_identity"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pid1"]["environ"] = [
        entry
        for entry in payload["pid1"]["environ"]
        if not entry.startswith("LUMO_NSYS_SESSION_NAME=")
    ]
    if session_name is not None:
        payload["pid1"]["environ"].append(
            f"LUMO_NSYS_SESSION_NAME={session_name}"
        )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        reducer.ReductionError,
        match="PID1 environment has no pinned Nsight session name",
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


def _publish_main_args(
    *,
    report: Path,
    output: Path,
    nsys_bin: Path,
    evidence: str,
    expected_report_identity: str,
    expected_report_sha256: str,
) -> list[str]:
    return [
        str(report),
        "--output",
        str(output),
        "--nsys-bin",
        str(nsys_bin),
        "--expected-report-identity",
        expected_report_identity,
        "--expected-report-sha256",
        expected_report_sha256,
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
    expected_report_identity = reducer._shell_report_identity(
        reducer._report_identity(report)
    )
    expected_report_sha256 = hashlib.sha256(report_content).hexdigest()

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
            _publish_main_args(
                report=report,
                output=output,
                nsys_bin=nsys_bin,
                evidence=evidence,
                expected_report_identity=expected_report_identity,
                expected_report_sha256=expected_report_sha256,
            )
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


def test_main_rejects_report_mutation_during_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"original report")
    nsys_bin = tmp_path / "nsys"
    nsys_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    nsys_bin.chmod(0o755)
    output = tmp_path / "attribution.json"
    expected_identity = reducer._shell_report_identity(
        reducer._report_identity(report)
    )
    expected_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()

    monkeypatch.setattr(
        reducer,
        "_run_stats",
        lambda **kwargs: _stats_csv()[str(kwargs["report_name"])],
    )

    def mutating_provenance(**_kwargs: object) -> dict[str, object]:
        report.write_bytes(b"mutated during provenance")
        return {"real_swe_verified": True, "schema": "fixture"}

    monkeypatch.setattr(
        reducer,
        "_build_attribution_provenance",
        mutating_provenance,
    )

    assert (
        reducer.main(
            _publish_main_args(
                report=report,
                output=output,
                nsys_bin=nsys_bin,
                evidence=str(tmp_path / "evidence"),
                expected_report_identity=expected_identity,
                expected_report_sha256=expected_sha256,
            )
        )
        == 2
    )
    assert not output.exists()
    assert "Nsight report changed during reduction" in capsys.readouterr().err


def test_main_rejects_replacement_after_lifecycle_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"same bytes")
    expected_identity = reducer._shell_report_identity(
        reducer._report_identity(report)
    )
    expected_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.nsys-rep"
    replacement.write_bytes(report.read_bytes())
    os.replace(replacement, report)
    nsys_bin = tmp_path / "nsys"
    nsys_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    nsys_bin.chmod(0o755)

    def unexpected_stats(**_kwargs: object) -> str:
        pytest.fail("stats must not run on a report that failed lifecycle binding")

    monkeypatch.setattr(reducer, "_run_stats", unexpected_stats)

    assert (
        reducer.main(
            [
                str(report),
                "--nsys-bin",
                str(nsys_bin),
                "--expected-report-identity",
                expected_identity,
                "--expected-report-sha256",
                expected_sha256,
            ]
        )
        == 2
    )
    assert "does not match the lifecycle-proven report" in capsys.readouterr().err


def test_main_rejects_report_mutation_between_stats_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"original report")
    nsys_bin = tmp_path / "nsys"
    nsys_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    nsys_bin.chmod(0o755)
    calls = 0

    def fake_run_stats(
        *,
        nsys_bin: Path,
        report_path: Path,
        report_name: str,
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            report_path.write_bytes(b"mutated report with a different size")
        return _stats_csv()[report_name]

    monkeypatch.setattr(reducer, "_run_stats", fake_run_stats)

    assert (
        reducer.main(
            [
                str(report),
                "--nsys-bin",
                str(nsys_bin),
            ]
        )
        == 2
    )
    assert calls == len(reducer.REPORT_NAMES)
    assert "Nsight report changed during reduction" in capsys.readouterr().err


def _write_hanging_nsys(
    nsys_bin: Path,
    *,
    parent_pid_path: Path,
    child_pid_path: Path,
) -> None:
    nsys_bin.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$$" > "$PARENT_PID_PATH"
trap '' TERM
bash -c 'trap "" TERM; printf "%s\\n" "$$" > "$CHILD_PID_PATH"; while :; do :; done' &
while :; do :; done
""",
        encoding="utf-8",
    )
    nsys_bin.chmod(0o755)


def _assert_no_live_processes(*pid_paths: Path) -> None:
    for pid_path in pid_paths:
        pid = int(pid_path.read_text(encoding="ascii"))
        for _ in range(100):
            stat_path = Path(f"/proc/{pid}/stat")
            if not stat_path.exists() or stat_path.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"timed-out nsys process remains live: {pid}")


def test_bounded_command_unblocks_guarded_signals_in_child() -> None:
    completed = reducer._run_bounded_command(
        [
            sys.executable,
            "-c",
            (
                "import signal;"
                "blocked=signal.pthread_sigmask(signal.SIG_BLOCK,set());"
                "print(int(signal.SIGINT in blocked or signal.SIGTERM in blocked))"
            ),
        ],
        label="signal-mask fixture",
        timeout_s=1,
        kill_after_s=0.2,
        env=os.environ,
    )

    assert completed.returncode == 0
    assert completed.stdout == "0\n"


def test_run_stats_timeout_kills_term_ignoring_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nsys_bin = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    _write_hanging_nsys(
        nsys_bin,
        parent_pid_path=parent_pid_path,
        child_pid_path=child_pid_path,
    )
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    monkeypatch.setenv("PARENT_PID_PATH", str(parent_pid_path))
    monkeypatch.setenv("CHILD_PID_PATH", str(child_pid_path))

    def reject_numeric_signal(*_args: object, **_kwargs: object) -> None:
        pytest.fail("cleanup must signal through pidfds, not numeric PIDs or PGIDs")

    monkeypatch.setattr(reducer.os, "kill", reject_numeric_signal)
    monkeypatch.setattr(reducer.os, "killpg", reject_numeric_signal)

    started = time.monotonic()
    with pytest.raises(reducer.ReductionError, match="timed out"):
        reducer._run_stats(
            nsys_bin=nsys_bin,
            report_path=report,
            report_name="nvtx_gpu_proj_sum",
            timeout_s=0.1,
            kill_after_s=0.2,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 2
    _assert_no_live_processes(parent_pid_path, child_pid_path)


def test_nsys_version_timeout_kills_term_ignoring_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nsys_bin = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    _write_hanging_nsys(
        nsys_bin,
        parent_pid_path=parent_pid_path,
        child_pid_path=child_pid_path,
    )
    monkeypatch.setenv("PARENT_PID_PATH", str(parent_pid_path))
    monkeypatch.setenv("CHILD_PID_PATH", str(child_pid_path))

    with pytest.raises(reducer.ReductionError, match="nsys --version timed out"):
        reducer._nsys_version(nsys_bin, timeout_s=0.1, kill_after_s=0.2)

    _assert_no_live_processes(parent_pid_path, child_pid_path)


def test_direct_pidfd_failure_kills_unreaped_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nsys_bin = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    escaped_pid_path = tmp_path / "escaped.pid"
    escaped_script = tmp_path / "escaped.py"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    escaped_script.write_text(
        """import os
import signal

if os.fork() != 0:
    os._exit(0)
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(os.environ["ESCAPED_PID_PATH"], "w", encoding="ascii") as target:
    target.write(str(os.getpid()))
while True:
    pass
""",
        encoding="utf-8",
    )
    nsys_bin.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$$" > "$PARENT_PID_PATH"
"$PYTHON_BIN" "$ESCAPED_SCRIPT" &
trap '' TERM
while :; do :; done
""",
        encoding="utf-8",
    )
    nsys_bin.chmod(0o755)
    monkeypatch.setenv("PARENT_PID_PATH", str(parent_pid_path))
    monkeypatch.setenv("ESCAPED_PID_PATH", str(escaped_pid_path))
    monkeypatch.setenv("ESCAPED_SCRIPT", str(escaped_script))
    monkeypatch.setenv("PYTHON_BIN", sys.executable)
    real_pidfd_open = reducer.os.pidfd_open
    real_unbound_cleanup = reducer._terminate_unbound_direct_process
    calls = 0
    cleanup_calls = 0

    def fail_direct_pidfd(pid: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            for _ in range(100):
                if escaped_pid_path.exists():
                    break
                time.sleep(0.01)
            raise PermissionError("injected direct pidfd failure")
        return real_pidfd_open(pid)

    def track_unbound_cleanup(
        process: subprocess.Popen[str],
        *,
        process_token: str,
        timeout_s: float,
    ) -> bool:
        nonlocal cleanup_calls
        cleanup_calls += 1
        return real_unbound_cleanup(
            process,
            process_token=process_token,
            timeout_s=timeout_s,
        )

    monkeypatch.setattr(reducer.os, "pidfd_open", fail_direct_pidfd)
    monkeypatch.setattr(
        reducer,
        "_terminate_unbound_direct_process",
        track_unbound_cleanup,
    )

    with pytest.raises(reducer.ReductionError, match="could not bind a pidfd"):
        reducer._run_stats(
            nsys_bin=nsys_bin,
            report_path=report,
            report_name="nvtx_gpu_proj_sum",
            timeout_s=30,
            kill_after_s=0.2,
        )

    assert cleanup_calls == 1
    _assert_no_live_processes(parent_pid_path, escaped_pid_path)


def test_direct_pidfd_cleanup_exception_uses_token_only_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nsys_bin = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    _write_hanging_nsys(
        nsys_bin,
        parent_pid_path=parent_pid_path,
        child_pid_path=child_pid_path,
    )
    monkeypatch.setenv("PARENT_PID_PATH", str(parent_pid_path))
    monkeypatch.setenv("CHILD_PID_PATH", str(child_pid_path))
    real_pidfd_open = reducer.os.pidfd_open
    calls = 0
    emergency_calls = 0

    def fail_direct_pidfd(pid: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            for _ in range(100):
                if parent_pid_path.exists() and child_pid_path.exists():
                    break
                time.sleep(0.01)
            raise PermissionError("injected direct pidfd failure")
        return real_pidfd_open(pid)

    def fail_emergency_cleanup(
        _process: subprocess.Popen[str],
        *,
        process_token: str,
        timeout_s: float,
    ) -> bool:
        del process_token, timeout_s
        nonlocal emergency_calls
        emergency_calls += 1
        raise RuntimeError("injected emergency cleanup failure")

    monkeypatch.setattr(reducer.os, "pidfd_open", fail_direct_pidfd)
    monkeypatch.setattr(
        reducer,
        "_terminate_unbound_direct_process",
        fail_emergency_cleanup,
    )

    with pytest.raises(
        reducer.ReductionError,
        match="direct pidfd binding failed during emergency cleanup",
    ):
        reducer._run_stats(
            nsys_bin=nsys_bin,
            report_path=report,
            report_name="nvtx_gpu_proj_sum",
            timeout_s=30,
            kill_after_s=0.2,
        )

    assert emergency_calls == 1
    _assert_no_live_processes(parent_pid_path, child_pid_path)


def test_pidfd_acquisition_failure_is_not_reported_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reducer,
        "_open_process_token_pidfds",
        lambda _token: ([], True),
    )

    assert reducer._kill_tracked_processes("fixture", timeout_s=0.01) is False


def test_run_stats_sigterm_kills_detached_process_group(tmp_path: Path) -> None:
    nsys_bin = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    _write_hanging_nsys(
        nsys_bin,
        parent_pid_path=parent_pid_path,
        child_pid_path=child_pid_path,
    )
    harness = """
import runpy
import sys
from pathlib import Path

reducer = runpy.run_path(sys.argv[1])
reducer["_run_stats"](
    nsys_bin=Path(sys.argv[2]),
    report_path=Path(sys.argv[3]),
    report_name="nvtx_gpu_proj_sum",
    timeout_s=30,
    kill_after_s=0.2,
)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            harness,
            str(REPO / "scripts" / "fr13_fixed32_nsys_reduce.py"),
            str(nsys_bin),
            str(report),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PARENT_PID_PATH": str(parent_pid_path),
            "CHILD_PID_PATH": str(child_pid_path),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    for _ in range(200):
        if parent_pid_path.exists() and child_pid_path.exists():
            break
        time.sleep(0.01)
    else:
        process.kill()
        pytest.fail("hanging nsys process group did not start")

    process.terminate()
    process.communicate(timeout=3)

    assert process.returncode == 143
    _assert_no_live_processes(parent_pid_path, child_pid_path)


def test_sigterm_during_direct_pidfd_bind_is_cleanup_safe(tmp_path: Path) -> None:
    nsys_bin = tmp_path / "nsys"
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    cleanup_mask_path = tmp_path / "cleanup-mask.txt"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    _write_hanging_nsys(
        nsys_bin,
        parent_pid_path=parent_pid_path,
        child_pid_path=child_pid_path,
    )
    harness = """
import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("reducer_under_test", sys.argv[1])
assert spec is not None and spec.loader is not None
reducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reducer)
real_pidfd_open = reducer.os.pidfd_open
real_terminate = reducer._terminate_process_group
calls = 0

def injected_pidfd_open(pid):
    global calls
    calls += 1
    pidfd = real_pidfd_open(pid)
    if calls == 2:
        for _ in range(200):
            if Path(os.environ["PARENT_PID_PATH"]).exists() and Path(
                os.environ["CHILD_PID_PATH"]
            ).exists():
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("fixture process group did not start")
        signal.raise_signal(signal.SIGTERM)
    return pidfd

def tracked_terminate(process, *, direct_pidfd, process_token, kill_after_s):
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    Path(os.environ["CLEANUP_MASK_PATH"]).write_text(
        str(int(signal.SIGINT in blocked and signal.SIGTERM in blocked)),
        encoding="ascii",
    )
    return real_terminate(
        process,
        direct_pidfd=direct_pidfd,
        process_token=process_token,
        kill_after_s=kill_after_s,
    )

reducer.os.pidfd_open = injected_pidfd_open
reducer._terminate_process_group = tracked_terminate
reducer._run_stats(
    nsys_bin=Path(sys.argv[2]),
    report_path=Path(sys.argv[3]),
    report_name="nvtx_gpu_proj_sum",
    timeout_s=30,
    kill_after_s=0.2,
)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            harness,
            str(REPO / "scripts" / "fr13_fixed32_nsys_reduce.py"),
            str(nsys_bin),
            str(report),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PARENT_PID_PATH": str(parent_pid_path),
            "CHILD_PID_PATH": str(child_pid_path),
            "CLEANUP_MASK_PATH": str(cleanup_mask_path),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(timeout=5)

    assert (stdout, stderr) == ("", "")
    assert process.returncode == 143
    assert cleanup_mask_path.read_text(encoding="ascii") == "1"
    _assert_no_live_processes(parent_pid_path, child_pid_path)


def test_sigterm_during_post_success_descendant_sweep_retries_cleanup(
    tmp_path: Path,
) -> None:
    nsys_bin = tmp_path / "nsys"
    escaped_script = tmp_path / "escaped.py"
    escaped_pid_path = tmp_path / "escaped.pid"
    sweep_calls_path = tmp_path / "sweep-calls.txt"
    cleanup_mask_path = tmp_path / "cleanup-mask.txt"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    escaped_script.write_text(
        """import os
import signal

if os.fork() != 0:
    os._exit(0)
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
devnull = os.open(os.devnull, os.O_RDWR)
for fd in (0, 1, 2):
    os.dup2(devnull, fd)
if devnull > 2:
    os.close(devnull)
with open(os.environ["ESCAPED_PID_PATH"], "w", encoding="ascii") as target:
    target.write(str(os.getpid()))
while True:
    pass
""",
        encoding="utf-8",
    )
    nsys_bin.write_text(
        """#!/usr/bin/env bash
"$PYTHON_BIN" "$ESCAPED_SCRIPT" &
for _ in $(seq 1 200); do
  test -s "$ESCAPED_PID_PATH" && exit 0
  sleep 0.01
done
exit 2
""",
        encoding="utf-8",
    )
    nsys_bin.chmod(0o755)
    harness = """
import importlib.util
import os
import signal
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("reducer_under_test", sys.argv[1])
assert spec is not None and spec.loader is not None
reducer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reducer)
real_kill = reducer._kill_tracked_processes
calls = 0

def injected_kill(token, *, timeout_s):
    global calls
    calls += 1
    Path(os.environ["SWEEP_CALLS_PATH"]).write_text(str(calls), encoding="ascii")
    if calls == 1:
        signal.raise_signal(signal.SIGTERM)
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    Path(os.environ["CLEANUP_MASK_PATH"]).write_text(
        str(int(signal.SIGINT in blocked and signal.SIGTERM in blocked)),
        encoding="ascii",
    )
    return real_kill(token, timeout_s=timeout_s)

reducer._kill_tracked_processes = injected_kill
reducer._run_stats(
    nsys_bin=Path(sys.argv[2]),
    report_path=Path(sys.argv[3]),
    report_name="nvtx_gpu_proj_sum",
    timeout_s=30,
    kill_after_s=0.2,
)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            harness,
            str(REPO / "scripts" / "fr13_fixed32_nsys_reduce.py"),
            str(nsys_bin),
            str(report),
        ],
        cwd=REPO,
        env={
            **os.environ,
            "PYTHON_BIN": sys.executable,
            "ESCAPED_SCRIPT": str(escaped_script),
            "ESCAPED_PID_PATH": str(escaped_pid_path),
            "SWEEP_CALLS_PATH": str(sweep_calls_path),
            "CLEANUP_MASK_PATH": str(cleanup_mask_path),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(timeout=5)

    assert (stdout, stderr) == ("", "")
    assert process.returncode == 143
    assert sweep_calls_path.read_text(encoding="ascii") == "2"
    assert cleanup_mask_path.read_text(encoding="ascii") == "1"
    _assert_no_live_processes(escaped_pid_path)


def test_post_kill_cleanup_is_bounded_when_escaped_process_holds_pipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    escaped_script = tmp_path / "escaped.py"
    escaped_pid_path = tmp_path / "escaped.pid"
    parent_pid_path = tmp_path / "parent.pid"
    nsys_bin = tmp_path / "nsys"
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")
    escaped_script.write_text(
        """import os
import signal

if os.fork() != 0:
    os._exit(0)
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(os.environ["ESCAPED_PID_PATH"], "w", encoding="ascii") as target:
    target.write(str(os.getpid()))
while True:
    pass
""",
        encoding="utf-8",
    )
    nsys_bin.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$$" > "$PARENT_PID_PATH"
"$PYTHON_BIN" "$ESCAPED_SCRIPT" &
trap '' TERM
while :; do :; done
""",
        encoding="utf-8",
    )
    nsys_bin.chmod(0o755)
    monkeypatch.setenv("PARENT_PID_PATH", str(parent_pid_path))
    monkeypatch.setenv("ESCAPED_PID_PATH", str(escaped_pid_path))
    monkeypatch.setenv("ESCAPED_SCRIPT", str(escaped_script))
    monkeypatch.setenv("PYTHON_BIN", sys.executable)

    started = time.monotonic()
    with pytest.raises(reducer.ReductionError, match="timed out"):
        reducer._run_stats(
            nsys_bin=nsys_bin,
            report_path=report,
            report_name="nvtx_gpu_proj_sum",
            timeout_s=0.2,
            kill_after_s=0.2,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 1.5
    _assert_no_live_processes(parent_pid_path, escaped_pid_path)


def test_reduced_output_must_not_alias_report(tmp_path: Path) -> None:
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(b"report")

    with pytest.raises(reducer.ReductionError, match="must not alias"):
        reducer._validate_output_path(report, report)

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
        "proxy_ledger": proxy_path,
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
    rendered = json.dumps(provenance, sort_keys=True)
    assert str(tmp_path) not in rendered
    assert "astropy__astropy" not in rendered


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

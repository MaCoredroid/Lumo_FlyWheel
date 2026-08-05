from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO
    / "results"
    / "fr13_fixed32_dfwd_k64_tc16x256x64_s2_sm121a_20260805"
)
CHECKER = (
    REPO / "scripts" / "fr13_check_bf16_gemm_k64_tc16x256x64_s2_codegen.py"
)


def _checker():
    spec = importlib.util.spec_from_file_location("fr13_tc_head_checker", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaged_codegen_passes_fail_closed_checker() -> None:
    audit = _checker().audit(
        (ARTIFACT / "candidate.sass").read_text(encoding="ascii"),
        (ARTIFACT / "candidate.resource.txt").read_text(encoding="ascii"),
    )
    packaged = json.loads((ARTIFACT / "codegen_audit.json").read_text())
    assert audit == packaged
    assert audit["status"] == "STATIC_CODEGEN_PASS_UNQUALIFIED"
    assert audit["sass"]["b1_static_instructions"] == 760
    assert audit["sass"]["b4_static_instructions"] == 760


def test_device_codegen_is_reproducible_without_overclaiming_host_elf() -> None:
    reproducibility = json.loads(
        (ARTIFACT / "reproducibility.json").read_text(encoding="ascii")
    )
    assert reproducibility["cubin_byte_identical"] is True
    assert reproducibility["sass_byte_identical"] is True
    assert reproducibility["resource_report_byte_identical"] is True
    assert reproducibility["linked_so_byte_identical"] is False
    assert reproducibility["gpu_runtime_used"] is False


def test_only_alpha_codegen_delta_is_bound_to_packaged_baseline() -> None:
    comparison = json.loads(
        (ARTIFACT / "only_alpha_comparison.json").read_text(encoding="ascii")
    )
    baseline = comparison["generic_epilogue_baseline"]
    candidate = comparison["candidate"]
    assert hashlib.sha256(
        (ARTIFACT / "generic_epilogue_baseline.cubin").read_bytes()
    ).hexdigest() == baseline["cubin_sha256"]
    assert hashlib.sha256((ARTIFACT / "candidate.cubin").read_bytes()).hexdigest() == (
        candidate["cubin_sha256"]
    )
    assert comparison["delta"]["static_instructions_removed_per_kernel"] == 192
    assert comparison["performance_measurement"] is False


def test_manifest_hashes_bind_every_packaged_file() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="ascii"))
    assert manifest["acceptance_valid"] is False
    assert manifest["performance_measurement"] is False
    assert manifest["production_default_enabled"] is False
    assert manifest["runtime_wired"] is False
    for record in manifest["files"]:
        path = REPO / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]

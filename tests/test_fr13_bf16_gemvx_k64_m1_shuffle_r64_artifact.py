from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO / "results" / "fr13_fixed32_dfwd_k64_m1_shuffle_r64_sm121a_20260805"
)
CHECKER = REPO / "scripts" / "fr13_check_bf16_gemvx_k64_m1_shuffle_r64_codegen.py"
R32_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle.cu"
R64_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle_r64.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle_r64.py"
SOURCE_TEST = REPO / "tests" / "test_fr13_bf16_gemvx_k64_m1_shuffle_r64_source.py"
CHECKER_TEST = REPO / "tests" / "test_fr13_bf16_gemvx_k64_m1_shuffle_r64_codegen.py"

SPEC = importlib.util.spec_from_file_location("fr13_r64_artifact_codegen", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, object]:
    return json.loads((ARTIFACT / "manifest.json").read_text(encoding="ascii"))


def test_artifact_is_default_off_unwired_and_unqualified() -> None:
    manifest = _manifest()

    assert manifest["status"] == "static_codegen_pass_default_off_runtime_unwired"
    for key in (
        "acceptance_valid",
        "byte_qualified",
        "timing_eligible",
        "performance_claim",
        "production_default_enabled",
        "runtime_wired",
        "gpu_used",
        "docker_workload_used",
    ):
        assert manifest[key] is False
    assert manifest["not_run"] == [
        "GPU kernel runtime",
        "real SWE-Verified task",
        "byte equality gate",
        "B1 or B4 full-step timing",
        "hardware-floor acceptance",
    ]


def test_artifact_binds_source_binary_object_and_cubin_bytes() -> None:
    manifest = _manifest()
    source = manifest["source"]
    candidate = manifest["candidate"]

    assert source["base_r32_commit"] == "960e99379d6880ed29a61f0acd8861eaa9657c89"
    assert _sha256(R32_CUDA) == source["r32_cuda_sha256"]
    assert _sha256(R64_CUDA) == source["r64_cuda_sha256"]
    assert _sha256(BUILDER) == source["builder_sha256"]
    assert _sha256(CHECKER) == source["checker_sha256"]
    assert _sha256(SOURCE_TEST) == source["source_test_sha256"]
    assert _sha256(CHECKER_TEST) == source["checker_test_sha256"]

    binary = ARTIFACT / "fr13_bf16_gemvx_k64_m1_shuffle_r64.abi3.so"
    obj = ARTIFACT / "fr13_bf16_gemvx_k64_m1_shuffle_r64.cuda.o"
    cubin = ARTIFACT / "fr13_bf16_gemvx_k64_m1_shuffle_r64.sm_121a.cubin"
    assert _sha256(binary) == candidate["binary_sha256"]
    assert binary.stat().st_size == candidate["binary_bytes"]
    assert _sha256(obj) == candidate["object_sha256"]
    assert obj.stat().st_size == candidate["object_bytes"]
    assert _sha256(cubin) == candidate["cubin_sha256"]
    assert cubin.stat().st_size == candidate["cubin_bytes"]


def test_raw_evidence_replays_the_fail_closed_static_checker() -> None:
    manifest = _manifest()
    args = argparse.Namespace(
        binary=ARTIFACT / "fr13_bf16_gemvx_k64_m1_shuffle_r64.abi3.so",
        object=ARTIFACT / "fr13_bf16_gemvx_k64_m1_shuffle_r64.cuda.o",
        cubin=ARTIFACT / "fr13_bf16_gemvx_k64_m1_shuffle_r64.sm_121a.cubin",
        source=R64_CUDA,
        r32_source=R32_CUDA,
        builder=BUILDER,
        source_test=SOURCE_TEST,
        build_attestation=ARTIFACT / "build_attestation.json",
        resource_usage=ARTIFACT / "resource_usage.txt",
        elf_list=ARTIFACT / "elf_list.txt",
        elf_dump=ARTIFACT / "elf_dump.txt",
        sass=ARTIFACT / "sass.txt",
        nvdisasm_version=ARTIFACT / "nvdisasm_version.txt",
        source_branch=manifest["source"]["branch"],
        source_commit=manifest["source"]["source_checkpoint_commit"],
        build_image_digest=manifest["toolchain"]["build_image_digest"],
    )

    observed = MODULE.audit(args)
    recorded = json.loads((ARTIFACT / "static_codegen.json").read_text("ascii"))
    assert observed == recorded
    assert observed["candidate"]["threads_per_cta"] == 1024
    assert observed["resources"]["stack_bytes_per_thread"] == 0
    assert observed["sass"]["cta_barrier"] == 0
    assert observed["sass"]["calls"] == 0


def test_sha256sums_is_complete_and_valid() -> None:
    entries: dict[str, str] = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest

    expected = {
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(entries) == expected
    for name, digest in entries.items():
        assert _sha256(ARTIFACT / name) == digest

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULTS = (
    REPO
    / "results"
    / "fr13_fixed32_verifier_head_m32_n256k32s3_sm121a_build_20260805"
)
CUDA = REPO / "csrc/fr13_bf16_verifier_head_m32_n256k32s3_sm121a.cu"
BUILDER = (
    REPO / "scripts/fr13_build_bf16_verifier_head_m32_n256k32s3_sm121a.py"
)
CURRENT_GATE_SOURCE = REPO / "csrc/fr13_bf16_verifier_head_m32_sm121a.cu"
CURRENT_GATE_LAUNCHER = REPO / "scripts/fr13_launch_forked_fa2_tree_server.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_build_is_source_bound_and_explicitly_unqualified() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="ascii"))
    attestation = json.loads(
        (RESULTS / "build_attestation.json").read_text(encoding="ascii")
    )

    assert manifest["status"] == "STATIC_BUILD_PASS_UNQUALIFIED"
    assert manifest["provenance"]["cuda_source_sha256"] == _sha256(CUDA)
    assert manifest["provenance"]["builder_sha256"] == _sha256(BUILDER)
    assert manifest["artifacts"]["build_attestation_sha256"] == _sha256(
        RESULTS / "build_attestation.json"
    )
    assert attestation["status"] == "BUILT_UNQUALIFIED"
    assert attestation["binary"] == {
        "bytes": 235328,
        "mode": "0555",
        "path": "/home/mark/fr13_bf16_verifier_head_m32_n256k32s3_cea04fef_20260805/fr13_bf16_verifier_head_m32_n256k32s3_sm121a.abi3.so",
        "sha256": "03f5d07a7f4029d7bc4a6a271a3c7e34f433c2f139d9adb374fcd0a80d1b91a7",
    }
    for key in (
        "performance_measurement",
        "performance_claim",
        "byte_equality_claim",
        "verifier_distribution_claim",
        "real_task_correctness",
        "production_default_enabled",
    ):
        assert attestation[key] is False
    assert manifest["eligibility"] == {
        "default_off": True,
        "timing_eligible": False,
        "production_eligible": False,
        "next_gate": "one real SWE-Verified B1 raw-BF16 shadow task with incumbent logits always served",
    }


def test_full_build_cpu_import_and_resources_are_closed() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="ascii"))
    assert (RESULTS / "cpu_import_schema.txt").read_text(encoding="ascii") == (
        "2.11.0+cu130\n"
        "13.0\n"
        "fr13_verifier_head::bf16_m32_n256k32s3_out(Tensor(a!) output, "
        "Tensor hidden, Tensor weight) -> ()\n"
    )
    assert manifest["cpu_import"]["status"] == "PASS"
    assert manifest["cpu_import"]["dynamic_dependencies_resolved"] is True
    assert manifest["cpu_import"]["gpu_tensor_created"] is False
    assert manifest["resources"] == {
        "cubin_count": 1,
        "registers": 128,
        "stack_bytes": 0,
        "static_local_bytes": 0,
        "static_shared_bytes": 1024,
        "dynamic_shared_bytes": 55296,
        "sass_instruction_lines": 1120,
        "sass_hmma_bf16": 32,
        "sass_async_global_to_shared_128b": 27,
        "sass_global_load_128b": 8,
        "sass_global_load_other": 4,
        "sass_global_store_128b": 16,
        "sass_ldl": 0,
        "sass_stl": 0,
        "sass_call": 0,
        "sass_matches_codegen": True,
    }


def test_full_build_does_not_rewire_current_gate_a() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="ascii"))
    preservation = manifest["gate_a_preservation"]
    assert preservation["current_source_sha256"] == _sha256(CURRENT_GATE_SOURCE)
    assert preservation["current_launcher_sha256"] == _sha256(CURRENT_GATE_LAUNCHER)
    assert preservation["source_or_launcher_changed_from_base"] is False
    assert preservation["current_binary_modified"] is False
    assert preservation["candidate_wired_into_gate"] is False
    assert "n256k32s3" not in CURRENT_GATE_LAUNCHER.read_text(encoding="utf-8")


def test_full_build_sha256sums_cover_every_committed_artifact() -> None:
    expected = {}
    for line in (RESULTS / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest
    committed = {
        path.name: _sha256(path)
        for path in RESULTS.iterdir()
        if path.name != "SHA256SUMS"
    }
    assert expected == committed

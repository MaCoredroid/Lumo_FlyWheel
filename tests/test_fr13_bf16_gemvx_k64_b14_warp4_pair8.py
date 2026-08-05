from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_gemvx_k64_b14_warp4_pair8.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_b14_warp4_pair8.py"
CHECKER = REPO / "scripts" / "fr13_check_bf16_gemvx_k64_b14_warp4_pair8_codegen.py"
ARTIFACT = (
    REPO
    / "results"
    / "fr13_fixed32_dfwd_k64_b14_warp4_pair8_sm121a_20260805"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("fr13_b14_pair8_codegen", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_dot(input_values: list[float], weight_values: list[float]) -> float:
    assert len(input_values) == len(weight_values)
    assert len(input_values) % (32 * 8) == 0
    octets = len(input_values) // 8
    partials = [0.0] * 32
    for lane in range(32):
        accumulator = 0.0
        for octet in range(lane, octets, 32):
            start = octet * 8
            for step in range(8):
                index = start + step
                accumulator += input_values[index] * weight_values[index]
        partials[lane] = accumulator
    for stride in (16, 8, 4, 2, 1):
        before = partials.copy()
        for lane in range(stride):
            partials[lane] = before[lane] + before[lane + stride]
    return partials[0]


def test_source_is_fixed32_k64_b1_b4_default_off_pair8() -> None:
    source = CUDA.read_text(encoding="ascii")
    for declaration in (
        "constexpr int kHidden = 5120;",
        "constexpr int kVocab = 65536;",
        "constexpr int kBatch4 = 4;",
        "constexpr int kLanes = 32;",
        "constexpr int kWarpsPerCta = 8;",
        "constexpr int kRowsPerWarp = 4;",
        "constexpr int kThreadsPerCta = kLanes * kWarpsPerCta;",
        "constexpr int kCtas = kVocab / kRowsPerCta;",
        "constexpr int kElementsPerLoad = 8;",
        "static_assert(kThreadsPerCta == 256);",
        "static_assert(kCtas == 2048);",
    ):
        assert declaration in source
    assert source.count("__global__ __launch_bounds__(kThreadsPerCta, 2)") == 2
    assert source.count("<<<kCtas, block, 0, at::cuda::getCurrentCUDAStream()>>>") == 2
    assert "FR13_DEVICE_CODEGEN_ONLY" in source
    assert "PRODUCTION" not in source


def test_packed_lane_partition_covers_k_once_and_rows_once() -> None:
    indices: list[int] = []
    for lane in range(32):
        for octet in range(lane, 640, 32):
            indices.extend(range(octet * 8, octet * 8 + 8))
    assert len(indices) == 5120
    assert sorted(indices) == list(range(5120))

    rows = [
        block * 32 + warp * 4 + row
        for block in range(2048)
        for warp in range(8)
        for row in range(4)
    ]
    assert rows == list(range(65536))


def test_cpu_oracle_covers_independent_b1_b4_projection_rows() -> None:
    hidden = 256
    inputs = [
        [((batch + 1) * ((index % 13) - 6)) / 16.0 for index in range(hidden)]
        for batch in range(4)
    ]
    weights = [
        [((row + 2) * ((index % 11) - 5)) / 32.0 for index in range(hidden)]
        for row in range(7)
    ]
    for batch in range(4):
        for row in range(7):
            reference = sum(
                inputs[batch][index] * weights[row][index]
                for index in range(hidden)
            )
            assert _candidate_dot(inputs[batch], weights[row]) == reference


def test_source_reuses_inputs_and_b4_weights_without_cross_row_alias() -> None:
    source = CUDA.read_text(encoding="ascii")
    m1 = source[
        source.index("fr13_bf16_gemvx_k64_m1_warp4_pair8_kernel") :
        source.index("fr13_bf16_gemvx_k64_m4_warp4_pair8_kernel")
    ]
    m4 = source[
        source.index("fr13_bf16_gemvx_k64_m4_warp4_pair8_kernel") :
        source.index("#if !defined(FR13_DEVICE_CODEGEN_ONLY)", source.index("fr13_bf16_gemvx_k64_m4_warp4_pair8_kernel"))
    ]
    assert m1.count("fr13_unpack_bf16_octet(input_octets[octet])") == 1
    assert m1.count("fr13_fma_octet(accumulator") == 4
    assert m4.count("fr13_unpack_bf16_octet(weight") == 4
    assert m4.count("FR13_ACCUMULATE_REQUEST(") == 5
    assert m4.count("FR13_STORE_REQUEST(") == 5
    for request in range(4):
        for row in range(4):
            assert f"accumulator{request}{row}" in m4
    assert "__syncthreads" not in source
    assert "atomic" not in source.lower()
    assert "__shared__" not in source


def test_ops_are_proposal_only_and_cannot_change_target_authority() -> None:
    source = CUDA.read_text(encoding="ascii")
    schemas = re.findall(r'"(gemvx_m[14]_warp4_pair8_out\(Tensor\(a!\).*?)"', source)
    assert len(schemas) == 2
    for schema in schemas:
        assert schema.endswith("Tensor input, Tensor weight) -> ()")
        assert "target" not in schema
        assert "accept" not in schema
        assert "random" not in schema
    builder = BUILDER.read_text(encoding="ascii")
    assert '"proposal_only": True' in builder
    assert '"target_authority_changed": False' in builder
    assert '"runtime_wired": False' in builder
    assert '"production_default_enabled": False' in builder


def test_builder_is_bound_to_exact_source() -> None:
    builder = BUILDER.read_text(encoding="ascii")
    match = re.search(r'^SOURCE_SHA256 = "([0-9a-f]{64})"$', builder, re.MULTILINE)
    assert match is not None
    assert hashlib.sha256(CUDA.read_bytes()).hexdigest() == match.group(1)
    assert '"gemvx_m1_warp4_pair8_out"' in builder
    assert '"gemvx_m4_warp4_pair8_out"' in builder
    assert '"--frandom-seed=fr13_bf16_k64_b14_warp4_pair8"' in builder


def test_codegen_checker_accepts_pinned_double_compile() -> None:
    first = _load_checker().audit(
        (ARTIFACT / "candidate.sass").read_text(encoding="ascii"),
        (ARTIFACT / "candidate.ptxas.txt").read_text(encoding="ascii"),
    )
    assert first["resources"]["m1_warp4_pair8_kernel"]["registers_per_thread"] == 40
    assert first["resources"]["m4_warp4_pair8_kernel"]["registers_per_thread"] == 80
    assert first["logical_global_load_model"]["b1"]["load_instruction_reduction_fraction"] == 0.921875
    assert first["logical_global_load_model"]["b4"]["load_instruction_reduction_fraction"] == 0.95

    reproducibility = json.loads(
        (ARTIFACT / "reproducibility.json").read_text(encoding="ascii")
    )
    assert reproducibility["cubin_byte_identical"] is True
    assert reproducibility["sass_byte_identical"] is True


def test_codegen_checker_rejects_packed_load_mutation() -> None:
    sass = (ARTIFACT / "candidate.sass").read_text(encoding="ascii")
    assert "LDG.E.128.CONSTANT" in sass
    mutated = sass.replace("LDG.E.128.CONSTANT", "LDG.E.U16.CONSTANT", 1)
    with pytest.raises(RuntimeError):
        _load_checker().audit(
            mutated,
            (ARTIFACT / "candidate.ptxas.txt").read_text(encoding="ascii"),
        )


def test_artifact_hashes_bind_every_packaged_file() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="ascii"))
    assert manifest["acceptance_valid"] is False
    assert manifest["performance_measurement"] is False
    assert manifest["production_default_enabled"] is False
    assert manifest["runtime_wired"] is False
    for record in manifest["files"]:
        path = REPO / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


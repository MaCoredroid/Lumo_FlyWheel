from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
CUDA = REPO / "csrc" / "fr13_bf16_verifier_head_m32_sm121a.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_verifier_head_m32_sm121a.py"
RESULTS = (
    REPO
    / "results"
    / "fr13_fixed32_b1_verifier_head_m32_sm121a_codegen_20260805"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _builder_module():
    spec = importlib.util.spec_from_file_location("fr13_verifier_builder", BUILDER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_source_is_exact_fixed32_full_vocab_no_split_k() -> None:
    source = CUDA.read_text(encoding="utf-8")

    for constant in (
        "constexpr int kRows = 32;",
        "constexpr int kVocab = 248320;",
        "constexpr int kHidden = 5120;",
        "constexpr int kThreadblockM = 32;",
        "constexpr int kThreadblockN = 128;",
        "constexpr int kThreadblockK = 64;",
        "constexpr int kWarpM = 32;",
        "constexpr int kWarpN = 64;",
        "constexpr int kWarpK = 64;",
        "constexpr int kStages = 3;",
    ):
        assert constant in source
    assert "using HiddenLayout = cutlass::layout::RowMajor;" in source
    assert "using WeightLayout = cutlass::layout::ColumnMajor;" in source
    assert "using OutputLayout = cutlass::layout::RowMajor;" in source
    assert "cutlass::arch::OpClassTensorOp" in source
    assert "cutlass::arch::Sm80" in source
    assert "cutlass::gemm::GemmShape<16, 8, 16>" in source
    assert "static_assert(kSharedStorageBytes == 61440);" in source
    assert "{kRows, kVocab, kHidden}" in source
    assert "{1.0f, 0.0f}" in source
    assert "get_workspace_size(arguments) == 0" in source
    assert source.count("      1);") == 1
    assert "split_k" not in source.lower()
    assert "float8" not in source.lower()
    assert "65536" not in source


def test_cuda_source_fails_closed_on_every_runtime_geometry() -> None:
    source = CUDA.read_text(encoding="utf-8")

    assert "output.is_cuda() && hidden.is_cuda() && weight.is_cuda()" in source
    assert "output.device() == hidden.device()" in source
    assert source.count("at::kBFloat16") == 3
    assert "hidden.sizes() == at::IntArrayRef({kRows, kHidden})" in source
    assert "hidden.strides() == at::IntArrayRef({kHidden, 1})" in source
    assert "weight.sizes() == at::IntArrayRef({kVocab, kHidden})" in source
    assert "weight.strides() == at::IntArrayRef({kHidden, 1})" in source
    assert "output.sizes() == at::IntArrayRef({kRows, kVocab})" in source
    assert "output.strides() == at::IntArrayRef({kVocab, 1})" in source
    assert "!output.is_alias_of(hidden) && !output.is_alias_of(weight)" in source
    assert "properties->major == 12" in source
    assert "properties->minor == 1" in source
    assert "can_implement(arguments)" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in source
    assert "bf16_m32_out(Tensor(a!) output" in source


def test_cutlass_views_match_existing_pytorch_storage() -> None:
    vocab = 11
    hidden = 7
    rows = 3

    # CUTLASS A is row-major over PyTorch hidden[rows,K].
    for row in range(rows):
        for k_index in range(hidden):
            pytorch_hidden_offset = row * hidden + k_index
            cutlass_a_offset = row * hidden + k_index
            assert cutlass_a_offset == pytorch_hidden_offset

    # CUTLASS B is [K,vocab] column-major over PyTorch weight[vocab,K].
    for vocab_index in range(vocab):
        for k_index in range(hidden):
            pytorch_weight_offset = vocab_index * hidden + k_index
            cutlass_b_offset = k_index + vocab_index * hidden
            assert cutlass_b_offset == pytorch_weight_offset

    # CUTLASS D is row-major over PyTorch output[rows,vocab].
    for row in range(rows):
        for vocab_index in range(vocab):
            pytorch_output_offset = row * vocab + vocab_index
            cutlass_d_offset = row * vocab + vocab_index
            assert cutlass_d_offset == pytorch_output_offset


def test_builder_is_pinned_and_records_unqualified_contract() -> None:
    module = _builder_module()
    source = BUILDER.read_text(encoding="utf-8")

    assert module.EXPECTED_TORCH == "2.11.0+cu130"
    assert module.EXPECTED_CUDA == "13.0"
    assert module.EXPECTED_ARCH == "12.1a"
    assert (
        module.EXPECTED_CUTLASS_COMMIT
        == "da5e086dab31d63815acafdac9a9c5893b1c69e2"
    )
    assert '"status": "BUILT_UNQUALIFIED"' in source
    assert '"performance_measurement": False' in source
    assert '"byte_equality_claim": False' in source
    assert '"verifier_distribution_claim": False' in source
    assert '"production_default_enabled": False' in source
    assert '"problem_mnk": [32, 248320, 5120]' in source
    assert '"threadblock_mnk": [32, 128, 64]' in source
    assert '"split_k_slices": 1' in source
    assert '"dynamic_shared_storage_bytes": 61440' in source
    assert '"logical_grid_mn": [1, 1940]' in source
    assert '"logical_grid_ctas": 1940' in source
    assert '"vocabulary_reduced": False' in source
    assert "one real SWE-Verified B1 shadow task" in source


def test_builder_rejects_unpinned_cutlass(tmp_path: Path) -> None:
    module = _builder_module()
    root = tmp_path / "cutlass"
    header = root / "include" / "cutlass" / "cutlass.h"
    header.parent.mkdir(parents=True)
    header.write_text("// not pinned\n", encoding="ascii")

    with pytest.raises(Exception):
        module.require_cutlass(root)


def test_builder_requires_clean_cutlass_source() -> None:
    source = BUILDER.read_text(encoding="utf-8")

    assert '"status", "--porcelain=v1", "--untracked-files=all"' in source
    assert "pinned build requires a clean CUTLASS source tree" in source


def test_codegen_artifact_is_fail_closed_and_self_consistent() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="ascii"))

    assert manifest["status"] == "CODEGEN_PASS_UNQUALIFIED"
    for key in (
        "performance_measurement",
        "performance_claim",
        "byte_equality_claim",
        "verifier_distribution_claim",
        "acceptance_valid",
        "gpu_runtime_used",
        "docker_used",
        "synthetic_or_probe_workload_used",
    ):
        assert manifest[key] is False
    assert manifest["codegen"]["status"] == "PASS"
    assert manifest["codegen"]["registers"] == 158
    assert manifest["codegen"]["spill_store_bytes"] == 0
    assert manifest["codegen"]["spill_load_bytes"] == 0
    assert manifest["codegen"]["symbol_threadblock_mnk"] == [32, 128, 64]
    assert manifest["codegen"]["logical_grid_mn"] == [1, 1940]
    assert manifest["codegen"]["logical_grid_ctas"] == 1940
    assert manifest["candidate"]["split_k_slices"] == 1
    assert manifest["candidate"]["full_vocabulary_preserved"] is True
    assert manifest["candidate"]["production_default_enabled"] is False

    for relative, expected in manifest["source_sha256"].items():
        assert _sha256(REPO / relative) == expected

    expected_sums = {}
    for line in (RESULTS / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        expected_sums[name] = digest
    assert expected_sums == {
        "README.md": _sha256(RESULTS / "README.md"),
        "manifest.json": _sha256(RESULTS / "manifest.json"),
        "codegen_resource.txt": _sha256(RESULTS / "codegen_resource.txt"),
    }

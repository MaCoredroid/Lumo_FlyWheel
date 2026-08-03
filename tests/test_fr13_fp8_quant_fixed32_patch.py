from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_patch_fp8_quant_fixed32.py")
    spec = importlib.util.spec_from_file_location("fr13_fp8_quant_patch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(module) -> str:
    return f"""#include <torch/csrc/stable/tensor.h>
{module.INCLUDE_ANCHOR}
{module.KERNEL_ANCHOR}
  const bool is_column_major = output_s.stride(0) < output_s.stride(1);
  const int scale_num_rows = output_s.size(1);
  const int scale_stride = output_s.stride(1);

#define LAUNCH_KERNEL(T, DST_DTYPE)                                        \\
  do {{}} while (0)
#undef LAUNCH_KERNEL
}}
"""


def test_patch_targets_pinned_libtorch_stable_fp8_quant_source() -> None:
    module = _module()
    assert module.TARGET_RELATIVE_PATH == Path(
        "csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu"
    )
    assert module.EXPECTED_UNPATCHED_SHA256 == (
        "f3516d6813c6d231b745558c488e98e80f0f61f07e4cc3509b8e2472dac9694e"
    )


def test_candidate_is_default_off_and_exactly_shape_gated() -> None:
    module = _module()
    patched, changed = module.patch_text(_fixture(module))
    assert changed
    assert 'std::getenv("FR13_FIXED32_B1_FP8_QUANT_REGCACHE")' in patched
    assert 'std::strcmp(value, "byte_ab") == 0' in patched
    assert 'std::strcmp(value, "1") == 0' in patched
    assert "input.scalar_type() == torch::headeronly::ScalarType::BFloat16" in patched
    assert "dst_type == torch::headeronly::ScalarType::Float8_e4m3fn" in patched
    assert "input.size(0) == 32 && input.size(1) == 5120" in patched
    assert "output_q.size(0) == 32" in patched
    assert "output_q.size(1) == 5120" in patched
    assert "group_size == 128" in patched
    assert "num_groups == 1280" in patched
    assert "groups_per_block == 16" in patched
    assert "!scale_ue8m0 && is_column_major" in patched
    assert "output_s.size(0) == 32 && output_s.size(1) == 40" in patched
    assert "output_s.stride(0) == 1 && scale_stride == 32" in patched
    assert "return;" in patched
    assert patched.index("if (fr13_regcache_shape &&") < patched.index(
        "#define LAUNCH_KERNEL"
    )


def test_byte_ab_is_real_task_armed_complete_and_stock_serving() -> None:
    module = _module()
    patched, _ = module.patch_text(_fixture(module))
    assert '"/logs/fr13_fixed32_cutlass_streamk.real_event.arm"' in patched
    assert "fr13.fixed32.b1_fp8_quant_regcache.byte_ab.v1" in patched
    assert "torch::stable::empty_like(output_q)" in patched
    assert "torch::stable::empty_like(output_s)" in patched
    assert "cudaStreamSynchronize(stream)" in patched
    assert r'\"output_byte_equal\"' in patched
    assert r'\"scale_byte_equal\"' in patched
    assert r'\"stock_served\":true' in patched
    assert r'\"comparison_sampled\":false' in patched
    assert "invocation / 128" in patched
    assert "invocation % 128" in patched
    assert "byte_ab_limit" not in patched


def test_candidate_preserves_stock_halfwarp_arithmetic_order() -> None:
    module = _module()
    patched, _ = module.patch_text(_fixture(module))
    candidate_start = patched.index(module.MARKER)
    candidate_end = patched.index(
        "template <typename T, typename DST_DTYPE, bool IS_COLUMN_MAJOR",
        candidate_start,
    )
    candidate = patched[candidate_start:candidate_end]
    assert "kThreadsPerGroup = 16" in candidate
    assert "kGroupsPerBlock = 16" in candidate
    assert "kValuesPerLane = 8" in candidate
    assert "GroupReduceMax(local_absmax) / max_8bit" in candidate
    assert "fmaxf(local_absmax, abs_value)" in candidate
    assert "fminf(fmaxf(value / y_s, min_8bit), max_8bit)" in candidate
    assert "DST_DTYPE(q)" in candidate
    assert "__syncthreads" not in candidate
    assert "extern __shared__" not in candidate
    assert "scale_col * kScaleStride + scale_row" in candidate


def test_fixed_group_to_column_major_scale_mapping_is_bijective() -> None:
    addresses = []
    for group in range(1280):
        row = group // 40
        col = group % 40
        addresses.append(col * 32 + row)
    assert min(addresses) == 0
    assert max(addresses) == 1279
    assert len(set(addresses)) == 1280


def test_patch_is_idempotent_and_detects_incomplete_marker() -> None:
    module = _module()
    patched, changed = module.patch_text(_fixture(module))
    assert changed
    same, changed_again = module.patch_text(patched)
    assert not changed_again
    assert same == patched
    with pytest.raises(module.PatchError, match="marker is incomplete"):
        module.patch_text(module.MARKER + " broken\n")


def test_apply_rejects_unpinned_source(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text("unrelated\n", encoding="ascii")
    with pytest.raises(module.PatchError, match="SHA-256 drift"):
        module.patch_vllm_root(tmp_path)


def test_apply_patches_exact_pinned_source(tmp_path: Path) -> None:
    module = _module()
    upstream = Path(
        "/home/mark/fr13_cutlass_b1_n5120_single_live_ready_20260803/vllm-source"
    ) / module.TARGET_RELATIVE_PATH
    raw = upstream.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == module.EXPECTED_UNPATCHED_SHA256
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)

    result, changed, digest = module.patch_vllm_root(tmp_path)
    assert result == target
    assert changed
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert module.MARKER in target.read_text(encoding="ascii")

    _, changed_again, digest_again = module.patch_vllm_root(tmp_path)
    assert not changed_again
    assert digest_again == digest

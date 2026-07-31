from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_patch_cutlass_fixed32_pdl.py")
    spec = importlib.util.spec_from_file_location("fr13_cutlass_pdl_patch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_fixture(module) -> str:
    return f"""#pragma once

{module.INCLUDE_ANCHOR}
namespace vllm::c3x {{

template <typename GemmKernel>
void cutlass_gemm_caller(
    cute::Shape<int, int, int, int> prob_shape) {{
  GemmOp gemm_op;
  auto workspace = get_workspace();
  auto stream = get_stream();

{module.RUN_ANCHOR}  CUTLASS_CHECK(status);
}}

}}  // namespace vllm::c3x
"""


def test_patch_targets_live_libtorch_stable_caller() -> None:
    module = _module()

    assert module.TARGET_RELATIVE_PATH == Path(
        "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/"
        "cutlass_gemm_caller.cuh"
    )
    assert module.EXPECTED_UNPATCHED_SHA256 == (
        "c3c606d787502fc7cebadd288f386e3913f5ed5539df12236e9bf0bd9d49fb8d"
    )


def test_patch_is_default_off_and_handles_swapped_fixed32_rows() -> None:
    module = _module()
    patched, changed = module.patch_text(_source_fixture(module))

    assert changed
    assert 'std::getenv("FR13_FIXED32_CUTLASS_PDL")' in patched
    assert 'std::strcmp(value, "1") == 0' in patched
    assert "problem_m < problem_n ? problem_m : problem_n" in patched
    for rows in (32, 64, 96, 128):
        assert f"rows == {rows}" in patched
    assert "problem_k >= 5120" in patched
    assert "args, workspace.data_ptr(), stream, nullptr, launch_with_pdl" in patched
    assert patched.count(module.MARKER) == 1


def test_patch_changes_only_launch_plumbing() -> None:
    module = _module()
    source = _source_fixture(module)
    patched, _ = module.patch_text(source)

    reconstructed = patched.replace(
        module.INCLUDE_REPLACEMENT, module.INCLUDE_ANCHOR, 1
    )
    reconstructed = reconstructed.replace(module.PDL_HELPER, module.NAMESPACE_ANCHOR, 1)
    reconstructed = reconstructed.replace(module.RUN_REPLACEMENT, module.RUN_ANCHOR, 1)
    assert reconstructed == source


def test_patch_is_idempotent() -> None:
    module = _module()
    first, changed = module.patch_text(_source_fixture(module))
    second, changed_again = module.patch_text(first)

    assert changed
    assert not changed_again
    assert second == first


def test_patch_fails_closed_on_anchor_drift() -> None:
    module = _module()
    source = _source_fixture(module).replace(
        module.RUN_ANCHOR, "  status = unknown_launch();\n"
    )
    with pytest.raises(RuntimeError, match="CUTLASS run anchor"):
        module.patch_text(source)


def test_source_root_fails_closed_on_pinned_digest_drift(tmp_path: Path) -> None:
    module = _module()
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(_source_fixture(module), encoding="utf-8")

    with pytest.raises(RuntimeError, match="caller SHA256 mismatch"):
        module.patch_source_root(tmp_path)


def test_source_root_applies_exact_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    source = _source_fixture(module)
    target = tmp_path / module.TARGET_RELATIVE_PATH
    target.parent.mkdir(parents=True)
    target.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        module,
        "EXPECTED_UNPATCHED_SHA256",
        hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )

    assert module.patch_source_root(tmp_path)
    assert module.MARKER in target.read_text(encoding="utf-8")
    assert not module.patch_source_root(tmp_path)

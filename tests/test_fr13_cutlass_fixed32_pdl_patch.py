from __future__ import annotations

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


def test_patch_is_default_off_and_fixed32_only() -> None:
    module = _module()
    patched, changed = module.patch_text(_source_fixture(module))

    assert changed
    assert 'std::getenv("FR13_FIXED32_CUTLASS_PDL")' in patched
    assert 'std::strcmp(value, "1") == 0' in patched
    assert "cute::get<0>(prob_shape) == 32" in patched
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

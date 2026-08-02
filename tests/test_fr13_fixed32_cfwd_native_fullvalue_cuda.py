from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumo_flywheel_serving import fr13_cfwd_native_fullvalue_cuda as candidate


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "native/fr13_fixed32_cfwd_native_fullvalue.cu"
PATCHER_SOURCE = ROOT / "scripts/fr13_patch_vllm_cfwd_native_fullvalue_cuda.py"
CODEGEN_CHECKER_SOURCE = (
    ROOT / "scripts/fr13_check_cfwd_native_fullvalue_codegen.py"
)


def _load_patcher():
    spec = importlib.util.spec_from_file_location(
        "cfwd_native_fullvalue_patcher", PATCHER_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_codegen_checker():
    spec = importlib.util.spec_from_file_location(
        "cfwd_native_fullvalue_codegen_checker", CODEGEN_CHECKER_SOURCE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection_kwargs() -> dict[str, object]:
    return {
        "mode": "hydra27_fixed32",
        "batch_size": 4,
        "num_layers": 48,
        "ring_nodes": 32,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "dim_k": 128,
        "dim_v": 128,
        "path_cap": 16,
        "max_accepted_length": 11,
        "bank_dtype": "float32",
        "ring_dtype": "bfloat16",
        "gate_dtype": "float32",
        "index_dtype": "int32",
        "use_qk_l2norm": True,
        "gate_coefficients_precomputed": True,
        "bank_offset_table_prevalidated": True,
        "accepted_values_device_guarded": True,
        "op_available": True,
    }


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_resource_contract_maps_full_state_to_one_cta(batch: int) -> None:
    contract = candidate.resource_contract(batch)
    assert contract["ctas_per_event"] == 48 * batch * 48
    assert contract["ctas_per_layer_request_value_head"] == 1
    assert contract["launches_per_event"] == 1
    assert contract["threads_per_cta"] == 512
    assert contract["warps_per_cta"] == 16
    assert contract["value_rows_per_warp"] == 8
    assert contract["key_columns_per_lane"] == 4
    assert contract["fp32_state_elements_per_thread"] == 32
    assert contract["normalized_k_shared_elements"] == 128
    assert contract["static_shared_bytes_source_model"] == 548
    assert contract["k_hbm_vector_loads_per_cta_step"] == 1
    assert contract["k_norms_per_cta_step"] == 1
    assert contract["state_hbm_traffic_removed"] is False
    assert contract["final_bank_store_dtype"] == "float32"
    assert contract["compile_gate"] == {
        "architecture": "sm_121",
        "registers_per_thread_at_most": 64,
        "minimum_ctas_per_sm_target": 2,
        "stack_frame_bytes": 0,
        "local_bytes": 0,
        "spill_load_bytes": 0,
        "spill_store_bytes": 0,
    }


def test_selector_is_default_off() -> None:
    assert candidate.resolve_candidate(**_selection_kwargs(), environ={}) is None
    assert candidate.resolve_candidate(
        **_selection_kwargs(), environ={candidate.SELECTOR_ENV: "0"}
    ) is None


def test_selector_is_diagnostic_only_and_never_authorizes_production() -> None:
    selection = candidate.resolve_candidate(
        **_selection_kwargs(),
        environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
    )
    assert selection is not None
    assert selection["candidate"] == candidate.CANDIDATE
    assert selection["source_only"] is True
    assert selection["default_off"] is True
    assert selection["production_authorized"] is False
    assert selection["fallback_on_error"] is False
    assert selection["timing_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", None),
        ("batch_size", 8),
        ("num_layers", 47),
        ("ring_nodes", 31),
        ("num_key_heads", 8),
        ("num_value_heads", 32),
        ("dim_k", 64),
        ("dim_v", 64),
        ("path_cap", 32),
        ("max_accepted_length", 12),
        ("bank_dtype", "bfloat16"),
        ("ring_dtype", "float32"),
        ("gate_dtype", "bfloat16"),
        ("index_dtype", "int64"),
        ("use_qk_l2norm", False),
        ("gate_coefficients_precomputed", False),
        ("bank_offset_table_prevalidated", False),
        ("accepted_values_device_guarded", False),
        ("op_available", False),
    ],
)
def test_armed_selector_fails_closed_on_any_contract_drift(
    field: str, value: object
) -> None:
    kwargs = _selection_kwargs()
    kwargs[field] = value
    with pytest.raises(RuntimeError):
        candidate.resolve_candidate(
            **kwargs,
            environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
        )


def test_unknown_selector_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="must be unset"):
        candidate.resolve_candidate(
            **_selection_kwargs(),
            environ={candidate.SELECTOR_ENV: "production"},
        )


def test_byte_gate_requires_real_work_and_full_fp32_bank_bytes() -> None:
    plan = candidate.incumbent_byte_gate_plan()
    assert plan["accepted_lengths_required"] == tuple(range(12))
    assert plan["surfaces"] == ("all_48_fp32_running_bank_rows",)
    assert plan["comparison"] == "raw_bytes"
    assert plan["qualification_work"]["b1"] == "real SWE-Verified task bracket"
    assert "exact4" in plan["qualification_work"]["b4"]
    assert plan["reference_always_served_during_qualification"] is True
    assert plan["pinned_compile_resource_gate_required"] is True
    assert plan["production_credential_emitted"] is False
    assert plan["timing_eligible"] is False


def test_operator_probe_and_launch_are_source_bound() -> None:
    calls: list[tuple[object, ...]] = []

    def op(*args: object) -> None:
        calls.append(args)

    torch_module = SimpleNamespace(
        ops=SimpleNamespace(
            _C=SimpleNamespace(fr13_fixed32_cfwd_native_fullvalue=op)
        )
    )
    selection = candidate.resolve_candidate(
        **_selection_kwargs(),
        environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
    )
    assert selection is not None
    tensors = {
        name: object()
        for name in (
            "bank_anchor",
            "bank_off16",
            "accepted_paths",
            "accepted_lens",
            "spec_state_indices",
            "k_rings",
            "v_rings",
            "a_rings",
            "b_rings",
            "gate_coeffs",
        )
    }
    candidate.launch_candidate(
        torch_module=torch_module,
        selection=selection,
        **tensors,
    )
    assert len(calls) == 1
    assert calls[0][:10] == tuple(tensors.values())
    assert calls[0][-3:] == (4, True, True)


def test_launch_rejects_unqualified_or_non_source_selection() -> None:
    torch_module = SimpleNamespace(
        ops=SimpleNamespace(
            _C=SimpleNamespace(fr13_fixed32_cfwd_native_fullvalue=lambda *_: None)
        )
    )
    selection = candidate.resolve_candidate(
        **_selection_kwargs(),
        environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
    )
    assert selection is not None
    tensors = {
        name: object()
        for name in (
            "bank_anchor",
            "bank_off16",
            "accepted_paths",
            "accepted_lens",
            "spec_state_indices",
            "k_rings",
            "v_rings",
            "a_rings",
            "b_rings",
            "gate_coeffs",
        )
    }
    for field, value in (
        ("candidate", "other"),
        ("production_authorized", True),
        ("timing_eligible", True),
        ("bank_offset_table_prevalidated", False),
        ("accepted_values_device_guarded", False),
    ):
        bad = dict(selection)
        bad[field] = value
        with pytest.raises(RuntimeError):
            candidate.launch_candidate(
                torch_module=torch_module,
                selection=bad,
                **tensors,
            )


def test_cuda_source_has_one_shared_k_load_and_cta_norm_per_step() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "constexpr int kWarpsPerBlock = 16;" in source
    assert "static_assert(kThreadsPerBlock == 512);" in source
    assert "static_assert(kStateElementsPerThread == 32);" in source
    assert "static_assert(kSharedBytes == 548);" in source
    assert "__launch_bounds__(kThreadsPerBlock, 2)" in source
    assert "float state[kValuesPerWarp][kKeyQuads];" in source
    assert "__shared__ float normalized_k[kDimK];" in source
    assert "if (thread_id < kDimK)" in source
    assert source.count("const float key_value = load_bf16(k_rings + key_offset)") == 1
    assert source.count("norm = triton_butterfly_four_sum(norm);") == 1
    assert "__fmul_rn(normalized_k[thread_id], norm_partials[0])" in source
    assert "All 16 warps consume the same normalized K vector" in source
    assert source.count("__syncthreads();") >= 6


def test_cuda_source_preserves_ordered_active_recurrence_and_fp32_store() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    loop = source[source.index("for (int step = 0;") :]
    assert "shared_steps = min(max(accepted, 0) + 1" in source
    assert "step - 1" in loop
    assert "__fmul_rn(state[value_lane][key_quad], decay_scale)" in loop
    assert "float partial02 = triton_butterfly_product_sum(" in loop
    assert "float partial13 = triton_butterfly_product_sum(" in loop
    assert "float state_k = __fadd_rn(partial02, partial13);" in loop
    assert "__fmul_rn(__fsub_rn(value, state_k), beta)" in loop
    assert "state[value_lane][key_quad] = __fmaf_rn(" in loop
    assert loop.index("decay_scale);") < loop.index("float partial02")
    assert loop.index("float partial02") < loop.index("const float residual")
    assert loop.index("const float residual") < loop.index("= __fmaf_rn(")
    assert "float* bank_anchor" in source
    assert "state_bank[state_offset] = state[value_lane][key_quad];" in source
    assert "__float2bfloat16" not in source
    assert "requires FP32 state banks" in source


def test_cuda_source_matches_incumbent_initial_zero_accumulate() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "including its -0.0 to +0.0 normalization" in source
    assert "load_state_bank[state_offset] + 0.0f;" in source


def test_cuda_source_pins_incumbent_ptx_math_contract() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "constexpr float kLog2E = 0x1.715476p+0f;" in source
    assert 'asm volatile("ex2.approx.f32 %0, %1;"' in source
    assert 'asm volatile("rsqrt.approx.ftz.f32 %0, %1;"' in source
    assert 'asm volatile("div.full.f32 %0, %1, %2;"' in source
    assert "expf(" not in source
    assert "rsqrtf(" not in source


def test_cuda_source_matches_incumbent_reduction_and_fma_order() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    helper = source[
        source.index("float triton_butterfly_product_sum"):
        source.index("float triton_butterfly_four_sum")
    ]
    assert "const float product = __fmul_rn(lhs, rhs);" in helper
    assert "__shfl_xor_sync(kFullWarpMask, product, 16)" in helper
    assert "float value = __fmaf_rn(lhs, rhs, partner);" in helper
    assert "for (int mask = 8; mask > 0; mask >>= 1)" in helper
    assert "__fadd_rn(" in helper
    loop = source[source.index("for (int step = 0;") :]
    assert "(group 0 + group 2) + (group 1 + group 3)" in loop
    assert loop.index("state[value_lane][0]") < loop.index(
        "state[value_lane][2]"
    )
    assert loop.index("state[value_lane][2]") < loop.index(
        "state[value_lane][1]"
    )
    assert loop.index("state[value_lane][1]") < loop.index(
        "state[value_lane][3]"
    )


def _codegen_fixture(checker) -> tuple[str, str]:
    resource_report = f"""
arch = sm_121a
Resource usage:
 Function mangled_{checker.KERNEL_MARKER}_symbol:
  REG:64 STACK:0 SHARED:1572 LOCAL:0 CONSTANT[0]:1084
"""
    sass = "\n".join(
        opcode
        for opcode, count in checker.EXPECTED_SASS_COUNTS.items()
        for _ in range(count)
    )
    return resource_report, sass


def test_codegen_checker_accepts_pinned_sm121_sass_contract() -> None:
    checker = _load_codegen_checker()
    resource_report, sass = _codegen_fixture(checker)
    receipt = checker.check_codegen(resource_report, sass)
    assert receipt["contract_pass"] is True
    assert receipt["resources"]["registers_per_thread"] == 64
    assert receipt["forbidden_sass_counts"] == {
        "LDL": 0,
        "STL": 0,
        "CALL": 0,
    }
    assert receipt["sass_counts"] == checker.EXPECTED_SASS_COUNTS


@pytest.mark.parametrize("forbidden", ["LDL.64", "STL.64", "CALL"])
def test_codegen_checker_rejects_local_or_call_drift(forbidden: str) -> None:
    checker = _load_codegen_checker()
    resource_report, sass = _codegen_fixture(checker)
    with pytest.raises(RuntimeError, match="local/call drift"):
        checker.check_codegen(resource_report, f"{sass}\n{forbidden}")


def test_codegen_checker_rejects_math_shape_drift() -> None:
    checker = _load_codegen_checker()
    resource_report, sass = _codegen_fixture(checker)
    with pytest.raises(RuntimeError, match="SASS shape drift"):
        checker.check_codegen(resource_report, sass.replace("MUFU.EX2", "NOP", 1))


def test_cuda_extension_contract_is_sm121_and_strict() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "properties->major == 12 && properties->minor == 1" in source
    assert "properties->maxThreadsPerBlock >= kThreadsPerBlock" in source
    assert "properties->sharedMemPerBlock >= kSharedBytes" in source
    assert "bank_offset_table_prevalidated" in source
    assert "accepted_paths.scalar_type() == torch::kInt32" in source
    assert "k_rings.scalar_type() == torch::kBFloat16" in source
    assert "gate_coeffs.scalar_type() == torch::kFloat32" in source


def test_patcher_adds_one_source_to_pinned_extension_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = _load_patcher()
    fixture = {
        Path("CMakeLists.txt"): patcher.CMAKE_ANCHOR,
        Path("csrc/ops.h"): patcher.OPS_ANCHOR,
        Path("csrc/torch_bindings.cpp"): patcher.BINDINGS_ANCHOR,
    }
    for relative, text in fixture.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        patcher,
        "PINNED_SHA256",
        {
            relative: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for relative, text in fixture.items()
        },
    )
    assert patcher.patch_source_root(tmp_path, CUDA_SOURCE) is True
    assert patcher.patch_source_root(tmp_path, CUDA_SOURCE) is False
    cmake = (tmp_path / "CMakeLists.txt").read_text(encoding="utf-8")
    ops = (tmp_path / "csrc/ops.h").read_text(encoding="utf-8")
    bindings = (tmp_path / "csrc/torch_bindings.cpp").read_text(
        encoding="utf-8"
    )
    assert cmake.count("csrc/fr13_fixed32_cfwd_native_fullvalue.cu") == 1
    assert "void fr13_fixed32_cfwd_native_fullvalue(" in ops
    assert bindings.count(
        'ops.impl("fr13_fixed32_cfwd_native_fullvalue"'
    ) == 1
    assert (
        tmp_path / "csrc/fr13_fixed32_cfwd_native_fullvalue.cu"
    ).read_bytes() == CUDA_SOURCE.read_bytes()


def test_patcher_rejects_partial_install(tmp_path: Path) -> None:
    patcher = _load_patcher()
    for relative in patcher.PINNED_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("anchor\n", encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text(
        f"# {patcher.MARKER}\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="partial"):
        patcher.patch_source_root(tmp_path, CUDA_SOURCE)

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from lumo_flywheel_serving import fr13_cfwd_native_fullvalue_cuda as candidate


ROOT = Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "native/fr13_fixed32_cfwd_native_fullvalue.cu"
PATCHER_SOURCE = ROOT / "scripts/fr13_patch_vllm_cfwd_native_fullvalue_cuda.py"
CODEGEN_CHECKER_SOURCE = (
    ROOT / "scripts/fr13_check_cfwd_native_keygroup_codegen.py"
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
        "cfwd_native_keygroup_codegen_checker", CODEGEN_CHECKER_SOURCE
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
        "binary_binding": _binary_binding(),
    }


def _binary_binding() -> dict[str, object]:
    return {
        "schema": candidate.BINARY_BINDING_SCHEMA,
        "candidate": candidate.CANDIDATE,
        "vllm_base_commit": candidate.VLLM_COMMIT,
        "operator": candidate.OPERATOR,
        "architecture": "sm_121a",
        "source_sha256": {
            candidate.CUDA_SOURCE_PATH: candidate.CUDA_SOURCE_SHA256,
            candidate.PATCHER_SOURCE_PATH: candidate.PATCHER_SOURCE_SHA256,
        },
        "patched_vllm_sha256": dict(candidate.PATCHED_VLLM_SHA256),
        "build": {
            "generator": "ninja",
            "candidate_source_in_build_graph": True,
            "candidate_source_forced_rebuild": True,
            "candidate_source_mtime_ns": 1,
            "candidate_object_outputs": ["CMakeFiles/candidate.cu.o"],
            "candidate_objects": [
                {
                    "path": "CMakeFiles/candidate.cu.o",
                    "sha256": "d" * 64,
                    "bytes": 123,
                    "mtime_ns": 2,
                }
            ],
            "full_vllm_extension_target": "_C.abi3.so",
            "full_extension_mtime_ns": 2,
            "cmake_cache_sha256": "b" * 64,
            "build_ninja_sha256": "c" * 64,
        },
        "binary": {"sha256": "a" * 64, "bytes": 123_456},
        "default_on": False,
        "production_authorized": False,
        "timing_eligible": False,
    }


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_resource_contract_maps_each_key_group_to_one_cta(batch: int) -> None:
    contract = candidate.resource_contract(batch)
    assert contract["ctas_per_event"] == 48 * batch * 16
    assert contract["ctas_per_layer_request_key_head"] == 1
    assert contract["value_heads_per_cta"] == 3
    assert contract["value_heads_processed_sequentially"] is True
    assert contract["launches_per_event"] == 2
    assert contract["event_gate_scalar_precompute_launches"] == 1
    assert contract["native_recurrence_launches"] == 1
    assert contract["threads_per_cta"] == 512
    assert contract["warps_per_cta"] == 16
    assert contract["value_rows_per_warp_per_head"] == 8
    assert contract["key_columns_per_lane"] == 4
    assert contract["fp32_state_elements_per_thread"] == 32
    assert contract["fp32_register_state_elements_per_thread"] == 32
    assert contract["fp32_shared_state_elements_per_thread"] == 0
    assert contract["precomputed_step_capacity"] == 12
    assert contract["precomputed_steps_per_wave"] == 4
    assert contract["precompute_waves"] == 3
    assert contract["normalized_k_shared_elements"] == 12 * 128
    assert contract["norm_partial_shared_elements"] == 16
    assert contract["inverse_norm_shared_elements"] == 4
    assert contract["precomputed_node_shared_elements"] == 12
    assert contract["precomputed_gate_scalar_shared_elements"] == 72
    assert contract["event_gate_scalar_elements"] == 48 * batch * 12 * 48 * 2
    assert contract["event_gate_scalar_math"] == "triton_incumbent_lowering"
    assert contract["native_gate_transcendentals"] == 0
    assert contract["static_shared_bytes_source_model"] == 6_568
    assert contract["k_hbm_vector_loads_per_cta_step"] == 1
    assert contract["k_norms_per_cta_step"] == 1
    assert contract["duplicate_value_head_k_loads_per_key_head_step"] == 0
    assert contract["persistent_shared_state_elements"] == 0
    assert contract["state_hbm_traffic_removed"] is False
    assert contract["final_bank_store_dtype"] == "float32"
    assert contract["compile_gate"] == {
        "architecture": "sm_121a",
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
    assert selection["source_bound"] is True
    assert selection["source_sha256"] == candidate.CUDA_SOURCE_SHA256
    assert selection["binary_sha256"] == "a" * 64


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
        ("binary_binding", None),
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


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "wrong"),
        (("candidate",), "wrong"),
        (("vllm_base_commit",), "0" * 40),
        (("operator",), "_C::wrong"),
        (("architecture",), "sm_120"),
        (("source_sha256", candidate.CUDA_SOURCE_PATH), "0" * 64),
        (("patched_vllm_sha256", "CMakeLists.txt"), "0" * 64),
        (("build", "generator"), "make"),
        (("build", "candidate_source_in_build_graph"), False),
        (("build", "candidate_source_forced_rebuild"), False),
        (("build", "candidate_object_outputs"), ["/tmp/candidate.o"]),
        (("build", "candidate_source_mtime_ns"), 3),
        (("build", "full_vllm_extension_target"), "_C_stable_libtorch.abi3.so"),
        (("build", "cmake_cache_sha256"), "bad"),
        (("binary", "sha256"), "bad"),
        (("binary", "bytes"), 0),
        (("default_on",), True),
        (("production_authorized",), True),
        (("timing_eligible",), True),
    ],
)
def test_binary_binding_fails_closed_on_identity_or_scope_drift(
    path: tuple[str, ...], value: object
) -> None:
    binding = _binary_binding()
    target = binding
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value  # type: ignore[index]
    kwargs = _selection_kwargs()
    kwargs["binary_binding"] = binding
    with pytest.raises(RuntimeError):
        candidate.resolve_candidate(
            **kwargs,
            environ={candidate.SELECTOR_ENV: candidate.SELECTOR_VALUE},
        )


def test_binary_binding_loader_requires_private_regular_file(
    tmp_path: Path,
) -> None:
    binding_path = tmp_path / "binding.json"
    binding_path.write_text(
        json.dumps(_binary_binding()), encoding="ascii"
    )
    binding_path.chmod(0o400)
    loaded = candidate.load_binary_binding(binding_path)
    assert loaded["binary"] == {"sha256": "a" * 64, "bytes": 123_456}

    binding_path.chmod(0o600)
    with pytest.raises(RuntimeError, match="private read-only"):
        candidate.load_binary_binding(binding_path)


def test_byte_gate_requires_real_work_and_full_fp32_bank_bytes() -> None:
    plan = candidate.incumbent_byte_gate_plan()
    assert plan["accepted_lengths_required"] == tuple(range(12))
    assert plan["surfaces"] == ("all_48_fp32_running_bank_rows",)
    assert plan["comparison"] == "raw_bytes"
    assert plan["qualification_work"]["b1"] == "real SWE-Verified task bracket"
    assert "exact4" in plan["qualification_work"]["b4"]
    assert plan["reference_always_served_during_qualification"] is True
    assert plan["source_binding_required"] == {
        "schema": candidate.BINARY_BINDING_SCHEMA,
        "vllm_base_commit": candidate.VLLM_COMMIT,
        "cuda_source_sha256": candidate.CUDA_SOURCE_SHA256,
        "patcher_source_sha256": candidate.PATCHER_SOURCE_SHA256,
    }
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
            "event_gate_scalars",
        )
    }
    candidate.launch_candidate(
        torch_module=torch_module,
        selection=selection,
        **tensors,
    )
    assert len(calls) == 1
    assert calls[0][:8] == tuple(tensors.values())
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
            "event_gate_scalars",
        )
    }
    for field, value in (
        ("candidate", "other"),
        ("production_authorized", True),
        ("timing_eligible", True),
        ("source_bound", False),
        ("source_sha256", "0" * 64),
        ("binary_sha256", "bad"),
        ("vllm_base_commit", "0" * 40),
        ("operator", "_C::wrong"),
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


def test_cuda_source_precomputes_four_steps_per_wave_without_shared_state() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "constexpr int kPrecomputedSteps = kMaxAcceptedLength + 1;" in source
    assert "constexpr int kWarpsPerBlock = 16;" in source
    assert "static_assert(kThreadsPerBlock == 512);" in source
    assert "static_assert(kValuesPerWarp == 8);" in source
    assert "static_assert(kStateElementsPerThread == 32);" in source
    assert "static_assert(kStepsPerWave == 4);" in source
    assert "static_assert(kPrecomputeWaves == 3);" in source
    assert "static_assert(kSharedBytes == 6568);" in source
    assert "__launch_bounds__(kThreadsPerBlock, 2)" in source
    assert "float state[kValuesPerWarp][kKeyQuads];" in source
    assert "__shared__ float shared_state" not in source
    assert "const int key_head = blockIdx.x % kKeyHeads;" in source
    assert "__shared__ float normalized_ks[kPrecomputedSteps][kDimK];" in source
    assert "__shared__ float norm_partials[kStepsPerWave]" in source
    assert "__shared__ float inverse_norms[kStepsPerWave];" in source
    assert "__shared__ float shared_recurrence_scalars[kPrecomputedSteps]" in source
    assert "__shared__ int32_t shared_nodes[kPrecomputedSteps];" in source
    assert "const int gate_task_count = steps * kHeadGroup;" in source
    assert "const int step_slot = warp / kNormPartialWarps;" in source
    assert "const int norm_warp = warp % kNormPartialWarps;" in source
    assert "for (int wave = 0; wave < kPrecomputeWaves; ++wave)" in source
    assert "const bool active_step = step < steps;" in source
    assert source.count("const float key_value = load_bf16(k_rings + key_offset)") == 1
    assert source.count("norm = triton_butterfly_four_sum(norm);") == 1
    assert "normalized_ks[step][key_index] = __fmul_rn(" in source
    assert "#pragma unroll 1" in source
    assert "for (int local_value_head = 0; local_value_head < kHeadGroup;" in source
    assert "kLayers * batch_size * kKeyHeads" in source
    assert source.count("__syncthreads();") == 5
    wave_loop = source[
        source.index("for (int wave = 0; wave < kPrecomputeWaves; ++wave)") :
        source.index("// Publish every immutable normalized K row")
    ]
    assert wave_loop.count("__syncthreads();") == 2
    assert "inverse_norms[step_slot]" in wave_loop


def test_cuda_source_preserves_ordered_active_recurrence_and_fp32_store() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    loop = source[source.index("// Process one value head at a time") :]
    assert "shared_steps = min(max(accepted, 0) + 1" in source
    assert "float state[kValuesPerWarp][kKeyQuads];" in loop
    assert "for (int step = 0; step < steps; ++step)" in loop
    assert "__fmul_rn(state[value_lane][key_quad], decay_scale)" in loop
    assert loop.count("float partial02 = triton_butterfly_product_sum(") == 1
    assert loop.count("float partial13 = triton_butterfly_product_sum(") == 1
    assert loop.count("float state_k = __fadd_rn(partial02, partial13);") == 1
    assert loop.count("__fmul_rn(__fsub_rn(value, state_k), beta)") == 1
    assert "state[value_lane][key_quad] = __fmaf_rn(" in loop
    assert loop.index("decay_scale);") < loop.index("float partial02")
    assert loop.index("float partial02") < loop.index("const float residual")
    assert loop.index("const float residual") < loop.index("= __fmaf_rn(")
    assert loop.index("state[value_lane][0]") < loop.index(
        "state[value_lane][2]"
    )
    assert loop.index("state[value_lane][2]") < loop.index(
        "state[value_lane][1]"
    )
    assert loop.index("state[value_lane][1]") < loop.index(
        "state[value_lane][3]"
    )
    assert "float* bank_anchor" in source
    assert "state_bank[state_offset] = state[value_lane][key_quad];" in source
    assert "__float2bfloat16" not in source
    assert "requires FP32 state banks" in source
    assert source.count("load_state_bank[state_offset] + 0.0f;") == 1
    assert 'asm volatile("mov.u32 %0, %%tid.x;"' in source


def test_cuda_source_pins_incumbent_reduction_order_without_gate_math() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert 'asm volatile("rsqrt.approx.ftz.f32 %0, %1;"' in source
    assert "expf(" not in source
    assert "rsqrtf(" not in source
    assert "logf(" not in source
    assert "softplus(" not in source
    assert "sigmoid(" not in source
    helper = source[
        source.index("float triton_butterfly_product_sum"):
        source.index("float triton_butterfly_four_sum")
    ]
    assert "const float product = __fmul_rn(lhs, rhs);" in helper
    assert "__shfl_xor_sync(kFullWarpMask, product, 16)" in helper
    assert "float value = __fmaf_rn(lhs, rhs, partner);" in helper
    assert "for (int mask = 8; mask > 0; mask >>= 1)" in helper
    assert "__fadd_rn(" in helper
    assert "step_k[key_quad] = normalized_ks[step][lane + key_quad * 32];" in source
    assert source.index("state[value_lane][0], step_k[0]") < source.index(
        "state[value_lane][2], step_k[2]"
    )
    assert source.index("state[value_lane][2], step_k[2]") < source.index(
        "state[value_lane][1], step_k[1]"
    )
    assert source.index("state[value_lane][1], step_k[1]") < source.index(
        "state[value_lane][3], step_k[3]"
    )


def test_cuda_source_loads_root_nodes_and_precomputed_three_head_gates() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    precompute = source[
        source.index("const int gate_task_count"):
        source.index("// Process one value head at a time")
    ]
    assert "const int step = thread_id / kHeadGroup;" in precompute
    assert "const int local_value_head = thread_id % kHeadGroup;" in precompute
    assert "if (step > 0)" in precompute
    assert "step - 1" in precompute
    assert "shared_nodes[step] = node;" in precompute
    assert "shared_recurrence_scalars[step][local_value_head][0]" in precompute
    assert "shared_recurrence_scalars[step][local_value_head][1]" in precompute
    assert "event_gate_scalars[gate_offset]" in precompute
    assert "event_gate_scalars[gate_offset + 1]" in precompute
    assert precompute.index("shared_recurrence_scalars") < precompute.index(
        "for (int wave = 0;"
    )


def test_cuda_source_checks_event_gate_scalar_contract() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "event_gate_scalars.scalar_type() == torch::kFloat32" in source
    assert "event_gate_scalars.size(2) == kPrecomputedSteps" in source
    assert "event_gate_scalars.size(3) == kValueHeads" in source
    assert "event_gate_scalars.size(4) == 2" in source
    assert "a_rings" not in source
    assert "b_rings" not in source


def _codegen_fixture(checker) -> tuple[str, str]:
    resource_report = f"""
arch = sm_121a
Resource usage:
 Function mangled_{checker.KERNEL_MARKER}_symbol:
  REG:64 STACK:0 SHARED:7592 LOCAL:0 CONSTANT[0]:1084
"""
    sass = "\n".join(
        opcode
        for opcode, count in checker.EXPECTED_SASS_COUNTS.items()
        for _ in range(count)
    )
    return resource_report, sass


def test_codegen_checker_accepts_pinned_sm121_precompute_contract() -> None:
    checker = _load_codegen_checker()
    resource_report, sass = _codegen_fixture(checker)
    receipt = checker.check_codegen(resource_report, sass)
    assert receipt["contract_pass"] is True
    assert receipt["resources"] == checker.EXPECTED_RESOURCES
    assert receipt["forbidden_sass_counts"] == {
        "LDL": 0,
        "STL": 0,
        "CALL": 0,
    }


@pytest.mark.parametrize("forbidden", ["LDL.64", "STL.64", "CALL"])
def test_codegen_checker_rejects_local_or_call_drift(forbidden: str) -> None:
    checker = _load_codegen_checker()
    resource_report, sass = _codegen_fixture(checker)
    with pytest.raises(RuntimeError, match="local/call drift"):
        checker.check_codegen(resource_report, f"{sass}\n{forbidden}")


def test_codegen_checker_rejects_resource_or_math_drift() -> None:
    checker = _load_codegen_checker()
    resource_report, sass = _codegen_fixture(checker)
    with pytest.raises(RuntimeError, match="resource drift"):
        checker.check_codegen(resource_report.replace("REG:64", "REG:63"), sass)
    with pytest.raises(RuntimeError, match="SASS shape drift"):
        checker.check_codegen(resource_report, f"{sass}\nMUFU.EX2")


def test_codegen_checker_cli_binds_source_object_command_and_toolchain() -> None:
    source = CODEGEN_CHECKER_SOURCE.read_text(encoding="utf-8")
    assert 'parser.add_argument("--compile-command", required=True)' in source
    assert '"object_sha256": _sha256(args.object)' in source
    assert '"source_sha256": _sha256(args.source)' in source
    assert '"compile_command_sha256": hashlib.sha256(' in source
    assert '"cuobjdump_version": _tool_version(args.cuobjdump)' in source
    assert '"nvcc_version": _tool_version(args.nvcc)' in source
    assert '"arch=compute_121a,code=sm_121a"' in source


def test_cuda_extension_contract_is_sm121_and_strict() -> None:
    source = CUDA_SOURCE.read_text(encoding="utf-8")
    assert "properties->major == 12 && properties->minor == 1" in source
    assert "properties->maxThreadsPerBlock >= kThreadsPerBlock" in source
    assert "properties->sharedMemPerBlock >= kSharedBytes" in source
    assert "bank_offset_table_prevalidated" in source
    assert "accepted_paths.scalar_type() == torch::kInt32" in source
    assert "k_rings.scalar_type() == torch::kBFloat16" in source
    assert "event_gate_scalars.scalar_type() == torch::kFloat32" in source


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

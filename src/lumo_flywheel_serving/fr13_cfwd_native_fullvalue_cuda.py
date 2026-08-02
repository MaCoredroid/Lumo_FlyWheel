"""Control contract for the default-off key-group precompute CFWD."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path


CANDIDATE = "fixed32_cfwd_native_keygroup_triton_scalar_cuda_v4"
SELECTOR_ENV = "FR13_FIXED32_CFWD_NATIVE_KEYGROUP_PRECOMPUTE_CUDA"
SELECTOR_VALUE = "diagnostic"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
VLLM_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
OPERATOR = "_C::fr13_fixed32_cfwd_native_fullvalue"
BINARY_BINDING_SCHEMA = "fr13.fixed32.cfwd_native_keygroup_binary.v1"
CUDA_SOURCE_PATH = "native/fr13_fixed32_cfwd_native_fullvalue.cu"
CUDA_SOURCE_SHA256 = (
    "5699ab062624bd2f6368143c48068bfccf1f9c3b5629e243d92616b94359bc54"
)
PATCHER_SOURCE_PATH = "scripts/fr13_patch_vllm_cfwd_native_fullvalue_cuda.py"
PATCHER_SOURCE_SHA256 = (
    "78aadbbf0cd5c1150ff26f6345804d309781a34de0469bd319897c2ad640a4e0"
)
PATCHED_VLLM_SHA256 = {
    "CMakeLists.txt": (
        "2a82eabaf9b6ab63bcb357932a5bf60b46506d81528845ac651d250759b9a1ba"
    ),
    "csrc/ops.h": (
        "01cf1cd80d4e78509bc9298c487a64d54e5622595d69bfe395b1a9bfaf57581f"
    ),
    "csrc/torch_bindings.cpp": (
        "3849a0d5a2e6928b311a76c34039b97b0156a61f56e33c94a6db41a88fde5e27"
    ),
    "csrc/fr13_fixed32_cfwd_native_fullvalue.cu": CUDA_SOURCE_SHA256,
}
MAX_BINDING_BYTES = 64 * 1024


def _require_sha256(value: object, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimeError(f"native key-group precompute CFWD {label} is invalid")
    return digest


def validate_binary_binding(binding: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact vLLM binary/source binding used by a live gate."""
    expected_keys = {
        "schema",
        "candidate",
        "vllm_base_commit",
        "operator",
        "architecture",
        "source_sha256",
        "patched_vllm_sha256",
        "build",
        "binary",
        "default_on",
        "production_authorized",
        "timing_eligible",
    }
    if set(binding) != expected_keys:
        raise RuntimeError("native key-group precompute CFWD binary keys drift")
    if binding.get("schema") != BINARY_BINDING_SCHEMA:
        raise RuntimeError("native key-group precompute CFWD binary schema drift")
    if binding.get("candidate") != CANDIDATE:
        raise RuntimeError("native key-group precompute CFWD binary candidate drift")
    if binding.get("vllm_base_commit") != VLLM_COMMIT:
        raise RuntimeError("native key-group precompute CFWD vLLM base drift")
    if binding.get("operator") != OPERATOR:
        raise RuntimeError("native key-group precompute CFWD operator binding drift")
    sources = binding.get("source_sha256")
    if not isinstance(sources, Mapping) or dict(sources) != {
        CUDA_SOURCE_PATH: CUDA_SOURCE_SHA256,
        PATCHER_SOURCE_PATH: PATCHER_SOURCE_SHA256,
    }:
        raise RuntimeError("native key-group precompute CFWD source binding drift")
    patched_vllm = binding.get("patched_vllm_sha256")
    if not isinstance(patched_vllm, Mapping) or dict(patched_vllm) != (
        PATCHED_VLLM_SHA256
    ):
        raise RuntimeError("native key-group precompute CFWD patched vLLM drift")
    build = binding.get("build")
    if not isinstance(build, Mapping) or set(build) != {
        "generator",
        "candidate_source_in_build_graph",
        "candidate_source_forced_rebuild",
        "candidate_source_mtime_ns",
        "candidate_object_outputs",
        "candidate_objects",
        "full_vllm_extension_target",
        "full_extension_mtime_ns",
        "cmake_cache_sha256",
        "build_ninja_sha256",
    }:
        raise RuntimeError("native key-group precompute CFWD build binding drift")
    if (
        build.get("generator") != "ninja"
        or build.get("candidate_source_in_build_graph") is not True
        or build.get("candidate_source_forced_rebuild") is not True
        or build.get("full_vllm_extension_target") != "_C.abi3.so"
    ):
        raise RuntimeError("native key-group precompute CFWD build target drift")
    cmake_cache_sha256 = _require_sha256(
        build.get("cmake_cache_sha256"), "CMake cache SHA-256"
    )
    build_ninja_sha256 = _require_sha256(
        build.get("build_ninja_sha256"), "build.ninja SHA-256"
    )
    candidate_source_mtime_ns = build.get("candidate_source_mtime_ns")
    full_extension_mtime_ns = build.get("full_extension_mtime_ns")
    candidate_object_outputs = build.get("candidate_object_outputs")
    candidate_objects = build.get("candidate_objects")
    if (
        type(candidate_source_mtime_ns) is not int
        or candidate_source_mtime_ns <= 0
        or type(full_extension_mtime_ns) is not int
        or not isinstance(candidate_object_outputs, list)
        or len(candidate_object_outputs) != 1
        or type(candidate_object_outputs[0]) is not str
        or Path(candidate_object_outputs[0]).is_absolute()
        or ".." in Path(candidate_object_outputs[0]).parts
        or Path(candidate_object_outputs[0]).suffix != ".o"
        or not isinstance(candidate_objects, list)
        or len(candidate_objects) != 1
        or not isinstance(candidate_objects[0], Mapping)
        or set(candidate_objects[0]) != {"path", "sha256", "bytes", "mtime_ns"}
        or candidate_objects[0].get("path") != candidate_object_outputs[0]
    ):
        raise RuntimeError("native key-group precompute CFWD forced rebuild drift")
    candidate_object_sha256 = _require_sha256(
        candidate_objects[0].get("sha256"), "candidate object SHA-256"
    )
    candidate_object_bytes = candidate_objects[0].get("bytes")
    candidate_object_mtime_ns = candidate_objects[0].get("mtime_ns")
    if (
        type(candidate_object_bytes) is not int
        or candidate_object_bytes <= 0
        or type(candidate_object_mtime_ns) is not int
        or candidate_object_mtime_ns < candidate_source_mtime_ns
        or full_extension_mtime_ns < candidate_object_mtime_ns
    ):
        raise RuntimeError("native key-group precompute CFWD object rebuild drift")
    binary = binding.get("binary")
    if not isinstance(binary, Mapping):
        raise RuntimeError("native key-group precompute CFWD binary binding is absent")
    binary_sha256 = _require_sha256(binary.get("sha256"), "binary SHA-256")
    binary_bytes = binary.get("bytes")
    if type(binary_bytes) is not int or binary_bytes <= 0:
        raise RuntimeError("native key-group precompute CFWD binary size is invalid")
    if binding.get("architecture") != "sm_121a":
        raise RuntimeError("native key-group precompute CFWD architecture drift")
    if (
        binding.get("default_on") is not False
        or binding.get("production_authorized") is not False
        or binding.get("timing_eligible") is not False
    ):
        raise RuntimeError("native key-group precompute CFWD qualification scope drift")
    return {
        "schema": BINARY_BINDING_SCHEMA,
        "candidate": CANDIDATE,
        "vllm_base_commit": VLLM_COMMIT,
        "operator": OPERATOR,
        "architecture": "sm_121a",
        "source_sha256": dict(sources),
        "patched_vllm_sha256": dict(patched_vllm),
        "build": {
            "generator": "ninja",
            "candidate_source_in_build_graph": True,
            "candidate_source_forced_rebuild": True,
            "candidate_source_mtime_ns": candidate_source_mtime_ns,
            "candidate_object_outputs": list(candidate_object_outputs),
            "candidate_objects": [
                {
                    "path": candidate_object_outputs[0],
                    "sha256": candidate_object_sha256,
                    "bytes": candidate_object_bytes,
                    "mtime_ns": candidate_object_mtime_ns,
                }
            ],
            "full_vllm_extension_target": "_C.abi3.so",
            "full_extension_mtime_ns": full_extension_mtime_ns,
            "cmake_cache_sha256": cmake_cache_sha256,
            "build_ninja_sha256": build_ninja_sha256,
        },
        "binary": {"sha256": binary_sha256, "bytes": binary_bytes},
        "default_on": False,
        "production_authorized": False,
        "timing_eligible": False,
    }


def load_binary_binding(path: str | os.PathLike[str]) -> dict[str, object]:
    """Read one private, non-symlinked live-gate binary binding."""
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(
                    "native key-group precompute CFWD duplicate binding key"
                )
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise RuntimeError(
            "native key-group precompute CFWD non-finite binding value"
        )

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as error:
        raise RuntimeError(
            "native key-group precompute CFWD cannot open binary binding"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or not 0 < metadata.st_size <= MAX_BINDING_BYTES
        ):
            raise RuntimeError(
                "native key-group precompute CFWD binary binding must be a "
                "private read-only regular file"
            )
        with os.fdopen(descriptor, encoding="ascii", closefd=False) as handle:
            payload = json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "native key-group precompute CFWD cannot read binary binding"
        ) from error
    finally:
        os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "native key-group precompute CFWD binary binding is not an object"
        )
    return validate_binary_binding(payload)


def resource_contract(batch_size: int) -> dict[str, object]:
    """Return source-level launch/resource facts, never measured results."""
    batch = int(batch_size)
    if batch not in (1, 2, 3, 4):
        raise ValueError(
            f"native key-group precompute CFWD requires B1-B4, got B={batch}"
        )
    ctas = 48 * batch * 16
    return {
        "candidate": CANDIDATE,
        "batch_size": batch,
        "layers": 48,
        "physical_rows_per_request": 32,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "value_heads_per_key_head": 3,
        "dim_k": 128,
        "dim_v": 128,
        "path_cap": 16,
        "max_accepted_length": 11,
        "root_inclusive_recurrence_steps_max": 12,
        "ctas_per_event": ctas,
        "ctas_per_layer": batch * 16,
        "ctas_per_layer_request_key_head": 1,
        "value_heads_per_cta": 3,
        "value_heads_processed_sequentially": True,
        "launches_per_event": 2,
        "event_gate_scalar_precompute_launches": 1,
        "native_recurrence_launches": 1,
        "threads_per_cta": 512,
        "warps_per_cta": 16,
        "value_rows_per_warp_per_head": 8,
        "key_columns_per_lane": 4,
        "fp32_state_elements_per_thread": 32,
        "fp32_register_state_elements_per_thread": 32,
        "fp32_shared_state_elements_per_thread": 0,
        "precomputed_step_capacity": 12,
        "precomputed_steps_per_wave": 4,
        "precompute_waves": 3,
        "normalized_k_shared_elements": 12 * 128,
        "norm_partial_shared_elements": 4 * 4,
        "inverse_norm_shared_elements": 4,
        "precomputed_node_shared_elements": 12,
        "precomputed_gate_scalar_shared_elements": 12 * 3 * 2,
        "event_gate_scalar_elements": 48 * batch * 12 * 48 * 2,
        "event_gate_scalar_math": "triton_incumbent_lowering",
        "native_gate_transcendentals": 0,
        "static_shared_bytes_source_model": 6_568,
        "k_hbm_vector_loads_per_cta_step": 1,
        "k_norms_per_cta_step": 1,
        "duplicate_value_head_k_loads_per_key_head_step": 0,
        "persistent_shared_state_elements": 0,
        "state_hbm_traffic_removed": False,
        "gate_coefficients_precomputed": True,
        "final_bank_store_dtype": "float32",
        "compile_gate": {
            "architecture": "sm_121a",
            "registers_per_thread_at_most": 64,
            "minimum_ctas_per_sm_target": 2,
            "stack_frame_bytes": 0,
            "local_bytes": 0,
            "spill_load_bytes": 0,
            "spill_store_bytes": 0,
        },
    }


def operator_available(torch_module: object) -> bool:
    try:
        namespace = getattr(getattr(torch_module, "ops"), "_C")
        getattr(namespace, "fr13_fixed32_cfwd_native_fullvalue")
    except (AttributeError, RuntimeError):
        return False
    return True


def resolve_candidate(
    *,
    mode: str | None,
    batch_size: int,
    num_layers: int,
    ring_nodes: int,
    num_key_heads: int,
    num_value_heads: int,
    dim_k: int,
    dim_v: int,
    path_cap: int,
    max_accepted_length: int,
    bank_dtype: str,
    ring_dtype: str,
    gate_dtype: str,
    index_dtype: str,
    use_qk_l2norm: bool,
    gate_coefficients_precomputed: bool,
    bank_offset_table_prevalidated: bool,
    accepted_values_device_guarded: bool,
    op_available: bool,
    binary_binding: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    """Resolve only an explicitly armed exact-geometry diagnostic candidate."""
    env = os.environ if environ is None else environ
    selector = env.get(SELECTOR_ENV, "")
    if selector in ("", "0"):
        return None
    if selector != SELECTOR_VALUE:
        raise RuntimeError(
            f"{SELECTOR_ENV} must be unset, 0, or {SELECTOR_VALUE!r}"
        )
    exact = (
        mode in FIXED32_MODES
        and int(batch_size) in (1, 2, 3, 4)
        and int(num_layers) == 48
        and int(ring_nodes) == 32
        and int(num_key_heads) == 16
        and int(num_value_heads) == 48
        and int(dim_k) == 128
        and int(dim_v) == 128
        and int(path_cap) == 16
        and int(max_accepted_length) == 11
        and bank_dtype == "float32"
        and ring_dtype == "bfloat16"
        and gate_dtype == "float32"
        and index_dtype == "int32"
        and bool(use_qk_l2norm)
        and bool(gate_coefficients_precomputed)
        and bool(bank_offset_table_prevalidated)
        and bool(accepted_values_device_guarded)
    )
    if not exact:
        raise RuntimeError(
            "armed native key-group precompute CFWD exact fixed32 contract drift"
        )
    if not op_available:
        raise RuntimeError(
            "armed native key-group precompute CFWD operator is absent from "
            "pinned vLLM _C"
        )
    if binary_binding is None:
        raise RuntimeError(
            "armed native key-group precompute CFWD has no binary binding"
        )
    binding = validate_binary_binding(binary_binding)
    return {
        **resource_contract(int(batch_size)),
        "mode": mode,
        "selector": SELECTOR_VALUE,
        "default_off": True,
        "source_only": True,
        "production_authorized": False,
        "fallback_on_error": False,
        "timing_eligible": False,
        "bank_offset_table_prevalidated": True,
        "accepted_values_device_guarded": True,
        "source_bound": True,
        "source_sha256": CUDA_SOURCE_SHA256,
        "binary_sha256": binding["binary"]["sha256"],
        "vllm_base_commit": VLLM_COMMIT,
        "operator": OPERATOR,
    }


def incumbent_byte_gate_plan() -> dict[str, object]:
    """Describe mandatory gates; this function does not issue credentials."""
    return {
        "reference_route": "force_incumbent_fixed32_cfwd_bv64",
        "candidate_route": CANDIDATE,
        "qualification_work": {
            "b1": "real SWE-Verified task bracket",
            "b4": "canonical real SWE-Verified exact4 campaign bracket",
        },
        "accepted_lengths_required": tuple(range(12)),
        "layers_required": 48,
        "surfaces": ("all_48_fp32_running_bank_rows",),
        "comparison": "raw_bytes",
        "restore_before_candidate": True,
        "reference_always_served_during_qualification": True,
        "same_server_process_required": True,
        "source_binding_required": {
            "schema": BINARY_BINDING_SCHEMA,
            "vllm_base_commit": VLLM_COMMIT,
            "cuda_source_sha256": CUDA_SOURCE_SHA256,
            "patcher_source_sha256": PATCHER_SOURCE_SHA256,
        },
        "pinned_compile_resource_gate_required": True,
        "production_credential_emitted": False,
        "timing_eligible": False,
    }


def launch_candidate(
    *,
    torch_module: object,
    selection: Mapping[str, object],
    bank_anchor: object,
    bank_off16: object,
    accepted_paths: object,
    accepted_lens: object,
    spec_state_indices: object,
    k_rings: object,
    v_rings: object,
    event_gate_scalars: object,
) -> None:
    """Launch a source-bound diagnostic selection or fail before dispatch."""
    if selection.get("candidate") != CANDIDATE:
        raise RuntimeError(
            "native key-group precompute CFWD selection is not source-bound"
        )
    if selection.get("production_authorized") is not False:
        raise RuntimeError(
            "native key-group precompute CFWD cannot authorize production"
        )
    if selection.get("timing_eligible") is not False:
        raise RuntimeError(
            "unqualified native key-group precompute CFWD cannot be timed"
        )
    if (
        selection.get("source_bound") is not True
        or selection.get("source_sha256") != CUDA_SOURCE_SHA256
        or selection.get("vllm_base_commit") != VLLM_COMMIT
        or selection.get("operator") != OPERATOR
    ):
        raise RuntimeError(
            "native key-group precompute CFWD launch source binding drift"
        )
    _require_sha256(selection.get("binary_sha256"), "launch binary SHA-256")
    if selection.get("bank_offset_table_prevalidated") is not True or (
        selection.get("accepted_values_device_guarded") is not True
    ):
        raise RuntimeError(
            "native key-group precompute CFWD safety guards are absent"
        )
    if not operator_available(torch_module):
        raise RuntimeError(
            "native key-group precompute CFWD op disappeared before launch"
        )
    op = torch_module.ops._C.fr13_fixed32_cfwd_native_fullvalue
    op(
        bank_anchor,
        bank_off16,
        accepted_paths,
        accepted_lens,
        spec_state_indices,
        k_rings,
        v_rings,
        event_gate_scalars,
        int(selection["batch_size"]),
        True,
        True,
    )

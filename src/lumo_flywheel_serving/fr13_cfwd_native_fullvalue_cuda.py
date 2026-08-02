"""Control contract for the default-off key-group precompute CFWD."""

from __future__ import annotations

import os
from collections.abc import Mapping


CANDIDATE = "fixed32_cfwd_native_keygroup_precompute_cuda_v3"
SELECTOR_ENV = "FR13_FIXED32_CFWD_NATIVE_KEYGROUP_PRECOMPUTE_CUDA"
SELECTOR_VALUE = "diagnostic"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))


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
        "launches_per_event": 1,
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
    a_rings: object,
    b_rings: object,
    gate_coeffs: object,
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
        a_rings,
        b_rings,
        gate_coeffs,
        int(selection["batch_size"]),
        True,
        True,
    )

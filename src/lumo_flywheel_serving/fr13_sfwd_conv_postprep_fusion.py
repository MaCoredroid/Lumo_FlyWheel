"""Source-only fixed32 tree-conv/post-conv-prep fusion candidate.

This module is intentionally not wired into the served patcher.  Its launcher
exists for a later authenticated byte gate and refuses graph capture or an
implicit production call.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton

from lumo_flywheel_serving.fr13_sfwd_conv_postprep_fusion_kernel import (
    _fr13_fixed32_sfwd_conv_postprep_fusion_kernel,
)
from lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless import (
    FIXED32_CHANNEL_SERIAL_ACTIVATION_WINDOW,
    FIXED32_CHANNEL_SERIAL_DEFERRED_STAGE_PEAK_LIVE_X,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER,
    FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
    FIXED32_CHANNEL_SERIAL_PEAK_LIVE_ACC,
    FIXED32_PARENT,
)


CANDIDATE = "fixed32_sfwd_conv_postprep_frontier5_direct_v1"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
QUALIFICATION_PROFILE = "k64_root"
DRAFT_VOCAB_K = 65536
DRAFT_VOCAB_ROOT = 1
LAYERS = 48
ROWS = 32
CHANNELS = 10240
CONV_WIDTH = 4
CONV_STATE_LEN = 34
SOURCE_ROWS = 36
X_ROW_STRIDE = 16384
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_K_DIM = 128
HEAD_V_DIM = 128
Q_DIM = NUM_K_HEADS * HEAD_K_DIM
V_DIM = NUM_V_HEADS * HEAD_V_DIM
GATE_BLOCK = 64
SOFTPLUS_THRESHOLD = 20.0
BF16_BYTES = 2
FP32_BYTES = 4


def _exact_host_parent(tree_parent: object) -> tuple[int, ...]:
    if not isinstance(tree_parent, (list, tuple)) or any(
        type(value) is not int for value in tree_parent
    ):
        raise ValueError(
            "fixed32 conv/post-prep fusion tree_parent must be a host int "
            "list/tuple"
        )
    actual = tuple(tree_parent)
    if actual != FIXED32_PARENT:
        raise RuntimeError(
            "fixed32 conv/post-prep fusion physical32 parent vector drifted"
        )
    return actual


def fixed32_sfwd_conv_postprep_fusion_contract(
    batch_size: int,
    *,
    fixed32_mode: str,
    tree_parent: object,
    qualification_profile: str,
    draft_vocab_k: int,
    draft_vocab_root: int,
    tree_rows: int = ROWS,
    conv_width: int = CONV_WIDTH,
    conv_state_len: int = CONV_STATE_LEN,
) -> dict[str, object]:
    """Close the only per-layer producer/consumer fusion geometry."""
    batch = int(batch_size)
    rows = int(tree_rows)
    width = int(conv_width)
    state_len = int(conv_state_len)
    _exact_host_parent(tree_parent)
    if batch not in (1, 2, 3, 4):
        raise ValueError("fixed32 conv/post-prep fusion requires B1-B4")
    if fixed32_mode not in FIXED32_MODES:
        raise RuntimeError("fixed32 conv/post-prep fusion requires an exact mode")
    if (
        str(qualification_profile) != QUALIFICATION_PROFILE
        or int(draft_vocab_k) != DRAFT_VOCAB_K
        or int(draft_vocab_root) != DRAFT_VOCAB_ROOT
    ):
        raise RuntimeError(
            "fixed32 conv/post-prep fusion requires physical32 K64/root1"
        )
    if (rows, width, state_len) != (ROWS, CONV_WIDTH, CONV_STATE_LEN):
        raise ValueError(
            "fixed32 conv/post-prep fusion requires rows/width/state "
            f"({ROWS}, {CONV_WIDTH}, {CONV_STATE_LEN})"
        )
    block_c = 128 if batch == 1 else 256
    num_warps = 2 if batch == 1 else 4
    channel_programs = CHANNELS // block_c
    return {
        "candidate": CANDIDATE,
        "source_only": True,
        "default_off": True,
        "production_eligible": False,
        "fixed32_mode": fixed32_mode,
        "qualification_profile": QUALIFICATION_PROFILE,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "batch_size": batch,
        "layers": LAYERS,
        "physical_rows_per_request": ROWS,
        "logical_rows": batch * ROWS,
        "channels": CHANNELS,
        "conv_width": CONV_WIDTH,
        "conv_state_len": CONV_STATE_LEN,
        "source_rows_per_request": SOURCE_ROWS,
        "num_k_heads": NUM_K_HEADS,
        "num_v_heads": NUM_V_HEADS,
        "head_k_dim": HEAD_K_DIM,
        "head_v_dim": HEAD_V_DIM,
        "query_channels": Q_DIM,
        "value_channels": V_DIM,
        "block_c": block_c,
        "num_warps": num_warps,
        "channel_programs_per_request": channel_programs,
        "gating_programs_per_request": ROWS,
        "programs_per_request": channel_programs + ROWS,
        "launches_per_layer": 1,
        "launches_for_all_layers": LAYERS,
        "cross_layer_fusion": False,
        "layer_recurrence_order": (
            "conv_postprep_then_gdn_then_residual_then_next_layer"
        ),
        "conv_product_dtype": "bfloat16",
        "conv_accumulator_dtype": "float32",
        "conv_add_order": ("bias", "tap0", "tap1", "tap2", "tap3"),
        "post_activation_boundary_dtype": "bfloat16",
        "query_key_l2norm": "deferred_to_existing_gdn_kernel",
        "postprep_dead_normalized_query_key_outputs": "not_materialized",
        "conv_tap_default": False,
        "commit_source_stage": True,
        "distinct_recurrence_output_storages": True,
        "conv_node_order": FIXED32_CHANNEL_SERIAL_FRONTIER5_ORDER,
        "conv_peak_live_x": FIXED32_CHANNEL_SERIAL_FRONTIER5_PEAK_LIVE_X,
        "conv_activation_window": FIXED32_CHANNEL_SERIAL_ACTIVATION_WINDOW,
        "conv_peak_live_acc": FIXED32_CHANNEL_SERIAL_PEAK_LIVE_ACC,
        "conv_peak_live_x_with_stage": (
            FIXED32_CHANNEL_SERIAL_DEFERRED_STAGE_PEAK_LIVE_X
        ),
        "algorithmic_shared_bytes": 0,
        "has_reduction": False,
        "has_barrier": False,
        "incumbent_codegen_registers_per_thread": 48,
        "candidate_codegen_registers_per_thread": 56,
        "source_register_ceiling_per_thread": 64,
        "source_register_ceiling_basis": (
            "frontier5_conv_branch_plus_direct_store_addresses_or_disjoint_"
            "scalar_gating_branch"
        ),
        "offline_codegen_target": "sm_121a",
        "offline_codegen_stack_bytes": 0,
        "offline_codegen_local_bytes": 0,
        "offline_codegen_shared_bytes": 0,
        "offline_codegen_profiles": ("B1", "B4", "B1_tap", "B4_tap"),
        "codegen_registers_verified": True,
        "timing_claim": False,
    }


def fixed32_sfwd_conv_postprep_static_ledger(
    batch_size: int,
    *,
    layers: int = LAYERS,
    store_conv_tap: bool = False,
) -> dict[str, object]:
    """Return deterministic logical global-byte and launch accounting."""
    batch = int(batch_size)
    layer_count = int(layers)
    if batch not in (1, 2, 3, 4):
        raise ValueError("fixed32 conv/post-prep ledger requires B1-B4")
    if layer_count != LAYERS:
        raise ValueError("fixed32 conv/post-prep ledger requires exactly 48 layers")
    if type(store_conv_tap) is not bool:
        raise TypeError("store_conv_tap must be bool")
    rows = batch * ROWS
    conv_surface = rows * CHANNELS * BF16_BYTES
    recurrence_surface = conv_surface
    value_tree_surface = rows * V_DIM * BF16_BYTES
    gate_surfaces = rows * NUM_V_HEADS * FP32_BYTES * 2
    source_stage_surface = batch * SOURCE_ROWS * CHANNELS * BF16_BYTES
    dead_normalized_qk_surface = rows * (2 * Q_DIM) * BF16_BYTES
    mandatory_direct_writes = (
        recurrence_surface + value_tree_surface + gate_surfaces
        + source_stage_surface
    )
    tap_write = conv_surface if store_conv_tap else 0
    removed_conv_write = 0 if store_conv_tap else conv_surface
    removed_conv_reads = conv_surface * 2
    removed_dead_qk_write = dead_normalized_qk_surface
    logical_traffic_removed = (
        removed_conv_write + removed_conv_reads + removed_dead_qk_write
    )
    return {
        "schema": "fr13.fixed32.sfwd_conv_postprep.static_ledger.v1",
        "candidate": CANDIDATE,
        "batch_size": batch,
        "layers": layer_count,
        "rows_per_layer": rows,
        "store_conv_tap": store_conv_tap,
        "per_layer_bytes": {
            "incumbent_conv_intermediate_write": conv_surface,
            "incumbent_rearrange_read": conv_surface,
            "incumbent_fused_postprep_read": conv_surface,
            "incumbent_dead_normalized_qk_write": dead_normalized_qk_surface,
            "direct_recurrence_qkv_write": recurrence_surface,
            "direct_value_tree_write": value_tree_surface,
            "direct_g_beta_write": gate_surfaces,
            "unchanged_commit_source_stage_write": source_stage_surface,
            "optional_conv_tap_write": tap_write,
            "mandatory_direct_writes": mandatory_direct_writes,
            "logical_traffic_removed": logical_traffic_removed,
        },
        "all_layer_bytes": {
            "conv_intermediate_write_removed": removed_conv_write * layer_count,
            "conv_intermediate_reads_removed": removed_conv_reads * layer_count,
            "dead_normalized_qk_write_removed": (
                removed_dead_qk_write * layer_count
            ),
            "logical_traffic_removed": logical_traffic_removed * layer_count,
        },
        "launches": {
            "incumbent_per_layer": 2,
            "candidate_per_layer": 1,
            "incumbent_all_layers": 2 * layer_count,
            "candidate_all_layers": layer_count,
            "removed_all_layers": layer_count,
        },
        "excludes": (
            "input_weight_reads",
            "cache_effects",
            "compiler_effects",
            "timing",
        ),
    }


def _storage_identity(tensor: torch.Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _storage_bound_failure(tensor: torch.Tensor) -> bool:
    if any(int(stride) < 0 for stride in tensor.stride()):
        return True
    if tensor.numel() == 0:
        return False
    last = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        last += (int(size) - 1) * int(stride)
    capacity = int(tensor.untyped_storage().nbytes()) // int(tensor.element_size())
    return last < 0 or last >= capacity


def fixed32_sfwd_conv_postprep_layout_contract(
    *,
    batch_size: int,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value_spec: torch.Tensor,
    value_tree: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    source_stage: torch.Tensor,
    conv_tap: torch.Tensor | None,
    expected_device_type: str = "cuda",
) -> dict[str, object]:
    """Fail closed over every tensor layout, storage bound, and output alias."""
    batch = int(batch_size)
    if batch not in (1, 2, 3, 4):
        raise ValueError("fixed32 conv/post-prep layout requires B1-B4")
    required_rows = batch * ROWS
    required_source_rows = batch * SOURCE_ROWS
    named: dict[str, torch.Tensor | None] = {
        "x": x,
        "conv_state": conv_state,
        "spec_state_indices": spec_state_indices,
        "conv_weights": conv_weights,
        "bias": bias,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "query": query,
        "key": key,
        "value_spec": value_spec,
        "value_tree": value_tree,
        "g": g,
        "beta": beta,
        "source_stage": source_stage,
        "conv_tap": conv_tap,
    }
    if any(
        tensor is not None and not torch.is_tensor(tensor)
        for tensor in named.values()
    ):
        raise TypeError("fixed32 conv/post-prep operands must be tensors or None")
    if not torch.is_tensor(x):
        raise TypeError("fixed32 conv/post-prep x must be a tensor")
    device = x.device
    present = {name: tensor for name, tensor in named.items() if tensor is not None}
    failures: list[str] = []
    if expected_device_type and device.type != expected_device_type:
        failures.append("device_type")
    if any(tensor.device != device for tensor in present.values()):
        failures.append("device_mismatch")

    exact_specs: Mapping[str, tuple[tuple[int, ...], tuple[int, ...], torch.dtype]] = {
        "x": ((required_rows, CHANNELS), (X_ROW_STRIDE, 1), torch.bfloat16),
        "spec_state_indices": ((batch, ROWS), (ROWS, 1), torch.int32),
        "conv_weights": ((CHANNELS, CONV_WIDTH), (CONV_WIDTH, 1), torch.bfloat16),
        "a": ((required_rows, NUM_V_HEADS), (NUM_V_HEADS, 1), torch.bfloat16),
        "b": ((required_rows, NUM_V_HEADS), (NUM_V_HEADS, 1), torch.bfloat16),
        "A_log": ((NUM_V_HEADS,), (1,), torch.float32),
        "dt_bias": ((NUM_V_HEADS,), (1,), torch.float32),
        "query": (
            (1, required_rows, NUM_K_HEADS, HEAD_K_DIM),
            (required_rows * Q_DIM, Q_DIM, HEAD_K_DIM, 1),
            torch.bfloat16,
        ),
        "key": (
            (1, required_rows, NUM_K_HEADS, HEAD_K_DIM),
            (required_rows * Q_DIM, Q_DIM, HEAD_K_DIM, 1),
            torch.bfloat16,
        ),
        "value_spec": (
            (1, required_rows, NUM_V_HEADS, HEAD_V_DIM),
            (required_rows * V_DIM, V_DIM, HEAD_V_DIM, 1),
            torch.bfloat16,
        ),
        "value_tree": (
            (required_rows, NUM_V_HEADS, HEAD_V_DIM),
            (V_DIM, HEAD_V_DIM, 1),
            torch.bfloat16,
        ),
        "g": (
            (required_rows, NUM_V_HEADS),
            (NUM_V_HEADS, 1),
            torch.float32,
        ),
        "beta": (
            (required_rows, NUM_V_HEADS),
            (NUM_V_HEADS, 1),
            torch.float32,
        ),
        "source_stage": (
            (required_source_rows, CHANNELS),
            (CHANNELS, 1),
            torch.bfloat16,
        ),
    }
    if conv_tap is not None:
        exact_specs = dict(exact_specs)
        exact_specs["conv_tap"] = (
            (required_rows, CHANNELS),
            (CHANNELS, 1),
            torch.bfloat16,
        )
    for name, (shape, stride, dtype) in exact_specs.items():
        tensor = present[name]
        if tuple(int(value) for value in tensor.shape) != shape:
            failures.append(f"{name}_shape")
        if tuple(int(value) for value in tensor.stride()) != stride:
            failures.append(f"{name}_stride")
        if tensor.dtype != dtype:
            failures.append(f"{name}_dtype")

    if conv_state.ndim != 3:
        failures.append("conv_state_ndim")
    else:
        if tuple(int(value) for value in conv_state.shape[1:]) != (
            CHANNELS,
            CONV_STATE_LEN,
        ):
            failures.append("conv_state_shape")
        conv_stride = tuple(int(value) for value in conv_state.stride())
        if (
            conv_stride[1:] != (1, CHANNELS)
            or conv_stride[0] < CHANNELS * CONV_STATE_LEN
        ):
            failures.append("conv_state_stride")
        if conv_state.dtype != torch.bfloat16:
            failures.append("conv_state_dtype")
    if bias is not None and (
        tuple(int(value) for value in bias.shape) != (CHANNELS,)
        or tuple(int(value) for value in bias.stride()) != (1,)
        or bias.dtype not in (torch.bfloat16, torch.float32)
    ):
        failures.append("bias_layout")
    if int(conv_weights.data_ptr()) % 4 != 0:
        failures.append("conv_weights_u32_alignment")
    for name, tensor in present.items():
        if _storage_bound_failure(tensor):
            failures.append(f"{name}_storage_bounds")

    writable_names = [
        "query",
        "key",
        "value_spec",
        "value_tree",
        "g",
        "beta",
        "source_stage",
    ]
    if conv_tap is not None:
        writable_names.append("conv_tap")
    readable_names = [
        "x",
        "conv_state",
        "spec_state_indices",
        "conv_weights",
        "a",
        "b",
        "A_log",
        "dt_bias",
    ]
    if bias is not None:
        readable_names.append("bias")
    writable_storage = [_storage_identity(present[name]) for name in writable_names]
    if len(set(writable_storage)) != len(writable_storage):
        failures.append("writable_storage_alias")
    readable_storage = {_storage_identity(present[name]) for name in readable_names}
    if any(identity in readable_storage for identity in writable_storage):
        failures.append("input_output_storage_alias")
    if failures:
        observed = {
            name: {
                "shape": tuple(int(value) for value in tensor.shape),
                "stride": tuple(int(value) for value in tensor.stride()),
                "dtype": str(tensor.dtype),
                "storage_offset": int(tensor.storage_offset()),
                "storage_bytes": int(tensor.untyped_storage().nbytes()),
            }
            for name, tensor in present.items()
        }
        raise ValueError(
            "fixed32 conv/post-prep operand contract drift: "
            f"failed={tuple(failures)!r}; observed={observed!r}"
        )
    return {
        "batch_size": batch,
        "device": str(device),
        "conv_state_stride_row": int(conv_state.stride(0)),
        "conv_tap": conv_tap is not None,
        "writable_storages": len(writable_storage),
        "input_aliases_allowed": True,
        "input_output_aliases_allowed": False,
    }


def launch_fixed32_sfwd_conv_postprep_fusion(
    *,
    x: torch.Tensor,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value_spec: torch.Tensor,
    value_tree: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    source_stage: torch.Tensor,
    conv_tap: torch.Tensor | None,
    batch_size: int,
    fixed32_mode: str,
    tree_parent: object,
    qualification_profile: str,
    draft_vocab_k: int,
    draft_vocab_root: int,
    physical32_guarded: bool,
    source_only_qualification: bool = False,
) -> dict[str, object]:
    """Launch one unqualified per-layer candidate kernel.

    A future runtime selector must replace the two explicit qualification
    booleans with an authenticated pass/guard binding.  This source revision
    refuses capture and cannot become serving-active by environment variable.
    """
    if source_only_qualification is not True:
        raise RuntimeError(
            "fixed32 conv/post-prep fusion is source-only and default-off"
        )
    if physical32_guarded is not True:
        raise RuntimeError(
            "fixed32 conv/post-prep fusion requires the upstream physical32 "
            "sticky guard"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "fixed32 conv/post-prep fusion is not graph-qualified"
        )
    contract = fixed32_sfwd_conv_postprep_fusion_contract(
        batch_size,
        fixed32_mode=fixed32_mode,
        tree_parent=tree_parent,
        qualification_profile=qualification_profile,
        draft_vocab_k=draft_vocab_k,
        draft_vocab_root=draft_vocab_root,
    )
    layout = fixed32_sfwd_conv_postprep_layout_contract(
        batch_size=batch_size,
        x=x,
        conv_state=conv_state,
        spec_state_indices=spec_state_indices,
        conv_weights=conv_weights,
        bias=bias,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        query=query,
        key=key,
        value_spec=value_spec,
        value_tree=value_tree,
        g=g,
        beta=beta,
        source_stage=source_stage,
        conv_tap=conv_tap,
    )
    block_c = int(contract["block_c"])
    num_warps = int(contract["num_warps"])
    channel_tasks = CHANNELS // block_c
    grid = (int(batch_size), channel_tasks + ROWS)
    bias_arg = bias if bias is not None else x
    conv_tap_arg = conv_tap if conv_tap is not None else query
    _fr13_fixed32_sfwd_conv_postprep_fusion_kernel[grid](
        x,
        conv_state,
        spec_state_indices,
        conv_weights,
        bias_arg,
        a,
        b,
        A_log,
        dt_bias,
        query,
        key,
        value_spec,
        value_tree,
        g,
        beta,
        source_stage,
        conv_tap_arg,
        CONV_STRIDE_ROW=int(conv_state.stride(0)),
        B=int(batch_size),
        N=ROWS,
        C=CHANNELS,
        WIDTH=CONV_WIDTH,
        STATE_LEN=CONV_STATE_LEN,
        SOURCE_ROWS=SOURCE_ROWS,
        H=NUM_K_HEADS,
        HV=NUM_V_HEADS,
        K=HEAD_K_DIM,
        V=HEAD_V_DIM,
        HAS_BIAS=bias is not None,
        STORE_CONV_TAP=conv_tap is not None,
        X_STRIDE_ROW=X_ROW_STRIDE,
        BLOCK_C=block_c,
        GATE_BLOCK=GATE_BLOCK,
        SOFTPLUS_THRESHOLD=SOFTPLUS_THRESHOLD,
        num_warps=num_warps,
    )
    return {**contract, "layout": layout, "conv_tap": conv_tap is not None}

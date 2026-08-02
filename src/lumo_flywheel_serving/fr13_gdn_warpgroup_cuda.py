"""Control and static contract for the fixed32 native SFWD candidate."""

from __future__ import annotations

import os
from collections.abc import Mapping


CANDIDATE = "fixed32_gdn_warpgroup_cuda_v1"
SELECTOR_ENV = "FR13_FIXED32_GDN_WARPGROUP_CUDA"
SELECTOR_VALUE = "diagnostic"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))

PARENT = (
    -1,
    0,
    0,
    0,
    1,
    1,
    1,
    2,
    3,
    4,
    4,
    4,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    14,
    14,
    17,
    18,
    19,
    23,
    24,
    25,
    26,
    28,
    29,
    30,
)
ROOT_PATH = (0, 1, 4, 9, 14)
GROUP_PATHS = (
    ((19, 24, 26, 28, 29, 30, 31), (20,), (21,)),
    ((2, 7, 12, 17, 22), (3, 8, 13, 18, 23, 25, 27)),
    ((5,), (6,)),
    ((10,), (11,)),
    ((15,), (16,)),
)
GROUP_PARENTS = (14, 0, 1, 4, 9)


def validate_static_schedule() -> dict[str, object]:
    writers = ROOT_PATH + tuple(
        node for group in GROUP_PATHS for path in group for node in path
    )
    if tuple(sorted(writers)) != tuple(range(32)) or len(set(writers)) != 32:
        raise RuntimeError("warp-group SFWD writer coverage drift")
    for parent, paths in zip(GROUP_PARENTS, GROUP_PATHS, strict=True):
        for path in paths:
            if PARENT[path[0]] != parent:
                raise RuntimeError("warp-group SFWD parent descriptor drift")
            for previous, node in zip(path, path[1:], strict=False):
                if PARENT[node] != previous:
                    raise RuntimeError("warp-group SFWD path ordering drift")
    if any(parent not in ROOT_PATH for parent in GROUP_PARENTS):
        raise RuntimeError("warp-group SFWD parent is absent from the root path")
    return {
        "root_path": ROOT_PATH,
        "root_depth": len(ROOT_PATH),
        "group_parents": GROUP_PARENTS,
        "group_sizes": tuple(len(group) for group in GROUP_PATHS),
        "active_member_warps": sum(len(group) for group in GROUP_PATHS),
        "max_branch_depth": max(
            len(path) for group in GROUP_PATHS for path in group
        ),
        "logical_depth": 12,
        "physical_recurrence_depth": 12,
        "writer_nodes": len(writers),
    }


def resource_contract(batch_size: int) -> dict[str, object]:
    batch = int(batch_size)
    if batch not in (1, 2, 3, 4):
        raise ValueError(f"warp-group SFWD requires B1-B4, got B={batch}")
    schedule = validate_static_schedule()
    state_tile_elements = 8 * 128
    parent_shared_bytes = 5 * state_tile_elements * 4
    return {
        "candidate": CANDIDATE,
        "batch_size": batch,
        "physical_rows_per_request": 32,
        "num_key_heads": 16,
        "num_value_heads": 48,
        "dim_k": 128,
        "dim_v": 128,
        "block_v": 8,
        "value_tiles_per_head": 16,
        "ctas_per_request_layer": 48 * 16,
        "ctas_per_layer": batch * 48 * 16,
        "launches_per_layer": 1,
        "warp_groups_per_cta": 5,
        "warps_per_group": 4,
        "warps_per_cta": 20,
        "threads_per_cta": 640,
        "active_member_warps": schedule["active_member_warps"],
        "inactive_member_warps": 20 - int(schedule["active_member_warps"]),
        "state_tile_fp32_elements": state_tile_elements,
        "state_tile_bytes": state_tile_elements * 4,
        "parent_tiles_in_shared": 5,
        "static_shared_bytes": parent_shared_bytes,
        "root_depth": schedule["root_depth"],
        "max_branch_depth": schedule["max_branch_depth"],
        "logical_depth": schedule["logical_depth"],
        "physical_recurrence_depth": schedule["physical_recurrence_depth"],
        "hbm_parent_state_exports": 0,
        "hbm_parent_state_reads": 0,
        "single_writer_nodes": schedule["writer_nodes"],
        "compile_gate": {
            "architecture": "sm_121",
            "max_threads_per_block_at_least": 640,
            "shared_memory_per_block_at_least": parent_shared_bytes,
            "registers_per_thread_at_most": 102,
            "local_bytes": 0,
            "spill_load_bytes": 0,
            "spill_store_bytes": 0,
        },
    }


def resolve_candidate(
    *,
    mode: str | None,
    batch_size: int,
    n_actual: int,
    n_pad: int,
    num_key_heads: int,
    num_value_heads: int,
    dim_k: int,
    dim_v: int,
    block_v: int,
    use_qk_l2norm: bool,
    scan_align: bool,
    ring_export: bool,
    flags_export: bool,
    h0_use_accepted_column: bool,
    op_available: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    env = os.environ if environ is None else environ
    selector = env.get(SELECTOR_ENV, "")
    if selector in ("", "0"):
        return None
    if selector != SELECTOR_VALUE:
        raise RuntimeError(
            f"{SELECTOR_ENV} must be unset, 0, or {SELECTOR_VALUE!r}"
        )
    expected = (
        mode in FIXED32_MODES
        and int(batch_size) in (1, 2, 3, 4)
        and int(n_actual) == 32
        and int(n_pad) == 32
        and int(num_key_heads) == 16
        and int(num_value_heads) == 48
        and int(dim_k) == 128
        and int(dim_v) == 128
        and int(block_v) == 8
        and bool(use_qk_l2norm)
        and bool(scan_align)
        and bool(ring_export)
        and bool(flags_export)
        and not bool(h0_use_accepted_column)
    )
    if not expected:
        raise RuntimeError(
            "armed warp-group SFWD exact fixed32 geometry/feature contract drift"
        )
    if not op_available:
        raise RuntimeError(
            "armed warp-group SFWD operator is absent from the pinned vLLM _C"
        )
    return {
        **resource_contract(int(batch_size)),
        "mode": mode,
        "selector": SELECTOR_VALUE,
        "default_off": True,
        "production_authorized": False,
        "fallback_on_error": False,
    }


def incumbent_byte_gate_plan() -> dict[str, object]:
    return {
        "reference_route": "force_incumbent_fixed32_path_bv8",
        "candidate_route": CANDIDATE,
        "qualification_work": {
            "b1": "real SWE-Verified task bracket",
            "b4": "canonical real SWE-Verified exact4 campaign bracket",
        },
        "layers_required": 48,
        "surfaces": (
            "output",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "flags",
            "invocation_counter",
        ),
        "comparison": "raw_bytes",
        "restore_before_candidate": True,
        "reference_always_served_during_qualification": True,
        "same_server_process_required": True,
        "production_credential_emitted": False,
        "timing_eligible": False,
    }


def operator_available(torch_module: object) -> bool:
    try:
        namespace = getattr(getattr(torch_module, "ops"), "_C")
        getattr(namespace, "fr13_fixed32_gdn_warpgroup")
    except (AttributeError, RuntimeError):
        return False
    return True


def launch_candidate(
    *,
    torch_module: object,
    selection: Mapping[str, object],
    out: object,
    q: object,
    k: object,
    v: object,
    raw_a: object,
    raw_b: object,
    a_log: object,
    dt_bias: object,
    h0: object,
    h0_indices: object,
    ring_k: object,
    ring_v: object,
    ring_a: object,
    ring_b: object,
    flags: object,
    invocation_counter: object,
    h0_index_row: int,
    h0_index_batch_stride: int,
    h0_bank_stride: int,
    output_scale: float,
    count_invocation: bool,
) -> None:
    if selection.get("candidate") != CANDIDATE:
        raise RuntimeError("warp-group SFWD launch selection is not source-bound")
    if selection.get("production_authorized") is not False:
        raise RuntimeError("warp-group SFWD source candidate cannot authorize production")
    if not operator_available(torch_module):
        raise RuntimeError("warp-group SFWD op disappeared before launch")
    op = torch_module.ops._C.fr13_fixed32_gdn_warpgroup
    op(
        out,
        q,
        k,
        v,
        raw_a,
        raw_b,
        a_log,
        dt_bias,
        h0,
        h0_indices,
        ring_k,
        ring_v,
        ring_a,
        ring_b,
        flags,
        invocation_counter,
        int(selection["batch_size"]),
        int(h0_index_row),
        int(h0_index_batch_stride),
        int(h0_bank_stride),
        float(output_scale),
        True,
        True,
        True,
        True,
        bool(count_invocation),
    )

"""FR13_REPLAY_ROUTE page-safe conv-state linear remap (wiring fix, no kernel).

Root cause (FR13 replay boundary trace, 2026-06-10, FR13_REPLAY_GPU_GATES_BIND
follow-up): vLLM builds the GDN conv (kv[0]) and ssm (kv[1]) caches as
``as_strided`` views over the SAME mamba page with
``stride(0) == num_element_per_page`` (gpu_model_runner._reshape_kv_cache_tensors,
MambaSpec branch: ``target_stride=(num_element_per_page, *stride[1:])``, the ssm
view at a storage offset after the conv slice). The frozen Triton remap in
``fr10_gdn_tree_kernel._remap_state_rows`` uses ``state.stride(0)`` as BOTH the
row-offset multiplier and the per-row copy extent, so a "conv-only" remap call
copies the WHOLE page -- conv slice + ssm slice -- from node columns to linear
columns. Under FR13_REPLAY_ROUTE=1 the ssm node columns are never written (the
committer replay publishes accepted ssm states directly to LINEAR columns and
``store_node_states=False`` compiles the scan's node export out), so the
page-wide copy drags never-written node-column ssm bytes (boot-fresh zeros or
stale reused-block leftovers) over the replay's just-published linear-column
states. Live byte prediction ``B.window[col c] == A.post.window[node path[c]]``
matched 581/581 with 0 mismatches on both probed layers.

This helper performs the IDENTICAL permutation as the Triton gather kernel
(_linear_remap_rows_gather_kernel: gather-then-scatter, same clamp/valid-lane
semantics) but in plain tensor ops that respect the VIEW's logical shape and
strides: ``index_select`` reads and ``index_copy_`` writes exactly the conv
slice's elements, never the page remainder. The kernel file stays frozen.

Legacy (FR13_REPLAY_ROUTE=0) keeps the whole-page Triton launch verbatim: there
the all-rows ssm publish refreshes every window column each event, which makes
the page-wide conv copy semantically identical to the intended ssm remap.

CUDA-graph safety: fixed-shape tensor ops only -- no host sync, no
data-dependent shapes, no Python branching on tensor values. Temporaries are
per-step intermediates (legal inside capture; the gate-4 lazy-persistent-alloc
ban does not apply to consumed-in-step temporaries).
"""

from __future__ import annotations

import torch


def replay_conv_state_linear_remap(
    *,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    max_path_len: int,
) -> None:
    """Materialize accepted tree-path rows into linear conv-state columns.

    Semantics (mirrors ``_linear_remap_rows_gather_kernel`` exactly): for each
    batch row ``b < num_spec_decodes`` and each linear column
    ``k < min(path_cols, spec_cols)`` with ``k < num_accepted_tokens[b]``::

        src_col = clamp(accepted_paths[b, k], 0, spec_cols - 1)
        conv_state[spec_state_indices[b, k]] = conv_state[spec_state_indices[b, src_col]]

    with every source row materialized BEFORE any destination row is written
    (race-free in-place overlapping permutation, the FR13_TREE_REMAP_SEQ
    license). Invalid lanes (``k >= len``) degrade to identity self-copies
    (byte-neutral) instead of the kernel's masked-off stores; destination rows
    are physically distinct within a window and windows are disjoint across
    requests, so the writes are conflict-free.

    Only the LOGICAL elements of ``conv_state`` are touched: on the
    page-sharing as_strided view this copies the conv slice and never the
    co-resident ssm slice.
    """
    if num_spec_decodes <= 0 or max_path_len <= 0:
        return
    if conv_state.ndim < 2:
        raise ValueError(
            f"conv_state bank must have row dimension plus payload, got {tuple(conv_state.shape)}"
        )
    if spec_state_indices.ndim != 2:
        raise ValueError(
            f"spec_state_indices must be 2D, got {tuple(spec_state_indices.shape)}"
        )
    if accepted_paths.ndim != 2:
        raise ValueError(
            f"accepted_paths must be 2D, got {tuple(accepted_paths.shape)}"
        )
    if accepted_paths.shape[0] < num_spec_decodes:
        raise ValueError(
            "accepted_paths batch rows must cover num_spec_decodes="
            f"{num_spec_decodes}, got {accepted_paths.shape[0]}"
        )
    if num_accepted_tokens.numel() < num_spec_decodes:
        raise ValueError(
            "num_accepted_tokens must cover num_spec_decodes="
            f"{num_spec_decodes}, got {num_accepted_tokens.numel()}"
        )
    spec_cols = int(spec_state_indices.shape[1])
    # The kernel's valid-lane mask requires ks < PATH_COLS, ks < SPEC_COLS and
    # ks < accepted_len; fold the static bounds into path_cols.
    path_cols = min(int(accepted_paths.shape[1]), int(max_path_len), spec_cols)
    if path_cols <= 0 or spec_cols <= 0:
        return

    b = int(num_spec_decodes)
    device = spec_state_indices.device
    ks = torch.arange(path_cols, device=device, dtype=torch.long)
    lens = num_accepted_tokens.reshape(-1)[:b].to(torch.long).view(b, 1)
    valid = ks.view(1, -1) < lens
    src_col = torch.clamp(
        accepted_paths[:b, :path_cols].to(torch.long), 0, spec_cols - 1
    )
    dst_col = ks.view(1, -1).expand(b, path_cols)
    # Invalid lanes become identity (src == dst): the subsequent self-copy
    # writes back the bytes it just read, matching the kernel's no-store.
    src_col = torch.where(valid, src_col, dst_col)
    window = spec_state_indices[:b].to(torch.long)
    src_rows = window.gather(1, src_col).reshape(-1)
    dst_rows = window.gather(1, dst_col).reshape(-1)
    # Gather-then-scatter: index_select MATERIALIZES every source row before
    # index_copy_ writes any destination row, making the in-place overlapping
    # permutation exact (spine paths map src cols [1..L] onto dst cols
    # [0..L-1]).
    vals = conv_state.index_select(0, src_rows)
    conv_state.index_copy_(0, dst_rows, vals)

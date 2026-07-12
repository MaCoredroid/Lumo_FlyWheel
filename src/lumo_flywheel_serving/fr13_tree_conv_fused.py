"""FR13_TREE_CONV_FUSED (FIX-3): fused tree causal-conv emulation ops.

Census target (FR13_B1_SPEED_ATTRIBUTION_BIND.md, holds=True): the tree
causal-conv Python torch-op emulation in
``scripts/fr10_phase4_patch_vllm_tree_gdn.py`` costs ~95 (chain5) /
~124-134 (cat9) captured device nodes PER GDN LAYER x48 vs native's single
``causal_conv1d_update`` node. Ranked contributors and their fused forms:

  (1) per-node state write-back loop (7 device nodes x tree_n x B, ~44-52%
      of the extra nodes) -> ONE init-time static-index gather over the
      shared ``(prior ++ x ++ zero_row)`` source + the existing
      ``index_copy_`` (playbook class 3 gather-then-scatter). Pure data
      movement, zero arithmetic: ``new_state[i][:, j] =
      (prior ++ x[path_i] ++ zeros)[path_len_i + j]`` is the closed-form
      composition of the loop's index math.
  (2) per-col tap/bias/silu emulation (~20%) -> one bf16 elementwise mul +
      one fp32 cast + bias broadcast-add + EXPLICIT ordered adds. Reduction
      ops are BANNED in this module (unspecified order breaks bit-exactness);
      the per-element operand pairs and the legacy add order
      ``(((bias + p0) + p1) + p2) + p3`` are preserved verbatim.
  (3) page-safe conv remap row math (~10.5%) and
  (4) committed-path prior gather row math (~9.5%) -> prepared-rows split:
      the shared row math runs once per kv-cache group per forward; each
      layer keeps only 1-2 bank ops (``index_select``/``index_copy_``).
      The frozen library fns (``fr13_replay_conv_remap`` /
      ``fr10_gdn_tree_kernel``) stay the byte-verbatim OFF arm.

THE BAR (class 10): BIT-EXACT to the current emulation — same bf16 tap
order, same cast boundaries (bf16 taps were THE FR11-overturning fix), same
accumulation order, same silu/bias op order. Math-correct != bit-exact;
per-stage tolerances are BANNED. Proof = int-view byte A/B
(tests/test_fr13_tree_conv_fused_byte_ab.py: synthetic + capture-payload),
then the live gate (B=1 same-seed repeat first).

SCOPE (tree-only, user 6c5aeaae): these helpers are consumed ONLY by the
tree-conv emulation branch (``use_fr10_tree_conv``); the native
``causal_conv1d_update`` path is untouched.

The ``legacy_*_reference`` functions are TEST-ONLY verbatim replicas of the
patcher's legacy emulation text / the frozen library row math (pinned by the
wiring test's substring asserts and anchored to the LIVE legacy bytes by the
capture-payload A/B arm). They must never be wired into serving.

CUDA-graph safety (class 6): every function here is fixed-shape tensor ops
with no host sync, no data-dependent shapes, no Python branching on tensor
values; per-step temporaries are consumed-in-step. The static index tables
and the zero source row are built ONCE (init-time / first non-capturing
forward, FIX-2 layer-keyed cache pattern) and passed in as arguments.
"""

from __future__ import annotations

import os

import torch


# ---------------------------------------------------------------------------
# Static index tables (init-time; value-static per tree topology).
# ---------------------------------------------------------------------------

# CUDA-graph capture-time staging retention (class 6, live-gate fix
# 2026-06-12): the cu130 nightly profiles cudagraph memory BEFORE any eager
# tree forward (gpu_model_runner "Profiling CUDA graph memory"), so the
# FIRST fused forward of a boot can run INSIDE stream capture and miss the
# layer-keyed table cache. A pageable host->device copy
# (torch.tensor(list, device="cuda")) is ILLEGAL during capture; a PINNED
# host source is legal but gets baked into the graph as an H2D copy node
# that re-reads the staging buffer on every replay — so every staging
# tensor must stay alive for the life of the process. Retained here
# (value-static, tree_n*state_len int64 each, a handful per boot). The
# DEVICE-side result stays unretained per the capture-pool license (the
# graph owns its pool address; the baked copy refills it on each replay).
_CAPTURE_STAGING_RETAIN: list[torch.Tensor] = []


def tree_paths_from_parent(parent: list[int]) -> list[list[int]]:
    """Root-to-node path (node ids) for every node, mirroring the builder."""
    paths: list[list[int]] = []
    for node in range(len(parent)):
        path: list[int] = []
        cur = node
        while cur >= 0:
            path.append(cur)
            cur = parent[cur]
        path.reverse()
        paths.append(path)
    return paths


def build_tree_conv_window_source_indices(
    *, parent: list[int], width: int, device: torch.device | str
) -> torch.Tensor:
    """TEST-ONLY replica of the builder's ``source_by_width`` math.

    Mirrors the metadata-builder init in
    ``scripts/fr10_phase4_patch_vllm_tree_gdn.py`` (``source_rows`` loop):
    per node, the last ``width`` rows of
    ``[prior rows 0..width-2] ++ [width-1+path_node for path_node in path]``.
    """
    rows = []
    for node, path in enumerate(tree_paths_from_parent(parent)):
        source = list(range(width - 1)) + [
            width - 1 + int(path_node) for path_node in path
        ]
        rows.append(source[-width:])
    return torch.tensor(rows, dtype=torch.long, device=device)


def build_tree_conv_state_src_indices(
    *,
    parent: list[int],
    width: int,
    state_len: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Static gather table replacing the per-node state write-back loop.

    Legacy semantics (proven, FR12_CONV_STATE_DIAGNOSTIC.md): for node ``i``
    and state column ``j``::

        new_state[i][:, j] = (prior ++ x[path_i] ++ zeros)[path_len_i + j]

    i.e. for ``j < width-1`` the last-(width-1) taps of
    ``(prior ++ committed-path tokens of node i)``; for ``j in
    [width-1, state_len)`` exact zeros (the live zero-pad branch:
    state_len=12 > width-1=3).

    Against the fused shared source
    ``source_z = cat(prior.T [width-1 rows], x [tree_n rows],
    zero_row [1 row])`` the closed form is, with ``p = path_len_i + j``::

        state_src[i, j] = p                                  if p < width-1
                          width-1 + path_i[p - (width-1)]    elif j < width-1
                          width-1 + tree_n   (the zero row)  otherwise

    Returns the FLAT [tree_n * state_len] int64 table (row-major (i, j)),
    matching ``fused_tree_conv_state_rows``'s
    ``index_select(0, ...).view(tree_n, state_len, dim)``.
    """
    if width < 2:
        raise ValueError(f"conv width must be >= 2, got {width}")
    if state_len < width - 1:
        raise ValueError(
            f"state_len {state_len} < width-1 {width - 1} is not the legacy "
            "write-back geometry (store_idx would read past the zero pad)"
        )
    paths = tree_paths_from_parent(parent)
    tree_n = len(parent)
    zero_row = width - 1 + tree_n
    flat: list[int] = []
    for i in range(tree_n):
        path_len = len(paths[i])
        for j in range(state_len):
            p = path_len + j
            if p < width - 1:
                flat.append(p)
            elif j < width - 1:
                flat.append(width - 1 + paths[i][p - (width - 1)])
            else:
                flat.append(zero_row)
    flat_cpu = torch.tensor(flat, dtype=torch.long)
    dev = torch.device(device)
    if dev.type != "cuda":
        return flat_cpu.to(dev)
    if torch.cuda.is_current_stream_capturing():
        # Capture-time miss (see _CAPTURE_STAGING_RETAIN): pinned staging is
        # the only capture-legal H2D; retain the staging buffer so the baked
        # copy node stays valid on every replay. Values are byte-identical
        # to the eager build (exact int64, no numerics).
        staging = flat_cpu.pin_memory()
        _CAPTURE_STAGING_RETAIN.append(staging)
        return staging.to(dev, non_blocking=True)
    return flat_cpu.to(dev)


# ---------------------------------------------------------------------------
# Fused per-layer ops (the FR13_TREE_CONV_FUSED=1 arm).
# ---------------------------------------------------------------------------


def fused_tree_conv_source(
    *, prior_window: torch.Tensor, x: torch.Tensor, zero_row: torch.Tensor
) -> torch.Tensor:
    """Shared source: ``cat(prior.T, x, zero_row)``.

    Reuses the window's existing cat with ONE appended zero row. Window /
    flat-source indices never reference the appended row (all are
    ``< width-1 + tree_n``), so every downstream window ``index_select``
    output is byte-identical to the legacy two-operand cat (CPU-executed
    invariance check in the byte A/B); the write-back gather reads the zero
    row for state columns ``>= width-1``.
    """
    if prior_window.dtype != x.dtype or zero_row.dtype != x.dtype:
        raise RuntimeError(
            "FR13_TREE_CONV_FUSED source dtype uniformity violated: "
            f"{prior_window.dtype}/{x.dtype}/{zero_row.dtype}"
        )
    return torch.cat((prior_window.transpose(0, 1), x, zero_row), dim=0)


def fused_tree_conv_taps_acc(
    *,
    window: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Vectorized bf16 taps + fp32 accumulation, legacy op/operand order.

    Legacy (native-bf16-taps arm of ``_fr11_conv_tap_product`` + the per-col
    loop): per col ``c``, ``tap_c = (x_col.to(bf16) * w_c.to(bf16))
    .to(bf16).to(f32)`` and ``acc = acc + tap_c`` with
    ``acc0 = bias.to(f32)`` (broadcast) or ``zeros``.

    Fused: ONE bf16 elementwise mul over [tree_n, width, dim] (per-element
    operand pairs identical to the per-col muls; bf16 mul is fp32-compute +
    RNE — exact: 8-bit x 8-bit mantissa products fit fp32), ONE fp32 cast
    (the same ``.to(f32)`` boundary), then EXPLICIT ordered adds in the
    legacy operand order — addition is never commuted and reduction ops are
    BANNED (unspecified order). The wasted ``x.float()`` materialization and
    the bias ``clone`` are dropped: the broadcast add has identical
    per-element operand pairs.

    Requires ``window.dtype == conv_weights.dtype`` (the legacy
    ``w.to(dtype)`` is a no-op cast then; fused skips the no-op entirely) —
    enforced fail-loud.
    """
    if window.dtype != conv_weights.dtype:
        raise RuntimeError(
            "FR13_TREE_CONV_FUSED tap dtype uniformity violated: "
            f"window={window.dtype} conv_weights={conv_weights.dtype}"
        )
    width = int(window.size(1))
    # conv_weights is [dim, width]; .t()/unsqueeze are VIEWS (no cast/copy
    # kernel). Elementwise mul in the window dtype = the legacy per-col mul.
    prod = window * conv_weights.t().unsqueeze(0)
    prod_f32 = prod.to(torch.float32)
    if bias is None:
        acc = torch.zeros(
            (int(window.size(0)), int(window.size(2))),
            dtype=torch.float32,
            device=window.device,
        )
        start = 0
    else:
        acc = bias.to(torch.float32).view(1, -1) + prod_f32[:, 0, :]
        start = 1
    # EXPLICIT ordered adds (legacy operand order: acc left, tap right).
    for col in range(start, width):
        acc = acc + prod_f32[:, col, :]
    return acc


def fused_tree_conv_state_rows(
    *,
    source_z: torch.Tensor,
    state_src: torch.Tensor,
    tree_n: int,
    state_len: int,
) -> torch.Tensor:
    """Per-node write-back rows via ONE static-index gather (class 3).

    Replaces the legacy 7-device-nodes-x-tree_n loop (per-node index_select
    / cat / new_zeros / arange / index_select / transpose + the final
    stack). Pure data movement: gathers are per-element moves with no
    arithmetic and no dtype change. Returns [tree_n, dim, state_len]
    contiguous, the legacy ``torch.stack`` layout.
    """
    return (
        source_z.index_select(0, state_src)
        .view(int(tree_n), int(state_len), int(source_z.size(1)))
        .transpose(1, 2)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# Prepared-rows split for the committed-prior gather + page-safe remap.
# The row math is remap-independent (reads only accepted paths/lens + spec
# indices, which no remap mutates), so once-per-kv-cache-group is exact; the
# per-layer remainder is 1-2 bank ops.
# ---------------------------------------------------------------------------


def prepare_committed_path_conv_rows(
    *,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    num_spec_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row math of ``gather_committed_path_conv_prior`` (frozen library fn)
    WITHOUT the per-layer bank ``index_select``.

    Identical op sequence to ``fr10_gdn_tree_kernel.
    gather_committed_path_conv_prior`` (the OFF arm): clamp/gather/where/
    clamp on the shared accepted paths/lens + spec indices. Returns
    ``(read_node_cols [B,1], bank_rows [B,1])`` int64.
    """
    b = int(num_spec_decodes)
    spec_cols = int(spec_state_indices.size(-1))
    device = spec_state_indices.device
    if accepted_paths is None or num_accepted_tokens is None:
        read_node_cols = torch.zeros((b, 1), dtype=torch.long, device=device)
    else:
        lens = num_accepted_tokens[:b].to(torch.long).view(-1, 1)
        path_cols = torch.clamp(
            lens - 1, min=0, max=int(accepted_paths.size(-1)) - 1
        )
        read_node_cols = accepted_paths[:b].to(torch.long).gather(1, path_cols)
        # len == 0 commits no draft node: read node column 0 (the committed
        # root token's window), matching the library fn exactly.
        read_node_cols = torch.where(
            lens > 0, read_node_cols, torch.zeros_like(read_node_cols)
        )
        read_node_cols = torch.clamp(read_node_cols, min=0, max=spec_cols - 1)
    if os.environ.get("FR13_TREE_RUNROW_INIT", "1") == "1":
        # STATELESS-TREE: seed the next-step conv prior from col 0 (the running
        # row, where the post-accept committer deposited this-step's committed
        # leaf window), not the accepted-leaf node column. Mirrors the SSM
        # RUNROW_INIT (h0_col=0). read_node_cols stays available for callers.
        read_node_cols = torch.zeros((b, 1), dtype=torch.long, device=device)
    bank_rows = spec_state_indices[:b].to(torch.long).gather(1, read_node_cols)
    return read_node_cols, bank_rows


def gather_committed_path_conv_prior_prepared(
    *, conv_state: torch.Tensor, bank_rows: torch.Tensor
) -> torch.Tensor:
    """Per-layer remainder of the committed-prior gather: the bank snapshot.

    MUST stay in-layer and PRE-remap (per-layer snapshot semantics): only
    the row math is hoisted to once-per-group.
    """
    return torch.index_select(conv_state, 0, bank_rows.reshape(-1))


def prepare_replay_conv_remap_rows(
    *,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    max_path_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row math of ``replay_conv_state_linear_remap`` (frozen library fn)
    WITHOUT the per-layer bank gather-then-scatter.

    Identical valid-lane/clamp/identity-self-copy semantics; returns flat
    ``(src_rows, dst_rows)`` int64 of length ``B * path_cols`` (empty on the
    library fn's early-return conditions, making the prepared remap the same
    no-op).
    """
    spec_cols = int(spec_state_indices.shape[1])
    path_cols = min(int(accepted_paths.shape[1]), int(max_path_len), spec_cols)
    b = int(num_spec_decodes)
    device = spec_state_indices.device
    if b <= 0 or path_cols <= 0 or spec_cols <= 0:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return empty, empty
    ks = torch.arange(path_cols, device=device, dtype=torch.long)
    lens = num_accepted_tokens.reshape(-1)[:b].to(torch.long).view(b, 1)
    valid = ks.view(1, -1) < lens
    src_col = torch.clamp(
        accepted_paths[:b, :path_cols].to(torch.long), 0, spec_cols - 1
    )
    dst_col = ks.view(1, -1).expand(b, path_cols)
    # Invalid lanes become identity (src == dst): byte-neutral self-copies,
    # matching the library fn / gather kernel's masked-off stores.
    src_col = torch.where(valid, src_col, dst_col)
    window = spec_state_indices[:b].to(torch.long)
    src_rows = window.gather(1, src_col).reshape(-1)
    dst_rows = window.gather(1, dst_col).reshape(-1)
    return src_rows, dst_rows


def replay_conv_state_linear_remap_prepared(
    *,
    conv_state: torch.Tensor,
    src_rows: torch.Tensor,
    dst_rows: torch.Tensor,
) -> None:
    """Per-layer remainder of the page-safe remap: 2 bank ops.

    Gather-then-scatter (class 3): ``index_select`` MATERIALIZES every
    source row before ``index_copy_`` writes any destination row — the same
    race-free in-place overlapping-permutation order as the library fn, and
    conv-view-logical-elements-only (page-safe) by the same torch-op
    construction.
    """
    if int(src_rows.numel()) == 0:
        return
    vals = conv_state.index_select(0, src_rows)
    conv_state.index_copy_(0, dst_rows, vals)


# ---------------------------------------------------------------------------
# TEST-ONLY legacy reference replicas (the byte A/B's Arm A).
# Pinned to the patcher's conv_replacement text / the frozen library fns by
# tests/test_fr13_tree_conv_fused_wiring.py substring asserts and anchored
# to LIVE legacy bytes by the capture-payload A/B arm. NEVER wire these into
# serving.
# ---------------------------------------------------------------------------


def legacy_tree_conv_taps_acc_reference(
    *,
    window: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    x: torch.Tensor,
    tap_dtype: torch.dtype,
) -> torch.Tensor:
    """Verbatim replica: per-col ``_fr11_conv_tap_product`` accumulation
    (native bf16 taps arm) with the legacy
    ``bias.to(f32).unsqueeze(0).expand_as(x.float()).clone()`` seed."""
    if bias is None:
        acc = torch.zeros_like(x, dtype=torch.float32)
    else:
        acc = bias.to(torch.float32).unsqueeze(0).expand_as(x.float()).clone()
    width = int(window.size(1))
    for col in range(width):
        w_cast = conv_weights[:, col].to(tap_dtype).unsqueeze(0)
        acc = acc + (
            (window[:, col, :].to(tap_dtype) * w_cast)
            .to(tap_dtype)
            .to(torch.float32)
        )
    return acc


def legacy_tree_conv_state_rows_reference(
    *,
    x: torch.Tensor,
    prior_window: torch.Tensor,
    path_node_tensors: list[torch.Tensor],
    state_len: int,
) -> torch.Tensor:
    """Verbatim replica of the patcher's per-node write-back loop."""
    node_state_rows = []
    for node_i in range(len(path_node_tensors)):
        node_path = path_node_tensors[node_i]
        node_x = x.index_select(0, node_path)
        node_state_source = torch.cat(
            (prior_window.transpose(0, 1), node_x),
            dim=0,
        )
        node_state_source = torch.cat(
            (
                node_state_source,
                x.new_zeros((int(state_len), int(x.size(1)))),
            ),
            dim=0,
        )
        node_store_idx = node_path.numel() + torch.arange(
            state_len, dtype=torch.long, device=x.device
        )
        node_state_rows.append(
            node_state_source.index_select(0, node_store_idx).transpose(0, 1)
        )
    return torch.stack(node_state_rows, dim=0)


def legacy_gather_committed_path_conv_prior_reference(
    *,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    num_spec_decodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verbatim replica of ``fr10_gdn_tree_kernel.
    gather_committed_path_conv_prior`` (that module imports triton at top,
    so CPU hosts compare against this replica; the GPU arm compares against
    the real library fn too)."""
    b = int(num_spec_decodes)
    spec_cols = int(spec_state_indices.size(-1))
    device = spec_state_indices.device
    if accepted_paths is None or num_accepted_tokens is None:
        read_node_cols = torch.zeros((b, 1), dtype=torch.long, device=device)
    else:
        lens = num_accepted_tokens[:b].to(torch.long).view(-1, 1)
        path_cols = torch.clamp(
            lens - 1, min=0, max=int(accepted_paths.size(-1)) - 1
        )
        read_node_cols = accepted_paths[:b].to(torch.long).gather(1, path_cols)
        read_node_cols = torch.where(
            lens > 0, read_node_cols, torch.zeros_like(read_node_cols)
        )
        read_node_cols = torch.clamp(read_node_cols, min=0, max=spec_cols - 1)
    bank_rows = spec_state_indices[:b].to(torch.long).gather(1, read_node_cols)
    prior_state_bank = torch.index_select(
        conv_state, 0, bank_rows.reshape(-1)
    )
    return read_node_cols, bank_rows, prior_state_bank


# ---------------------------------------------------------------------------
# Full per-layer drivers (T5 synthetic pipeline A/B; also CPU data movement).
# Op ORDER mirrors the live conv_replacement exactly: committed-prior
# snapshot (pre-remap) -> page-safe remap -> per-request window/taps/
# activation/out-slice/write-back/index_copy_.
# ---------------------------------------------------------------------------


def legacy_tree_conv_layer_reference(
    *,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    x: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    source_flat: torch.Tensor,
    path_node_tensors: list[torch.Tensor],
    num_spec_decodes: int,
    tree_n: int,
    width: int,
    state_len: int,
    activation_fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from lumo_flywheel_serving.fr13_replay_conv_remap import (
        replay_conv_state_linear_remap,
    )

    (
        read_cols,
        bank_rows,
        prior_bank,
    ) = legacy_gather_committed_path_conv_prior_reference(
        conv_state=conv_state,
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        num_accepted_tokens=accepted_lens,
        num_spec_decodes=num_spec_decodes,
    )
    replay_conv_state_linear_remap(
        conv_state=conv_state,
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        num_accepted_tokens=accepted_lens,
        num_spec_decodes=num_spec_decodes,
        max_path_len=int(spec_state_indices.size(-1)),
    )
    prior_cols = torch.arange(width - 1, dtype=torch.long, device=x.device)
    out = torch.empty_like(x)
    for b in range(int(num_spec_decodes)):
        start = b * tree_n
        end = start + tree_n
        xb = x[start:end]
        prior_window = prior_bank[b].index_select(1, prior_cols)
        source = torch.cat((prior_window.transpose(0, 1), xb), dim=0)
        window = source.index_select(0, source_flat).view(
            tree_n, width, xb.size(1)
        )
        acc = legacy_tree_conv_taps_acc_reference(
            window=window,
            conv_weights=conv_weights,
            bias=bias,
            x=xb,
            tap_dtype=x.dtype,
        )
        out[start:end] = activation_fn(acc)
        new_state = legacy_tree_conv_state_rows_reference(
            x=xb,
            prior_window=prior_window,
            path_node_tensors=path_node_tensors,
            state_len=state_len,
        ).to(dtype=conv_state.dtype)
        conv_state.index_copy_(
            0,
            spec_state_indices[b, :tree_n].to(torch.long),
            new_state,
        )
    return out, read_cols, bank_rows


def fused_tree_conv_layer(
    *,
    conv_state: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    x: torch.Tensor,
    conv_weights: torch.Tensor,
    bias: torch.Tensor | None,
    source_flat: torch.Tensor,
    state_src: torch.Tensor,
    zero_row: torch.Tensor,
    num_spec_decodes: int,
    tree_n: int,
    width: int,
    state_len: int,
    activation_fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    read_cols, bank_rows = prepare_committed_path_conv_rows(
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        num_accepted_tokens=accepted_lens,
        num_spec_decodes=num_spec_decodes,
    )
    src_rows, dst_rows = prepare_replay_conv_remap_rows(
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        num_accepted_tokens=accepted_lens,
        num_spec_decodes=num_spec_decodes,
        max_path_len=int(spec_state_indices.size(-1)),
    )
    prior_bank = gather_committed_path_conv_prior_prepared(
        conv_state=conv_state, bank_rows=bank_rows
    )
    replay_conv_state_linear_remap_prepared(
        conv_state=conv_state, src_rows=src_rows, dst_rows=dst_rows
    )
    prior_cols = torch.arange(width - 1, dtype=torch.long, device=x.device)
    out = torch.empty_like(x)
    for b in range(int(num_spec_decodes)):
        start = b * tree_n
        end = start + tree_n
        xb = x[start:end]
        prior_window = prior_bank[b].index_select(1, prior_cols)
        source_z = fused_tree_conv_source(
            prior_window=prior_window, x=xb, zero_row=zero_row
        )
        window = source_z.index_select(0, source_flat).view(
            tree_n, width, xb.size(1)
        )
        acc = fused_tree_conv_taps_acc(
            window=window, conv_weights=conv_weights, bias=bias
        )
        out[start:end] = activation_fn(acc)
        new_state = fused_tree_conv_state_rows(
            source_z=source_z,
            state_src=state_src,
            tree_n=tree_n,
            state_len=state_len,
        ).to(dtype=conv_state.dtype)
        conv_state.index_copy_(
            0,
            spec_state_indices[b, :tree_n].to(torch.long),
            new_state,
        )
    return out, read_cols, bank_rows

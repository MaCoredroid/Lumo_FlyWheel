"""FR13_TREE_CONV_FUSED (FIX-3) byte A/B — int-view equality, NEVER atol.

Class-10 bar: the fused tree causal-conv emulation must be BIT-EXACT to the
legacy emulation — same bf16 tap order, same cast boundaries (bf16 taps were
THE FR11-overturning fix), same accumulation order, same silu/bias op order.
Math-correct != bit-exact; per-stage tolerances are BANNED. Views: bf16 ->
.view(torch.int16), fp32 -> .view(torch.int32), index rows -> int64 equal.

CPU-runnable arms (this host, no GPU):
  T1 index-composition identity (static state_src table vs the legacy loop's
     index math via unique-row-id value tensors) + appended-zero-row window
     invariance,
  T2 write-back data movement (legacy loop vs fused static gather),
  T3 taps/bias accumulation (legacy per-col loop vs batched mul + explicit
     ordered adds; the CPU silu fallback is NOT ex2 so activation equality is
     GPU-only — both arms call the same fn, making CPU equality
     data-movement-meaningful only),
  T4 prepared rows (committed-prior + page-safe remap) vs the frozen library
     fn / its verbatim replica, whole-bank int-view equality including
     untouched rows and the PAGE-BOUNDARY sentinel case.

GPU-gated arms (skipif CUDA+triton; the test_fr13_eager_pack_replay_byte_ab
pattern): T5 full per-layer pipeline synthetic A/B with the real
triton_ex2_silu_bf16, T6 capture-payload A/B (the live-legacy anchor), T7
B=2 disjoint-window batching.

CPU identities are necessary-not-sufficient (class 8/10): the binding gates
are the GPU byte A/B and then the live B=1 same-seed repeat + ON-vs-OFF
served-stream byte identity.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr13_replay_conv_remap import (
    replay_conv_state_linear_remap,
)
from lumo_flywheel_serving.fr13_tree_conv_fused import (
    build_tree_conv_state_src_indices,
    build_tree_conv_window_source_indices,
    fused_tree_conv_layer,
    fused_tree_conv_source,
    fused_tree_conv_state_rows,
    fused_tree_conv_taps_acc,
    gather_committed_path_conv_prior_prepared,
    legacy_gather_committed_path_conv_prior_reference,
    legacy_tree_conv_layer_reference,
    legacy_tree_conv_state_rows_reference,
    legacy_tree_conv_taps_acc_reference,
    prepare_committed_path_conv_rows,
    prepare_replay_conv_remap_rows,
    replay_conv_state_linear_remap_prepared,
    tree_paths_from_parent,
)

try:  # pragma: no cover - host venv has no triton
    import triton  # noqa: F401

    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

_GPU = pytest.mark.skipif(
    not (_TRITON_OK and torch.cuda.is_available()),
    reason="FR13_TREE_CONV_FUSED GPU byte A/B requires CUDA + triton",
)


# ---------------------------------------------------------------------------
# Topologies: the exact deployed trees read from the launch script at test
# time (capture-once rule for configs), plus branchy + single-node.
# ---------------------------------------------------------------------------


def _parent_from_tree_choices(choices) -> list[int]:
    """The builder's sorted-index parent derivation (patcher init replica)."""
    tree_choices = sorted(choices, key=lambda p: (len(p), p))
    index = {choice: i + 1 for i, choice in enumerate(tree_choices)}
    parent = [-1]
    for choice in tree_choices:
        parent.append(0 if len(choice) == 1 else index[choice[:-1]])
    return parent


def _deployed_cat9_parent() -> list[int]:
    text = (REPO / "scripts" / "fr10_launch_speed_server.sh").read_text()
    match = re.search(r'TREE=\$\{TREE:-"(.*)"\}', text)
    assert match, "deployed TREE default not found in fr10_launch_speed_server.sh"
    choices = ast.literal_eval(match.group(1))
    return _parent_from_tree_choices([tuple(c) for c in choices])


CHAIN5_PARENT = _parent_from_tree_choices(
    [(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]
)
CAT9_PARENT = _deployed_cat9_parent()
BRANCHY_PARENT = _parent_from_tree_choices(
    [(0,), (1,), (0, 0), (0, 1), (0, 0, 0)]
)
SINGLE_PARENT = [-1]

TOPOLOGIES = {
    "chain5": CHAIN5_PARENT,
    "cat9": CAT9_PARENT,
    "branchy": BRANCHY_PARENT,
    "single": SINGLE_PARENT,
}

WIDTHS = (2, 3, 4, 5, 6)
SEEDS = (0, 1, 2)
DIM = 24  # CPU-sized stand-in for 10240 (per-element ops are dim-agnostic)


def _state_lens(width: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((width - 1, 8, 12)))  # dedup, keep order


def _bits16(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.bfloat16
    return t.contiguous().view(torch.int16)


def _bits32(t: torch.Tensor) -> torch.Tensor:
    assert t.dtype == torch.float32
    return t.contiguous().view(torch.int32)


def _bf16_edges(device="cpu") -> torch.Tensor:
    """bf16 edge values: zeros, subnormals, 1ulp neighbors, RNE-tie food,
    silu saturation, infs, NaN."""
    min_sub = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)[0]
    neg_min_sub = torch.tensor([-32767], dtype=torch.int16).view(torch.bfloat16)[0]
    max_sub = torch.tensor([0x007F], dtype=torch.int16).view(torch.bfloat16)[0]
    vals = torch.tensor(
        [
            0.0,
            -0.0,
            float(min_sub),
            float(neg_min_sub),
            float(max_sub),
            1.0,
            1.0078125,  # 1.0 + 1ulp
            0.99609375,  # 1.0 - 1ulp
            1.5,  # RNE-tie-prone products with .5-mantissa weights
            88.0,
            -88.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype=torch.bfloat16,
        device=device,
    )
    # -0.0 survives the python-float round trip via direct bit poke.
    vals[1] = torch.tensor([-32768], dtype=torch.int16).view(torch.bfloat16)[0]
    return vals


def _inject_edges(t: torch.Tensor, seed: int) -> None:
    flat = t.view(-1)
    edges = _bf16_edges(t.device)
    gen = torch.Generator().manual_seed(seed + 977)
    pos = torch.randperm(flat.numel(), generator=gen)[: edges.numel()]
    flat[pos] = edges[: pos.numel()]


def _make_inputs(parent, width, seed, dim=DIM, dtype=torch.bfloat16):
    n = len(parent)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((n, dim), generator=gen).to(dtype)
    prior_window = torch.randn((dim, width - 1), generator=gen).to(dtype)
    conv_weights = torch.randn((dim, width), generator=gen).to(dtype)
    bias = torch.randn((dim,), generator=gen).to(dtype)
    _inject_edges(x, seed)
    _inject_edges(prior_window, seed + 1)
    return x, prior_window, conv_weights, bias


def _path_node_tensors(parent, device="cpu"):
    return [
        torch.tensor(p, dtype=torch.long, device=device)
        for p in tree_paths_from_parent(parent)
    ]


# ---------------------------------------------------------------------------
# T1: index-composition identity + appended-zero-row window invariance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
@pytest.mark.parametrize("width", WIDTHS)
def test_t1_state_src_index_composition_identity(topo: str, width: int) -> None:
    parent = TOPOLOGIES[topo]
    n = len(parent)
    for state_len in _state_lens(width):
        # Unique-row-id value tensors: prior row p carries value p+1, x node
        # i carries width+i, zeros are 0.0 (every NONZERO id distinct from
        # the zero row, so equality on every (node, col) == exhaustive
        # index-composition identity).
        dim = 3
        prior_window = (
            torch.arange(1, width, dtype=torch.float32)
            .view(1, -1)
            .expand(dim, width - 1)
            .contiguous()
        )
        x = (
            torch.arange(width, width + n, dtype=torch.float32)
            .view(-1, 1)
            .expand(n, dim)
            .contiguous()
        )
        legacy = legacy_tree_conv_state_rows_reference(
            x=x,
            prior_window=prior_window,
            path_node_tensors=_path_node_tensors(parent),
            state_len=state_len,
        )
        zero_row = torch.zeros((1, dim), dtype=torch.float32)
        source_z = fused_tree_conv_source(
            prior_window=prior_window, x=x, zero_row=zero_row
        )
        state_src = build_tree_conv_state_src_indices(
            parent=parent, width=width, state_len=state_len, device="cpu"
        )
        fused = fused_tree_conv_state_rows(
            source_z=source_z, state_src=state_src, tree_n=n, state_len=state_len
        )
        assert torch.equal(_bits32(legacy), _bits32(fused)), (
            f"{topo} w={width} sl={state_len}: state_src table != legacy "
            "loop index math"
        )


@pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_t1_zero_row_leaves_window_and_flat_indices_unchanged(
    topo: str, width: int, seed: int
) -> None:
    parent = TOPOLOGIES[topo]
    n = len(parent)
    x, prior_window, _, _ = _make_inputs(parent, width, seed)
    source_legacy = torch.cat((prior_window.transpose(0, 1), x), dim=0)
    zero_row = torch.zeros((1, x.size(1)), dtype=x.dtype)
    source_z = fused_tree_conv_source(
        prior_window=prior_window, x=x, zero_row=zero_row
    )
    source_flat = build_tree_conv_window_source_indices(
        parent=parent, width=width, device="cpu"
    ).reshape(-1)
    assert int(source_flat.max()) < width - 1 + n  # never the appended row
    win_a = source_legacy.index_select(0, source_flat)
    win_b = source_z.index_select(0, source_flat)
    assert torch.equal(_bits16(win_a), _bits16(win_b))


# ---------------------------------------------------------------------------
# T2: write-back data movement, random bf16 + edge injections.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seed", SEEDS)
def test_t2_writeback_rows_byte_identical(topo: str, width: int, seed: int) -> None:
    parent = TOPOLOGIES[topo]
    n = len(parent)
    for state_len in _state_lens(width):
        x, prior_window, _, _ = _make_inputs(parent, width, seed)
        legacy = legacy_tree_conv_state_rows_reference(
            x=x,
            prior_window=prior_window,
            path_node_tensors=_path_node_tensors(parent),
            state_len=state_len,
        )
        zero_row = torch.zeros((1, x.size(1)), dtype=x.dtype)
        source_z = fused_tree_conv_source(
            prior_window=prior_window, x=x, zero_row=zero_row
        )
        state_src = build_tree_conv_state_src_indices(
            parent=parent, width=width, state_len=state_len, device="cpu"
        )
        fused = fused_tree_conv_state_rows(
            source_z=source_z, state_src=state_src, tree_n=n, state_len=state_len
        )
        assert torch.equal(_bits16(legacy), _bits16(fused)), (
            f"{topo} w={width} sl={state_len} seed={seed}"
        )
        # Zero-pad columns are EXACT zeros (the live state_len=12 > width-1
        # branch), bit pattern +0x0000.
        if state_len > width - 1:
            pad = fused[:, :, width - 1 :]
            assert torch.equal(
                _bits16(pad), torch.zeros_like(pad).view(torch.int16)
            )


# ---------------------------------------------------------------------------
# T3: taps/bias accumulation, int32-view of the fp32 acc (pre-silu).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("with_bias", (True, False))
def test_t3_taps_accumulation_byte_identical(
    topo: str, width: int, seed: int, with_bias: bool
) -> None:
    parent = TOPOLOGIES[topo]
    n = len(parent)
    x, prior_window, conv_weights, bias = _make_inputs(parent, width, seed)
    if not with_bias:
        bias = None
    source = torch.cat((prior_window.transpose(0, 1), x), dim=0)
    source_flat = build_tree_conv_window_source_indices(
        parent=parent, width=width, device="cpu"
    ).reshape(-1)
    window = source.index_select(0, source_flat).view(n, width, x.size(1))
    legacy = legacy_tree_conv_taps_acc_reference(
        window=window,
        conv_weights=conv_weights,
        bias=bias,
        x=x,
        tap_dtype=x.dtype,
    )
    fused = fused_tree_conv_taps_acc(
        window=window, conv_weights=conv_weights, bias=bias
    )
    # NaN payloads must match bit-for-bit too: identical per-element op
    # sequences, so int-view equality holds (TAPS_BYTE_EQUAL).
    assert torch.equal(_bits32(legacy), _bits32(fused)), (
        f"{topo} w={width} seed={seed} bias={with_bias}"
    )


def test_t3_fused_taps_dtype_mismatch_fails_loud() -> None:
    parent = CHAIN5_PARENT
    x, prior_window, conv_weights, bias = _make_inputs(parent, 4, 0)
    source = torch.cat((prior_window.transpose(0, 1), x), dim=0)
    source_flat = build_tree_conv_window_source_indices(
        parent=parent, width=4, device="cpu"
    ).reshape(-1)
    window = source.index_select(0, source_flat).view(len(parent), 4, x.size(1))
    with pytest.raises(RuntimeError, match="dtype uniformity"):
        fused_tree_conv_taps_acc(
            window=window, conv_weights=conv_weights.to(torch.float32), bias=bias
        )


# ---------------------------------------------------------------------------
# T4: prepared rows vs frozen library fns; whole-bank + page-boundary
# sentinels.
# ---------------------------------------------------------------------------


def _page_shared_conv_bank(
    bank_rows: int, dim: int, state_len: int, *, pad_elems: int = 7, seed: int = 0
):
    """vLLM MambaSpec page layout replica: conv bank as an as_strided view
    over a padded page; the gap carries sentinel bytes (ssm slice stand-in).
    """
    conv_elems = dim * state_len
    per_page = conv_elems + pad_elems
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn((bank_rows * per_page,), generator=gen).to(torch.bfloat16)
    # Sentinels in the non-conv remainder of every page.
    raw_view = raw.view(bank_rows, per_page)
    raw_view[:, conv_elems:] = torch.tensor(
        float("inf"), dtype=torch.bfloat16
    )
    conv = raw.as_strided((bank_rows, dim, state_len), (per_page, state_len, 1))
    return raw, conv


def _accepted_cases(parent, spec_cols):
    """accepted_len cases: zero-accept, 1, partial, full; spine and
    branch-winner (non-monotone) paths."""
    paths = tree_paths_from_parent(parent)
    spine = max(
        (p for p in paths), key=lambda p: (len(p), [-x for x in p])
    )
    branch = max((p for p in paths), key=lambda p: (len(p), p))
    cases = [("zero", [0] * spec_cols, 0)]
    for name, path in (("spine", spine), ("branch", branch)):
        for ln in sorted({1, max(1, len(path) // 2), len(path)}):
            row = list(path[:ln]) + [0] * (spec_cols - ln)
            cases.append((f"{name}_len{ln}", row[:spec_cols], min(ln, spec_cols)))
    if spec_cols > 9:
        # FIX-2's non-monotone branch-winner shape ([2, 6, 7, 9]): the remap
        # permutation must not assume monotone node ids.
        cases.append(
            ("nonmono_branchwinner", [2, 6, 7, 9] + [0] * (spec_cols - 4), 4)
        )
    return cases


@pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
@pytest.mark.parametrize("b", (1, 2))
def test_t4_prepared_rows_match_frozen_library(topo: str, b: int) -> None:
    parent = TOPOLOGIES[topo]
    n = len(parent)
    spec_cols = n + 1
    dim, state_len = DIM, 12
    bank_rows = b * spec_cols + 3
    for case_name, path_row, ln in _accepted_cases(parent, spec_cols):
        # Disjoint per-request windows (deployed KV-pool semantics).
        spec_idx = torch.stack(
            [
                torch.arange(
                    bi * spec_cols, (bi + 1) * spec_cols, dtype=torch.int32
                )
                for bi in range(b)
            ]
        )
        accepted_paths = torch.tensor(
            [path_row] * b, dtype=torch.int32
        )
        accepted_lens = torch.tensor([ln] * b, dtype=torch.int32)
        if b == 2:
            # Distinct second-request case: zero-accept (root window read).
            accepted_paths[1].zero_()
            accepted_lens[1] = 0

        raw_a, conv_a = _page_shared_conv_bank(bank_rows, dim, state_len, seed=11)
        raw_b, conv_b = _page_shared_conv_bank(bank_rows, dim, state_len, seed=11)
        assert torch.equal(raw_a.view(torch.int16), raw_b.view(torch.int16))

        # Committed-prior: replica (frozen library math) vs prepared split.
        (
            cols_a,
            rows_a,
            prior_a,
        ) = legacy_gather_committed_path_conv_prior_reference(
            conv_state=conv_a,
            spec_state_indices=spec_idx,
            accepted_paths=accepted_paths,
            num_accepted_tokens=accepted_lens,
            num_spec_decodes=b,
        )
        cols_b, rows_b = prepare_committed_path_conv_rows(
            spec_state_indices=spec_idx,
            accepted_paths=accepted_paths,
            num_accepted_tokens=accepted_lens,
            num_spec_decodes=b,
        )
        prior_b = gather_committed_path_conv_prior_prepared(
            conv_state=conv_b, bank_rows=rows_b
        )
        assert torch.equal(cols_a, cols_b), case_name
        assert torch.equal(rows_a, rows_b), case_name
        assert torch.equal(_bits16(prior_a), _bits16(prior_b)), case_name

        # Remap: frozen library fn vs prepared split, WHOLE-PAGE int view
        # (untouched rows + sentinel gap bytes = bleed detection).
        replay_conv_state_linear_remap(
            conv_state=conv_a,
            spec_state_indices=spec_idx,
            accepted_paths=accepted_paths,
            num_accepted_tokens=accepted_lens,
            num_spec_decodes=b,
            max_path_len=spec_cols,
        )
        src_rows, dst_rows = prepare_replay_conv_remap_rows(
            spec_state_indices=spec_idx,
            accepted_paths=accepted_paths,
            num_accepted_tokens=accepted_lens,
            num_spec_decodes=b,
            max_path_len=spec_cols,
        )
        assert int(src_rows.numel()) == b * min(
            accepted_paths.shape[1], spec_cols
        )
        replay_conv_state_linear_remap_prepared(
            conv_state=conv_b, src_rows=src_rows, dst_rows=dst_rows
        )
        assert torch.equal(raw_a.view(torch.int16), raw_b.view(torch.int16)), (
            f"{topo} b={b} {case_name}: whole-page bytes diverged "
            "(conv remap bleed)"
        )
        # Sentinels in the page gap byte-untouched in BOTH arms.
        conv_elems = dim * state_len
        gap_a = raw_a.view(bank_rows, -1)[:, conv_elems:]
        assert bool(torch.isinf(gap_a.float()).all())


# ---------------------------------------------------------------------------
# T5/T6/T7: GPU-gated (real triton_ex2_silu_bf16 / capture payload / B=2).
# ---------------------------------------------------------------------------


def _silu_gpu(out_dtype):
    from lumo_flywheel_serving.fr13_ex2_silu import triton_ex2_silu_bf16

    def _fn(acc):
        return triton_ex2_silu_bf16(acc, out_dtype=out_dtype)

    return _fn


def _run_pipeline_pair(parent, width, state_len, b, seed, device, dim=10240):
    n = len(parent)
    spec_cols = n + 1
    bank_rows = b * spec_cols + 4
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((b * n, dim), generator=gen).to(torch.bfloat16).to(device)
    conv_weights = (
        torch.randn((dim, width), generator=gen).to(torch.bfloat16).to(device)
    )
    bias = torch.randn((dim,), generator=gen).to(torch.bfloat16).to(device)
    _inject_edges(x, seed)
    bank0 = (
        torch.randn((bank_rows, dim, state_len), generator=gen)
        .to(torch.bfloat16)
        .to(device)
    )
    spec_idx = torch.stack(
        [
            torch.arange(bi * spec_cols, (bi + 1) * spec_cols, dtype=torch.int32)
            for bi in range(b)
        ]
    ).to(device)
    paths = tree_paths_from_parent(parent)
    branch = max((p for p in paths), key=lambda p: (len(p), p))
    accepted_paths = torch.zeros((b, spec_cols), dtype=torch.int32)
    accepted_lens = torch.zeros((b,), dtype=torch.int32)
    accepted_paths[0, : len(branch)] = torch.tensor(branch, dtype=torch.int32)
    accepted_lens[0] = len(branch)
    accepted_paths, accepted_lens = accepted_paths.to(device), accepted_lens.to(device)
    source_flat = build_tree_conv_window_source_indices(
        parent=parent, width=width, device=device
    ).reshape(-1)
    state_src = build_tree_conv_state_src_indices(
        parent=parent, width=width, state_len=state_len, device=device
    )
    zero_row = torch.zeros((1, dim), dtype=torch.bfloat16, device=device)
    silu = _silu_gpu(torch.bfloat16)

    bank_a = bank0.clone()
    out_a, cols_a, rows_a = legacy_tree_conv_layer_reference(
        conv_state=bank_a,
        spec_state_indices=spec_idx,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        x=x.clone(),
        conv_weights=conv_weights,
        bias=bias,
        source_flat=source_flat,
        path_node_tensors=_path_node_tensors(parent, device),
        num_spec_decodes=b,
        tree_n=n,
        width=width,
        state_len=state_len,
        activation_fn=silu,
    )
    bank_b = bank0.clone()
    out_b, cols_b, rows_b = fused_tree_conv_layer(
        conv_state=bank_b,
        spec_state_indices=spec_idx,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        x=x.clone(),
        conv_weights=conv_weights,
        bias=bias,
        source_flat=source_flat,
        state_src=state_src,
        zero_row=zero_row,
        num_spec_decodes=b,
        tree_n=n,
        width=width,
        state_len=state_len,
        activation_fn=silu,
    )
    torch.cuda.synchronize()
    return (out_a, cols_a, rows_a, bank_a), (out_b, cols_b, rows_b, bank_b)


@_GPU
@pytest.mark.parametrize("topo", sorted(TOPOLOGIES))
@pytest.mark.parametrize("seed", SEEDS)
def test_t5_full_pipeline_synthetic_gpu_byte_ab(topo: str, seed: int) -> None:
    parent = TOPOLOGIES[topo]
    a, b_ = _run_pipeline_pair(
        parent, width=4, state_len=12, b=1, seed=seed, device="cuda"
    )
    (out_a, cols_a, rows_a, bank_a) = a
    (out_b, cols_b, rows_b, bank_b) = b_
    # Written rows [0, B*tree_n) only; trailing mixed_qkv rows are
    # uninitialized in BOTH live arms (excluded by design, documented).
    assert torch.equal(_bits16(out_a), _bits16(out_b))
    assert torch.equal(cols_a, cols_b) and torch.equal(rows_a, rows_b)
    # FULL bank int-view (untouched rows included = bleed detection).
    assert torch.equal(_bits16(bank_a), _bits16(bank_b))


@_GPU
def test_t7_b2_disjoint_window_batching_gpu_byte_ab() -> None:
    parent = CAT9_PARENT
    a, b_ = _run_pipeline_pair(
        parent, width=4, state_len=12, b=2, seed=7, device="cuda"
    )
    assert torch.equal(_bits16(a[0]), _bits16(b_[0]))
    assert torch.equal(_bits16(a[3]), _bits16(b_[3]))


@_GPU
def test_t6_capture_payload_anchor_gpu_byte_ab() -> None:
    """Live-legacy anchor: feed a captured FR12_SUBKERNEL_CAPTURE
    tree_conv_detail payload (pre_conv + prior windows + weights/bias) to
    both arms. If absent on this host, capture a fresh PAIRED payload first
    (capture-once pinned-prompt rule)."""
    import glob as _glob
    import os as _os

    cand = _os.environ.get("FR13_TCF_PAYLOAD")
    paths = [cand] if cand else sorted(
        _glob.glob(str(REPO / "output" / "**" / "*subkernel*.pt"), recursive=True)
    ) + sorted(_glob.glob("/logs/*subkernel*.pt"))
    payload = None
    for p in paths:
        if not p or not _os.path.exists(p):
            continue
        try:
            cand_payload = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        detail_cand = (
            cand_payload.get("meta", {}).get("tree_conv_detail")
            if isinstance(cand_payload, dict)
            else None
        )
        # Older FR12 captures lack the selected-window fields this anchor
        # consumes; require the exact keys (conv_bias may be None but the
        # key must exist) so the sorted scan lands on a live FR13 capture.
        if isinstance(detail_cand, dict) and all(
            k in detail_cand
            for k in (
                "window_selected",
                "pre_conv_selected",
                "conv_weights",
                "conv_bias",
            )
        ):
            payload = cand_payload
            break
    if payload is None:
        pytest.skip(
            "no FR12_SUBKERNEL_CAPTURE tree_conv_detail payload found; "
            "capture a fresh PAIRED payload on the GPU host "
            "(FR13_TCF_PAYLOAD=<path> to pin one)"
        )
    detail = payload["meta"]["tree_conv_detail"]
    device = torch.device("cuda")
    # Stored fp32 copies of bf16 live values: the bf16 round-trip is exact.
    window = detail["window_selected"].to(torch.bfloat16).to(device)
    conv_weights = detail["conv_weights"].to(torch.bfloat16).to(device)
    bias = (
        None
        if detail["conv_bias"] is None
        else detail["conv_bias"].to(torch.bfloat16).to(device)
    )
    x_rows = detail["pre_conv_selected"].to(torch.bfloat16).to(device)
    legacy = legacy_tree_conv_taps_acc_reference(
        window=window,
        conv_weights=conv_weights,
        bias=bias,
        x=x_rows,
        tap_dtype=torch.bfloat16,
    )
    fused = fused_tree_conv_taps_acc(
        window=window, conv_weights=conv_weights, bias=bias
    )
    assert torch.equal(_bits32(legacy.cpu()), _bits32(fused.cpu()))
    silu = _silu_gpu(torch.bfloat16)
    assert torch.equal(
        _bits16(silu(legacy).cpu()), _bits16(silu(fused).cpu())
    )


@_GPU
def test_t8_state_src_build_capture_legal_byte_ab() -> None:
    """T8 (class 6, live-gate fix 2026-06-12): the static-table fallback must
    be LEGAL inside CUDA-graph capture and byte-identical on replay.

    The cu130 nightly profiles cudagraph memory BEFORE any eager tree
    forward, so the first fused forward of a boot runs INSIDE stream capture
    with a cold table cache; the original pageable
    ``torch.tensor(list, device="cuda")`` raised "Cannot copy between CPU
    and CUDA tensors during CUDA graph capture" and killed the cat9_on boot.
    The fix stages through RETAINED pinned host memory. This test captures a
    graph whose body builds the table and bakes a copy into a persistent out
    buffer, then replays TWICE (after trashing the buffer) and asserts exact
    int64 equality with the eager build — proving both capture legality and
    replay-stable values (the baked H2D re-reads the retained staging).
    """
    parent = CAT9_PARENT
    eager = build_tree_conv_state_src_indices(
        parent=parent, width=4, state_len=12, device="cuda"
    )
    out = torch.empty_like(eager)
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        captured = build_tree_conv_state_src_indices(
            parent=parent, width=4, state_len=12, device="cuda"
        )
        out.copy_(captured)
    for _ in range(2):
        out.fill_(-1)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, eager)

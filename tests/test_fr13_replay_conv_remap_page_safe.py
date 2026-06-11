"""FR13_REPLAY_ROUTE page-safe conv remap — CPU unit tests.

The fixture reproduces vLLM's MambaSpec kv-cache layout byte-for-byte
(gpu_model_runner._reshape_kv_cache_tensors, MambaSpec branch): conv (kv[0])
and ssm (kv[1]) are ``as_strided`` views over the SAME raw page tensor with
``stride(0) == num_element_per_page`` and the ssm storage offset right after
the conv slice. This is the layout under which the frozen Triton remap's
``state.stride(0)`` copy extent drags the co-resident ssm slice along
(boundary-trace root cause: 581/581 corrupted-window byte predictions).

The contract proven here: ``replay_conv_state_linear_remap`` applies EXACTLY
the gather-kernel permutation to the conv view's logical elements and leaves
every other byte of the raw page tensor untouched (ssm slice + page padding +
non-window rows).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from lumo_flywheel_serving.fr13_replay_conv_remap import (
    replay_conv_state_linear_remap,
)

HELPER = Path("src/lumo_flywheel_serving/fr13_replay_conv_remap.py")


def _page_shared_bank(
    num_blocks: int,
    conv_shape: tuple[int, ...],
    ssm_shape: tuple[int, ...],
    *,
    pad_elems: int = 7,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
):
    """Mirror gpu_model_runner._reshape_kv_cache_tensors (MambaSpec branch)."""
    conv_elems = math.prod(conv_shape)
    ssm_elems = math.prod(ssm_shape)
    num_element_per_page = conv_elems + ssm_elems + pad_elems
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn((num_blocks * num_element_per_page,), generator=gen).to(dtype)
    views = []
    storage_offset = 0
    for shape in (conv_shape, ssm_shape):
        target_shape = (num_blocks, *shape)
        stride = torch.empty(target_shape).stride()
        target_stride = (num_element_per_page, *stride[1:])
        views.append(
            torch.as_strided(
                raw,
                size=target_shape,
                stride=target_stride,
                storage_offset=storage_offset,
            )
        )
        storage_offset += stride[0]
    conv_view, ssm_view = views
    return raw, conv_view, ssm_view, num_element_per_page


def _reference_remap(
    conv_view: torch.Tensor,
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_spec_decodes: int,
    max_path_len: int,
) -> None:
    """Pure-python transcription of _linear_remap_rows_gather_kernel."""
    spec_cols = int(spec_state_indices.shape[1])
    path_cols = min(int(accepted_paths.shape[1]), int(max_path_len), spec_cols)
    for b in range(num_spec_decodes):
        length = int(num_accepted_tokens[b])
        # Gather-then-scatter: all sources loaded before any store.
        loaded = []
        for k in range(path_cols):
            if k >= length:
                continue
            src_col = max(0, min(int(accepted_paths[b, k]), spec_cols - 1))
            src_bank = int(spec_state_indices[b, src_col])
            loaded.append((k, conv_view[src_bank].clone()))
        for k, val in loaded:
            dst_bank = int(spec_state_indices[b, k])
            conv_view[dst_bank].copy_(val)


def _random_case(seed: int, *, dtype: torch.dtype = torch.float32):
    gen = torch.Generator().manual_seed(seed)
    b = int(torch.randint(1, 4, (1,), generator=gen))
    spec_cols = 10
    path_cols = 10
    num_blocks = 64
    raw, conv_view, ssm_view, page = _page_shared_bank(
        num_blocks, (6, 3), (2, 4, 4), dtype=dtype, seed=seed + 1000
    )
    # Disjoint per-request windows of unique physical blocks (block tables
    # never alias live blocks across requests).
    perm = torch.randperm(num_blocks, generator=gen)[: b * spec_cols]
    windows = perm.reshape(b, spec_cols).to(torch.int32)
    lens = torch.zeros((b,), dtype=torch.int32)
    paths = torch.full((b, path_cols), -7, dtype=torch.int32)  # garbage filler
    for row in range(b):
        length = int(torch.randint(0, 6, (1,), generator=gen))
        lens[row] = length
        spine = bool(torch.randint(0, 2, (1,), generator=gen))
        node = 0
        for k in range(length):
            if spine:
                node = k + 1
            else:
                # branch-shaped: strictly increasing but skipping nodes
                node = min(node + int(torch.randint(1, 3, (1,), generator=gen)), spec_cols - 1)
            paths[row, k] = node
    return raw, conv_view, ssm_view, windows, paths, lens, b, spec_cols


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7])
def test_remap_matches_kernel_permutation_and_touches_only_conv_bytes(
    seed: int, dtype: torch.dtype
) -> None:
    raw, conv_view, ssm_view, windows, paths, lens, b, spec_cols = _random_case(
        seed, dtype=dtype
    )
    # Build the expected raw page bytes: apply the reference permutation to a
    # CLONED raw's conv view only. Any deviation anywhere in the page (ssm
    # slice, padding, non-window rows) fails the whole-tensor compare.
    expected_raw = raw.clone()
    exp_conv = torch.as_strided(
        expected_raw,
        size=conv_view.shape,
        stride=conv_view.stride(),
        storage_offset=conv_view.storage_offset(),
    )
    _reference_remap(exp_conv, windows, paths, lens, b, spec_cols)

    ssm_before = ssm_view.clone()
    replay_conv_state_linear_remap(
        conv_state=conv_view,
        spec_state_indices=windows,
        accepted_paths=paths,
        num_accepted_tokens=lens,
        num_spec_decodes=b,
        max_path_len=spec_cols,
    )
    assert torch.equal(raw, expected_raw)
    # Explicit headline assertion: the co-resident ssm slice is byte-frozen.
    assert torch.equal(ssm_view, ssm_before)


def test_zero_accept_is_byte_noop_on_the_whole_page_tensor() -> None:
    raw, conv_view, _, windows, paths, lens, b, spec_cols = _random_case(11)
    lens.zero_()
    before = raw.clone()
    replay_conv_state_linear_remap(
        conv_state=conv_view,
        spec_state_indices=windows,
        accepted_paths=paths,
        num_accepted_tokens=lens,
        num_spec_decodes=b,
        max_path_len=spec_cols,
    )
    assert torch.equal(raw, before)


def test_spine_overlap_permutation_is_gather_then_scatter_exact() -> None:
    # Spine path [1..L] maps src cols [1..L] onto dst cols [0..L-1]: the
    # racy-overlap shape. Destination col k must receive the PRE-remap content
    # of col k+1, not a cascaded copy.
    raw, conv_view, ssm_view, page = _page_shared_bank(16, (6, 3), (2, 4, 4), seed=5)
    windows = torch.tensor([[3, 9, 4, 1, 12, 7, 2, 8, 5, 0]], dtype=torch.int32)
    length = 4
    paths = torch.full((1, 10), 0, dtype=torch.int32)
    paths[0, :length] = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    lens = torch.tensor([length], dtype=torch.int32)
    pre = {int(c): conv_view[int(windows[0, c])].clone() for c in range(10)}
    ssm_before = ssm_view.clone()
    replay_conv_state_linear_remap(
        conv_state=conv_view,
        spec_state_indices=windows,
        accepted_paths=paths,
        num_accepted_tokens=lens,
        num_spec_decodes=1,
        max_path_len=10,
    )
    for k in range(length):
        assert torch.equal(conv_view[int(windows[0, k])], pre[k + 1]), k
    for c in range(length, 10):
        assert torch.equal(conv_view[int(windows[0, c])], pre[c]), c
    assert torch.equal(ssm_view, ssm_before)


def test_whole_page_copy_would_have_clobbered_ssm() -> None:
    # Regression demonstration of the root cause at layout level: the conv
    # view's row stride exceeds its logical row payload (page-sharing), so a
    # stride(0)-extent copy (the frozen Triton remap's contract) necessarily
    # overwrites the co-resident ssm row. The page-safe helper must therefore
    # never derive its extent from stride(0).
    raw, conv_view, ssm_view, page = _page_shared_bank(8, (6, 3), (2, 4, 4))
    assert conv_view.stride(0) == page
    assert conv_view.stride(0) > conv_view[0].numel()
    # The two views alias one storage.
    assert conv_view.untyped_storage().data_ptr() == ssm_view.untyped_storage().data_ptr()
    # Simulate the stride(0)-extent row copy 1 -> 0 (what the Triton kernel
    # does): ssm row 0 is clobbered by ssm row 1.
    sim = raw.clone()
    sim[0:page] = sim[page : 2 * page]
    sim_ssm = torch.as_strided(
        sim, ssm_view.shape, ssm_view.stride(), ssm_view.storage_offset()
    )
    assert not torch.equal(sim_ssm[0], ssm_view[0])
    assert torch.equal(sim_ssm[0], ssm_view[1])


def test_validation_raises_loud() -> None:
    raw, conv_view, _, windows, paths, lens, b, spec_cols = _random_case(21)
    with pytest.raises(ValueError):
        replay_conv_state_linear_remap(
            conv_state=conv_view[0, 0],
            spec_state_indices=windows,
            accepted_paths=paths,
            num_accepted_tokens=lens,
            num_spec_decodes=b,
            max_path_len=spec_cols,
        )
    with pytest.raises(ValueError):
        replay_conv_state_linear_remap(
            conv_state=conv_view,
            spec_state_indices=windows,
            accepted_paths=paths[:0],
            num_accepted_tokens=lens,
            num_spec_decodes=max(b, 1),
            max_path_len=spec_cols,
        )
    with pytest.raises(ValueError):
        replay_conv_state_linear_remap(
            conv_state=conv_view,
            spec_state_indices=windows,
            accepted_paths=paths,
            num_accepted_tokens=lens[:0],
            num_spec_decodes=max(b, 1),
            max_path_len=spec_cols,
        )


def test_helper_is_capture_safe_and_kernel_file_untouched() -> None:
    import ast

    tree = ast.parse(HELPER.read_text())
    # No host syncs / data-dependent shapes in the helper body (it runs
    # inside the potentially CUDA-graph-captured forward), and stride() must
    # never be the copy-extent basis (the root cause it replaces).
    banned_attrs = {
        "item",
        "tolist",
        "cpu",
        "numpy",
        "synchronize",
        "nonzero",
        "masked_select",
        "unique",
        "stride",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in banned_attrs, node.attr
    # The helper is pure torch: it must not import the frozen kernel module
    # or triton.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert "fr10_gdn_tree_kernel" not in module
            assert "triton" not in module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "triton" not in alias.name
                assert "fr10_gdn_tree_kernel" not in alias.name

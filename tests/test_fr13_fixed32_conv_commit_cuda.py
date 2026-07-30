from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("triton")

from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA and Triton"
)

DEPLOYED_CONV_SHAPE = (10_240, 34)
DEPLOYED_PAGE_ELEMS = 2_097_152
GAP_SENTINEL = -4096.0


def _page_shared_bank(
    *,
    rows: int,
    seed: int,
    conv_shape: tuple[int, int] = (3, 2),
    channel_contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ssm_shape = (2, 2)
    page_elems = math.prod(conv_shape) + math.prod(ssm_shape) + 5
    raw = (
        torch.arange(
            rows * page_elems,
            dtype=torch.float32,
            device="cuda",
        )
        .add_(seed * 1000)
        .to(torch.bfloat16)
    )
    if channel_contiguous:
        # vLLM stores SD as [row, state_len, dim], then GDN consumes its
        # [row, dim, state_len] transpose.
        conv = torch.as_strided(
            raw,
            size=(rows, conv_shape[1], conv_shape[0]),
            stride=(page_elems, conv_shape[0], 1),
        ).transpose(1, 2)
    else:
        conv = torch.as_strided(
            raw,
            size=(rows, *conv_shape),
            stride=(page_elems, conv_shape[1], 1),
        )
    ssm = torch.as_strided(
        raw,
        size=(rows, *ssm_shape),
        stride=(page_elems, ssm_shape[1], 1),
        storage_offset=math.prod(conv_shape),
    )
    return raw, conv, ssm


def _install_preseed(
    monkeypatch: pytest.MonkeyPatch,
    *,
    channel_contiguous: bool,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_CONV_PREGATHER", {})
    monkeypatch.setattr(kernel, "_FR13_FIXED32_CONV_SSI_GROUPS", {})
    order = tuple(f"gdn.{index:02d}" for index in range(48))
    raws = []
    banks = []
    for index in range(48):
        raw, bank, _ = _page_shared_bank(
            rows=4,
            seed=index,
            conv_shape=DEPLOYED_CONV_SHAPE,
            channel_contiguous=channel_contiguous,
        )
        raws.append(raw)
        banks.append(bank)
    bank_tuple = tuple(banks)
    assert bank_tuple[0][0].numel() == 348_160
    expected_inner_stride = (
        (1, DEPLOYED_CONV_SHAPE[0])
        if channel_contiguous
        else (DEPLOYED_CONV_SHAPE[1], 1)
    )
    assert bank_tuple[0].stride()[1:] == expected_inner_stride
    for group_index in range(3):
        names = order[group_index * 16 : (group_index + 1) * 16]
        kernel.register_fixed32_conv_col0_ssi_group(
            layer_names=names,
            spec_state_indices=torch.zeros(
                (4, 1), dtype=torch.int32, device="cuda"
            ),
            max_batch_size=4,
        )
    commit_ssi = torch.zeros(
        (48, 4, 32), dtype=torch.int32, device="cuda"
    )
    accepted_paths = torch.zeros(
        (4, 16), dtype=torch.int32, device="cuda"
    )
    accepted_lens = torch.zeros((4,), dtype=torch.int32, device="cuda")
    before_preseed = tuple(raw.clone() for raw in raws)
    contract = kernel.preseed_fixed32_conv_col0_pregather(
        conv_banks=bank_tuple,
        layer_order=order,
        max_batch_size=4,
        commit_spec_state_indices=commit_ssi,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
    )
    assert contract["commit_route"] == "fixed32_two_launch_col0"
    assert contract["commit_bank_nonoverlap"] is True
    assert kernel.audit_fixed32_conv_commit_lease()["lease_audited"] is True
    assert all(
        torch.equal(raw, before)
        for raw, before in zip(raws, before_preseed, strict=True)
    )
    return (
        tuple(raws),
        bank_tuple,
        commit_ssi,
        accepted_paths,
        accepted_lens,
    )


def _bank_views_for_raw_clones(
    raws: tuple[torch.Tensor, ...],
    banks: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.as_strided(
            raw,
            size=bank.shape,
            stride=bank.stride(),
            storage_offset=bank.storage_offset(),
        )
        for raw, bank in zip(raws, banks, strict=True)
    )


def _legacy_commit_reference(
    *,
    banks: tuple[torch.Tensor, ...],
    spec_state_indices: torch.Tensor,
    accepted_paths: torch.Tensor,
    accepted_lens: torch.Tensor,
    batch: int,
) -> None:
    """Transcribe the removed per-layer index_select/index_copy_ route."""
    rows = torch.arange(batch, device="cuda")
    accepted = accepted_lens[:batch]
    leaf_pos = (accepted - 1).clamp(min=0)
    leaf_node = accepted_paths[:batch].gather(
        1, leaf_pos.view(-1, 1).to(torch.long)
    ).view(-1)
    leaf_node = torch.where(
        accepted > 0, leaf_node, torch.zeros_like(leaf_node)
    ).clamp(0, int(spec_state_indices.shape[2]) - 1).to(torch.long)
    for layer, bank in enumerate(banks):
        layer_ssi = spec_state_indices[layer]
        dst = layer_ssi[:batch, 0].to(torch.long)
        src = layer_ssi[rows, leaf_node].to(torch.long)
        bank.index_copy_(0, dst, bank.index_select(0, src))


def _exact_deployed_page_bank(
    *, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.full(
        (rows * DEPLOYED_PAGE_ELEMS,),
        GAP_SENTINEL,
        dtype=torch.bfloat16,
        device="cuda",
    )
    channels, state_length = DEPLOYED_CONV_SHAPE
    bank = torch.as_strided(
        raw,
        size=(rows, state_length, channels),
        stride=(DEPLOYED_PAGE_ELEMS, channels, 1),
    ).transpose(1, 2)
    values = (
        torch.arange(
            channels * state_length, dtype=torch.int32, device="cuda"
        )
        .remainder_(251)
        .to(torch.bfloat16)
        .view(channels, state_length)
    )
    for row in range(rows):
        bank[row].copy_(values + row * 256)
    return raw, bank


def _assert_deployed_page_gaps_are_sentinel(
    raw: torch.Tensor, *, rows: int
) -> None:
    row_elems = math.prod(DEPLOYED_CONV_SHAPE)
    gaps = raw.view(rows, DEPLOYED_PAGE_ELEMS)[:, row_elems:]
    assert torch.all(gaps == GAP_SENTINEL)


def test_exact_outer_stride_high_row_and_page_gaps_are_kernel_exact() -> None:
    rows = 8
    high_row = rows - 1
    raw, bank = _exact_deployed_page_bank(rows=rows)
    assert bank.shape == (rows, *DEPLOYED_CONV_SHAPE)
    assert bank.stride() == (
        DEPLOYED_PAGE_ELEMS,
        1,
        DEPLOYED_CONV_SHAPE[0],
    )
    _assert_deployed_page_gaps_are_sentinel(raw, rows=rows)

    row_elems = math.prod(DEPLOYED_CONV_SHAPE)
    block = 1024
    grid = (1, 1, (row_elems + block - 1) // block)
    off16 = torch.zeros((1,), dtype=torch.int64, device="cuda")
    ssi = torch.tensor([[high_row]], dtype=torch.int32, device="cuda")
    ssi_ptrs = torch.tensor(
        [int(ssi.data_ptr())], dtype=torch.int64, device="cuda"
    )
    ssi_strides = torch.tensor(
        [int(ssi.stride(0))], dtype=torch.int64, device="cuda"
    )
    staging = torch.empty(
        (1, 1, row_elems), dtype=bank.dtype, device="cuda"
    )
    kernel._fr13_conv_col0_pregather_kernel[grid](
        bank,
        off16,
        ssi_ptrs,
        ssi_strides,
        staging,
        staging.stride(0),
        staging.stride(1),
        bank.stride(0),
        bank.stride(1),
        bank.stride(2),
        ROW_ELEMS=row_elems,
        CONV_L=DEPLOYED_CONV_SHAPE[1],
        ELEM_BYTES=bank.element_size(),
        B=1,
        BLOCK=block,
    )
    assert torch.equal(staging[0, 0], bank[high_row].reshape(-1))
    _assert_deployed_page_gaps_are_sentinel(raw, rows=rows)

    spec_state_indices = torch.zeros(
        (1, 1, 2), dtype=torch.int32, device="cuda"
    )
    spec_state_indices[0, 0] = torch.tensor(
        [0, high_row], dtype=torch.int32, device="cuda"
    )
    accepted_paths = torch.zeros(
        (1, 16), dtype=torch.int32, device="cuda"
    )
    accepted_paths[0, 0] = 1
    accepted_lens = torch.ones((1,), dtype=torch.int32, device="cuda")
    expected_raw = raw.clone()
    expected_bank = torch.as_strided(
        expected_raw, size=bank.shape, stride=bank.stride()
    )
    _legacy_commit_reference(
        banks=(expected_bank,),
        spec_state_indices=spec_state_indices,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        batch=1,
    )
    kernel._fr13_fixed32_conv_commit_gather_kernel[grid](
        bank,
        off16,
        spec_state_indices,
        accepted_paths,
        accepted_lens,
        staging,
        staging.stride(0),
        staging.stride(1),
        spec_state_indices.stride(0),
        spec_state_indices.stride(1),
        spec_state_indices.stride(2),
        accepted_paths.stride(0),
        accepted_paths.stride(1),
        accepted_lens.stride(0),
        bank.stride(0),
        ROW_ELEMS=row_elems,
        ELEM_BYTES=bank.element_size(),
        SPEC_COLS=spec_state_indices.shape[2],
        PATH_COLS=accepted_paths.shape[1],
        B=1,
        BLOCK=block,
    )
    kernel._fr13_fixed32_conv_commit_scatter_kernel[grid](
        bank,
        off16,
        spec_state_indices,
        staging,
        staging.stride(0),
        staging.stride(1),
        spec_state_indices.stride(0),
        spec_state_indices.stride(1),
        spec_state_indices.stride(2),
        bank.stride(0),
        ROW_ELEMS=row_elems,
        ELEM_BYTES=bank.element_size(),
        B=1,
        BLOCK=block,
    )
    assert torch.equal(raw, expected_raw)
    _assert_deployed_page_gaps_are_sentinel(raw, rows=rows)
    assert not torch.equal(staging[0, 0], bank[high_row].reshape(-1))

    kernel._fr13_conv_col0_pregather_kernel[grid](
        bank,
        off16,
        ssi_ptrs,
        ssi_strides,
        staging,
        staging.stride(0),
        staging.stride(1),
        bank.stride(0),
        bank.stride(1),
        bank.stride(2),
        ROW_ELEMS=row_elems,
        CONV_L=DEPLOYED_CONV_SHAPE[1],
        ELEM_BYTES=bank.element_size(),
        B=1,
        BLOCK=block,
    )
    assert torch.equal(staging[0, 0], bank[high_row].reshape(-1))
    _assert_deployed_page_gaps_are_sentinel(raw, rows=rows)


@pytest.mark.parametrize(
    "channel_contiguous",
    (False, True),
    ids=("last-dimension-contiguous", "canonical-channel-contiguous"),
)
def test_deployed_layout_pregather_and_two_launch_commit_are_page_exact(
    monkeypatch: pytest.MonkeyPatch,
    channel_contiguous: bool,
) -> None:
    raws, banks, commit_ssi, paths, lens = _install_preseed(
        monkeypatch, channel_contiguous=channel_contiguous
    )

    state = kernel._FR13_FIXED32_CONV_PREGATHER["state"]
    seen_sources: set[int] = set()
    for source in state["ssi_sources"]:
        if id(source) in seen_sources:
            continue
        seen_sources.add(id(source))
        source[:, 0].copy_(
            torch.tensor([3, 1, 2, 0], dtype=torch.int32, device="cuda")
        )
    expected_staging = tuple(
        bank.index_select(0, source[:4, 0].to(torch.long)).reshape(4, -1)
        for bank, source in zip(
            banks, state["ssi_sources"], strict=True
        )
    )
    before_pregather = tuple(raw.clone() for raw in raws)
    kernel.selfcheck_fixed32_conv_col0_ssi_sources(num_spec_decodes=4)
    assert all(
        torch.equal(state["staging"][layer, :4], expected)
        for layer, expected in enumerate(expected_staging)
    )
    assert all(
        torch.equal(raw, before)
        for raw, before in zip(raws, before_pregather, strict=True)
    )

    paths.fill_(31)
    lens.zero_()
    zero_expected = tuple(raw.clone() for raw in raws)
    _legacy_commit_reference(
        banks=_bank_views_for_raw_clones(zero_expected, banks),
        spec_state_indices=commit_ssi,
        accepted_paths=paths,
        accepted_lens=lens,
        batch=1,
    )
    kernel.launch_fixed32_conv_commit_to_col0(
        conv_banks=banks,
        spec_state_indices=commit_ssi,
        accepted_paths=paths,
        accepted_lens=lens,
        num_spec_decodes=1,
    )
    assert all(
        torch.equal(raw, expected)
        for raw, expected in zip(raws, zero_expected, strict=True)
    )

    # Max accepted length with an explicit src==dst leaf remains byte-neutral.
    commit_ssi[:, 0, 0] = 0
    commit_ssi[:, 0, 2] = 0
    paths.zero_()
    paths[0, 15] = 2
    lens.zero_()
    lens[0] = 16
    max_len_expected = tuple(raw.clone() for raw in raws)
    _legacy_commit_reference(
        banks=_bank_views_for_raw_clones(max_len_expected, banks),
        spec_state_indices=commit_ssi,
        accepted_paths=paths,
        accepted_lens=lens,
        batch=1,
    )
    kernel.launch_fixed32_conv_commit_to_col0(
        conv_banks=banks,
        spec_state_indices=commit_ssi,
        accepted_paths=paths,
        accepted_lens=lens,
        num_spec_decodes=1,
    )
    assert all(
        torch.equal(raw, expected)
        for raw, expected in zip(raws, max_len_expected, strict=True)
    )

    commit_ssi[:, :, 0] = torch.arange(
        4, dtype=torch.int32, device="cuda"
    )
    commit_ssi[:, :, 1] = torch.tensor(
        [1, 2, 3, 0], dtype=torch.int32, device="cuda"
    )
    paths.zero_()
    paths[:, 0] = 1
    lens.fill_(1)
    expected_raws = tuple(raw.clone() for raw in raws)
    _legacy_commit_reference(
        banks=_bank_views_for_raw_clones(expected_raws, banks),
        spec_state_indices=commit_ssi,
        accepted_paths=paths,
        accepted_lens=lens,
        batch=4,
    )
    kernel.launch_fixed32_conv_commit_to_col0(
        conv_banks=banks,
        spec_state_indices=commit_ssi,
        accepted_paths=paths,
        accepted_lens=lens,
        num_spec_decodes=4,
    )
    assert all(
        torch.equal(raw, expected)
        for raw, expected in zip(raws, expected_raws, strict=True)
    )
    counters = kernel.fixed32_conv_col0_commit_counters()
    assert counters["gather_launches_by_batch"][1] == 2
    assert counters["scatter_launches_by_batch"][1] == 2
    assert counters["gather_launches_by_batch"][4] == 1
    assert counters["scatter_launches_by_batch"][4] == 1


def test_preseed_rejects_overlapping_layer_banks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_FIXED32_CONV_PREGATHER", {})
    monkeypatch.setattr(kernel, "_FR13_FIXED32_CONV_SSI_GROUPS", {})
    order = tuple(f"gdn.{index:02d}" for index in range(48))
    _, bank, _ = _page_shared_bank(rows=8, seed=0)
    for group_index in range(3):
        kernel.register_fixed32_conv_col0_ssi_group(
            layer_names=order[group_index * 16 : (group_index + 1) * 16],
            spec_state_indices=torch.zeros(
                (4, 1), dtype=torch.int32, device="cuda"
            ),
            max_batch_size=4,
        )
    with pytest.raises(ValueError, match="bank spans overlap"):
        kernel.preseed_fixed32_conv_col0_pregather(
            conv_banks=(bank,) * 48,
            layer_order=order,
            max_batch_size=4,
            commit_spec_state_indices=torch.zeros(
                (48, 4, 32), dtype=torch.int32, device="cuda"
            ),
            accepted_paths=torch.zeros(
                (4, 16), dtype=torch.int32, device="cuda"
            ),
            accepted_lens=torch.zeros(
                (4,), dtype=torch.int32, device="cuda"
            ),
        )

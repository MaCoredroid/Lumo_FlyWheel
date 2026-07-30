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


def _page_shared_bank(
    *,
    rows: int,
    seed: int,
    conv_shape: tuple[int, int] = (3, 2),
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
        )
        raws.append(raw)
        banks.append(bank)
    bank_tuple = tuple(banks)
    assert bank_tuple[0][0].numel() == 348_160
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


def test_b1_zero_accept_and_b4_alias_cycle_are_whole_page_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raws, banks, commit_ssi, paths, lens = _install_preseed(monkeypatch)

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

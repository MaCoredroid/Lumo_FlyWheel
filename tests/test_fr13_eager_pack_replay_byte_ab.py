"""FR13_EAGER_PACK 2b batched-replay byte A/B (SKIPPED on CPU hosts -- GPU gate).

FIX-2 item 2b pre-live gate (FR13_B1_FIX1_GATE_BIND.md pattern, class 10):
the batched all-layer replay kernel is source-identical per program to the
single-layer kernel, but a NEW COMPILE is never assumed codegen-identical.
This test runs the legacy per-layer launch loop and the ONE batched launch
from bit-identical inputs (each layer its own bank tensor, exercising the
int64 pointer table) and asserts every layer's published bank bytes are
INT-VIEW BYTE-IDENTICAL (never atol). It must pass on the GPU host BEFORE
any live boot with FR13_EAGER_PACK=1; the live B=1 same-seed repeat remains
the first binding gate (class 8: offline proofs are necessary, not
sufficient).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import triton  # noqa: F401

    _TRITON_OK = True
except Exception:  # pragma: no cover - host venv has no triton
    _TRITON_OK = False

pytestmark = pytest.mark.skipif(
    not (_TRITON_OK and torch.cuda.is_available()),
    reason="FR13_EAGER_PACK replay byte A/B requires CUDA + triton (GPU gate)",
)

NUM_KH = 16
NUM_VH = 48
DIM_K = 128
DIM_V = 128
N_PAD = 16
SPEC_COLS = 16
PATH_COLS = 10  # deployed num_spec + 1
BANK_ROWS = SPEC_COLS + 4
OUTPUT_SCALE = DIM_K**-0.5


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize(
    "num_layers,num_spec,seed", [(6, 1, 1313), (48, 1, 1414), (6, 2, 1515)]
)
def test_batched_replay_bank_bytes_match_legacy_loop(
    num_layers: int, num_spec: int, seed: int
) -> None:
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        build_replay_bank_pointer_table,
        launch_tree_gdn_replay,
        launch_tree_gdn_replay_all_layers,
    )

    device = torch.device("cuda")
    torch.manual_seed(seed)

    ring_bs = max(num_spec, 2)
    k_rings = torch.randn(
        (num_layers, ring_bs, N_PAD, NUM_KH, DIM_K), device=device, dtype=torch.bfloat16
    )
    v_rings = torch.randn(
        (num_layers, ring_bs, N_PAD, NUM_VH, DIM_V), device=device, dtype=torch.bfloat16
    )
    a_rings = torch.randn(
        (num_layers, ring_bs, N_PAD, NUM_VH), device=device, dtype=torch.bfloat16
    )
    b_rings = torch.randn(
        (num_layers, ring_bs, N_PAD, NUM_VH), device=device, dtype=torch.bfloat16
    )
    A_logs = (torch.randn((num_layers, NUM_VH), device=device).abs() * 0.5).to(
        torch.float32
    )
    dt_biases = torch.randn((num_layers, NUM_VH), device=device, dtype=torch.float32)
    spec_idx = (
        torch.arange(SPEC_COLS, device=device, dtype=torch.int32)
        .view(1, 1, -1)
        .expand(num_layers, ring_bs, SPEC_COLS)
        .contiguous()
    )
    prev_lens = (
        torch.tensor([3, 1][:ring_bs], device=device, dtype=torch.int32)
        .view(1, -1)
        .expand(num_layers, ring_bs)
        .contiguous()
    )
    # Branch-shaped accepted paths (non-monotone node ids) + a zero-accept row.
    paths_host = [[2, 6, 7, 9, 0, 0, 0, 0, 0, 0], [0] * PATH_COLS][:ring_bs]
    lens_host = [4, 0][:ring_bs]
    accepted_paths = torch.tensor(paths_host, device=device, dtype=torch.int32)
    accepted_lens = torch.tensor(lens_host, device=device, dtype=torch.int32)

    base_banks = [
        torch.randn((BANK_ROWS, NUM_VH, DIM_V, DIM_K), device=device, dtype=torch.float32)
        for _ in range(num_layers)
    ]

    # Arm A: legacy per-layer launch loop (the 48-launch storm).
    banks_legacy = [b.clone() for b in base_banks]
    for li in range(num_layers):
        launch_tree_gdn_replay(
            state_bank=banks_legacy[li],
            spec_state_indices=spec_idx[li],
            prev_lens=prev_lens[li],
            accepted_paths=accepted_paths,
            accepted_lens=accepted_lens,
            k_ring=k_rings[li],
            v_ring=v_rings[li],
            a_ring=a_rings[li],
            b_ring=b_rings[li],
            A_log=A_logs[li],
            dt_bias=dt_biases[li],
            num_spec_decodes=num_spec,
            output_scale=OUTPUT_SCALE,
            use_qk_l2norm_in_kernel=True,
        )

    # Arm B: ONE batched all-layer launch via the int64 bank pointer table.
    banks_batched = [b.clone() for b in base_banks]
    ptr_list, bank_shape, bank_stride = build_replay_bank_pointer_table(banks_batched)
    bank_ptrs = torch.tensor(ptr_list, dtype=torch.int64, device=device)
    launch_tree_gdn_replay_all_layers(
        bank_ptrs=bank_ptrs,
        bank_shape=bank_shape,
        bank_stride=bank_stride,
        spec_state_indices=spec_idx,
        prev_lens=prev_lens,
        accepted_paths=accepted_paths,
        accepted_lens=accepted_lens,
        k_rings=k_rings,
        v_rings=v_rings,
        a_rings=a_rings,
        b_rings=b_rings,
        A_logs=A_logs,
        dt_biases=dt_biases,
        num_layers=num_layers,
        num_spec_decodes=num_spec,
        output_scale=OUTPUT_SCALE,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    # INT-VIEW byte identity of EVERY layer's full bank (never atol; class 10
    # bar) -- includes untouched rows, so any cross-layer/row bleed fails too.
    for li in range(num_layers):
        assert torch.equal(_bits(banks_legacy[li]), _bits(banks_batched[li])), (
            f"layer {li}: batched replay bank bytes != legacy per-layer loop"
        )


def test_pointer_table_validates_stacking_preconditions() -> None:
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        build_replay_bank_pointer_table,
    )

    device = torch.device("cuda")
    good = [
        torch.zeros((4, 2, 8, 8), device=device, dtype=torch.float32)
        for _ in range(2)
    ]
    ptrs, shape, stride = build_replay_bank_pointer_table(good)
    assert ptrs == [int(b.data_ptr()) for b in good]
    assert shape == (4, 2, 8, 8)
    assert stride == good[0].stride(0)

    # Non-fp32 bank -> fail-loud (no silent per-layer fallback, class 9).
    with pytest.raises(ValueError):
        build_replay_bank_pointer_table(
            [good[0], good[1].to(torch.bfloat16)]
        )
    # Shape mismatch across layers -> fail-loud.
    with pytest.raises(ValueError):
        build_replay_bank_pointer_table(
            [good[0], torch.zeros((5, 2, 8, 8), device=device)]
        )

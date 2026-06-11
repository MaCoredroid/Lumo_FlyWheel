"""FR13 replay-route GPU byte A/B (SKIPPED on CPU hosts -- GPU TODO).

Unit-level form of the one-time codegen-identity gate: run the scan with
STORE_NODE_STATES=True to get the per-node export, then run the replay kernel
from the same bank/h0/ring bytes and assert the published linear columns are
BYTE-identical to the exported accepted-path states. This is the
kernel-level half of the gate; the official gate also replays CAPTURED
serving payloads (FR10_TREE_GDN_CAPTURE_PAYLOAD harness) -- see
FR13_REPLAY_ROUTE_BUILD.md GPU TODO list.
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
    reason="FR13 replay byte A/B requires CUDA + triton (listed as GPU TODO)",
)

NUM_KH = 16
NUM_VH = 48
DIM_K = 128
DIM_V = 128
PARENTS = (-1, 0, 1, 2, 3, 4, 0, 1, 2, 3)  # deployed 10-node gdn tree
SPEC_COLS = 16


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


@pytest.mark.parametrize("acc_len,prev_len", [(0, 3), (1, 1), (3, 3), (5, 1), (5, 6)])
def test_replay_publishes_byte_identical_accepted_states(acc_len: int, prev_len: int) -> None:
    from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
        Tree,
        launch_tree_gdn_prepared,
        launch_tree_gdn_replay,
        padded_nodes,
    )

    device = torch.device("cuda")
    torch.manual_seed(1313 + acc_len + 10 * prev_len)
    tree = Tree(PARENTS)
    n = tree.n
    n_pad = padded_nodes(n)
    strict, visible = tree.masks(device, n_pad)

    q = torch.randn((n, NUM_KH, DIM_K), device=device, dtype=torch.bfloat16)
    k = torch.randn((n, NUM_KH, DIM_K), device=device, dtype=torch.bfloat16)
    v = torch.randn((n, NUM_VH, DIM_V), device=device, dtype=torch.bfloat16)
    raw_a = torch.randn((n, NUM_VH), device=device, dtype=torch.bfloat16)
    raw_b = torch.randn((n, NUM_VH), device=device, dtype=torch.bfloat16)
    g_dummy = torch.zeros((n, NUM_VH), device=device, dtype=torch.float32)
    beta_dummy = torch.zeros((n, NUM_VH), device=device, dtype=torch.float32)
    a_log = (torch.randn((NUM_VH,), device=device).abs() * 0.5).to(torch.float32)
    dt_bias = torch.randn((NUM_VH,), device=device, dtype=torch.float32)

    bank = torch.randn(
        (SPEC_COLS + 4, NUM_VH, DIM_V, DIM_K), device=device, dtype=torch.float32
    )
    spec_idx = torch.arange(SPEC_COLS, device=device, dtype=torch.int32).view(1, -1)
    prev_lens = torch.tensor([prev_len], device=device, dtype=torch.int32)

    # Scan with the export ON (legacy path) to obtain the per-node states.
    out = torch.empty((n_pad, NUM_VH, DIM_V), device=device, dtype=q.dtype)
    state = torch.empty(
        (n_pad, NUM_VH, DIM_V, DIM_K), device=device, dtype=torch.float32
    )
    launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=v,
        g=g_dummy,
        beta=beta_dummy,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=a_log,
        dt_bias=dt_bias,
        h0=bank,
        h0_indices=spec_idx,
        h0_num_accepted_tokens=prev_lens,
        h0_is_bank=True,
        h0_index_row=0,
        h0_batch_index=0,
        h0_use_accepted_column=True,
        n_actual=n,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        out=out,
        state=state,
        output_scale=DIM_K**-0.5,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    export = state.clone()

    # Scan with the export OFF must produce the identical out (pure-export
    # claim) -- and must not touch the state buffer.
    out2 = torch.empty_like(out)
    launch_tree_gdn_prepared(
        q=q,
        k=k,
        v=v,
        g=g_dummy,
        beta=beta_dummy,
        raw_a=raw_a,
        raw_b=raw_b,
        A_log=a_log,
        dt_bias=dt_bias,
        h0=bank,
        h0_indices=spec_idx,
        h0_num_accepted_tokens=prev_lens,
        h0_is_bank=True,
        h0_index_row=0,
        h0_batch_index=0,
        h0_use_accepted_column=True,
        n_actual=n,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        out=out2,
        state=None,
        store_node_states=False,
        output_scale=DIM_K**-0.5,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), out2.view(torch.int16))

    # Accepted spine path of length acc_len (committer convention: gdn node
    # ids excluding the root).
    path = list(range(1, acc_len + 1))
    paths = torch.zeros((1, 6), device=device, dtype=torch.int32)
    for t, node in enumerate(path):
        paths[0, t] = node
    lens = torch.tensor([acc_len], device=device, dtype=torch.int32)

    ring_k = torch.zeros((1, n_pad, NUM_KH, DIM_K), device=device, dtype=k.dtype)
    ring_v = torch.zeros((1, n_pad, NUM_VH, DIM_V), device=device, dtype=v.dtype)
    ring_a = torch.zeros((1, n_pad, NUM_VH), device=device, dtype=raw_a.dtype)
    ring_b = torch.zeros((1, n_pad, NUM_VH), device=device, dtype=raw_b.dtype)
    ring_k[0, :n].copy_(k)
    ring_v[0, :n].copy_(v)
    ring_a[0, :n].copy_(raw_a)
    ring_b[0, :n].copy_(raw_b)

    untouched = bank.clone()
    launch_tree_gdn_replay(
        state_bank=bank,
        spec_state_indices=spec_idx,
        prev_lens=prev_lens,
        accepted_paths=paths,
        accepted_lens=lens,
        k_ring=ring_k,
        v_ring=ring_v,
        a_ring=ring_a,
        b_ring=ring_b,
        A_log=a_log,
        dt_bias=dt_bias,
        num_spec_decodes=1,
        output_scale=DIM_K**-0.5,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    # Written columns: 0..acc_len-1 (and 0 on zero-accept = root state).
    if acc_len == 0:
        expected = {0: export[0]}
    else:
        expected = {t: export[path[t]] for t in range(acc_len)}
    written_rows = set()
    for col, exp in expected.items():
        row = int(spec_idx[0, col].item())
        written_rows.add(row)
        assert torch.equal(
            _bits(bank[row]), _bits(exp)
        ), f"col {col} not byte-identical (acc_len={acc_len}, prev_len={prev_len})"
    for row in range(bank.shape[0]):
        if row not in written_rows:
            assert torch.equal(_bits(bank[row]), _bits(untouched[row])), (
                f"row {row} mutated outside the publish set"
            )

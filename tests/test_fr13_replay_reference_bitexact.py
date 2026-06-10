"""FR13 replay-route CPU proofs: scan-chain vs replay-chain BITWISE identity.

These tests prove, on IEEE-754 fp32 (pure torch, no triton/CUDA), that the
replay restructuring is bit-identical to the scan chain it replaces:
identical per-node op order (softplus x<=20 branch, sigmoid, k-l2norm
rsqrt(sum+1e-6), exp, multiply-then-rank-1), the per-edge +0.0 parent-handoff
normalization (the scan's masked-sum -0.0 flip), root consuming h0
UNnormalized, the zero-accept row-0 publish, and bf16 ring roundtrips.

They intentionally do NOT touch Triton: torch<->Triton codegen identity is
the GPU-gated byte A/B (see FR13_REPLAY_ROUTE_BUILD.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr13_replay_reference import (  # noqa: E402
    accepted_path_to_root_chain,
    gdn_node_step_ref,
    handoff_ref,
    legacy_consumed_columns,
    tree_replay_chain_ref,
    tree_scan_chain_ref,
)

DIM_K = 128
DIM_V = 128
# Deployed 9-draft caterpillar+leaves gdn tree (root + 9 drafts = 10 nodes);
# second vector is an arbitrary bushier topology (parent[i] < i).
PARENTS_DEPLOYED = [-1, 0, 1, 2, 3, 4, 0, 1, 2, 3]
PARENTS_BUSHY = [-1, 0, 0, 1, 1, 2, 4, 4, 6, 6, 7, 0, 3, 5]
SPINE_PARENTS = [-1, 0, 1, 2, 3, 4]


def _bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float32:
        return t.contiguous().view(torch.int32)
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16)
    raise AssertionError(f"unexpected dtype {t.dtype}")


def assert_bits_equal(a: torch.Tensor, b: torch.Tensor, msg: str = "") -> None:
    assert a.dtype == b.dtype and a.shape == b.shape, msg
    assert torch.equal(_bits(a), _bits(b)), msg


def _random_inputs(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    # Consumed precision: activations are bf16-roundtripped before the fp32
    # widen, exactly like the activation ring path.
    k = torch.randn((n, DIM_K), generator=g).to(torch.bfloat16).to(torch.float32)
    v = torch.randn((n, DIM_V), generator=g).to(torch.bfloat16).to(torch.float32)
    raw_a = torch.randn((n,), generator=g).to(torch.bfloat16).to(torch.float32)
    raw_b = torch.randn((n,), generator=g).to(torch.bfloat16).to(torch.float32)
    a_log = torch.randn((), generator=g).abs() * 0.5
    dt_bias = torch.randn((), generator=g)
    h0 = torch.randn((DIM_V, DIM_K), generator=g)
    return h0, k, v, raw_a, raw_b, a_log, dt_bias


def test_handoff_masked_sum_equals_plus_zero_and_flips_neg_zero() -> None:
    g = torch.Generator().manual_seed(7)
    h = torch.randn((4, 8, 8), generator=g)
    h[1, 0, 0] = -0.0
    h[1, 2, 3] = 0.0
    via_sum = handoff_ref(h, 1)
    via_add = h[1] + 0.0
    assert_bits_equal(via_sum, via_add, "masked-sum handoff != (+0.0) emulation")
    # Non-vacuous: the -0.0 really flips, so a replay WITHOUT the +0.0 would
    # NOT be bit-identical.
    assert not torch.equal(_bits(via_sum), _bits(h[1]))
    assert via_sum[0, 0].item() == 0.0
    assert _bits(via_sum)[0, 0].item() == 0  # +0.0 bit pattern


def test_replay_chain_matches_scan_chain_all_paths_deployed_tree() -> None:
    for seed, parents in ((1313, PARENTS_DEPLOYED), (4242, PARENTS_BUSHY)):
        n = len(parents)
        h0, k, v, raw_a, raw_b, a_log, dt_bias = _random_inputs(n, seed)
        scan_states = tree_scan_chain_ref(
            h0, parents, k, v, raw_a, raw_b, a_log, dt_bias, output_scale=DIM_K**-0.5
        )
        for node in range(n):
            path = accepted_path_to_root_chain(parents, node)
            states, bank = tree_replay_chain_ref(
                h0, path, k, v, raw_a, raw_b, a_log, dt_bias,
                output_scale=DIM_K**-0.5,
            )
            chain = [0] + path
            for t, nd in enumerate(chain):
                assert_bits_equal(
                    states[t],
                    scan_states[nd],
                    f"replay step {t} != scan node {nd} (parents={parents})",
                )


def test_replay_bank_columns_match_legacy_publish_plus_remap() -> None:
    parents = PARENTS_DEPLOYED
    n = len(parents)
    h0, k, v, raw_a, raw_b, a_log, dt_bias = _random_inputs(n, 99)
    scan_states = tree_scan_chain_ref(
        h0, parents, k, v, raw_a, raw_b, a_log, dt_bias
    )
    for node in range(n):
        path = accepted_path_to_root_chain(parents, node)
        _, bank = tree_replay_chain_ref(
            h0, path, k, v, raw_a, raw_b, a_log, dt_bias
        )
        legacy = legacy_consumed_columns(scan_states, path)
        assert set(bank.keys()) == set(legacy.keys())
        for col in legacy:
            assert_bits_equal(
                bank[col], legacy[col], f"bank col {col} mismatch (node {node})"
            )


def test_zero_accept_replays_root_into_column_zero() -> None:
    parents = SPINE_PARENTS
    h0, k, v, raw_a, raw_b, a_log, dt_bias = _random_inputs(len(parents), 5)
    scan_states = tree_scan_chain_ref(h0, parents, k, v, raw_a, raw_b, a_log, dt_bias)
    states, bank = tree_replay_chain_ref(
        h0, [], k, v, raw_a, raw_b, a_log, dt_bias
    )
    # Empty accepted path: exactly one replay step (the root), published to
    # column 0 (the next h0 read clamps accepted_len-1 to column 0).
    assert len(states) == 1
    assert set(bank.keys()) == {0}
    assert_bits_equal(bank[0], scan_states[0], "zero-accept row-0 != root state")


def test_replay_chain_every_accepted_len_on_spine() -> None:
    parents = SPINE_PARENTS
    n = len(parents)
    h0, k, v, raw_a, raw_b, a_log, dt_bias = _random_inputs(n, 21)
    scan_states = tree_scan_chain_ref(h0, parents, k, v, raw_a, raw_b, a_log, dt_bias)
    for acc_len in range(0, n):  # 0..MAX (5 on the 6-node spine)
        path = list(range(1, acc_len + 1))
        states, bank = tree_replay_chain_ref(
            h0, path, k, v, raw_a, raw_b, a_log, dt_bias
        )
        assert len(states) == acc_len + 1
        for t, nd in enumerate([0] + path):
            assert_bits_equal(states[t], scan_states[nd], f"len={acc_len} t={t}")
        consumed_col = max(acc_len - 1, 0)
        assert consumed_col in bank


def test_neg_zero_in_parent_state_handoff_is_bit_exact() -> None:
    # Craft a root update that PRESERVES -0.0 elements of h0 (k = zeros so
    # the rank-1 increment rows are -0.0 for negative v): the child handoff
    # must then flip them (+0.0) identically in both chains.
    parents = [-1, 0]
    n = 2
    _, k, v, raw_a, raw_b, a_log, dt_bias = _random_inputs(n, 33)
    k = k.clone()
    v = v.clone()
    k[0] = 0.0
    v[0] = -1.0
    h0 = torch.zeros((DIM_V, DIM_K))
    h0[3, 5] = -0.0
    h0[7, 11] = -0.0
    assert _bits(h0)[3, 5].item() != 0  # really -0.0 in the seed
    scan_states = tree_scan_chain_ref(h0, parents, k, v, raw_a, raw_b, a_log, dt_bias)
    # The root state must still carry -0.0 (otherwise this test is vacuous).
    assert _bits(scan_states[0])[3, 5].item() != 0, "-0.0 did not survive the root update"
    states, _ = tree_replay_chain_ref(h0, [1], k, v, raw_a, raw_b, a_log, dt_bias)
    assert_bits_equal(states[0], scan_states[0], "root state mismatch (-0.0 case)")
    assert_bits_equal(states[1], scan_states[1], "child state mismatch (-0.0 handoff)")


def test_node_step_softplus_branch_and_q_independence() -> None:
    # The x<=20 branch must be exercised on BOTH sides, and q must not
    # influence the state (the replay discards out and feeds zeros q).
    state = torch.randn((DIM_V, DIM_K), generator=torch.Generator().manual_seed(3))
    k = torch.randn((DIM_K,), generator=torch.Generator().manual_seed(4))
    v = torch.randn((DIM_V,), generator=torch.Generator().manual_seed(5))
    for raw_a_val in (-3.0, 0.5, 25.0, 19.999, 20.0, 20.001):
        raw_a = torch.tensor(raw_a_val)
        raw_b = torch.tensor(0.25)
        a_log = torch.tensor(0.1)
        dt_bias = torch.tensor(0.0)
        s_no_q, out_none = gdn_node_step_ref(
            state, k, v, raw_a, raw_b, a_log, dt_bias
        )
        s_zero_q, out_zero = gdn_node_step_ref(
            state, k, v, raw_a, raw_b, a_log, dt_bias, q=torch.zeros((DIM_K,))
        )
        s_real_q, out_real = gdn_node_step_ref(
            state, k, v, raw_a, raw_b, a_log, dt_bias,
            q=torch.randn((DIM_K,), generator=torch.Generator().manual_seed(6)),
        )
        assert out_none is None and out_zero is not None and out_real is not None
        assert_bits_equal(s_no_q, s_zero_q, f"q changed state (raw_a={raw_a_val})")
        assert_bits_equal(s_no_q, s_real_q, f"q changed state (raw_a={raw_a_val})")


def test_activation_ring_bf16_roundtrip_is_byte_exact() -> None:
    # The ring stores activations at consumed precision (bf16 byte-copy);
    # store/load must reproduce the exact bytes and widen to the exact fp32
    # bits the scan consumed.
    g = torch.Generator().manual_seed(77)
    n_pad, n = 16, 10
    for shape in ((n, 16, 128), (n, 48, 128), (n, 48)):
        src = torch.randn(shape, generator=g).to(torch.bfloat16)
        ring = torch.zeros((n_pad, *shape[1:]), dtype=torch.bfloat16)
        ring[:n].copy_(src)
        assert torch.equal(ring[:n].view(torch.int16), src.view(torch.int16))
        assert torch.equal(
            ring[:n].to(torch.float32).view(torch.int32),
            src.to(torch.float32).view(torch.int32),
        )
    # Edge payloads survive the roundtrip byte-exactly too.
    edge = torch.tensor(
        [0.0, -0.0, 1e-38, -1e-38, 65280.0, -65280.0, 3.3895e38],
        dtype=torch.bfloat16,
    )
    ring = torch.zeros((16,), dtype=torch.bfloat16)
    ring[: edge.numel()].copy_(edge)
    assert torch.equal(ring[: edge.numel()].view(torch.int16), edge.view(torch.int16))


def test_replay_differs_without_plus_zero_normalization() -> None:
    # Adversarial self-check that the +0.0 edge normalization is load-bearing
    # in the -0.0 construction: dropping it must break bit-identity.
    parents = [-1, 0]
    _, k, v, raw_a, raw_b, a_log, dt_bias = _random_inputs(2, 55)
    k = k.clone()
    v = v.clone()
    k[0] = 0.0
    v[0] = -1.0
    # Also zero the CHILD's k so the -0.0/+0.0 parent bits flow through the
    # child update unchanged (decay-multiply keeps the sign distinction).
    k[1] = 0.0
    v[1] = -1.0
    h0 = torch.zeros((DIM_V, DIM_K))
    h0[3, 5] = -0.0
    scan_states = tree_scan_chain_ref(h0, parents, k, v, raw_a, raw_b, a_log, dt_bias)

    # Replay WITHOUT the +0.0: replicate the chain by hand.
    state, _ = gdn_node_step_ref(h0, k[0], v[0], raw_a[0], raw_b[0], a_log, dt_bias)
    state_no_norm, _ = gdn_node_step_ref(
        state, k[1], v[1], raw_a[1], raw_b[1], a_log, dt_bias
    )
    assert not torch.equal(
        _bits(state_no_norm), _bits(scan_states[1])
    ), "+0.0 normalization was not load-bearing; -0.0 construction is vacuous"

    states, _ = tree_replay_chain_ref(h0, [1], k, v, raw_a, raw_b, a_log, dt_bias)
    assert_bits_equal(states[1], scan_states[1])

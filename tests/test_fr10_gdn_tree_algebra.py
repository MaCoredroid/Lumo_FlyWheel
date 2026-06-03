from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fr10_gdn_tree_algebra_reference import (  # noqa: E402
    VLLM_CPU_RULE,
    accepted_path_final_state,
    assert_close_tree,
    chunk_oracle_eval,
    make_layer_inputs,
    make_tree,
    packed_tree_eval,
    read_qwen_gdn_config,
    serial_per_path_eval,
    trunk_sharing_eval,
)


pytestmark = pytest.mark.skipif(not VLLM_CPU_RULE.exists(), reason="vLLM 0.22 CPU GDN oracle missing")


@pytest.mark.parametrize("total_nodes", [2, 3, 6, 8, 14])
def test_fr10_packed_and_trunk_tree_match_serial_fp32(total_nodes: int):
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=total_nodes, spine_depth=min(6, total_nodes), seed=100 + total_nodes)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=torch.float32, seed=200 + total_nodes)

    serial = serial_per_path_eval(tree, tensors, cfg)
    packed = packed_tree_eval(tree, tensors, cfg)
    trunk = trunk_sharing_eval(tree, tensors, cfg)
    chunk = chunk_oracle_eval(tree, tensors, cfg)

    assert_close_tree(packed, serial, atol=2e-5, rtol=2e-5)
    assert_close_tree(trunk, serial, atol=2e-5, rtol=2e-5)
    assert_close_tree(chunk, serial, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 2e-5, 2e-5),
        (torch.bfloat16, 8e-2, 8e-2),
        (torch.float16, 8e-2, 8e-2),
    ],
)
def test_fr10_dtype_sweep(dtype: torch.dtype, atol: float, rtol: float):
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=14, spine_depth=6, seed=301)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=dtype, seed=302)

    serial = serial_per_path_eval(tree, tensors, cfg)
    packed = packed_tree_eval(tree, tensors, cfg)
    trunk = trunk_sharing_eval(tree, tensors, cfg)

    assert_close_tree(packed, serial, atol=atol, rtol=rtol)
    assert_close_tree(trunk, serial, atol=atol, rtol=rtol)


def test_fr10_random_small_tree_generator_required_shapes():
    cfg = read_qwen_gdn_config()
    for total_nodes in [2, 3, 6, 8, 14]:
        for spine_depth in range(1, 7):
            tree = make_tree(total_nodes=total_nodes, spine_depth=spine_depth, seed=900 + 31 * total_nodes + spine_depth)
            assert len(tree.nodes) == total_nodes
            assert max(node.depth for node in tree.nodes) <= 6
            assert max(node.sibling_index for node in tree.nodes) <= 2
            tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=torch.float32, seed=950 + 31 * total_nodes + spine_depth)
            serial = serial_per_path_eval(tree, tensors, cfg)
            trunk = trunk_sharing_eval(tree, tensors, cfg)
            assert_close_tree(trunk, serial, atol=2e-5, rtol=2e-5)


def test_fr10_appending_sibling_leaf_does_not_change_trunk_nodes():
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=8, spine_depth=5, seed=401)
    parent_id = tree.trunk_nodes()[-2]
    extended = tree.append_sibling_leaf(parent_id=parent_id)
    tensors = make_layer_inputs(cfg, len(extended.nodes), dtype=torch.float32, seed=402)

    before = packed_tree_eval(tree, {k: v[:, : len(tree.nodes)] if k != "initial_state" else v for k, v in tensors.items()}, cfg)
    after = packed_tree_eval(extended, tensors, cfg)

    for node_id in tree.trunk_nodes():
        torch.testing.assert_close(after["state"][node_id], before["state"][node_id], atol=0.0, rtol=0.0)
        torch.testing.assert_close(after["logits"][:, node_id], before["logits"][:, node_id], atol=0.0, rtol=0.0)


def test_fr10_accepted_path_final_state_equals_serial_native_decode():
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=14, spine_depth=6, seed=501)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=torch.float32, seed=502)
    leaf_id = max(range(len(tree.nodes)), key=lambda i: tree.nodes[i].depth)

    serial = serial_per_path_eval(tree, tensors, cfg)
    packed = packed_tree_eval(tree, tensors, cfg)

    torch.testing.assert_close(
        accepted_path_final_state(tree, packed, leaf_id),
        accepted_path_final_state(tree, serial, leaf_id),
        atol=2e-5,
        rtol=2e-5,
    )


def test_fr10_negative_control_shared_parent_mutation_fails_loudly():
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=8, spine_depth=4, seed=601)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=torch.float32, seed=602)

    serial = serial_per_path_eval(tree, tensors, cfg)
    contaminated = packed_tree_eval(tree, tensors, cfg, contaminate_shared_parent=True)

    with pytest.raises(AssertionError):
        assert_close_tree(contaminated, serial, atol=2e-5, rtol=2e-5)


def test_fr10_negative_control_linear_mask_leaks_across_siblings_and_fails():
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=8, spine_depth=4, seed=701)
    tree = tree.append_sibling_leaf(parent_id=1)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=torch.float32, seed=702)

    serial = serial_per_path_eval(tree, tensors, cfg)
    linear_leak = trunk_sharing_eval(tree, tensors, cfg, linear_mask_leak=True)

    with pytest.raises(AssertionError):
        assert_close_tree(linear_leak, serial, atol=2e-5, rtol=2e-5)


def test_fr10_greedy_logit_margin_exceeds_observed_tree_tolerance():
    cfg = read_qwen_gdn_config()
    tree = make_tree(total_nodes=14, spine_depth=6, seed=801)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=torch.float32, seed=802)

    serial = serial_per_path_eval(tree, tensors, cfg)
    packed = trunk_sharing_eval(tree, tensors, cfg)
    logit_delta = (packed["logits"] - serial["logits"]).abs().max()
    top2 = torch.topk(serial["logits"].reshape(-1, serial["logits"].shape[-1]), k=2, dim=-1).values
    min_margin = (top2[:, 0] - top2[:, 1]).min()

    # Real Gate B is byte-exact-by-identical-kernel: public path0 uses the same
    # verifier kernel. The synthetic margin is diagnostic only; token identity
    # is the greedy losslessness gate here.
    assert logit_delta.item() <= 2e-5
    torch.testing.assert_close(
        packed["logits"].argmax(dim=-1),
        serial["logits"].argmax(dim=-1),
        atol=0,
        rtol=0,
    )
    assert min_margin.item() >= 0.0


def test_fr10_negative_control_longest_accepted_hidden_winner_fails_distribution_gate():
    target = torch.tensor([0.40, 0.30, 0.15, 0.10, 0.05], dtype=torch.float64)
    biased = []
    tail = 1.0
    for prob in target.tolist():
        biased.append(prob * (2.0 * tail - prob))
        tail -= prob
    biased = torch.tensor(biased, dtype=torch.float64)
    biased = biased / biased.sum()
    noncentrality = 100_000 * torch.sum((biased - target) ** 2 / target)
    total_variation = 0.5 * torch.sum(torch.abs(biased - target))

    assert noncentrality.item() > 1_000
    assert total_variation.item() > 0.05
    assert not math.isclose(float(biased[0]), float(target[0]), rel_tol=0.01)

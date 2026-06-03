#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


VLLM_CPU_RULE = Path(
    "/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/ops/cpu/"
    "recurrent_gated_delta_rule.py"
)
QWEN_CONFIG = Path("/models/qwen3.6-27b-fp8/config.json")


def load_vllm_cpu_rule():
    spec = importlib.util.spec_from_file_location("fr10_vllm_cpu_gdn_rule", VLLM_CPU_RULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import vLLM CPU rule from {VLLM_CPU_RULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class QwenGDNConfig:
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    num_gdn_layers: int
    scale: float
    use_qk_l2norm_in_kernel: bool = True


def read_qwen_gdn_config(path: Path = QWEN_CONFIG) -> QwenGDNConfig:
    text_config = json.loads(path.read_text())["text_config"]
    layer_types = text_config["layer_types"]
    head_k_dim = int(text_config["linear_key_head_dim"])
    return QwenGDNConfig(
        num_k_heads=int(text_config["linear_num_key_heads"]),
        num_v_heads=int(text_config["linear_num_value_heads"]),
        head_k_dim=head_k_dim,
        head_v_dim=int(text_config["linear_value_head_dim"]),
        conv_kernel_size=int(text_config["linear_conv_kernel_dim"]),
        num_gdn_layers=sum(1 for x in layer_types if x == "linear_attention"),
        scale=head_k_dim**-0.5,
        use_qk_l2norm_in_kernel=True,
    )


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    parent_id: int
    depth: int
    sibling_index: int
    token_id: int
    position: int
    ancestor_offsets: tuple[int, ...]


@dataclass(frozen=True)
class TreeDescriptor:
    nodes: tuple[TreeNode, ...]

    def __post_init__(self) -> None:
        ids = [n.node_id for n in self.nodes]
        if ids != list(range(len(self.nodes))):
            raise ValueError("node ids must be topological and dense")
        for node in self.nodes:
            if node.parent_id >= node.node_id:
                raise ValueError("parent must precede child")

    def ancestors(self, node_id: int) -> tuple[int, ...]:
        out = []
        cur = self.nodes[node_id]
        while cur.parent_id >= 0:
            out.append(cur.parent_id)
            cur = self.nodes[cur.parent_id]
        return tuple(reversed(out))

    def path_to(self, node_id: int) -> tuple[int, ...]:
        return (*self.ancestors(node_id), node_id)

    def trunk_nodes(self) -> tuple[int, ...]:
        return tuple(n.node_id for n in self.nodes if n.sibling_index == 0)

    def append_sibling_leaf(self, parent_id: int, token_id: int = 999_983) -> "TreeDescriptor":
        siblings = [n for n in self.nodes if n.parent_id == parent_id]
        sibling_index = 0 if not siblings else max(n.sibling_index for n in siblings) + 1
        node_id = len(self.nodes)
        parent = self.nodes[parent_id]
        ancestors = (*self.ancestors(parent_id), parent_id)
        node = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=parent.depth + 1,
            sibling_index=sibling_index,
            token_id=token_id,
            position=node_id,
            ancestor_offsets=ancestors,
        )
        return TreeDescriptor((*self.nodes, node))


def make_tree(total_nodes: int, spine_depth: int, seed: int) -> TreeDescriptor:
    if total_nodes < 2:
        raise ValueError("FR10 random tree gate requires at least 2 nodes")
    rng = random.Random(seed)
    spine_depth = max(1, min(spine_depth, total_nodes, 6))
    nodes: list[TreeNode] = []
    child_counts: dict[int, int] = {}

    def add(parent_id: int) -> None:
        node_id = len(nodes)
        if parent_id < 0:
            depth = 1
            sibling_index = 0
            ancestors: tuple[int, ...] = ()
        else:
            parent = nodes[parent_id]
            depth = parent.depth + 1
            sibling_index = child_counts.get(parent_id, 0)
            ancestors = (*nodes[parent_id].ancestor_offsets, parent_id)
            child_counts[parent_id] = sibling_index + 1
        nodes.append(
            TreeNode(
                node_id=node_id,
                parent_id=parent_id,
                depth=depth,
                sibling_index=sibling_index,
                token_id=rng.randrange(1024, 248_000),
                position=node_id,
                ancestor_offsets=ancestors,
            )
        )

    add(-1)
    while len(nodes) < spine_depth:
        add(len(nodes) - 1)
    while len(nodes) < total_nodes:
        candidates = [n for n in nodes if n.depth < 6 and child_counts.get(n.node_id, 0) < 3]
        if not candidates:
            candidates = list(nodes)
        parent = rng.choice(candidates)
        add(parent.node_id)
    return TreeDescriptor(tuple(nodes))


def make_layer_inputs(
    cfg: QwenGDNConfig,
    num_nodes: int,
    *,
    dtype: torch.dtype,
    seed: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Keep magnitudes modest so fp16 CPU recurrence stays inside documented tolerances.
    q = torch.randn((1, num_nodes, cfg.num_k_heads, cfg.head_k_dim), generator=gen) * 0.2
    k = torch.randn((1, num_nodes, cfg.num_k_heads, cfg.head_k_dim), generator=gen) * 0.2
    v = torch.randn((1, num_nodes, cfg.num_v_heads, cfg.head_v_dim), generator=gen) * 0.2
    a = torch.randn((1, num_nodes, cfg.num_v_heads), generator=gen) * 0.2
    b = torch.randn((1, num_nodes, cfg.num_v_heads), generator=gen) * 0.2
    initial_state = (
        torch.randn((1, cfg.num_v_heads, cfg.head_v_dim, cfg.head_k_dim), generator=gen) * 0.05
    )
    A_log = torch.linspace(-3.0, -1.0, cfg.num_v_heads)
    dt_bias = torch.linspace(-2.0, 0.5, cfg.num_v_heads)
    vllm_rule = load_vllm_cpu_rule()
    g, beta = vllm_rule.gdn_gating(A_log=A_log, a=a, b=b, dt_bias=dt_bias)
    return {
        "q": q.to(device=device, dtype=dtype),
        "k": k.to(device=device, dtype=dtype),
        "v": v.to(device=device, dtype=dtype),
        "g": g.to(device=device, dtype=dtype),
        "beta": beta.to(device=device, dtype=dtype),
        "initial_state": initial_state.to(device=device, dtype=torch.float32),
    }


def _select_nodes(x: torch.Tensor, path: Iterable[int]) -> torch.Tensor:
    index = torch.tensor(list(path), dtype=torch.long, device=x.device)
    return x.index_select(1, index)


def _one_token(vllm_rule, tensors: dict[str, torch.Tensor], state: torch.Tensor, idx: int, cfg: QwenGDNConfig):
    token = (idx,)
    return vllm_rule.recurrent_gated_delta_rule(
        query=_select_nodes(tensors["q"], token),
        key=_select_nodes(tensors["k"], token),
        value=_select_nodes(tensors["v"], token),
        g=_select_nodes(tensors["g"], token),
        beta=_select_nodes(tensors["beta"], token),
        initial_state=state.clone(),
        scale=cfg.scale,
        use_qk_l2norm_in_kernel=cfg.use_qk_l2norm_in_kernel,
    )


def logits_from_output(output: torch.Tensor, vocab_size: int = 256) -> torch.Tensor:
    flat = output.reshape(output.shape[0], output.shape[1], -1).to(torch.float32)
    # Deterministic projection, independent of random test seed.
    cols = torch.arange(
        flat.shape[-1] * vocab_size, dtype=torch.float32, device=flat.device
    ).reshape(flat.shape[-1], vocab_size)
    weight = torch.sin(cols * 0.017) * 0.1
    return flat @ weight


def serial_per_path_eval(
    tree: TreeDescriptor,
    tensors: dict[str, torch.Tensor],
    cfg: QwenGDNConfig,
) -> dict[str, torch.Tensor]:
    vllm_rule = load_vllm_cpu_rule()
    states = []
    outputs = []
    logits = []
    for node in tree.nodes:
        path = tree.path_to(node.node_id)
        output, state = vllm_rule.recurrent_gated_delta_rule(
            query=_select_nodes(tensors["q"], path),
            key=_select_nodes(tensors["k"], path),
            value=_select_nodes(tensors["v"], path),
            g=_select_nodes(tensors["g"], path),
            beta=_select_nodes(tensors["beta"], path),
            initial_state=tensors["initial_state"].clone(),
            scale=cfg.scale,
            use_qk_l2norm_in_kernel=cfg.use_qk_l2norm_in_kernel,
        )
        node_output = output[:, -1:].contiguous()
        outputs.append(node_output)
        states.append(state)
        logits.append(logits_from_output(node_output))
    return {
        "state": torch.cat(states, dim=0),
        "output": torch.cat(outputs, dim=1),
        "logits": torch.cat(logits, dim=1),
    }


def packed_tree_eval(
    tree: TreeDescriptor,
    tensors: dict[str, torch.Tensor],
    cfg: QwenGDNConfig,
    *,
    contaminate_shared_parent: bool = False,
) -> dict[str, torch.Tensor]:
    vllm_rule = load_vllm_cpu_rule()
    states_by_node: dict[int, torch.Tensor] = {}
    outputs = []
    logits = []
    shared_parent_ref: torch.Tensor | None = None
    for node in tree.nodes:
        if node.parent_id < 0:
            parent_state = tensors["initial_state"]
        else:
            parent_state = states_by_node[node.parent_id]
        if contaminate_shared_parent and node.parent_id >= 0 and node.sibling_index > 0:
            # Negative control: mutate an already-published parent/trunk row.
            parent_state.add_(0.125)
            shared_parent_ref = parent_state
        output, state = _one_token(vllm_rule, tensors, parent_state, node.node_id, cfg)
        states_by_node[node.node_id] = state
        node_output = output[:, -1:].contiguous()
        outputs.append(node_output)
        logits.append(logits_from_output(node_output))
    if shared_parent_ref is not None:
        # Keep the mutation live so parity catches trunk contamination too.
        pass
    return {
        "state": torch.cat([states_by_node[i] for i in range(len(tree.nodes))], dim=0),
        "output": torch.cat(outputs, dim=1),
        "logits": torch.cat(logits, dim=1),
    }


def _tree_strict_ancestor_mask(tree: TreeDescriptor, device: torch.device) -> torch.Tensor:
    num_nodes = len(tree.nodes)
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    for node in tree.nodes:
        for ancestor_id in tree.ancestors(node.node_id):
            mask[node.node_id, ancestor_id] = True
    return mask


def _path_cum_g(tree: TreeDescriptor, g: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(g)
    for node in tree.nodes:
        path = tree.path_to(node.node_id)
        index = torch.tensor(path, dtype=torch.long, device=g.device)
        out[:, :, node.node_id] = g.index_select(2, index).sum(dim=2)
    return out


def trunk_sharing_eval(
    tree: TreeDescriptor,
    tensors: dict[str, torch.Tensor],
    cfg: QwenGDNConfig,
    *,
    linear_mask_leak: bool = False,
) -> dict[str, torch.Tensor]:
    vllm_rule = load_vllm_cpu_rule()
    initial_dtype = tensors["q"].dtype
    q = tensors["q"]
    k = tensors["k"]
    v = tensors["v"]
    beta = tensors["beta"]
    g = tensors["g"]
    if cfg.use_qk_l2norm_in_kernel:
        q = vllm_rule.l2norm(q, dim=-1, eps=1e-6)
        k = vllm_rule.l2norm(k, dim=-1, eps=1e-6)
    if q.shape[2] != v.shape[2]:
        repeat_factor = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(repeat_factor, dim=2)
        k = k.repeat_interleave(repeat_factor, dim=2)

    q, k, v, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (q, k, v, beta, g)
    ]
    q = q * cfg.scale
    state0 = tensors["initial_state"].to(torch.float32)
    batch, heads, num_nodes, key_dim = k.shape
    value_dim = v.shape[-1]
    if batch != 1:
        raise ValueError("FR10 P1 tree proof currently expects one packed request")

    if linear_mask_leak:
        strict_mask = torch.tril(
            torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=q.device),
            diagonal=-1,
        )
    else:
        strict_mask = _tree_strict_ancestor_mask(tree, q.device)
    diag = torch.eye(num_nodes, dtype=torch.bool, device=q.device)
    with_diag = strict_mask | diag

    cum_g = g.cumsum(dim=2) if linear_mask_leak else _path_cum_g(tree, g)
    decay = (cum_g.unsqueeze(-1) - cum_g.unsqueeze(-2)).exp()
    strict = strict_mask.view(1, 1, num_nodes, num_nodes)
    visible = with_diag.view(1, 1, num_nodes, num_nodes)

    interaction = (k * beta.unsqueeze(-1)) @ k.transpose(-1, -2)
    system = torch.eye(num_nodes, dtype=torch.float32, device=q.device).view(1, 1, num_nodes, num_nodes)
    system = system + torch.where(strict, interaction * decay, torch.zeros_like(interaction))
    solved_values = torch.linalg.solve_triangular(
        system,
        v * beta.unsqueeze(-1),
        upper=False,
    )
    solved_keys = torch.linalg.solve_triangular(
        system,
        (k * beta.unsqueeze(-1)) * cum_g.exp().unsqueeze(-1),
        upper=False,
    )
    incoming_memory = torch.einsum("bhvk,bhck->bhcv", state0, solved_keys)
    transformed_values = solved_values - incoming_memory

    inter_chunk = torch.einsum("bhvk,bhck->bhcv", state0, q * cum_g.exp().unsqueeze(-1))
    intra = (q @ k.transpose(-1, -2)) * decay
    output = inter_chunk + torch.where(visible, intra, torch.zeros_like(intra)) @ transformed_values

    state_rows = []
    for node_id in range(num_nodes):
        row_mask = with_diag[node_id].view(1, 1, num_nodes, 1).to(torch.float32)
        row_decay = (cum_g[:, :, node_id : node_id + 1] - cum_g).exp().unsqueeze(-1)
        decayed_keys = k * row_decay * row_mask
        state_i = state0 * cum_g[:, :, node_id, None, None].exp() + torch.einsum(
            "bhcv,bhck->bhvk", transformed_values, decayed_keys
        )
        state_rows.append(state_i)
    states = torch.cat(state_rows, dim=0).contiguous()
    output_blhv = output.transpose(1, 2).contiguous().to(initial_dtype)
    return {
        "state": states,
        "output": output_blhv,
        "logits": logits_from_output(output_blhv),
    }


def chunk_oracle_eval(
    tree: TreeDescriptor,
    tensors: dict[str, torch.Tensor],
    cfg: QwenGDNConfig,
) -> dict[str, torch.Tensor]:
    vllm_rule = load_vllm_cpu_rule()
    paths = [tree.path_to(n.node_id) for n in tree.nodes]
    flat_nodes = [idx for path in paths for idx in path]
    cu = [0]
    for path in paths:
        cu.append(cu[-1] + len(path))
    q = _select_nodes(tensors["q"], flat_nodes)
    k = _select_nodes(tensors["k"], flat_nodes)
    v = _select_nodes(tensors["v"], flat_nodes)
    g = _select_nodes(tensors["g"], flat_nodes)
    beta = _select_nodes(tensors["beta"], flat_nodes)
    initial = tensors["initial_state"].repeat(len(paths), 1, 1, 1)
    output, state = vllm_rule.chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial,
        scale=cfg.scale,
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=q.device),
        use_qk_l2norm_in_kernel=cfg.use_qk_l2norm_in_kernel,
    )
    node_outputs = []
    for end in cu[1:]:
        node_outputs.append(output[:, end - 1 : end])
    outputs = torch.cat(node_outputs, dim=1)
    return {"state": state, "output": outputs, "logits": logits_from_output(outputs)}


def assert_close_tree(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    for key in ("state", "output", "logits"):
        torch.testing.assert_close(actual[key], expected[key], atol=atol, rtol=rtol)


def accepted_path_final_state(tree: TreeDescriptor, result: dict[str, torch.Tensor], leaf_id: int) -> torch.Tensor:
    return result["state"][leaf_id]


def main() -> None:
    cfg = read_qwen_gdn_config()
    dtype = torch.float32
    tree = make_tree(total_nodes=8, spine_depth=4, seed=10)
    tensors = make_layer_inputs(cfg, len(tree.nodes), dtype=dtype, seed=11)
    serial = serial_per_path_eval(tree, tensors, cfg)
    packed = packed_tree_eval(tree, tensors, cfg)
    trunk = trunk_sharing_eval(tree, tensors, cfg)
    assert_close_tree(packed, serial, atol=1e-5, rtol=1e-5)
    assert_close_tree(trunk, serial, atol=1e-5, rtol=1e-5)
    print(
        json.dumps(
            {
                "qwen_gdn_config": cfg.__dict__,
                "nodes": len(tree.nodes),
                "max_abs_state_delta": float((packed["state"] - serial["state"]).abs().max()),
                "max_abs_logit_delta": float((packed["logits"] - serial["logits"]).abs().max()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

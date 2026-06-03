#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch


H = 48
HV = 48
K = 128
V = 128
PACKED_QKV = 2 * H * K + HV * V


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def _make_inputs(tokens: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    mixed_qkv = (
        torch.randn((tokens, PACKED_QKV), device=device, generator=gen, dtype=torch.float32)
        * 0.2
    ).to(torch.bfloat16).contiguous()
    a = (
        torch.randn((tokens, HV), device=device, generator=gen, dtype=torch.float32) * 0.1
        - 0.2
    ).to(torch.bfloat16).contiguous()
    b = torch.randn((tokens, HV), device=device, generator=gen, dtype=torch.float32).to(
        torch.bfloat16
    ).contiguous()
    # Production uses A_log and dt_bias from the layer; use stable synthetic
    # ranges that exercise the same dtype/cast path without extreme decays.
    a_log = (
        torch.randn((HV,), device=device, generator=gen, dtype=torch.float32) * 0.1 - 2.0
    ).contiguous()
    dt_bias = (
        torch.randn((HV,), device=device, generator=gen, dtype=torch.float32) * 0.1
    ).contiguous()
    state_bank = torch.zeros((2, HV, V, K), device=device, dtype=torch.float32).contiguous()
    state_bank[1] = (
        torch.randn((HV, V, K), device=device, generator=gen, dtype=torch.float32) * 0.05
    )
    return {
        "mixed_qkv": mixed_qkv,
        "a": a,
        "b": b,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "state_bank": state_bank,
    }


def _split_mixed_qkv(mixed_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = mixed_qkv.shape[0]
    q = mixed_qkv[:, : H * K].view(1, tokens, H, K).contiguous()
    k = mixed_qkv[:, H * K : 2 * H * K].view(1, tokens, H, K).contiguous()
    v = mixed_qkv[:, 2 * H * K :].view(1, tokens, HV, V).contiguous()
    return q, k, v


def _g_beta_from_decode_params(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirrors fused_recurrent_gated_delta_rule_packed_decode_kernel:
    # g = -exp(A_log) * softplus(a + dt_bias)
    # beta = sigmoid(b).to(b.dtype).to(fp32)
    g = -torch.exp(A_log.float()).view(1, 1, HV) * _softplus(
        a.float().view(1, a.shape[0], HV) + dt_bias.float().view(1, 1, HV)
    )
    beta = torch.sigmoid(b.float()).to(b.dtype).view(1, b.shape[0], HV).contiguous()
    return g.contiguous(), beta


def packed_decode_commit(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_bank: torch.Tensor,
    accepted_tokens: int,
    *,
    capture_one_token: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode

    out_rows: list[torch.Tensor] = []
    state_indices = torch.tensor([1], device=mixed_qkv.device, dtype=torch.int32)
    graph_bit_exact = False

    for token_idx in range(accepted_tokens):
        pre_state = state_bank.clone() if capture_one_token and token_idx == 0 else None
        out = torch.empty((1, 1, HV, V), device=mixed_qkv.device, dtype=mixed_qkv.dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[token_idx : token_idx + 1],
            a=a[token_idx : token_idx + 1],
            b=b[token_idx : token_idx + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=K**-0.5,
            initial_state=state_bank,
            out=out,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        if capture_one_token and token_idx == 0:
            assert pre_state is not None
            eager_out = out.clone()
            eager_state = state_bank.clone()
            graph_out = torch.empty_like(out)
            graph_state = pre_state.clone()
            graph = torch.cuda.CUDAGraph()
            # Capture a same-shape one-token canonical commit into a separate
            # state buffer. This is the graph-capture contract the integration
            # needs for warmed accepted-path lengths.
            with torch.cuda.graph(graph):
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv[token_idx : token_idx + 1],
                    a=a[token_idx : token_idx + 1],
                    b=b[token_idx : token_idx + 1],
                    A_log=A_log,
                    dt_bias=dt_bias,
                    scale=K**-0.5,
                    initial_state=graph_state,
                    out=graph_out,
                    ssm_state_indices=state_indices,
                    use_qk_l2norm_in_kernel=True,
            )
            graph_out.zero_()
            graph_state.copy_(pre_state)
            graph.replay()
            torch.cuda.synchronize()
            graph_bit_exact = bool(torch.equal(graph_out, eager_out) and torch.equal(graph_state, eager_state))
        out_rows.append(out.squeeze(0).squeeze(0).clone())
    return torch.stack(out_rows, dim=0), state_bank[1].clone(), graph_bit_exact


def recurrent_sequence_reference(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    accepted_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule

    q, k, v = _split_mixed_qkv(mixed_qkv[:accepted_tokens])
    g, beta = _g_beta_from_decode_params(a[:accepted_tokens], b[:accepted_tokens], A_log, dt_bias)
    state = initial_state[1:2].clone().contiguous()
    out, final_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state,
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return out.squeeze(0).float(), final_state[-1].float()


def run_probe(tokens: int, accepted_tokens: int, seed: int, capture: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for canonical state commit probe")
    if not (1 <= accepted_tokens <= tokens):
        raise ValueError(f"accepted_tokens must be in [1, {tokens}], got {accepted_tokens}")

    device = torch.device("cuda")
    inputs = _make_inputs(tokens, seed, device)
    commit_state_a = inputs["state_bank"].clone()
    commit_state_b = inputs["state_bank"].clone()

    out_a, state_a, graph_bit_exact = packed_decode_commit(
        inputs["mixed_qkv"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        commit_state_a,
        accepted_tokens,
        capture_one_token=capture,
    )
    out_b, state_b, _ = packed_decode_commit(
        inputs["mixed_qkv"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        commit_state_b,
        accepted_tokens,
        capture_one_token=False,
    )
    torch.cuda.synchronize()

    ref_out, ref_state = recurrent_sequence_reference(
        inputs["mixed_qkv"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        inputs["state_bank"],
        accepted_tokens,
    )
    return {
        "schema": "fr10.canonical_state_commit_probe.v1",
        "tokens": tokens,
        "accepted_tokens": accepted_tokens,
        "seed": seed,
        "device": torch.cuda.get_device_name(0),
        "input_dtype": "bfloat16",
        "state_dtype": "float32",
        "state_slots": 2,
        "valid_state_index": 1,
        "decode_kernel": "vllm.model_executor.layers.fla.ops.fused_recurrent_gated_delta_rule_packed_decode",
        "scale": K**-0.5,
        "use_qk_l2norm_in_kernel": True,
        "packed_replay_out_bit_exact": bool(torch.equal(out_a, out_b)),
        "packed_replay_state_bit_exact": bool(torch.equal(state_a, state_b)),
        "packed_replay_max_out_abs": (out_a.float() - out_b.float()).abs().max().item(),
        "packed_replay_max_state_abs": (state_a - state_b).abs().max().item(),
        "packed_vs_sequence_max_out_abs": (out_a.float() - ref_out).abs().max().item(),
        "packed_vs_sequence_max_state_abs": (state_a - ref_state).abs().max().item(),
        "graph_one_token_bit_exact": graph_bit_exact if capture else None,
        "state_read_bytes_per_token": HV * V * K * 4,
        "state_write_bytes_per_token": HV * V * K * 4,
        "accepted_path_state_read_bytes": accepted_tokens * HV * V * K * 4,
        "accepted_path_state_write_bytes": accepted_tokens * HV * V * K * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=14)
    parser.add_argument("--accepted-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    result = run_probe(args.tokens, args.accepted_tokens, args.seed, args.capture)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

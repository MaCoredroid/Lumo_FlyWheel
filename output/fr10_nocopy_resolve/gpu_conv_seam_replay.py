#!/usr/bin/env python3
"""FR11 Probe alpha: propagate the conv tap-dtype seam to residual hidden.

Boot-free GPU replay. This script reconstructs layer-0 linear-attention inputs
from captured hidden states and local safetensors, runs two conv branches:

  native: bf16 tap products, fp32 accumulate
  tree:   fp32 tap products, fp32 accumulate

Both branches then pass through the same real GDN recurrence, RMSNormGated,
real dequantized out_proj, and the layer-0 residual add.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LAYER_CAPTURE = REPO / "output/fr10_match0_layer_native_20260605T101521Z/logs/layer_native.call0.pt"
DEFAULT_HANDOFF = REPO / "output/fr10_conv_handoff_confirm_20260605T023734Z/logs/fr10_src_native_handoff.pt"
DEFAULT_MODEL = Path("/models/qwen3.6-27b-fp8")

HIDDEN = 5120
QK_HEADS = 16
V_HEADS = 48
HEAD_K = 128
HEAD_V = 128
CONV_W = 4
QKV_DIM = QK_HEADS * HEAD_K * 2 + V_HEADS * HEAD_V
VALUE_DIM = V_HEADS * HEAD_V
SCALE = HEAD_K**-0.5


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _rms(a: torch.Tensor, b: torch.Tensor) -> float:
    d = a.float() - b.float()
    return float(d.pow(2).mean().sqrt().item())


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(a.shape),
        "left_dtype": str(a.dtype),
        "right_dtype": str(b.dtype),
        "max_abs": _max_abs(a, b),
        "rms": _rms(a, b),
        "num_mismatched_elements": int((a != b).sum().item()) if a.shape == b.shape else None,
        "bit_exact": bool(a.shape == b.shape and torch.equal(a, b)),
    }


def _gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y * (1.0 + weight.float())
    return y.to(orig_dtype)


def _rms_norm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    orig_dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = y * weight.float()
    y = y * F.silu(z.float())
    return y.to(orig_dtype)


def _dequant_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape
    block_out = (out_features + scale_inv.shape[0] - 1) // scale_inv.shape[0]
    block_in = (in_features + scale_inv.shape[1] - 1) // scale_inv.shape[1]
    scales = scale_inv.float().repeat_interleave(block_out, dim=0)
    scales = scales[:out_features].repeat_interleave(block_in, dim=1)[:, :in_features]
    return weight.float() * scales


def _load_tensor(model_dir: Path, name: str) -> torch.Tensor:
    with safe_open(model_dir / "layers-0.safetensors", framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _linear_fp8(
    x: torch.Tensor,
    model_dir: Path,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    w = _load_tensor(model_dir, f"{name}.weight")
    s = _load_tensor(model_dir, f"{name}.weight_scale_inv")
    w_f = _dequant_block_fp8(w, s).to(device)
    y = x.to(device).float().matmul(w_f.t())
    del w_f, w, s
    return y


def _linear_bf16_weight(
    x: torch.Tensor,
    model_dir: Path,
    name: str,
    device: torch.device,
) -> torch.Tensor:
    w = _load_tensor(model_dir, f"{name}.weight").to(device).float()
    y = x.to(device).float().matmul(w.t())
    del w
    return y


def _conv_branches(
    x: torch.Tensor,
    prior_conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    read_col: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_bf = x.to(torch.bfloat16)
    w_bf = conv_weight.to(torch.bfloat16)
    hist_native = prior_conv_state[:, read_col : read_col + CONV_W - 1].to(torch.bfloat16)
    hist_tree = hist_native.clone()
    native_rows: list[torch.Tensor] = []
    tree_rows: list[torch.Tensor] = []
    for token in range(int(x_bf.shape[0])):
        window_native = torch.cat((hist_native, x_bf[token].view(-1, 1)), dim=1)
        window_tree = torch.cat((hist_tree, x_bf[token].view(-1, 1)), dim=1)
        acc_native = torch.zeros((QKV_DIM,), device=x.device, dtype=torch.float32)
        acc_tree = torch.zeros((QKV_DIM,), device=x.device, dtype=torch.float32)
        for col in range(CONV_W):
            prod_native = (
                window_native[:, col] * w_bf[:, col]
            ).to(torch.bfloat16)
            acc_native = acc_native + prod_native.float()
            acc_tree = acc_tree + window_tree[:, col].float() * w_bf[:, col].float()
        native_rows.append(F.silu(acc_native).to(torch.bfloat16))
        tree_rows.append(F.silu(acc_tree).to(torch.bfloat16))
        hist_native = torch.cat((hist_native[:, 1:], x_bf[token].view(-1, 1)), dim=1)
        hist_tree = torch.cat((hist_tree[:, 1:], x_bf[token].view(-1, 1)), dim=1)
    return torch.stack(native_rows), torch.stack(tree_rows)


def _split_qkv(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = x[:, : QK_HEADS * HEAD_K].reshape(1, -1, QK_HEADS, HEAD_K)
    k = x[:, QK_HEADS * HEAD_K : 2 * QK_HEADS * HEAD_K].reshape(
        1, -1, QK_HEADS, HEAD_K
    )
    v = x[:, 2 * QK_HEADS * HEAD_K :].reshape(1, -1, V_HEADS, HEAD_V)
    return q, k, v


def _gdn_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h0: torch.Tensor,
) -> torch.Tensor:
    q = q.squeeze(0).float()
    k = k.squeeze(0).float()
    v = v.squeeze(0).float()
    q = q * torch.rsqrt((q * q).sum(dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(dim=-1, keepdim=True) + 1e-6)
    repeat = V_HEADS // QK_HEADS
    q = q.repeat_interleave(repeat, dim=1) * SCALE
    k = k.repeat_interleave(repeat, dim=1)
    g = -torch.exp(a_log.float()).view(1, V_HEADS) * F.softplus(
        a.float() + dt_bias.float().view(1, V_HEADS), beta=1.0, threshold=20.0
    )
    beta = torch.sigmoid(b.float())
    state = h0.float().clone()
    outs: list[torch.Tensor] = []
    for token in range(int(q.shape[0])):
        state = state * torch.exp(g[token]).view(V_HEADS, 1, 1)
        kv = (state * k[token].view(V_HEADS, 1, HEAD_K)).sum(dim=-1)
        delta = (v[token] - kv) * beta[token].view(V_HEADS, 1)
        state = state + delta.view(V_HEADS, HEAD_V, 1) * k[token].view(
            V_HEADS, 1, HEAD_K
        )
        outs.append((state * q[token].view(V_HEADS, 1, HEAD_K)).sum(dim=-1))
    return torch.stack(outs, dim=0).to(torch.bfloat16)


def evaluate(
    *,
    layer_capture: Path,
    handoff_payload: Path,
    model_dir: Path,
    out: Path,
    tokens: int,
    eps: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Probe alpha GPU replay")
    device = torch.device("cuda")

    layer = torch.load(layer_capture, map_location="cpu", weights_only=False)
    handoff = torch.load(handoff_payload, map_location="cpu", weights_only=False)
    if not handoff.get("tree_parent"):
        raise RuntimeError("tree engagement assertion failed: missing tree_parent")
    if handoff.get("next_read_conv_state") is None:
        raise RuntimeError("handoff payload missing next_read_conv_state")
    n = min(tokens, int(layer["input_hidden"].shape[0]))
    input_hidden = layer["input_hidden"][:n].to(device).to(torch.bfloat16)
    residual_in = input_hidden.clone()

    prefix = "model.language_model.layers.0"
    input_norm_weight = _load_tensor(model_dir, f"{prefix}.input_layernorm.weight").to(device)
    hidden_norm = _gemma_rms_norm(input_hidden, input_norm_weight, eps)

    qkv_name = f"{prefix}.linear_attn.in_proj_qkv"
    z_name = f"{prefix}.linear_attn.in_proj_z"
    a_name = f"{prefix}.linear_attn.in_proj_a"
    b_name = f"{prefix}.linear_attn.in_proj_b"
    mixed_preconv = _linear_fp8(hidden_norm, model_dir, qkv_name, device).to(torch.bfloat16)
    z = _linear_fp8(hidden_norm, model_dir, z_name, device).to(torch.bfloat16)
    a = _linear_bf16_weight(hidden_norm, model_dir, a_name, device).to(torch.bfloat16)
    b = _linear_bf16_weight(hidden_norm, model_dir, b_name, device).to(torch.bfloat16)

    conv_weight = _load_tensor(model_dir, f"{prefix}.linear_attn.conv1d.weight")
    conv_weight = conv_weight.view(QKV_DIM, CONV_W).to(device)
    accepted_len = int(handoff["accepted_len"])
    read_col = max(0, accepted_len - 1)
    prior_conv = handoff["prev_conv_prior"].to(device)
    native_conv, tree_conv = _conv_branches(
        mixed_preconv, prior_conv, conv_weight, read_col=read_col
    )

    q_native, k_native, v_native = _split_qkv(native_conv)
    q_tree, k_tree, v_tree = _split_qkv(tree_conv)
    a_log = _load_tensor(model_dir, f"{prefix}.linear_attn.A_log").to(device).float()
    dt_bias = _load_tensor(model_dir, f"{prefix}.linear_attn.dt_bias").to(device)
    h0_row = handoff["prev_h0"].to(device)
    core_native = _gdn_recurrent(q_native, k_native, v_native, a, b, a_log, dt_bias, h0_row)
    core_tree = _gdn_recurrent(q_tree, k_tree, v_tree, a, b, a_log, dt_bias, h0_row)

    norm_weight = _load_tensor(model_dir, f"{prefix}.linear_attn.norm.weight").to(device)
    z_3d = z.reshape(n, V_HEADS, HEAD_V)
    gated_native = _rms_norm_gated(
        core_native.reshape(-1, HEAD_V),
        z_3d.reshape(-1, HEAD_V),
        norm_weight,
        eps,
    ).reshape(n, VALUE_DIM)
    gated_tree = _rms_norm_gated(
        core_tree.reshape(-1, HEAD_V),
        z_3d.reshape(-1, HEAD_V),
        norm_weight,
        eps,
    ).reshape(n, VALUE_DIM)

    out_name = f"{prefix}.linear_attn.out_proj"
    attn_native = _linear_fp8(gated_native, model_dir, out_name, device).to(torch.bfloat16)
    attn_tree = _linear_fp8(gated_tree, model_dir, out_name, device).to(torch.bfloat16)
    residual_native = (attn_native + residual_in).to(torch.bfloat16)
    residual_tree = (attn_tree + residual_in).to(torch.bfloat16)
    torch.cuda.synchronize()

    result = {
        "schema": "fr11.probe_alpha_conv_seam_residual_replay.v1",
        "layer_capture": str(layer_capture),
        "handoff_payload": str(handoff_payload),
        "model_dir": str(model_dir),
        "tokens": n,
        "tree_engaged": True,
        "accepted_len": accepted_len,
        "conv_read_col": read_col,
        "conv_window_columns": [read_col, read_col + CONV_W - 1],
        "input_hidden_absmean": float(input_hidden.float().abs().mean().item()),
        "mixed_preconv_absmean": float(mixed_preconv.float().abs().mean().item()),
        "conv_native_absmean": float(native_conv.float().abs().mean().item()),
        "core_native_absmean": float(core_native.float().abs().mean().item()),
        "attn_native_absmean": float(attn_native.float().abs().mean().item()),
        "residual_native_absmean": float(residual_native.float().abs().mean().item()),
        "conv_native_vs_tree": _stats(native_conv, tree_conv),
        "core_native_vs_tree": _stats(core_native, core_tree),
        "rmsnorm_gated_native_vs_tree": _stats(gated_native, gated_tree),
        "attn_out_native_vs_tree": _stats(attn_native, attn_tree),
        "residual_hidden_native_vs_tree": _stats(residual_native, residual_tree),
        "interpretation": (
            "PRECISION_FLOOR_CANDIDATE"
            if _max_abs(residual_native, residual_tree) >= 0.01
            else "CONV_SEAM_NOT_CAUSALLY_SUFFICIENT"
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-capture", type=Path, default=DEFAULT_LAYER_CAPTURE)
    parser.add_argument("--handoff-payload", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=REPO / "output/fr10_nocopy_resolve/gpu_conv_seam_replay_result.json")
    parser.add_argument("--tokens", type=int, default=11)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    args = parser.parse_args()
    result = evaluate(
        layer_capture=args.layer_capture,
        handoff_payload=args.handoff_payload,
        model_dir=args.model_dir,
        out=args.out,
        tokens=args.tokens,
        eps=args.eps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Same-boot byte gate for the fixed32 FA2 qrow16 candidate.

The input must be a paged-KV capture from a real fixed32 SWE-Verified B1
decode event.  B1 selects qrow16.  Repeating the same independent request to
B2 and B4 preserves its operands while forcing the stock dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


Q_ROWS = 32
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
PAGE_ROWS = 1024
EXACT_SAFE_FA2_SHA256 = "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"


def _load_capture(path: Path) -> dict[str, Any]:
    capture = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(capture, dict):
        raise TypeError("capture must be a dictionary")
    required = {
        "query",
        "key_cache",
        "value_cache",
        "block_table",
        "seq_lens",
        "max_seq_len",
        "tree_bias",
        "scale",
    }
    missing = sorted(required - capture.keys())
    if missing:
        raise KeyError(f"capture is missing keys: {missing}")
    return capture


def _validate(capture: dict[str, Any]) -> None:
    provenance = capture.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("capture provenance is required")
    if provenance.get("suite") != "SWE-Verified":
        raise ValueError("capture must come from SWE-Verified")
    if not provenance.get("instance_id"):
        raise ValueError("capture provenance has no instance_id")
    if provenance.get("concurrency") != 1 or provenance.get("physical_nodes") != 32:
        raise ValueError("capture must be fixed32 B1")
    if provenance.get("source_fa2_sha256") != EXACT_SAFE_FA2_SHA256:
        raise ValueError("capture source is not the pinned exact-safe FA2 binary")
    query = capture["query"]
    key_cache = capture["key_cache"]
    value_cache = capture["value_cache"]
    block_table = capture["block_table"]
    seq_lens = capture["seq_lens"]
    tree_bias = capture["tree_bias"]
    if tuple(query.shape) != (Q_ROWS, Q_HEADS, HEAD_DIM):
        raise ValueError(f"query shape is {tuple(query.shape)}")
    if query.dtype != torch.bfloat16:
        raise ValueError(f"query dtype is {query.dtype}, expected bfloat16")
    expected_cache_tail = (PAGE_ROWS, KV_HEADS, HEAD_DIM)
    if tuple(key_cache.shape[1:]) != expected_cache_tail:
        raise ValueError(f"key_cache tail is {tuple(key_cache.shape[1:])}")
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        raise ValueError("value_cache shape differs from key_cache")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("Q/K/V dtypes differ")
    if block_table.ndim != 2 or tuple(block_table.shape[:1]) != (1,):
        raise ValueError(f"block_table shape is {tuple(block_table.shape)}")
    if block_table.dtype != torch.int32:
        raise ValueError("block_table must be int32")
    if tuple(seq_lens.reshape(-1).shape) != (1,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be one int32 value")
    if tree_bias.ndim == 3:
        if tuple(tree_bias.shape) != (1, Q_ROWS, Q_ROWS):
            raise ValueError(f"tree_bias shape is {tuple(tree_bias.shape)}")
    elif tuple(tree_bias.shape) != (Q_ROWS, Q_ROWS):
        raise ValueError(f"tree_bias shape is {tuple(tree_bias.shape)}")
    if tree_bias.dtype != torch.float32:
        raise ValueError("tree_bias must be fp32")
    if int(capture["max_seq_len"]) < int(seq_lens.item()):
        raise ValueError("max_seq_len is smaller than seq_lens")


def _call(flash_attn_varlen_func: Any, capture: dict[str, Any], copies: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    query = capture["query"].to(device).repeat((copies, 1, 1)).contiguous()
    key_cache = capture["key_cache"].to(device).contiguous()
    value_cache = capture["value_cache"].to(device).contiguous()
    block_table = capture["block_table"].to(device).repeat((copies, 1)).contiguous()
    seq_lens = capture["seq_lens"].to(device).reshape(1).repeat(copies).contiguous()
    bias = capture["tree_bias"]
    if bias.ndim == 2:
        bias = bias.unsqueeze(0)
    bias = bias.to(device).repeat((copies, 1, 1)).contiguous()
    cu_q = torch.arange(
        0,
        (copies + 1) * Q_ROWS,
        Q_ROWS,
        dtype=torch.int32,
        device=device,
    )
    out, lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_q,
        max_seqlen_q=Q_ROWS,
        seqused_k=seq_lens,
        max_seqlen_k=int(capture["max_seq_len"]),
        softmax_scale=float(capture["scale"]),
        causal=False,
        window_size=[-1, -1],
        softcap=float(capture.get("softcap", 0.0)),
        block_table=block_table,
        num_splits=1,
        return_softmax_lse=True,
        fa_version=2,
        tree_bias=bias,
    )
    torch.cuda.synchronize()
    return out.detach().cpu(), lse.detach().cpu()


def _first_request(out: torch.Tensor, lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return out[:Q_ROWS].contiguous(), lse[:, :Q_ROWS].contiguous()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capture = _load_capture(args.capture)
    _validate(capture)
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    b1_out, b1_lse = _call(flash_attn_varlen_func, capture, 1)
    rows: dict[str, Any] = {}
    passed = True
    for copies in (2, 4):
        stock_out, stock_lse = _call(flash_attn_varlen_func, capture, copies)
        out0, lse0 = _first_request(stock_out, stock_lse)
        out_equal = torch.equal(b1_out, out0)
        lse_equal = torch.equal(b1_lse, lse0)
        replicas_equal = all(
            torch.equal(stock_out[:Q_ROWS], stock_out[i * Q_ROWS : (i + 1) * Q_ROWS])
            and torch.equal(stock_lse[:, :Q_ROWS], stock_lse[:, i * Q_ROWS : (i + 1) * Q_ROWS])
            for i in range(1, copies)
        )
        rows[f"b{copies}"] = {
            "output_byte_equal": out_equal,
            "lse_byte_equal": lse_equal,
            "stock_replicas_byte_equal": replicas_equal,
            "output_max_abs": float((b1_out.float() - out0.float()).abs().max().item()),
            "lse_max_abs": float((b1_lse.float() - lse0.float()).abs().max().item()),
        }
        passed &= out_equal and lse_equal and replicas_equal
    result = {
        "schema": "fr13.fixed32.fa2_qrow16_same_boot_byte_ab.v1",
        "capture": str(args.capture),
        "capture_provenance": capture.get("provenance"),
        "candidate_arm": "B1 qrow16",
        "stock_arms": ["B2 duplicated request", "B4 duplicated request"],
        "passed": passed,
        "comparisons": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

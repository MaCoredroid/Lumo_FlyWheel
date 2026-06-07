#!/usr/bin/env python3
"""Compare stock FA2 and the FR13 fork when no tree bias is supplied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    delta = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "torch_equal": bool(torch.equal(a, b)),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "nonzero": int((delta != 0).sum().item()) if delta.numel() else 0,
    }


def make_input(path: Path) -> int:
    torch.manual_seed(1313)
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        q_lens = [9, 3]
        k_lens = [71, 17]
        total_q = sum(q_lens)
        total_k = sum(k_lens)
        q = torch.randn(total_q, 64, 128, dtype=dtype)
        k = torch.randn(total_k, 8, 128, dtype=dtype)
        v = torch.randn(total_k, 8, 128, dtype=dtype)
        cu_q = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32)
        cu_k = torch.tensor([0, k_lens[0], total_k], dtype=torch.int32)
        cases.append(
            {
                "name": str(dtype).replace("torch.", ""),
                "q": q,
                "k": k,
                "v": v,
                "cu_q": cu_q,
                "cu_k": cu_k,
                "max_q": max(q_lens),
                "max_k": max(k_lens),
                "scale": 1.0 / (128**0.5),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "fr13.fa2_no_bias_input.v1", "cases": cases}, path)
    print(path)
    return 0


def run_one(input_path: Path, output_path: Path, label: str) -> int:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    payload = torch.load(input_path, map_location="cpu")
    results = []
    for case in payload["cases"]:
        q = case["q"].cuda()
        k = case["k"].cuda()
        v = case["v"].cuda()
        cu_q = case["cu_q"].cuda()
        cu_k = case["cu_k"].cuda()
        out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(case["max_q"]),
            max_seqlen_k=int(case["max_k"]),
            softmax_scale=float(case["scale"]),
            causal=True,
            window_size=[-1, -1],
            softcap=0.0,
            fa_version=2,
        )
        torch.cuda.synchronize()
        results.append(
            {
                "name": case["name"],
                "out": out.detach().cpu(),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "fr13.fa2_no_bias_output.v1",
            "label": label,
            "results": results,
        },
        output_path,
    )
    print(output_path)
    return 0


def compare(stock_path: Path, fork_path: Path, out_json: Path) -> int:
    stock = torch.load(stock_path, map_location="cpu")
    fork = torch.load(fork_path, map_location="cpu")
    rows = []
    passed = True
    for stock_case, fork_case in zip(stock["results"], fork["results"], strict=True):
        if stock_case["name"] != fork_case["name"]:
            raise RuntimeError(
                f"case mismatch: {stock_case['name']} != {fork_case['name']}"
            )
        stats = _stats(stock_case["out"], fork_case["out"])
        passed = passed and bool(stats["torch_equal"]) and stats["max_abs"] == 0.0
        rows.append({"name": stock_case["name"], "output": stats})
    result = {
        "schema": "fr13.fa2_no_bias_pristine_compare.v1",
        "stock_output": str(stock_path),
        "fork_output": str(fork_path),
        "passed": bool(passed),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    make = sub.add_parser("make-input")
    make.add_argument("--out", required=True, type=Path)

    run = sub.add_parser("run-one")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--label", required=True)

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--stock", required=True, type=Path)
    cmp_parser.add_argument("--fork", required=True, type=Path)
    cmp_parser.add_argument("--out-json", required=True, type=Path)

    args = parser.parse_args()
    if args.cmd == "make-input":
        return make_input(args.out)
    if args.cmd == "run-one":
        return run_one(args.input, args.out, args.label)
    if args.cmd == "compare":
        return compare(args.stock, args.fork, args.out_json)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())

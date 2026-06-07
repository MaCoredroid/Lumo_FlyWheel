#!/usr/bin/env python3
"""Boot-free FA2 references for a captured TREE_ATTN decode event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


TREE_PARENT = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]
TREE_SPINE_ROWS = [0, 1, 2, 4, 6]


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    d = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }


def _ancestors(parent: list[int], row: int) -> list[int]:
    path = []
    cur = row
    while cur >= 0:
        path.append(cur)
        cur = parent[cur]
    return list(reversed(path))


def _parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _dense_spine(
    cap: dict[str, Any],
    spine_rows: list[int],
    kv_rows: torch.Tensor,
    context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = cap["query"].index_select(0, torch.tensor(spine_rows)).to(torch.bfloat16)
    k = cap["dense_key"][0].index_select(0, kv_rows).to(torch.bfloat16)
    v = cap["dense_value"][0].index_select(0, kv_rows).to(torch.bfloat16)
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    out = torch.empty_like(qf)
    lse = torch.empty((len(spine_rows), qf.shape[1]), dtype=torch.float32)
    n_q_per_kv = int(cap["num_queries_per_kv"])
    scale = float(cap["scale"])
    for qi in range(len(spine_rows)):
        allowed = context_len + qi + 1
        for head in range(qf.shape[1]):
            kv_head = head // n_q_per_kv
            scores = (qf[qi, head].unsqueeze(0) @ kf[:, kv_head].T).squeeze(0)
            scores = scores * scale
            scores = scores.masked_fill(
                torch.arange(kf.shape[0]) >= allowed, float("-inf")
            )
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
            out[qi, head] = probs @ vf[:, kv_head]
            lse[qi, head] = torch.logsumexp(scores, dim=-1)
    return out, lse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-op-capture", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-tensor", type=Path)
    parser.add_argument("--parent", default=",".join(str(x) for x in TREE_PARENT))
    parser.add_argument("--spine-rows", default=",".join(str(x) for x in TREE_SPINE_ROWS))
    args = parser.parse_args()

    from vllm.v1.attention.backends.fa_utils import (  # noqa: PLC0415
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    cap = torch.load(args.tree_op_capture, map_location="cpu")
    parent = _parse_ints(args.parent)
    spine_rows = _parse_ints(args.spine_rows)
    q_len = int(cap["query"].shape[0])
    seq_len = int(cap["seq_lens"][0].item())
    context_len = seq_len - q_len
    scale = float(cap["scale"])
    tree = cap["output"].to(torch.float32)
    q_all = cap["query"].to(torch.bfloat16)
    k_all = cap["dense_key"][0].to(torch.bfloat16)
    v_all = cap["dense_value"][0].to(torch.bfloat16)

    cu_q_single = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    refs = []
    path_refs = []
    rows = []
    for row in range(q_len):
        path = _ancestors(parent, row)
        kv_rows = torch.tensor(
            list(range(context_len)) + [context_len + x for x in path],
            dtype=torch.long,
        )
        q = q_all[row : row + 1].cuda()
        k = k_all.index_select(0, kv_rows).cuda()
        v = v_all.index_select(0, kv_rows).cuda()
        cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda")
        out = torch.empty_like(q)
        ret = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q_single,
            max_seqlen_q=1,
            cu_seqlens_k=cu_k,
            max_seqlen_k=int(k.shape[0]),
            softmax_scale=scale,
            causal=True,
            window_size=[-1, -1],
            softcap=0.0,
            fa_version=2,
        )
        if isinstance(ret, torch.Tensor):
            out = ret
        ref = out.detach().cpu().to(torch.float32)[0]
        refs.append(ref)

        path_q_rows = torch.tensor(path, dtype=torch.long)
        q_path = q_all.index_select(0, path_q_rows).cuda()
        k_path = k_all.index_select(0, kv_rows).cuda()
        v_path = v_all.index_select(0, kv_rows).cuda()
        cu_q_path = torch.tensor([0, len(path)], dtype=torch.int32, device="cuda")
        out_path = torch.empty_like(q_path)
        ret_path = flash_attn_varlen_func(
            q=q_path,
            k=k_path,
            v=v_path,
            out=out_path,
            cu_seqlens_q=cu_q_path,
            max_seqlen_q=len(path),
            cu_seqlens_k=cu_k,
            max_seqlen_k=int(k_path.shape[0]),
            softmax_scale=scale,
            causal=True,
            window_size=[-1, -1],
            softcap=0.0,
            fa_version=2,
        )
        if isinstance(ret_path, torch.Tensor):
            out_path = ret_path
        path_ref = out_path.detach().cpu().to(torch.float32)[-1]
        path_refs.append(path_ref)
        rows.append(
            {
                "row": row,
                "parent": parent[row] if row < len(parent) else None,
                "path": path,
                "kv_len": int(k.shape[0]),
                "tree_vs_fa2": _stats(tree[row], ref),
                "tree_vs_fa2_path": _stats(tree[row], path_ref),
            }
        )

    fa2_rows = torch.stack(refs, dim=0)
    fa2_path_rows = torch.stack(path_refs, dim=0)

    spine = torch.tensor(spine_rows, dtype=torch.long)
    spine_kv_rows = torch.tensor(
        list(range(context_len)) + [context_len + x for x in spine_rows],
        dtype=torch.long,
    )
    q = q_all.index_select(0, spine).cuda()
    k = k_all.index_select(0, spine_kv_rows).cuda()
    v = v_all.index_select(0, spine_kv_rows).cuda()
    cu_q = torch.tensor([0, len(spine_rows)], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda")
    spine_fa2, spine_lse = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        max_seqlen_q=len(spine_rows),
        cu_seqlens_k=cu_k,
        max_seqlen_k=int(k.shape[0]),
        softmax_scale=scale,
        causal=True,
        window_size=[-1, -1],
        softcap=0.0,
        fa_version=2,
        return_softmax_lse=True,
    )
    spine_fa2 = spine_fa2.detach().cpu().to(torch.float32)
    spine_lse = spine_lse.detach().cpu().to(torch.float32).T.contiguous()
    dense_spine, dense_lse = _dense_spine(cap, spine_rows, spine_kv_rows, context_len)
    tree_spine = tree.index_select(0, spine)

    payload = {
        "schema": "fr13.fa2_tree_path_ref.v1",
        "capture": str(args.tree_op_capture),
        "fa_version": get_flash_attn_version(),
        "context_len": context_len,
        "spine_rows": spine_rows,
        "tree_vs_fa2_all_rows": _stats(tree, fa2_rows),
        "tree_vs_fa2_path_rows": _stats(tree, fa2_path_rows),
        "tree_vs_fa2_spine": _stats(tree_spine, spine_fa2),
        "fa2_spine_vs_dense_spine": _stats(spine_fa2, dense_spine),
        "fa2_lse_vs_dense_lse": _stats(spine_lse, dense_lse),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.out_tensor:
        torch.save(
            {
                "fa2_rows": fa2_rows,
                "spine_fa2": spine_fa2,
                "dense_spine": dense_spine,
                "tree": tree,
                "rows": rows,
            },
            args.out_tensor,
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

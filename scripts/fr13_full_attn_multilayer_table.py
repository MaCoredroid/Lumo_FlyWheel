#!/usr/bin/env python3
"""Build FR13 full-attention multi-layer drift table from capture artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


TREE_PARENT = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]
TREE_SPINE_ROWS = [0, 1, 2, 4, 6]
NATIVE_SPINE_ROWS = [0, 1, 2, 3, 4]
STAGES = [
    "input_hidden",
    "qkv_proj",
    "q_norm_out",
    "k_norm_out",
    "v",
    "positions",
    "q_after_rope",
    "k_after_rope",
    "attn_out_raw",
    "gate_sigmoid",
    "attn_out_gated",
    "o_proj_out",
]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _layer_idx(name: str) -> int:
    match = re.search(r"layers\.(\d+)\.self_attn", name)
    if not match:
        raise ValueError(f"cannot parse layer index from {name!r}")
    return int(match.group(1))


def _stage(cap: dict[str, Any], stage: str) -> torch.Tensor | None:
    item = cap.get("stages", {}).get(stage)
    if item is None:
        return None
    return item["tensor"]


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.float() - b.float()).abs().mean().item())


def _index_rows(t: torch.Tensor, rows: list[int]) -> torch.Tensor:
    return t.index_select(0, torch.tensor(rows, dtype=torch.long))


def _position_tokens(pos: torch.Tensor) -> torch.Tensor:
    if pos.ndim == 2:
        return pos[0]
    return pos.reshape(-1)


def _stage_diff(
    tree: torch.Tensor,
    native: torch.Tensor,
    *,
    tree_rows: list[int],
    native_rows: list[int],
) -> dict[str, Any]:
    if tree.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        tree = _position_tokens(tree)
        native = _position_tokens(native)
    tr = _index_rows(tree, tree_rows)
    nr = _index_rows(native, native_rows)
    by_depth = []
    for depth, (tree_row, native_row) in enumerate(zip(tree_rows, native_rows)):
        a = tree[tree_row]
        b = native[native_row]
        by_depth.append({
            "depth": int(depth),
            "tree_row": int(tree_row),
            "native_row": int(native_row),
            "max_abs": _max_abs(a, b),
            "mean_abs": _mean_abs(a, b),
        })
    return {
        "max_abs": _max_abs(tr, nr),
        "mean_abs": _mean_abs(tr, nr),
        "by_depth": by_depth,
    }


def _captures_by_layer(paths: list[Path], key: str) -> dict[int, dict[str, Any]]:
    out = {}
    for path in paths:
        cap = _load(path)
        name = str(cap.get(key) or cap.get("layer_prefix") or "")
        out[_layer_idx(name)] = cap | {"_capture_path": str(path)}
    return out


def _ancestors(parent: list[int], row: int) -> list[int]:
    path = []
    cur = int(row)
    while cur >= 0:
        path.append(cur)
        cur = int(parent[cur])
    return list(reversed(path))


def _fa2_tree_path_rows(cap: dict[str, Any], parent: list[int]) -> dict[str, Any] | None:
    try:
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None
    q_all = cap["query"].to(torch.bfloat16)
    k_all = cap["dense_key"][0].to(torch.bfloat16)
    v_all = cap["dense_value"][0].to(torch.bfloat16)
    tree = cap["output"].to(torch.float32)
    q_len = int(q_all.shape[0])
    seq_len = int(cap["seq_lens"][0].item())
    context_len = seq_len - q_len
    scale = float(cap["scale"])
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    rows = []
    refs = []
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
        out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            max_seqlen_q=1,
            cu_seqlens_k=cu_k,
            max_seqlen_k=int(k.shape[0]),
            softmax_scale=scale,
            causal=True,
            window_size=[-1, -1],
            softcap=0.0,
            fa_version=2,
        )
        ref = out.detach().cpu().float()[0]
        refs.append(ref)
        rows.append({
            "row": int(row),
            "path": path,
            "max_abs": _max_abs(tree[row], ref),
            "mean_abs": _mean_abs(tree[row], ref),
        })
    refs_t = torch.stack(refs, dim=0)
    branch_rows = [idx for idx in range(q_len) if idx not in TREE_SPINE_ROWS]
    return {
        "all_rows_max_abs": _max_abs(tree, refs_t),
        "spine_max_abs": _max_abs(_index_rows(tree, TREE_SPINE_ROWS), _index_rows(refs_t, TREE_SPINE_ROWS)),
        "branch_max_abs": _max_abs(_index_rows(tree, branch_rows), _index_rows(refs_t, branch_rows)) if branch_rows else 0.0,
        "rows": rows,
    }


def _dense_spine_math(tree_op: dict[str, Any], native_op: dict[str, Any]) -> dict[str, float]:
    tree_q = _index_rows(tree_op["query"], TREE_SPINE_ROWS).float()
    native_q = _index_rows(native_op["query"], NATIVE_SPINE_ROWS).float()
    tree_k = tree_op["dense_key"][0].float()
    native_k = native_op["dense_key"][0].float()
    tree_v = tree_op["dense_value"][0].float()
    native_v = native_op["dense_value"][0].float()
    scale = float(tree_op["scale"])
    q_len = int(tree_op["query"].shape[0])
    tree_context = int(tree_op["seq_lens"][0].item()) - q_len
    native_context = int(native_op["seq_lens"][0].item()) - int(native_op["query"].shape[0])
    qk_max = 0.0
    p_max = 0.0
    pv_max = 0.0
    for depth in range(len(TREE_SPINE_ROWS)):
        t_allowed = tree_context + depth + 1
        n_allowed = native_context + depth + 1
        n_q_per_kv = int(tree_op["num_queries_per_kv"])
        t_scores_rows = []
        n_scores_rows = []
        for head in range(tree_q.shape[1]):
            kvh = head // n_q_per_kv
            t_scores_rows.append(tree_k[:t_allowed, kvh].float() @ tree_q[depth, head].float())
            n_scores_rows.append(native_k[:n_allowed, kvh].float() @ native_q[depth, head].float())
        t_scores = torch.stack(t_scores_rows, dim=0)
        n_scores = torch.stack(n_scores_rows, dim=0)
        t_scores = t_scores * scale
        n_scores = n_scores * scale
        min_len = min(t_scores.shape[-1], n_scores.shape[-1])
        qk_max = max(qk_max, _max_abs(t_scores[:, :min_len], n_scores[:, :min_len]))
        t_p = torch.softmax(t_scores, dim=-1)
        n_p = torch.softmax(n_scores, dim=-1)
        p_max = max(p_max, _max_abs(t_p[:, :min_len], n_p[:, :min_len]))
        # Dense PV proxy using query-head-to-kv-head mapping.
        t_pv = []
        n_pv = []
        for head in range(tree_q.shape[1]):
            kvh = head // n_q_per_kv
            t_pv.append(t_p[head] @ tree_v[:t_allowed, kvh].float())
            n_pv.append(n_p[head] @ native_v[:n_allowed, kvh].float())
        pv_max = max(pv_max, _max_abs(torch.stack(t_pv), torch.stack(n_pv)))
    return {
        "dense_qk_scores_spine_max_abs": qk_max,
        "dense_softmax_p_spine_max_abs": p_max,
        "dense_pv_spine_max_abs": pv_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-full", nargs="+", required=True, type=Path)
    parser.add_argument("--native-full", nargs="+", required=True, type=Path)
    parser.add_argument("--tree-op", nargs="*", default=[], type=Path)
    parser.add_argument("--native-op", nargs="*", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    tree_full = _captures_by_layer(args.tree_full, "layer_prefix")
    native_full = _captures_by_layer(args.native_full, "layer_prefix")
    tree_ops = _captures_by_layer(args.tree_op, "layer_name") if args.tree_op else {}
    native_ops = _captures_by_layer(args.native_op, "layer_name") if args.native_op else {}

    layers = []
    for layer in sorted(set(tree_full) & set(native_full)):
        tcap = tree_full[layer]
        ncap = native_full[layer]
        stage_rows = []
        first = None
        for stage in STAGES:
            tt = _stage(tcap, stage)
            nt = _stage(ncap, stage)
            if tt is None or nt is None:
                continue
            stats = _stage_diff(
                tt,
                nt,
                tree_rows=TREE_SPINE_ROWS,
                native_rows=NATIVE_SPINE_ROWS,
            )
            row = {
                "stage": stage,
                "max_abs": stats["max_abs"],
                "mean_abs": stats["mean_abs"],
                "by_depth": stats["by_depth"],
            }
            stage_rows.append(row)
            if first is None and stats["max_abs"] > args.threshold:
                first = row
        op_math = {}
        if layer in tree_ops and layer in native_ops:
            op_math.update(_dense_spine_math(tree_ops[layer], native_ops[layer]))
        tree_vs_fa2 = None
        if layer in tree_ops:
            tree_vs_fa2 = _fa2_tree_path_rows(tree_ops[layer], TREE_PARENT)
        layers.append({
            "layer": int(layer),
            "tree_capture": tcap["_capture_path"],
            "native_capture": ncap["_capture_path"],
            "first_nonzero_stage": None if first is None else {
                "stage": first["stage"],
                "max_abs": first["max_abs"],
            },
            "stages": stage_rows,
            "dense_math_spine": op_math,
            "tree_vs_fa2_path_rows": tree_vs_fa2,
        })

    out = {
        "schema": "fr13.full_attn_multilayer_table.v1",
        "tree_spine_rows": TREE_SPINE_ROWS,
        "native_spine_rows": NATIVE_SPINE_ROWS,
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "layers": [
            {
                "layer": row["layer"],
                "first_nonzero_stage": row["first_nonzero_stage"],
                "attn_out_raw": next(
                    (s["max_abs"] for s in row["stages"] if s["stage"] == "attn_out_raw"),
                    None,
                ),
                "o_proj_out": next(
                    (s["max_abs"] for s in row["stages"] if s["stage"] == "o_proj_out"),
                    None,
                ),
                "tree_vs_fa2_branch": None
                if row["tree_vs_fa2_path_rows"] is None
                else row["tree_vs_fa2_path_rows"]["branch_max_abs"],
            }
            for row in layers
        ]
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

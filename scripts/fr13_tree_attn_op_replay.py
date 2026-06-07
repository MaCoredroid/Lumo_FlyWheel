#!/usr/bin/env python3
"""Replay FR13 TREE_ATTN op captures with dense masked attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


TREE_PARENT = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]
TREE_SPINE_ROWS = [0, 1, 2, 4, 6]
NATIVE_SPINE_ROWS = [0, 1, 2, 3, 4]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def _parse_rows(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _stage(capture: dict[str, Any], name: str) -> torch.Tensor | None:
    item = capture.get("stages", {}).get(name)
    if item is None:
        return None
    return item["tensor"]


def _apply_softcap(scores: torch.Tensor, softcap: float) -> torch.Tensor:
    if softcap <= 0:
        return scores
    return softcap * torch.tanh(scores / softcap)


def replay_dense(capture: dict[str, Any]) -> dict[str, Any]:
    query = capture["query"].to(torch.float32)
    kernel_output = capture["output"].to(torch.float32)
    q_start = capture["query_start_loc"].to(torch.long)
    seq_lens = capture["seq_lens"].to(torch.long)
    dense_key = [x.to(torch.float32) for x in capture["dense_key"]]
    dense_value = [x.to(torch.float32) for x in capture["dense_value"]]
    bias = capture.get("tree_attn_bias")
    if bias is not None:
        bias = bias.to(torch.float32)
    scale = float(capture["scale"])
    softcap = float(capture.get("logits_soft_cap", 0.0))
    num_queries_per_kv = int(capture["num_queries_per_kv"])
    sliding_window = tuple(int(x) for x in capture.get("sliding_window", (-1, -1)))
    sliding_window_size = 1 + sliding_window[0] if sliding_window[0] >= 0 else 0

    dense_output = torch.empty_like(kernel_output)
    seq_summaries = []
    qk_min = float("inf")
    qk_max = float("-inf")
    p_min = float("inf")
    p_max = float("-inf")

    for seq_idx in range(int(seq_lens.numel())):
        qs = int(q_start[seq_idx].item())
        qe = int(q_start[seq_idx + 1].item())
        q_seq = query[qs:qe]
        out_seq = dense_output[qs:qe]
        key_seq = dense_key[seq_idx]
        value_seq = dense_value[seq_idx]
        q_len = int(q_seq.shape[0])
        seq_len = int(seq_lens[seq_idx].item())
        context_len = seq_len - q_len
        if key_seq.shape[0] != seq_len or value_seq.shape[0] != seq_len:
            raise ValueError(
                f"seq {seq_idx}: dense KV len mismatch "
                f"k={key_seq.shape[0]} v={value_seq.shape[0]} seq_len={seq_len}"
            )
        seq_pv = []
        last_scores = None
        last_probs = None
        for q_row in range(q_len):
            head_out = []
            q_abs = context_len + q_row
            key_pos = torch.arange(seq_len, dtype=torch.long)
            base_mask = key_pos <= q_abs
            if sliding_window_size > 0:
                base_mask &= (q_abs - key_pos) < sliding_window_size
            for q_head in range(int(q_seq.shape[1])):
                kv_head = q_head // num_queries_per_kv
                scores = (
                    q_seq[q_row, q_head].unsqueeze(0)
                    @ key_seq[:, kv_head, :].transpose(0, 1)
                ).squeeze(0)
                scores = _apply_softcap(scores * scale, softcap)
                scores = scores.masked_fill(~base_mask, float("-inf"))
                if bias is not None:
                    qkey_len = min(q_len, int(bias.shape[1]))
                    if qkey_len > 0:
                        scores[context_len : context_len + qkey_len] += bias[
                            min(q_row, int(bias.shape[0]) - 1), :qkey_len
                        ]
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
                pv = probs @ value_seq[:, kv_head, :]
                head_out.append(pv)
                last_scores = scores
                last_probs = probs
            seq_pv.append(torch.stack(head_out, dim=0))
        out_seq.copy_(torch.stack(seq_pv, dim=0))
        if last_scores is not None:
            finite = last_scores[torch.isfinite(last_scores)]
            if finite.numel():
                qk_min = min(qk_min, float(finite.min().item()))
                qk_max = max(qk_max, float(finite.max().item()))
        if last_probs is not None:
            p_min = min(p_min, float(last_probs.min().item()))
            p_max = max(p_max, float(last_probs.max().item()))
        seq_summaries.append(
            {
                "seq_idx": seq_idx,
                "q_start": qs,
                "q_end": qe,
                "q_len": q_len,
                "seq_len": seq_len,
                "context_len": context_len,
                "kernel_vs_dense_max_abs": _max_abs(kernel_output[qs:qe], out_seq),
                "kernel_vs_dense_mean_abs": _mean_abs(kernel_output[qs:qe], out_seq),
            }
        )

    return {
        "dense_output": dense_output,
        "summary": {
            "kernel_vs_dense_max_abs": _max_abs(kernel_output, dense_output),
            "kernel_vs_dense_mean_abs": _mean_abs(kernel_output, dense_output),
            "qk_scores_finite_min": None if qk_min == float("inf") else qk_min,
            "qk_scores_finite_max": None if qk_max == float("-inf") else qk_max,
            "softmax_prob_min": None if p_min == float("inf") else p_min,
            "softmax_prob_max": None if p_max == float("-inf") else p_max,
            "seqs": seq_summaries,
        },
    }


def row_compare(
    tree_tensor: torch.Tensor,
    native_tensor: torch.Tensor,
    tree_rows: list[int],
    native_rows: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for depth, (tree_row, native_row) in enumerate(zip(tree_rows, native_rows)):
        rows.append(
            {
                "depth": depth,
                "tree_row": tree_row,
                "native_row": native_row,
                "max_abs": _max_abs(tree_tensor[tree_row], native_tensor[native_row]),
                "mean_abs": _mean_abs(tree_tensor[tree_row], native_tensor[native_row]),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-op-capture", required=True, type=Path)
    parser.add_argument("--native-full-attn-capture", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tree-rows", default=",".join(str(x) for x in TREE_SPINE_ROWS))
    parser.add_argument(
        "--native-rows", default=",".join(str(x) for x in NATIVE_SPINE_ROWS)
    )
    args = parser.parse_args()

    capture = _load(args.tree_op_capture)
    replay = replay_dense(capture)
    dense = replay["dense_output"]
    kernel = capture["output"].to(torch.float32)
    tree_rows = _parse_rows(args.tree_rows)
    native_rows = _parse_rows(args.native_rows)

    out: dict[str, Any] = {
        "schema": "fr13.tree_attn_op_replay.v1",
        "tree_op_capture": str(args.tree_op_capture),
        "layer_name": capture.get("layer_name"),
        "summary": replay["summary"],
        "kernel_vs_dense_by_spine_depth": row_compare(
            kernel.reshape(kernel.shape[0], -1),
            dense.reshape(dense.shape[0], -1),
            tree_rows,
            tree_rows,
        ),
    }

    if args.native_full_attn_capture:
        native = _load(args.native_full_attn_capture)
        native_attn = _stage(native, "attn_out_raw")
        if native_attn is not None:
            out["dense_tree_vs_native_attn_out_by_spine_depth"] = row_compare(
                dense.reshape(dense.shape[0], -1),
                native_attn.reshape(native_attn.shape[0], -1).to(torch.float32),
                tree_rows,
                native_rows,
            )
            out["kernel_tree_vs_native_attn_out_by_spine_depth"] = row_compare(
                kernel.reshape(kernel.shape[0], -1),
                native_attn.reshape(native_attn.shape[0], -1).to(torch.float32),
                tree_rows,
                native_rows,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

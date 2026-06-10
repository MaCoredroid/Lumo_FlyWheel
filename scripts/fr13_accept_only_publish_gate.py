#!/usr/bin/env python3
"""Validate accepted-only GDN recurrent-state publish semantics.

This is a deterministic reducer for the FR13 accepted-path publish change. It
compares the new committer-side accepted-only `ssm_state.index_copy_` against the
previous all-tree-row publish, but only on rows that survive into the next
`launch_tree_state_linear_remap` read (`pid_k < accepted_len`). Rejected rows are
checked separately to prove the durable write reduction is real.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = (left.float() - right.float()).abs()
    return {
        "torch_equal": bool(torch.equal(left, right)),
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "nonzero": int((delta != 0).sum().item()) if delta.numel() else 0,
    }


def evaluate() -> dict[str, Any]:
    torch.manual_seed(1313)
    batch = 3
    tree_n = 10
    bank_rows = 64
    row_shape = (2, 3, 5)
    spec_state_indices = torch.tensor(
        [
            [17, 3, 41, 8, 22, 9, 55, 12, 29, 6],
            [4, 18, 31, 5, 47, 11, 53, 7, 36, 20],
            [1, 14, 27, 33, 42, 48, 54, 60, 10, 24],
        ],
        dtype=torch.long,
    )
    accepted_paths = [
        [1, 2, 4, 6],
        [1],
        [],
    ]
    initial = torch.randn((bank_rows, *row_shape), dtype=torch.float32).bfloat16()
    tree_state = torch.randn((batch, tree_n, *row_shape), dtype=torch.float32)

    old_ssm = initial.clone()
    for b in range(batch):
        old_ssm.index_copy_(
            0,
            spec_state_indices[b, :tree_n],
            tree_state[b, :tree_n].to(dtype=old_ssm.dtype),
        )

    new_ssm = initial.clone()
    total_old_rows = 0
    total_new_rows = 0
    accepted_bank_rows: list[int] = []
    rejected_bank_rows: list[int] = []
    per_batch = []
    for b, path in enumerate(accepted_paths):
        total_old_rows += tree_n
        # Row 0 is live even when accepted_len is zero: the next h0 read clamps
        # accepted_len - 1 to column 0. The optimized durable publish keeps row
        # 0 plus the accepted path rows and skips only truly rejected branches.
        publish_path = sorted(set([0, *path]))
        path_t = torch.tensor(publish_path, dtype=torch.long)
        if path_t.numel():
            bank_t = spec_state_indices[b].index_select(0, path_t)
            vals = tree_state[b].index_select(0, path_t).to(dtype=new_ssm.dtype)
            new_ssm.index_copy_(0, bank_t, vals)
            accepted_bank_rows.extend(int(x) for x in bank_t.tolist())
            total_new_rows += int(path_t.numel())
        rejected = sorted(set(range(tree_n)) - set(int(x) for x in publish_path))
        rejected_bank_rows.extend(int(x) for x in spec_state_indices[b, rejected].tolist())
        accepted_stats = (
            _stats(
                old_ssm.index_select(0, bank_t),
                new_ssm.index_select(0, bank_t),
            )
            if path_t.numel()
            else {"torch_equal": True, "max_abs": 0.0, "nonzero": 0}
        )
        per_batch.append(
            {
                "batch": b,
                "accepted_path": path,
                "published_path": publish_path,
                "accepted_rows": len(path),
                "published_rows": int(path_t.numel()),
                "accepted_rows_vs_old_publish": accepted_stats,
            }
        )

    accepted_idx = torch.tensor(accepted_bank_rows, dtype=torch.long)
    rejected_idx = torch.tensor(rejected_bank_rows, dtype=torch.long)
    accepted_stats = _stats(
        old_ssm.index_select(0, accepted_idx),
        new_ssm.index_select(0, accepted_idx),
    )
    rejected_unchanged = _stats(
        initial.index_select(0, rejected_idx),
        new_ssm.index_select(0, rejected_idx),
    )
    rejected_old_changed = _stats(
        initial.index_select(0, rejected_idx),
        old_ssm.index_select(0, rejected_idx),
    )
    passed = (
        accepted_stats["torch_equal"]
        and accepted_stats["max_abs"] == 0.0
        and rejected_unchanged["torch_equal"]
        and rejected_old_changed["nonzero"] > 0
        and total_new_rows < total_old_rows
    )
    return {
        "schema": "fr13.accept_only_publish_gate.v1",
        "passed": bool(passed),
        "batch": batch,
        "tree_n": tree_n,
        "old_durable_rows": total_old_rows,
        "new_durable_rows": total_new_rows,
        "row_reduction_factor": float(total_old_rows / max(total_new_rows, 1)),
        "accepted_rows_vs_old_publish": accepted_stats,
        "rejected_rows_vs_initial_after_new_publish": rejected_unchanged,
        "rejected_rows_vs_initial_after_old_publish": rejected_old_changed,
        "per_batch": per_batch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

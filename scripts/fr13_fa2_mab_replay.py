#!/usr/bin/env python3
"""FR13 forked-FA2 M-invariance A/B (M=9 vs M=5 spine-slice) replay verifier.

DECISIVE controlled test for the cat9 22-flip SPINE_PERTURBATION carrier,
re-using the banked ``FR13_TREE_ATTN_OP_CAPTURE`` per-layer artifacts (each one
records ``query`` [M=9], the block-gathered ``dense_key``/``dense_value``, the
M x M ``tree_attn_bias``, and the served kernel ``output`` for one
full_attention layer at the deep-accept carrier event).

It re-calls OUR forked-FA2 op (the tree-bias varlen forward,
``flash_attn_varlen_func(..., tree_bias=...)``) TWICE on the SAME captured K/V:

  (a) M=9 -- the full tree: all 9 query rows + the 9x9 ancestry bias.
  (b) M=5 -- the SPINE-SLICE: only the spine query rows [0,1,2,4,6] + the 5x5
            spine-ancestry sub-bias; the spine-suffix KV rows [0,1,2,4,6].

The spine rows attend ONLY to spine ancestors (strict mask), so the M=5 slice's
spine bias == the M=9 spine bias for those keys; the ONLY thing that changes
between (a) and (b) is M (the live query-row occupancy / kBlockM MMA fragment
tile). It reports the deep-spine row (cat9 flat row 6 = node5) attn_out RAW
max_abs (NOT atol) between (a) and (b), per full_attention layer.

VERDICT:
  RAW != 0 at the full-attn layers => the forked-FA2 IS M-dependent (the
    query-tile fragment realization) => the query-pad fix (FR13_FA2_QPAD) is
    REAL and will drop 22 -> ~native floor at FULL ACCEPT.
  RAW == 0 (bit-exact)          => the fork is M-invariant => NOT the carrier
    => the 22 is genuinely DIFFUSE (per-layer GDN ~1-ULP accumulation) =>
    reshape (cat8/cat7) is the lossless route.

This is the IN-CONTAINER (GPU) counterpart of the in-process A/B hook
(FR13_FA2_MAB in scripts/fr10_phase4_patch_vllm_tree_gdn.py); both use the SAME
captured K/V from ONE cat9 boot's carrier event (NO second served stream => no
stream decoherence) and the SAME forked op + contiguous-KV re-call (so they are
unaffected by paged-KV block_table / varlen cu_seqlens).  The replay verifier
needs only the banked capture artifacts and a GPU with the forked FA2 binary;
it does NOT boot the server.

Run inside the vLLM container (the forked ``flash_attn_varlen_func`` with the
``tree_bias`` param lives there; the host venv is CPU-only and has no vllm).

Reward-hacks BANNED: this re-calls OUR kernel as-is; no copy / dense rerun /
splice / oracle / routing through native.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


# cat9 caterpillar: parent[i], spine chain [0,1,2,4,6], deep spine row = node5.
TREE_PARENT = (-1, 0, 1, 1, 2, 2, 4, 4, 6, 6)
SPINE_ROWS = (0, 1, 2, 4, 6)
DEEP_ROW = 6


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _layer_idx(name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.self_attn", str(name or ""))
    return int(match.group(1)) if match else None


def _recall_fork(
    flash_attn_varlen_func,
    q_all: torch.Tensor,
    k_all: torch.Tensor,
    v_all: torch.Tensor,
    context_len: int,
    q_rows: list[int],
    suffix_rows: list[int],
    bias_sub: torch.Tensor,
    scale: float,
    softcap: float,
    device: torch.device,
) -> torch.Tensor:
    """One forked-FA2 call on contiguous (non-paged) KV. Returns fp32 [m,H,D]."""
    m = len(q_rows)
    q = q_all.index_select(0, torch.tensor(q_rows, dtype=torch.long)).to(device).contiguous()
    kv_idx = list(range(context_len)) + [context_len + r for r in suffix_rows]
    kv_sel = torch.tensor(kv_idx, dtype=torch.long)
    k = k_all.index_select(0, kv_sel).to(device).contiguous()
    v = v_all.index_select(0, kv_sel).to(device).contiguous()
    cu_q = torch.tensor([0, m], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, k.shape[0]], dtype=torch.int32, device=device)
    tb = bias_sub.to(device).contiguous()
    out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=m,
        max_seqlen_k=int(k.shape[0]),
        softmax_scale=scale,
        causal=True,
        window_size=[-1, -1],
        softcap=softcap,
        fa_version=2,
        tree_bias=tb,
    )
    torch.cuda.synchronize()
    return out.detach().to(torch.float32).cpu()


def ab_one_capture(
    flash_attn_varlen_func,
    cap: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    query = cap["query"].to(torch.float32)  # [M, H, D]
    m_full = int(query.shape[0])
    bias = cap.get("tree_attn_bias")
    if bias is None:
        raise ValueError("capture has no tree_attn_bias (not a tree-bias decode)")
    bias = bias.to(torch.float32)
    if bias.shape[0] < m_full or bias.shape[1] < m_full:
        raise ValueError(
            f"bias {tuple(bias.shape)} smaller than M={m_full}"
        )
    seq_lens = cap["seq_lens"].reshape(-1)
    seq_len = int(seq_lens[0].item())
    context_len = seq_len - m_full
    if context_len < 0:
        raise ValueError(f"context_len {context_len} < 0 (seq_len={seq_len}, M={m_full})")

    spine_rows = [r for r in SPINE_ROWS if r < m_full]
    if DEEP_ROW not in spine_rows or len(spine_rows) < 2:
        raise ValueError(
            f"deep row {DEEP_ROW} / spine {list(SPINE_ROWS)} not present at M={m_full}"
        )

    scale = float(cap["scale"])
    softcap = float(cap.get("logits_soft_cap", 0.0))
    q_all = query.to(torch.bfloat16)
    k_all = cap["dense_key"][0].to(torch.bfloat16)
    v_all = cap["dense_value"][0].to(torch.bfloat16)
    bias_full = bias[:m_full, :m_full].contiguous()

    full_rows = list(range(m_full))
    out_m9 = _recall_fork(
        flash_attn_varlen_func, q_all, k_all, v_all, context_len,
        full_rows, full_rows, bias_full, scale, softcap, device,
    )
    spine_idx = torch.tensor(spine_rows, dtype=torch.long)
    bias_spine = bias_full.index_select(0, spine_idx).index_select(1, spine_idx).contiguous()
    out_m5 = _recall_fork(
        flash_attn_varlen_func, q_all, k_all, v_all, context_len,
        spine_rows, spine_rows, bias_spine, scale, softcap, device,
    )

    m5_deep = spine_rows.index(DEEP_ROW)
    row_m9 = out_m9[DEEP_ROW].reshape(-1)
    row_m5 = out_m5[m5_deep].reshape(-1)
    raw_max_abs = float((row_m9 - row_m5).abs().max().item())
    raw_mean_abs = float((row_m9 - row_m5).abs().mean().item())

    served = cap["output"].to(torch.float32)
    recall_vs_served = float(
        (row_m9 - served[DEEP_ROW].reshape(-1)).abs().max().item()
    )

    by_depth = []
    for depth, sr in enumerate(spine_rows):
        a = out_m9[sr].reshape(-1)
        b = out_m5[depth].reshape(-1)
        by_depth.append(
            {
                "depth": int(depth),
                "m9_row": int(sr),
                "m5_row": int(depth),
                "max_abs": float((a - b).abs().max().item()),
                "mean_abs": float((a - b).abs().mean().item()),
            }
        )

    return {
        "layer_name": cap.get("layer_name"),
        "layer": _layer_idx(cap.get("layer_name")),
        "m_full": m_full,
        "context_len": context_len,
        "seq_len": seq_len,
        "spine_rows": list(spine_rows),
        "deep_row": DEEP_ROW,
        "deep_spine_raw_max_abs": raw_max_abs,
        "deep_spine_raw_mean_abs": raw_mean_abs,
        "recall_m9_vs_served_deep_max_abs": recall_vs_served,
        "by_spine_depth": by_depth,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        nargs="+",
        required=True,
        type=Path,
        help="banked FR13_TREE_ATTN_OP_CAPTURE artifacts (one per full_attn layer)",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="RAW max_abs above this counts the layer as M-DEPENDENT (default 0.0)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA not available: run inside the vLLM container (the forked "
            "flash_attn_varlen_func with tree_bias lives there)."
        )
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    device = torch.device("cuda")
    rows: list[dict[str, Any]] = []
    for path in args.capture:
        cap = _load(path)
        try:
            res = ab_one_capture(flash_attn_varlen_func, cap, device)
            res["capture_path"] = str(path)
            rows.append(res)
        except Exception as exc:  # noqa: BLE001 - report and continue
            rows.append({"capture_path": str(path), "error": str(exc)})

    ok_rows = [r for r in rows if "error" not in r]
    ok_rows.sort(key=lambda r: (r.get("layer") is None, r.get("layer", 0)))
    max_over_layers = max(
        (r["deep_spine_raw_max_abs"] for r in ok_rows), default=0.0
    )
    m_dependent_layers = [
        r["layer"]
        for r in ok_rows
        if r["deep_spine_raw_max_abs"] > args.threshold
    ]
    verdict = (
        "M_DEPENDENT (forked-FA2 query-tile is the carrier; FR13_FA2_QPAD is REAL)"
        if max_over_layers > args.threshold
        else "M_INVARIANT (forked-FA2 NOT the carrier; 22 is DIFFUSE -> reshape route)"
    )

    out = {
        "schema": "fr13.fa2_mab_replay.v1",
        "spine_rows": list(SPINE_ROWS),
        "deep_row": DEEP_ROW,
        "threshold": args.threshold,
        "deep_spine_raw_max_abs_over_layers": max_over_layers,
        "m_dependent_layers": m_dependent_layers,
        "verdict": verdict,
        "layers": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "verdict": verdict,
                "deep_spine_raw_max_abs_over_layers": max_over_layers,
                "m_dependent_layers": m_dependent_layers,
                "per_layer": [
                    {
                        "layer": r.get("layer"),
                        "deep_spine_raw_max_abs": r.get("deep_spine_raw_max_abs"),
                        "recall_m9_vs_served": r.get(
                            "recall_m9_vs_served_deep_max_abs"
                        ),
                        **({"error": r["error"]} if "error" in r else {}),
                    }
                    for r in rows
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

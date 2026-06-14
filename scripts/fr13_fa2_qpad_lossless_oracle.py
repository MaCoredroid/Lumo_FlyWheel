#!/usr/bin/env python3
"""FR13_FA2_QPAD lossless-by-construction CPU oracle.

Proves that the query-pad transform (``_fr13_fa2_qpad_prepare`` in the patched
``vllm/vllm_flash_attn/flash_attn_interface.py``) leaves the REAL query-row
outputs of the forked-FA2 tree-bias decode BIT-IDENTICAL to the unpadded call,
under a kernel-faithful attention model.  This is the design proof behind the
FR13_FA2_QPAD M-invariance fix (FR13_FA2_MDEPENDENT_BIND): the deep-spine row
was M_DEPENDENT because its kBlockM tile occupancy / Is_even_MN predication /
tree_bias lane offsets varied with the live tree size; padding the query AND
the suffix-key extent to a fixed N_PAD_Q removes that variance while the
padded rows/keys are -inf-masked from every real row.

Kernel-faithful model (why CPU torch is representative HERE):
  * Score S[i,j] = q_i . k_j is computed PER (i,j) over the fixed head dim, so
    it does NOT depend on the total key count (a CUTLASS MMA computes each tile
    element independently).  We therefore compute scores with a per-column dot
    product, NOT a width-dependent batched GEMM (a plain ``q @ k.T`` introduces
    a ~6e-8 width artifact that the kernel does not have).
  * Softmax treats -inf scores as an EXACT 0 contribution and accumulates in a
    fixed column order; a padded column adds exactly 0.0 (IEEE-754 x+0.0==x),
    so it is a true no-op.

This oracle does NOT replace the live GPU gates (GATE-1 = the FR13_FA2_MAB A/B
re-run with QPAD; GATE-2 = the e2e per-token argmax flip count).  It only
proves the value-preservation of the padding construction.  The decisive
verify-vs-DECODE question (does padding to N_PAD_Q match the M=1 decode tile
geometry) is settled by GATE-2 on a live boot, per FR13_FA2_MDEPENDENT_BIND.

Run on CPU (host venv is fine; no CUDA / no forked kernel needed)::

    python3 scripts/fr13_fa2_qpad_lossless_oracle.py \
        --interface /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TREE_PARENT = (-1, 0, 1, 1, 2, 2, 4, 4, 6, 6)
SPINE_ROWS = (0, 1, 2, 4, 6)


def _load_qpad_prepare(interface_path: Path):
    """Exec the injected QPAD helpers out of the patched interface file.

    The full module imports the CUDA extensions (unavailable on CPU), so we
    exec only the helper-function span verbatim.
    """
    src = interface_path.read_text()
    marker = "def _fr13_fa2_qpad_should_apply"
    end_marker = "def flash_attn_varlen_func("
    if marker not in src:
        raise SystemExit(
            f"{interface_path} is not QPAD-patched (run fr13_patch_fa2_tree_bias.py)"
        )
    ns: dict[str, object] = {"torch": torch, "os": os}
    chunk = src[src.index(marker) : src.index(end_marker)]
    exec(compile(chunk, str(interface_path), "exec"), ns)  # noqa: S102
    return ns["_fr13_fa2_qpad_prepare"]


def _ancestry_bias(m: int) -> torch.Tensor:
    bias = torch.full((m, m), float("-inf"))
    for r in range(m):
        bias[r, r] = 0.0
        bias[r, 0] = 0.0
        a = r
        while TREE_PARENT[a] != -1:
            a = TREE_PARENT[a]
            bias[r, a] = 0.0
    return bias


def _dense_fork(q, k, v, cu_q, cu_k, seqused_k, tree_bias, scale, odtype):
    """Kernel-faithful dense forked-FA2 oracle (per-column dot + -inf-exact softmax)."""
    n_seq = cu_q.numel() - 1
    _, h_n, d = q.shape
    out = torch.zeros((q.shape[0], h_n, d), dtype=odtype)
    for i in range(n_seq):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        sq = qe - qs
        if cu_k is not None:
            ks, ke = int(cu_k[i]), int(cu_k[i + 1])
            kk, vv = k[ks:ke].float(), v[ks:ke].float()
            sk = ke - ks
        else:
            sk = int(seqused_k[i])
            kk, vv = k[i, :sk].float(), v[i, :sk].float()
        qq = q[qs:qe].float()
        ctx = sk - sq
        for r in range(sq):
            for h in range(h_n):
                sc = torch.stack(
                    [(qq[r, h] * kk[c, h]).sum() for c in range(sk)]
                ) * scale
                col = torch.arange(sk)
                sc = sc.masked_fill(col >= r + 1 + ctx, float("-inf"))
                if tree_bias is not None and r < tree_bias.shape[0]:
                    bc = min(sk - ctx, tree_bias.shape[1])
                    if bc > 0:
                        br = tree_bias[r, :bc]
                        seg = sc[ctx : ctx + bc]
                        seg = torch.where(
                            torch.isneginf(br),
                            torch.full_like(seg, float("-inf")),
                            seg + br,
                        )
                        sc = sc.clone()
                        sc[ctx : ctx + bc] = seg
                fin = torch.isfinite(sc)
                mx = sc[fin].max()
                acc = torch.zeros(d)
                ll = torch.tensor(0.0)
                for c in range(sk):
                    if not bool(fin[c]):
                        ll = ll + 0.0  # IEEE-exact no-op
                        acc = acc + 0.0
                        continue
                    e = torch.exp(sc[c] - mx)
                    ll = ll + e
                    acc = acc + e * vv[c, h, :]
                out[qs + r, h] = (acc / ll).to(odtype)
    return out


def _check(qpad_prepare, m, ctxs, n_pad_q, odtype, paged):
    h_n, d, scale = 2, 4, 0.5
    bias = _ancestry_bias(m)
    qs, ks, vs, cu_q, cu_k = [], [], [], [0], [0]
    for c in ctxs:
        sk = c + m
        qs.append(torch.randn(m, h_n, d, dtype=torch.bfloat16))
        ks.append(torch.randn(sk, h_n, d, dtype=torch.bfloat16))
        vs.append(torch.randn(sk, h_n, d, dtype=torch.bfloat16))
        cu_q.append(cu_q[-1] + m)
        cu_k.append(cu_k[-1] + sk)
    q = torch.cat(qs)
    k = torch.cat(ks)
    v = torch.cat(vs)
    cu_q = torch.tensor(cu_q, dtype=torch.int32)
    cu_k = torch.tensor(cu_k, dtype=torch.int32)
    max_q, max_k = m, max(c + m for c in ctxs)
    ref = _dense_fork(q, k, v, cu_q, cu_k, None, bias, scale, odtype)

    os.environ["FR13_FA2_QPAD"] = "1"
    if not paged:
        pk = qpad_prepare(
            q, k, v, None, cu_q, cu_k, None, max_q, max_k, bias, None, n_pad_q
        )
        got = _dense_fork(
            pk["q"], pk["k"], pk["v"], pk["cu_seqlens_q"],
            pk["cu_seqlens_k"], pk["seqused_k"], pk["tree_bias"], scale, odtype,
        )
    else:
        # per-seq KV blocks with GARBAGE in the padded slots [sk:sk+pad]; the
        # paged QPAD inflates seqused_k so the kernel reads them, but they are
        # -inf-masked from every real row.
        pad = n_pad_q - m
        n_seq = len(ctxs)
        blk_len = max_k + pad
        k_blk = torch.randn(n_seq, blk_len, h_n, d, dtype=torch.bfloat16)
        v_blk = torch.randn(n_seq, blk_len, h_n, d, dtype=torch.bfloat16)
        seqused = torch.zeros(n_seq, dtype=torch.int32)
        for i, c in enumerate(ctxs):
            sk = c + m
            ks0 = int(cu_k[i])
            k_blk[i, :sk] = k[ks0 : ks0 + sk]
            v_blk[i, :sk] = v[ks0 : ks0 + sk]
            seqused[i] = sk
        pk = qpad_prepare(
            q, k_blk, v_blk, None, cu_q, None, seqused, max_q, max_k, bias, True, n_pad_q
        )
        got = _dense_fork(
            pk["q"], pk["k"], pk["v"], pk["cu_seqlens_q"], None,
            pk["seqused_k"], pk["tree_bias"], scale, odtype,
        )
    real = pk["unpad"](got).reshape(len(ctxs) * m, h_n, d)
    diff = (real - ref).abs().max().item()
    tag = "paged" if paged else "contig"
    print(
        f"[{tag:6s}] M={m:2d} N_PAD={n_pad_q} B={len(ctxs)} "
        f"odtype={str(odtype).split('.')[-1]:8s} real-row max_abs = {diff:.3e}"
    )
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interface",
        type=Path,
        default=Path(
            "/usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/flash_attn_interface.py"
        ),
        help="path to the QPAD-patched flash_attn_interface.py",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.interface.exists():
        raise SystemExit(f"interface file not found: {args.interface}")
    torch.manual_seed(args.seed)
    qpad_prepare = _load_qpad_prepare(args.interface)

    cases = [
        # (m, ctxs, n_pad_q, odtype, paged)
        (9, [23], 64, torch.bfloat16, False),
        (9, [23], 64, torch.float32, False),
        (5, [23], 64, torch.bfloat16, False),  # spine slice
        (9, [17, 40, 5], 64, torch.bfloat16, False),  # multi-request
        (9, [17, 40, 5], 64, torch.float32, False),
        (9, [23], 64, torch.bfloat16, True),  # paged, garbage padded KV
        (9, [17, 40, 5], 64, torch.float32, True),  # paged multi-request
        (10, [31], 64, torch.bfloat16, False),  # cat9 full M=10
    ]
    worst = 0.0
    for m, ctxs, n_pad_q, odtype, paged in cases:
        worst = max(worst, _check(qpad_prepare, m, ctxs, n_pad_q, odtype, paged))
    print(f"\nworst real-row max_abs over all cases = {worst:.3e}")
    if worst != 0.0:
        raise SystemExit(
            f"FR13_FA2_QPAD NOT lossless-by-construction: worst={worst}"
        )
    print(
        "PASS: FR13_FA2_QPAD real-row outputs are BIT-IDENTICAL to the unpadded "
        "forked-FA2 call (padded query rows + padded/garbage keys are inert)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

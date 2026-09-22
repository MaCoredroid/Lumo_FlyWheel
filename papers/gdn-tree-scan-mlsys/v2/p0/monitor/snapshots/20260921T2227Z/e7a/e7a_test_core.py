#!/usr/bin/env python3
"""Narrow CPU regression tests for review-02 findings (pure torch; no GPU, no model).
Run: python3 e7a_test_core.py [--json-out path]. Exit code 0 iff all checks pass."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e7a_core as C  # noqa: E402


def check(results: dict, name: str, cond: bool, detail=None) -> None:
    results[name] = {"pass": bool(cond), "detail": detail}
    print(("PASS " if cond else "FAIL ") + name + ("" if detail is None else f"  {detail}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    R: dict = {}

    # ---- finding 4: signed ULP mapping (fp32 and bf16)
    f = torch.tensor
    one = f([1.0]); nxt = torch.nextafter(one, f([2.0]))
    check(R, "ulp32_adjacent_positive", int(C.ulp_distance_fp32(nxt, one)) == 1, int(C.ulp_distance_fp32(nxt, one)))
    m1 = f([-1.0]); mnxt = torch.nextafter(m1, f([-2.0]))
    check(R, "ulp32_adjacent_negative", int(C.ulp_distance_fp32(mnxt, m1)) == 1, int(C.ulp_distance_fp32(mnxt, m1)))
    tiny = torch.nextafter(f([0.0]), f([1.0])); mtiny = torch.nextafter(f([0.0]), f([-1.0]))
    check(R, "ulp32_cross_zero_min_subnormals", int(C.ulp_distance_fp32(tiny, mtiny)) == 2, int(C.ulp_distance_fp32(tiny, mtiny)))
    check(R, "ulp32_neg_min_subnormal_to_zero", int(C.ulp_distance_fp32(mtiny, f([0.0]))) == 1, int(C.ulp_distance_fp32(mtiny, f([0.0]))))
    check(R, "ulp32_signed_zero_equal", int(C.ulp_distance_fp32(f([-0.0]), f([0.0]))) == 0, int(C.ulp_distance_fp32(f([-0.0]), f([0.0]))))
    xs = torch.tensor([-3.0, -1.5, -1.0, -1e-30, -0.0, 0.0, 1e-30, 1.0, 1.5, 3.0], dtype=torch.float32)
    ords = C._fp32_ordinal(xs)
    check(R, "ulp32_ordinal_monotone", bool((ords[1:] >= ords[:-1]).all()), ords.tolist())
    b1 = torch.tensor([1.0], dtype=torch.bfloat16); b1n = torch.tensor([1.0 + 2 ** -7], dtype=torch.bfloat16)
    check(R, "ulp16_adjacent_positive", int(C.ulp_distance_bf16(b1n, b1)) == 1, int(C.ulp_distance_bf16(b1n, b1)))
    check(R, "ulp16_signed_zero_equal", int(C.ulp_distance_bf16(torch.tensor([-0.0], dtype=torch.bfloat16), torch.tensor([0.0], dtype=torch.bfloat16))) == 0)
    bm = torch.tensor([-1.0], dtype=torch.bfloat16); bmn = torch.tensor([-(1.0 + 2 ** -7)], dtype=torch.bfloat16)
    check(R, "ulp16_adjacent_negative", int(C.ulp_distance_bf16(bmn, bm)) == 1, int(C.ulp_distance_bf16(bmn, bm)))
    xs16 = torch.tensor([-3.0, -1.0, -0.0, 0.0, 1.0, 3.0], dtype=torch.bfloat16)
    o16 = C._bf16_ordinal(xs16)
    check(R, "ulp16_ordinal_monotone", bool((o16[1:] >= o16[:-1]).all()), o16.tolist())
    # non-finite handling in compare(): counted and excluded
    c = C.compare(torch.tensor([1.0, float("inf"), 2.0]), torch.tensor([1.0, 1.0, float("nan")]), ulp="fp32")
    check(R, "compare_counts_and_excludes_nonfinite", c["nonfinite_candidate"] == 1 and c["nonfinite_reference"] == 1 and math.isfinite(c["max_abs"]), c)

    # ---- finding 3: off-path exponent overflow must not produce NaN (reviewer's two-node case)
    for dtype in (torch.float32, torch.float64):
        cum = torch.tensor([[-100.0], [-200.0]], dtype=dtype)  # (N=2, VH=1): cum_0=-100, cum_1=-200
        g = torch.tensor([[-100.0], [-100.0]], dtype=dtype)
        vis = C.visible_mask([-1, 0])
        d = C.decay_ratios(cum, g, [-1, 0], vis, "expdiff")
        ok = bool(torch.isfinite(d).all()) and d[0, 0, 1].item() == 0.0 and d[0, 0, 0].item() == 1.0 and d[0, 1, 1].item() == 1.0
        check(R, f"decay_expdiff_offpath_finite_zero_{str(dtype)[6:]}", ok, d.tolist())
        d_ratio = C.decay_ratios(cum, g, [-1, 0], vis, "ratio")
        check(R, f"decay_ratio_offpath_zero_{str(dtype)[6:]}", d_ratio[0, 0, 1].item() == 0.0, d_ratio.tolist())
    # full system build on a synthetic tiny-gate depth-11 chain must be finite in fp32 (expdiff) — labeled synthetic
    pay = C.synth_payload(C.chain_parents(11), seed=7, regime="tiny-gates")
    ops32 = C.lift(pay, torch.float32)
    sys32 = C.build_system(ops32)
    gs = C.gate_stats(ops32, sys32)
    U32 = C.solve_forward_substitution(sys32, pay.parents)
    S32 = C.reconstruct_all_states(ops32, sys32, U32)
    check(R, "tiny_gates_chain11_fp32_expdiff_all_finite", bool(torch.isfinite(sys32.G).all() and torch.isfinite(sys32.R).all() and torch.isfinite(U32).all() and torch.isfinite(S32).all()),
          {"P_min": gs["P_min"], "cum_g_min": gs["cum_g_min"], "fp32_underflow_risk": gs["fp32_underflow_risk"]})
    sys_ratio = C.build_system(ops32, ratio_mode="ratio")
    check(R, "tiny_gates_chain11_fp32_literal_ratio_flagged_nonfinite_or_finite(diagnostic)", True,
          {"nonfinite_decay_entries": int((~torch.isfinite(sys_ratio.decay)).sum().item()), "P_min": gs["P_min"]})

    # ---- finding 1: compare() must refuse mismatched shapes (padded-vs-actual axis bugs surface loudly)
    try:
        C.compare(torch.zeros(48, 16, 128), torch.zeros(10, 48, 128))
        check(R, "compare_shape_mismatch_raises", False)
    except ValueError as e:
        check(R, "compare_shape_mismatch_raises", True, str(e)[:80])

    # ---- algebra identities on a synthetic depth-11 chain and a binary depth-3 tree (fp64) — sanity after edits
    for label, parents in (("chain11", C.chain_parents(11)), ("binary3", C.binary_parents(3)), ("caterpillar10", list(C.CATERPILLAR_10))):
        p = C.synth_payload(parents, seed=11, regime="historical-like")
        ops = C.lift(p, torch.float64)
        orc = C.sequential_tree(ops, checkpoint_parent=False)
        sysd = C.build_system(ops)
        UB = C.solve_forward_substitution(sysd, parents)
        UC = C.solve_neumann(sysd, parents)
        eB = C.compare(UB.permute(1, 0, 2), orc.u)["max_abs"]
        eC = C.compare(UC.permute(1, 0, 2), orc.u)["max_abs"]
        eS = C.compare(C.reconstruct_all_states(ops, sysd, UB), orc.state)["max_abs"]
        worst_commit = max(C.compare(C.commit_compact(ops, sysd, UC, path), orc.state[path[-1]])["max_abs"] for path in C.all_accept_paths(parents))
        # sibling reorder invariance (relabeling only)
        newp, perm = C.sibling_reorder_permutation(parents, "reverse")
        pp = p.permuted(perm, newp)
        orc_p = C.sequential_tree(C.lift(pp, torch.float64), checkpoint_parent=False)
        inv = torch.empty(len(perm), dtype=torch.long); inv[torch.tensor(perm)] = torch.arange(len(perm))
        eP = C.compare(orc_p.state.index_select(0, inv), orc.state)["max_abs"]
        check(R, f"fp64_identities_{label}", max(eB, eC, eS, worst_commit) < 1e-12 and eP == 0.0,
              {"B_U": eB, "C_U": eC, "B_state": eS, "C_commit_worst": worst_commit, "sibling_reverse_relabel": eP})

    ok = all(v["pass"] for v in R.values())
    print("ALL PASS" if ok else "SOME FAILED")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps({"all_pass": ok, "torch": torch.__version__, "checks": C.to_jsonable(R)}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

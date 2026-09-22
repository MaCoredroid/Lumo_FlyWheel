#!/usr/bin/env python3
"""E7a core self-test (CPU, pure torch): algebraic identities A == B == C == oracle at fp64,
and fp32 torch-mirror errors, on one payload. Not a GPU result."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e7a_core as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", default="/home/mark/shared/lumoFlyWheel/output/fr13_replay_gpu_gates/boot1a_eager_capture_logs/tree_gdn_capture_payload.pt")
    ap.add_argument("--synthetic", default=None, help="chain:<depth> | binary:<depth> | caterpillar")
    ap.add_argument("--regime", default="historical-like")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.time()
    if args.synthetic:
        kind, _, arg = args.synthetic.partition(":")
        parents = {"chain": lambda: C.chain_parents(int(arg)), "binary": lambda: C.binary_parents(int(arg)),
                   "caterpillar": lambda: list(C.CATERPILLAR_10)}[kind]()
        pay = C.synth_payload(parents, args.seed, regime=args.regime)
    else:
        pay = C.load_payload(args.payload)
    print(f"payload: {pay.label} synthetic={pay.synthetic} n={pay.n} parents={pay.parents} depth={C.max_strict_ancestors(pay.parents)}")
    res = {"payload": C.to_jsonable(pay.provenance), "parents": pay.parents}

    # ---- fp64 oracle (A, per-node replay from h0 = the serial recurrence on identical rounded operands)
    ops64 = C.lift(pay, torch.float64)
    orc = C.sequential_tree(ops64, checkpoint_parent=False)
    orc_ckpt = C.sequential_tree(ops64, checkpoint_parent=True)
    sys64 = C.build_system(ops64)
    res["gates"] = C.gate_stats(ops64, sys64)
    print("gates:", json.dumps(res["gates"]))
    UB64 = C.solve_forward_substitution(sys64, pay.parents)
    UC64 = C.solve_neumann(sys64, pay.parents)
    res["fp64_identities"] = {
        "A_ckpt_vs_A_replay_state": C.compare(orc_ckpt.state, orc.state)["max_abs"],
        "B_U_vs_oracle_u": C.compare(UB64.permute(1, 0, 2), orc.u)["max_abs"],
        "C_U_vs_oracle_u": C.compare(UC64.permute(1, 0, 2), orc.u)["max_abs"],
        "B_out_vs_oracle": C.compare(C.outputs_from_factors(ops64, sys64, UB64), orc.out)["max_abs"],
        "C_out_vs_oracle": C.compare(C.outputs_from_factors(ops64, sys64, UC64), orc.out)["max_abs"],
        "B_state_vs_oracle": C.compare(C.reconstruct_all_states(ops64, sys64, UB64), orc.state)["max_abs"],
        "C_state_vs_oracle": C.compare(C.reconstruct_all_states(ops64, sys64, UC64), orc.state)["max_abs"],
        "oracle_out_max_abs": orc.out.abs().max().item(), "oracle_state_max_abs": orc.state.abs().max().item(),
    }
    for path in C.all_accept_paths(pay.parents):
        a = path[-1]
        cb = C.commit_compact(ops64, sys64, UB64, path)
        cc = C.commit_compact(ops64, sys64, UC64, path)
        ra = C.replay_commit(ops64, path)
        res["fp64_identities"][f"commit_path_to_{a}"] = {
            "B_compact_vs_oracle_state": C.compare(cb, orc.state[a])["max_abs"],
            "C_compact_vs_oracle_state": C.compare(cc, orc.state[a])["max_abs"],
            "A_replay_vs_oracle_state": C.compare(ra, orc.state[a])["max_abs"],
        }
    print("fp64 identities:", json.dumps({k: v for k, v in res["fp64_identities"].items() if not k.startswith("commit")}))
    worst_commit = max(max(d.values()) for k, d in res["fp64_identities"].items() if k.startswith("commit"))
    print("fp64 worst commit identity error over all accept paths:", worst_commit)

    # ---- fp32 torch mirrors vs fp64 oracle (mathematical/implementation error of the fp32 realization)
    ops32 = C.lift(pay, torch.float32)
    A32 = C.sequential_tree(ops32, checkpoint_parent=True)
    sys32 = C.build_system(ops32)
    UB32 = C.solve_forward_substitution(sys32, pay.parents)
    UC32 = C.solve_neumann(sys32, pay.parents)
    res["fp32_torch_mirror_vs_oracle"] = {
        "A_u": C.compare(A32.u, orc.u, ulp="fp32"),
        "B_U": C.compare(UB32.permute(1, 0, 2), orc.u, ulp="fp32"),
        "C_U": C.compare(UC32.permute(1, 0, 2), orc.u, ulp="fp32"),
        "A_out": C.compare(A32.out, orc.out, ulp="fp32"),
        "B_out": C.compare(C.outputs_from_factors(ops32, sys32, UB32), orc.out, ulp="fp32"),
        "C_out": C.compare(C.outputs_from_factors(ops32, sys32, UC32), orc.out, ulp="fp32"),
        "A_state": C.compare(A32.state, orc.state, ulp="fp32"),
        "B_state": C.compare(C.reconstruct_all_states(ops32, sys32, UB32), orc.state, ulp="fp32"),
        "C_state": C.compare(C.reconstruct_all_states(ops32, sys32, UC32), orc.state, ulp="fp32"),
    }
    # stage 2: identical reference factors (oracle u rounded to fp32) into each fp32 commit
    Uref32 = orc.u.float().permute(1, 0, 2).contiguous()
    st2 = {}
    for path in C.all_accept_paths(pay.parents):
        a = path[-1]
        st2[f"to_{a}"] = {
            "compact_with_oracle_factors": C.compare(C.commit_compact(ops32, sys32, Uref32, path), orc.state[a], ulp="fp32")["max_abs"],
            "A_replay_fp32": C.compare(C.replay_commit(ops32, path), orc.state[a], ulp="fp32")["max_abs"],
            "B_compact_own_factors": C.compare(C.commit_compact(ops32, sys32, UB32, path), orc.state[a], ulp="fp32")["max_abs"],
            "C_compact_own_factors": C.compare(C.commit_compact(ops32, sys32, UC32, path), orc.state[a], ulp="fp32")["max_abs"],
        }
    res["fp32_commit_stages"] = st2
    for k, v in res["fp32_torch_mirror_vs_oracle"].items():
        print(f"fp32 {k:8s} max_abs={v['max_abs']:.3e} rel={v['max_rel_to_refmax']:.3e} ulp32_max_sig={v['ulp32_max_sig']} frac>=1ulp_sig={v['ulp32_frac_ge1_sig']:.3f} exact={v['exact_frac_all']:.3f}")
    print("fp32 commit stages (max_abs vs oracle) per accepted leaf:")
    for k, v in st2.items():
        print(f"  {k:6s} " + " ".join(f"{kk}={vv:.2e}" for kk, vv in v.items()))

    # ---- served-kernel bytes from the historical capture (June 2026 kernel): context only
    if pay.serving_out is not None:
        res["historical_served_vs_oracle"] = {
            "served_out_bf16_vs_oracle": C.compare(pay.serving_out, orc.out, ulp="bf16"),
            # the production stateless route (FR13_REPLAY_ROUTE=1) materializes NO per-node tree state: fresh payloads
            # carry serving_state=None and these rows are reported as N/A (never as agreement).
            "served_state_fp32_vs_oracle": C.compare(pay.serving_state, orc.state, ulp="fp32") if pay.serving_state is not None else "N/A (stateless route: no served tree state)",
            "served_state_vs_A32_mirror": C.compare(pay.serving_state, A32.state, ulp="fp32") if pay.serving_state is not None else "N/A (stateless route: no served tree state)",
        }
        for k, v in res["historical_served_vs_oracle"].items():
            tag = str(pay.provenance.get("status", "")).split(" ")[0] or "payload"
    if isinstance(v, dict):
        print(f"{tag} {k}: max_abs={v['max_abs']:.3e} rel={v['max_rel_to_refmax']:.3e}", {kk: vv for kk, vv in v.items() if 'ulp' in kk and 'max' in kk})
    else:
        print(f"{tag} {k}: {v}")
    res["elapsed_s"] = time.time() - t0
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(C.to_jsonable(res), indent=1))
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

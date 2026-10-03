#!/usr/bin/env python3
"""q1v3 reducer (CPU only): authenticate raw records, compute distances, apply the frozen PROTOCOL.md rules.

  reduce.py --run RUN_ROOT --out RUN_ROOT/VERDICT.json

All thresholds below are frozen by PROTOCOL.md section 5 (do not edit after a run has started).
Distances are always taken against native arm A (the arm whose cold prefill defined the common O0).
"""
from __future__ import annotations

import argparse, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1v3_common as Q  # noqa: E402

# ---------------------------------------------------------------- frozen constants (PROTOCOL.md section 5)
M = 4.0                                   # tolerance = max(floor, M x calibrated native envelope)
FLOORS = {"next_kl": 0.02, "gdn_rel": 0.02, "conv_rel": 0.02, "kv_new_rel": 0.02, "kv_hist_rel": 0.02}
MARGIN_FLOOR = 0.25                       # greedy tie band floor (2 bf16 ULPs at |logit| in [16, 32))
MARGIN_MULT = 2.0                         # band = max(floor, 2 x envelope of next-token max-abs logit difference)
MAX_AMBIGUOUS_FRACTION = 0.10             # natives disagreeing on greedy outside the band
REQUIRED_ARMS = ("A", "B", "V", "CAND")
OPTIONAL_ARMS = ("P",)
NATIVE_PAIRS = ("B", "V", "P")
REQUIRED_NCS = ("NC_CONV", "NC_GDN", "NC_KV", "NC_STALE")
NC_SURFACE = {"conv": "conv_rel", "gdn": "gdn_rel", "kv": "kv_new_rel", "logits": "next_kl"}
SURFACES = tuple(FLOORS)


# ---------------------------------------------------------------- record access
class Obs:
    def __init__(self, arm_dir, doc):
        self.dir, self.doc, self._cache = arm_dir, doc, {}

    def _t(self, desc):
        k = desc["file"]
        if k not in self._cache:
            self._cache[k] = Q.load_tensor(self.dir, desc, verify=True)
        return self._cache[k]

    def logits(self):
        out = {}
        for r in self.doc["records"]["logits"]:
            t = self._t(r["obj"]).float()
            for row, i in enumerate(r["i"]):
                out[i] = t[row]
        return out

    def o1(self, k, part):
        d = self.doc["records"]["o1"][str(k)][part]
        return None if d is None else self._t(d)

    def drop(self):
        self._cache.clear()


def load_obs(run, arm, obs_id):
    p = os.path.join(run, arm, "cases", f"{obs_id}.json")
    if not os.path.exists(p):
        return None, "missing"
    doc = json.load(open(p))
    if Q.record_digest(doc) != doc.get("record_sha256"):
        return None, "record digest mismatch"
    return Obs(os.path.join(run, arm), doc), None


# ---------------------------------------------------------------- metrics
def layer_rel_max(x, ref):
    vals = [Q.rel_l2(x[l], ref[l]) for l in range(int(ref.shape[0]))]
    j = max(range(len(vals)), key=lambda l: vals[l])
    return vals[j], j


def cell(case, k, ref, x, lref, lx):
    cy = case["cycles"][k]
    s, b = case["cycle_start"][k], case["cycle_start"][k] + cy["L"] + 1
    out = {"k": k, "path": cy["path_name"], "L": cy["L"], "z_kind": cy["z_kind"]}
    a, c = lref[b], lx[b]
    out.update(next_kl=Q.kl_div(a, c), next_maxabs=Q.max_abs(c, a), greedy_ref=Q.greedy(a), greedy_x=Q.greedy(c), margin_ref=Q.top_margin(a),
               top5_overlap=len(set(Q.topk_ids(a, 5)) & set(Q.topk_ids(c, 5))))
    out["verify_kl_max"] = max(Q.kl_div(lref[i], lx[i]) for i in range(s, b))
    out["gdn_rel"], out["gdn_worst_layer"] = layer_rel_max(x.o1(k, "ssm"), ref.o1(k, "ssm"))
    cr, cx = ref.o1(k, "conv"), x.o1(k, "conv")
    out["conv_rel"], out["conv_worst_layer"] = layer_rel_max(cx, cr)
    out["conv_rel_reversed_taps"] = layer_rel_max(cx.flip(1), cr)[0]          # layout diagnostic only
    kr, kx = ref.o1(k, "kv"), x.o1(k, "kv")                                     # [16, 2, b, H, D], stream idx [0, b)
    out["kv_new_rel"] = max(Q.rel_l2(kx[l, p, s:b], kr[l, p, s:b]) for l in range(kr.shape[0]) for p in (0, 1))
    out["kv_hist_rel"] = max((Q.rel_l2(kx[l, p, :s], kr[l, p, :s]) for l in range(kr.shape[0]) for p in (0, 1)), default=0.0) if s else 0.0
    out["bitwise_equal"] = bool(out["next_maxabs"] == 0 and out["gdn_rel"] == 0 and out["conv_rel"] == 0 and out["kv_new_rel"] == 0 and out["kv_hist_rel"] == 0)
    return out


def compare(case, ref, x):
    lref, lx = ref.logits(), x.logits()
    cells = [cell(case, k, ref, x, lref, lx) for k in range(len(case["cycles"]))]
    v0 = case["cycles"][0]["L"] + 1                              # cycle-0 rows: depend only on the common O0, no commit
    verify0 = {"kl": max(Q.kl_div(lref[i], lx[i]) for i in range(v0)), "maxabs": max(Q.max_abs(lx[i], lref[i]) for i in range(v0))}
    return cells, verify0


# ---------------------------------------------------------------- structure checks
def structural(doc, case, kind, mutation=None):
    pr = []
    if doc["structural_mismatches"]:
        pr.append(f"{len(doc['structural_mismatches'])} consumed-token mismatches (first {doc['structural_mismatches'][0]})")
    if kind == "cand":
        for pub in doc["records"]["publication"]:
            j = pub["after_taw_call"]
            if 0 <= j < len(case["cycles"]):
                nodes = case["cycles"][j]["nodes"]
                if mutation and mutation["mutation"] == "NC_SIB" and j == mutation["cycle"]:
                    nodes = mutation["sibling_nodes"]
                L = len(nodes) - 1
                if pub["paths"] is None or pub["paths"][:L] != nodes[1:] or pub["len"] != L:
                    pr.append(f"published path after TAW {j}: {pub['paths'][:L] if pub['paths'] else None}/{pub['len']} != {nodes[1:]}/{L}")
    return pr


# ---------------------------------------------------------------- main reduction
def reduce(run, cases_doc, plan_fn):
    V = {"schema": "q1v3.verdict.v1", "run": run, "cases_sha256": Q.sha256_file(os.path.join(HERE, "cases.v1.json")), "reasons": []}
    case_by = {c["case_id"]: c for c in cases_doc["cases"]}
    prefixes = cases_doc["prefixes"]
    run_id = os.path.basename(os.path.abspath(run))
    arms_present = [a for a in REQUIRED_ARMS + OPTIONAL_ARMS if os.path.isdir(os.path.join(run, a, "cases"))]
    # ---- R1 integrity
    integ = {}
    for arm in REQUIRED_ARMS + OPTIONAL_ARMS:
        exp = plan_fn(cases_doc, arm, run_id)
        got = {"expected": len(exp), "valid": 0, "invalid": [], "missing": []}
        for o in exp:
            ob, err = load_obs(run, arm, o["obs_id"])
            if ob is None:
                got["missing"].append(f"{o['obs_id']}: {err}")
            elif not ob.doc["valid"]:
                got["invalid"].append({"obs_id": o["obs_id"], "problems": ob.doc["problems"][:5]})
            else:
                got["valid"] += 1
        integ[arm] = got
    V["integrity"] = integ
    r1 = all(integ[a]["valid"] == integ[a]["expected"] for a in REQUIRED_ARMS)
    if not r1:
        V["reasons"].append("R1: a required arm has missing/invalid observations")
    # ---- R2 prompt parity and R3 O0 import, natural-O0 diagnostics
    parity, imports, natural = {}, [], {}
    for arm in arms_present:
        for o in plan_fn(cases_doc, arm, run_id):
            ob, _ = load_obs(run, arm, o["obs_id"])
            if ob is None:
                continue
            d = ob.doc; pfx = case_by[d["case_id"]]["prefix"]
            parity.setdefault(pfx, set()).add((d["P"], d["prompt_sha256"]))
            imp = d["records"]["o0"].get("import") or {}
            imports.append(bool(imp.get("readback_equal")))
            if "natural" in d["records"]["o0"]:
                natural.setdefault(pfx, {})[arm] = d["records"]["o0"]["natural"]
    V["prompt_parity"] = {p: {"distinct": len(s), "P": sorted(x[0] for x in s), "reference_prompt_len": prefixes[p]["reference_prompt_len"],
                              "matches_codex_token_file": any(x[1] == prefixes[p]["reference_prompt_sha256"] for x in s)} for p, s in parity.items()}
    r2 = all(v["distinct"] == 1 for v in V["prompt_parity"].values())
    r3 = bool(imports) and all(imports)
    if not r2:
        V["reasons"].append("R2: rendered prompts differ across arms")
    if not r3:
        V["reasons"].append("R3: a common-O0 import readback differed")
    nat = {}
    for p, d in natural.items():
        a = d.get("A")
        nat[p] = {arm: (None if a is None else {k: d[arm][k] == a[k] for k in ("conv_sha256", "ssm_sha256", "kv_sha256")}) for arm in d if arm != "A"}
    V["natural_o0_equal_to_A"] = nat
    # ---- distances: natives vs A and candidate vs A
    pos_cases = [c for c in cases_doc["cases"]]
    rows = {arm: {} for arm in NATIVE_PAIRS + ("CAND",)}
    verify0 = {}
    for c in pos_cases:
        ref, err = load_obs(run, "A", f"A__{c['case_id']}__r0")
        if ref is None or not ref.doc["valid"]:
            continue
        for arm in NATIVE_PAIRS + ("CAND",):
            x, err = load_obs(run, arm, f"{arm}__{c['case_id']}__r0")
            if x is None or not x.doc["valid"]:
                continue
            cells, v0 = compare(c, ref, x)
            rows[arm][c["case_id"]] = cells
            verify0.setdefault(arm, {})[c["case_id"]] = v0
            x.drop()
        ref.drop()
    # ---- envelope from calibration prefixes, tolerances
    calib = {c["case_id"] for c in pos_cases if prefixes[c["prefix"]]["block"] == "calibration"}
    env = {s: 0.0 for s in SURFACES}; env_maxabs = 0.0; ab_exact = True
    for arm in NATIVE_PAIRS:
        for cid, cells in rows[arm].items():
            for ce in cells:
                if arm == "B" and not ce["bitwise_equal"]:
                    ab_exact = False
                if cid in calib:
                    for s in SURFACES:
                        env[s] = max(env[s], ce[s])
                    env_maxabs = max(env_maxabs, ce["next_maxabs"])
    tau_native = {s: max(FLOORS[s], M * env[s]) for s in SURFACES}
    # Candidate kernel noise on bitwise-identical state (cycle-0 verify rows, no commit involved), calibration prefixes only.
    dv0 = {k: max((v[k] for cid, v in verify0.get("CAND", {}).items() if cid in calib), default=0.0) for k in ("kl", "maxabs")}
    tau = dict(tau_native, next_kl=max(tau_native["next_kl"], M * dv0["kl"]))
    band = max(MARGIN_FLOOR, MARGIN_MULT * env_maxabs, MARGIN_MULT * dv0["maxabs"])
    V["reference"] = {"A_vs_B": "EXACT" if (ab_exact and rows["B"]) else "SPREAD", "envelope_calibration": env, "envelope_next_maxabs": env_maxabs,
                      "candidate_verify_only_calibration": dv0, "tolerance_native": tau_native, "tolerance": tau, "greedy_band": band,
                      "native_arms_present": [a for a in NATIVE_PAIRS if rows[a]]}
    spread = {}
    for arm in NATIVE_PAIRS:
        spread[arm] = {s: max((ce[s] for cells in rows[arm].values() for ce in cells), default=None) for s in SURFACES + ("next_maxabs",)}
        spread[arm]["bitwise_equal_cells"] = sum(ce["bitwise_equal"] for cells in rows[arm].values() for ce in cells)
        spread[arm]["cells"] = sum(len(cells) for cells in rows[arm].values())
    V["native_spread_vs_A"] = spread
    # ---- R4 held-out generalization of the native envelope
    viol = []
    for arm in NATIVE_PAIRS:
        for cid, cells in rows[arm].items():
            if cid in calib:
                continue
            for ce in cells:
                for s in SURFACES:
                    if ce[s] > tau_native[s]:
                        viol.append(f"{arm} {cid} k{ce['k']} {s}={ce[s]:.4g} > {tau_native[s]:.4g}")
    V["heldout_native_violations"] = viol
    r4 = not viol
    if not r4:
        V["reasons"].append(f"R4: {len(viol)} held-out native cells exceed the calibrated tolerance")
    # ---- R5 native categorical stability; candidate cells
    cand_fail, cand_cells, ambiguous, cand_struct, ref_struct = [], 0, 0, [], []
    for cid, cells in rows["CAND"].items():
        c = case_by[cid]
        for ce in cells:
            cand_cells += 1
            unstable = any(rows[a].get(cid) and rows[a][cid][ce["k"]]["greedy_x"] != ce["greedy_ref"] for a in NATIVE_PAIRS) and ce["margin_ref"] > band
            if unstable:
                ambiguous += 1
            fails = [f"{s}={ce[s]:.4g}>{tau[s]:.4g}" for s in SURFACES if ce[s] > tau[s]]
            if ce["greedy_x"] != ce["greedy_ref"] and ce["margin_ref"] > band and not unstable:
                fails.append(f"greedy {ce['greedy_x']}!={ce['greedy_ref']} (margin {ce['margin_ref']:.3g} > band {band:.3g})")
            if fails:
                cand_fail.append({"case": cid, "k": ce["k"], "path": ce["path"], "held_out": prefixes[c["prefix"]]["held_out"], "fails": fails})
        ob, _ = load_obs(run, "CAND", f"CAND__{cid}__r0")
        sp = structural(ob.doc, c, "cand") if ob else ["missing"]
        if sp:
            cand_struct.append({"case": cid, "problems": sp})
    for arm in ("A",) + NATIVE_PAIRS:
        for c in pos_cases:
            ob, _ = load_obs(run, arm, f"{arm}__{c['case_id']}__r0")
            sp = structural(ob.doc, c, "native") if ob is not None else []
            if sp:
                ref_struct.append({"case": c["case_id"], "arm": arm, "problems": sp})
    if ref_struct:
        r1 = False
        V["reasons"].append("R1: a native reference did not consume exactly the forced stream")
    V["reference_structural_failures"] = ref_struct
    amb_frac = ambiguous / cand_cells if cand_cells else 1.0
    r5 = amb_frac <= MAX_AMBIGUOUS_FRACTION
    if not r5:
        V["reasons"].append(f"R5: {amb_frac:.1%} of cells have native-unstable greedy decisions")
    V["candidate"] = {"cells": cand_cells, "failing_cells": cand_fail, "structural_failures": cand_struct, "ambiguous_cells": ambiguous,
                      "max_by_surface": {s: max((ce[s] for cells in rows["CAND"].values() for ce in cells), default=None) for s in SURFACES + ("next_maxabs",)},
                      "by_cycle_index": _by_cycle(rows["CAND"]),
                      "verify_only_cycle0_kl_max": max((v["kl"] for v in verify0.get("CAND", {}).values()), default=None),
                      "verify_only_within_native_tolerance": all(v["kl"] <= tau_native["next_kl"] for v in verify0.get("CAND", {}).values()),
                      "native_verify_only_cycle0_kl_max": {a: max((v["kl"] for v in verify0.get(a, {}).values()), default=None) for a in NATIVE_PAIRS}}
    # ---- R6 negative controls (discrimination)
    ncs, detected_all = [], True
    for n in cases_doc["negative_controls"]:
        c = case_by[n["base_case"]]
        ref, _ = load_obs(run, "A", f"A__{c['case_id']}__r0")
        x, err = load_obs(run, "CAND", f"CAND__{c['case_id']}__{n['mutation']}")
        rec = {"nc_id": n["nc_id"], "mutation": n["mutation"], "cycle": n["cycle"], "executed": bool(x and x.doc["valid"])}
        if rec["executed"] and ref is not None:
            cells, _ = compare(c, ref, x)
            ce = cells[n["cycle"]]
            rec["target"] = {s: {"value": ce[NC_SURFACE[s]], "tau": tau[NC_SURFACE[s]], "detected": ce[NC_SURFACE[s]] > tau[NC_SURFACE[s]]} for s in n["must_fail_surfaces_at_cycle"]}
            rec["detected"] = all(v["detected"] for v in rec["target"].values())
            rec["any_surface_fails_by_cycle"] = [[s for s in SURFACES if cc[s] > tau[s]] for cc in cells]
            rec["greedy_flip_by_cycle"] = [cc["greedy_x"] != cc["greedy_ref"] for cc in cells]
            rec["cycle0_clean"] = not rec["any_surface_fails_by_cycle"][0]
            rec["mutation_log"] = x.doc["records"]["mutation_log"]
            rec["structural"] = structural(x.doc, c, "cand", {"mutation": n["mutation"], "cycle": n["cycle"], "sibling_nodes": n.get("sibling_nodes")})
            x.drop(); ref.drop()
        else:
            rec["detected"] = None
        if n["mutation"] in REQUIRED_NCS and rec["detected"] is not True:
            detected_all = False
        if n["mutation"] not in REQUIRED_NCS and rec["executed"] and rec["detected"] is False:
            detected_all = False
        ncs.append(rec)
    V["negative_controls"] = ncs
    r6 = detected_all
    if not r6:
        V["reasons"].append("R6: a required negative control was not executed or not detected on its target surface")
    # ---- candidate within-process repeat (diagnostic)
    reps = []
    for cid in cases_doc["candidate_repeats"]:
        a, _ = load_obs(run, "CAND", f"CAND__{cid}__r0"); b, _ = load_obs(run, "CAND", f"CAND__{cid}__r1")
        if a and b and a.doc["valid"] and b.doc["valid"]:
            cells, _ = compare(case_by[cid], a, b)
            reps.append({"case": cid, "bitwise_equal_all_cycles": all(ce["bitwise_equal"] for ce in cells), "max_next_kl": max(ce["next_kl"] for ce in cells)})
    V["candidate_repeat"] = reps
    # ---- MTP / next-draft diagnostic (not gating)
    V["mtp_diagnostic"] = _mtp_diag(run, cases_doc, rows)
    # ---- verdict
    attributable = [f for f in cand_fail if (not f["held_out"]) or r4]
    gates = {"R1_integrity": r1, "R2_prompt_parity": r2, "R3_o0_import": r3, "R4_heldout_reference": r4, "R5_native_categorical": r5, "R6_discrimination": r6}
    V["gates"] = gates
    if all(gates.values()) and not cand_fail and not cand_struct:
        V["verdict"] = "EQUIVALENT"
    elif r1 and r2 and r3 and r6 and (attributable or cand_struct):
        V["verdict"] = "NOT_EQUIVALENT"
    else:
        V["verdict"] = "INCONCLUSIVE"
    V["cells_native"] = {a: rows[a] for a in NATIVE_PAIRS}
    V["cells_candidate"] = rows["CAND"]
    return V


def _by_cycle(rows):
    out = {}
    for cells in rows.values():
        for ce in cells:
            d = out.setdefault(ce["k"], {s: 0.0 for s in SURFACES})
            for s in SURFACES:
                d[s] = max(d[s], ce[s])
    return {str(k): v for k, v in sorted(out.items())}


def _mtp_diag(run, cases_doc, rows):
    """Agreement of the candidate's natural depth-1 spine draft (MTP top-1 after the commit) with native A's greedy
    next token after the same consumed prefix. Reported only (no native MTP reference exists in this design)."""
    agree = tot = 0
    for c in cases_doc["cases"]:
        ob, _ = load_obs(run, "CAND", f"CAND__{c['case_id']}__r0")
        if ob is None or c["case_id"] not in rows["CAND"]:
            continue
        cells = rows["CAND"][c["case_id"]]
        for d in ob.doc["records"]["natural_drafts"]:
            j = d["for_step"]
            if 1 <= j <= len(cells):
                tot += 1; agree += int(d["tokens"][0] == cells[j - 1]["greedy_ref"])
    return {"post_commit_steps": tot, "mtp_top1_equals_native_greedy": agree, "rate": (agree / tot) if tot else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import cases as CS
    import client as CL
    V = reduce(a.run, CS.load(), CL.plan)
    V["utc"] = Q.utc()
    Q.write_json(a.out, V)
    print(json.dumps({"verdict": V["verdict"], "gates": V["gates"], "reasons": V["reasons"], "tolerance": V["reference"]["tolerance"],
                      "candidate_failing_cells": len(V["candidate"]["failing_cells"])}, indent=1))
    return 0 if V["verdict"] in ("EQUIVALENT", "NOT_EQUIVALENT") else 3


if __name__ == "__main__":
    sys.exit(main())

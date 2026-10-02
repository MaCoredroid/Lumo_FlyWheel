#!/usr/bin/env python3
"""Post-hoc re-scoring of q1v3 v2 (DISCLOSED: defined after the frozen verdict and after seeing the data).

Frozen rule (unchanged, INCONCLUSIVE): tau_s = 4 x worst native-pair distance over ALL calibration cells.
Post-hoc rule: compare each candidate cell with the native arms at the SAME cell (case, cycle):
    R_s(cell) = max( max_{arm in B,V,P} d_s(arm vs A, cell), F_s ),   F_s = 0.02 (the pre-registered floors)
    candidate ratio  r_s(cell) = d_s(candidate vs A, cell) / R_s(cell)
    fault ratio      q(NC)     = d_target(NC, mutation cell) / R_target(base cell C2-nc-base, same cycle)
A multiplier c separates clean from faulty iff max_cells,s r_s < c < min_NC q. The analysis reports the whole
separating window instead of choosing c, plus the greedy-token agreement at every cell.
Usage: q1v3_posthoc.py VERDICT.json OUT.json"""
import json, sys

SURF = ["next_kl", "gdn_rel", "conv_rel", "kv_new_rel", "kv_hist_rel"]
NC_TARGET = {"conv": "conv_rel", "gdn": "gdn_rel", "kv": "kv_new_rel", "logits": "next_kl"}
F = 0.02


def cells(d):
    return d if isinstance(d, list) else [d[k] for k in sorted(d, key=int)]


def main():
    v = json.load(open(sys.argv[1]))
    cc, cn = v["cells_candidate"], v["cells_native"]
    arms = sorted(cn)
    rows, worst = [], {s: (0.0, None) for s in SURF}
    greedy = {"cells": 0, "greedy_equal": 0, "flips": []}
    for case, cl in cc.items():
        for c in cells(cl):
            k = c["k"]
            nat = {a: cells(cn[a][case])[k] for a in arms if case in cn[a]}
            row = {"case": case, "k": k, "path": c["path"]}
            for s in SURF:
                N = max(nat[a][s] for a in nat)
                R = max(N, F)
                r = c[s] / R
                row[s] = {"cand": round(c[s], 4), "native_max": round(N, 4), "ratio": round(r, 3)}
                if r > worst[s][0]:
                    worst[s] = (r, f"{case} k={k} cand={c[s]:.3f} native={N:.3f}")
            greedy["cells"] += 1
            if c["greedy_x"] == c["greedy_ref"]:
                greedy["greedy_equal"] += 1
            else:
                nat_flip = [a for a in nat if nat[a]["greedy_x"] != nat[a]["greedy_ref"]]
                greedy["flips"].append({"case": case, "k": k, "margin_ref": c["margin_ref"], "native_arms_also_flipping": nat_flip})
            rows.append(row)
    c_cand = max(w[0] for w in worst.values())
    ncs = []
    for n in v["negative_controls"]:
        prefix = n["nc_id"].split("__")[0]
        base = f"{prefix}__C2-nc-base"
        k = n["cycle"]
        nat = {a: cells(cn[a][base])[k] for a in arms}
        best = None
        for tgt, val in n["target"].items():
            s = NC_TARGET[tgt]
            R = max(max(nat[a][s] for a in nat), F)
            q = val["value"] / R
            cand_clean = cells(cc[base])[k][s]
            ent = {"surface": s, "value": round(val["value"], 4), "native_max": round(R, 4),
                   "clean_candidate": round(cand_clean, 4), "ratio": round(q, 3)}
            if best is None or q > best["ratio"]:
                best = ent
        ncs.append({"nc": n["nc_id"], "detect_if_c_below": best["ratio"], "via": best})
    c_nc = min(x["detect_if_c_below"] for x in ncs)
    out = {"schema": "q1v3.posthoc.v1", "disclosure": __doc__.split("Usage")[0].strip(),
           "frozen_verdict": v["verdict"], "frozen_reasons": v["reasons"], "floor": F, "native_arms": arms,
           "candidate_worst_ratio_by_surface": {s: {"ratio": round(w[0], 3), "where": w[1]} for s, w in worst.items()},
           "candidate_max_ratio": round(c_cand, 3), "nc_min_ratio": round(c_nc, 3),
           "separating_window": [round(c_cand, 3), round(c_nc, 3)] if c_cand < c_nc else None,
           "negative_controls": ncs, "greedy": greedy,
           "verdict_by_c": {str(c): ("EQUIVALENT" if (c_cand <= c and all(x["detect_if_c_below"] > c for x in ncs))
                                     else ("NOT_EQUIVALENT" if all(x["detect_if_c_below"] > c for x in ncs) else "INCONCLUSIVE (fault not detected)"))
                            for c in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0)},
           "cells": rows}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("frozen_verdict", "candidate_worst_ratio_by_surface", "candidate_max_ratio", "nc_min_ratio",
                                          "separating_window", "verdict_by_c")}, indent=1))
    for x in ncs:
        print(f"  {x['nc']:22s} q={x['detect_if_c_below']:6.2f}  via {x['via']}")
    print("greedy equal", greedy["greedy_equal"], "/", greedy["cells"], "flips:", greedy["flips"])


if __name__ == "__main__":
    main()

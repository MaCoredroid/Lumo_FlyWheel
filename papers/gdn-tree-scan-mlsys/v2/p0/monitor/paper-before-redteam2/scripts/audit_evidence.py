#!/usr/bin/env python3
"""Recompute paper accounting from archived data; does not run model inference."""
from pathlib import Path
import csv
import hashlib
import json
import math

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[2]
ARMS = {
    "native5": "native5_control_kvr1",
    "native11": "native11_control_kvr1",
    "tree": "kvremap_tail6_kvr1",
}


def main():
    rows, sources = [], []
    for arm, directory in ARMS.items():
        path = REPO / "output/fr13_kvremap_tail6" / directory / "deploy_speed_kvr1.json"
        d = json.loads(path.read_text())
        a, c, t, e = (d[k] for k in (
            "accept_per_event", "committed_per_event", "wall_s_per_event", "events_per_step"))
        assert math.isclose(c / t, d["measured_tps_fullstep_wall"], rel_tol=1e-10)
        assert math.isclose(c, a + 1, rel_tol=1e-10)
        row = dict(arm=arm, accepted=a, committed=c, event_ms=t * 1000,
                   tps=c / t, events_per_step=e,
                   verify_ms=d["s_per_fwd_gpu"] * 1000,
                   draft_ms=d["drafter_gpu_ms_per_step"] / e,
                   commit_ms=d["committer_gpu_ms_per_step"] / e,
                   host_ms=d["overhead_other_ms_per_event"],
                   alignment=d["fullstep_alignment_ratio"])
        assert math.isclose(sum(row[k] for k in ("verify_ms", "draft_ms", "commit_ms", "host_ms")),
                            row["event_ms"], rel_tol=1e-9)
        assert d["n_tasks"] == 16
        assert all(x.startswith("astropy__astropy-") for x in d["task_instance_ids"])
        rows.append(row)
        sources.append(dict(evidence_id="H3", path=str(path.relative_to(REPO)),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            scope="archived nominal-B4 observation; effective sampling audit pending"))
    by_arm = {r["arm"]: r for r in rows}
    n, t = by_arm["native5"], by_arm["tree"]
    derived = {
        "tree_accepted_break_even": n["committed"] * t["event_ms"] / n["event_ms"] - 1,
        "tree_event_ms_break_even": t["committed"] * n["event_ms"] / n["committed"],
        "committed_gain_vs_native5_pct": 100 * (t["committed"] / n["committed"] - 1),
        "event_cost_increase_vs_native5_pct": 100 * (t["event_ms"] / n["event_ms"] - 1),
    }
    derived["required_event_cost_reduction_pct"] = 100 * (
        1 - derived["tree_event_ms_break_even"] / t["event_ms"])
    for eid, rel in [
        ("H1", "FR13_REMAP_SHIP_RESULTS.md"),
        ("H2", "FR13_SLOT_REORDER_ARTIFACTS/VERDICT_TABLE.md"),
        ("H3", "FR13_POSTSNAPFIX3_CLOSEOUT.md"),
        ("H4", "results/fr13_series_closeout_20260815/README.md"),
        ("H5", "results/fr13_b4_exact16_qc_20260814/fr13_b4_exact16_qc.json"),
        ("H6", "FR13_REPLAY_CHASEDOWN_BANK.md"),
        ("H6", "docs/archive/wy/FR13_WY_VS_SEQUENTIAL_VERDICT.md"),
        ("H6", "FR13_DIFFUSE_GDN_EXPLAINED.md"),
        ("H6", "FR13_REALIZATION_AGREEMENT.md"),
        ("H6", "results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md"),
        ("H6", "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"),
    ]:
        p = REPO / rel
        sources.append(dict(evidence_id=eid, path=rel,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            scope=("documentary/code evidence; raw numerical captures not reproduced"
                                   if eid == "H6" else "archived source; not a fresh experiment")))
    result_dir = PAPER / "results"
    result_dir.mkdir(exist_ok=True)
    with (result_dir / "historical-metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (result_dir / "derived-accounting.json").write_text(json.dumps(derived, indent=2) + "\n")
    (PAPER / "notes/evidence-sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    commands = []
    for arm, prefix in [("native5", "NativeFive"), ("native11", "NativeEleven"), ("tree", "Tree")]:
        r = by_arm[arm]
        for field, suffix, fmt in [("accepted", "Accepted", ".3f"), ("committed", "Committed", ".3f"),
                                   ("event_ms", "EventMs", ".2f"), ("tps", "TPS", ".2f"),
                                   ("events_per_step", "Occupancy", ".2f")]:
            commands.append("\\newcommand{\\" + prefix + suffix + "}{" + format(r[field], fmt) + "}")
    commands += ["\\newcommand{\\BreakEvenAccepted}{%.2f}" % derived["tree_accepted_break_even"],
                 "\\newcommand{\\BreakEvenEventMs}{%.2f}" % derived["tree_event_ms_break_even"],
                 "\\newcommand{\\RequiredReduction}{%.1f}" % derived["required_event_cost_reduction_pct"]]
    (result_dir / "historical-macros.tex").write_text("% Generated from archived JSON.\n" + "\n".join(commands) + "\n")
    print(json.dumps({"archived_arms_checked": len(rows), "source_hashes": len(sources),
                      "derived": derived}, indent=2))


if __name__ == "__main__":
    main()

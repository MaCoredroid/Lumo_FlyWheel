#!/usr/bin/env python3
"""Assemble the STRUCTURED pilot freeze record from evidence on disk (review 10). It never invents criteria: the
judgment fields (candidate selection, tolerances, decision changes, timing precision, metric variance) are read from a
criteria JSON the operator writes AFTER reading the pilot SUMMARY; everything hash-bound is computed here from the run
dirs and the harness summary. Output is validated with check_pilot_freeze.py before it is written as FROZEN.
Usage: fill_pilot_freeze.py <frozen_prefixes.json> <harness_run_dir> <criteria.json> <out PILOT_FREEZE.json> <root>...
"""
import datetime as dt, hashlib, json, os, re, statistics, subprocess, sys
from pathlib import Path
frozen_p, harness, criteria_p, out_p = sys.argv[1:5]; roots = sys.argv[5:]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
pilot_ids = [r["id"] for r in json.load(open(frozen_p))["pilot"]]
ev = []
for root in roots:
    for d in sorted(Path(root).glob("capture_*_p*")):
        pid = d.name.split("_")[-1]
        if pid not in pilot_ids or not (d / "capture_provenance.json").exists() or not (d / "logs" / "tree_gdn_capture_payload.pt").exists():
            continue
        pv = json.load(open(d / "capture_provenance.json"))
        if not pv.get("all_pass"):
            continue
        ev.append({"prefix_id": pid, "run_dir": str(d), "payload_sha256": sha(d / "logs" / "tree_gdn_capture_payload.pt"),
                   "provenance_sha256": sha(d / "capture_provenance.json"), "inspector_sha256": pv.get("inspector_sha256")})
# review 11: a prefix with more than one PASSING run is REJECTED (never silently choose a favorable rerun)
dups = {e["prefix_id"] for e in ev if sum(x["prefix_id"] == e["prefix_id"] for x in ev) > 1}
if dups:
    print(f"REFUSING: duplicate passing runs for pilot prefixes {sorted(dups)}; resolve explicitly (record which run is evidence and why)", file=sys.stderr); sys.exit(41)
by = {e["prefix_id"]: e for e in ev}
ev = [by[p] for p in pilot_ids if p in by]
# costs are computed from the SELECTED one-run-per-prefix evidence set only
costs = {"boot_health_s": [], "request_s": [], "driver_cycle_s": [], "payload_bytes": []}
for e in ev:
    d = Path(e["run_dir"])
    tr = (d / "driver_trace.txt").read_text(); h = (d / "health.txt").read_text() if (d / "health.txt").exists() else ""
    m = re.search(r"healthy after (\d+)s", h); costs["boot_health_s"].append(int(m.group(1)) if m else None)
    req = json.load(open(d / "capture_request.json")); costs["request_s"].append(float(req.get("latency_s", 0)))
    t0 = re.search(r"start_utc=(\S+)", tr).group(1); t1 = re.search(r"exit_utc=(\S+)", tr).group(1)
    f = lambda x: dt.datetime.fromisoformat(x.replace("Z", "+00:00")); costs["driver_cycle_s"].append((f(t1) - f(t0)).total_seconds())
    costs["payload_bytes"].append((d / "logs" / "tree_gdn_capture_payload.pt").stat().st_size)
crit = json.load(open(criteria_p))
def agg(v):
    v = [x for x in v if x is not None]; return {"n": len(v), "median": statistics.median(v) if v else None, "min": min(v) if v else None, "max": max(v) if v else None}
cost = {k: (statistics.median([x for x in v if x is not None]) if v else None) for k, v in costs.items()}
cost["detail"] = {k: agg(v) for k, v in costs.items()}
cost["metric_variance"] = crit.get("metric_variance")
summ = Path(harness) / "summary.json"
rec = {"schema": "e7a.pilot_freeze.v1", "status": "FROZEN", "frozen_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "resampling_unit": "frozen prefix (one boot, one request, one one-shot payload per prefix; never layer/file/row)",
       "pilot_evidence": ev, "pilot_harness_summary": {"path": str(summ), "sha256": sha(summ) if summ.exists() else None, "summary_md_sha256": sha(Path(harness) / "SUMMARY.md") if (Path(harness) / "SUMMARY.md").exists() else None},
       "criteria": {"candidate_selection": crit["candidate_selection"], "tolerances": crit["tolerances"], "unacceptable_decision_changes": crit["unacceptable_decision_changes"],
                    "timing_precision_target": crit["timing_precision_target"], "cost_variance": cost},
       "notes": crit.get("notes", "")}
json.dump(rec, open(out_p, "w"), indent=2)
r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "check_pilot_freeze.py"), out_p, frozen_p, "--json", out_p + ".check.json"], capture_output=True, text=True)
print(r.stdout.strip().splitlines()[-1]); print(f"wrote {out_p} ({len(ev)} pilot entries); validator rc={r.returncode}")
sys.exit(r.returncode)

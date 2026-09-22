#!/usr/bin/env python3
"""E1 offline JOINER v4 (CPU). Binds recorder v4 events. Two supports, never conflated (review-17 E1 recheck):
  * OUTPUT LEDGER = every forward's output_rows (pure AND nonpure: prefill/mixed forwards emit tokens too, e.g. the API's
    first token) → per-request COMPLETE emitted sequence → bound to the API token stream (exact equality, or the API ids
    are a strict prefix whose missing tail lies entirely within the request's LAST recorded row = EOS/max_tokens; tail
    reported, not counted). Token evidence is MANDATORY for a throughput result.
  * PURE WALL SUPPORT = physical steps (pure-decode forwards with the timer's fwd_index) whose successor is consecutive
    in BOTH fwd_index and forward seq, no chain_break between, same cohort, expected occupancy, no discarded row; only
    THOSE steps' output rows count toward throughput. Long intervals (> --idle-cap-s, default 1.5 s) on such steps are
    RETAINED in the primary rate (they can be real slow compute/JIT/host work) and reported as a diagnostic together with
    a trimmed rate; idle is excluded only via an explicit scheduler/phase break (seq gap / chain_break).
  Validity (any failure → run INVALID, rc 2): first event recorder_probe; no error events; no errors/sink_failures
  counters; forward seq contiguous; every physical_step has a forward_entry and exactly one pure output_rows record
  with EXACTLY one row per active request (no duplicates/foreign ids); at most one output_rows per forward seq; a pure
  output_rows must reference an existing physical step (no orphans); per request, step_idx is 0,1,2,… over ALL its rows
  in forward order; every retained physical step has rows. Missing rows / orphan pure rows are instrumentation loss.
  CLOSE SEAL + FAIL FLAG (recheck case 2): event write counters `n` must be contiguous 1..N and the LAST event must be a
  run_close whose n_events == N and whose forward/physical-step/output-record totals match the file; a fail-flag sidecar
  (`e1_recorder_FAILED.flag` next to the events file, or --failflag) → INVALID.
  PHASE MANIFEST (recheck case 3): --manifest {"phases": {"warmup": [request ids], "timed": [request ids]}}: every recorded
  request must belong to exactly one phase; the API token map must cover ALL recorded requests (complete reconciliation,
  warm-up included); throughput uses only physical steps whose active set is entirely within the TIMED phase (steps with
  any warm-up request are the prespecified "warmup phase" exclusion; mixed warm-up/timed sets are excluded as "phase
  overlap"); per-phase token/wall summaries are reported.
Usage: e1_join.py <events.jsonl> --api-tokens <map.json> --manifest <manifest.json> [--expect-reqs N] [--idle-cap-s 1.5 (diagnostic only)] [--failflag PATH] [--json out]"""
import argparse, json, sys
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("events"); ap.add_argument("--json"); ap.add_argument("--api-tokens", default=None); ap.add_argument("--expect-reqs", type=int, default=None); ap.add_argument("--idle-cap-s", type=float, default=1.5)
    ap.add_argument("--manifest", default=None); ap.add_argument("--failflag", default=None); a = ap.parse_args()
    import os
    failflag = a.failflag or os.path.join(os.path.dirname(os.path.abspath(a.events)), "e1_recorder_FAILED.flag")
    ev = [json.loads(l) for l in open(a.events) if l.strip()]
    rep = {"n_events": len(ev), "invalid": None, "refused": None}
    def out(rc):
        if a.json: json.dump(rep, open(a.json, "w"), indent=1)
        print(json.dumps({k: v for k, v in rep.items() if k != "steps"})[:800]); sys.exit(rc)
    def invalid(msg): rep["invalid"] = msg; rep["refused"] = msg; out(2)
    if os.path.exists(failflag): invalid(f"recorder fail-flag sidecar present: {failflag}")
    if not ev or ev[0].get("event") != "recorder_probe": invalid("first event is not recorder_probe (sink not validated / file truncated)")
    ns = [e.get("n") for e in ev]
    if any(n is None for n in ns) or ns != list(range(1, len(ev) + 1)): invalid("event write counters not contiguous 1..N (lost or reordered writes)")
    last = ev[-1]
    if last.get("event") != "run_close": invalid("file does not end with a run_close seal (tail lost or unclean stop)")
    if int(last.get("n_events", -1)) != len(ev): invalid(f"run_close n_events {last.get('n_events')} != {len(ev)} events in file")
    if last.get("sink_failed"): invalid("run_close reports a sink failure")
    if any(e.get("event") == "error" for e in ev): invalid(f"{sum(1 for e in ev if e.get('event') == 'error')} recorder error event(s)")
    if any((e.get("errors") or 0) > 0 or (e.get("sink_failures") or 0) > 0 for e in ev): invalid("recorder reported errors/sink failures")
    fe, ps, orow, breaks = {}, {}, {}, []
    for e in ev:
        k = e.get("event")
        if k == "forward_entry":
            if e["seq"] in fe: invalid(f"duplicate forward seq {e['seq']}")
            fe[e["seq"]] = e
        elif k == "physical_step":
            if e["physical_step_id"] in ps: invalid(f"duplicate physical_step_id {e['physical_step_id']}")
            if int(e["num_reqs"]) != len(e["request_ids"]) or len(set(e["request_ids"])) != len(e["request_ids"]): invalid(f"physical step {e['physical_step_id']}: num_reqs/request_ids inconsistent")
            ps[e["physical_step_id"]] = e
        elif k == "output_rows":
            if e["seq"] in orow: invalid(f"duplicate output_rows for forward seq {e['seq']}")
            orow[e["seq"]] = e
        elif k == "chain_break": breaks.append(e["seq"])
    if not fe or sorted(fe) != list(range(min(fe), max(fe) + 1)): invalid("forward sequence missing or not contiguous")
    if int(last.get("n_forwards", -1)) != len(fe) or int(last.get("n_physical_steps", -1)) != len(ps) or int(last.get("n_output_records", -1)) != len(orow): invalid(f"run_close totals (forwards {last.get('n_forwards')}, steps {last.get('n_physical_steps')}, outputs {last.get('n_output_records')}) != file ({len(fe)}, {len(ps)}, {len(orow)})")
    for seq, o in orow.items():
        if seq not in fe: invalid(f"output_rows for unknown forward seq {seq}")
        if o["kind"] == "pure":
            if o["physical_step_id"] not in ps or ps[o["physical_step_id"]]["seq"] != seq: invalid(f"orphan pure output_rows (seq {seq}, physical step {o['physical_step_id']})")
        elif o["kind"] != "nonpure": invalid(f"unknown output kind {o['kind']}")
    pure_rows = {}
    for pid, p in ps.items():
        if p.get("seq") not in fe: invalid(f"physical step {pid} has no forward_entry")
        o = orow.get(p["seq"])
        if o is None or o["kind"] != "pure": invalid(f"physical step {pid} has no pure output rows (instrumentation loss)")
        req = list(p["request_ids"]); got = [x["request_id"] for x in o["rows"]]
        if len(got) != len(req) or len(set(got)) != len(got) or set(got) != set(req) or int(o.get("num_reqs", len(got))) != len(req): invalid(f"physical step {pid}: rows {sorted(got)} do not match the active request set {sorted(req)} exactly")
        pure_rows[pid] = o
    step_seen = {}
    for seq in sorted(orow):
        for x in orow[seq]["rows"]:
            exp = step_seen.get(x["request_id"], 0)
            if int(x["step_idx"]) != exp: invalid(f"request {x['request_id']}: step_idx {x['step_idx']} at forward {seq}, expected {exp}")
            step_seen[x["request_id"]] = exp + 1
            if len(x.get("emitted_ids", [])) != int(x.get("n_emitted", len(x.get("emitted_ids", [])))): invalid(f"row count field inconsistent at forward {seq}")
    phases = None
    if a.manifest:
        man = json.load(open(a.manifest)); phases = man.get("phases") or {}
        warm, timed = set(map(str, phases.get("warmup", []))), set(map(str, phases.get("timed", [])))
        if warm & timed: invalid(f"manifest: requests in both phases: {sorted(warm & timed)}")
        recorded = {x["request_id"] for o in orow.values() for x in o["rows"]}
        unk = recorded - warm - timed
        if unk: invalid(f"manifest: recorded requests in no phase: {sorted(unk)}")
        missing_timed = timed - recorded
        if missing_timed: invalid(f"manifest: timed requests never recorded: {sorted(missing_timed)}")
        rep["phases"] = {"warmup": sorted(warm), "timed": sorted(timed)}
    usable, excluded = [], {}
    def excl(pid, reason, wall=None): excluded[str(pid)] = {"reason": reason, "wall_s": wall}
    for pid in sorted(ps):
        p, o = ps[pid], pure_rows[pid]; nxt = ps.get(pid + 1)
        if nxt is None: excl(pid, "terminal: no successor physical step id+1"); continue
        wall = nxt["t_start"] - p["t_start"]
        if phases is not None:
            act = set(p["request_ids"])
            if act & warm and act & timed: excl(pid, "phase overlap (warm-up and timed requests active together)", wall); continue
            if act & warm: excl(pid, "warmup phase (prespecified exclusion)", wall); continue
        if nxt["seq"] != p["seq"] + 1: excl(pid, "mixed/prefill forward between (seq gap): not bridged", wall); continue
        if any(p["seq"] <= b <= nxt["seq"] for b in breaks): excl(pid, "chain_break between this step and its successor", wall); continue
        if set(nxt["request_ids"]) != set(p["request_ids"]): excl(pid, "cohort change at the successor (ramp/drain)", wall); continue
        if a.expect_reqs is not None and len(p["request_ids"]) != a.expect_reqs: excl(pid, f"occupancy {len(p['request_ids'])} != expected {a.expect_reqs} (ramp/drain)", wall); continue
        if not (wall > 0): invalid(f"non-positive wall {wall} at physical step {pid}")
        if any(x.get("discarded") for x in o["rows"]): excl(pid, "discarded row(s) in this step", wall); continue
        # E1 freeze decision (parent, 2026-09-22): a long wall on consecutive pure same-cohort forwards is NOT excluded
        # from the primary rate (it can be real slow compute / JIT / host work); it is RETAINED and reported over the
        # diagnostic cap; idle is excluded only by an explicit scheduler/phase break (seq gap / chain_break above).
        usable.append({"physical_step_id": pid, "seq": p["seq"], "wall_s": wall, "num_reqs": len(p["request_ids"]), "rows": o["rows"], "over_cap": wall > a.idle_cap_s})
    rep.update({"n_forwards": len(fe), "n_output_records": len(orow), "n_nonpure_output_records": sum(1 for o in orow.values() if o["kind"] == "nonpure"), "n_physical_steps": len(ps), "n_chain_breaks": len(breaks), "n_usable": len(usable), "n_excluded": len(excluded), "excluded": excluded,
                "excluded_wall_s_total": sum(v["wall_s"] for v in excluded.values() if v["wall_s"])})
    if not a.api_tokens: rep["refused"] = "no API token evidence supplied: structure reported, no throughput"; rep["steps"] = usable; out(3)
    api = json.load(open(a.api_tokens)); seq_by_req = {}
    for seq in sorted(orow):                       # the COMPLETE ledger (pure + nonpure), forward order
        for x in orow[seq]["rows"]: seq_by_req.setdefault(x["request_id"], []).append((seq, x))
    bound = {}
    for rid, lst in seq_by_req.items():
        if rid not in api: invalid(f"request {rid} has no API token evidence")
        ids = api[rid]; cat = [t for _, x in lst for t in x["emitted_ids"]]
        if cat == ids: bound[rid] = {seq: len(x["emitted_ids"]) for seq, x in lst}; continue
        last_seq, last = lst[-1]; head = cat[:len(cat) - len(last["emitted_ids"])]
        if len(ids) < len(cat) and cat[:len(ids)] == ids and len(ids) >= len(head):
            bound[rid] = {seq: len(x["emitted_ids"]) for seq, x in lst[:-1]}; bound[rid][last_seq] = len(ids) - len(head); rep.setdefault("truncated_tail_tokens", {})[rid] = len(cat) - len(ids); continue
        invalid(f"token identity: request {rid} recorder ids differ from the API stream (not a last-row truncation)")
    for rid in api:
        if rid not in seq_by_req: invalid(f"API request {rid} has no recorder rows")
    tot_wall = sum(u["wall_s"] for u in usable); tot_tok = sum(bound[x["request_id"]][u["seq"]] for u in usable for x in u["rows"]); n_rows = sum(len(u["rows"]) for u in usable)
    over = [u for u in usable if u["over_cap"]]; kept = [u for u in usable if not u["over_cap"]]
    tw_t = sum(u["wall_s"] for u in kept); tt_t = sum(bound[x["request_id"]][u["seq"]] for u in kept for x in u["rows"])
    rep["over_cap_diagnostic"] = {"cap_s": a.idle_cap_s, "n_intervals_over_cap": len(over), "wall_s_over_cap": sum(u["wall_s"] for u in over), "physical_step_ids": [u["physical_step_id"] for u in over],
                                  "tokens_per_wall_second_trimmed_DIAGNOSTIC_ONLY": (tt_t / tw_t) if tw_t > 0 else None}
    if phases is not None:
        rep["phase_summary"] = {ph: {"requests": len(ids_), "ledger_tokens": sum(len(x["emitted_ids"]) for o in orow.values() for x in o["rows"] if x["request_id"] in ids_), "api_tokens": sum(len(api[r]) for r in ids_ if r in api)} for ph, ids_ in (("warmup", warm), ("timed", timed))}
    rep.update({"token_evidence": {rid: len(ids) for rid, ids in api.items()}, "ledger_tokens_all_forwards": sum(len(x["emitted_ids"]) for o in orow.values() for x in o["rows"]),
                "sum_wall_s_unique_physical_steps": tot_wall, "sum_emitted_tokens_api_bound_pure_support": tot_tok, "n_request_rows": n_rows,
                "tokens_per_wall_second": (tot_tok / tot_wall) if tot_wall > 0 else None, "steps": usable})
    out(0 if usable else 3)
if __name__ == "__main__":
    main()

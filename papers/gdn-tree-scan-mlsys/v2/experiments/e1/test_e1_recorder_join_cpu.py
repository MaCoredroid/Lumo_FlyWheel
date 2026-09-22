#!/usr/bin/env python3
"""CPU controls for recorder v3 + joiner v3 — the review-17 E1 counterexample table plus the earlier cases. Every malformed
case must make the run INVALID (rc 2) or excluded by a prespecified rule; no case may return a biased success.
Usage: test_e1_recorder_join_cpu.py <out_json>"""
import json, os, sys, subprocess, tempfile, importlib, time
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here); res = {"checks": {}}
def check(n, ok, d=None): res["checks"][n] = {"pass": bool(ok), "detail": d}; print(("PASS " if ok else "FAIL ") + n, "" if d is None else str(d)[:150])
def fresh(path):
    os.environ["E1_RECORD"] = path; os.environ.pop("E1_RECORD_FAILFLAG", None); sys.modules.pop("e1_recorder", None); return importlib.import_module("e1_recorder")
def join(path, api=None, *extra):
    try:
        import e1_recorder as _m
        if os.environ.get("E1_RECORD") == path: _m.close("test")
    except Exception: pass
    out = path + ".join.json"; args = [sys.executable, os.path.join(here, "e1_join.py"), path, "--json", out] + (["--api-tokens", api] if api else []) + list(extra)
    r = subprocess.run(args, capture_output=True, text=True); return r.returncode, json.load(open(out))
PROBE = {"event": "recorder_probe", "t": 0.0}
def W(path, recs, seal=True, tamper_n=None):
    evs = [PROBE] + list(recs)
    if seal: evs.append({"event": "run_close", "reason": "test", "n_forwards": sum(1 for e in recs if e.get("event") == "forward_entry"), "n_physical_steps": sum(1 for e in recs if e.get("event") == "physical_step"), "n_output_records": sum(1 for e in recs if e.get("event") == "output_rows"), "sink_failed": False})
    with open(path, "w") as f:
        for i, r in enumerate(evs, 1):
            r = dict(r); r["n"] = i if tamper_n is None else tamper_n(i)
            if r.get("event") == "run_close": r["n_events"] = len(evs)
            f.write(json.dumps(r) + "\n")
def PS(seq, pid, t, rids): return {"event": "physical_step", "seq": seq, "physical_step_id": pid, "t_start": t, "num_reqs": len(rids), "request_ids": rids}
def FE(seq, n=1): return {"event": "forward_entry", "seq": seq, "num_reqs": n}
def CR(seq, pid, rows): return {"event": "output_rows", "seq": seq, "kind": "pure", "physical_step_id": pid, "num_reqs": len(rows), "rows": rows}
def NP(seq, rows): return {"event": "output_rows", "seq": seq, "kind": "nonpure", "physical_step_id": None, "num_reqs": len(rows), "rows": rows}
def row(b, rid, k, ids): return {"row": b, "request_id": rid, "step_idx": k, "emitted_ids": ids, "n_emitted": len(ids), "num_draft_tokens": None, "discarded": len(ids) == 0}
def api_of(td, name, m): p = os.path.join(td, name); json.dump(m, open(p, "w")); return p
with tempfile.TemporaryDirectory() as td:
    # --- B1 chain via the REAL recorder (probe first event; explicit req ids; dedupe error on a second commit)
    p = os.path.join(td, "b1.jsonl"); m = fresh(p); step_ids = [[100, 101], [102, 103, 104, 105], [106], [107, 108, 109], [110, 111, 112, 113, 114]]
    for k, ids in enumerate(step_ids):
        m.forward_entry(num_reqs=1, num_tokens=10, max_num_scheduled_tokens=10); m.physical_step(fwd_index=100 + k, num_reqs=1, request_ids=["r1"], cg_mode="NONE"); m.commit_rows(req_ids=["r1"], sampled_lists=[ids], num_draft_tokens=[9]); time.sleep(0.005)
    api = api_of(td, "api_b1.json", {"r1": [t for ids in step_ids for t in ids]}); rc, j = join(p, api, "--expect-reqs", "1")
    check("b1_chain_4_usable_api_bound_tokens_probe_first", rc == 0 and j["n_usable"] == 4 and j["sum_emitted_tokens_api_bound_pure_support"] == sum(len(x) for x in step_ids[:4]) and j["n_forwards"] == 5 and j["excluded"]["104"]["reason"].startswith("terminal"), {k: j.get(k) for k in ("n_usable", "sum_emitted_tokens_api_bound_pure_support")})
    rc, j0 = join(p); check("no_api_evidence_no_throughput_rc3", rc == 3 and "tokens_per_wall_second" not in j0)
    p2 = os.path.join(td, "b1dup.jsonl"); m = fresh(p2); m.forward_entry(num_reqs=1); m.physical_step(fwd_index=1, num_reqs=1, request_ids=["a"]); m.commit_rows(req_ids=["a"], sampled_lists=[[1]]); m.commit_rows(req_ids=["a"], sampled_lists=[[1]]); rc, j = join(p2); check("second_output_record_same_forward_is_error_invalid", rc == 2 and "error" in j["invalid"])
    # --- ORDINARY ROUTE (review-17 recheck): prefill forward (no pure physical step) emits the FIRST token; then pure decode steps; the API stream starts with that first token
    p4 = os.path.join(td, "prefill.jsonl"); m = fresh(p4)
    m.forward_entry(num_reqs=1, num_tokens=256, max_num_scheduled_tokens=256); m.chain_break(reason="non-pure step"); m.commit_rows(req_ids=["r1"], sampled_lists=[[500]])   # prefill: first token, no physical step
    ids_steps = [[501, 502], [503], [504, 505, 506]]
    for k, ids in enumerate(ids_steps):
        m.forward_entry(num_reqs=1, num_tokens=10, max_num_scheduled_tokens=10); m.physical_step(fwd_index=10 + k, num_reqs=1, request_ids=["r1"]); m.commit_rows(req_ids=["r1"], sampled_lists=[ids]); time.sleep(0.003)
    api = api_of(td, "api_prefill.json", {"r1": [500] + [t for ids in ids_steps for t in ids]}); rc, j = join(p4, api)
    check("ordinary_route_prefill_first_token_in_ledger_not_in_pure_support", rc == 0 and j["n_nonpure_output_records"] == 1 and j["ledger_tokens_all_forwards"] == 7 and j["n_usable"] == 2 and j["sum_emitted_tokens_api_bound_pure_support"] == 3 and j["token_evidence"] == {"r1": 7}, {k: j.get(k) for k in ("n_nonpure_output_records", "ledger_tokens_all_forwards", "n_usable", "sum_emitted_tokens_api_bound_pure_support")})
    p5 = os.path.join(td, "prefill_mid.jsonl"); m = fresh(p5)   # mixed forward in the MIDDLE (e.g. chunked prefill of another request) with an output row: recorded nonpure; interval not bridged
    m.forward_entry(num_reqs=1); m.physical_step(fwd_index=1, num_reqs=1, request_ids=["a"]); m.commit_rows(req_ids=["a"], sampled_lists=[[1]])
    m.forward_entry(num_reqs=2); m.chain_break(reason="non-pure step"); m.commit_rows(req_ids=["a", "b"], sampled_lists=[[2], [90]])
    m.forward_entry(num_reqs=2); m.physical_step(fwd_index=2, num_reqs=2, request_ids=["a", "b"]); m.commit_rows(req_ids=["a", "b"], sampled_lists=[[3], [91]])
    m.forward_entry(num_reqs=2); m.physical_step(fwd_index=3, num_reqs=2, request_ids=["a", "b"]); m.commit_rows(req_ids=["a", "b"], sampled_lists=[[4], [92]])
    api = api_of(td, "api_mid.json", {"a": [1, 2, 3, 4], "b": [90, 91, 92]}); rc, j = join(p5, api)
    check("mixed_forward_output_in_ledger_interval_not_bridged", rc == 0 and j["n_nonpure_output_records"] == 1 and "seq gap" in j["excluded"]["1"]["reason"] and j["n_usable"] == 1 and j["sum_emitted_tokens_api_bound_pure_support"] == 2, (j.get("excluded"), j.get("n_usable")))
    p6 = os.path.join(td, "orphan_pure.jsonl"); W(p6, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), {"event": "output_rows", "seq": 2, "kind": "pure", "physical_step_id": 999, "num_reqs": 1, "rows": [row(0, "a", 1, [2])]}])
    rc, j = join(p6); check("orphan_pure_output_rows_invalid", rc == 2 and "orphan" in j["invalid"])
    # --- I3: explicit ids; a permuted list at the call site is the caller's coordinate (rows keyed by the given ids); inconsistent length → error
    p3 = os.path.join(td, "i3.jsonl"); m = fresh(p3); m.forward_entry(num_reqs=2); m.physical_step(fwd_index=1, num_reqs=2, request_ids=["a", "b"]); m.commit_rows(req_ids=["b", "a"], sampled_lists=[[7], [8]])
    recs = [json.loads(l) for l in open(p3)]; cr = [r for r in recs if r["event"] == "commit_rows"][0]; check("i3_rows_labeled_by_explicit_call_site_ids", [(x["request_id"], x["emitted_ids"]) for x in cr["rows"]] == [("b", [7]), ("a", [8])])
    p3b = os.path.join(td, "i3b.jsonl"); m = fresh(p3b); m.forward_entry(num_reqs=2); m.physical_step(fwd_index=1, num_reqs=2, request_ids=["a", "b"]); m.commit_rows(req_ids=["a"], sampled_lists=[[7], [8]]); rc, j = join(p3b); check("i3_short_id_list_is_error_invalid", rc == 2)
    # --- I2: pure -> MIXED -> pure with consecutive fwd_index: not bridged (seq gap); chain_break also not bridged
    p = os.path.join(td, "mixed.jsonl"); W(p, [FE(1), PS(1, 50, 0.0, ["a"]), CR(1, 50, [row(0, "a", 0, [1, 2])]), FE(2, 2), {"event": "chain_break", "seq": 2, "reason": "non-pure step"}, FE(3), PS(3, 51, 0.5, ["a"]), CR(3, 51, [row(0, "a", 1, [3, 4])]), FE(4), PS(4, 52, 0.6, ["a"]), CR(4, 52, [row(0, "a", 2, [5])])])
    api = api_of(td, "api_m.json", {"a": [1, 2, 3, 4, 5]}); rc, j = join(p, api); check("i2_mixed_forward_between_pure_steps_not_bridged", rc == 0 and "seq gap" in j["excluded"]["50"]["reason"] and j["n_usable"] == 1 and abs(j["sum_wall_s_unique_physical_steps"] - 0.1) < 1e-9)
    p = os.path.join(td, "brk.jsonl"); W(p, [FE(1), PS(1, 60, 0.0, ["a"]), CR(1, 60, [row(0, "a", 0, [1])]), {"event": "chain_break", "seq": 1, "reason": "x"}, FE(2), PS(2, 61, 0.1, ["a"]), CR(2, 61, [row(0, "a", 1, [2])]), FE(3), PS(3, 62, 0.2, ["a"]), CR(3, 62, [row(0, "a", 2, [3])])])
    api = api_of(td, "api_brk.json", {"a": [1, 2, 3]}); rc, j = join(p, api); check("i2_chain_break_marker_not_bridged", rc == 0 and "chain_break" in j["excluded"]["60"]["reason"] and j["n_usable"] == 1)
    # --- reviewer table: B4 four rows, 0.24 s; with token evidence → 83.33; without → rc 3
    b4rows = lambda k: [row(i, r, k, [10 * i + 5 * k + t for t in range(5)]) for i, r in enumerate("abcd")]
    p = os.path.join(td, "b4.jsonl"); W(p, [FE(1, 4), PS(1, 200, 10.0, list("abcd")), CR(1, 200, b4rows(0)), FE(2, 4), PS(2, 201, 10.24, list("abcd")), CR(2, 201, b4rows(1))])
    api = api_of(td, "api_b4.json", {r: [10 * i + t for t in range(10)] for i, r in enumerate("abcd")}); rc, j = join(p, api, "--expect-reqs", "4")
    check("b4_wall_once_20_tokens_per_0p24s", rc == 0 and j["n_usable"] == 1 and j["sum_emitted_tokens_api_bound_pure_support"] == 20 and abs(j["tokens_per_wall_second"] - 83.3333) < 0.01)
    rc, j = join(p); check("b4_without_token_evidence_no_rate", rc == 3 and "tokens_per_wall_second" not in j)
    # --- reviewer table: one request duplicated twice in commit rows → INVALID
    p = os.path.join(td, "dupreq.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1]), row(1, "a", 0, [2])]), FE(2), PS(2, 2, 0.24, ["a"]), CR(2, 2, [row(0, "a", 1, [3])])])
    rc, j = join(p, api_of(td, "api_d.json", {"a": [1, 2, 3]})); check("duplicated_request_row_invalid", rc == 2 and "exactly" in j["invalid"])
    # --- reviewer table: num_reqs=4 but one request id and one row → INVALID (physical_step inconsistent)
    p = os.path.join(td, "nreq.jsonl"); W(p, [FE(1, 4), {"event": "physical_step", "seq": 1, "physical_step_id": 1, "t_start": 0.0, "num_reqs": 4, "request_ids": ["a"]}, CR(1, 1, [row(0, "a", 0, [1])]), FE(2), PS(2, 2, 0.24, ["a"]), CR(2, 2, [row(0, "a", 1, [2])])])
    rc, j = join(p, api_of(td, "api_n.json", {"a": [1, 2]})); check("num_reqs_mismatch_invalid", rc == 2 and "num_reqs" in j["invalid"])
    # --- reviewer table: four-request step followed by one-request successor → cohort change exclusion (not bridged)
    p = os.path.join(td, "cohort.jsonl"); W(p, [FE(1, 4), PS(1, 300, 0.0, list("abcd")), CR(1, 300, b4rows(0)), FE(2), PS(2, 301, 0.24, ["a"]), CR(2, 301, [row(0, "a", 1, [5, 6, 7, 8, 9])]), FE(3), PS(3, 302, 0.3, ["a"]), CR(3, 302, [row(0, "a", 2, [10])])])
    api = api_of(td, "api_c.json", {"a": list(range(0, 11)), "b": [10, 11, 12, 13, 14], "c": [20, 21, 22, 23, 24], "d": [30, 31, 32, 33, 34]}); rc, j = join(p, api)
    check("cohort_change_excluded_not_bridged", rc == 0 and "cohort" in j["excluded"]["300"]["reason"] and j["n_usable"] == 1 and j["steps"][0]["physical_step_id"] == 301)
    # --- reviewer table: missing one of two rows on a 100 s step, then a complete 0.1 s step → INVALID (instrumentation loss), never a biased rate
    p = os.path.join(td, "miss.jsonl"); W(p, [FE(1, 2), PS(1, 1, 0.0, ["a", "b"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2, 2), PS(2, 2, 100.0, ["a", "b"]), CR(2, 2, [row(0, "a", 1, [2]), row(1, "b", 0, [3])]), FE(3, 2), PS(3, 3, 100.1, ["a", "b"]), CR(3, 3, [row(0, "a", 2, [4]), row(1, "b", 1, [5])])])
    rc, j = join(p, api_of(td, "api_miss.json", {"a": [1, 2, 4], "b": [3, 5]})); check("missing_row_on_slow_step_invalidates_run", rc == 2 and "exactly" in j["invalid"])
    # --- reviewer table: orphan commit at nonexistent physical id 999 → INVALID
    p = os.path.join(td, "orphan.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), CR(2, 999, [row(0, "a", 1, [9])]), FE(3), PS(3, 2, 0.1, ["a"]), CR(3, 2, [row(0, "a", 2, [2])])])
    rc, j = join(p, api_of(td, "api_o.json", {"a": [1, 9, 2]})); check("orphan_commit_rows_invalid", rc == 2 and "orphan" in j["invalid"])
    # --- physical step without commit rows → INVALID (instrumentation loss)
    p = os.path.join(td, "nocr.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), FE(2), PS(2, 2, 0.1, ["a"]), CR(2, 2, [row(0, "a", 0, [2])])])
    rc, j = join(p, api_of(td, "api_nc.json", {"a": [2]})); check("physical_step_without_rows_invalid", rc == 2 and "instrumentation loss" in j["invalid"])
    # --- step_idx progression violated → INVALID
    p = os.path.join(td, "stepidx.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), PS(2, 2, 0.1, ["a"]), CR(2, 2, [row(0, "a", 5, [2])])])
    rc, j = join(p, api_of(td, "api_s.json", {"a": [1, 2]})); check("step_idx_progression_invalid", rc == 2 and "step_idx" in j["invalid"])
    # --- token evidence: recorder ids 10,11 vs API 999,888 (equal count) → INVALID; EOS truncation in last row → accepted, tail not counted; not-last-row → INVALID
    p = os.path.join(td, "tok.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [10])]), FE(2), PS(2, 2, 0.1, ["a"]), CR(2, 2, [row(0, "a", 1, [11])]), FE(3), PS(3, 3, 0.2, ["a"]), CR(3, 3, [row(0, "a", 2, [12])])])
    rc, j = join(p, api_of(td, "api_bad.json", {"a": [999, 888, 777]})); check("equal_count_wrong_ids_invalid", rc == 2 and "token identity" in j["invalid"])
    p = os.path.join(td, "trunc.jsonl"); W(p, [FE(1), PS(1, 70, 0.0, ["a"]), CR(1, 70, [row(0, "a", 0, [1, 2])]), FE(2), PS(2, 71, 0.1, ["a"]), CR(2, 71, [row(0, "a", 1, [3, 4, 5])]), FE(3), PS(3, 72, 0.2, ["b"]), CR(3, 72, [row(0, "b", 0, [6])])])
    rc, j = join(p, api_of(td, "api_t.json", {"a": [1, 2, 3, 4], "b": [6]})); check("eos_truncation_in_last_row_tail_not_counted", rc == 0 and j.get("truncated_tail_tokens") == {"a": 1} and j["sum_emitted_tokens_api_bound_pure_support"] == 2, (j.get("truncated_tail_tokens"), j.get("sum_emitted_tokens_api_bound_pure_support"), j.get("excluded")))
    p = os.path.join(td, "trunc2.jsonl"); W(p, [FE(1), PS(1, 70, 0.0, ["a"]), CR(1, 70, [row(0, "a", 0, [1, 2])]), FE(2), PS(2, 71, 0.1, ["a"]), CR(2, 71, [row(0, "a", 1, [3, 4, 5])]), FE(3), PS(3, 72, 0.2, ["a"]), CR(3, 72, [row(0, "a", 2, [6])])])
    rc, j = join(p, api_of(td, "api_t2.json", {"a": [1, 2, 3, 4]})); check("truncation_not_in_last_row_invalid", rc == 2)
    # --- idle cap and occupancy exclusions (prespecified, reported with wall)
    p = os.path.join(td, "idle.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), PS(2, 2, 3.0, ["a"]), CR(2, 2, [row(0, "a", 1, [2])]), FE(3), PS(3, 3, 3.1, ["a"]), CR(3, 3, [row(0, "a", 2, [3])])])
    rc, j = join(p, api_of(td, "api_i.json", {"a": [1, 2, 3]})); check("long_same_cohort_interval_retained_and_reported", rc == 0 and j["n_usable"] == 2 and abs(j["sum_wall_s_unique_physical_steps"] - 3.1) < 1e-9 and j["over_cap_diagnostic"]["n_intervals_over_cap"] == 1 and abs(j["over_cap_diagnostic"]["wall_s_over_cap"] - 3.0) < 1e-9 and abs(j["over_cap_diagnostic"]["tokens_per_wall_second_trimmed_DIAGNOSTIC_ONLY"] - 10.0) < 1e-6 and abs(j["tokens_per_wall_second"] - 2 / 3.1) < 1e-6, (j.get("n_usable"), j.get("over_cap_diagnostic")))
    rc, j = join(p, api_of(td, "api_i.json", {"a": [1, 2, 3]}), "--expect-reqs", "4"); check("occupancy_mismatch_excluded", rc == 3 and all("occupancy" in v["reason"] or "terminal" in v["reason"] for v in j["excluded"].values()))
    # --- I5: recorder never raises on a bad sink; a run whose file lacks the probe / carries sink failures is INVALID
    os.environ["E1_RECORD_FAILFLAG"] = os.path.join(td, "fail.flag"); bad = fresh("/dev/null/child"); os.environ["E1_RECORD_FAILFLAG"] = os.path.join(td, "fail.flag")
    try: bad.forward_entry(num_reqs=1); bad.physical_step(fwd_index=1, num_reqs=1, request_ids=["a"]); bad.commit_rows(req_ids=["a"], sampled_lists=[[1]]); bad.chain_break(reason="x"); raised = False
    except Exception: raised = True
    check("i5_bad_sink_never_raises_flag_written", not raised and bad._CUR["sink_failed"] and os.path.exists(os.path.join(td, "fail.flag")))
    p = os.path.join(td, "noprobe.jsonl"); open(p, "w").write(json.dumps(FE(1)) + "\n"); rc, j = join(p); check("i5_missing_probe_invalid", rc == 2 and "recorder_probe" in j["invalid"])
    p = os.path.join(td, "sinkfail.jsonl"); W(p, [dict(FE(1), sink_failures=1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])])]); rc, j = join(p); check("i5_sink_failure_counter_invalid", rc == 2)
    # ---- recheck case 2: close seal + fail flag
    p = os.path.join(td, "noseal.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), PS(2, 2, 0.1, ["a"]), CR(2, 2, [row(0, "a", 1, [2])])], seal=False)
    rc, j = join(p, api_of(td, "api_ns.json", {"a": [1, 2]})); check("missing_close_seal_invalid", rc == 2 and "run_close" in j["invalid"])
    p = os.path.join(td, "lost_tail.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), PS(2, 2, 0.1, ["a"]), CR(2, 2, [row(0, "a", 1, [2])])])
    lines = open(p).read().splitlines(); open(p, "w").write("\n".join(lines[:-2]) + "\n" + lines[-1] + "\n")   # drop the last output record but keep the seal (counters now non-contiguous)
    rc, j = join(p, api_of(td, "api_lt.json", {"a": [1, 2]})); check("lost_middle_write_counter_gap_invalid", rc == 2 and ("counters" in j["invalid"] or "n_events" in j["invalid"]))
    p = os.path.join(td, "sealmismatch.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])])]); lines = open(p).read().splitlines(); last = json.loads(lines[-1]); last["n_output_records"] = 5; lines[-1] = json.dumps(last); open(p, "w").write("\n".join(lines) + "\n")
    rc, j = join(p, api_of(td, "api_sm.json", {"a": [1]})); check("seal_totals_mismatch_invalid", rc == 2 and "run_close totals" in j["invalid"])
    p = os.path.join(td, "flagged.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["a"]), CR(1, 1, [row(0, "a", 0, [1])]), FE(2), PS(2, 2, 0.1, ["a"]), CR(2, 2, [row(0, "a", 1, [2])])]); open(os.path.join(td, "e1_recorder_FAILED.flag"), "w").write("x")
    rc, j = join(p, api_of(td, "api_fl.json", {"a": [1, 2]})); check("fail_flag_sidecar_invalid", rc == 2 and "fail-flag" in j["invalid"]); os.remove(os.path.join(td, "e1_recorder_FAILED.flag"))
    # real recorder: explicit close writes a matching seal; a second close is idempotent
    p = os.path.join(td, "sealed.jsonl"); m = fresh(p); m.forward_entry(num_reqs=1); m.physical_step(fwd_index=1, num_reqs=1, request_ids=["a"]); m.commit_rows(req_ids=["a"], sampled_lists=[[1]]); m.forward_entry(num_reqs=1); m.physical_step(fwd_index=2, num_reqs=1, request_ids=["a"]); m.commit_rows(req_ids=["a"], sampled_lists=[[2]]); m.close("test"); m.close("test")
    rc, j = join(p, api_of(td, "api_sd.json", {"a": [1, 2]})); check("real_recorder_seal_matches_file", rc == 0 and json.loads(open(p).read().splitlines()[-1])["event"] == "run_close" and open(p).read().count('"run_close"') == 1)
    # ---- recheck case 3: phase manifest (warm-up vs timed) with complete API reconciliation
    p = os.path.join(td, "phases.jsonl"); W(p, [FE(1), PS(1, 1, 0.0, ["w"]), CR(1, 1, [row(0, "w", 0, [1])]), FE(2), PS(2, 2, 0.1, ["w"]), CR(2, 2, [row(0, "w", 1, [2])]), FE(3), PS(3, 3, 0.2, ["w", "t"]), CR(3, 3, [row(0, "w", 2, [3]), row(1, "t", 0, [10])]), FE(4), PS(4, 4, 0.3, ["t"]), CR(4, 4, [row(0, "t", 1, [11])]), FE(5), PS(5, 5, 0.4, ["t"]), CR(5, 5, [row(0, "t", 2, [12])]), FE(6), PS(6, 6, 0.5, ["t"]), CR(6, 6, [row(0, "t", 3, [13])])])
    man = api_of(td, "manifest.json", {"phases": {"warmup": ["w"], "timed": ["t"]}}); full = api_of(td, "api_full.json", {"w": [1, 2, 3], "t": [10, 11, 12, 13]}); timed_only = api_of(td, "api_timed.json", {"t": [10, 11, 12, 13]})
    rc, j = join(p, full, "--manifest", man); check("phase_manifest_full_api_map_timed_only_support", rc == 0 and j["n_usable"] == 2 and j["sum_emitted_tokens_api_bound_pure_support"] == 2 and "warmup phase" in j["excluded"]["1"]["reason"] and "phase overlap" in j["excluded"]["3"]["reason"] and j["phase_summary"]["warmup"]["ledger_tokens"] == 3 and j["phase_summary"]["timed"]["api_tokens"] == 4, (j.get("n_usable"), j.get("excluded"), j.get("phase_summary")))
    rc, j = join(p, timed_only, "--manifest", man); check("timed_only_api_map_refused_incomplete_reconciliation", rc == 2 and "no API token evidence" in j["invalid"])
    man2 = api_of(td, "manifest2.json", {"phases": {"warmup": [], "timed": ["t"]}}); rc, j = join(p, full, "--manifest", man2); check("request_in_no_phase_invalid", rc == 2 and "no phase" in j["invalid"])
    os.environ.pop("E1_RECORD", None); sys.modules.pop("e1_recorder", None); m = importlib.import_module("e1_recorder"); m.forward_entry(num_reqs=1); check("recorder_inert_without_env", m._CUR["seq"] == 0)
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(sys.argv[1], "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)

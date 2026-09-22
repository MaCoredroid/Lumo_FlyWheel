#!/usr/bin/env bash
# Real-data smoke test of e1_cell_summary.py on a SCRATCH COPY of the policy-B B1 pilot (token-id API domain): synthesizes
# the workload manifest (timed = the one request; warm-up none) and expects the pipeline to run (map -> phase translation to
# engine ids -> joiner -> per-prompt accounting) and to return INSUFFICIENT_SUPPORT (1 of 8 prompts), never INVALID.
# Also a negative: a manifest phase id that maps to no engine id must give INVALID. Nothing in the pilot root is written.
set -uo pipefail
EXP=/home/mark/lumo-paper-v2-20260921/papers/gdn-tree-scan-mlsys/v2/experiments; B1=$EXP/out-20260922T082815Z-e2-b1-policyB-sync-p072/e7b_001_p072_arm1_none-B_fs_ieee-all
S=$(mktemp -d /tmp/claude-1000/-home-mark-lumo-paper-v2-20260921/c8955ff9-f3ae-4551-869b-1a98b38ca2c3/scratchpad/e1sum.XXXX); OUT=$1
mk() { local d=$1 badid=$2; mkdir -p "$d/logs" "$d/cohort/req_t0_p072"; cp "$B1/logs/e1_events.jsonl" "$d/logs/"; cp "$B1/capture_request.json" "$d/cohort/req_t0_p072/capture_request.json"
  python3 - "$d" "$badid" <<'PY'
import json, sys
d, bad = sys.argv[1], sys.argv[2] == "1"
cr = json.load(open(d + "/cohort/req_t0_p072/capture_request.json")); rid = ("cmpl-DOESNOTEXIST" if bad else cr["response_id"])
n = len([t for t in cr["response_logprobs_tokens"] if str(t).startswith("token_id:")])
man = {"phases": {"warmup": [], "timed": [rid]}, "batch": 1, "submission": [{"phase": "timed", "slots": ["t0"], "utc": "x"}],
       "requests": [{"phase": "timed", "slot": "t0", "prefix_id": "p072", "response_id": rid, "n_api_tokens": n, "finish_reason": cr["finish_reason"], "max_tokens": 32}]}
json.dump(man, open(d + "/e1_manifest.json", "w"))
PY
}
mk "$S/ok" 0; mk "$S/bad" 1
# C6 fixture: complete, reconciled single request with ONE pure forward (terminal only) -> joiner rc 3 -> INSUFFICIENT_SUPPORT
mkdir -p "$S/zero/logs" "$S/zero/cohort/req_t0_p072"; python3 - "$S/zero" <<'PY2'
import json, sys
d = sys.argv[1]; rid = "cmpl-zero-0-eng"; ids = [11, 12, 13]
ev = [{"event": "recorder_probe", "n": 1}, {"event": "forward_entry", "seq": 1, "num_reqs": 1, "num_tokens": 256, "n": 2}, {"event": "output_rows", "seq": 1, "kind": "nonpure", "physical_step_id": None, "num_reqs": 1, "rows": [{"row": 0, "request_id": rid, "step_idx": 0, "emitted_ids": [11], "n_emitted": 1, "num_draft_tokens": None, "discarded": False}], "n": 3},
      {"event": "forward_entry", "seq": 2, "num_reqs": 1, "num_tokens": 10, "n": 4}, {"event": "physical_step", "seq": 2, "physical_step_id": 0, "t_start": 1.0, "t_wall_utc": "x", "num_reqs": 1, "request_ids": [rid], "n": 5}, {"event": "output_rows", "seq": 2, "kind": "pure", "physical_step_id": 0, "num_reqs": 1, "rows": [{"row": 0, "request_id": rid, "step_idx": 1, "emitted_ids": [12, 13], "n_emitted": 2, "num_draft_tokens": 9, "discarded": False}], "n": 6},
      {"event": "run_close", "reason": "atexit", "n_events": 7, "n_forwards": 2, "n_physical_steps": 1, "n_output_records": 2, "sink_failed": False, "n": 7}]
open(d + "/logs/e1_events.jsonl", "w").write("".join(json.dumps(e) + "\n" for e in ev))
json.dump({"response_id": "cmpl-zero", "finish_reason": "stop", "usage": {"prompt_tokens": 256, "completion_tokens": 3}, "response_logprobs_tokens": [f"token_id:{t}" for t in ids]}, open(d + "/cohort/req_t0_p072/capture_request.json", "w"))
json.dump({"phases": {"warmup": [], "timed": ["cmpl-zero"]}, "batch": 1, "submission": [{"phase": "timed", "slots": ["t0"], "utc": "x"}], "requests": [{"phase": "timed", "slot": "t0", "prefix_id": "p072", "response_id": "cmpl-zero", "n_api_tokens": 3, "finish_reason": "stop", "max_tokens": 128}]}, open(d + "/e1_manifest.json", "w"))
PY2
python3 "$EXP/e1/e1_cell_summary.py" "$S/zero" --batch 1 --snapshot "$EXP/e1" > "$S/zero.log" 2>&1; rc3=$?
python3 "$EXP/e1/e1_cell_summary.py" "$S/ok" --batch 1 --snapshot "$EXP/e1" > "$S/ok.log" 2>&1; rc1=$?
python3 "$EXP/e1/e1_cell_summary.py" "$S/bad" --batch 1 --snapshot "$EXP/e1" > "$S/bad.log" 2>&1; rc2=$?
python3 - "$S" "$rc1" "$rc2" "$OUT" "$rc3" <<'PY'
import json, sys, os
S, rc1, rc2, out, rc3 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
zero = json.load(open(S + "/zero/cell_result.preliminary.json"))
ok = json.load(open(S + "/ok/cell_result.preliminary.json")); bad = json.load(open(S + "/bad/cell_result.preliminary.json"))
res = {"checks": {}}
def check(n, c, d=None): res["checks"][n] = {"pass": bool(c), "detail": d}; print(("PASS " if c else "FAIL ") + n, "" if d is None else str(d)[:200])
check("b1_pilot_pipeline_runs_insufficient_support_not_invalid", rc1 == 0 and ok["status"] == "INSUFFICIENT_SUPPORT" and ok["floor"]["pass"] is False and ok.get("per_prompt", {}).get("p072", {}).get("intervals", 0) == 11 and ok["primary"]["n_usable"] == 11, {"status": ok["status"], "floor": ok["floor"], "per_prompt": ok.get("per_prompt")})
check("phase_translated_to_engine_ids", all(k.startswith("cmpl-") and "-0-" in k for k in json.load(open(S + "/ok/e1_join_manifest.json"))["phases"]["timed"]), json.load(open(S + "/ok/e1_join_manifest.json")))
check("unmapped_phase_id_is_invalid", rc2 != 0 and bad["status"] == "INVALID", bad.get("reasons"))
check("summary_writes_preliminary_only_no_terminal_result", ok.get("preliminary") is True and not os.path.exists(S + "/ok/cell_result.json"))
check("zero_usable_intervals_complete_reconciliation_is_insufficient_support_not_invalid", rc3 == 0 and zero["status"] == "INSUFFICIENT_SUPPORT" and zero["join"]["invalid"] is None and zero["join"]["refused"] is None, (rc3, zero.get("status"), zero.get("reasons")))
res["all_pass"] = all(v["pass"] for v in res["checks"].values()); json.dump(res, open(out, "w"), indent=2); print("ALL PASS" if res["all_pass"] else "SOME FAILED")
PY

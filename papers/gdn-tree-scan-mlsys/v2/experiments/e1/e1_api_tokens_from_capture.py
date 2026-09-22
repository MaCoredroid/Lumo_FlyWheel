#!/usr/bin/env python3
"""Build the joiner's --api-tokens map {request_id: [emitted token ids]} from the driver's capture_request.json files
(single request: <run>/capture_request.json; cohort: <run>/cohort/req_*/capture_request.json). The recorder labels rows
with the ENGINE request id; the OpenAI server derives it from the API response id (`cmpl-…`, optionally suffixed
`-<n>` for n>1). We key by the API response id and ALSO emit every recorder id seen in the events file that matches an
API id by exact or prefix relation (fail closed: an ambiguous or unmatched recorder request is reported and the map is
refused). Usage: e1_api_tokens_from_capture.py <run_dir> <events.jsonl> <out_map.json>"""
import glob, json, os, sys
def ids_of(rec):
    t = rec.get("response_logprobs_tokens") or []
    return [int(x[9:]) for x in t if isinstance(x, str) and x.startswith("token_id:")]
def main():
    run, events, out = sys.argv[1:4]
    files = [os.path.join(run, "capture_request.json")] if os.path.exists(os.path.join(run, "capture_request.json")) else []
    files += sorted(glob.glob(os.path.join(run, "cohort", "req_*", "capture_request.json")))
    api = {}
    for f in files:
        rec = json.load(open(f)); rid = rec.get("response_id") or rec.get("id")
        if not rid: print("REFUSING: capture_request without a response id:", f); sys.exit(2)
        api[str(rid)] = ids_of(rec)
    seen = set()
    for l in open(events):
        if not l.strip(): continue
        e = json.loads(l)
        if e.get("event") == "output_rows":
            for x in e["rows"]: seen.add(str(x["request_id"]))
        elif e.get("event") == "physical_step":
            for r in e["request_ids"]: seen.add(str(r))
    mapped = {}; problems = []
    for rid in sorted(seen):
        cands = [a for a in api if rid == a or rid.startswith(a + "-") or a.startswith(rid + "-")]
        if len(cands) == 1: mapped[rid] = api[cands[0]]
        else: problems.append({"recorder_request_id": rid, "candidates": cands})
    if problems or not mapped:
        json.dump({"refused": "unmatched or ambiguous recorder request ids", "problems": problems, "api_ids": sorted(api)}, open(out, "w"), indent=1); print("REFUSING:", problems[:3], "api ids", sorted(api)[:4]); sys.exit(2)
    json.dump(mapped, open(out, "w"), indent=1); print("api-tokens map:", {k: len(v) for k, v in mapped.items()}); sys.exit(0)
if __name__ == "__main__":
    main()

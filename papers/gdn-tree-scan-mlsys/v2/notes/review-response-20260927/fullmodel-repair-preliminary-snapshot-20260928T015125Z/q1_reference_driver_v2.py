#!/usr/bin/env python3
"""Sequential native-only calibration/smoke driver **v2** for the spec-off reference engine (host side).

R5 lifecycle/accounting repairs: inputs validated before any request (job sha, fixture canonical, positive repeats, exact case ids from
the JOB, fresh output namespace); one control per (case, repeat) carrying run/arm/process/repeat identity; after each request the driver
AUTHENTICATES the sealed document (path derived from the observation id, record sha recomputed, run/arm/process/repeat/case fields, `valid`
true, referenced raw objects present with matching byte length) and requires `usage.prompt_tokens == |P|`; on an HTTP error, timeout or an
unsealed/invalid case the run STOPS (no further control/request while the prior request may be active) and a failure receipt is kept;
a final DRIVER-VERDICT binds every expected observation to its authenticated seal.  Nothing here interprets O2 outcomes.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, struct, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_reference_hooks_v2 import authenticate_objects  # noqa: E402  (same length/hash contract the hooks use when reconstructing)

SCHEMA = "lumo.q1.fullmodel.reference-driver.v2"
HERE = os.path.dirname(os.path.abspath(__file__)); CAMPAIGN = os.path.dirname(HERE)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(open(p, "rb").read())


def http_post(url, payload, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def authenticate_seal(path, expect, objects_root):
    """Return (ok, problems, doc) for a sealed case document."""
    pr = []
    if not os.path.exists(path):
        return False, ["seal file missing"], None
    doc = json.load(open(path)); body = dict(doc); rs = body.pop("record_sha256", None)
    if rs != hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest():
        pr.append("record_sha256 mismatch")
    for k, v in expect.items():
        if doc.get(k) != v:
            pr.append(f"{k}={doc.get(k)!r} != expected {v!r}")
    if doc.get("valid") is not True:
        pr.append(f"seal not valid: {doc.get('problems')}")
    refs = []
    for tag in ("o0", "o1"):
        o = doc.get(tag) or {}
        for name, g in (o.get("gdn") or {}).items():
            refs += [g["conv"]["sha256"], g["ssm"]["sha256"]]
        for name, a in (o.get("attention") or {}).items():
            for blk in a.get("full_blocks", []):
                refs += [blk["k"], blk["v"]]
            if a.get("tail"):
                refs += [a["tail"]["k"], a["tail"]["v"]]
    if doc.get("o2") and doc["o2"].get("raw"):
        refs.append(doc["o2"]["raw"]["sha256"])
    missing = [r for r in set(refs) if not os.path.exists(os.path.join(objects_root, r + ".bin"))]
    if missing:
        pr.append(f"{len(missing)} referenced raw objects missing")
    # v2 repair (preliminary review defect 1): every referenced GDN / K / V / O2 object is hash + exact-length authenticated (dedup by sha)
    opr, n_obj, n_bytes = authenticate_objects(doc, objects_root)
    pr += opr
    if not opr and len(set(refs)) != n_obj:
        pr.append(f"authenticated {n_obj} objects != {len(set(refs))} distinct references")
    doc["_raw_objects"] = {"authenticated": n_obj, "bytes": n_bytes, "problems": len(opr)}
    return (not pr), pr, doc


def run(job, job_sha, prefix_plan_dir, out_dir, control_path, url, hooks_out_dir, post_fn=http_post, wait_fn=None, poll_s=0.5, seal_timeout_s=1800.0):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise SystemExit("output namespace not fresh")
    os.makedirs(out_dir, exist_ok=True)
    if int(job["repeats"]) <= 0 or not job["cases"]:
        raise SystemExit("job has no positive repeats / no cases")
    fx = json.load(open(os.path.join(CAMPAIGN, job["fixtures"]["path"])))
    if fx["canonical_sha256_excluding_timestamp"] != job["fixtures"]["canonical"] or sha256_file(os.path.join(CAMPAIGN, job["fixtures"]["path"])) != job["fixtures"]["sha256"]:
        raise SystemExit("fixtures changed since the job was built")
    objects_root = os.path.join(hooks_out_dir, "objects"); cases_root = os.path.join(hooks_out_dir, "cases")
    receipts, stop_reason = [], None
    started = utc()
    for c in job["cases"]:
        pre = job["prefixes"][c["prefix_id"]]
        raw = open(os.path.join(prefix_plan_dir, pre["token_ids_u32le"]), "rb").read(); ids = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
        if sha256_bytes(raw) != pre["token_ids_sha256"] or len(ids) != pre["prefix_len"]:
            raise SystemExit(f"{c['prefix_id']}: prefix bytes changed")
        for r in range(int(job["repeats"])):
            if stop_reason:
                break
            obs_id = f"{job['run_id']}__{job['arm']}__p{job['process']}__r{r}__{c['case_id']}"
            control = {"case_id": c["case_id"], "chain_tokens": c["chain_tokens"], "chain_positions": c["chain_positions"], "prefix_len": pre["prefix_len"], "prefix_sha256": pre["token_ids_sha256"],
                       "arm": job["arm"], "repeat": r, "process": job["process"], "run_id": job["run_id"], "obs_id": obs_id, "written_utc": utc()}
            tmp = control_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(control, f)
            os.replace(tmp, control_path)
            payload = {"model": "q1-spec-off-reference", "prompt": ids, "max_tokens": len(c["chain_tokens"]) + 1, "temperature": 0.0, "top_p": 1.0, "seed": 0, "stream": False, "logprobs": None, "skip_special_tokens": False, "ignore_eos": False}
            t0 = time.time(); err = resp = None
            try:
                resp = post_fn(url, payload)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            seal_path = os.path.join(cases_root, obs_id + ".json")
            sealed = (wait_fn or _wait_for_file)(seal_path, poll_s, seal_timeout_s)
            ok, problems, doc = authenticate_seal(seal_path, {"obs_id": obs_id, "case_id": c["case_id"], "run_id": job["run_id"], "arm": job["arm"], "process": job["process"], "repeat": r, "job_sha256": job_sha}, objects_root) if sealed else (False, ["not sealed within timeout"], None)
            usage = (resp or {}).get("usage") or {}
            rec = {"obs_id": obs_id, "case_id": c["case_id"], "repeat": r, "prompt_tokens_sent": len(ids), "usage": usage, "prompt_token_parity": usage.get("prompt_tokens") == pre["prefix_len"],
                   "http_error": err, "wall_s": round(dt, 3), "sealed": bool(sealed), "seal_authenticated": ok, "seal_problems": problems, "raw_objects_authenticated": (doc or {}).get("_raw_objects"),
                   "o2_argmax": ((doc or {}).get("o2") or {}).get("argmax_smallest_id"), "o0_digest": ((doc or {}).get("o0") or {}).get("logical_digest"), "utc": utc()}
            receipts.append(rec)
            with open(os.path.join(out_dir, f"driver.{obs_id}.json"), "x") as f:
                json.dump({"receipt": rec, "response": resp}, f, indent=1)
            if err or not sealed or not ok or not rec["prompt_token_parity"]:
                stop_reason = f"{obs_id}: " + ("http error" if err else "unsealed" if not sealed else "seal not authenticated/valid" if not ok else "prompt parity failed")
                break
        if stop_reason:
            break
    expected = int(job["expected_requests"])
    verdict = {"schema": SCHEMA, "started_utc": started, "ended_utc": utc(), "run_id": job["run_id"], "arm": job["arm"], "process": job["process"], "job_sha256": job_sha, "expected_requests": expected,
               "requests": len(receipts), "authenticated_valid": sum(1 for x in receipts if x["seal_authenticated"]), "stop_reason": stop_reason,
               "complete": stop_reason is None and len(receipts) == expected and all(x["seal_authenticated"] and x["prompt_token_parity"] for x in receipts),
               "repeat_o0_digests_identical": len({x["o0_digest"] for x in receipts}) == 1 if receipts and all(x["o0_digest"] for x in receipts) else None,
               "repeat_o2_argmax_identical": len({x["o2_argmax"] for x in receipts}) == 1 if receipts and all(x["o2_argmax"] is not None for x in receipts) else None,
               "note": "native instrumentation/calibration receipts only; no candidate, no qualification"}
    with open(os.path.join(out_dir, f"DRIVER-VERDICT.{job['arm']}.p{job['process']}.json"), "x") as f:
        json.dump(verdict, f, indent=1)
    return verdict, receipts


def _wait_for_file(path, poll_s, timeout_s):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if os.path.exists(path):
            return True
        time.sleep(poll_s)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True); ap.add_argument("--prefix-plan-dir", default=os.path.join(CAMPAIGN, "prefix-plan"))
    ap.add_argument("--out-dir", required=True); ap.add_argument("--control-path", required=True); ap.add_argument("--hooks-out-dir", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:9950/v1/completions")
    a = ap.parse_args()
    job = json.load(open(a.job)); job_sha = sha256_file(a.job)
    v, _ = run(job, job_sha, a.prefix_plan_dir, a.out_dir, a.control_path, a.url, a.hooks_out_dir)
    print(json.dumps(v, indent=1))
    return 0 if v["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())

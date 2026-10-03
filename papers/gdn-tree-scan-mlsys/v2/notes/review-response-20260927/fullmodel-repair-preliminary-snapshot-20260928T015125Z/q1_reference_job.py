#!/usr/bin/env python3
"""Hash-bound hooks JOB for the spec-off reference engine (CPU-only builder).

The instrumented engine refuses to boot without this file (config v2 requires it and checks its sha); the hooks module (v2) asserts
run/arm identity, expected packed flag, archival policy, fixture identity and the case list at boot.  The native smoke
(Q1-NATIVE-SMOKE-DESIGN.json) is ONE case (`calibration-short_available__c0__root-only`), R=2, one process, primary arm.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); CAMPAIGN = os.path.dirname(HERE)
SCHEMA = "lumo.q1.fullmodel.reference-hooks-job.v2"
SMOKE_CASE = "calibration-short_available__c0__root-only"
TERMINAL_TOKEN = 248046   # <|im_end|>


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


FA2_SOURCE_INDEX = os.path.join(CAMPAIGN, "identity", "native_source", "fa2_source_index.json")


def fa2_sources():
    """Pinned FA2 source/binary identities the hooks must find at runtime (extracted byte copies of the pinned image; see the index)."""
    idx = json.load(open(FA2_SOURCE_INDEX)); f = idx["files"]
    return {"index_sha256": sha256_file(FA2_SOURCE_INDEX), "flash_attn_interface_stock": f["flash_attn_interface_stock"]["sha256"], "flash_attn_interface_patched": idx["flash_attn_interface_patched_sha256"],
            "fa_utils": f["fa_utils"]["sha256"], "flash_attn_backend": f["flash_attn_backend"]["sha256"], "attention_layer": f["attention_layer"]["sha256"],
            "fa2_patcher": idx["fa2_patcher"]["sha256"], "fork_binary": idx["fork_binary"]["sha256"], "selector": idx["api_facts"]["selector"]}


def build(fixtures_path, run_id, arm, process, repeats, case_ids, out_dir_in_container, control_path_in_container, save_kv_bytes, smoke):
    fx = json.load(open(fixtures_path))
    cases = {c["case_id"]: c for c in fx["cases"]}
    for cid in case_ids:
        if cid not in cases:
            raise SystemExit(f"case {cid} not in the frozen token fixtures")
        if cases[cid]["held_out"]:
            raise SystemExit(f"case {cid} is held out")
    if smoke and (case_ids != [SMOKE_CASE] or repeats != 2 or process != "A" or arm != "aligned_nonpacked"):
        raise SystemExit("native smoke scope is exactly one root-only calibration case, R=2, process A, primary arm")
    prefixes = {p["prefix_id"]: p for p in fx["prefixes"]}
    job = {"schema": SCHEMA, "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "run_id": run_id, "arm": arm, "process": process, "repeats": int(repeats),
           "expect_packed_flag": "0" if arm == "aligned_nonpacked" else "1", "smoke": bool(smoke),
           "fixtures": {"path": os.path.relpath(fixtures_path, CAMPAIGN), "sha256": sha256_file(fixtures_path), "canonical": fx["canonical_sha256_excluding_timestamp"]},
           "cases": [{k: cases[c][k] for k in ("case_id", "prefix_id", "block", "path_id", "accepted_len", "chain_tokens", "chain_positions", "z_position", "chain_sha256")} for c in case_ids],
           "prefixes": {cases[c]["prefix_id"]: {k: prefixes[cases[c]["prefix_id"]][k] for k in ("prefix_len", "token_ids_sha256", "token_ids_u32le")} for c in case_ids},
           "expected_requests": len(case_ids) * int(repeats),
           "out_dir": out_dir_in_container, "control_path": control_path_in_container, "terminal_token_id": TERMINAL_TOKEN,
           "archive": {"save_state_bytes": True, "save_kv_bytes": bool(save_kv_bytes), "kv_policy": "content-addressed full logical blocks + valid tail; per-case manifest; no silent truncation"},
           "layer_coverage_required": {"gdn": 48, "attention": 16}, "vocab_required": 248320, "declared_geometry": {"attn_kv_heads": 4, "head_dim": 256, "kv_dtype": "torch.bfloat16", "ssm_dtype": "torch.float32", "conv_dtype": "torch.bfloat16"},
           "fa2_expected_sha256": fa2_sources()["fork_binary"], "fa2_sources": fa2_sources(),
           "failure_rule": "any missing/invalid observation seals the case INVALID exactly once and the driver stops the bounded run; no automatic expansion"}
    body = {k: v for k, v in job.items() if k != "built_utc"}
    job["canonical_sha256_excluding_timestamp"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=os.path.join(CAMPAIGN, "fullmodel", "fixtures", "token-fixtures.v1.json"))
    ap.add_argument("--run-id", required=True); ap.add_argument("--arm", default="aligned_nonpacked", choices=["aligned_nonpacked", "native_default_packed"]); ap.add_argument("--process", default="A")
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--case-id", action="append", default=None); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir-in-container", default="/logs/q1_ref"); ap.add_argument("--control-path-in-container", default="/logs/q1_ref_control.json"); ap.add_argument("--save-kv-bytes", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    case_ids = a.case_id or ([SMOKE_CASE] if a.smoke else None)
    if not case_ids:
        raise SystemExit("--case-id or --smoke required")
    job = build(a.fixtures, a.run_id, a.arm, a.process, a.repeats, case_ids, a.out_dir_in_container, a.control_path_in_container, a.save_kv_bytes or a.smoke, a.smoke)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists (jobs are immutable per run)")
    with open(a.out, "x") as f:
        json.dump(job, f, indent=1); f.write("\n")
    print(json.dumps({"out": a.out, "sha256": sha256_file(a.out), "canonical": job["canonical_sha256_excluding_timestamp"], "cases": len(job["cases"]), "expected_requests": job["expected_requests"]}))


if __name__ == "__main__":
    sys.exit(main())

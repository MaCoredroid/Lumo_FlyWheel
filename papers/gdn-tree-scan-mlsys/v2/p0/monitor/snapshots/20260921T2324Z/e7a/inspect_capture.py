#!/usr/bin/env python3
"""Provenance inspector for ONE fresh capture run directory (review 05 item 2): proves the payload is complete,
bound to the single real request, and not consumed by warm-up. CPU only. Exit 0 iff every check passes."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch


def main() -> int:
    run = Path(sys.argv[1])
    out = {"run": str(run), "checks": {}}
    ok = True

    def check(name, cond, detail=None):
        nonlocal ok
        out["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)

    pay_p = run / "logs" / "tree_gdn_capture_payload.pt"
    check("payload_exists", pay_p.exists() and pay_p.stat().st_size > 0, pay_p.stat().st_size if pay_p.exists() else None)
    check("no_warmup_invalid_marker", not (run / "CAPTURE_INVALID_warmup.txt").exists())
    if not pay_p.exists():
        print(json.dumps(out, indent=1)); return 1
    p = torch.load(pay_p, map_location="cpu", weights_only=True)
    out["payload_sha256"] = hashlib.sha256(pay_p.read_bytes()).hexdigest()
    req = json.load(open(run / "capture_request.json")) if (run / "capture_request.json").exists() else None
    check("request_record_exists", req is not None)
    check("schema", p.get("schema") == "fr10.tree_gdn_scan_capture.v1", p.get("schema"))
    need = ["query_spec", "key_spec", "value_tree", "a", "b", "A_log", "dt_bias", "h0", "tree_parent", "n_actual", "n_pad", "output_scale", "serving_out"]
    missing = [k for k in need if k not in p or p[k] is None]
    check("required_keys_present", not missing, missing)
    n = int(p.get("n_actual", 0)); parents = [int(x) for x in p.get("tree_parent", [])]
    check("topology_consistent", len(parents) == n and parents[0] == -1 and all(0 <= parents[i] < i for i in range(1, n)), {"n": n, "parents": parents})
    shp = {k: (tuple(p[k].shape), str(p[k].dtype)) for k in ("query_spec", "key_spec", "value_tree", "a", "b", "A_log", "dt_bias", "h0", "serving_out") if k in p and p[k] is not None}
    out["shapes"] = shp
    check("shapes", shp.get("query_spec", ((0,),))[0][0] >= n and shp.get("h0", ((0,),))[0] == (48, 128, 128), shp)
    fin = all(torch.isfinite(p[k].float()).all().item() for k in ("query_spec", "key_spec", "value_tree", "a", "b", "A_log", "dt_bias", "h0", "serving_out"))
    check("all_operands_finite", fin)
    check("serving_state_is_None_on_stateless_route", p.get("serving_state") is None, type(p.get("serving_state")).__name__)
    out["layer_prefix"] = p.get("layer_prefix"); out["state_index"] = p.get("state_index"); out["batch_index"] = p.get("batch_index")
    # timing/binding: payload mtime must fall inside [request_start, request_end + slack]
    trace = (run / "driver_trace.txt").read_text() if (run / "driver_trace.txt").exists() else ""
    m_s = re.search(r"request_start_utc=(\S+)", trace); m_e = re.search(r"request_end_utc=(\S+)", trace); m_p = re.search(r"payload_mtime_utc=([^\n]+?) size=", trace)
    out["trace"] = {"request_start": m_s.group(1) if m_s else None, "request_end": m_e.group(1) if m_e else None, "payload_mtime": m_p.group(1) if m_p else None}
    import datetime as dt
    try:
        rs = dt.datetime.fromisoformat(m_s.group(1).replace("Z", "+00:00")); re_ = dt.datetime.fromisoformat(m_e.group(1).replace("Z", "+00:00"))
        pm = dt.datetime.fromisoformat(m_p.group(1).strip().replace(" ", "T", 1)) if m_p else None
        if pm is not None and pm.tzinfo is None:
            pm = pm.replace(tzinfo=dt.timezone.utc)
        check("payload_written_during_request_window", pm is not None and rs - dt.timedelta(seconds=1) <= pm <= re_ + dt.timedelta(seconds=60), out["trace"])
    except Exception as exc:
        check("payload_written_during_request_window", False, repr(exc))
    # per-request server logs must have grown by the request (lines_before vs after)
    for f in ("tree_path_lcp.jsonl", "fr10_tree_depth_positions.jsonl"):
        before = int((run / "logs" / f"{f}.lines_before_request").read_text().strip() or 0) if (run / "logs" / f"{f}.lines_before_request").exists() else None
        after = sum(1 for _ in open(run / "logs" / f)) if (run / "logs" / f).exists() else None
        check(f"{f}_grew_after_request", before is not None and after is not None and after > before, {"before": before, "after": after})
    # prompt token count binding: depth-position log base position (num_computed_tokens) vs usage.prompt_tokens
    if req and (run / "logs" / "fr10_tree_depth_positions.jsonl").exists():
        lines = [json.loads(l) for l in open(run / "logs" / "fr10_tree_depth_positions.jsonl") if l.strip()]
        bases = [l.get("base") or l.get("num_computed_tokens") or l.get("base_position") for l in lines]
        out["depth_position_bases"] = bases[:4]
        out["usage"] = req.get("usage")
    out["request"] = {k: req.get(k) for k in ("prefix_id", "prefix_sha256", "prefix_tokens_pool", "response_id", "usage")} if req else None
    if (run / "e7a_capture_shim.json").exists():
        out["shim_report"] = json.load(open(run / "e7a_capture_shim.json"))
        check("shim_applied", len(out["shim_report"].get("edits", [])) == 2)
    else:
        check("shim_applied", False, "report missing")
    out["all_pass"] = ok
    (run / "capture_provenance.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "shapes"}, indent=1, default=str)[:4000])
    print("PROVENANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

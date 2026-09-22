#!/usr/bin/env python3
"""Provenance inspector for ONE fresh capture run directory (review 05 item 2; strengthened after the review-06
negative-control fixture). Proves the payload is complete AND bound to the single real request:
  * layer_prefix == the layer the driver requested (driver_trace.txt / capture_expected.json)
  * exact model geometry: q,k (n,16,128), v (n,48,128), a,b (n,48), A_log,dt_bias (48,), h0 (48,128,128) fp32,
    serving_out (n,48,128); n == n_actual == len(tree_parent) == expected tree size; n_pad == 16; batch_index == 0
  * topology == the served TREE (expected parents recorded by the driver)
  * prefix_sha256 == prompt_sha256 == the pool entry's hash (pool re-read and re-hashed); usage.prompt_tokens ==
    pool token count; response_id present
  * per-request server logs grew after the request; the depth-position log's base position equals prompt_tokens
  * payload mtime inside the request window; not consumed by warm-up; shim applied (2 edits)
CPU only. Exit 0 iff every check passes. Writes capture_provenance.json."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

KH, VH, DK, DV, N_PAD_EXPECTED = 16, 48, 128, 128, 16


def main() -> int:
    run = Path(sys.argv[1])
    out = {"run": str(run), "checks": {}}
    ok = True

    def check(name, cond, detail=None):
        nonlocal ok
        out["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok &= bool(cond)

    exp = json.load(open(run / "capture_expected.json")) if (run / "capture_expected.json").exists() else None
    check("expected_record_exists", exp is not None)
    pay_p = run / "logs" / "tree_gdn_capture_payload.pt"
    check("payload_exists", pay_p.exists() and pay_p.stat().st_size > 0, pay_p.stat().st_size if pay_p.exists() else None)
    check("no_warmup_invalid_marker", not (run / "CAPTURE_INVALID_warmup.txt").exists())
    req = json.load(open(run / "capture_request.json")) if (run / "capture_request.json").exists() else None
    check("request_record_exists", req is not None and bool(req.get("response_id")), (req or {}).get("response_id"))
    if not pay_p.exists() or exp is None or req is None:
        out["all_pass"] = False
        (run / "capture_provenance.json").write_text(json.dumps(out, indent=1, default=str))
        print(json.dumps(out, indent=1, default=str)); print("PROVENANCE: FAIL"); return 1
    p = torch.load(pay_p, map_location="cpu", weights_only=True)
    out["payload_sha256"] = hashlib.sha256(pay_p.read_bytes()).hexdigest()
    check("schema", p.get("schema") == "fr10.tree_gdn_scan_capture.v1", p.get("schema"))
    check("layer_binding", p.get("layer_prefix") == exp.get("layer_prefix"), {"payload": p.get("layer_prefix"), "expected": exp.get("layer_prefix")})
    check("batch_index_zero", p.get("batch_index") == 0, p.get("batch_index"))
    parents = [int(x) for x in p.get("tree_parent", [])]
    n = int(p.get("n_actual", -1))
    check("topology_binding", parents == exp.get("tree_parents_expected") and n == len(parents), {"payload": parents, "expected": exp.get("tree_parents_expected"), "n_actual": n})
    check("n_pad", int(p.get("n_pad", -1)) == N_PAD_EXPECTED, p.get("n_pad"))
    check("state_index_int", isinstance(p.get("state_index"), int), p.get("state_index"))
    expected_shapes = {"query_spec": ((n, KH, DK), torch.bfloat16), "key_spec": ((n, KH, DK), torch.bfloat16), "value_tree": ((n, VH, DV), torch.bfloat16),
                       "a": ((n, VH), torch.bfloat16), "b": ((n, VH), torch.bfloat16), "A_log": ((VH,), torch.float32), "dt_bias": ((VH,), torch.bfloat16),
                       "h0": ((VH, DV, DK), torch.float32), "serving_out": ((n, VH, DV), torch.bfloat16)}
    shp = {}
    for k, (es, edt) in expected_shapes.items():
        t = p.get(k)
        got = (tuple(t.shape), str(t.dtype)) if isinstance(t, torch.Tensor) else None
        shp[k] = {"got": got, "expected": (es, str(edt))}
        check(f"shape_dtype_{k}", isinstance(t, torch.Tensor) and tuple(t.shape) == es and t.dtype == edt, shp[k])
    out["shapes"] = shp
    fin = all(isinstance(p.get(k), torch.Tensor) and torch.isfinite(p[k].float()).all().item() for k in expected_shapes)
    check("all_operands_finite", fin)
    check("serving_state_is_None_on_stateless_route", p.get("serving_state") is None, type(p.get("serving_state")).__name__)
    check("output_scale", abs(float(p.get("output_scale", 0)) - DK ** -0.5) < 1e-12, p.get("output_scale"))
    # ---- prefix / prompt binding against the pool
    pool_p = Path(exp.get("pool_path", ""))
    pool_ok = pool_p.exists() and hashlib.sha256(pool_p.read_bytes()).hexdigest() == exp.get("pool_sha256")
    check("pool_hash_chain", pool_ok, {"pool": str(pool_p), "expected_sha": exp.get("pool_sha256")})
    if pool_ok:
        entry = next((e for e in json.load(open(pool_p))["prefixes"] if e["id"] == req.get("prefix_id")), None)
        check("prefix_in_pool", entry is not None, req.get("prefix_id"))
        if entry:
            txt_sha = hashlib.sha256(entry["text"].encode("utf-8")).hexdigest()
            check("prefix_hash_binding", req.get("prefix_sha256") == txt_sha == req.get("prompt_sha256") == entry["prefix_sha256"],
                  {"request_prefix_sha": req.get("prefix_sha256"), "request_prompt_sha": req.get("prompt_sha256"), "pool_text_sha": txt_sha})
            pt = (req.get("usage") or {}).get("prompt_tokens")
            check("prompt_token_count_binding", pt == entry["tokens"], {"usage.prompt_tokens": pt, "pool_tokens": entry["tokens"]})
            check("prefix_id_expected", req.get("prefix_id") == exp.get("prefix_id"), {"request": req.get("prefix_id"), "expected": exp.get("prefix_id")})
    # ---- request window
    trace = (run / "driver_trace.txt").read_text() if (run / "driver_trace.txt").exists() else ""
    m_s = re.search(r"request_start_utc=(\S+)", trace); m_e = re.search(r"request_end_utc=(\S+)", trace); m_p = re.search(r"payload_mtime_utc=([^\n]+?) size=", trace)
    out["trace"] = {"request_start": m_s.group(1) if m_s else None, "request_end": m_e.group(1) if m_e else None, "payload_mtime": m_p.group(1) if m_p else None}
    try:
        rs = dt.datetime.fromisoformat(m_s.group(1).replace("Z", "+00:00")); re_ = dt.datetime.fromisoformat(m_e.group(1).replace("Z", "+00:00"))
        pm = dt.datetime.fromisoformat(m_p.group(1).strip().replace(" ", "T", 1))
        if pm.tzinfo is None:
            pm = pm.replace(tzinfo=dt.timezone.utc)
        check("payload_written_during_request_window", rs - dt.timedelta(seconds=1) <= pm <= re_ + dt.timedelta(seconds=60), out["trace"])
    except Exception as exc:
        check("payload_written_during_request_window", False, repr(exc))
    # ---- per-request server logs
    for f in ("tree_path_lcp.jsonl", "fr10_tree_depth_positions.jsonl"):
        bp = run / "logs" / f"{f}.lines_before_request"
        before = int(bp.read_text().strip() or 0) if bp.exists() else None
        after = sum(1 for _ in open(run / "logs" / f)) if (run / "logs" / f).exists() else None
        check(f"{f}_grew_after_request", before is not None and after is not None and after > before, {"before": before, "after": after})
    pt = (req.get("usage") or {}).get("prompt_tokens")
    dp = run / "logs" / "fr10_tree_depth_positions.jsonl"
    if dp.exists():
        bp = run / "logs" / "fr10_tree_depth_positions.jsonl.lines_before_request"
        before = int(bp.read_text().strip() or 0) if bp.exists() else 0
        lines = [json.loads(l) for l in open(dp) if l.strip()][before:]
        bases = []
        for l in lines[:3]:
            for k, v in l.items():
                if isinstance(v, int) and ("computed" in k or k in ("base", "base_position", "root_position")):
                    bases.append((k, v))
        out["depth_position_bases_after_request"] = bases
        check("depth_position_base_equals_prompt_tokens", any(v == pt for _, v in bases) if bases else False, {"bases": bases, "prompt_tokens": pt})
    lcp = run / "logs" / "tree_path_lcp.jsonl"
    if lcp.exists():
        bp = run / "logs" / "tree_path_lcp.jsonl.lines_before_request"
        before = int(bp.read_text().strip() or 0) if bp.exists() else 0
        first = [json.loads(l) for l in open(lcp) if l.strip()][before:before + 1]
        out["first_lcp_after_request"] = first[0] if first else None
        check("lcp_req_index_zero", bool(first) and first[0].get("req_index") == 0 and first[0].get("node_count") == n - 1, first[0] if first else None)
    if (run / "e7a_capture_shim.json").exists():
        rep = json.load(open(run / "e7a_capture_shim.json")); out["shim_report"] = rep
        check("shim_applied", len(rep.get("edits", [])) == 2 and rep.get("sha256_after") != rep.get("sha256_before"))
    else:
        check("shim_applied", False, "report missing")
    out["request"] = {k: req.get(k) for k in ("prefix_id", "prefix_sha256", "prompt_sha256", "prefix_tokens_pool", "response_id", "usage")}
    out["all_pass"] = ok
    (run / "capture_provenance.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("shapes",)}, indent=1, default=str)[:5000])
    print("PROVENANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

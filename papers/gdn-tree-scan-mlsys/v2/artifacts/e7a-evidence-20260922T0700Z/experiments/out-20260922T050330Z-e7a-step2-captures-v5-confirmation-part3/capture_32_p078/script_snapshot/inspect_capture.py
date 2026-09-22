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
CPU only. Exit 0 iff EVERY check passes (pass is True; None never qualifies). Writes capture_provenance.json.
Token binding is mandatory and exact (EOS/stop-aware) — review 08."""
from __future__ import annotations
import os

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

KH, VH, DK, DV, N_PAD_EXPECTED = 16, 48, 128, 128, 16


def _anc(parents, i):
    out = []
    cur = parents[i]
    while cur >= 0:
        out.append(cur); cur = parents[cur]
    return out


def main() -> int:
    run = Path(sys.argv[1])
    out = {"run": str(run), "inspector_path": str(Path(__file__).resolve()),
           "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "checks": {},
           "position_contract": "HEAD emitter base_contract 'state=num_computed_tokens_cpu-1,mrope=num_computed_tokens_cpu': the captured h0 is the recurrent state after the prompt's last token (position prompt_tokens-1); node 0 (root) is the first generated token at position prompt_tokens; drafts follow. No absolute position is emitted; binding uses tree size, scheduled-token count, per-request spec-trace rows (rid embeds the response id), draft records, accepted-count-implied token counts, API-side emitted ids decoded exactly to the response text, window and hashes."}
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
    si = p.get("state_index")
    check("state_index_nonnegative_int", isinstance(si, int) and not isinstance(si, bool) and si >= 0, si)
    h0t = p.get("h0")
    check("h0_pre_state_nontrivial", isinstance(h0t, torch.Tensor) and torch.isfinite(h0t.float()).all().item() and float(h0t.float().abs().max()) > 0.0,
          {"h0_abs_max": float(h0t.float().abs().max()) if isinstance(h0t, torch.Tensor) else None, "note": "a NULL/zero bank row would indicate an unselected pre-state"})
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
    # ---- per-request server logs. Pre-request baseline per log: the driver's `<log>.lines_before_request` when recorded,
    # else the byte size from the driver's pre-request directory listing (logs_listing_before_request.txt: a log absent
    # from the listing had 0 bytes). No baseline evidence at all -> FAIL (never assumed).
    def _listing_sizes():
        lp = run / "logs_listing_before_request.txt"
        if not lp.exists():
            return None
        sizes = {}
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 9 and parts[0].startswith("-"):
                sizes[parts[-1]] = int(parts[4])
        return sizes
    _sizes = _listing_sizes()
    def _baseline(name):
        """-> (new_rows or None, detail)"""
        f = run / "logs" / name
        if not f.exists():
            return None, {"reason": "log missing"}
        bp = run / "logs" / f"{name}.lines_before_request"
        rows_all = [l for l in open(f) if l.strip()]
        if bp.exists():
            before = int(bp.read_text().strip() or 0)
            return [json.loads(l) for l in rows_all[before:]], {"method": "lines_before_request", "before_lines": before, "after_lines": len(rows_all)}
        if _sizes is not None:
            nbytes = _sizes.get(name, 0)
            data = f.read_bytes()
            head = data[:nbytes]
            if nbytes and not head.endswith(b"\n"):
                return None, {"method": "listing_bytes", "reason": "pre-request byte boundary is not a line boundary", "before_bytes": nbytes}
            before = head.count(b"\n")
            return [json.loads(l) for l in rows_all[before:]], {"method": "listing_bytes", "before_bytes": nbytes, "before_lines": before, "after_lines": len(rows_all)}
        return None, {"reason": "no pre-request baseline recorded (no .lines_before_request, no listing)"}
    _new_cache = {}
    def _new_rows(name):
        if name not in _new_cache:
            _new_cache[name] = _baseline(name)[0]
        return _new_cache[name]
    for f in ("per_req_spec_trace.jsonl", "fr10_mtp_draft_trace.jsonl", "fr10_tree_depth_positions.jsonl"):
        rows, det = _baseline(f)
        _new_cache[f] = rows
        check(f"{f}_grew_after_request", rows is not None and len(rows) > 0, det)
    usage = req.get("usage") or {}
    pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
    # ---- depth-position records emitted for THIS request (real schema: event, tree_n, num_scheduled_tokens, offsets, base_contract)
    dp = run / "logs" / "fr10_tree_depth_positions.jsonl"
    if dp.exists() and _new_rows("fr10_tree_depth_positions.jsonl") is not None:
        recs = _new_rows("fr10_tree_depth_positions.jsonl")
        first = recs[0] if recs else None
        out["first_depth_position_record_after_request"] = first
        check("depth_position_first_step_is_full_tree", bool(first) and first.get("event") == "tree_depth_positions" and first.get("tree_n") == n
              and first.get("num_scheduled_tokens") == [n] and "num_computed_tokens_cpu-1" in str(first.get("base_contract", "")),
              {"tree_n": (first or {}).get("tree_n"), "num_scheduled_tokens": (first or {}).get("num_scheduled_tokens"), "n_actual": n, "base_contract": (first or {}).get("base_contract")})
        check("depth_position_offsets_match_topology", bool(first) and first.get("flat_first_tree") == list(range(n))
              and first.get("depth_first_tree") == [0] + [1 + len([x for x in _anc(parents, i)]) - 1 for i in range(1, n)] if first else False,
              {"depth_first_tree": (first or {}).get("depth_first_tree")})
    else:
        check("depth_position_first_step_is_full_tree", False, "log missing")
    # ---- request binding on the logs THIS configuration actually emits (review 08). At temperature 0 on the served
    # tree route the tree_path_lcp.jsonl writer never fires (0 lines in attempt 3 and in the June boot1a capture), so the
    # per-request binding is built from: (a) per_req_spec_trace.jsonl rows whose rid embeds the API response id (explicit
    # request identity in server logs), one row per spec step with draft == n-1; (b) fr10_mtp_draft_trace.jsonl tree_mtp
    # draft records (shape [1, n-1]) inside the request window; (c) the emitted-token count implied by the accepted
    # counts (1 prefill token + sum(acc+1) per step, truncated only in the final step by max_tokens/EOS);
    # (d) API-side emitted token ids (logprobs tokens as "token_id:<id>", driver v4) decoded EXACTLY to response_text,
    # EOS/stop-aware; (e) first-step draft/accept consistency between server logs and the API-side sequence.
    try:
        rs_ = dt.datetime.fromisoformat(m_s.group(1).replace("Z", "+00:00")).timestamp()
        re__ = dt.datetime.fromisoformat(m_e.group(1).replace("Z", "+00:00")).timestamp()
    except Exception:
        rs_, re__ = None, None
    resp_id = req.get("response_id")
    per = _new_rows("per_req_spec_trace.jsonl")
    if per is None:
        check("spec_trace_rows_for_this_request", False, "per_req_spec_trace.jsonl missing")
        per = []
    else:
        mine = [r for r in per if isinstance(r.get("rid"), str) and resp_id and r["rid"].startswith(resp_id + "-")]
        others = [r.get("rid") for r in per if r not in mine]
        check("spec_trace_rows_for_this_request", len(mine) >= 1 and len(others) == 0, {"n_rows": len(mine), "foreign_rids": others[:3], "rid_example": (mine[0].get("rid") if mine else None)})
        check("spec_trace_every_step_drafts_n_minus_1", bool(mine) and all(r.get("draft") == n - 1 and r.get("proposal_width") == n - 1 for r in mine), {"drafts": sorted({r.get("draft") for r in mine})})
        check("spec_trace_first_step_within_request_window", bool(mine) and rs_ is not None and rs_ - 1 <= float(mine[0].get("ts", 0)) <= re__ + 5, {"ts": (mine[0].get("ts") if mine else None), "window": [rs_, re__]})
        per = mine
    acc = [int(r.get("acc", 0)) for r in per]
    out["spec_steps"] = {"n_steps": len(per), "acc": acc}
    drafts = _new_rows("fr10_mtp_draft_trace.jsonl")
    # ---- chunked prefill (review 11, p085): the served scheduler logs its own token budget
    # ("max_num_scheduled_tokens is set to <C> based on the speculative decoding settings"); a prompt of P tokens is
    # prefilled in ceil(P/C) forwards. The drafter runs after EVERY forward (gpu_model_runner: propose_draft_token_ids
    # after sampling, unconditionally), so each prefill forward leaves one draft record whose drafts are never
    # scheduled (partial-prefill requests have their sampled token discarded via discard_request_mask =
    # num_computed_tokens < num_prompt_tokens; the scheduler keeps them in prefill without spec tokens). Hence the
    # number of draft records BEFORE the first tree step (first gpu_tree_metadata event) must equal ceil(P/C), the
    # verified draft for spec step 1 is the LAST of them (offset = ceil(P/C) - 1, fixed, never searched), and the
    # one-shot payload (NUM_TOKENS filter 10 = tree steps only) must be written inside tree step 1: between that
    # first gpu_tree_metadata event and the first sampler_metadata event that follows it.
    import math as _math
    dl = run / "docker_logs.txt"
    mC = re.search(r"max_num_scheduled_tokens is set to (\d+)", dl.read_text(errors="replace")) if dl.exists() else None
    C = int(mC.group(1)) if mC else None
    check("served_token_budget_logged", C is not None and C > 0, {"max_num_scheduled_tokens": C, "source": "docker_logs.txt (server WARNING vllm.py)"})
    sd = _new_rows("tree_sampler_debug.jsonl")
    tree_meta = [float(e.get("ts", 0)) for e in (sd or []) if e.get("event") == "gpu_tree_metadata"]
    first_tree = tree_meta[0] if tree_meta else None
    samp_after = [float(e.get("ts", 0)) for e in (sd or []) if e.get("event") == "sampler_metadata" and first_tree is not None and float(e.get("ts", 0)) > first_tree]
    expected_prefill = (_math.ceil(pt / C) if (isinstance(pt, int) and C) else None)
    n_prefill = (len([r for r in (drafts or []) if float(r.get("ts", 0)) < first_tree]) if (first_tree is not None and drafts) else None)
    check("prefill_forward_count_matches_prompt_over_token_budget", expected_prefill is not None and n_prefill == expected_prefill and n_prefill >= 1,
          {"prompt_tokens": pt, "token_budget": C, "expected_prefill_forwards": expected_prefill, "draft_records_before_first_tree_step": n_prefill, "first_tree_step_ts": first_tree})
    leading = (n_prefill - 1) if n_prefill else 0
    try:
        pm_ts = dt.datetime.fromisoformat(re.search(r"payload_mtime_utc=(\S+ \S+)", trace).group(1) + "+00:00").timestamp()
    except Exception:
        pm_ts = None
    check("payload_written_inside_first_tree_step", first_tree is not None and bool(samp_after) and pm_ts is not None and first_tree - 0.05 <= pm_ts <= samp_after[0] + 0.1,
          {"first_tree_step_ts": first_tree, "first_sampler_after_tree_ts": samp_after[0] if samp_after else None, "payload_mtime_ts": pm_ts})
    out["prefill_binding"] = {"token_budget": C, "prefill_forwards": n_prefill, "leading_unused_draft_records": leading}
    if drafts is None:
        check("draft_trace_rows_for_this_request", False, "fr10_mtp_draft_trace.jsonl missing")
        drafts = []
    else:
        # Draft record k is the tree verified in spec step k+1 (established by exact path alignment on attempts 3 and 4);
        # the drafter also runs at the end of the final step(s), leaving 1–2 trailing unused records (async scheduling
        # can overrun the finish by one step: 1 trailing in attempt 3, 2 in attempt 4). Token-level alignment is
        # checked separately below (per_step_accepted_paths_match_api_sequence).
        trailing = len(drafts) - leading - len(per)
        check("draft_trace_rows_for_this_request", len(drafts) >= 1 and all(r.get("mode") == "tree_mtp" and r.get("shape") == [1, n - 1] for r in drafts)
              and 1 <= trailing <= 2, {"n_draft_records": len(drafts), "n_steps": len(per), "leading_prefill_records": leading, "trailing_unused_records": trailing, "modes": sorted({str(r.get("mode")) for r in drafts})})
        check("draft_trace_first_record_within_request_window", bool(drafts) and rs_ is not None and rs_ - 1 <= float(drafts[0].get("ts", 0)) <= re__ + 5, {"ts": (drafts[0].get("ts") if drafts else None)})
    out["first_draft_record_after_request"] = drafts[0] if drafts else None
    out["first_verified_draft_record"] = drafts[leading] if len(drafts) > leading else None
    # (c) emitted-token count implied by server-side accepted counts
    if per:
        full = 1 + sum(a + 1 for a in acc)              # prefill token + every step's accepted + bonus
        lo = 1 + sum(a + 1 for a in acc[:-1]) + 1       # final step contributes at least one token (truncation by max_tokens/EOS)
        check("completion_tokens_consistent_with_accepted_counts", isinstance(ct, int) and lo <= ct <= full, {"usage.completion_tokens": ct, "implied_min": lo, "implied_max": full, "acc": acc})
    else:
        check("completion_tokens_consistent_with_accepted_counts", False, "no spec steps")
    # (d) API-side emitted ids, EOS/stop-aware EXACT decode == response_text (mandatory; tokenizer must load)
    resp = req.get("response_text")
    tok_path = exp.get("tokenizer_path", "/models/qwen3.6-27b-fp8")
    tok = None
    try:
        import transformers  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        tj = Path(tok_path) / "tokenizer.json"
        out["tokenizer_identity"] = {"path": tok_path, "class": type(tok).__name__, "transformers_version": getattr(transformers, "__version__", None),
                                     "tokenizer_json_sha256": hashlib.sha256(tj.read_bytes()).hexdigest() if tj.exists() else None}
        check("tokenizer_loaded", True, out["tokenizer_identity"])
    except Exception as exc:
        check("tokenizer_loaded", False, f"tokenizer unavailable -> token binding FAILS (never unverified): {type(exc).__name__}: {exc}"[:300])
    eos_ids = set()
    gc_p = Path(tok_path) / "generation_config.json"
    if gc_p.exists():
        try:
            g = json.load(open(gc_p)).get("eos_token_id")
            eos_ids |= set(g if isinstance(g, list) else [g]) if g is not None else set()
        except Exception:
            pass
    te = getattr(tok, "eos_token_id", None)
    if isinstance(te, int):
        eos_ids.add(te)
    out["eos_token_ids_used"] = sorted(eos_ids)
    lpt = req.get("response_logprobs_tokens")
    ids = []
    if isinstance(lpt, list) and lpt and all(isinstance(x, str) and x.startswith("token_id:") and x[9:].isdigit() for x in lpt):
        ids = [int(x[9:]) for x in lpt]
    check("api_emitted_token_ids_recorded", len(ids) > 0 and isinstance(ct, int) and len(ids) == ct, {"n_ids": len(ids), "usage.completion_tokens": ct, "note": "driver v4 records logprobs.tokens as token_id:<id>"})
    max_tokens = (req.get("request") or {}).get("max_tokens")
    fr = req.get("finish_reason")
    if tok is not None and ids:
        if ids[-1] in eos_ids:
            finish, body = "stop", ids[:-1]
        else:
            finish, body = "length", ids
        no_inner_eos = not any(t in eos_ids for t in body)
        try:
            dec = tok.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception as exc:
            dec = None
            check("tokenizer_decode_ok", False, repr(exc)[:200])
        exact = isinstance(resp, str) and len(resp) > 0 and dec is not None and len(body) > 0 and dec == resp
        check("emitted_tokens_decode_exactly_to_response_text", exact and no_inner_eos,
              {"finish_inferred": finish, "n_ids": len(ids), "n_body": len(body), "no_inner_eos": no_inner_eos, "decoded_len": None if dec is None else len(dec),
               "response_len": None if not isinstance(resp, str) else len(resp), "decoded_head": None if dec is None else dec[:80], "response_head": resp[:80] if isinstance(resp, str) else resp,
               "first_mismatch_index": next((i for i, (x, y) in enumerate(zip(dec or "", resp or "")) if x != y), min(len(dec or ""), len(resp or ""))) if not exact else None})
        check("finish_consistent_with_request_max_tokens", (finish == "stop") or (isinstance(max_tokens, int) and len(ids) == max_tokens),
              {"finish_inferred": finish, "n_ids": len(ids), "request.max_tokens": max_tokens, "note": "length finish requires n_ids == recorded max_tokens; missing max_tokens -> FAIL"})
        check("recorded_finish_reason_matches_inferred", fr == finish, {"recorded": fr, "inferred": finish})
        # (f) per-step accepted-path alignment (temperature 0, greedy tree verification): walking draft record k's tree
        # from the root along the API-side sequence must match EXACTLY acc[k] tokens for every non-final step (a
        # rejected node's token differs from the emitted bonus by construction), each step consuming acc[k]+1 tokens
        # (accepted path + bonus); the final step may be truncated by max_tokens/EOS: it consumes 1..acc+1 tokens and
        # its matched path length equals min(acc, remaining). The sequence must be exhausted exactly at the last step.
        if drafts and acc and ids:
            def _children(pp, node):
                return [i for i in range(1, len(pp)) if pp[i] == node]
            def _consistent_paths(d, seq):
                """All root-anchored tree paths whose node tokens equal the emitted sequence prefix, with a flag telling
                whether each path is TERMINAL for greedy verification: it ends at a leaf, or no child of its last node
                carries the next emitted token (that token is then the bonus). Sibling nodes may carry IDENTICAL draft
                tokens (the drafter can propose the same token at the spine and a leaf; observed p087 step 9: node 6 and
                node 7 both 198), and the served LCP committer's tie-break decides which is taken — so the accepted
                length is not unique from the token sequence alone; the check is EXISTENCE of a consistent terminal path
                of exactly the server-reported accepted length."""
                out = []
                def rec(node, depth):
                    kids = _children(parents, node)
                    matching = [c for c in kids if c - 1 < len(d) and depth < len(seq) and d[c - 1] == seq[depth]]
                    terminal = (len(kids) == 0) or (not matching)
                    out.append((depth, terminal))
                    for c in matching:
                        rec(c, depth + 1)
                rec(0, 0)
                return out
            ptr, steps_detail, ok_all = 1, [], True   # ids[0] is the prefill-emitted token (no draft verified)
            for k, a in enumerate(acc):
                if leading + k >= len(drafts):
                    ok_all = False; steps_detail.append({"step": k + 1, "error": "no draft record"}); break
                d = (drafts[leading + k].get("draft") or [[]])[0]
                remaining = len(ids) - ptr
                last = (k == len(acc) - 1)
                paths = _consistent_paths(d, ids[ptr:])
                max_depth = max(dp for dp, _ in paths)
                terminal_lens = sorted({dp for dp, t in paths if t})
                if not last:
                    ok = (a in terminal_lens) and remaining >= a + 1
                    consumed = a + 1
                else:
                    # final step may be truncated by max_tokens/EOS: the kept tokens are a prefix of the accepted path (+bonus)
                    ok = (1 <= remaining <= a + 1) and (min(a, remaining) <= max_depth)
                    consumed = remaining
                steps_detail.append({"step": k + 1, "acc": a, "consistent_terminal_path_lens": terminal_lens, "max_consistent_depth": max_depth,
                                     "consumed": consumed, "ok": ok})
                ok_all &= ok
                ptr += consumed
            ok_all &= (ptr == len(ids))
            check("per_step_accepted_paths_match_api_sequence", ok_all, {"n_steps": len(acc), "draft_offset_fixed_from_prefill_count": leading, "exhausted_exactly": ptr == len(ids), "steps": steps_detail})
        else:
            check("per_step_accepted_paths_match_api_sequence", False, {"n_drafts": len(drafts), "n_acc": len(acc), "n_ids": len(ids)})
        # (e) first-step consistency: at temperature 0 the first draft's root child (node 1, parent 0) is accepted iff
        # the second emitted token equals it (acc[0] >= 1); a rejected root child cannot equal the bonus token.
        if drafts and acc and len(ids) >= 2:
            d0 = (drafts[leading].get("draft") or [[None]])[0] if len(drafts) > leading else [None]
            root_child = d0[0] if isinstance(d0, list) and d0 else None
            check("first_step_draft_accept_consistent_with_api_sequence", root_child is not None and ((acc[0] >= 1) == (ids[1] == root_child)),
                  {"first_draft_root_child": root_child, "api_token_1": ids[1], "acc_0": acc[0]})
        else:
            check("first_step_draft_accept_consistent_with_api_sequence", False, {"n_drafts": len(drafts), "n_acc": len(acc), "n_ids": len(ids)})
    else:
        check("emitted_tokens_decode_exactly_to_response_text", False, {"reason": "tokenizer unavailable" if tok is None else "no API-side emitted ids", "n_ids": len(ids)})
        check("finish_consistent_with_request_max_tokens", False, "not evaluated")
        check("recorded_finish_reason_matches_inferred", False, "not evaluated")
        check("first_step_draft_accept_consistent_with_api_sequence", False, "not evaluated")
        check("per_step_accepted_paths_match_api_sequence", False, "not evaluated")
    if (run / "e7a_capture_shim.json").exists():
        rep = json.load(open(run / "e7a_capture_shim.json")); out["shim_report"] = rep
        anchors = {e.get("anchor") for e in rep.get("edits", [])}
        check("shim_applied", {"guard", "serving_state"} <= anchors and rep.get("sha256_after") != rep.get("sha256_before"), sorted(anchors))
        out["shim_edit_anchors"] = sorted(anchors)
    else:
        check("shim_applied", False, "report missing")
    out["request"] = {k: req.get(k) for k in ("prefix_id", "prefix_sha256", "prompt_sha256", "prefix_tokens_pool", "response_id", "usage")}
    # every check is mandatory: all_pass requires pass is True for ALL checks (None/unverified never qualifies)
    out["unverified"] = [k for k, v in out["checks"].items() if v["pass"] is not True and v["pass"] is not False]
    out["failed"] = [k for k, v in out["checks"].items() if v["pass"] is False]
    out["all_pass"] = bool(ok) and all(v["pass"] is True for v in out["checks"].values())
    ok = out["all_pass"]
    out_path = Path(os.environ.get("INSPECT_OUT") or (run / "capture_provenance.json"))
    out_path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("shapes",)}, indent=1, default=str)[:5000])
    print("PROVENANCE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

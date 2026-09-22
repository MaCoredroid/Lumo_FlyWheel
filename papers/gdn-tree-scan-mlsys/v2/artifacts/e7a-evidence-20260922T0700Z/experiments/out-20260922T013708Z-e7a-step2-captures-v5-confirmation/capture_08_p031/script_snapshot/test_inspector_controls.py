#!/usr/bin/env python3
"""Versioned CPU controls for inspect_capture.py (review 08): positive + negative fixtures in the DRIVER's real record
shape (capture_request.json carries request.max_tokens; driver v4 also finish_reason/logprobs tokens).
A fake `transformers.AutoTokenizer` decodes by a fixed id->string map so EOS handling can be exercised; EOS ids come from
the checkpoint's generation_config.json exactly as the inspector reads them. Zero model inference.
Evidence shape (what the served configuration emits): per_req_spec_trace.jsonl rows (rid = response_id + "-0-<hash>",
draft/proposal_width = n-1, acc per step), fr10_mtp_draft_trace.jsonl tree_mtp records ([1, n-1]), depth-position records,
API-side logprobs tokens "token_id:<id>" (driver v4), finish_reason, request.max_tokens. tree_path_lcp.jsonl is NOT emitted.
Cases (expected verdict):
  positive_length / positive_eos_stop / positive_two_trailing_drafts    -> PASS
  neg_tokenizer_unavailable, neg_empty_response, neg_truncated_response, neg_extended_response, neg_inner_eos,
  neg_length_not_max, neg_missing_max_tokens, neg_state_index_negative, neg_h0_all_zero, neg_finish_reason_mismatch,
  neg_api_ids_missing (no token_id strings), neg_api_ids_count (31 vs 32), neg_foreign_rid (spec rows of another request),
  neg_acc_count_inconsistent (accepted counts cannot produce completion_tokens), neg_first_step_draft_inconsistent
  (acc[0]>=1 but api token 1 != first draft root child), neg_draft_mode_not_tree, neg_step3_path_mismatch -> FAIL
Usage: test_inspector_controls.py <out_json>
"""
from pathlib import Path
import os, contextlib, datetime as dt, hashlib, importlib.util, io, json, sys, tempfile, types, torch
torch.set_num_threads(1)
here = Path(__file__).resolve().parent; ins = here / "inspect_capture.py"
spec = importlib.util.spec_from_file_location("ctl_inspector", ins); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
EOS = 248046
VOCAB = {i: f"w{i} " for i in range(1, 40)}; VOCAB[EOS] = "<|im_end|>"
def decode(ids): return "".join(VOCAB.get(i, f"<{i}>") for i in ids)
parents = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]; n = len(parents)
OUT = Path(sys.argv[1]).resolve()  # captured BEFORE the loop rebinds sys.argv for the inspector
res = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "inspector_sha256": hashlib.sha256(ins.read_bytes()).hexdigest(), "cases": {}}
body32 = list(range(1, 33)); body31 = list(range(1, 32))
def ids_to_lp(ids): return [f"token_id:{i}" for i in ids]
cases = {
    "positive_length":            dict(ids=body32, text=decode(body32), avail=True, expect=True, finish="length"),
    "positive_eos_stop":          dict(ids=body31 + [EOS], text=decode(body31), avail=True, expect=True, finish="stop"),
    "neg_tokenizer_unavailable":  dict(ids=body32, text=decode(body32), avail=False, expect=False, finish="length"),
    "neg_empty_response":         dict(ids=body32, text="", avail=True, expect=False, finish="length"),
    "neg_truncated_response":     dict(ids=body32, text=decode(body32)[:-3], avail=True, expect=False, finish="length"),
    "neg_extended_response":      dict(ids=body32, text=decode(body32) + "x", avail=True, expect=False, finish="length"),
    "neg_inner_eos":              dict(ids=body31[:5] + [EOS] + body31[5:], text=decode(body31[:5] + [EOS] + body31[5:]), avail=True, expect=False, finish="length"),
    "neg_length_not_max":         dict(ids=body31, text=decode(body31), avail=True, expect=False, finish="length", completion_tokens=31, acc=[1, 2, 1, 0, 2, 1, 4, 2, 2, 0, 4, 1]),
    "neg_missing_max_tokens":     dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", no_max_tokens=True),
    "neg_state_index_negative":   dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", state_index=-1),
    "neg_h0_all_zero":            dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", h0_zero=True),
    "neg_finish_reason_mismatch": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="stop"),
    "neg_api_ids_missing":        dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", lp_strings=True),
    "neg_api_ids_count":          dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", lp_ids=body31),
    "neg_foreign_rid":            dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", foreign_rid=True),
    "neg_acc_count_inconsistent": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", acc=[0, 0, 0]),
    "neg_first_step_draft_inconsistent": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", first_draft_root=999),
    "neg_draft_mode_not_tree":    dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", draft_mode="naive_mtp"),
    "neg_step3_path_mismatch":    dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", step3_mismatch=True),
    "positive_chunked_prefill_4096": dict(ids=body32, text=decode(body32), avail=True, expect=True, finish="length", prompt_tokens=4096),
    "positive_chunked_prefill_2049_three_forwards": dict(ids=body32, text=decode(body32), avail=True, expect=True, finish="length", prompt_tokens=4097),
    "neg_budget_line_missing":    dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", no_budget_line=True),
    "neg_prefill_records_fewer_than_prompt_over_budget": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", prompt_tokens=4096, prefill_records=1),
    "neg_prefill_records_more_than_prompt_over_budget": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", prompt_tokens=256, prefill_records=2),
    "neg_payload_written_before_first_tree_step": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", payload_ts_offset=-5.0),
    "neg_payload_written_after_step1_sampler": dict(ids=body32, text=decode(body32), avail=True, expect=False, finish="length", payload_ts_offset=+2.0),
    "positive_two_trailing_drafts": dict(ids=body32, text=decode(body32), avail=True, expect=True, finish="length", trailing2=True),
}
# real attempt-3 accepted counts: 12 steps, 1 + sum(acc+1) = 34 >= 32 >= 1 + 31 + 1 = 33?  -> use counts consistent with 32:
# acc below: 1 + sum(acc+1) = 1 + (11 steps: 1,2,1,0,2,1,4,2,2,0,4 -> 18+11=29) + (last 2+1=3) = 33 >= 32 >= 1+29+1 = 31  OK
DEFAULT_ACC = [1, 2, 1, 0, 2, 1, 4, 2, 2, 0, 4, 2]
all_ok = True
for name, c in cases.items():
    with tempfile.TemporaryDirectory(prefix="e7a-ctl-") as td:
        run = Path(td); (run / "logs").mkdir()
        txt = "fixture prefix text"; hs = hashlib.sha256(txt.encode()).hexdigest()
        pool = run / "pool.json"; pool.write_text(json.dumps({"prefixes": [{"id": "fx", "text": txt, "prefix_sha256": hs, "tokens": c.get("prompt_tokens", 256)}]}))
        exp = {"prefix_id": "fx", "layer_prefix": "language_model.model.layers.62.linear_attn", "tree_parents_expected": parents, "pool_path": str(pool),
               "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest()}
        (run / "capture_expected.json").write_text(json.dumps(exp))
        p = {"schema": "fr10.tree_gdn_scan_capture.v1", "layer_prefix": exp["layer_prefix"], "batch_index": 0, "tree_parent": parents, "n_actual": n, "n_pad": 16,
             "state_index": c.get("state_index", 0), "output_scale": 128 ** -0.5, "serving_state": None}
        for k, shape, dtype in [("query_spec", (n, 16, 128), torch.bfloat16), ("key_spec", (n, 16, 128), torch.bfloat16), ("value_tree", (n, 48, 128), torch.bfloat16),
                                ("a", (n, 48), torch.bfloat16), ("b", (n, 48), torch.bfloat16), ("A_log", (48,), torch.float32), ("dt_bias", (48,), torch.bfloat16),
                                ("h0", (48, 128, 128), torch.float32), ("serving_out", (n, 48, 128), torch.bfloat16)]:
            p[k] = torch.zeros(shape, dtype=dtype) if (k == "h0" and c.get("h0_zero")) else torch.ones(shape, dtype=dtype)
        pay = run / "logs/tree_gdn_capture_payload.pt"; torch.save(p, pay)
        now = dt.datetime.now(dt.timezone.utc); stamp = now.isoformat()
        # driver_trace is written after the timeline is known (see below); placeholder here
        (run / "driver_trace.txt").write_text(f"request_start_utc={stamp}\nrequest_end_utc={stamp}\npayload_mtime_utc={stamp} size={pay.stat().st_size}\n")
        depth = [len(mod._anc(parents, i)) for i in range(n)]
        dp = {"event": "tree_depth_positions", "tree_n": n, "num_scheduled_tokens": [n], "base_contract": "state=num_computed_tokens_cpu-1,mrope=num_computed_tokens_cpu",
              "flat_first_tree": list(range(n)), "depth_first_tree": depth}
        ids = c["ids"]
        acc = c.get("acc", DEFAULT_ACC)
        rid = ("cmpl-OTHER-0-deadbeef" if c.get("foreign_rid") else "fixture-response-0-b1438620")
        per_rows = [{"ts": now.timestamp() + 4.5 + 0.32 * k, "rid": rid, "draft": n - 1, "proposal_width": n - 1, "verify_width": n - 1, "acc": a, "inv": 0} for k, a in enumerate(acc)]
        # drafts consistent with acc: spine slots (nodes 1,2,4,6,8 -> d[0],d[1],d[3],d[5],d[7]) carry the accepted path
        # tokens, every other slot a dummy id that never equals an emitted token; the bonus follows the path.
        spine_slots = [0, 1, 3, 5, 7]
        prompt_tokens = c.get("prompt_tokens", 256); budget = c.get("token_budget", 2048)
        import math as _m
        n_prefill = c.get("prefill_records", _m.ceil(prompt_tokens / budget))   # draft records before the first tree step
        t0 = now.timestamp()
        first_tree_ts = t0 + 0.5 * n_prefill + 0.3; sampler1_ts = first_tree_ts + 3.8       # step-1 window (matches real runs)
        (run / "docker_logs.txt").write_text("" if c.get("no_budget_line") else f"(APIServer pid=1) WARNING x [vllm.py:1406] max_num_scheduled_tokens is set to {budget} based on the speculative decoding settings. This may lead to suboptimal performance.\n")
        sd = [{"event": "gpu_tree_metadata", "ts": first_tree_ts}, {"event": "sampler_metadata", "ts": sampler1_ts}]
        (run / "logs" / "tree_sampler_debug.jsonl").write_text("".join(json.dumps(e) + "\n" for e in sd)); (run / "logs" / "tree_sampler_debug.jsonl.lines_before_request").write_text("0")
        pm_ts = c.get("payload_ts_offset", 0.0) + sampler1_ts - 0.01
        draft_rows, ptr = [], 1
        for j in range(n_prefill):   # prefill-forward draft records (never scheduled); the LAST one is verified in step 1
            draft_rows.append({"event": "mtp_draft", "idx": j, "ts": t0 + 0.5 * j + 0.1, "mode": "tree_mtp", "shape": [1, n - 1], "draft": [[800000 + j * 10 + q for q in range(n - 1)]]})
        for k in range(len(acc) + 2 if c.get("trailing2") else len(acc) + 1):
            d = [900000 + 10 * k + j for j in range(n - 1)]
            if k < len(acc):
                a = acc[k]
                for t in range(min(a, len(spine_slots))):
                    if ptr + t < len(ids): d[spine_slots[t]] = ids[ptr + t]
                if k == 0 and "first_draft_root" in c: d[0] = c["first_draft_root"]
                if k == 2 and c.get("step3_mismatch"): d[0] = 900999
                ptr += a + 1
            if k == 0 and draft_rows:   # the verified step-1 draft IS the last prefill record
                draft_rows[-1]["draft"] = [d]; draft_rows[-1]["mode"] = c.get("draft_mode", "tree_mtp"); continue
            draft_rows.append({"event": "mtp_draft", "idx": len(draft_rows), "ts": first_tree_ts + 6.5 + 0.32 * k, "mode": c.get("draft_mode", "tree_mtp"), "shape": [1, n - 1], "draft": [d]})
        end = dt.datetime.fromtimestamp(t0 + 30, dt.timezone.utc); pm = dt.datetime.fromtimestamp(pm_ts, dt.timezone.utc)
        (run / "driver_trace.txt").write_text(f"request_start_utc={stamp}\nrequest_end_utc={end.isoformat()}\npayload_mtime_utc={pm.strftime('%Y-%m-%d %H:%M:%S.%f')} +0000 size={pay.stat().st_size}\n")
        os.utime(run / "logs" / "tree_gdn_capture_payload.pt", (pm_ts, pm_ts))
        (run / "logs" / "fr10_tree_depth_positions.jsonl").write_text(json.dumps(dp) + "\n"); (run / "logs" / "fr10_tree_depth_positions.jsonl.lines_before_request").write_text("0")
        (run / "logs" / "per_req_spec_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_rows)); (run / "logs" / "per_req_spec_trace.jsonl.lines_before_request").write_text("0")
        (run / "logs" / "fr10_mtp_draft_trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in draft_rows)); (run / "logs" / "fr10_mtp_draft_trace.jsonl.lines_before_request").write_text("0")
        (run / "e7a_capture_shim.json").write_text(json.dumps({"edits": [{"anchor": "guard"}, {"anchor": "serving_state"}], "sha256_before": "a", "sha256_after": "b"}))
        req = {"prefix_id": "fx", "prefix_sha256": hs, "prompt_sha256": hs, "response_id": "fixture-response", "prefix_tokens_pool": c.get("prompt_tokens", 256),
               "usage": {"prompt_tokens": c.get("prompt_tokens", 256), "completion_tokens": c.get("completion_tokens", len(ids))}, "response_text": c["text"],
               "request": {} if c.get("no_max_tokens") else {"max_tokens": 32, "temperature": 0.0, "logprobs": 5, "return_tokens_as_token_ids": True},
               "finish_reason": c["finish"],
               "response_logprobs_tokens": ([VOCAB.get(i, "?") for i in ids] if c.get("lp_strings") else ids_to_lp(c.get("lp_ids", ids)))}
        (run / "capture_request.json").write_text(json.dumps(req))
        class Tok:
            eos_token_id = EOS
            @staticmethod
            def from_pretrained(*a, **k):
                if not c["avail"]: raise OSError("controls: tokenizer forced unavailable")
                return Tok()
            def decode(self, ids_, **k): return decode(list(ids_))
        fake = types.ModuleType("transformers"); fake.AutoTokenizer = Tok; fake.__version__ = "fake-controls"; sys.modules["transformers"] = fake
        sys.argv = [str(ins), str(run)]
        with contextlib.redirect_stdout(io.StringIO()): rc = mod.main()
        out = json.loads((run / "capture_provenance.json").read_text())
        ok = (out["all_pass"] is c["expect"]) and (rc == (0 if c["expect"] else 1))
        all_ok &= ok
        res["cases"][name] = {"expected_all_pass": c["expect"], "all_pass": out["all_pass"], "rc": rc, "failed": out.get("failed"), "unverified": out.get("unverified"),
                              "token_binding": out["checks"].get("emitted_tokens_decode_exactly_to_response_text", {}).get("detail"), "control_ok": ok}
        print(("OK   " if ok else "BAD  ") + f"{name}: all_pass={out['all_pass']} rc={rc} failed={out.get('failed')}")
res["all_controls_ok"] = all_ok
OUT.write_text(json.dumps(res, indent=2)); print("ALL CONTROLS OK" if all_ok else "CONTROLS FAILED"); sys.exit(0 if all_ok else 1)

#!/usr/bin/env python3
"""pendiag: record-only diagnostic of how sampling penalties reach LumoTree's tree rows (applied in the container
after every production patcher, via the q1v3 launch chain; no production source is modified).
Appends a wrapper around RejectionSampler.apply_logits_processors that, for the first PD_MAX calls with a
multiple-of-31 row count, records per row: the token ids whose logit the processors changed and the change, the
top-256 pre-processing logits, and the request's output-token tail and spec (draft) token ids. Nothing is altered.
If /q1v3/HOOKS_ON exists the passive tree trace hooks are installed afterwards."""
import glob, json, os, subprocess, sys

SITE = "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py"
WRAP = r'''

# ---- v2exp pendiag (record-only) ------------------------------------------------------------------
import json as _pd_json
_PD_OUT = "/q1run/CAND/pen_trace.jsonl"
_PD_MAX = 80
_pd_state = {"n": 0}
_pd_orig_alp = RejectionSampler.apply_logits_processors


def _pd_apply_logits_processors(self, logits, sampling_metadata, metadata):
    rec = _pd_state["n"] < _PD_MAX and logits.dim() == 2 and logits.shape[0] > 0 and logits.shape[0] % 31 == 0
    pre = logits.detach().float().clone() if rec else None
    out = _pd_orig_alp(self, logits, sampling_metadata, metadata)
    if rec:
        try:
            post = out.detach().float()
            d = post - pre
            rows = []
            for i in range(pre.shape[0]):
                nz = (d[i] != 0).nonzero().flatten()
                tv, ti = pre[i].topk(256)
                rows.append({"pen_ids": [int(x) for x in nz.tolist()],
                             "pen_vals": sorted({round(float(x), 4) for x in d[i][nz].tolist()}),
                             "top_ids": [int(x) for x in ti.tolist()], "top_vals": [round(float(x), 5) for x in tv.tolist()]})
            outs = sampling_metadata.output_token_ids
            spec = sampling_metadata.spec_token_ids
            pen = {k: [float(x) for x in getattr(sampling_metadata, k).tolist()] for k in
                   ("presence_penalties", "frequency_penalties", "repetition_penalties") if getattr(sampling_metadata, k, None) is not None}
            with open(_PD_OUT, "a") as f:
                f.write(_pd_json.dumps({"n": _pd_state["n"], "nrows": len(rows), "no_penalties": bool(sampling_metadata.no_penalties),
                                        "penalties": pen, "num_draft_tokens": list(metadata.num_draft_tokens),
                                        "out_len": [len(o) for o in outs], "out_tail": [list(o[-96:]) for o in outs],
                                        "spec": [list(s) for s in spec] if spec is not None else None, "rows": rows}) + "\n")
        except Exception as e:  # never disturb serving
            with open(_PD_OUT, "a") as f:
                f.write(_pd_json.dumps({"n": _pd_state["n"], "error": repr(e)}) + "\n")
        _pd_state["n"] += 1
    return out


RejectionSampler.apply_logits_processors = _pd_apply_logits_processors
# ---- end v2exp pendiag -----------------------------------------------------------------------------
'''
rec = {"schema": "v2exp.pendiag.v1", "applied": False, "file": SITE}
s = open(SITE).read()
if "class RejectionSampler" not in s or "v2exp pendiag" in s:
    rec["error"] = "anchor missing or already patched"
else:
    open(SITE, "w").write(s + WRAP)
    rec["applied"] = True
os.makedirs("/q1run/CAND", exist_ok=True)
json.dump(rec, open("/q1run/CAND/pendiag_receipt.json", "w"), indent=1)
print("[pendiag]", json.dumps(rec), flush=True)
if not rec["applied"]:
    sys.exit(3)
if os.path.exists("/q1v3/HOOKS_ON"):
    sys.exit(subprocess.call([sys.executable, "/q1v3/patch_runner_q1v3.py"] + sys.argv[1:]))
json.dump({"applied": True, "problems": [], "mode": "pendiag-only"}, open("/q1run/CAND/patch_receipt.json", "w"))

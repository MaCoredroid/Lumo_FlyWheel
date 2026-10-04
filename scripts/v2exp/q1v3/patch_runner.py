#!/usr/bin/env python3
"""Install the q1v3 hook calls into the vLLM sources inside a container (run AFTER all production patchers).

  --mode native : stock gpu_model_runner.py (vLLM 0.19.2rc1.dev134, sha 904d7d35...)      -> 4 edits
  --mode cand   : production-patched (fixed32) gpu_model_runner.py + rejection_sampler.py  -> 6 + 2 edits

Every anchor must occur exactly once at line start; a marker already present refuses (no double patch); all
edited texts are compiled before any file is written. Observed pre-patch sha256s are recorded and compared with
the reference copies Codex captured (identity/*); a sha difference is recorded, not fatal -- anchor uniqueness
is the gate. No edit changes a production statement: each inserts one hook call next to its anchor.
CPU-testable: --runner/--rejection point at copies, --dry-run patches in memory only.
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys

MARK = "# Q1V3_HOOKS"
SITE = "/usr/local/lib/python3.12/dist-packages/vllm"
RUNNER = f"{SITE}/v1/worker/gpu_model_runner.py"
REJECTION = f"{SITE}/v1/sample/rejection_sampler.py"
REF_SHA = {"native_runner": "904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0",
           "cand_runner": "3d9df101ebb1b2d5", "cand_rejection": None}   # cand runner: prefix of the Cqc10 probe copy

IMPORT = ("from vllm.forward_context import (\n", "before", f"import q1v3_hooks as _q1v3  {MARK}\n")
NATIVE_RUNNER = [
    ("import", *IMPORT),
    ("pre_forward", "            model_output = self._model_forward(\n", "before",
     f"            _q1v3.H().on_pre_forward(self, scheduler_output, attn_metadata, input_ids, positions, logits_indices)  {MARK}\n"),
    ("logits", "                sample_hidden_states = hidden_states[logits_indices]\n                logits = self.model.compute_logits(sample_hidden_states)\n", "after",
     f"                _q1v3.H().on_logits(self, logits, logits_indices)  {MARK}\n"),
    ("sampled", "            sampler_output = self._sample(logits, spec_decode_metadata)\n", "after",
     f"            _q1v3.H().on_sampled(self, sampler_output, scheduler_output, grammar_output)  {MARK}\n"),
]
CAND_RUNNER = [
    ("import", *IMPORT),
    ("pre_forward", "            model_output = self._model_forward(\n", "before",
     f"            _q1v3.H().on_pre_forward(self, scheduler_output, attn_metadata, input_ids, positions, logits_indices)  {MARK}\n"),
    ("logits", "                logits = self.model.compute_logits(sample_hidden_states)\n", "after",
     f"                _q1v3.H().on_logits(self, logits, logits_indices)  {MARK}\n"),
    ("sampled", "                sampler_output = self._sample(logits, spec_decode_metadata)\n", "after",
     f"                _q1v3.H().on_sampled(self, sampler_output, scheduler_output)  {MARK}\n"),
    ("drafts", "                # # FR13_REPLAY_DRAFT_REQKEY: remember the row owners for the GPU\n", "before",
     f"                _q1v3.H().on_drafts(self, self._draft_token_ids)  {MARK}\n"),
    ("sealed", "                # FR13_FIXED32_DRAFTER_PROPOSAL_SEALED\n", "before",
     f"                _q1v3.H().on_sealed(self)  {MARK}\n"),
]
CAND_REJECTION = [
    ("import", "# LUMO_TREE_PATH_LCP_MAX\n", "after", f"import q1v3_hooks as _q1v3  {MARK}\n"),
    ("taw", "        _fr13_f32_output = _fr13_fixed32_device_commit_route(\n            _fr13_f32_commit_result,\n", "before",
     f"        _fr13_f32_commit_result = _q1v3.H().on_taw_products(_fr13_f32_commit_result)  {MARK}\n"),
]
# context that must also be unique/present so an anchor cannot silently bind to a different copy
CAND_RUNNER_CONTEXT = [("pp_branch_logits", "                    logits = self.model.compute_logits(sample_hidden_states)\n", 1)]


def _pat(a):
    return re.compile("^" + re.escape(a), re.M)


def patch_text(text, edits, context=()):
    problems, lines = [], {}
    if MARK in text:
        problems.append("marker already present (double patch refused)")
    for name, anchor, where, ins in edits:
        hits = list(_pat(anchor).finditer(text))
        if len(hits) != 1:
            problems.append(f"{name}: anchor occurs {len(hits)} times (need 1)")
        else:
            lines[name] = text[:hits[0].start()].count("\n") + 1
    for name, needle, n in context:
        k = len(_pat(needle).findall(text))
        if k != n:
            problems.append(f"context {name}: {k} != {n}")
    if problems:
        return None, problems, lines
    for name, anchor, where, ins in edits:
        m = _pat(anchor).search(text)
        new = ins + anchor if where == "before" else anchor + ins
        text = text[:m.start()] + new + text[m.end():]
    return text, [], lines


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(mode, runner, rejection, apply, receipt):
    targets = [("runner", runner, NATIVE_RUNNER if mode == "native" else CAND_RUNNER, () if mode == "native" else CAND_RUNNER_CONTEXT)]
    if mode == "cand":
        targets.append(("rejection_sampler", rejection, CAND_REJECTION, ()))
    rec = {"schema": "q1v3.patch-receipt.v1", "mode": mode, "applied": False, "problems": [], "files": {}}
    out = {}
    for key, path, edits, ctx in targets:
        text = open(path, encoding="utf-8").read()
        pre = sha(path)
        ref = REF_SHA["native_runner" if mode == "native" else ("cand_runner" if key == "runner" else "cand_rejection")]
        new, problems, lines = patch_text(text, edits, ctx)
        rec["files"][key] = {"path": path, "pre_sha256": pre, "reference_sha256": ref,
                             "matches_reference_copy": (None if ref is None else pre.startswith(ref)), "anchor_lines": lines}
        rec["problems"] += [f"{key}: {p}" for p in problems]
        if new is not None:
            try:
                compile(new, path, "exec")
            except SyntaxError as e:
                rec["problems"].append(f"{key}: patched text does not compile: {e}")
            out[path] = new
    if apply and not rec["problems"]:
        for path, new in out.items():
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            rec["files"][[k for k, v in rec["files"].items() if v["path"] == path][0]]["post_sha256"] = sha(path)
        rec["applied"] = True
    hooks = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q1v3_hooks.py")
    rec["hooks_sha256"] = sha(hooks) if os.path.exists(hooks) else None
    if receipt:
        os.makedirs(os.path.dirname(os.path.abspath(receipt)), exist_ok=True)
        with open(receipt, "w") as f:
            json.dump(rec, f, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["native", "cand"], required=True)
    ap.add_argument("--runner", default=RUNNER)
    ap.add_argument("--rejection", default=REJECTION)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--receipt")
    a = ap.parse_args()
    if a.apply == a.dry_run:
        ap.error("exactly one of --apply / --dry-run")
    rec = run(a.mode, a.runner, a.rejection, a.apply, a.receipt)
    print(json.dumps({k: rec[k] for k in ("mode", "applied", "problems")}))
    return 0 if not rec["problems"] else 2


if __name__ == "__main__":
    sys.exit(main())

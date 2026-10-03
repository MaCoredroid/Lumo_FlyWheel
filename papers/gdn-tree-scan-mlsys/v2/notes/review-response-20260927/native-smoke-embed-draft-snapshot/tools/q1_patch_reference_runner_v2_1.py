#!/usr/bin/env python3
"""Q1 reference-runner diagnostic patcher **v2.1** (= v2 anchors unchanged; imports q1_reference_hooks_v2_1 -- the embed-path token-source repair). fr10 literal-anchor idiom: every anchor exactly once.

Applies five host-side edits to the pinned stock `vllm/v1/worker/gpu_model_runner.py`
(sha 904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0, vLLM 0.19.2rc1.dev134) so the
spec-off reference engine can capture O0/O1/O2 and teacher-force the fixture chain.  It refuses to
run against any other runner source (byte hash check) and refuses to patch twice.

  A1  import                    after `from vllm.forward_context import (` block -> import q1_reference_hooks
  A2  O0/O1 pre-forward hook    before `            model_output = self._model_forward(`   (:4035)
  A3  O2 logits hook            after  `                logits = self.model.compute_logits(sample_hidden_states)` (:4071)
  A4  grammar guard             before `        if grammar_output is not None:\n            apply_grammar_bitmask(` (:4162)
  A5  teacher-forcing hook      after  `            sampler_output = self._sample(logits, spec_decode_metadata)` (:4167)

No edit touches the model, attention, GDN operators, cache allocation, scheduler or sampler
internals; A5 only calls a hook that writes the forced token in place into the sampler output.
CPU-testable: `--dry-run --source <copy>` patches a copy and reports anchor counts + py_compile.
"""
from __future__ import annotations

import argparse, hashlib, json, os, py_compile, sys, time

EXPECTED_RUNNER_SHA = "904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0"
DEFAULT_TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
MARK = "# Q1_REFERENCE_HOOKS_PATCH v2.1"

A1_ANCHOR = "from vllm.forward_context import (\n"
A1_INSERT_BEFORE = ("import q1_reference_hooks_v2_1 as _q1_ref_hooks  " + MARK + " A1\n")

A2_ANCHOR = "            model_output = self._model_forward(\n"
A2_INSERT_BEFORE = (
    "            _q1_ref_hooks.instance().on_pre_forward(self, scheduler_output, attn_metadata, input_ids, positions, logits_indices)  " + MARK + " A2\n"
)

A3_ANCHOR = ("                sample_hidden_states = hidden_states[logits_indices]\n"
             "                logits = self.model.compute_logits(sample_hidden_states)\n")
A3_INSERT_AFTER = ("                _q1_ref_hooks.instance().on_logits(self, logits, logits_indices)  " + MARK + " A3\n")

A4_ANCHOR = ("        if grammar_output is not None:\n"
             "            apply_grammar_bitmask(\n")
A4_INSERT_BEFORE = ("        _q1_ref_hooks.instance().on_grammar(self, grammar_output)  " + MARK + " A4\n")

A5_ANCHOR = "            sampler_output = self._sample(logits, spec_decode_metadata)\n"
A5_INSERT_AFTER = ("            _q1_ref_hooks.instance().on_sampled(self, sampler_output, scheduler_output, grammar_output)  " + MARK + " A5\n")

ANCHORS = [("A1", A1_ANCHOR), ("A2", A2_ANCHOR), ("A3", A3_ANCHOR), ("A4", A4_ANCHOR), ("A5", A5_ANCHOR)]


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _replace_once(src: str, needle: str, replacement: str, tag: str) -> str:
    n = src.count(needle)
    if n != 1:
        raise RuntimeError(f"Q1 reference patch: anchor {tag} occurs {n} times (need exactly 1)")
    return src.replace(needle, replacement)


def patch_text(src: str) -> tuple[str, dict]:
    if MARK in src:
        raise RuntimeError("Q1 reference patch already applied")
    counts = {tag: src.count(a) for tag, a in ANCHORS}
    out = src
    out = _replace_once(out, A1_ANCHOR, A1_INSERT_BEFORE + A1_ANCHOR, "A1")
    out = _replace_once(out, A2_ANCHOR, A2_INSERT_BEFORE + A2_ANCHOR, "A2")
    out = _replace_once(out, A3_ANCHOR, A3_ANCHOR + A3_INSERT_AFTER, "A3")
    out = _replace_once(out, A4_ANCHOR, A4_INSERT_BEFORE + A4_ANCHOR, "A4")
    out = _replace_once(out, A5_ANCHOR, A5_ANCHOR + A5_INSERT_AFTER, "A5")
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_TARGET)
    ap.add_argument("--apply", action="store_true", help="write the patched file in place (in-container only)")
    ap.add_argument("--dry-run", action="store_true", help="patch a copy under --out and report")
    ap.add_argument("--out", default=None)
    ap.add_argument("--receipt", default=None)
    ap.add_argument("--allow-hash-mismatch", action="store_true", help="tests only; never in a launch")
    a = ap.parse_args()
    src_sha = sha256_file(a.source)
    if src_sha != EXPECTED_RUNNER_SHA and not a.allow_hash_mismatch:
        print(json.dumps({"ok": False, "reason": f"runner sha {src_sha} != expected {EXPECTED_RUNNER_SHA}"}))
        return 3
    src = open(a.source, encoding="utf-8").read()
    try:
        patched, counts = patch_text(src)
    except RuntimeError as e:
        print(json.dumps({"ok": False, "reason": str(e)}))
        return 4
    if a.apply and a.dry_run:
        print(json.dumps({"ok": False, "reason": "choose one of --apply / --dry-run"})); return 2
    target = a.source if a.apply else (a.out or (a.source + ".q1patched"))
    if not (a.apply or a.dry_run):
        print(json.dumps({"ok": True, "would_patch": target, "anchor_counts": counts})); return 0
    with open(target, "w", encoding="utf-8") as f:
        f.write(patched)
    py_compile.compile(target, doraise=True)
    receipt = {"schema": "lumo.q1.fullmodel.reference-patch-receipt.v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "source": a.source, "source_sha256": src_sha, "target": target, "patched_sha256": sha256_file(target),
               "anchor_counts_before": counts, "marks_after": patched.count(MARK), "applied_in_place": bool(a.apply)}
    if a.receipt:
        with open(a.receipt, "w") as f:
            json.dump(receipt, f, indent=1)
    print(json.dumps(receipt, indent=1))
    return 0 if receipt["marks_after"] == 5 else 5


if __name__ == "__main__":
    sys.exit(main())

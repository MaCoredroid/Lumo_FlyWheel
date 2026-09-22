#!/usr/bin/env python3
"""E1 capture-only shim v2 (applied IN-CONTAINER after the patcher, like e7b_substitute_shim.py). FIVE anchors, each required
EXACTLY once (newline-exact), else the shim refuses (exit 3) and the boot must abort:
  F (gpu_model_runner.py): first statement of _fr13_sfwd_begin → forward_entry at EVERY forward (pure or mixed);
  B1/B2 (gpu_model_runner.py): both wall_break sites (non-pure step; start unavailable) → chain_break;
  W (gpu_model_runner.py): the single wall_mark call → physical_step (pure steps; fwd_index);
  R (gpu_model_runner.py, v3): the model runner's synchronous output bookkeeping (`_bookkeeping_sync`, before the
     per-request loop) → commit_rows with `req_ids_output_copy` (the runner's own row order, explicit) and
     `valid_sampled_token_ids` (HOST lists of the ACTUAL emitted ids; native and tree routes alike); async scheduling →
     error event (run invalid). (v2's sampler-site anchor is gone: it covered the tree committer only.)
  W (gpu_model_runner.py): the single `_fr13_timer.wall_mark(` call → insert BEFORE it a call to e1_recorder.physical_step
     with the same fwd_index/num_reqs/request_ids (and cg_mode from the start tuple).
  C (rejection_sampler.py): the committer anchor `_fr13_flags[0].fill_(0)` + `if _fr13_bnd_layer_on:` → insert between
     them a call to e1_recorder.commit_rows with the replay rows/paths/lens (the E7b shim uses the same anchor; both may be
     applied — each preserves the anchor text).
Guarded import: `_e1_rec()` returns None unless E1_RECORD is set, so an unset env is byte-inert in behaviour.
Usage: e1_event_recorder_shim.py [--runner PATH] [--sampler PATH] [--report PATH]"""
import argparse, hashlib, json, sys
from pathlib import Path
RUNNER = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
SAMPLER = "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/rejection_sampler.py"
GUARD = '''
def _e1_rec():
    """E1 event recorder (capture-only): None unless E1_RECORD is set."""
    import os as _e1_os
    if not _e1_os.environ.get("E1_RECORD"):
        return None
    import importlib.util as _e1_ilu, sys as _e1_sys
    _d = _e1_os.environ.get("E1_RUNTIME_DIR", "/workspace/papers/gdn-tree-scan-mlsys/v2/experiments/e1")
    if "e1_recorder" not in _e1_sys.modules:
        _s = _e1_ilu.spec_from_file_location("e1_recorder", _e1_os.path.join(_d, "e1_recorder.py")); _m = _e1_ilu.module_from_spec(_s); _s.loader.exec_module(_m); _e1_sys.modules["e1_recorder"] = _m
    return _e1_sys.modules["e1_recorder"]
'''
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def edit(path, anchor, insert_before, name, report):
    src = Path(path).read_text(); n = src.count(anchor)
    if n != 1: raise SystemExit(f"E1 shim REFUSING: anchor {name} found {n} times in {path}")
    before = sha(path)
    # guard once, before the first top-level `import torch` (module scope)
    if "def _e1_rec():" not in src:
        i = src.index("\nimport torch\n"); src = src[:i] + "\n" + GUARD + src[i:]
    src = src.replace(anchor, insert_before + anchor, 1); Path(path).write_text(src)
    report["edits"].append({"anchor": name, "file": str(path), "sha256_before": before, "sha256_after": sha(path)})
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runner", default=RUNNER); ap.add_argument("--sampler", default=SAMPLER); ap.add_argument("--report", default="/logs/e1_event_recorder_shim.json"); a = ap.parse_args()
    rep = {"shim_sha256": sha(__file__), "edits": []}
    def apply(path, edits):
        src = Path(path).read_text(); before = sha(path)
        if "def _e1_rec():" not in src: i = src.index("\nimport torch\n"); src = src[:i] + "\n" + GUARD + src[i:]
        for name, anchor, insert, where in edits:
            n = src.count(anchor)
            if n != 1: raise SystemExit(f"E1 shim REFUSING: anchor {name} found {n} times in {path}")
            src = src.replace(anchor, (insert + anchor) if where == "before" else (anchor + insert), 1); rep["edits"].append({"anchor": name, "file": path})
        Path(path).write_text(src); rep["edits"].append({"anchor": "file", "file": path, "sha256_before": before, "sha256_after": sha(path)})
    I4, I8, I12 = " " * 4, " " * 8, " " * 12
    apply(a.runner, [
        ("F-forward-entry", '\n    if __import__("os").environ.get("FR13_SFWD_GPU_TIMER", "0") != "1":\n        return None\n',
         "\n" + I4 + "_e1_m = _e1_rec()\n" + I4 + "if _e1_m is not None:\n" + I8 + "_e1_m.forward_entry(num_reqs=num_reqs, num_tokens=num_tokens, max_num_scheduled_tokens=max_num_scheduled_tokens)", "before"),
        ("B1-chain-break-nonpure", "\n            _fr13_sfwd_timer().wall_break()\n", "\n" + I12 + "_e1_m = _e1_rec()\n" + I12 + "if _e1_m is not None:\n" + I12 + "    _e1_m.chain_break(reason=\"non-pure step\")", "before"),
        ("B2-chain-break-no-start", "\n    if _fr13_start is None:\n        _fr13_timer.wall_break()\n        return None\n", "\n" + I4 + "if _fr13_start is None:\n" + I8 + "_e1_m = _e1_rec()\n" + I8 + "if _e1_m is not None:\n" + I8 + "    _e1_m.chain_break(reason=\"timer start unavailable\")", "before"),
        ("W-physical-step", "\n        _fr13_timer.wall_mark(\n", "\n" + I8 + "_e1_m = _e1_rec()\n" + I8 + "if _e1_m is not None:\n" + I12 + "_e1_m.physical_step(fwd_index=_fr13_start[3], num_reqs=num_reqs, request_ids=request_ids[:num_reqs], cg_mode=_fr13_start[2])", "before"),
        ("R-output-boundary", "\n        req_ids = self.input_batch.req_ids\n        for req_idx in range(num_sampled_tokens):\n",
         "\n" + I8 + "_e1_m = _e1_rec()\n" + I8 + "if _e1_m is not None:\n" + I12 + "_e1_m.commit_rows(req_ids=req_ids_output_copy, sampled_lists=(valid_sampled_token_ids if not self.use_async_scheduling else []), num_draft_tokens=(getattr(spec_decode_metadata, \"num_draft_tokens\", None) if spec_decode_metadata is not None else None), async_scheduling=bool(self.use_async_scheduling))", "before"),
    ])
    Path(a.report).parent.mkdir(parents=True, exist_ok=True); Path(a.report).write_text(json.dumps(rep, indent=1)); print("E1 shim v3 applied:", [e["anchor"] for e in rep["edits"]])
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E7a capture-only shim. Runs INSIDE the serving container AFTER the original patcher
(scripts/fr10_phase4_patch_vllm_tree_gdn.py) and only when E7A_CAPTURE_SHIM=1.

Why: at HEAD the production tree route is STATELESS (FR13_REPLAY_ROUTE=1: no per-node state scratch). The
one-shot GDN operand capture (FR10_TREE_GDN_CAPTURE_PAYLOAD) predates that change: (a) a diagnostic guard
raises when the capture env is set while the replay route is on, and (b) the capture dict serializes
`tree_state[:tree_n]`, which is None on the stateless route, so the dict construction raises inside its
try-block and NO payload is written. Turning the replay route OFF is not an option at HEAD: the legacy publish
path also dereferences the removed state.

What this shim changes (exact-text, fail-closed, each anchor must match EXACTLY once):
  1. the diagnostic guard ignores FR10_TREE_GDN_CAPTURE_PAYLOAD when E7A_CAPTURE_SHIM=1 (the other three
     diagnostics stay guarded);
  2. `"serving_state": tree_state[:tree_n]...` becomes None-safe (None on the stateless route).
Nothing outside the capture-only try-block and its guard is touched; the serving arithmetic, flags, gates and
routes are unchanged. A report with before/after sha256 and anchor line numbers is written to /logs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

TARGET = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py")
REPORT = Path(os.environ.get("E7A_CAPTURE_SHIM_REPORT", "/logs/e7a_capture_shim.json"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    if os.environ.get("E7A_CAPTURE_SHIM") != "1":
        print("E7A_CAPTURE_SHIM != 1: shim not applied")
        return 0
    src = TARGET.read_text()
    before = sha(TARGET)
    edits = []

    # --- edit 1: guard bypass for the payload capture only
    guard = re.compile(
        r'(?P<ind>[ \t]*)os\.environ\.get\("FR10_TREE_GDN_CAPTURE_PAYLOAD"\)\n'
        r'(?P=ind)or os\.environ\.get\("FR10_TREE_GDN_COMMIT_HANDOFF_LOG"\)\n'
    )
    m = list(guard.finditer(src))
    if len(m) != 1:
        raise SystemExit(f"shim anchor 1 (guard) matched {len(m)} times; refusing")
    ind = m[0].group("ind")
    repl = (f'{ind}(os.environ.get("FR10_TREE_GDN_CAPTURE_PAYLOAD") and os.environ.get("E7A_CAPTURE_SHIM") != "1")\n'
            f'{ind}or os.environ.get("FR10_TREE_GDN_COMMIT_HANDOFF_LOG")\n')
    edits.append({"anchor": "guard", "line": src[: m[0].start()].count("\n") + 1})
    src = src[: m[0].start()] + repl + src[m[0].end():]

    # --- edit 2: None-safe serving_state in the capture dict
    st = re.compile(r'"serving_state":\s*tree_state\[:tree_n\]\s*\.detach\(\)\s*\.cpu\(\)\s*\.clone\(\),')
    m2 = list(st.finditer(src))
    if len(m2) != 1:
        raise SystemExit(f"shim anchor 2 (serving_state) matched {len(m2)} times; refusing")
    edits.append({"anchor": "serving_state", "line": src[: m2[0].start()].count("\n") + 1})
    src = src[: m2[0].start()] + '"serving_state": (tree_state[:tree_n].detach().cpu().clone() if tree_state is not None else None),' + src[m2[0].end():]

    compile(src, str(TARGET), "exec")  # syntax check before writing
    TARGET.write_text(src)
    after = sha(TARGET)
    rep = {"schema": "e7a.capture_shim.v1", "applied_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "target": str(TARGET), "sha256_before": before, "sha256_after": after, "edits": edits,
           "shim_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "scope": "capture-only guard + None-safe serving_state; serving arithmetic untouched"}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1))
    print("e7a capture shim applied:", json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())

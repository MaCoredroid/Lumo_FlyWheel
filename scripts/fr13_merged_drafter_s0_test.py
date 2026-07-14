#!/usr/bin/env python3
"""FR13 merged-drafter patcher SELF-TEST (gate c, host-runnable, no GPU).

Applies the runner-side patch chain (tree_reqkey -> merged_drafter) to a FRESH copy of a pristine
vLLM gpu_model_runner.py and asserts: the anchor matched, the FR13_MERGED_DRAFTER_LIFECYCLE marker
is present, the merged logic is behind the `if _fr13_md.merged_on():` sidecar gate (=> byte-identical
when off), the patched file COMPILES, and the patch is idempotent. Catches anchor drift / syntax
errors BEFORE a GPU boot.
"""
import importlib.util
import os
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
PRISTINE_CANDIDATES = [
    "/tmp/lumo_tree_patch_probe_t11i0n5r/vllm/v1/worker/gpu_model_runner.py",
    "/tmp/lumo_tree_patch_probe_gweqrziq/vllm/v1/worker/gpu_model_runner.py",
    "/tmp/fr13_bnd_pristine/v1/worker/gpu_model_runner.py",
]

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {msg}")
    else:
        FAIL += 1; print(f"  [FAIL] {msg}")


pristine = next((p for p in PRISTINE_CANDIDATES if os.path.exists(p)), None)
if pristine is None:
    print("SKIP: no pristine gpu_model_runner.py found on host (need a container/probe copy)")
    sys.exit(0)
print(f"pristine: {pristine}")

# import the patcher module (definitions only; __main__ guard is inert on import)
spec = importlib.util.spec_from_file_location("fr10patch_selftest", str(PATCHER))
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

tmp = Path(tempfile.mkdtemp())
work = tmp / "gpu_model_runner.py"
shutil.copy(pristine, work)
patcher.GPU_MODEL_RUNNER_PATH = work

# reqkey MUST run first (my hook anchors on its output line)
r_reqkey = patcher._patch_gpu_model_runner_tree_reqkey()
check(bool(r_reqkey), "tree_reqkey patch applied (produces the anchor)")
try:
    r_merged = patcher._patch_gpu_model_runner_merged_drafter()
    applied_ok = True
except Exception as e:
    r_merged = None; applied_ok = False
    print(f"  [FAIL] merged_drafter patch raised: {e!r}")
check(applied_ok and bool(r_merged), "merged_drafter patch applied (anchor matched)")

text = work.read_text()
check("# FR13_TREE_REQKEY_REWRITE" in text, "reqkey marker present")
check("# FR13_MERGED_DRAFTER_LIFECYCLE" in text, "merged-drafter marker present")
check("if _fr13_md.merged_on():" in text, "merged logic behind merged_on() sidecar gate (byte-id off)")
check("_fr13_md.note_new_requests" in text and "_fr13_md.ingest_from_sequence" in text
      and "_fr13_md.retire_requests" in text, "all 3 lifecycle calls injected")
# the merged block must be inside its OWN try/except (can't disable reqkey)
_idx = text.find("# FR13_MERGED_DRAFTER_LIFECYCLE")
_seg = text[_idx:_idx + 3000]
_try = _seg.find("\n                    try:")
_exc = _seg.find("\n                    except Exception:")
check(0 <= _try < _exc, "merged block wrapped in own try/except (own error isolation)")

# COMPILES (the decisive syntax/indent check in real context)
try:
    py_compile.compile(str(work), doraise=True)
    check(True, "patched gpu_model_runner.py COMPILES")
except py_compile.PyCompileError as e:
    check(False, f"compile FAILED: {e}")

# idempotent: re-running merged returns False (sentinel guard)
check(patcher._patch_gpu_model_runner_merged_drafter() is False, "merged patch idempotent (sentinel)")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{PASS}/{PASS+FAIL} checks PASS")
sys.exit(0 if FAIL == 0 else 1)

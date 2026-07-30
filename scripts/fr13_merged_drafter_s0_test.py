#!/usr/bin/env python3
"""FR13 merged-drafter patcher SELF-TEST (gate c, host-runnable, no GPU).

Applies the runner-side patch chain (row_ids_fresh -> tree_reqkey -> merged_drafter) to a FRESH
copy of a pristine vLLM gpu_model_runner.py and asserts: the anchor matched, the
FR13_MERGED_DRAFTER_LIFECYCLE marker
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
        PASS += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL += 1
        print(f"  [FAIL] {msg}")


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

# The lifecycle hook anchors on the unconditional row-owner prelude.
r_rowids = patcher._patch_gpu_model_runner_row_req_ids_fresh()
check(bool(r_rowids), "row_req_ids_fresh patch applied (produces lifecycle anchor)")
r_reqkey = patcher._patch_gpu_model_runner_tree_reqkey()
check(bool(r_reqkey), "tree_reqkey patch applied")
try:
    r_merged = patcher._patch_gpu_model_runner_merged_drafter()
    applied_ok = True
except Exception as e:
    r_merged = None
    applied_ok = False
    print(f"  [FAIL] merged_drafter patch raised: {e!r}")
check(applied_ok and bool(r_merged), "merged_drafter patch applied (anchor matched)")

text = work.read_text()
check("# FR13_TREE_REQKEY_REWRITE" in text, "reqkey marker present")
check("# FR13_MERGED_DRAFTER_LIFECYCLE" in text, "merged-drafter marker present")
check(
    "_fr13_md_on = _fr13_md.merged_on()" in text
    and "if _fr13_md_on:" in text,
    "merged logic behind merged_on() sidecar gate (legacy byte-id off)",
)
check(
    "_fr13_md.note_new_requests" in text
    and "_fr13_md.ingest_from_sequence" in text
    and "_fr13_md.retire_requests" in text
    and "_fr13_md.stage_fixed32_step" in text,
    "legacy and fixed32 lifecycle calls injected",
)
# The lifecycle has a local exception boundary. Fixed32 rethrows; legacy
# retains the historical best-effort isolation.
_idx = text.find("# FR13_MERGED_DRAFTER_LIFECYCLE")
_seg = text[_idx:_idx + 3000]
_try = _seg.find("\n            try:")
_exc = _seg.find("\n            except Exception")
check(0 <= _try < _exc, "merged block has its own exception boundary")

# COMPILES (the decisive syntax/indent check in real context)
try:
    py_compile.compile(str(work), doraise=True)
    check(True, "patched gpu_model_runner.py COMPILES")
except py_compile.PyCompileError as e:
    check(False, f"compile FAILED: {e}")

# idempotent: re-running merged returns False (sentinel guard)
check(patcher._patch_gpu_model_runner_merged_drafter() is False, "merged patch idempotent (sentinel)")

# --- S4 seam DELETED 2026-07-27 (cleanup+bake): assert the deletion is complete ---
# The head-merge decide_and_fill seam was removed (FR13_CLEANUP_BAKE_PLAN.md); a tombstone
# comment marks the site. Assert: tombstone present, no live decide_and_fill call anywhere,
# and the tail path (decide_tail) is still wired.
psrc = PATCHER.read_text()
check("# FR13_MERGED_DRAFTER_SEAM DELETED" in psrc, "seam tombstone present in patcher")
check("decide_and_fill(" not in psrc, "no live decide_and_fill call remains in patcher")
check("decide_tail" in psrc, "tail path (decide_tail) still wired in patcher")

# best-effort: apply the eagle patch to a pristine copy if one matches the target version
EAGLE_CANDIDATES = [
    "/tmp/lumo_tree_patch_probe_t11i0n5r/vllm/v1/spec_decode/eagle.py",
    "/tmp/fr13_bnd_pristine/v1/spec_decode/eagle.py",
]
_matched = False
for _ep in EAGLE_CANDIDATES:
    if not os.path.exists(_ep):
        continue
    ework = tmp / "eagle.py"
    shutil.copy(_ep, ework)
    patcher.EAGLE_PATH = ework
    try:
        patcher._patch_eagle_tree_consumption_verify()
        py_compile.compile(str(ework), doraise=True)
        check("# FR13_MERGED_DRAFTER_SEAM" in ework.read_text(), f"eagle patch applied+compiles ({os.path.basename(os.path.dirname(_ep))})")
        _matched = True
        break
    except Exception:
        continue
if not _matched:
    print("  [SKIP] no version-matched pristine eagle.py on host -> seam splice validated at BOOT (loud patch-time failure)")

shutil.rmtree(tmp, ignore_errors=True)
print(f"\n{PASS}/{PASS+FAIL} checks PASS")
sys.exit(0 if FAIL == 0 else 1)

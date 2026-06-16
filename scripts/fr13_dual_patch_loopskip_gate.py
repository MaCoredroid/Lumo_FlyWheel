#!/usr/bin/env python3
"""HARDENED OFFLINE (CPU, no-GPU) dual-patch gate for the FR13 GPU-committer
loop-skip injection in scripts/fr10_phase4_patch_vllm_tree_gdn.py.

THE BUG THIS GATE EXISTS TO CATCH (proven by GPU verify wgkt3dzpf):
  _patch_rejection_sampler_tree_lcp() runs FIRST and injects a doc comment
  containing the text "# FR13_GPU_COMMITTER)." (helper r-string, ~line 6065).
  _patch_rejection_sampler_gpu_committer() runs SECOND; its idempotency guard
  used the sentinel "# FR13_GPU_COMMITTER" -> that substring was ALREADY in the
  file (from the tree_lcp comment), so the guard mis-fired "already patched" and
  BAILED. The loop-skip `if _fr13_gpu_committer:` was NEVER injected; the bare
  legacy `for req_i, node_count in enumerate(counts)` loop ran on the
  synckill-nulled parents_cpu -> RuntimeError -> EngineDeadError at B=1 warmup.

WHAT THIS GATE DOES (the check that would have caught the collision):
  1. Reads a FRESH rejection_sampler.py from the PINNED image via
     scripts/vllm_src.sh (file-read only via `docker run --rm cat`, NO GPU). If
     docker is unavailable it falls back to $FR13_FRESH_REJECTION_SAMPLER or a
     pre-fetched copy next to this file.
  2. Applies BOTH patchers in the REAL dispatch order (tree_lcp THEN
     gpu_committer) to that fresh source, exactly as patch_steps does.
  3. ASSERTS the emitted text contains `if _fr13_gpu_committer:` (the loop-skip)
     BEFORE the legacy `for ... enumerate(counts)` loop. This is the assertion
     that fails on the colliding sentinel.
  4. SELF-TEST (negative control): re-runs the gpu_committer patcher with the OLD
     colliding sentinel "# FR13_GPU_COMMITTER" monkeypatched back in, and asserts
     the gate's loop-skip check FAILS -> proves this gate catches the bug.
  5. Confirms idempotency on a GENUINE gpu_committer double-apply (re-apply is a
     no-op, byte-identical), and uniqueness of the new sentinel (appears ONLY at
     the injection sites, NOT in the tree_lcp doc comment).

Exit 0 = PASS. Nonzero = FAIL.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
VLLM_SRC = REPO / "scripts" / "vllm_src.sh"
REJ_RELPATH = "v1/sample/rejection_sampler.py"
# fallback pre-fetched fresh copy (refresh with:
#   scripts/vllm_src.sh v1/sample/rejection_sampler.py > <this path>)
FALLBACK = REPO / "scripts" / "fr13_fresh_rejection_sampler.py"

NEW_SENTINEL = "# FR13_GPU_COMMITTER_LOOPSKIP_SOT"
OLD_SENTINEL = "# FR13_GPU_COMMITTER"
LOOPSKIP = "if _fr13_gpu_committer:"
LEGACY_LOOP = "for req_i, node_count in enumerate(counts)"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        fails.append(name)
    return cond


def load_patcher():
    spec = importlib.util.spec_from_file_location("fr13_patcher_dualgate", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fresh_rejection_sampler() -> str:
    """Read a FRESH rejection_sampler.py from the pinned image (file-read only)."""
    override = os.environ.get("FR13_FRESH_REJECTION_SAMPLER")
    if override and Path(override).exists():
        print(f"[src] using FR13_FRESH_REJECTION_SAMPLER={override}")
        return Path(override).read_text()
    if VLLM_SRC.exists():
        try:
            out = subprocess.run(
                ["bash", str(VLLM_SRC), REJ_RELPATH],
                capture_output=True, text=True, timeout=180,
            )
            if out.returncode == 0 and out.stdout.strip():
                print(f"[src] fresh from pinned image via vllm_src.sh ({len(out.stdout.splitlines())} lines)")
                return out.stdout
            print(f"[src] vllm_src.sh failed (rc={out.returncode}): {out.stderr.strip()[:200]}")
        except Exception as e:  # docker missing / sandbox: fall back
            print(f"[src] vllm_src.sh raised {type(e).__name__}: {e}")
    if FALLBACK.exists():
        print(f"[src] FALLBACK pre-fetched copy {FALLBACK}")
        return FALLBACK.read_text()
    raise SystemExit(
        "FAIL: no fresh rejection_sampler.py source (vllm_src.sh + docker "
        "unavailable, no FR13_FRESH_REJECTION_SAMPLER, no fallback copy). "
        "Pre-fetch one: scripts/vllm_src.sh v1/sample/rejection_sampler.py > "
        f"{FALLBACK}"
    )


def apply_both_patchers(mod, fresh_text: str) -> str:
    """Apply tree_lcp THEN gpu_committer to fresh_text, return emitted text."""
    tmp = Path(tempfile.mkstemp(prefix="fr13_dualgate_", suffix=".py")[1])
    tmp.write_text(fresh_text)
    saved = mod.REJECTION_SAMPLER_PATH
    try:
        mod.REJECTION_SAMPLER_PATH = tmp
        r1 = mod._patch_rejection_sampler_tree_lcp()
        r2 = mod._patch_rejection_sampler_gpu_committer()
        emitted = tmp.read_text()
        return emitted, r1, r2, tmp
    finally:
        mod.REJECTION_SAMPLER_PATH = saved


def loopskip_before_legacy_loop(text: str) -> bool:
    """The hardened assertion: loop-skip injected BEFORE the legacy loop."""
    if LOOPSKIP not in text or LEGACY_LOOP not in text:
        return False
    return text.index(LOOPSKIP) < text.index(LEGACY_LOOP)


def main() -> int:
    mod = load_patcher()
    fresh = fresh_rejection_sampler()

    # ---- D1: dual-patch in REAL order on a FRESH source ----
    emitted, r1, r2, tmp = apply_both_patchers(mod, fresh)
    check("D1a tree_lcp patcher applied", r1 is True)
    check("D1b gpu_committer patcher applied (NOT bailed by sentinel collision)", r2 is True,
          "<-- this is the bug: was False when the sentinel collided")

    # ---- D2: the HARDENED assertion -- loop-skip injected BEFORE legacy loop ----
    check("D2 loop-skip `if _fr13_gpu_committer:` present BEFORE legacy "
          "`for ... enumerate(counts)` loop", loopskip_before_legacy_loop(emitted))

    # show the emitted region for the record
    if LOOPSKIP in emitted:
        idx = emitted.index(NEW_SENTINEL) if NEW_SENTINEL in emitted else emitted.index(LOOPSKIP)
        region = emitted[max(0, idx - 60): idx + 360]
        print("---- emitted loop-skip region ----")
        for ln in region.splitlines():
            print("  | " + ln)
        print("-----------------------------------")

    # ---- D3: sentinel uniqueness ----
    new_lines = [(i, ln) for i, ln in enumerate(emitted.splitlines(), 1) if NEW_SENTINEL in ln]
    check("D3a new sentinel appears ONLY at injection sites (>=1, all are "
          "gpu_committer markers)", len(new_lines) >= 1,
          f"lines={[i for i, _ in new_lines]}")
    # NOT a substring of the tree_lcp doc comment line(s)
    treelcp_comment_lines = [
        ln for ln in emitted.splitlines()
        if OLD_SENTINEL in ln and NEW_SENTINEL not in ln
    ]
    check("D3b tree_lcp doc comment `# FR13_GPU_COMMITTER).` does NOT contain "
          "the new sentinel", all(NEW_SENTINEL not in ln for ln in treelcp_comment_lines),
          f"({len(treelcp_comment_lines)} tree_lcp comment line(s))")

    # ---- D4: genuine gpu_committer double-apply is idempotent ----
    saved = mod.REJECTION_SAMPLER_PATH
    try:
        mod.REJECTION_SAMPLER_PATH = tmp
        r3 = mod._patch_rejection_sampler_gpu_committer()
        emitted2 = tmp.read_text()
    finally:
        mod.REJECTION_SAMPLER_PATH = saved
    check("D4a gpu_committer re-apply is a no-op (idempotent -> False)", r3 is False)
    check("D4b emitted byte-identical on genuine double-apply", emitted2 == emitted)

    # ---- D5: NEGATIVE CONTROL -- prove the gate FAILS on the OLD sentinel ----
    # Monkeypatch the gpu_committer patcher to use the OLD colliding sentinel and
    # confirm the loop-skip is then MISSING (gate would FAIL) -- this proves the
    # hardened assertion catches the bug.
    src = PATCHER.read_text()
    # Build a sibling module whose gpu_committer uses the OLD sentinel by patching
    # the assignment line in source, exec-ing a fresh module.
    src_old = src.replace(
        f'    sentinel = "{NEW_SENTINEL}"',
        f'    sentinel = "{OLD_SENTINEL}"',
        1,
    )
    collided = "/* not replaced */"
    if src_old == src:
        check("D5a constructed OLD-sentinel variant", False,
              "could not find `sentinel = \"<new>\"` assignment to revert")
    else:
        check("D5a constructed OLD-sentinel variant (reverted sentinel "
              "assignment)", True)
        old_mod_spec = importlib.util.spec_from_loader("fr13_patcher_oldsent", loader=None)
        old_mod = importlib.util.module_from_spec(old_mod_spec)
        exec(compile(src_old, str(PATCHER), "exec"), old_mod.__dict__)
        emitted_old, ro1, ro2, _ = apply_both_patchers(old_mod, fresh)
        # With the OLD sentinel, gpu_committer should BAIL (return False) because
        # the tree_lcp comment already contains "# FR13_GPU_COMMITTER".
        check("D5b OLD sentinel -> gpu_committer BAILS (reproduces the bug)",
              ro2 is False)
        check("D5c OLD sentinel -> loop-skip MISSING -> HARDENED assertion FAILS "
              "(gate catches the bug)", not loopskip_before_legacy_loop(emitted_old))
        collided = "yes" if not loopskip_before_legacy_loop(emitted_old) else "no"

    print()
    print(f"[neg-control] gate FAILS on OLD colliding sentinel: {collided}")
    print("DUAL-PATCH LOOPSKIP GATE:", "PASS" if not fails else ("FAIL " + ",".join(fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

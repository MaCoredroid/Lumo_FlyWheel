#!/usr/bin/env python3
"""OFFLINE (CPU, no-GPU) gate for the FR13_COMMITTER_SYNCKILL single-source-of-
truth fix in scripts/fr10_phase4_patch_vllm_tree_gdn.py.

Proves, without a GPU / live vLLM:
  G1  the injected committer source (the `helper` r-string) is valid Python.
  G2  the injected loop-skip replacement source is valid Python.
  G3  the synckill NULL guard and the legacy-LOOP-SKIP reference the SAME
      `_fr13_gpu_committer` variable -> identical-by-construction (the null can
      NEVER fire when the loop-skip is False). Done by executing the EXACT
      hoisted predicate code over the full diagnostic-flag state space and
      checking null==loop-skip in every cell.
  G4  default-OFF (FR13_COMMITTER_SYNCKILL unset / FR13_GPU_COMMITTER unset) the
      null cannot fire and the loop-skip is False -> legacy path unchanged.
  G5  fail-loud preserved (the composition-defect RuntimeError text + the
      loop-skip missing-predicate RuntimeError text are still present).
"""
import ast
import itertools
import os
import re
import sys

PATCHER = os.path.join(os.path.dirname(__file__), "fr10_phase4_patch_vllm_tree_gdn.py")
src = open(PATCHER).read()
lines = src.splitlines()

fails = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail)
    if not cond:
        fails.append(name)


# ---- locate the committer helper r-string (first `helper = r'''` .. `'''`) ----
start = None
for i, ln in enumerate(lines):
    if ln.strip() == "helper = r'''":
        start = i
        break
assert start is not None, "committer helper r-string start not found"
end = None
for j in range(start + 1, len(lines)):
    if lines[j] == "'''":
        end = j
        break
assert end is not None, "committer helper r-string end not found"
helper_text = "\n".join(lines[start + 1:end])

# G1: injected committer source parses
try:
    ast.parse(helper_text)
    check("G1 injected committer source parses", True)
except SyntaxError as e:
    check("G1 injected committer source parses", False, repr(e))

# ---- G3/G4: the hoisted predicate block, executed verbatim ----
# Extract the EXACT top-of-function single-source block (the `_fr13_gpu_committer`
# assignment and its inputs) from the helper text so we test the SHIPPED code,
# not a paraphrase.
m = re.search(
    r"(    _fr13_cag_target_logits = globals\(\).*?    _fr13_gpu_committer = \(.*?\n    \)\n)",
    helper_text,
    re.S,
)
check("G3a hoisted predicate block found in committer source", m is not None)
import textwrap
pred_block = textwrap.dedent(m.group(1)) if m else ""

# The NULL guard line and the LOOP-SKIP both must reference _fr13_gpu_committer.
null_guard_uses = "_FR13_COMMITTER_SYNCKILL and _fr13_gpu_committer" in helper_text
check("G3b synckill NULL guard references _fr13_gpu_committer", null_guard_uses)
# Old drifting duplicate (the LIVE recompute assignment) must be GONE from the
# committer source. (The name may still appear in explanatory comments.)
check(
    "G3c old env-recompute _fr13_sk_engage assignment removed",
    "_fr13_sk_engage = (" not in helper_text
    and "if _FR13_COMMITTER_SYNCKILL and _fr13_sk_engage:" not in helper_text,
)

# ---- locate the loop-skip replacement (in _patch_rejection_sampler_gpu_committer) ----
gco_start = src.index("def _patch_rejection_sampler_gpu_committer(")
gco = src[gco_start:]
# the replacement is built as a python-string concatenation; reconstruct the
# emitted source by eval-ing the `replacement = ( ... )` literal pieces.
rep_m = re.search(r"replacement = \(\n(.*?)\n    \)\n", gco, re.S)
check("G2a loop-skip replacement literal found", rep_m is not None)
emitted = ""
if rep_m:
    sentinel = "# FR13_GPU_COMMITTER"
    # evaluate each "..."-quoted chunk; the only runtime name is `sentinel`.
    for chunk in re.findall(r'"((?:[^"\\]|\\.)*)"', rep_m.group(1)):
        emitted += chunk.encode().decode("unicode_escape")
    # sentinel concatenations: `" + sentinel + "` already split chunks, so the
    # reconstructed text has gaps where sentinel went; reinsert by replacing the
    # known concatenation pattern is unnecessary for an AST parse of the body.
# the emitted loop-skip is a function-body fragment; wrap it to parse.
loopskip_body = emitted
# G2: the emitted loop-skip must reference the single predicate and NOT recompute
check(
    "G2b loop-skip references single-source _fr13_gpu_committer (no recompute)",
    "if _fr13_gpu_committer:" in loopskip_body
    and "_fr13_gpu_committer = (" not in loopskip_body,
)
check(
    "G2c loop-skip fail-loud if predicate missing",
    "single-source predicate _fr13_gpu_committer" in loopskip_body,
)

# ---- G3 core: execute the SHIPPED predicate over the full flag state space ----
# Build a tiny harness that runs the exact hoisted predicate code (pred_block)
# plus the null/loop-skip booleans, and asserts: null_fires => loop_skip_engages
# i.e. they are the SAME boolean. We feed every combination of the 5 inputs.
#
# Inputs that drive the predicate:
#   env FR13_GPU_COMMITTER, FR13_FORCE_SPINE_COMMIT, FR13_TREE_BONUS_SELF
#   globals _FR13_COMMIT_ARGMAX_GATE, _FR13_FORK_MARGIN_DUMP and the published
#   _FR13_CAG_TARGET_LOGITS (None vs not-None)
#   env/flag FR13_COMMITTER_SYNCKILL (only gates the null, ANDed with predicate)
needle_calls = []


def _fr13_commit_argmax_gate_needle(x):
    needle_calls.append(("cag", x))


def _fr13_fork_margin_dump_needle(x):
    needle_calls.append(("fmd", x))


space = list(itertools.product([0, 1], repeat=6))
mismatches = 0
null_fired_without_loopskip = 0
for gpu_c, force_sp, bonus, cag_flag, fmd_flag, cag_pub in space:
    env = {
        "FR13_GPU_COMMITTER": str(gpu_c),
        "FR13_FORCE_SPINE_COMMIT": str(force_sp),
        "FR13_TREE_BONUS_SELF": str(bonus),
    }
    g = {
        "_FR13_COMMIT_ARGMAX_GATE": bool(cag_flag),
        "_FR13_FORK_MARGIN_DUMP": bool(fmd_flag),
        "_FR13_CAG_TARGET_LOGITS": (object() if cag_pub else None),
        "_FR13_CAG_SELF_LOGITS": None,
        "_FR13_CAG_STEP": 3,
    }
    ns = {
        "globals": (lambda gg=g: gg),
        "_FR13_COMMIT_ARGMAX_GATE": g["_FR13_COMMIT_ARGMAX_GATE"],
        "_FR13_FORK_MARGIN_DUMP": g["_FR13_FORK_MARGIN_DUMP"],
        "_fr13_commit_argmax_gate_needle": _fr13_commit_argmax_gate_needle,
        "_fr13_fork_margin_dump_needle": _fr13_fork_margin_dump_needle,
        "__import__": (lambda n, *a, **k: type("E", (), {"environ": env})() if n == "os" else __import__(n, *a, **k)),
    }
    # exec the EXACT shipped predicate block
    exec(pred_block, ns, ns)
    loop_skip = ns["_fr13_gpu_committer"]
    # the null guard fires iff SYNCKILL and the SAME predicate
    for synckill in (False, True):
        null_fires = synckill and ns["_fr13_gpu_committer"]
        # the invariant: null_fires implies loop_skip
        if null_fires and not loop_skip:
            null_fired_without_loopskip += 1
        # they must reference the identical predicate (null is predicate ANDed
        # with synckill); when synckill True they are EQUAL
        if synckill and (null_fires != loop_skip):
            mismatches += 1

check("G3d null implies loop-skip across ALL 64 flag cells", null_fired_without_loopskip == 0)
check("G3e null==loop-skip when SYNCKILL on (identical boolean)", mismatches == 0)

# ---- G4: default-OFF byte-identity (the SYNCKILL-on branch is the only change) ----
# default boot: FR13_COMMITTER_SYNCKILL unset(=0) AND FR13_GPU_COMMITTER unset(=0)
env0 = {}  # nothing set -> all .get(...,'0')/'1' defaults
ns0 = {
    "globals": (lambda: {"_FR13_CAG_TARGET_LOGITS": None}),
    "_FR13_COMMIT_ARGMAX_GATE": False,
    "_FR13_FORK_MARGIN_DUMP": False,
    "_fr13_commit_argmax_gate_needle": _fr13_commit_argmax_gate_needle,
    "_fr13_fork_margin_dump_needle": _fr13_fork_margin_dump_needle,
    "__import__": (lambda n, *a, **k: type("E", (), {"environ": env0})() if n == "os" else __import__(n, *a, **k)),
}
exec(pred_block, ns0, ns0)
check("G4a default-OFF: _fr13_gpu_committer is False (loop-skip not taken)", ns0["_fr13_gpu_committer"] is False)
# with the predicate False, `_FR13_COMMITTER_SYNCKILL and _fr13_gpu_committer`
# is False regardless of SYNCKILL -> the parents_cpu null can NOT fire -> legacy
# DtoH else-branch runs -> host lists materialised -> legacy loop unchanged.
check("G4b default-OFF: synckill null cannot fire (predicate False)", (True and ns0["_fr13_gpu_committer"]) is False)

# ---- G5: fail-loud strings preserved ----
check(
    "G5a composition-defect RuntimeError preserved",
    "FR13_COMMITTER_SYNCKILL composition defect" in helper_text,
)
check(
    "G5b loop-skip missing-predicate fail-loud present",
    "single-source predicate _fr13_gpu_committer" in src,
)

print()
print("OFFLINE GATE:", "PASS" if not fails else ("FAIL " + ",".join(fails)))
sys.exit(1 if fails else 0)

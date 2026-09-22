#!/usr/bin/env python3
"""Produce a FIXED copy of scripts/fr10_phase4_patch_vllm_tree_gdn.py with ONE declared edit and record both hashes.

Defect (HEAD): `_patch_eagle_tree_consumption_verify` emits the drafter module constant `_FR13_FIXED32_MODE` into
eagle.py only `if _FR13_FIXED32_MODE:` (i.e. only when FR13_FIXED32_MODE is set at patch time), while the emitted
`propose` body references the name unconditionally -> NameError on the first draft proposal of the PLAIN tree route.
Fix: emit the constant unconditionally with the patcher's own value (the env-unset default is ""), which selects the
plain-route branches the emitted code already implements. No flag, gate, refusal check or numerical path is altered.
Usage: make_fixed_patcher.py <orig> <fixed_out> <report_json>
"""
import difflib, hashlib, json, sys

orig_p, fixed_p, rep_p = sys.argv[1:4]
src = open(orig_p, encoding="utf-8").read()
OLD = '''    if _FR13_FIXED32_MODE:
        text = text.replace(
            "import os\\n",
            f"import os\\n_FR13_FIXED32_MODE = {_FR13_FIXED32_MODE!r}\\n",
            1,
        )

    tree_parse_old = """        # Parse the speculative token tree.'''
NEW = '''    # E7A-FIX (2026-09-21): emit the drafter constant UNCONDITIONALLY. The emitted propose body references
    # _FR13_FIXED32_MODE on every route; HEAD emitted its definition only when FR13_FIXED32_MODE was set, so the
    # plain tree_mtp route raised NameError on the first proposal. The value is the patcher's own (env-unset -> "").
    text = text.replace(
        "import os\\n",
        f"import os\\n_FR13_FIXED32_MODE = {_FR13_FIXED32_MODE!r}\\n",
        1,
    )

    tree_parse_old = """        # Parse the speculative token tree.'''
assert src.count(OLD) == 1, f"anchor count {src.count(OLD)}"
fixed = src.replace(OLD, NEW)
compile(fixed, fixed_p, "exec")
open(fixed_p, "w", encoding="utf-8").write(fixed)
diff = list(difflib.unified_diff(src.splitlines(True), fixed.splitlines(True), "a/scripts/fr10_phase4_patch_vllm_tree_gdn.py", "b/scripts/fr10_phase4_patch_vllm_tree_gdn.py"))
rep = {"orig": orig_p, "orig_sha256": hashlib.sha256(src.encode()).hexdigest(), "fixed": fixed_p,
       "fixed_sha256": hashlib.sha256(fixed.encode()).hexdigest(), "diff_lines": len(diff), "diff": "".join(diff),
       "declared_change": "unconditional emission of _FR13_FIXED32_MODE constant into eagle.py (value unchanged: patcher's own, '' when env unset)"}
json.dump(rep, open(rep_p, "w"), indent=2)
print(f"fixed patcher written: orig {rep['orig_sha256'][:16]} -> fixed {rep['fixed_sha256'][:16]}, diff {len(diff)} lines")

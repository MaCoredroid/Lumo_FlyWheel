#!/usr/bin/env python3
"""Bounded CPU regression for the HEAD plain-route drafter defect (NameError: _FR13_FIXED32_MODE, eagle.py propose).

Evidence base: a CPU-only dry-run of the patcher inside the pinned image (`patcher_dryrun.sh`) that emits the patched
vLLM sources under the EXACT container env of the failed boot (attempt 2) for the HEAD patcher (arm A) and the
FIXED patcher (arm B). This test re-audits those emitted files with `emitted_globals_audit.py` and asserts:
  1. HEAD emission: eagle.py has exactly ONE new unbound global, `_FR13_FIXED32_MODE`, referenced inside `propose`
     and at the traceback line of the failed boot (1174);
  2. FIXED emission: eagle.py has no new unbound globals; the A->B eagle.py diff is exactly one added definition line;
  3. FIXED emission introduces no unbound name in ANY emitted file that HEAD did not already have (no regression);
  4. the remaining unbound names (fixed32 SFWD helpers in gdn_linear_attn.py) are identical between A and B and are
     documented as latent/unreachable on the plain route (gated by module constants that are False/'' here);
  5. `make_fixed_patcher.py` applied to the HEAD patcher reproduces the FIXED patcher byte-for-byte (declared edit).
Usage: test_patcher_plain_route_globals.py <dryrun_run_dir> <head_patcher> <out_json>
"""
import hashlib, json, os, subprocess, sys, difflib, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emitted_globals_audit as G

run, head_patcher, out_json = sys.argv[1:4]
A, B = os.path.join(run, "A_head_plain", "files"), os.path.join(run, "B_fixed_plain", "files")
res = {"run_dir": run, "checks": {}}
def check(name, ok, detail=None):
    res["checks"][name] = {"pass": bool(ok), "detail": detail}; print(("PASS " if ok else "FAIL ") + name, "" if detail is None else json.dumps(detail)[:220])

eag = "vllm/v1/spec_decode/eagle.py"
ea, eb = G.audit(os.path.join(A, eag)), G.audit(os.path.join(B, eag))
orig_eagle = None
for cand in ("/tmp/none",):
    pass
# (1) HEAD emission defect
und_a = ea["undefined"]
check("head_eagle_exactly_one_unbound_global", set(und_a) == {"_FR13_FIXED32_MODE"}, sorted(und_a))
rec = und_a.get("_FR13_FIXED32_MODE", {})
check("head_eagle_unbound_in_propose", any(s.endswith(".propose") for s in rec.get("scopes", [])), rec.get("scopes"))
check("head_eagle_unbound_at_failed_boot_traceback_line_1174", 1174 in rec.get("lines", []), rec.get("lines"))
# (2) FIXED emission
check("fixed_eagle_no_unbound_globals", eb["undefined"] == {}, sorted(eb["undefined"]))
la, lb = open(os.path.join(A, eag)).read().splitlines(), open(os.path.join(B, eag)).read().splitlines()
d = [l for l in difflib.unified_diff(la, lb, lineterm="", n=0) if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))]
check("eagle_A_to_B_diff_is_single_added_definition", d == ["+_FR13_FIXED32_MODE = ''"], d)
# (3)/(4) whole-emission comparison
mods_a = sorted(open(os.path.join(run, "A_head_plain", "modified_files.txt")).read().split())
mods_b = sorted(open(os.path.join(run, "B_fixed_plain", "modified_files.txt")).read().split())
check("same_modified_file_set", mods_a == mods_b, {"n": len(mods_a), "only_a": sorted(set(mods_a) - set(mods_b)), "only_b": sorted(set(mods_b) - set(mods_a))})
regress, per_file = {}, {}
for rel in mods_b:
    ua = set(G.audit(os.path.join(A, rel))["undefined"]) if rel in mods_a else set()
    ub = set(G.audit(os.path.join(B, rel))["undefined"])
    per_file[rel] = {"A": sorted(ua), "B": sorted(ub)}
    if ub - ua: regress[rel] = sorted(ub - ua)
check("fixed_introduces_no_new_unbound_name_in_any_file", regress == {}, regress)
latent = per_file.get("vllm/model_executor/layers/mamba/gdn_linear_attn.py", {})
check("latent_fixed32_sfwd_names_identical_A_B", latent.get("A") == latent.get("B") and all(("fixed32_sfwd" in n) for n in latent.get("B", [])), latent)
res["per_file_unbound"] = per_file
# (5) declared edit reproduces the FIXED patcher
with tempfile.TemporaryDirectory() as td:
    fixed_p, rep_p = os.path.join(td, "fixed.py"), os.path.join(td, "rep.json")
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_fixed_patcher.py"), head_patcher, fixed_p, rep_p], check=True, capture_output=True)
    rep = json.load(open(rep_p))
    snap_fixed = os.path.join(run, "script_snapshot", "fr10_phase4_patch_vllm_tree_gdn.FIXED.py")
    check("declared_edit_reproduces_fixed_patcher_sha", rep["fixed_sha256"] == hashlib.sha256(open(snap_fixed, "rb").read()).hexdigest(),
          {"orig_sha256": rep["orig_sha256"], "fixed_sha256": rep["fixed_sha256"], "diff_lines": rep["diff_lines"]})
res["emitted_sha256"] = {"A_eagle": ea["sha256"], "B_eagle": eb["sha256"]}
res["patcher_sha256"] = {"head": rep["orig_sha256"], "fixed": rep["fixed_sha256"]}
res["audit_tool_sha256"] = hashlib.sha256(open(G.__file__, "rb").read()).hexdigest()
res["all_pass"] = all(c["pass"] for c in res["checks"].values())
json.dump(res, open(out_json, "w"), indent=2)
print("ALL PASS" if res["all_pass"] else "SOME FAILED"); sys.exit(0 if res["all_pass"] else 1)

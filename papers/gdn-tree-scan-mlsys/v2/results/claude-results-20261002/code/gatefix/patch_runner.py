#!/usr/bin/env python3
"""gatefix: runtime fix for the FR14 suffix pass gate (applied in the container after every production
patcher, through the q1v3 launch chain; no production source is modified).

Defect: on a gated step the depth-4/5 runner-up columns are filled with copies of their spine token.
Under sampled TAW the walk picks among identical children roughly uniformly, and the copies are
leaves, so ~2/3 of accepted depth-4 paths dead-end there (passive trace 2026-10-02: 57 of 76).
Fix: fill those columns with a token the model never samples in text chat (<|vision_pad|>), so the
walk can only take the spine child. Ungated steps are untouched (the edited block runs only when
_fr14_gate_fired). If /q1v3/HOOKS_ON exists, the passive trace hooks are installed afterwards."""
import glob, json, os, re, subprocess, sys

NEVER = 248055  # <|vision_pad|>
PAT = re.compile(r"_fr10_spine_tokens\[_fr14_pd\]\s*\.reshape\(-1,\s*1\)\s*\.repeat\(1,\s*_fr14_pw\)")
NEW = ("torch.cat([_fr10_spine_tokens[_fr14_pd].reshape(-1, 1), torch.full("
       "(_fr10_spine_tokens[_fr14_pd].numel(), _fr14_pw - 1), " + str(NEVER) + ", "
       "dtype=_fr10_spine_tokens[_fr14_pd].dtype, device=_fr10_spine_tokens[_fr14_pd].device)], dim=1)")
rec = {"schema": "v2exp.gatefix.v1", "never_token": NEVER, "applied": False, "files": []}
site = "/usr/local/lib/python3.12/dist-packages/vllm"
cands = [f for f in glob.glob(site + "/**/*.py", recursive=True) if "_fr14_pw" in open(f, errors="ignore").read()]
for f in cands:
    s = open(f).read()
    n = len(PAT.findall(s))
    rec["files"].append({"path": f, "matches": n})
    if n == 1:
        open(f, "w").write(PAT.sub(NEW, s))
        rec["applied"] = True
        rec["patched"] = f
total = sum(x["matches"] for x in rec["files"])
if total != 1:
    rec["applied"] = False
    rec["error"] = f"expected exactly one duplicate-fill site, found {total}"
os.makedirs("/q1run/CAND", exist_ok=True)
json.dump(rec, open("/q1run/CAND/gatefix_receipt.json", "w"), indent=1)
print("[gatefix]", json.dumps(rec), flush=True)
if not rec["applied"]:
    sys.exit(3)
if os.path.exists("/q1v3/HOOKS_ON"):
    sys.exit(subprocess.call([sys.executable, "/q1v3/patch_runner_q1v3.py"] + sys.argv[1:]))
json.dump({"applied": True, "problems": [], "mode": "gatefix-only"}, open("/q1run/CAND/patch_receipt.json", "w"))

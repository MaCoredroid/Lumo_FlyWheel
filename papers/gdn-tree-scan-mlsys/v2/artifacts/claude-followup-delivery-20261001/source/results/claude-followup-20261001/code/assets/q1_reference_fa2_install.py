#!/usr/bin/env python3
"""Install the PINNED PATCHED FA2 binary + the minimal compatible Python interface edit inside the spec-off reference container.

Implementation review R1 (q1-fullmodel-implementation-review.md): both native reference arms must load the SAME forked FA2 binary the
candidate uses (sha 28570f83..., size 300123792) through the ordinary one-token causal FLASH_ATTN backend.  This script, run once before
`vllm serve` in the container's ephemeral layer:
  1. verifies the mounted fork (sha + size), records the stock binary's identity, copies the fork over
     /usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so and re-verifies;
  2. applies ONLY `_patch_flash_attn_interface` from scripts/fr13_patch_fa2_tree_bias.py (imported, not executed as a script) to
     flash_attn_interface.py: the fork's `varlen_fwd` keeps the stock signature and the patched interface dispatches
     `varlen_fwd_tree_bias` only when a tree bias is passed (never on this reference); no tree-attention backend / committer / runner
     edits of the candidate patcher are imported;
  3. writes a receipt (before/after hashes, fork identity, interface diff summary) and exits non-zero on any mismatch.
The in-process boot attestation (hooks) must additionally prove the executed FA version is 2 and the fork symbols are loaded.
"""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

EXPECTED_FORK_SHA = "28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857"
EXPECTED_FORK_SIZE = 300123792
SP = Path("/usr/local/lib/python3.12/dist-packages/vllm")
SO = SP / "vllm_flash_attn" / "_vllm_fa2_C.abi3.so"
IFACE = SP / "vllm_flash_attn" / "flash_attn_interface.py"
EXPECTED_STOCK_IFACE_SHA = "9caca9584061cfd60c9ebd404d9a8881a62fe7894242efefcba5b88fc086d5b6"   # pinned image, stock bytes
EXPECTED_PATCHED_IFACE_SHA = "dadab8aff63b7f608274834929c954335247366411382f390af92501361044a1"   # deterministic result of _patch_flash_attn_interface on the stock bytes
PINNED_SOURCES = {"v1/attention/backends/fa_utils.py": "379bdccb69048f7bf6746816d146b383adceadef5e03bcd1ab1aa216ebbdfb27",
                  "v1/attention/backends/flash_attn.py": "c949e39a0b0b3dfa316017c422b1cb1368cd3a4a1328a1c691f5d7f44ed571e7",
                  "model_executor/layers/attention/attention.py": "1e5894a5ddc0dcbd343ba4b3e4c6d674d87604232dabede2aed1383a5ac4e713"}   # identity/native_source/fa2_source_index.json


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", default="/tmp/fr13_fork_fa2.so"); ap.add_argument("--patcher", default="/workspace/scripts/fr13_patch_fa2_tree_bias.py")
    ap.add_argument("--receipt", required=True); ap.add_argument("--expected-fork-sha256", default=EXPECTED_FORK_SHA)
    a = ap.parse_args()
    rc = {"schema": "lumo.q1.fullmodel.reference-fa2-install-receipt.v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "problems": []}
    if not os.path.exists(a.fork):
        rc["problems"].append(f"fork not mounted at {a.fork}")
    else:
        rc["fork"] = {"path": a.fork, "sha256": sha256_file(a.fork), "bytes": os.path.getsize(a.fork)}
        if rc["fork"]["sha256"] != a.expected_fork_sha256 or rc["fork"]["bytes"] != EXPECTED_FORK_SIZE:
            rc["problems"].append("fork sha/size != pinned")
    if SO.exists():
        rc["stock_binary_before"] = {"sha256": sha256_file(SO), "bytes": SO.stat().st_size}
    else:
        rc["problems"].append("stock FA2 binary path missing")
    if IFACE.exists():
        rc["interface_before_sha256"] = sha256_file(IFACE)
        if rc["interface_before_sha256"] != EXPECTED_STOCK_IFACE_SHA:
            rc["problems"].append("flash_attn_interface.py is not the pinned stock bytes before patching")
    else:
        rc["problems"].append("flash_attn_interface.py missing")
    if rc["problems"]:
        json.dump(rc, open(a.receipt, "w"), indent=1); print(json.dumps(rc)); return 3
    shutil.copyfile(a.fork, SO)
    rc["installed_binary"] = {"sha256": sha256_file(SO), "bytes": SO.stat().st_size}
    if rc["installed_binary"]["sha256"] != a.expected_fork_sha256:
        rc["problems"].append("installed binary sha != fork"); json.dump(rc, open(a.receipt, "w"), indent=1); return 4
    sys.path.insert(0, os.path.dirname(os.path.abspath(a.patcher)))
    import importlib.util
    spec = importlib.util.spec_from_file_location("fr13_patch_fa2_tree_bias", a.patcher); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rc["patcher"] = {"path": a.patcher, "sha256": sha256_file(a.patcher), "function": "_patch_flash_attn_interface"}
    changed = mod._patch_flash_attn_interface(IFACE)
    rc["interface_after_sha256"] = sha256_file(IFACE); rc["interface_changed"] = bool(changed)
    txt = IFACE.read_text()
    rc["interface_markers"] = {"tree_bias_param": "tree_bias=None" in txt, "chooses_tree_op_only_with_bias": "varlen_fwd_tree_bias\n            if tree_bias is not None" in txt,
                               "split_guard_helper": "_fr13_fa2_qrow32_b1_split2_interface_allowed" in txt}
    if not (changed and all(rc["interface_markers"].values())):
        rc["problems"].append("interface patch did not apply the expected minimal edits")
    if rc["interface_after_sha256"] != EXPECTED_PATCHED_IFACE_SHA:
        rc["problems"].append(f"patched interface sha {rc['interface_after_sha256'][:12]} != pinned expected {EXPECTED_PATCHED_IFACE_SHA[:12]}")
    rc["pinned_sources"] = {}
    for rel, want in PINNED_SOURCES.items():
        p = SP / rel; got = sha256_file(p) if p.exists() else None
        rc["pinned_sources"][rel] = {"sha256": got, "expected": want, "match": got == want}
        if got != want:
            rc["problems"].append(f"pinned source {rel} sha {str(got)[:12]} != {want[:12]}")
    json.dump(rc, open(a.receipt, "w"), indent=1); print(json.dumps({k: rc[k] for k in ("problems", "installed_binary", "interface_after_sha256", "interface_markers")}))
    return 0 if not rc["problems"] else 5


if __name__ == "__main__":
    sys.exit(main())

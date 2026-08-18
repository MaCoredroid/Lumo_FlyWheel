"""ARM S reachability probe — execute the DEPLOYED selector helpers, do not read them.

Question: at this HEAD, is there ANY route by which the `gqa_pair_splitk` kernel
produces SERVED generation tokens (the only thing a degenerate eyeball can read)?

Method: the qrow32-B1 selector helper block is a source blob the FA2 patcher
injects into vllm's tree_attn.py. It is real Python, so it is exec'd into a
namespace and the deployed predicates are DRIVEN, exactly as
suffix_pass_gating.md 11.4 drives the drafter blob on CPU.
"""
import json
import os
import sys
import types
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816")
sys.path.insert(0, str(REPO / "scripts"))

import fr13_patch_fa2_tree_bias as patcher  # noqa: E402

blob = patcher.FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS

# Minimal namespace: the block needs os and torch at import time only for
# attribute lookups inside function bodies we do not call.
fake_torch = types.SimpleNamespace(
    cuda=types.SimpleNamespace(
        is_available=lambda: False,
        is_current_stream_capturing=lambda: False,
        synchronize=lambda: None,
    ),
    uint8=None,
)
ns = {"os": os, "torch": fake_torch, "__name__": "fr13_qrow32_b1_selectors"}
compile(blob, "<FIXED32_QUERY_TILE32_B1_SELECTOR_HELPERS>", "exec")
exec(blob, ns)  # noqa: S102 -- driving the deployed blob is the point

out = {
    "schema": "fr14.promotion_ab.arm_s_reachability.v1",
    "head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
}

arms = ns["_FR13_FA2_QROW32_B1_ARMS"]
prod_arms = ns["_FR13_FA2_QROW32_B1_PRODUCTION_ARMS"]
out["arms_known"] = sorted(arms)
out["production_arms"] = list(prod_arms)
out["splitk_is_a_production_arm"] = "gqa_pair_splitk" in prod_arms
out["splitk_num_splits"] = int(arms["gqa_pair_splitk"]["num_splits"])

# ROUTE 1 -- production arm. This is the ONLY route where the deployed code sets
# candidate_served=True, i.e. the only route whose kernel output reaches a token.
os.environ["FR13_FA2_QROW32_B1_PRODUCTION_ARM"] = "gqa_pair_splitk"
try:
    ns["_fr13_fa2_qrow32_b1_arm"]("FR13_FA2_QROW32_B1_PRODUCTION_ARM")
    out["route_production"] = {"refused": False, "error": None}
except RuntimeError as exc:
    out["route_production"] = {"refused": True, "error": str(exc)}
os.environ.pop("FR13_FA2_QROW32_B1_PRODUCTION_ARM")

# ROUTE 2 -- live A/B arm. Accepted as an ARM NAME ...
os.environ["FR13_FA2_QROW32_B1_LIVE_AB_ARM"] = "gqa_pair_splitk"
try:
    got = ns["_fr13_fa2_qrow32_b1_arm"]("FR13_FA2_QROW32_B1_LIVE_AB_ARM")
    out["route_live_name_accepted"] = {"refused": False, "arm": got}
except RuntimeError as exc:
    out["route_live_name_accepted"] = {"refused": True, "error": str(exc)}
os.environ.pop("FR13_FA2_QROW32_B1_LIVE_AB_ARM")

# ... but the live route's FIRST act on every captured layer is the reduction
# check. The served reference is the qrow16 FULL graph at num_splits=0.
for ref in (0, 1, 2, 4):
    key = f"route_live_same_reduction_ref{ref}"
    try:
        ns["_fr13_fa2_qrow32_b1_require_same_reduction"]("gqa_pair_splitk", ref)
        out[key] = {"refused": False}
    except RuntimeError as exc:
        out[key] = {"refused": True, "error": str(exc)}

# CONTROL: the promoted arm passes the same check at the served topology, so the
# refusal above is specific to split-K, not an artifact of the probe.
try:
    ns["_fr13_fa2_qrow32_b1_require_same_reduction"]("gqa_pair", 0)
    out["control_gqa_pair_ref0"] = {"refused": False}
except RuntimeError as exc:
    out["control_gqa_pair_ref0"] = {"refused": True, "error": str(exc)}

out["verdict"] = (
    "UNREACHABLE: split-K is refused as a production arm (the only "
    "candidate_served=True route) and is refused by the live-A/B route's "
    "reduction-topology check against the served qrow16 reference. No served "
    "token at this HEAD can carry split-K attention."
    if (
        out["route_production"]["refused"]
        and out["route_live_same_reduction_ref0"]["refused"]
        and not out["control_gqa_pair_ref0"]["refused"]
    )
    else "REACHABLE — re-examine"
)
print(json.dumps(out, indent=2, sort_keys=True))

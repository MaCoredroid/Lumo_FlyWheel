#!/usr/bin/env python3
"""Generate the q1v3 copy of the deployed LumoTree serve-only chain (CPU only, writes only into --out).

Deployed chain (run_tree_arm.sh):  scripts/v2exp/promoab_tail10_serve_only.sh  (Cqc10 vehicle, REPO = port worktree)
  -> scripts/v2exp_serve_only_variant.sh -> $REPO/scripts/fr13_launch_forked_fa2_tree_server.sh  (docker run + vllm serve)

The generated copies differ from those sources ONLY by these exact-once substitutions (same idiom as
scripts/v2exp/make_*.py); every source is sha256-pinned and every anchor must match exactly once:
  promoab : run root -> $Q1V3_SERVE_ROOT;  variant path -> $Q1V3_VARIANT
  variant : SCRIPT_DIR -> the real v2exp scripts dir (the generated copy lives elsewhere);  launcher -> $Q1V3_LAUNCHER
  launcher: SCRIPT_DIR / gpu_oom_guard sibling -> the real port scripts dir;  +2 mounts (/q1v3 ro, /q1run rw) and
            +5 Q1V3_* env vars on docker run;  in the container, AFTER every production patcher and the fixed32
            runtime attestation and immediately before `exec vllm serve`:
              python3 /q1v3/patch_runner.py --mode cand --apply ...  &&  export PYTHONPATH=/q1v3:$PYTHONPATH
Serving flags, env pins, exact-pair checks (incl. KV_CACHE_MEMORY_BYTES == "" at B1), image, FA2 .so and
middleware are untouched. KV capacity of the candidate therefore stays GPU_UTIL-derived (unpinnable at B1).
"""
from __future__ import annotations

import argparse, hashlib, json, os, sys

V2EXP = "/home/mark/shared/lumotree-v2exp-20260930"
PORT = "/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816"
SOURCES = {
    "promoab": (f"{V2EXP}/scripts/v2exp/promoab_tail10_serve_only.sh", "dfc9558868f0739787f64fcce5ba099227cc95c17dff12f33a3973b48f40c024"),
    "variant": (f"{V2EXP}/scripts/v2exp_serve_only_variant.sh", "e614a8e93cbcc9c2f34f1da496055e536cab02ebd629a7545432e3c506616001"),
    "launcher": (f"{PORT}/scripts/fr13_launch_forked_fa2_tree_server.sh", "55c1b67fb4ae7e3c177638cbfcae2ec03ea17f6b49568e68a8676e700416d4b3"),
}
EDITS = {
    "promoab": [
        ("RUNROOT=/home/mark/shared/lumotree-v2exp-runs/tree/fr14_promoab_${ARM_KIND}${PROMOAB_ARM_SUFFIX:-}_$TS\n",
         "RUNROOT=${Q1V3_SERVE_ROOT:?}/fr14_promoab_${ARM_KIND}${PROMOAB_ARM_SUFFIX:-}_$TS\n"),
        ('  bash /home/mark/shared/lumotree-v2exp-20260930/scripts/v2exp_serve_only_variant.sh "$ARM" "$PROMOAB_KIND" "$SUBSET" \\\n',
         '  bash "${Q1V3_VARIANT:?}" "$ARM" "$PROMOAB_KIND" "$SUBSET" \\\n'),
    ],
    "variant": [
        ('SCRIPT_DIR=${FR13_SNAPSHOT_SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}\n',
         f"SCRIPT_DIR={V2EXP}/scripts  # q1v3: generated copy; siblings resolve in the real tree\n"),
        ('  scripts/fr13_launch_forked_fa2_tree_server.sh > "$ARMDIR/launch.log" 2>&1\n',
         '  bash "${Q1V3_LAUNCHER:?}" > "$ARMDIR/launch.log" 2>&1\n'),
    ],
    "launcher": [
        ('SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\n',
         f"SCRIPT_DIR={PORT}/scripts  # q1v3: generated copy; siblings resolve in the real tree\n"),
        ('      setsid bash "$(dirname "$0")/gpu_oom_guard.sh" \\\n      >/dev/null 2>&1 </dev/null &\n  else\n',
         '      setsid bash "$SCRIPT_DIR/gpu_oom_guard.sh" \\\n      >/dev/null 2>&1 </dev/null &\n  else\n'),
        ('    GPU_GUARD_NAME_GLOB="$CONTAINER" \\\n      setsid bash "$(dirname "$0")/gpu_oom_guard.sh" \\\n',
         '    GPU_GUARD_NAME_GLOB="$CONTAINER" \\\n      setsid bash "$SCRIPT_DIR/gpu_oom_guard.sh" \\\n'),
        ('  "${FR13_FIXED32_DOCKER_ARGS[@]}" \\\n',
         '  "${FR13_FIXED32_DOCKER_ARGS[@]}" \\\n'
         '  -v "${Q1V3_SRC:?}:/q1v3:ro" -v "${Q1V3_RUN:?}:/q1run" \\\n'
         '  -e Q1V3_ARM_KIND=cand -e Q1V3_ARM=CAND -e Q1V3_OUT=/q1run/CAND -e Q1V3_RUN=/q1run -e Q1V3_CASES=/q1v3/cases.json \\\n'),
        ("NSYS_PREFIX=()\n",
         "# q1v3: hook installation after all production patchers and the runtime attestation\n"
         "python3 /q1v3/patch_runner.py --mode cand --apply --receipt /q1run/CAND/patch_receipt.json\n"
         "export PYTHONPATH=/q1v3:\\${PYTHONPATH:-}\n"
         "NSYS_PREFIX=()\n"),
    ],
}
NAMES = {"promoab": "promoab_q1v3.sh", "variant": "variant_q1v3.sh", "launcher": "launcher_q1v3.sh"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def generate(out, sources=SOURCES, check_sha=True):
    os.makedirs(out, exist_ok=True)
    receipt = {"schema": "q1v3.cand-launch-generation.v1", "files": {}}
    for key, (path, want) in sources.items():
        raw = open(path, "rb").read()
        got = sha(raw)
        if check_sha and got != want:
            raise SystemExit(f"{key}: source {path} sha {got} != pinned {want}")
        text = raw.decode()
        for old, new in EDITS[key]:
            n = text.count(old)
            if n != 1:
                raise SystemExit(f"{key}: anchor occurs {n} times: {old[:70]!r}")
            text = text.replace(old, new)
        dst = os.path.join(out, NAMES[key])
        with open(dst, "w") as f:
            f.write(text)
        os.chmod(dst, 0o755)
        receipt["files"][key] = {"source": path, "source_sha256": got, "generated": dst, "generated_sha256": sha(text.encode()), "edits": len(EDITS[key])}
    with open(os.path.join(out, "GENERATION-RECEIPT.json"), "w") as f:
        json.dump(receipt, f, indent=1)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(generate(a.out), indent=1))

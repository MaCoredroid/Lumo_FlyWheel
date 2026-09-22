#!/usr/bin/env bash
# E7a: CPU-only dry-run of the HEAD patcher vs a FIXED copy inside the pinned image (no --gpus, no model, no server).
# Reuses the EXACT container env of a recorded failed boot (docker_inspect.json -> Config.Env) so the emitted sources
# match the intended plain-route configuration. Usage: patcher_dryrun.sh <out_dir> <recorded_boot_dir>
set -euo pipefail
WT=/home/mark/lumo-paper-v2-20260921; E=$WT/papers/gdn-tree-scan-mlsys/v2/experiments/e7a
OUT=$1; BOOT=$2; IMAGE="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
SCR=${SCRATCH:-/tmp/claude-1000/-home-mark-lumo-paper-v2-20260921/c8955ff9-f3ae-4551-869b-1a98b38ca2c3/scratchpad}/patcher_dryrun_$(basename "$OUT")
mkdir -p "$OUT/script_snapshot" "$SCR"
cp "$E/patcher_dryrun.sh" "$E/patcher_dryrun_incontainer.sh" "$E/emitted_globals_audit.py" "$E/make_fixed_patcher.py" "$E/e7a_capture_shim.py" "$OUT/script_snapshot/"
cp "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" "$OUT/script_snapshot/fr10_phase4_patch_vllm_tree_gdn.HEAD.py"
if grep -q 'E7A-FIX (2026-09-21)' "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py"; then
  # the worktree patcher already carries the declared fix (applied 2026-09-22T00:06Z): the FIXED copy IS the worktree file
  cp "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" "$OUT/script_snapshot/fr10_phase4_patch_vllm_tree_gdn.FIXED.py"
  git -C "$WT" show HEAD:scripts/fr10_phase4_patch_vllm_tree_gdn.py > "$OUT/script_snapshot/fr10_phase4_patch_vllm_tree_gdn.HEAD.py"
  echo "{\"note\": \"worktree patcher already fixed; HEAD copy from git, FIXED copy = worktree file\"}" > "$OUT/patcher_fix_report.json"
else
  python3 "$E/make_fixed_patcher.py" "$OUT/script_snapshot/fr10_phase4_patch_vllm_tree_gdn.HEAD.py" "$OUT/script_snapshot/fr10_phase4_patch_vllm_tree_gdn.FIXED.py" "$OUT/patcher_fix_report.json"
fi
( cd "$OUT/script_snapshot" && sha256sum * > SHA256SUMS )
git -C "$WT" rev-parse HEAD > "$OUT/worktree_head.txt"; git -C "$WT" hash-object "$WT/scripts/fr10_phase4_patch_vllm_tree_gdn.py" > "$OUT/patcher_worktree_blob.txt"
git -C "$WT" rev-parse HEAD:scripts/fr10_phase4_patch_vllm_tree_gdn.py > "$OUT/patcher_HEAD_blob.txt"
# exact recorded env of the failed boot (no credential-like keys present: verified; values copied verbatim)
python3 - "$BOOT/docker_inspect.json" "$OUT/container_env.list" <<'PY'
import json, sys, re
d = json.load(open(sys.argv[1])); d = d[0] if isinstance(d, list) else d
env = d["Config"]["Env"]; keep = []
for e in env:
    k = e.split("=", 1)[0]
    if re.search(r"(TOKEN|SECRET|PASS|CRED|AUTH|API_KEY)$", k) and not k.endswith("_NUM_TOKENS"):
        print("dropping credential-like key", k, file=sys.stderr); continue
    if "\n" in e: print("dropping multi-line value", k, file=sys.stderr); continue
    keep.append(e)
open(sys.argv[2], "w").write("\n".join(keep) + "\n"); print(f"{len(keep)} env vars copied from recorded boot")
PY
echo "source_boot=$BOOT" > "$OUT/env_source.txt"; sha256sum "$BOOT/docker_inspect.json" >> "$OUT/env_source.txt"
NAME="e7a-patcher-dryrun-$(date -u +%H%M%S)"; echo "$NAME" > "$OUT/container_name.txt"
docker image inspect "$IMAGE" --format '{{.Id}} {{.RepoDigests}}' > "$OUT/image_identity.txt"
echo "start $(date -u +%FT%TZ)" > "$OUT/timing.txt"
docker run --rm --name "$NAME" --entrypoint bash --env-file "$OUT/container_env.list" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" ${EXTRA_SHIM:+-e EXTRA_SHIM="$EXTRA_SHIM"} \
  -v "$WT:/workspace:ro" -v "$OUT/script_snapshot:/snap:ro" -v "$OUT:/out" -v "$SCR:/scratch" \
  "$IMAGE" /snap/patcher_dryrun_incontainer.sh > "$OUT/container_stdout.txt" 2>&1 || echo "container rc=$?" | tee -a "$OUT/timing.txt"
echo "end $(date -u +%FT%TZ)" >> "$OUT/timing.txt"; tail -25 "$OUT/dryrun.log"

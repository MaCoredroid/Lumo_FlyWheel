#!/usr/bin/env bash
# MARK'S DRAFTER-NEUTRALITY PROBE — plain MTP-5, same NVFP4 weights.
#
# THE QUESTION. 13236's thinking runaway appeared under our MERGED drafter (arctic
# tail + hydra tree). Mark's hypothesis: it was absent under plain MTP-5 on 3.6, so
# either our drafter machinery is implicated or the behaviour is 3.8-particular. This
# arm removes the drafter machinery and keeps everything else.
#
# ROUTE: KIND=nativemtp5, which the vehicle already defines (variant :415) as "STOCK
# vLLM native MTP-5 (qwen3_5_mtp, num_speculative_tokens=5, NO tree)... no forked-fa2,
# no tree_attn, no APC, NO IN-CONTAINER PATCHER". I am using the kind rather than
# hand-assembling SPEC_CONFIG/FR10_DECODE_MODE_DEFAULT, because the bdca0bd50 probe
# burned ten boots proving that reconstructing an environment from its shadow instead
# of its source is how you get a serve that is not the one you meant.
#
# WHY nativemtp5 AND NOT nativemtp5_exseed. The _exseed variant reaches the same
# decode path but runs LAUNCHER=forked to get the in-container patcher. For THIS
# question the patcher is part of what is under suspicion, so the arm that removes it
# entirely is the stronger test.
#
# WHAT THIS ARM DOES NOT CARRY, and why that is correct rather than a gap:
#   * no tier-B arm, no split-K, no credential -- LAUNCHER=native never reaches the
#     FA2 selector, so the whole apparatus is bypassed. NO RE-SEAL IS NEEDED for this
#     arm, and its provenance records the stack it actually ran.
#   * no fixed32 topology pins -- there is no tree.
# The coordinator's note that "split-K auto-disarms under DIAGNOSTIC=1" is true but
# understates it here: on the native launcher split-K is not in the picture at all.
#
# INSTRUMENTS. c5 IS NOT APPLICABLE and the artifact must say so: c5 is a SEAM
# conditional (accept[pos5]/accept[pos4]) and a chain drafter has no seam -- its
# per-position curve decays smoothly. Degeneration is read here from the trace screens
# (ttr, top-12-gram, max-block, tool cadence) plus the 24k ceiling's length events.
#
# Usage: MTP5_TASK=astropy13236 MTP5_REP=a bash promotion_ab_arm_mtp5.sh
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

MTP5_TASK=${MTP5_TASK:?set MTP5_TASK to a B1 diagnostic profile name}
MTP5_REP=${MTP5_REP:?set MTP5_REP to a replicate label}
case "$MTP5_TASK" in
  astropy13236) SUBSET=config/fr13_fixed32/subset_b1_diagnostic_astropy13236.json ;;
  astropy12907) SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json ;;
  *) echo "no B1 diagnostic profile for $MTP5_TASK -- see the spec: 14369 needs one minted first" >&2
     exit 2 ;;
esac
[[ -f "$SUBSET" && ! -L "$SUBSET" ]] || { echo "subset missing: $SUBSET" >&2; exit 2; }

ARM="nativemtp5_probe_${MTP5_TASK}_${MTP5_REP}"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT_ABS="$REPO/output/fr14_mtp5_${MTP5_TASK}_${MTP5_REP}_$TS"
export TAG=mtp5probe

# ---- gates ------------------------------------------------------------------
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ -z "$(docker ps -aq)" ]] || { echo "docker must be empty before boot" >&2; exit 2; }
sync; sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
PYTHONPATH="$REPO/src" .venv/bin/python -c \
  "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" \
  >/dev/null 2>&1 || true
_free=$(awk '/^MemFree:/{printf "%.1f", $2/1048576}' /proc/meminfo)
awk '/^MemFree:/{exit ($2/1048576 < 102.8)}' /proc/meminfo \
  || { echo "unified-memory preflight failed: MemFree=${_free}GiB < 102.8GiB" >&2; exit 2; }
echo "[mtp5] preflight OK: MemFree=${_free}GiB"

# ---- campaign env, SOURCED not reconstructed --------------------------------
export BSIZE=1 CONC=1 WALL=9000
export FR13_CAMPAIGN_TASK_BUDGET_S=9000
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant

mkdir -p "$RUNROOT_ABS"
.venv/bin/python results/fr14_nvfp4_port_20260816/promotion_ab_closure_watch.py \
  snapshot "$RUNROOT_ABS/closure.json" >/dev/null 2>&1 \
  || echo "[mtp5] WARNING: closure snapshot failed" >&2

{
  printf 'probe=drafter_neutrality\nhypothesis=plain MTP-5 removes the merged drafter; does the thinking runaway persist?\n'
  printf 'kind=nativemtp5 (LAUNCHER=native: stock vLLM MTP-5, qwen3_5_mtp, 5 tokens, NO tree, NO patcher)\n'
  printf 'task=%s\nreplicate=%s\nsubset=%s\n' "$MTP5_TASK" "$MTP5_REP" "$SUBSET"
  printf 'tier_b_arm=NONE (native launcher never reaches the FA2 selector)\n'
  printf 'split_k=NOT PRESENT (not merely disarmed)\ncredential=NOT REQUIRED (no re-seal for this arm)\n'
  printf 'c5_applicable=NO -- c5 is a SEAM conditional and a chain drafter has no seam\n'
  printf 'degeneration_instruments=trace screens (ttr, top-12-gram, max-block, tool cadence) + 24k ceiling length events\n'
  printf 'output_ceiling=%s\n' "${DEPLOY_MAX_OUTPUT_TOKENS:-24000}"
  printf 'boot_head=%s\nstarted=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
} > "$RUNROOT_ABS/MTP5_PROBE.txt"

echo "===== $ARM (drafter-neutrality probe) $(date -u +%FT%TZ) ====="
env RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=9000 \
  # NO FR13_FIXED32_B1_DIAGNOSTIC HERE, and this is not an omission.
  # The diagnostic route is a FIXED32 concept. The variant sets FIXED32_MODE only
  # for the fixed32 kinds (:519/:534), so on LAUNCHER=native it stays empty, the
  # --fixed32-* runner args are never passed (:2824), fixed32_enabled is False, and
  # run_swe_bench_q36_a.py:9981 hard-errors: 'FR13_FIXED32_B1_DIAGNOSTIC=1 requires
  # fixed32 runtime binding'. Single-task selection here comes from the SUBSET
  # alone. Verified by reading the gate rather than by burning a boot on it.
  FR10_METRICS=0 \
  TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" nativemtp5 "$SUBSET" \
  > "$RUNROOT_ABS/$ARM.runlog" 2>&1
rc=$?
echo "[mtp5] serve rc=$rc $(date -u +%FT%TZ)"
printf 'serve_rc=%s\nended=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/MTP5_PROBE.txt"
echo "[mtp5] containers after: $(docker ps -aq | wc -l)"
echo "[mtp5] done -> $RUNROOT_ABS"

#!/usr/bin/env bash
# Default-off dual real SWE-Verified B3+B4 raw-byte gate for GQA-pair FA2.
#
# FR13_FA2_QROW32_B34_PADDED -- Mark's ruling 2026-08-13. This is the width-4
# dual gate (scripts/fr13_run_b4_fa2_qrow32_gqa_pair_live_gate.sh) with ONE
# addition: the same two topology arms also gate the width-3 FULL graph, which
# reaches the sealed .so by PADDING to the canonical (b == 4, total_q == 128)
# geometry with a single inert shadow request in slot 3.
#
# What the added arm proves, per topology, on all 16 target tree layers:
#   * output[0:96] and softmax_lse[:, 0:96] are byte-identical between the
#     NATIVE width-3 stock call and the PADDED width-4 candidate call;
#   * the same real rows are byte-identical between a CLEAN shadow and a
#     POISONED shadow (NaN query rows + an impossible block-table page). At
#     seqused_k[3] == 0 the kernel reads neither, so any difference means the
#     n_block_min >= n_block_max early exit (flash_fwd_kernel.h:759) was not
#     taken -- which is exactly the thing the shadow doctrine rests on;
#   * the shadow half is precisely what that early return writes: zeros in its
#     32 O rows, +INF in its 32 LSE entries.
#
# CAVEAT 4 OF THE DE-RISK, CARRIED FORWARD SO THE TIMING ARM INHERITS IT. The
# width-3 wave-recovery fraction is 0.37 +/- 0.05, NOT 0.3682. The four-figure
# form came from a wave model whose own in-family residual is 10-15% (b=3 vs
# b=4 at equal wave count: +9.9% on medians, +15.4% on per-step totals). So the
# expected gross return is 25.6 +/- 3.5 ms/step (16 layers x 4.3441 ms/launch x
# 0.37 +/- 0.05), less 0.14-0.28 ms/step of staging and 0.023 ms/step of shadow
# stores. Even the pessimistic end clears the sealed four-pass MDE (4.20 ms
# hydra27 / 6.42 ms tail6) by 3-5x. These numbers are stamped into
# launcher_meta.txt below so the timing arm that follows this gate cannot
# quietly re-acquire the lost significant digits.
#
# NOTHING about the binary moves: the .so sha256, its size, the six per-file
# source digests and the C++ 33-clause TORCH_CHECK are all re-validated here
# unchanged, before and after. Zero rebuild, zero re-seal.
set -euo pipefail

case "${FR13_RUN_B34_QROW32_GQA_PAIR_LIVE_GATE:-0}" in
  1) ;;
  0)
    echo "B3+B4 qrow32 GQA-pair live gate is disabled" >&2
    echo "set FR13_RUN_B34_QROW32_GQA_PAIR_LIVE_GATE=1 to run it" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B34_QROW32_GQA_PAIR_LIVE_GATE must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
INNER=$(realpath "$SCRIPT_DIR/fr13_run_b4_fa2_qrow32_live_gate.sh")
GATE=$(realpath "$SCRIPT_DIR/fr13_fa2_qrow32_gqa_pair_gate.py")
PATCHER=$(realpath "$SCRIPT_DIR/fr13_patch_fa2_tree_bias.py")
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new output directory}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW32_GQA_PAIR_FA2_SO:?set QROW32_GQA_PAIR_FA2_SO to the pinned binary}"
: "${QROW32_GQA_PAIR_FA2_SOURCE:?set QROW32_GQA_PAIR_FA2_SOURCE to the regenerated FA2 source}"

PYTHON_BIN=${PYTHON_BIN:-/home/mark/lumoFlyWheel-b4-compactionfix-rerun/.venv/bin/python}
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81
SOURCE_COMMIT=$(git rev-parse HEAD)
RUNROOT_ABS=$(realpath -m "$RUNROOT")

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ "$QROW32_GQA_PAIR_FA2_SO" == /* \
   && -f "$QROW32_GQA_PAIR_FA2_SO" \
   && ! -L "$QROW32_GQA_PAIR_FA2_SO" ]] \
  || { echo "candidate SO must be an absolute regular non-symlink file" >&2; exit 2; }
[[ "$QROW32_GQA_PAIR_FA2_SOURCE" == /* \
   && -d "$QROW32_GQA_PAIR_FA2_SOURCE" \
   && ! -L "$QROW32_GQA_PAIR_FA2_SOURCE" ]] \
  || { echo "FA2 source must be an absolute non-symlink directory" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
  || { echo "source commit must be pushed before the dual gate" >&2; exit 2; }
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before the dual gate" >&2; exit 2; }

RUNNER_SHA256=$(sha256sum "$RUNNER" | awk '{print $1}')
INNER_SHA256=$(sha256sum "$INNER" | awk '{print $1}')
GATE_SHA256=$(sha256sum "$GATE" | awk '{print $1}')
PATCHER_SHA256=$(sha256sum "$PATCHER" | awk '{print $1}')

mkdir -p "$RUNROOT_ABS"
"$PYTHON_BIN" "$GATE" validate-candidate \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  > "$RUNROOT_ABS/candidate_identity.at_launch.json"

printf 'classification=real_swe_verified_exact4_b34_fa2_qrow32_gqa_pair_dual_byte_gate\nacceptance_valid=0\ntiming_eligible=0\nproduction_enabled=0\nreference_always_served=1\nqualified_topologies=Tail23,Hydra27\ntask_count_per_topology=4\ngated_widths=3,4\ncanonical_width=4\nexpected_width3_recovery_fraction=0.37\nexpected_width3_recovery_fraction_sigma=0.05\nexpected_width3_gross_ms_per_step=25.6\nexpected_width3_gross_ms_per_step_sigma=3.5\nstaging_cost_ms_per_step_range=0.14-0.28\nstaging_footprint_bytes=3145728\nshadow_slot=3\nshadow_seqused_k=0\npoisoned_shadow_arm=1\nbatch_size=4\nconcurrency=4\nphysical_rows_per_slot=32\ndraft_vocab_k=65536\ndraft_vocab_root=1\nfa2_head=%s\nfa2_source_closure_sha256=%s\nsource_commit=%s\nrunner_sha256=%s\ninner_runner_sha256=%s\ngate_sha256=%s\npatcher_sha256=%s\nlauncher_pid=%s\nstarted=%s\n' \
  "$FA2_HEAD" "$SOURCE_CLOSURE_SHA256" "$SOURCE_COMMIT" \
  "$RUNNER_SHA256" "$INNER_SHA256" "$GATE_SHA256" "$PATCHER_SHA256" \
  "$$" "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"

run_arm() {
  local mode=$1
  local label=$2
  local arm_tag="${TAG}_${label}"
  local arm_root="$RUNROOT_ABS/$label"
  local arm_dir="$arm_root/${mode}_fa2_qrow32_gate_${arm_tag}"

  env \
    RUNROOT="$arm_root" \
    TAG="$arm_tag" \
    FORKED_FA2_SO="$QROW32_GQA_PAIR_FA2_SO" \
    PYTHON_BIN="$PYTHON_BIN" \
    FR13_QROW32_FIXED32_MODE="$mode" \
    FR13_QROW32_LIVE_AB_ARM=gqa_pair \
    FR13_QROW32_LIVE_AB_B3=1 \
    FR13_QROW32_FA2_HEAD="$FA2_HEAD" \
    FR13_QROW32_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
    bash "$INNER"

  # The width-4 arm, verbatim the sealed lineage.
  "$PYTHON_BIN" "$GATE" verify-arm \
    --result "$arm_dir/logs/fr13_fa2_qrow32_live_paged_ab.json" \
    --campaign-arm "$arm_dir/swe_out/verified/campaign_summary.json" \
    --campaign-provenance "$arm_dir/swe_out/verified/fixed32_qwen_campaign_provenance.json" \
    --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
    --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
    --fixed32-mode "$mode" \
    --source-commit "$SOURCE_COMMIT" \
    > "$RUNROOT_ABS/${label}_verification.json"

  # The width-3 PADDED arm, including the poisoned-shadow clauses. The result
  # lands in its own file, so neither arm can overwrite the other and a
  # missing width-3 replay is a MISSING FILE, not a silent pass.
  "$PYTHON_BIN" "$GATE" verify-arm-b3 \
    --result "$arm_dir/logs/fr13_fa2_qrow32_live_paged_ab_b3.json" \
    --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
    --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
    --fixed32-mode "$mode" \
    --source-commit "$SOURCE_COMMIT" \
    > "$RUNROOT_ABS/${label}_b3_verification.json"
}

run_arm tail6_fixed32 tail23
run_arm hydra27_fixed32 hydra27

"$PYTHON_BIN" "$GATE" verify-dual \
  --tail-verification "$RUNROOT_ABS/tail23_verification.json" \
  --hydra-verification "$RUNROOT_ABS/hydra27_verification.json" \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  --source-commit "$SOURCE_COMMIT" \
  > "$RUNROOT_ABS/dual_gate_verification.json"

"$PYTHON_BIN" "$GATE" validate-candidate \
  --candidate-so "$QROW32_GQA_PAIR_FA2_SO" \
  --fa2-source "$QROW32_GQA_PAIR_FA2_SOURCE" \
  > "$RUNROOT_ABS/candidate_identity.at_end.json"
cmp -s "$RUNROOT_ABS/candidate_identity.at_launch.json" \
  "$RUNROOT_ABS/candidate_identity.at_end.json" \
  || { echo "candidate binary/source identity changed during the dual gate" >&2; exit 14; }
[[ "$(sha256sum "$RUNNER" | awk '{print $1}')" == "$RUNNER_SHA256" \
   && "$(sha256sum "$INNER" | awk '{print $1}')" == "$INNER_SHA256" \
   && "$(sha256sum "$GATE" | awk '{print $1}')" == "$GATE_SHA256" \
   && "$(sha256sum "$PATCHER" | awk '{print $1}')" == "$PATCHER_SHA256" ]] \
  || { echo "dual gate source changed during execution" >&2; exit 14; }
[[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" \
   && "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" \
   && -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "frozen pushed source changed during the dual gate" >&2; exit 14; }

printf 'ended=%s\nstatus=PASS\n' "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

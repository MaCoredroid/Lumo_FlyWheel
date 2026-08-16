#!/usr/bin/env bash
# PASS-gated real SWE-Verified exact4 timing for Hydra27 qrow32 no-split.
# Despite the retained historical filename, this isolates tree attention in FULL graph mode.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
RUNNER_PATH=$(realpath "$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")")
cd "$REPO"

case "${FR13_RUN_QROW32_NOSPLIT_TIMING:-0}" in
  1) ;;
  0)
    echo "qrow32 no-split timing is disabled; set FR13_RUN_QROW32_NOSPLIT_TIMING=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_QROW32_NOSPLIT_TIMING must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

COMPOSED_STACK=${FR13_B1_COMPOSED_STACK_TIMING:-0}
case "$COMPOSED_STACK" in
  0|1) ;;
  *) echo "FR13_B1_COMPOSED_STACK_TIMING must be exactly 0 or 1" >&2; exit 2 ;;
esac
CFWD_PRODUCTION=${FR13_B1_COMPOSED_CFWD_PRODUCTION:-0}
case "$CFWD_PRODUCTION" in
  0|1) ;;
  *) echo "FR13_B1_COMPOSED_CFWD_PRODUCTION must be exactly 0 or 1" >&2; exit 2 ;;
esac
PRODUCTION_SMOKE=${FR13_B1_COMPOSED_CFWD_SMOKE:-0}
case "$PRODUCTION_SMOKE" in
  0|1) ;;
  *) echo "FR13_B1_COMPOSED_CFWD_SMOKE must be exactly 0 or 1" >&2; exit 2 ;;
esac
if [[ "$CFWD_PRODUCTION" == "1" && "$COMPOSED_STACK" != "1" ]]; then
  echo "composed CFWD production requires the composed B1 stack" >&2
  exit 2
fi
if [[ "$PRODUCTION_SMOKE" == "1" && "$CFWD_PRODUCTION" != "1" ]]; then
  echo "composed CFWD smoke requires composed CFWD production" >&2
  exit 2
fi

: "${RUNROOT:?set RUNROOT to a new path under output/}"
: "${TAG:?set TAG to a unique run tag}"
: "${QROW32_B1_FA2_SO:?set QROW32_B1_FA2_SO to the pinned combined binary}"
: "${QROW32_B1_FA2_SOURCE:?set QROW32_B1_FA2_SOURCE to the pinned FA2 source closure}"
: "${QROW32_B1_PASS:?set QROW32_B1_PASS to the qrow32 nosplit real-task live PASS}"
: "${QROW32_B1_PASS_SHA256:?set QROW32_B1_PASS_SHA256 to its raw SHA-256}"
if [[ "$COMPOSED_STACK" == "1" ]]; then
  : "${QROW32_B1_COMPOSED_CREDENTIAL:?set the Gate-A Qrow32 composed credential}"
  : "${QROW32_B1_COMPOSED_CREDENTIAL_SHA256:?set its raw SHA-256}"
  : "${GQA3_PASS:?set the Gate-A GQA3 production credential}"
  : "${GQA3_PASS_SHA256:?set its raw SHA-256}"
  : "${DFWD_TOP3_SO:?set the pinned DFWD K64 top3 binary}"
  : "${DFWD_TOP3_CREDENTIAL:?set the Gate-A DFWD top3 credential}"
  : "${DFWD_TOP3_CREDENTIAL_SHA256:?set its raw SHA-256}"
  : "${DFWD_TOP3_BUILD_ATTESTATION:?set the pinned DFWD build attestation}"
  : "${CUTLASS_TARGET_SO:?set the pinned wide256 full-grid target binary}"
  : "${CUTLASS_TARGET_PASS:?set the Gate-B target live PASS}"
  : "${CUTLASS_TARGET_PASS_SHA256:?set its raw SHA-256}"
  : "${SFWD_CONV_POSTPREP_PASS:?set the Gate-B SFWD production PASS}"
  : "${SFWD_CONV_POSTPREP_PASS_SHA256:?set its raw SHA-256}"
  : "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST:?set the Gate-B SFWD source manifest}"
  : "${SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256:?set its raw SHA-256}"
  : "${TARGET_SFWD_COMBINED_SUMMARY:?set the Gate-B combined target/SFWD summary}"
  : "${TARGET_SFWD_COMBINED_SUMMARY_SHA256:?set its raw SHA-256}"
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  : "${TAW_B1_CREDENTIAL:?set the source-bound Hydra27 B1 TAW credential}"
  : "${TAW_B1_CREDENTIAL_SHA256:?set its raw SHA-256}"
  : "${TAW_B1_LIVE_BUNDLE:?set the credentialed Hydra27 B1 TAW replay}"
  : "${TAW_B1_LIVE_BUNDLE_SHA256:?set its raw SHA-256}"
  : "${TAW_REVIEWED_B4_PASS:?set the reviewed Hydra27 exact4 TAW bundle}"
  : "${TAW_REVIEWED_B4_PASS_SHA256:?set its raw SHA-256}"
  : "${TAW_REVIEWED_B4_VERDICT:?set the reviewed Hydra27 exact4 TAW verdict}"
  : "${TAW_REVIEWED_B4_VERDICT_SHA256:?set its raw SHA-256}"
  : "${TAW_MERGE_BINDING:?set the Hydra27 B1/B4 TAW merge binding}"
  : "${TAW_MERGE_BINDING_SHA256:?set its raw SHA-256}"
  : "${TAW_PASS_JSON:?set the merged Hydra27 TAW production bundle}"
  : "${TAW_PASS_SHA256:?set its raw SHA-256}"
  : "${CFWD_PASS_JSON:?set the source-bound CFWD production credential}"
  : "${CFWD_PASS_SHA256:?set its raw SHA-256}"
  if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
    : "${COMPOSED_CFWD_SMOKE_PASS:?set the prior one-task production smoke credential}"
    : "${COMPOSED_CFWD_SMOKE_PASS_SHA256:?set its raw SHA-256}"
  fi
fi

PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
EXACT4_SUBSET=config/fr13_fixed32/subset_b4_four.json
EXACT4_SUBSET_SHA256=0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5
ONE_TASK_SUBSET=config/fr13_fixed32/subset_b1_diagnostic_one.json
ONE_TASK_SUBSET_SHA256=cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb
SUBSET=$EXACT4_SUBSET
SUBSET_SHA256=$EXACT4_SUBSET_SHA256
TASK_COUNT=4
TIMING_ELIGIBLE=1
if [[ "$PRODUCTION_SMOKE" == "1" ]]; then
  SUBSET=$ONE_TASK_SUBSET
  SUBSET_SHA256=$ONE_TASK_SUBSET_SHA256
  TASK_COUNT=1
  TIMING_ELIGIBLE=0
fi
BLOCK_MAP=scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_CONTAINER=/workspace/scripts/fr13_dvk_subset_blocks.json
BLOCK_MAP_SHA256=85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff
CANDIDATE_SHA256=a9d8a6887b8b27b3a83af60bba7945eb66caff174ba710c2ee2aea92b8e7081a
CANDIDATE_BYTES=300154616
TARGET_SHA256=7d762dfa793671d75d1e353bd37d76fc07370cbe387ad1e315e32584d27927d4
TARGET_BYTES=119781296
TARGET_SELECTOR=identity_wide256_fullgrid_b1
DFWD_TOP3_SHA256=c0ed75cafdd926eceafcf28671869d54f37addb51bfef5a37c0b07c34f5420ff
DFWD_TOP3_BYTES=159288
FA2_HEAD=29210221863736a08f71a866459e368ad1ac4a95
SOURCE_CLOSURE_SHA256=22b8c2016443a151bf50f62166f7cc3b9ce45137138d948b76fdfded74c395ff
BASELINE=$REPO/results/fr13_fixed32_qrow16_prod_exact4_b1_20260731T182827Z/hydra_valid/deploy_speed_qrow16_prod_exact4_b1_20260731T182827Z.json
BASELINE_SHA256=0350e791bc825083bfc3635e11c875617fa1d3823eba5f93ebd7f392c50f18d0
MANDATORY_WEIGHT_BYTES=27977022848
MANDATORY_WEIGHT_FLOOR_MS=102.479937172
ONE_SIDED_U95_CAP_MS=117.8519277478
SEQUENCE=scripts/fr13_fixed32_floor_timers_seq.sh
SOURCE_COMMIT=$(git rev-parse HEAD)
PATCH_SOURCE_SHA256=$(sha256sum scripts/fr13_patch_fa2_tree_bias.py | awk '{print $1}')
RUNNER_SHA256=$(sha256sum "$RUNNER_PATH" | awk '{print $1}')
RUNROOT_ABS=$(realpath -m "$RUNROOT")
RUNROOT_REL=${RUNROOT_ABS#"$REPO/"}
if [[ "$COMPOSED_STACK" == "1" ]]; then
  if [[ "$PRODUCTION_SMOKE" == "1" ]]; then
    ARM="hydra27_fixed32_k64_composed_cfwd_production_smoke_${TAG}"
  elif [[ "$CFWD_PRODUCTION" == "1" ]]; then
    ARM="hydra27_fixed32_k64_composed_qrow32_gqa3_dfwd3_target_sfwd_cfwd_exact4_${TAG}"
  else
    ARM="hydra27_fixed32_k64_composed_qrow32_gqa3_dfwd3_target_sfwd_exact4_${TAG}"
  fi
else
  ARM="hydra27_fixed32_k64_qrow32_nosplit_exact4_${TAG}"
fi
ARMDIR="$RUNROOT_ABS/$ARM"

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] \
  || { echo "TAG contains unsafe characters" >&2; exit 2; }
[[ "$RUNROOT_ABS" == "$REPO/output/"* ]] \
  || { echo "RUNROOT must resolve below $REPO/output" >&2; exit 2; }
[[ ! -e "$RUNROOT_ABS" && ! -L "$RUNROOT_ABS" ]] \
  || { echo "RUNROOT must be new: $RUNROOT_ABS" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] \
  || { echo "Python environment is unavailable: $PYTHON_BIN" >&2; exit 2; }
for required in "$QROW32_B1_FA2_SO" "$QROW32_B1_PASS"; do
  [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
    || { echo "required input must be an absolute regular file: $required" >&2; exit 2; }
done
unset required
if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
  [[ "$BASELINE" == /* && -f "$BASELINE" && ! -L "$BASELINE" ]] \
    || { echo "baseline must be an absolute regular file: $BASELINE" >&2; exit 2; }
fi
if [[ "$COMPOSED_STACK" == "1" ]]; then
  for required in \
      "$QROW32_B1_COMPOSED_CREDENTIAL" "$GQA3_PASS" \
      "$DFWD_TOP3_SO" "$DFWD_TOP3_CREDENTIAL" \
      "$DFWD_TOP3_BUILD_ATTESTATION" "$CUTLASS_TARGET_SO" \
      "$CUTLASS_TARGET_PASS" "$SFWD_CONV_POSTPREP_PASS" \
      "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST" \
      "$TARGET_SFWD_COMBINED_SUMMARY"; do
    [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
      || { echo "composed input must be an absolute regular file: $required" >&2; exit 2; }
  done
  unset required
  SFWD_PASS_ABS=$(realpath "$SFWD_CONV_POSTPREP_PASS")
  SFWD_MANIFEST_ABS=$(realpath "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST")
  [[ "$SFWD_PASS_ABS" == "$REPO/"* && "$SFWD_MANIFEST_ABS" == "$REPO/"* ]] \
    || { echo "SFWD PASS and source manifest must resolve inside the repository" >&2; exit 2; }
  SFWD_PASS_CONTAINER="/workspace/${SFWD_PASS_ABS#"$REPO/"}"
  SFWD_MANIFEST_CONTAINER="/workspace/${SFWD_MANIFEST_ABS#"$REPO/"}"
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  for required in \
      "$TAW_B1_CREDENTIAL" "$TAW_B1_LIVE_BUNDLE" \
      "$TAW_REVIEWED_B4_PASS" "$TAW_REVIEWED_B4_VERDICT" \
      "$TAW_MERGE_BINDING" "$TAW_PASS_JSON" "$CFWD_PASS_JSON"; do
    [[ "$required" == /* && -f "$required" && ! -L "$required" ]] \
      || { echo "TAW/CFWD input must be an absolute regular file: $required" >&2; exit 2; }
  done
  unset required
  if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
    [[ "$COMPOSED_CFWD_SMOKE_PASS" == /* \
       && -f "$COMPOSED_CFWD_SMOKE_PASS" \
       && ! -L "$COMPOSED_CFWD_SMOKE_PASS" ]] \
      || { echo "production smoke credential must be an absolute regular file" >&2; exit 2; }
  fi
fi
[[ "$QROW32_B1_FA2_SOURCE" == /* \
   && -d "$QROW32_B1_FA2_SOURCE" \
   && ! -L "$QROW32_B1_FA2_SOURCE" ]] \
  || { echo "QROW32_B1_FA2_SOURCE must be an absolute non-symlink directory" >&2; exit 2; }
[[ "$(stat -c '%s' "$QROW32_B1_FA2_SO")" == "$CANDIDATE_BYTES" \
   && "$(sha256sum "$QROW32_B1_FA2_SO" | awk '{print $1}')" == "$CANDIDATE_SHA256" \
   && "$(sha256sum "$QROW32_B1_PASS" | awk '{print $1}')" == "$QROW32_B1_PASS_SHA256" \
   && "$(sha256sum "$SUBSET" | awk '{print $1}')" == "$SUBSET_SHA256" \
   && "$(sha256sum "$EXACT4_SUBSET" | awk '{print $1}')" == "$EXACT4_SUBSET_SHA256" \
   && "$(sha256sum "$ONE_TASK_SUBSET" | awk '{print $1}')" == "$ONE_TASK_SUBSET_SHA256" \
   && "$(sha256sum "$BLOCK_MAP" | awk '{print $1}')" == "$BLOCK_MAP_SHA256" ]] \
  || { echo "qrow32 exact4 timing prerequisite identity drifted" >&2; exit 2; }
if [[ "$PRODUCTION_SMOKE" == "0" \
      && "$(sha256sum "$BASELINE" | awk '{print $1}')" != "$BASELINE_SHA256" ]]; then
  echo "qrow16 historical baseline identity drifted" >&2
  exit 2
fi
if [[ "$COMPOSED_STACK" == "1" ]]; then
  [[ "$(stat -c '%s' "$DFWD_TOP3_SO")" == "$DFWD_TOP3_BYTES" \
     && "$(sha256sum "$DFWD_TOP3_SO" | awk '{print $1}')" == "$DFWD_TOP3_SHA256" \
     && "$(stat -c '%s' "$CUTLASS_TARGET_SO")" == "$TARGET_BYTES" \
     && "$(sha256sum "$CUTLASS_TARGET_SO" | awk '{print $1}')" == "$TARGET_SHA256" \
     && "$(sha256sum "$QROW32_B1_COMPOSED_CREDENTIAL" | awk '{print $1}')" == "$QROW32_B1_COMPOSED_CREDENTIAL_SHA256" \
     && "$(sha256sum "$GQA3_PASS" | awk '{print $1}')" == "$GQA3_PASS_SHA256" \
     && "$(sha256sum "$DFWD_TOP3_CREDENTIAL" | awk '{print $1}')" == "$DFWD_TOP3_CREDENTIAL_SHA256" \
     && "$(sha256sum "$CUTLASS_TARGET_PASS" | awk '{print $1}')" == "$CUTLASS_TARGET_PASS_SHA256" \
     && "$(sha256sum "$SFWD_PASS_ABS" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
     && "$(sha256sum "$SFWD_MANIFEST_ABS" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" ]] \
    || { echo "composed B1 credential or binary identity drifted" >&2; exit 2; }
  [[ "$(sha256sum "$TARGET_SFWD_COMBINED_SUMMARY" | awk '{print $1}')" \
       == "$TARGET_SFWD_COMBINED_SUMMARY_SHA256" ]] \
    || { echo "combined target/SFWD summary identity drifted" >&2; exit 2; }
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  [[ "$TAW_B1_CREDENTIAL_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$TAW_B1_CREDENTIAL" | awk '{print $1}')" == "$TAW_B1_CREDENTIAL_SHA256" \
     && "$TAW_B1_LIVE_BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$TAW_B1_LIVE_BUNDLE" | awk '{print $1}')" == "$TAW_B1_LIVE_BUNDLE_SHA256" \
     && "$TAW_REVIEWED_B4_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$TAW_REVIEWED_B4_PASS" | awk '{print $1}')" == "$TAW_REVIEWED_B4_PASS_SHA256" \
     && "$TAW_REVIEWED_B4_VERDICT_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$TAW_REVIEWED_B4_VERDICT" | awk '{print $1}')" == "$TAW_REVIEWED_B4_VERDICT_SHA256" \
     && "$TAW_MERGE_BINDING_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$TAW_MERGE_BINDING" | awk '{print $1}')" == "$TAW_MERGE_BINDING_SHA256" \
     && "$TAW_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$TAW_PASS_JSON" | awk '{print $1}')" == "$TAW_PASS_SHA256" \
     && "$CFWD_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
     && "$(sha256sum "$CFWD_PASS_JSON" | awk '{print $1}')" == "$CFWD_PASS_SHA256" ]] \
    || { echo "TAW/CFWD credential identity drifted" >&2; exit 2; }
  if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
    [[ "$COMPOSED_CFWD_SMOKE_PASS_SHA256" =~ ^[0-9a-f]{64}$ \
       && "$(sha256sum "$COMPOSED_CFWD_SMOKE_PASS" | awk '{print $1}')" \
          == "$COMPOSED_CFWD_SMOKE_PASS_SHA256" ]] \
      || { echo "composed CFWD smoke credential identity drifted" >&2; exit 2; }
  fi
fi
CFWD_COMPONENT_HASH_ARGS=()
CFWD_COMPONENT_ARGS=()
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  CFWD_COMPONENT_HASH_ARGS=(
    --qrow-credential-sha256 "$QROW32_B1_COMPOSED_CREDENTIAL_SHA256"
    --gdn-credential-sha256 "$GQA3_PASS_SHA256"
    --dfwd-credential-sha256 "$DFWD_TOP3_CREDENTIAL_SHA256"
    --target-live-sha256 "$CUTLASS_TARGET_PASS_SHA256"
    --sfwd-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256"
    --source-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256"
    --combined-summary-sha256 "$TARGET_SFWD_COMBINED_SUMMARY_SHA256"
    --taw-b1-credential-sha256 "$TAW_B1_CREDENTIAL_SHA256"
    --taw-b1-live-bundle-sha256 "$TAW_B1_LIVE_BUNDLE_SHA256"
    --taw-b4-pass-sha256 "$TAW_REVIEWED_B4_PASS_SHA256"
    --taw-b4-verdict-sha256 "$TAW_REVIEWED_B4_VERDICT_SHA256"
    --taw-merge-binding-sha256 "$TAW_MERGE_BINDING_SHA256"
    --taw-production-sha256 "$TAW_PASS_SHA256"
    --cfwd-credential-sha256 "$CFWD_PASS_SHA256"
  )
  CFWD_COMPONENT_ARGS=(
    --qrow-credential "$QROW32_B1_COMPOSED_CREDENTIAL"
    --gdn-credential "$GQA3_PASS"
    --dfwd-credential "$DFWD_TOP3_CREDENTIAL"
    --target-live "$CUTLASS_TARGET_PASS"
    --sfwd-pass "$SFWD_PASS_ABS"
    --source-manifest "$SFWD_MANIFEST_ABS"
    --combined-summary "$TARGET_SFWD_COMBINED_SUMMARY"
    --taw-b1-credential "$TAW_B1_CREDENTIAL"
    --taw-b1-live-bundle "$TAW_B1_LIVE_BUNDLE"
    --taw-b4-pass "$TAW_REVIEWED_B4_PASS"
    --taw-b4-verdict "$TAW_REVIEWED_B4_VERDICT"
    --taw-merge-binding "$TAW_MERGE_BINDING"
    --taw-production "$TAW_PASS_JSON"
    --cfwd-credential "$CFWD_PASS_JSON"
    "${CFWD_COMPONENT_HASH_ARGS[@]}"
  )
fi
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
if [[ "$COMPOSED_STACK" == "1" ]]; then
  [[ "$(git rev-parse '@{upstream}')" == "$SOURCE_COMMIT" ]] \
    || { echo "composed timing source commit must be pushed to upstream" >&2; exit 2; }
fi

"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py validate-source \
  --source-root "$QROW32_B1_FA2_SOURCE" >/dev/null
"$PYTHON_BIN" - \
  "$QROW32_B1_PASS" "$CANDIDATE_SHA256" "$SOURCE_COMMIT" \
  "$PATCH_SOURCE_SHA256" "$QROW32_B1_FA2_SO" <<'PY'
import sys
from pathlib import Path

from scripts import fr13_qrow32_b1_pass_sidecar as qrow

payload, _ = qrow.load_json(Path(sys.argv[1]))
qrow.validate_live_result(
    payload,
    candidate_sha256=sys.argv[2],
    arm=qrow.ARM,
    source_commit=sys.argv[3],
    patch_source_sha256=sys.argv[4],
)
qrow.validate_candidate(Path(sys.argv[5]), sys.argv[2])
qrow.validate_patch_source(
    Path("scripts/fr13_patch_fa2_tree_bias.py"),
    expected_source_commit=sys.argv[3],
)
PY
if [[ "$COMPOSED_STACK" == "1" ]]; then
  "$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py validate-graph-credentials \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --qrow-live "$QROW32_B1_PASS" \
    --qrow-live-sha256 "$QROW32_B1_PASS_SHA256" \
    --qrow-credential "$QROW32_B1_COMPOSED_CREDENTIAL" \
    --qrow-credential-sha256 "$QROW32_B1_COMPOSED_CREDENTIAL_SHA256" \
    --gdn-credential "$GQA3_PASS" \
    --gdn-credential-sha256 "$GQA3_PASS_SHA256" \
    --dfwd-credential "$DFWD_TOP3_CREDENTIAL" \
    --dfwd-credential-sha256 "$DFWD_TOP3_CREDENTIAL_SHA256" \
    --candidate-so "$DFWD_TOP3_SO" \
    --build-attestation "$DFWD_TOP3_BUILD_ATTESTATION" >/dev/null
  "$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py validate-eager-credentials \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --combined-summary "$TARGET_SFWD_COMBINED_SUMMARY" \
    --combined-summary-sha256 "$TARGET_SFWD_COMBINED_SUMMARY_SHA256" \
    --target-live "$CUTLASS_TARGET_PASS" \
    --target-live-sha256 "$CUTLASS_TARGET_PASS_SHA256" \
    --sfwd-pass "$SFWD_PASS_ABS" \
    --sfwd-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256" \
    --source-manifest "$SFWD_MANIFEST_ABS" \
    --source-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" >/dev/null
  "$PYTHON_BIN" scripts/fr13_gdn_gqa_group3_production_credential.py \
    --credential "$GQA3_PASS" \
    --source-commit "$SOURCE_COMMIT" \
    --profile fixed32 \
    --mode hydra27_fixed32 \
    --batch 1
  "$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py validate \
    --live-result "$CUTLASS_TARGET_PASS" \
    --expected-live-sha256 "$CUTLASS_TARGET_PASS_SHA256" \
    --candidate-so "$CUTLASS_TARGET_SO" \
    --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py \
    --expected-source-commit "$SOURCE_COMMIT" \
    --candidate-selector "$TARGET_SELECTOR" \
    --qualification-profile k64_root \
    --diagnostic-task-profile astropy12907 \
    --draft-vocab-blocks "$BLOCK_MAP" >/dev/null
  "$PYTHON_BIN" scripts/fr13_sfwd_conv_postprep_gate.py validate-pass \
    --repo "$REPO" \
    --live-pass "$SFWD_PASS_ABS" \
    --expected-live-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256" \
    --source-manifest "$SFWD_MANIFEST_ABS" \
    --expected-source-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
    --source-commit "$SOURCE_COMMIT"
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  "$PYTHON_BIN" scripts/fr13_cfwd_logit_direct_gate.py validate \
    --credential "$CFWD_PASS_JSON" \
    --expected-sha256 "$CFWD_PASS_SHA256" \
    --source-commit "$SOURCE_COMMIT" \
    --timing-subset "$EXACT4_SUBSET" >/dev/null
  "$PYTHON_BIN" scripts/fr13_taw_b1_credential.py validate-production \
    --mode hydra27_fixed32 \
    --source scripts/fr13_device_multidraft_kernel.py \
    --credential "$TAW_B1_CREDENTIAL" \
    --b1-live-bundle "$TAW_B1_LIVE_BUNDLE" \
    --b4-production-pass "$TAW_REVIEWED_B4_PASS" \
    --b4-gate-verdict "$TAW_REVIEWED_B4_VERDICT" \
    --merge-binding "$TAW_MERGE_BINDING" \
    --production-pass "$TAW_PASS_JSON" >/dev/null
  if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
    "$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py validate-production-smoke \
      --repo "$REPO" \
      --source-commit "$SOURCE_COMMIT" \
      --credential "$COMPOSED_CFWD_SMOKE_PASS" \
      --expected-sha256 "$COMPOSED_CFWD_SMOKE_PASS_SHA256" \
      "${CFWD_COMPONENT_HASH_ARGS[@]}" >/dev/null
  fi
fi
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] \
  || { echo "all Docker containers must be absent before timing" >&2; exit 2; }

export BSIZE=1
export CONC=1
export WALL=0
export FR13_DRAFT_VOCAB_ROOT=1
export FR13_DRAFT_VOCAB_K=65536
export FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER"
export FR13_FLOOR_ORDER=HT
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source "$SEQUENCE"
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_ROOT" == "1" \
   && "$FR13_DRAFT_VOCAB_K" == "65536" \
   && "$FR13_DRAFT_VOCAB_BLOCKS" == "$BLOCK_MAP_CONTAINER" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "ROOT=1 K64 hardware-floor contract drifted" >&2; exit 2; }

mkdir -p "$RUNROOT_ABS/sidecars"
"$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
  --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
  --output "$RUNROOT_ABS/runtime_manifest.at_launch.json"
"$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
  --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_launch.json"
CLASSIFICATION=real_swe_verified_exact4_qrow32_nosplit
if [[ "$COMPOSED_STACK" == "1" ]]; then
  CLASSIFICATION=real_swe_verified_exact4_b1_composed_kernel_stack
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  CLASSIFICATION=real_swe_verified_exact4_b1_composed_cfwd_kernel_stack
fi
if [[ "$PRODUCTION_SMOKE" == "1" ]]; then
  CLASSIFICATION=real_swe_verified_one_task_b1_composed_cfwd_production_smoke
fi
printf 'classification=%s\ntask_count=%s\nbatch_size=1\nconcurrency=1\ntiming_eligible=%s\nformal_floor_acceptance_eligible=0\ntopology=hydra27_fixed32\nphysical_rows=32\nlogical_drafts=27\nvalid_mask=0x7abdffff\ndraft_vocab_root=1\ndraft_vocab_k=65536\nqrow32_nosplit_production=1\nruntime=FULL_graph_exact_geometry\nmandatory_weight_bytes=%s\nmandatory_weight_floor_ms=%s\none_sided_u95_cap_ms=%s\nexact16_rule=only_after_exact4_u95_clears_cap\narm=%s\nsource=%s\npatch_source_sha256=%s\nrunner_sha256=%s\nsubset_sha256=%s\ncandidate_so_sha256=%s\ncandidate_so_bytes=%s\nfa2_head=%s\nfa2_source_closure_sha256=%s\npass_sha256=%s\nqrow16_historical_baseline_sha256=%s\nstarted=%s\n' \
  "$CLASSIFICATION" "$TASK_COUNT" "$TIMING_ELIGIBLE" \
  "$MANDATORY_WEIGHT_BYTES" "$MANDATORY_WEIGHT_FLOOR_MS" \
  "$ONE_SIDED_U95_CAP_MS" "$ARM" "$SOURCE_COMMIT" \
  "$PATCH_SOURCE_SHA256" "$RUNNER_SHA256" "$SUBSET_SHA256" \
  "$CANDIDATE_SHA256" "$CANDIDATE_BYTES" "$FA2_HEAD" \
  "$SOURCE_CLOSURE_SHA256" "$QROW32_B1_PASS_SHA256" "$BASELINE_SHA256" \
  "$(date -u +%FT%TZ)" > "$RUNROOT_ABS/launcher_meta.txt"
if [[ "$COMPOSED_STACK" == "1" ]]; then
  printf 'composed_stack=1\ngqa3_production=1\ndfwd_k64_top3=1\ntarget_selector=%s\nsfwd_conv_postprep_production=1\nqrow_composed_credential_sha256=%s\ngqa3_pass_sha256=%s\ndfwd_top3_credential_sha256=%s\ndfwd_top3_so_sha256=%s\ntarget_so_sha256=%s\ntarget_pass_sha256=%s\nsfwd_pass_sha256=%s\nsfwd_source_manifest_sha256=%s\ntarget_sfwd_combined_summary_sha256=%s\n' \
    "$TARGET_SELECTOR" "$QROW32_B1_COMPOSED_CREDENTIAL_SHA256" \
    "$GQA3_PASS_SHA256" "$DFWD_TOP3_CREDENTIAL_SHA256" \
    "$DFWD_TOP3_SHA256" "$TARGET_SHA256" "$CUTLASS_TARGET_PASS_SHA256" \
    "$SFWD_CONV_POSTPREP_PASS_SHA256" \
    "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" \
    "$TARGET_SFWD_COMBINED_SUMMARY_SHA256" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  printf 'taw_native_precompute_production=1\ncfwd_logit_direct_production=1\ntaw_b1_credential_sha256=%s\ntaw_b1_live_bundle_sha256=%s\ntaw_reviewed_b4_pass_sha256=%s\ntaw_reviewed_b4_verdict_sha256=%s\ntaw_merge_binding_sha256=%s\ntaw_pass_sha256=%s\ncfwd_pass_sha256=%s\n' \
    "$TAW_B1_CREDENTIAL_SHA256" "$TAW_B1_LIVE_BUNDLE_SHA256" \
    "$TAW_REVIEWED_B4_PASS_SHA256" "$TAW_REVIEWED_B4_VERDICT_SHA256" \
    "$TAW_MERGE_BINDING_SHA256" "$TAW_PASS_SHA256" "$CFWD_PASS_SHA256" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
fi

MANIFEST_FINALIZED=0
finalize_manifests() {
  (( MANIFEST_FINALIZED == 0 )) || return 0
  "$PYTHON_BIN" scripts/fr13_runtime_manifest.py \
    --repo "$PWD" --profile fixed32 --sequence "$SEQUENCE" \
    --output "$RUNROOT_ABS/runtime_manifest.at_end.json"
  "$PYTHON_BIN" scripts/fr13_fixed32_contract.py external-manifest \
    --repo "$PWD" --output "$RUNROOT_ABS/external_manifest.at_end.json"
  cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    || { echo "runtime/source manifest changed during timing" >&2; return 14; }
  cmp -s "$RUNROOT_ABS/external_manifest.at_launch.json" \
    "$RUNROOT_ABS/external_manifest.at_end.json" \
    || { echo "external manifest changed during timing" >&2; return 14; }
  [[ "$(sha256sum "$RUNNER_PATH" | awk '{print $1}')" == "$RUNNER_SHA256" ]] \
    || { echo "timing runner changed during execution" >&2; return 14; }
  MANIFEST_FINALIZED=1
}
runner_exit() {
  local rc=$?
  trap - EXIT
  if (( MANIFEST_FINALIZED == 0 )); then
    finalize_manifests || { local mrc=$?; (( rc == 0 )) && rc=$mrc; }
  fi
  exit "$rc"
}
trap runner_exit EXIT

STACK_ENV=(
  FR10_METRICS=0
  FR13_RING_EXPORT=1
  FR13_FLAGS_INKERNEL=1
  FR13_SCAN_ALIGN=0
  FR13_NPAD_INVARIANT=0
  FR13_TREE_GDN_GEOM_OVERRIDE=
  FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0
  FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=
  FR13_FIXED32_GDN_GQA_GROUP3_PASS_JSON=
  FR13_DFWD_K64_TOP3=0
  FR13_DFWD_K64_TOP3_SO=
  FR13_DFWD_K64_TOP3_SHA256=
  FR13_FIXED32_CUTLASS_WAVE=stock
  FR13_FIXED32_CUTLASS_WAVE_SO=
  FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=0
  FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root
  FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=astropy12907
  FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON=
  FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256=
  FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0
  FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0
  FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON=
  FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256=
  FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH=
  FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256=
  FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT=
  FR13_CONV_WB_BATCHED=0
)
if [[ "$COMPOSED_STACK" == "1" ]]; then
  STACK_ENV=(
    FR10_METRICS=1
    FR13_RING_EXPORT=1
    FR13_FLAGS_INKERNEL=1
    FR13_SCAN_ALIGN=0
    FR13_NPAD_INVARIANT=0
    FR13_TREE_GDN_GEOM_OVERRIDE=BV=8
    FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=1
    FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=1
    FR13_FIXED32_GDN_GQA_GROUP3_PASS_JSON="$GQA3_PASS"
    FR13_DFWD_K64_TOP3=1
    FR13_DFWD_K64_TOP3_SO="$DFWD_TOP3_SO"
    FR13_DFWD_K64_TOP3_SHA256="$DFWD_TOP3_SHA256"
    FR13_FIXED32_CUTLASS_WAVE="$TARGET_SELECTOR"
    FR13_FIXED32_CUTLASS_WAVE_SO="$CUTLASS_TARGET_SO"
    FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=1
    FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root
    FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=astropy12907
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_JSON="$CUTLASS_TARGET_PASS"
    FR13_FIXED32_CUTLASS_WAVE_LIVE_PASS_SHA256="$CUTLASS_TARGET_PASS_SHA256"
    FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1
    FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0
    FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON="$SFWD_PASS_CONTAINER"
    FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_SHA256="$SFWD_CONV_POSTPREP_PASS_SHA256"
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH="$SFWD_MANIFEST_CONTAINER"
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256="$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256"
    FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT="$SOURCE_COMMIT"
    FR13_CONV_WB_BATCHED=1
  )
fi
CFWD_ENV=(
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
  FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON=
  FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0
  FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0
  FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON=
  FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256=
)
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  CFWD_ENV=(
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1
    FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$TAW_PASS_JSON"
    FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION=1
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON="$CFWD_PASS_JSON"
    FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256="$CFWD_PASS_SHA256"
  )
fi

if env \
    OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
    FR13_FIXED32_B1_DIAGNOSTIC=0 \
    FR13_DRAFT_VOCAB_ROOT=1 FR13_DRAFT_VOCAB_K=65536 \
    FR13_DRAFT_VOCAB_BLOCKS="$BLOCK_MAP_CONTAINER" \
    FR13_MANDATORY_WEIGHT_BYTES="$MANDATORY_WEIGHT_BYTES" \
    FR13_WEIGHT_FLOOR_MS="$MANDATORY_WEIGHT_FLOOR_MS" \
    ENFORCE_EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE \
    FR13_SFWD_GPU_TIMER=1 FR13_DFWD_GPU_TIMER="$COMPOSED_STACK" FR13_CFWD_GPU_TIMER="$COMPOSED_STACK" \
    FR13_SFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}.json" \
    FR13_DFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_dfwd.json" \
    FR13_CFWD_GPU_TIMER_JSON="/workspace/$RUNROOT_REL/sidecars/${ARM}_cfwd.json" \
    FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0 \
    FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0 \
    FR13_FIXED32_CONV_SOURCE_BATCH=0 \
    FR13_TREE_CONV_FUSED=1 \
    FR13_FA2_QROW16_LIVE_PAGED_AB=0 FR13_FA2_QROW16_PRODUCTION=0 \
    FR13_FA2_QROW32_LIVE_PAGED_AB=0 \
    FR13_FA2_QROW32_B1_LIVE_AB_ARM= \
    FR13_FA2_QROW32_B1_SO_SHA256="$CANDIDATE_SHA256" \
    FR13_FA2_QROW32_B1_SO_SIZE="$CANDIDATE_BYTES" \
    FR13_FA2_QROW32_B1_FA2_HEAD="$FA2_HEAD" \
    FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256="$SOURCE_CLOSURE_SHA256" \
    FR13_FA2_QROW32_B1_SOURCE_COMMIT="$SOURCE_COMMIT" \
    FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256="$PATCH_SOURCE_SHA256" \
    FR13_FA2_QROW32_B1_PRODUCTION_ARM=nosplit \
    FR13_FA2_QROW32_B1_LIVE_PASS_JSON="$QROW32_B1_PASS" \
    FR13_FA2_QROW32_B1_LIVE_PASS_SHA256="$QROW32_B1_PASS_SHA256" \
    FR13_FA2_QROW32_B1_EXACT4_TASK_IDS=astropy__astropy-12907,astropy__astropy-13033,astropy__astropy-13236,astropy__astropy-13398 \
    FR13_FA2_QROW32_B1_EXACT4_SUBSET_SHA256="$EXACT4_SUBSET_SHA256" \
    FR13_DFWD_UNIFIED_BM8_LIVE_AB=0 FR13_DFWD_UNIFIED_BM8_PRODUCTION=0 \
    FR13_DRAFT_HEAD_PAD_ROWS=0 FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB=0 \
    FR13_DRAFT_HEAD_M32_LIVE_AB=0 FR13_DRAFT_HEAD_M32_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB=0 \
    FR13_FIXED32_BATCH_GDN_BV_CANDIDATE= \
    FR13_FIXED32_BATCH_GDN_PRODUCTION=0 \
    FR13_FIXED32_BATCH_GDN_BV_PRODUCTION= \
    FR13_FIXED32_GDN_PATH_BV_CANDIDATE= \
    FR13_FIXED32_GDN_PATH_BV_PRODUCTION= \
    FR13_FIXED32_ATTRIBUTION_ONLY=0 \
    "${CFWD_ENV[@]}" \
    "${STACK_ENV[@]}" \
    FORKED_FA2_SO="$QROW32_B1_FA2_SO" RUNROOT="$RUNROOT_ABS" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh \
      "$ARM" hydra27_fixed32 "$SUBSET" \
      > "$RUNROOT_ABS/$ARM.runlog" 2>&1; then
  serve_rc=0
else
  serve_rc=$?
fi
printf 'serve_rc=%s ended=%s\n' "$serve_rc" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"
(( serve_rc == 0 )) || exit "$serve_rc"

CONTAINER_ENV="$ARMDIR/container_env.txt"
for expected in \
  'FR13_FIXED32_MODE=hydra27_fixed32' \
  'FR13_FIXED32_B1_DIAGNOSTIC=0' \
  'FR13_DRAFT_VOCAB_ROOT=1' \
  'FR13_DRAFT_VOCAB_K=65536' \
  'MAX_NUM_SEQS=1' \
  'SWE_CONCURRENCY=1' \
  'ENFORCE_EAGER=0' \
  'CUDAGRAPH_MODE=FULL_AND_PIECEWISE' \
  'FR13_FA2_QROW16_LIVE_PAGED_AB=0' \
  'FR13_FA2_QROW16_PRODUCTION=0' \
  'FR13_FA2_QROW32_B1_LIVE_AB_ARM=' \
  'FR13_FA2_QROW32_B1_PRODUCTION_ARM=nosplit' \
  'FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0' \
  "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=$CFWD_PRODUCTION" \
  'FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0' \
  "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=$CFWD_PRODUCTION" \
  "FR13_FA2_QROW32_B1_SO_SHA256=$CANDIDATE_SHA256" \
  "FR13_FA2_QROW32_B1_SO_SIZE=$CANDIDATE_BYTES" \
  "FR13_FA2_QROW32_B1_FA2_HEAD=$FA2_HEAD" \
  "FR13_FA2_QROW32_B1_SOURCE_CLOSURE_SHA256=$SOURCE_CLOSURE_SHA256" \
  "FR13_FA2_QROW32_B1_SOURCE_COMMIT=$SOURCE_COMMIT" \
  "FR13_FA2_QROW32_B1_PATCH_SOURCE_SHA256=$PATCH_SOURCE_SHA256"; do
  [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks exact qrow32 timing pin: $expected" >&2; exit 4; }
done
unset expected
if [[ "$COMPOSED_STACK" == "0" ]]; then
  [[ "$(grep -Fxc 'FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0' "$CONTAINER_ENV")" -eq 1 ]] \
    || { echo "container lacks disabled SFWD conv/post-prep pin" >&2; exit 4; }
else
  for expected in \
    'FR10_METRICS=1' \
    'FR13_RING_EXPORT=1' \
    'FR13_FLAGS_INKERNEL=1' \
    'FR13_SCAN_ALIGN=0' \
    'FR13_NPAD_INVARIANT=0' \
    'FR13_TREE_GDN_GEOM_OVERRIDE=BV=8' \
    'FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=1' \
    'FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=1' \
    'FR13_DFWD_K64_TOP3=1' \
    "FR13_DFWD_K64_TOP3_SHA256=$DFWD_TOP3_SHA256" \
    "FR13_FIXED32_CUTLASS_WAVE=$TARGET_SELECTOR" \
    'FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=1' \
    'FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=k64_root' \
    'FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE=astropy12907' \
    'FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1' \
    'FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0' \
    'FR13_FIXED32_SFWD_CONV_POSTPREP_LIVE_PASS_JSON=/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json' \
    'FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_PATH=/logs/fr13_fixed32_sfwd_conv_postprep.source_manifest.json' \
    "FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT=$SOURCE_COMMIT" \
    'FR13_CONV_WB_BATCHED=1' \
    'FR13_TREE_CONV_FUSED=1' \
    'FR13_FIXED32_CONV_SOURCE_BATCH=0' \
    'FR13_SFWD_GPU_TIMER=1' \
    'FR13_DFWD_GPU_TIMER=1' \
    'FR13_CFWD_GPU_TIMER=1'; do
    [[ "$(grep -Fxc "$expected" "$CONTAINER_ENV")" -eq 1 ]] \
      || { echo "container lacks exact composed-stack pin: $expected" >&2; exit 4; }
  done
  unset expected
fi

MEASURE="$ARMDIR/deploy_speed_fullwall.json"
if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
  "$PYTHON_BIN" scripts/fr13_measure.py deploy-speed \
    --arm "$ARM" \
    --out-root "$ARMDIR/swe_out" \
    --expected-tok-per-draft 31 \
    --batch-size 1 \
    --out "$MEASURE"
fi

SIDECAR="$ARMDIR/logs/fr13_fa2_qrow32_b1_production_pass.json"
ENGAGEMENT="$ARMDIR/logs/fr13_fa2_qrow32_b1_production_engagement.json"
HEALTH="$ARMDIR/health.json"
TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"
COMMON_ARTIFACTS=("$SIDECAR" "$ENGAGEMENT" "$HEALTH" "$TRAFFIC_AUDIT")
if [[ "$PRODUCTION_SMOKE" == "0" ]]; then
  COMMON_ARTIFACTS+=("$MEASURE")
fi
for artifact in "${COMMON_ARTIFACTS[@]}"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] \
    || { echo "exact4 timing artifact is missing or unsafe: $artifact" >&2; exit 4; }
done
unset artifact
SIDECAR_SHA256=$(sha256sum "$SIDECAR" | awk '{print $1}')
"$PYTHON_BIN" scripts/fr13_qrow32_b1_pass_sidecar.py verify \
  --sidecar "$SIDECAR" \
  --expected-sidecar-sha256 "$SIDECAR_SHA256" \
  --candidate-so "$QROW32_B1_FA2_SO" \
  --expected-candidate-sha256 "$CANDIDATE_SHA256" \
  --arm nosplit \
  --patch-source scripts/fr13_patch_fa2_tree_bias.py \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-patch-source-sha256 "$PATCH_SOURCE_SHA256" >/dev/null
if [[ "$COMPOSED_STACK" == "1" ]]; then
  GQA3_PRODUCTION_CREDENTIAL="$ARMDIR/logs/fr13_fixed32_gdn_gqa_group3.production_credential.json"
  GQA3_PRODUCTION_ARM="$ARMDIR/logs/fr13_fixed32_gdn_gqa_group3.production.arm"
  GQA3_PRODUCTION_BATCH="$ARMDIR/logs/fr13_fixed32_gdn_gqa_group3.production_batch.flag"
  TARGET_PRODUCTION_SIDECAR="$ARMDIR/logs/fr13_fixed32_cutlass_streamk.production_pass.json"
  TARGET_BINARY_RECORD="$ARMDIR/logs/fr13_fixed32_cutlass_streamk_binary.json"
  TARGET_SELECTOR_RECORD="$ARMDIR/logs/fr13_fixed32_cutlass_wave.selector"
  SFWD_PRODUCTION_PASS="$ARMDIR/logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json"
  SFWD_PRODUCTION_MANIFEST="$ARMDIR/logs/fr13_fixed32_sfwd_conv_postprep.source_manifest.json"
  DOCKER_LOG="$ARMDIR/docker_after_tasks.log"
  for artifact in \
      "$GQA3_PRODUCTION_CREDENTIAL" "$GQA3_PRODUCTION_ARM" \
      "$GQA3_PRODUCTION_BATCH" "$TARGET_PRODUCTION_SIDECAR" \
      "$TARGET_BINARY_RECORD" "$TARGET_SELECTOR_RECORD" \
      "$SFWD_PRODUCTION_PASS" "$SFWD_PRODUCTION_MANIFEST" "$DOCKER_LOG"; do
    [[ -f "$artifact" && ! -L "$artifact" ]] \
      || { echo "composed production evidence is missing or unsafe: $artifact" >&2; exit 4; }
  done
  unset artifact
  [[ "$(sha256sum "$GQA3_PRODUCTION_CREDENTIAL" | awk '{print $1}')" == "$GQA3_PASS_SHA256" \
     && "$(cat "$GQA3_PRODUCTION_ARM")" == "1" \
     && "$(cat "$GQA3_PRODUCTION_BATCH")" == "1" \
     && "$(cat "$TARGET_SELECTOR_RECORD")" == "$TARGET_SELECTOR" \
     && "$(sha256sum "$SFWD_PRODUCTION_PASS" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_PASS_SHA256" \
     && "$(sha256sum "$SFWD_PRODUCTION_MANIFEST" | awk '{print $1}')" == "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256" ]] \
    || { echo "composed production sidecar identity drifted" >&2; exit 4; }
  TARGET_PRODUCTION_SIDECAR_SHA256=$(sha256sum "$TARGET_PRODUCTION_SIDECAR" | awk '{print $1}')
  "$PYTHON_BIN" scripts/fr13_cutlass_streamk_pass.py verify \
    --sidecar "$TARGET_PRODUCTION_SIDECAR" \
    --expected-sidecar-sha256 "$TARGET_PRODUCTION_SIDECAR_SHA256" \
    --candidate-so "$CUTLASS_TARGET_SO" \
    --patch-source scripts/fr13_patch_cutlass_fixed32_wave.py \
    --candidate-selector "$TARGET_SELECTOR" \
    --qualification-profile k64_root \
    --diagnostic-task-profile astropy12907 \
    --draft-vocab-blocks "$BLOCK_MAP" >/dev/null
  for marker in \
    '[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3' \
    '[FR13_DFWD_K64_TOP3] engaged stock_argmax_topk_map_copy=0' \
    '[FR13_DFWD_K64_TOP3] graph captured_calls=4'; do
    grep -Fq "$marker" "$DOCKER_LOG" \
      || { echo "DFWD top3 production marker is missing: $marker" >&2; exit 4; }
  done
  unset marker
  [[ "$(grep -Fc '[FR13_SFWD_CONV_POSTPREP] production engaged layer=' "$DOCKER_LOG")" -eq 48 ]] \
    || { echo "SFWD conv/post-prep production did not engage exactly 48 layers" >&2; exit 4; }
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  TAW_PRODUCTION_PASS="$ARMDIR/logs/fr13_fixed32_taw_native_precompute.production_pass.json"
  TAW_PRODUCTION_ARM="$ARMDIR/logs/fr13_fixed32_taw_native_precompute_production.arm"
  CFWD_PRODUCTION_PASS="$ARMDIR/logs/fr13_cfwd_logit_direct.production_pass.json"
  CFWD_ENGAGEMENT="$ARMDIR/logs/fr13_cfwd_logit_direct.production_engagement.json"
  for artifact in \
      "$TAW_PRODUCTION_PASS" "$TAW_PRODUCTION_ARM" \
      "$CFWD_PRODUCTION_PASS" "$CFWD_ENGAGEMENT"; do
    [[ -f "$artifact" && ! -L "$artifact" ]] \
      || { echo "TAW/CFWD production evidence is missing or unsafe: $artifact" >&2; exit 4; }
  done
  unset artifact
  [[ "$(sha256sum "$TAW_PRODUCTION_PASS" | awk '{print $1}')" == "$TAW_PASS_SHA256" \
     && "$(cat "$TAW_PRODUCTION_ARM")" == "1" \
     && "$(sha256sum "$CFWD_PRODUCTION_PASS" | awk '{print $1}')" == "$CFWD_PASS_SHA256" ]] \
    || { echo "TAW/CFWD copied production credential drifted" >&2; exit 4; }
fi
finalize_manifests

if [[ "$PRODUCTION_SMOKE" == "1" ]]; then
  SMOKE_PASS="$RUNROOT_ABS/composed_cfwd_production_smoke.json"
  "$PYTHON_BIN" scripts/fr13_b1_composed_stack_gate.py issue-production-smoke \
    --repo "$REPO" \
    --source-commit "$SOURCE_COMMIT" \
    --arm "$ARMDIR" \
    --runtime-launch "$RUNROOT_ABS/runtime_manifest.at_launch.json" \
    --runtime-end "$RUNROOT_ABS/runtime_manifest.at_end.json" \
    --external-launch "$RUNROOT_ABS/external_manifest.at_launch.json" \
    --external-end "$RUNROOT_ABS/external_manifest.at_end.json" \
    --output "$SMOKE_PASS" \
    "${CFWD_COMPONENT_ARGS[@]}"
  SMOKE_PASS_SHA256=$(sha256sum "$SMOKE_PASS" | awk '{print $1}')
  printf 'production_smoke=%s\nproduction_smoke_sha256=%s\nended=%s\n' \
    "$SMOKE_PASS" "$SMOKE_PASS_SHA256" "$(date -u +%FT%TZ)" \
    >> "$RUNROOT_ABS/launcher_meta.txt"
  exit 0
fi

TIMING_REDUCER=scripts/fr13_qrow32_split2_timing.py
if [[ "$COMPOSED_STACK" == "1" ]]; then
  TIMING_REDUCER=scripts/fr13_b1_composed_stack_timing.py
fi
REDUCER_ARGS=(
  --subset "$SUBSET"
  --measure "$MEASURE"
  --baseline "$BASELINE"
  --engagement "$ENGAGEMENT"
  --health "$HEALTH"
  --traffic-audit "$TRAFFIC_AUDIT"
  --source-commit "$SOURCE_COMMIT"
  --patch-source-sha256 "$PATCH_SOURCE_SHA256"
  --pass-sha256 "$QROW32_B1_PASS_SHA256"
  --pass-sidecar-sha256 "$SIDECAR_SHA256"
  --runner-sha256 "$RUNNER_SHA256"
  --block-map-sha256 "$BLOCK_MAP_SHA256"
  --floor-ms "$MANDATORY_WEIGHT_FLOOR_MS"
  --cap-ms "$ONE_SIDED_U95_CAP_MS"
  --arm "$ARM"
  --out "$RUNROOT_ABS/timing_summary.json"
)
if [[ "$COMPOSED_STACK" == "1" ]]; then
  REDUCER_ARGS+=(
    --container-env "$CONTAINER_ENV"
    --docker-log "$DOCKER_LOG"
    --gqa3-production-credential "$GQA3_PRODUCTION_CREDENTIAL"
    --gqa3-production-arm "$GQA3_PRODUCTION_ARM"
    --gqa3-production-batch "$GQA3_PRODUCTION_BATCH"
    --gqa3-pass-sha256 "$GQA3_PASS_SHA256"
    --target-production-sidecar "$TARGET_PRODUCTION_SIDECAR"
    --target-production-sidecar-sha256 "$TARGET_PRODUCTION_SIDECAR_SHA256"
    --target-binary-record "$TARGET_BINARY_RECORD"
    --sfwd-production-pass "$SFWD_PRODUCTION_PASS"
    --sfwd-pass-sha256 "$SFWD_CONV_POSTPREP_PASS_SHA256"
    --sfwd-production-manifest "$SFWD_PRODUCTION_MANIFEST"
    --sfwd-manifest-sha256 "$SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256"
    --qrow-composed-credential "$QROW32_B1_COMPOSED_CREDENTIAL"
    --qrow-composed-credential-sha256 "$QROW32_B1_COMPOSED_CREDENTIAL_SHA256"
    --dfwd-credential "$DFWD_TOP3_CREDENTIAL"
    --dfwd-credential-sha256 "$DFWD_TOP3_CREDENTIAL_SHA256"
    --target-sfwd-combined-summary "$TARGET_SFWD_COMBINED_SUMMARY"
    --target-sfwd-combined-summary-sha256 "$TARGET_SFWD_COMBINED_SUMMARY_SHA256"
  )
fi
if [[ "$CFWD_PRODUCTION" == "1" ]]; then
  REDUCER_ARGS+=(
    --cfwd-production
    --taw-production-pass "$TAW_PRODUCTION_PASS"
    --taw-production-pass-sha256 "$TAW_PASS_SHA256"
    --cfwd-production-pass "$CFWD_PRODUCTION_PASS"
    --cfwd-production-pass-sha256 "$CFWD_PASS_SHA256"
    --cfwd-engagement "$CFWD_ENGAGEMENT"
    --cfwd-smoke-credential "$COMPOSED_CFWD_SMOKE_PASS"
    --cfwd-smoke-credential-sha256 "$COMPOSED_CFWD_SMOKE_PASS_SHA256"
  )
fi
"$PYTHON_BIN" "$TIMING_REDUCER" "${REDUCER_ARGS[@]}"
printf 'summary=%s ended=%s\n' \
  "$RUNROOT_ABS/timing_summary.json" "$(date -u +%FT%TZ)" \
  >> "$RUNROOT_ABS/launcher_meta.txt"

#!/usr/bin/env bash
# Same-build SWE-bench Verified agent study, TreeHost arm: the Cqc10 vehicle (tail10, ARM_KIND=C,
# PROMOAB_FA2=default split-K, exact16_qc_remainder_10, Cqc10 PROMOAB_EXTRA_ENV) on latest code.
set -uo pipefail
ROOT=/home/user/shared/treehost-v2exp-runs
WT=/home/user/shared/treehost-v2exp-20260930
TS=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ROOT/swe"; LOG=$ROOT/swe/tree-driver-$TS.log
exec > >(tee -a "$LOG") 2>&1
[[ -z "$(docker ps -q)" ]] || { echo "a container is running; refusing"; exit 3; }
PROMOAB_EXTRA_ENV="FR13_B1_CREDENTIAL_POINTER=/nonexistent
FR13_FA2_QROW32_B1_TIERB_WORKLOAD=exact16_qc_remainder_10
FR13_FA2_QROW32_B1_TIERB_TASK_IDS=astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=716503a46a991e3b187e14777f96f074c1a3359d9f8b7928f4453a6b9da1ee9b" \
ARM_KIND=C PROMOAB_FA2=default PROMOAB_SUBSET=exact16_qc_remainder_10 PROMOAB_ARM_SUFFIX=_v2swe$TS \
  bash "$WT/scripts/v2exp/promoab_tail10_swe.sh" > "$ROOT/swe/tree-promoab-$TS.out" 2>&1
RC=$?
echo "promoab rc=$RC"; tail -8 "$ROOT/swe/tree-promoab-$TS.out"
R=$(ls -td "$ROOT"/swe/fr14_promoab_C_v2swe${TS}_* 2>/dev/null | head -1)
grep -h "B1 arm unnamed\|INCUMBENT" "$R"/*/launch.log "$ROOT/swe/tree-promoab-$TS.out" 2>/dev/null | head -2
grep -q "gqa_pair_splitk" "$R"/*/launch.log 2>/dev/null || echo "WARNING: split-K not confirmed in launch.log"
for c in $(docker ps -aq --filter "name=fr13-bigdenom-hydra27_fixed32_promoab_C_v2swe$TS"); do
  docker logs "$c" > "$R/container_after_teardown.log" 2>&1; docker stop -t 30 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null && echo "removed own container $c"
done
exit $RC

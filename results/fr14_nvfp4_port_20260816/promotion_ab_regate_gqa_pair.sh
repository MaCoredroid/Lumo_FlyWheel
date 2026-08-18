#!/usr/bin/env bash
# FR14 PROMOTION A/B — step 0: re-earn the gqa_pair K0 byte gate AT THIS HEAD.
#
# WHY THIS IS STEP 0 AND NOT OPTIONAL.
# "Today's promoted stack" at B1 is hydra27 K0 + gqa_pair (promoted 2026-08-13,
# Mark's "B1 flip Yes"). gqa_pair arms only from a credential whose
# FR13_FA2_QROW32_B1_SOURCE_COMMIT == $(git rev-parse HEAD) -- the launcher
# checks serviceability, not mere presence. The banked pointer
# output/fr13_b1_gqa_pair_credential.env was earned at 05987f682, and HEAD has
# moved since (pass 51/52/53). Without this re-earn the promoted default
# DEGRADES TO THE INCUMBENT and says so in the runlog -- which would mean the
# control arm of a promotion campaign did not carry the promoted kernel.
#
# 14m21s on the last run (output/fr14_gqa_k0_gate_20260817T235503Z). It serves
# ONE real SWE task (astropy__astropy-12907) with the qrow16 incumbent and
# byte-compares the candidate in shadow; it produces no timing evidence.
#
# CONSEQUENCE FOR CAMPAIGN DISCIPLINE: the credential this mints dies the moment
# HEAD moves, so NO COMMIT may be taken between here and the last arm's drain.
# That supersedes "commit between arms" for this campaign; the reason is banked
# in promotion_ab_campaign.md.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_promoab_gate_$TS

echo "[regate] $(date -u +%FT%TZ) HEAD=$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "[regate] FAIL: tracked worktree must be clean"; exit 2; }
[[ -z "$(docker ps -aq)" ]] || { echo "[regate] FAIL: docker not empty"; exit 2; }
sync
sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
free -g
awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
  || { echo "[regate] FAIL: unified-memory preflight"; exit 2; }

FR13_RUN_QROW32_K0_LIVE_GATE=1 \
FR13_QROW32_B1_LIVE_ARM=gqa_pair \
RUNROOT="$RUNROOT" \
TAG=promoab \
QROW32_B1_FA2_SO=/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so \
QROW32_B1_FA2_SOURCE=/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810/source \
TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr14_run_b1_k0_qrow32_live_gate.sh > "$RUNROOT.driver.log" 2>&1
rc=$?
echo "[regate] rc=$rc $(date -u +%FT%TZ)"
echo "[regate] containers after: $(docker ps -aq | wc -l)"
tail -30 "$RUNROOT.driver.log"
exit "$rc"

#!/usr/bin/env bash
# Boot the deployed LumoTree (hydra27 fixed32, promoted stack) in serve-only mode,
# replay the request corpus with signed requests, then stop it.
# Usage: run_tree_arm.sh <mode greedy|sampled> [max_tokens]
set -uo pipefail
MODE=${1:-sampled}
[[ $MODE == sampled ]] || { echo "fixed32 route requires temperature>0; only sampled mode is supported"; exit 2; }; MAXTOK=${2:-1024}
ROOT=/home/mark/shared/lumotree-v2exp-runs
WT=/home/mark/shared/lumotree-v2exp-20260930
PORTWT=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
TS=$(date -u +%Y%m%dT%H%M%SZ)
TAGW=$(printf "%s" "${V2_TAG:-}" | tr -cd "A-Za-z0-9")
echo "variant: FA2=${V2_FA2:-default} TOPK=${V2_TOPK:-promoted} KIND=${V2_KIND:-hydra27_fixed32} TAG=${V2_TAG:-} EXTRA=${V2_EXTRA_ENV:-}"
LOG=$ROOT/tree-driver-$MODE-$TS.log
exec > >(tee -a "$LOG") 2>&1
[[ -z "$(docker ps -q)" ]] || { echo "a container is running; refusing"; exit 3; }
mkdir -p "$ROOT/tree"
before=$(ls -d "$ROOT"/tree/fr14_promoab_C_v2exp${TAGW}${MODE}${TS}_* 2>/dev/null | sort | tail -1)
# Extra env copied verbatim from the Cqc10 arm_env.txt (ten-task deployment).
PROMOAB_EXTRA_ENV="FR13_B1_CREDENTIAL_POINTER=/nonexistent
FR13_FA2_QROW32_B1_TIERB_WORKLOAD=exact16_qc_remainder_10
FR13_FA2_QROW32_B1_TIERB_TASK_IDS=astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=716503a46a991e3b187e14777f96f074c1a3359d9f8b7928f4453a6b9da1ee9b${V2_EXTRA_ENV:+$'\n'$V2_EXTRA_ENV}" \
ARM_KIND=C PROMOAB_FA2=${V2_FA2:-default} PROMOAB_TOPK=${V2_TOPK:-promoted} PROMOAB_KIND=${V2_KIND:-hydra27_fixed32} PROMOAB_SUBSET=exact16_qc_remainder_10 PROMOAB_ARM_SUFFIX=_v2exp${TAGW}${MODE}${TS} setsid bash "$WT/scripts/v2exp/promoab_tail10_serve_only.sh" \
  > "$ROOT/tree-promoab-$TS.out" 2>&1 < /dev/null &
DRV=$!
echo "driver pid=$DRV"
RUNROOT=""
t0=$(date +%s)
while :; do
  sleep 15
  cand=$(ls -d "$ROOT"/tree/fr14_promoab_C_v2exp${TAGW}${MODE}${TS}_* 2>/dev/null | sort | tail -1)
  [[ -n "$cand" && "$cand" != "$before" ]] && RUNROOT=$cand
  [[ -n "$RUNROOT" && -f "$RUNROOT/READY.json" ]] && break
  kill -0 $DRV 2>/dev/null || { echo "driver exited before READY"; tail -40 "$ROOT/tree-promoab-$TS.out"; exit 1; }
  (( $(date +%s) - t0 > 3000 )) && { echo "READY timeout"; [[ -n "$RUNROOT" ]] && touch "$RUNROOT/STOP"; exit 1; }
done
echo "READY after $(( $(date +%s) - t0 ))s: $(cat "$RUNROOT/READY.json")"
L=$(ls "$RUNROOT"/*/launch.log 2>/dev/null | head -1)
grep -h "B1 arm\|PROMOTED DEFAULT\|split-K\|INCUMBENT" "$L" "$ROOT"/tree-promoab-$TS.out 2>/dev/null | head -5
[[ "${V2_FA2:-default}" != default ]] || grep -q "gqa_pair_splitk" "$L" || { echo "ABORT: deployed split-K attention arm not engaged"; touch "$RUNROOT/STOP"; wait $DRV; exit 4; }
KINDDIR=replay; [[ "${V2_DIST:-0}" == 1 ]] && KINDDIR=dist
OUT=$ROOT/$KINDDIR/tree${V2_TAG:-}-$MODE-$TS; mkdir -p "$OUT"; echo "$RUNROOT" > "$OUT/serve_runroot.txt"
export V2EXP_READY_FILE="$RUNROOT/READY.json"
cd "$WT/scripts/v2exp"
PY=$PORTWT/.venv/bin/python
curl -s http://127.0.0.1:9950/metrics > "$OUT/metrics_boot.txt"
$PY replay.py --arm tree-warmup --requests "$ROOT/corpus/requests" --out "$OUT/warmup.jsonl" \
  --mode sampled --max-tokens 64 --limit 1 --auth-hook fixed32_auth:headers
if [[ "${V2_DIST:-0}" == 1 ]]; then
  $PY dist_sample.py --arm tree --requests "$ROOT/corpus/requests" --out "$OUT/dist.jsonl" \
    --samples "${V2_DIST_SAMPLES:-40}" --max-tokens "${V2_DIST_MAXTOK:-24}" --order "${V2_DIST_ORDER:-sample-outer}" --auth-hook fixed32_auth:headers
else
  $PY replay.py --arm tree --requests "$ROOT/corpus/requests" --out "$OUT/replay.jsonl" \
    --mode "$MODE" --max-tokens "$MAXTOK" --auth-hook fixed32_auth:headers
fi
REPLAY_RC=$?
curl -s http://127.0.0.1:9950/metrics > "$OUT/metrics_end.txt"
touch "$RUNROOT/STOP"
echo "stop requested; waiting for driver teardown"
wait $DRV; echo "driver rc=$? $(date -u +%FT%TZ)"
# remove only this run's own preserved (exited) container; its log is already in docker_full.log
# Serve-only exits before the SWE finalize, so the harness may preserve this run's own
# container (even running). Save its log, then stop/remove only that container.
for c in $(docker ps -aq --filter "name=fr13-bigdenom-${V2_KIND:-hydra27_fixed32}_promoab_C_v2exp${TAGW}${MODE}${TS}"); do
  docker logs "$c" > "$OUT/container_after_teardown.log" 2>&1
  docker stop -t 30 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null && echo "removed own container $c"
done
# fail the step if any replay record errored or the replay itself failed
EXPECT=43; RFILE="$OUT/replay.jsonl"
[[ "${V2_DIST:-0}" == 1 ]] && { EXPECT=$(( ${V2_DIST_SAMPLES:-40} * 20 )); RFILE="$OUT/dist.jsonl"; }
python3 - "$RFILE" "$EXPECT" <<'PY' || REPLAY_RC=7
import json,sys
recs=[json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
bad=[r for r in recs if r.get("error")]
print(f"replay records={len(recs)} errors={len(bad)}")
sys.exit(1 if (bad or len(recs)<int(sys.argv[2])) else 0)
PY
exit $REPLAY_RC

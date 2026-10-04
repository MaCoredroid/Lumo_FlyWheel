#!/usr/bin/env bash
# Penalty-history diagnostic: deployed LumoTree (ARM_KIND=C) + record-only penalty capture (patch_runner.py in
# duplicate-padding defect (patch_runner.py in this dir). Replays the corpus like run_tree_arm.sh.
# Env: PD_TAG (e.g. -pd1), PD_HOOKS=1 (passive trace),
#      PD_LIMIT (requests, default all), PD_MAXTOK (default 1024), PD_REQUESTS (default tuning corpus)
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; WT=/home/mark/shared/lumotree-v2exp-20260930; PORT=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
Q1V3=$WT/scripts/v2exp/q1v3; GD=$WT/scripts/v2exp/gatediag; GFD=$WT/scripts/v2exp/pendiag
TAG=${PD_TAG:?set PD_TAG}; TAGW=$(printf "%s" "$TAG" | tr -cd "A-Za-z0-9"); TS=$(date -u +%Y%m%dT%H%M%SZ)
REQ=${PD_REQUESTS:-$ROOT/corpus/requests}; NREQ=$(ls "$REQ"/*.json | wc -l); [[ -n "${PD_LIMIT:-}" ]] && NREQ=$PD_LIMIT
RUN=$ROOT/pendiag/$TAGW-$TS; SRC=$RUN/src; OUT=$ROOT/replay/tree$TAG-sampled-$TS
mkdir -p "$RUN/CAND" "$OUT"; exec > >(tee -a "$RUN/driver.log") 2>&1
log(){ echo "[pendiag $(date -u +%FT%TZ)] $*"; }
log "tag=$TAG hooks=${PD_HOOKS:-0} req=$REQ n=$NREQ"
[[ -z "$(docker ps -q)" ]] || { log "a container is running; refusing"; exit 3; }
cp -r "$Q1V3" "$SRC"; rm -rf "$SRC/__pycache__" "$SRC/tests/__pycache__"
mv "$SRC/patch_runner.py" "$SRC/patch_runner_q1v3.py"; cp "$GFD/patch_runner.py" "$SRC/patch_runner.py"
cp "$GD/q1v3_hooks.py" "$SRC/q1v3_hooks.py"; [[ "${PD_HOOKS:-0}" == 1 ]] && touch "$SRC/HOOKS_ON"
PYTHONDONTWRITEBYTECODE=1 python3 "$Q1V3/make_cand_launch.py" --out "$RUN/CAND/gen" > "$RUN/CAND/gen.log" || { log "launch-chain generation refused"; cat "$RUN/CAND/gen.log"; exit 1; }
PYTHONPATH=$WT/src $PORT/.venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" || log "recover_host_memory failed"
SUF=_pd${TAGW}$TS
PROMOAB_EXTRA_ENV="FR13_B1_CREDENTIAL_POINTER=/nonexistent
FR13_FA2_QROW32_B1_TIERB_WORKLOAD=exact16_qc_remainder_10
FR13_FA2_QROW32_B1_TIERB_TASK_IDS=astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=716503a46a991e3b187e14777f96f074c1a3359d9f8b7928f4453a6b9da1ee9b" \
ARM_KIND=C PROMOAB_FA2=default PROMOAB_TOPK=promoted PROMOAB_KIND=hydra27_fixed32 PROMOAB_SUBSET=exact16_qc_remainder_10 \
PROMOAB_ARM_SUFFIX=$SUF Q1V3_SRC="$SRC" Q1V3_RUN="$RUN" Q1V3_SERVE_ROOT="$RUN/CAND/serve" \
Q1V3_VARIANT="$RUN/CAND/gen/variant_q1v3.sh" Q1V3_LAUNCHER="$RUN/CAND/gen/launcher_q1v3.sh" \
  setsid bash "$RUN/CAND/gen/promoab_q1v3.sh" > "$RUN/CAND/promoab.out" 2>&1 < /dev/null &
DRV=$!; RR=""; t0=$(date +%s)
while :; do
  sleep 15
  RR=$(ls -d "$RUN"/CAND/serve/fr14_promoab_C${SUF}_* 2>/dev/null | tail -1)
  [[ -n "$RR" && -f "$RR/READY.json" ]] && break
  kill -0 $DRV 2>/dev/null || { log "vehicle exited before READY"; tail -30 "$RUN/CAND/promoab.out"; cat "$RUN/CAND/pendiag_receipt.json" 2>/dev/null; exit 1; }
  (( $(date +%s) - t0 > 3000 )) && { log "READY timeout"; [[ -n "$RR" ]] && touch "$RR/STOP"; wait $DRV; exit 1; }
done
log "READY after $(( $(date +%s) - t0 ))s"; cat "$RUN/CAND/pendiag_receipt.json"; echo
grep -q '"applied": true' "$RUN/CAND/pendiag_receipt.json" || { log "ABORT: pendiag not applied"; touch "$RR/STOP"; wait $DRV; exit 4; }
L=$(ls "$RR"/*/launch.log | head -1); grep -q "gqa_pair_splitk" "$L" || { log "ABORT: split-K attention not engaged"; touch "$RR/STOP"; wait $DRV; exit 4; }
echo "$RR" > "$OUT/serve_runroot.txt"; echo "$RUN" > "$OUT/gatefix_run.txt"
cd "$WT/scripts/v2exp"; export V2EXP_READY_FILE="$RR/READY.json"; PY=$PORT/.venv/bin/python
curl -s http://127.0.0.1:9950/metrics > "$OUT/metrics_boot.txt"
$PY replay.py --arm tree-warmup --requests "$REQ" --out "$OUT/warmup.jsonl" --mode sampled --max-tokens 64 --limit 1 --auth-hook fixed32_auth:headers
$PY replay.py --arm tree$TAG --requests "$REQ" --out "$OUT/replay.jsonl" --mode sampled --max-tokens "${PD_MAXTOK:-1024}" \
  ${PD_LIMIT:+--limit $PD_LIMIT} --auth-hook fixed32_auth:headers
RC=$?
curl -s http://127.0.0.1:9950/metrics > "$OUT/metrics_end.txt"
touch "$RR/STOP"; log "stop requested"; wait $DRV; log "vehicle rc=$?"
for c in $(docker ps -aq --filter "name=fr13-bigdenom-hydra27_fixed32_promoab_C$SUF"); do
  docker logs "$c" > "$OUT/container_after_teardown.log" 2>&1; docker stop -t 30 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null && log "removed own container $c"
done
cp $PORT/output/fr13_sfwd_sidecar/*_promoab_G${SUF}*.json.* "$OUT/" 2>/dev/null || log "no timer sidecars"
cp "$RUN"/CAND/trace.*.jsonl "$RUN"/CAND/pen_trace.jsonl "$OUT/" 2>/dev/null
python3 - "$OUT/replay.jsonl" "$NREQ" <<'PY' || RC=7
import json,sys,os
recs=[json.loads(l) for l in open(sys.argv[1])] if os.path.exists(sys.argv[1]) else []
bad=[r for r in recs if r.get("error")]
print(f"replay records={len(recs)} errors={len(bad)}"); sys.exit(1 if (bad or len(recs)<int(sys.argv[2])) else 0)
PY
exit $RC

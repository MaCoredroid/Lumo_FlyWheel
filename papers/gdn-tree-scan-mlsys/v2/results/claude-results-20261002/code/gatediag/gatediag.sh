#!/usr/bin/env bash
# Passive gated-step diagnostic: deployed LumoTree vehicle with the suffix pass gate on (permissive
# predicate, ~all steps gated) plus record-only runner hooks (q1v3 launch chain, passive hooks module).
# Replays a few corpus requests and records per-step tree inputs, drafts, gated flag and TAW products.
set -uo pipefail
ROOT=/home/mark/shared/lumotree-v2exp-runs; WT=/home/mark/shared/lumotree-v2exp-20260930
Q1V3=$WT/scripts/v2exp/q1v3; SRC=$ROOT/gatediag-src
TS=$(date -u +%Y%m%dT%H%M%SZ); RUN=$ROOT/gatediag/$TS; mkdir -p "$RUN/CAND"
exec > >(tee -a "$RUN/driver.log") 2>&1
log(){ echo "[gatediag $(date -u +%FT%TZ)] $*"; }
[[ -z "$(docker ps -q)" ]] || { log "a container is running; refusing"; exit 3; }
rm -rf "$SRC"; cp -r "$Q1V3" "$SRC"; rm -rf "$SRC/__pycache__" "$SRC/tests/__pycache__"
cp "$WT/scripts/v2exp/gatediag/q1v3_hooks.py" "$SRC/q1v3_hooks.py"
PYTHONDONTWRITEBYTECODE=1 python3 "$Q1V3/make_cand_launch.py" --out "$RUN/CAND/gen" > "$RUN/CAND/gen.log" || { log "launch-chain generation refused"; cat "$RUN/CAND/gen.log"; exit 1; }
PYTHONPATH=$WT/src /home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/.venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" || log "recover_host_memory failed"
PROMOAB_EXTRA_ENV="FR13_B1_CREDENTIAL_POINTER=/nonexistent
FR13_FA2_QROW32_B1_TIERB_WORKLOAD=exact16_qc_remainder_10
FR13_FA2_QROW32_B1_TIERB_TASK_IDS=astropy__astropy-13977,astropy__astropy-14096,astropy__astropy-14182,astropy__astropy-14309,astropy__astropy-14365,astropy__astropy-14369,astropy__astropy-14508,astropy__astropy-14539,astropy__astropy-14598,astropy__astropy-14995
FR13_FA2_QROW32_B1_TIERB_SUBSET_SHA256=716503a46a991e3b187e14777f96f074c1a3359d9f8b7928f4453a6b9da1ee9b
FR14_SUFFIX_PASS_GATE_NGRAM=${GD_NGRAM:-1}
FR14_SUFFIX_PASS_GATE_MIN_AGREE=${GD_AGREE:-0}
FR14_SUFFIX_PASS_GATE_MIN_HISTORY=${GD_HIST:-1}" \
ARM_KIND=${GD_ARM_KIND:-G} PROMOAB_FA2=default PROMOAB_TOPK=promoted PROMOAB_KIND=hydra27_fixed32 PROMOAB_SUBSET=exact16_qc_remainder_10 \
PROMOAB_ARM_SUFFIX=_gdiag$TS Q1V3_SRC="$SRC" Q1V3_RUN="$RUN" Q1V3_SERVE_ROOT="$RUN/CAND/serve" \
Q1V3_VARIANT="$RUN/CAND/gen/variant_q1v3.sh" Q1V3_LAUNCHER="$RUN/CAND/gen/launcher_q1v3.sh" \
  setsid bash "$RUN/CAND/gen/promoab_q1v3.sh" > "$RUN/CAND/promoab.out" 2>&1 < /dev/null &
DRV=$!; RR=""; t0=$(date +%s)
while :; do
  sleep 15
  RR=$(ls -d "$RUN"/CAND/serve/fr14_promoab_*_gdiag${TS}_* 2>/dev/null | tail -1)
  [[ -n "$RR" && -f "$RR/READY.json" ]] && break
  kill -0 $DRV 2>/dev/null || { log "vehicle exited before READY"; tail -40 "$RUN/CAND/promoab.out"; exit 1; }
  (( $(date +%s) - t0 > 3000 )) && { log "READY timeout"; [[ -n "$RR" ]] && touch "$RR/STOP"; wait $DRV; exit 1; }
done
log "READY after $(( $(date +%s) - t0 ))s"; cat "$RUN/CAND/patch_receipt.json" | head -c 300; echo
grep -h "SUFFIX PASS GATE" "$RR"/*/launch.log | head -2
cd "$WT/scripts/v2exp"; export V2EXP_READY_FILE="$RR/READY.json"
/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816/.venv/bin/python replay.py --arm gatediag --requests "$ROOT/corpus/requests" \
  --out "$RUN/replay.jsonl" --mode sampled --max-tokens "${GD_MAXTOK:-512}" --limit "${GD_LIMIT:-8}" --auth-hook fixed32_auth:headers
touch "$RR/STOP"; log "stop requested"; wait $DRV; log "vehicle rc=$?"
for c in $(docker ps -aq --filter "name=fr13-bigdenom-hydra27_fixed32_promoab_${GD_ARM_KIND:-G}_gdiag$TS"); do
  docker logs "$c" > "$RUN/container_after_teardown.log" 2>&1; docker stop -t 30 "$c" >/dev/null 2>&1; docker rm "$c" >/dev/null && log "removed own container $c"
done
ls -la "$RUN/CAND"/trace.*.jsonl 2>/dev/null; wc -l "$RUN/CAND"/trace.*.jsonl 2>/dev/null
exit 0

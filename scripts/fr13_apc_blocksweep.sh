#!/usr/bin/env bash
# FR13 APC BLOCK-SIZE SWEEP — does scaling mamba_block_size UP reduce the cache-ON
# char-8 / empty-patch failure rate? Tests the "block 1024 = bad zone" hypothesis
# (historical spec+cache: 1024=0/3 all char-8, 8192=1/2). Mechanism under test:
# bigger block => fewer cache snapshot/restore events per turn => less perturbation
# => closer to cache-OFF behavior. Single task (12907), temp 0.6, N rollouts/arm.
# Arms: OFF (no cache, native ~816 page) + cache-ON @ block {1024,2048,4096,8192}
# (the drift-curve align points, so resolve-by-block pairs with state-drift-by-block).
# Tally resolve-rate + char-8-rate by block + Fisher (ON1024 vs ON8192, ON8192 vs OFF).
# OOM-hardened (DOCKER_MEM_CAP + watchdog + recover_host_memory between boots).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
source .lumo.local.env 2>/dev/null || true
N=${N:-6}
SUBSET=${SUBSET:-output/fr13_b1_gold_swe/subset_astropy12907.json}
RUNROOT=${RUNROOT:-output/fr13_apc_blocksweep/run_$(date -u +%Y%m%dT%H%M%SZ)}
DOCKER_MEM_CAP=${DOCKER_MEM_CAP:-105g}; export DOCKER_MEM_CAP
mkdir -p "$RUNROOT"
SUMMARY="$RUNROOT/blocksweep_rollouts.tsv"
printf 'arm\tblock\trollout\tverdict\tchar8\trun_log\n' > "$SUMMARY"
ARMS=${ARMS:-"OFF:0:0 ON1024:1:1024 ON2048:1:2048 ON4096:1:4096 ON8192:1:8192"}  # tag:APC:BLOCK
echo "=== FR13 APC BLOCK SWEEP  N=$N/arm  arms=[$ARMS]  subset=$SUBSET  -> $RUNROOT ==="

_hygiene() {
  pgrep -af 'oom_protect_watchdog\.sh' | grep -qv pgrep || { setsid bash scripts/oom_protect_watchdog.sh >/dev/null 2>&1 </dev/null & disown; }
  bash scripts/oom_protect_session.sh >/dev/null 2>&1 || true
  PYTHONPATH="$PWD/src" .venv/bin/python -c "from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()" >/dev/null 2>&1 || true
  echo "  [hygiene] avail=$(free -g | awk '/Mem/{print $7}')GiB"
}

# interleave by rollout so an early stop still has all arms partially covered
for i in $(seq 1 "$N"); do
  for spec in $ARMS; do
    tag=${spec%%:*}; rest=${spec#*:}; APC=${rest%%:*}; BLOCK=${rest##*:}
    ARM="bs_${tag}_r${i}"; RLOG="$RUNROOT/${ARM}.log"
    echo "--- [$tag rollout $i/$N] APC=$APC block=$BLOCK arm=$ARM @ $(date -u +%H:%M:%S) ---"
    _hygiene
    docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
    BLKENV=""; [ "$APC" = "1" ] && BLKENV="MAMBA_BLOCK_SIZE=$BLOCK APC_BLOCK_SIZE=$BLOCK"
    env FR13_ENABLE_APC=$APC FR13_APC_CONFIG_ONLY=0 $BLKENV MAMBA_SSM_CACHE_DTYPE=float32 \
      DEPLOY_FORCE_TEMP=0.6 OFFLOAD_CODEX=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 \
      FR10_METRICS=0 BATCH_INVARIANT=0 DOCKER_MEM_CAP="$DOCKER_MEM_CAP" RUNROOT="$RUNROOT" \
      bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" cat6root "$SUBSET" > "$RLOG" 2>&1 </dev/null
    V=$(grep -hoE "resolved_rate=[0-9.]+" "$RLOG" 2>/dev/null | tail -1)
    VERD=$(echo "$V" | grep -qE "resolved_rate=1" && echo resolved || echo failed)
    TR=$(find "$RUNROOT/$ARM" -path "*per_task*/codex_trace.jsonl" 2>/dev/null | head -1)
    C8=0; [ -n "$TR" ] && C8=$(grep -cE 'Unterminated|column 9 \(char 8\)|EOF while parsing a string' "$TR" 2>/dev/null || echo 0)
    printf '%s\t%s\t%d\t%s\t%s\t%s\n' "$tag" "$BLOCK" "$i" "$VERD" "$C8" "$RLOG" >> "$SUMMARY"
    echo "  => verdict=$VERD char8=$C8"
    docker rm -f "fr13-bigdenom-$ARM" >/dev/null 2>&1 || true
    # incremental tally after every rollout (so progress is visible / early-stoppable)
    .venv/bin/python scripts/fr13_blocksweep_reduce.py "$SUMMARY" 2>/dev/null || true
  done
done
echo "=== BLOCK SWEEP DONE @ $(date -u +%H:%M:%S) -> $SUMMARY ==="
.venv/bin/python scripts/fr13_blocksweep_reduce.py "$SUMMARY"

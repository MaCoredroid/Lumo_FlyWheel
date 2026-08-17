#!/usr/bin/env bash
# FR14 composed-arm BOOT PROBE -- shake out the max-stack arm at 10-minute cost.
#
# WHY A BOOT PROBE AND NOT THE 1-TASK DIAGNOSTIC SUBSET. The composed arm cannot
# be exercised on config/fr13_fixed32/subset_b1_diagnostic_one.json at all: the
# campaign driver refuses any subset that is not 4 or 16 tasks; the 1-task subset
# only runs under FR13_FIXED32_B1_DIAGNOSTIC=1; and EVERY B1 production arm
# requires B1_DIAGNOSTIC == 0 plus the exact4 identity verbatim, as do both
# promoted-default scope guards. A diagnostic-mode run can exercise a lever only
# as a LIVE A/B arm -- which is what the byte gate already does -- never as a
# production arm, so it would test the wrong path even if it ran.
#
# So this probes the REAL production identity and simply stops before draining:
# boot exact4 with every lever armed, assert each engagement artifact and each
# inert path's zero-needle, tear down. ~10-12 minutes, and a defect costs that
# instead of a 2-hour serve.
#
# THIS IS NOT ACCEPTANCE EVIDENCE. It never completes a task, so it yields no
# timing, acceptance, or resolve verdict -- only "does the composed arm boot and
# engage what it claims". The runroot is labelled accordingly.
#
# Usage:
#   FR13_FIXED32_GDN_SINGLE_LAUNCH_PASS_JSON=<single_launch credential> \
#   bash scripts/fr14_run_b1_composed_boot_probe.sh
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

TAG=${TAG:-bootprobe}
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "TAG unsafe" >&2; exit 2; }
PYTHON_BIN=${PYTHON_BIN:-.venv/bin/python}
FA2_DIR=${FA2_DIR:-/home/mark/fr13_fa2_qrow32_gqa_pair_b1_sm121a_20260810}
CANDIDATE_SO="$FA2_DIR/_vllm_fa2_qrow32_gqa_pair_b1_sm121a.abi3.so"
SUBSET=config/fr13_fixed32/subset_b4_four.json
MANDATORY_WEIGHT_BYTES=25430574256
MANDATORY_WEIGHT_FLOOR_MS=93.15228665201465
BOOT_WAIT_S=${BOOT_WAIT_S:-1500}
# How long to wait for ONE request to complete after the campaign begins.
REQUEST_WAIT_S=${REQUEST_WAIT_S:-900}

# Levers under test. gqa_pair arms itself from the run-local credential pointer
# via the promoted default, so it is deliberately NOT named here -- naming it
# would test a path production does not take.
SINGLE_LAUNCH_PASS_JSON=${FR13_FIXED32_GDN_SINGLE_LAUNCH_PASS_JSON:-}
PROBE_PREP_BAKE=${PROBE_PREP_BAKE:-1}
PROBE_DEFER=${PROBE_DEFER:-1}
PROBE_SINGLE_LAUNCH=${PROBE_SINGLE_LAUNCH:-1}
if [[ "$PROBE_SINGLE_LAUNCH" == "1" && -z "$SINGLE_LAUNCH_PASS_JSON" ]]; then
  echo "PROBE_SINGLE_LAUNCH=1 requires FR13_FIXED32_GDN_SINGLE_LAUNCH_PASS_JSON" >&2
  exit 2
fi

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUNROOT=output/fr14_composed_boot_probe_$TS
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ ! -e "$RUNROOT_ABS" ]] || { echo "RUNROOT must be new" >&2; exit 2; }
mkdir -p "$RUNROOT_ABS"
ARM="hydra27_fixed32_composed_${TAG}"
ARMDIR="$RUNROOT_ABS/$ARM"

SOURCE_COMMIT=$(git rev-parse HEAD)
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] \
  || { echo "tracked worktree must be clean" >&2; exit 2; }
[[ -z "$(docker ps -aq)" ]] || { echo "docker must be empty" >&2; exit 2; }
awk '/^MemFree:/{exit ($2/1048576 < 82.3)}' /proc/meminfo \
  || { echo "unified-memory preflight failed" >&2; exit 2; }

# ---- publish the fixed32 route pins (see the lever-pair runner's note) -------
export BSIZE=1 CONC=1 WALL=0
export FR13_DRAFT_VOCAB_ROOT=0 FR13_DRAFT_VOCAB_K=0
export FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"
export FR13_FA2_QROW32_B1_QUALIFICATION_PROFILE=full_vocab
export FR13_FIXED32_GDN_SINGLE_LAUNCH_QUALIFICATION_PROFILE=full_vocab
export FR13_FLOOR_ORDER=TH
source scripts/fr13_canonical_env.sh
run_variant() { :; }
source scripts/fr13_fixed32_floor_timers_seq.sh
unset -f run_variant
[[ "$FR13_DRAFT_VOCAB_K" == "0" && "$FR13_DRAFT_VOCAB_ROOT" == "0" \
   && "$FR13_FIXED32_TAW_WALK_CAP" == "12" \
   && "$FR13_MANDATORY_WEIGHT_BYTES" == "$MANDATORY_WEIGHT_BYTES" \
   && "$FR13_WEIGHT_FLOOR_MS" == "$MANDATORY_WEIGHT_FLOOR_MS" ]] \
  || { echo "K0 composed floor contract drifted" >&2; exit 2; }

printf 'classification=composed_arm_boot_probe_not_acceptance_evidence\n' \
  > "$RUNROOT_ABS/probe_meta.txt"
printf 'source_commit=%s\nstarted=%s\nlevers=gqa_pair(promoted),single_launch=%s,prep_bake=%s,defer=%s\n' \
  "$SOURCE_COMMIT" "$(date -u +%FT%TZ)" \
  "$PROBE_SINGLE_LAUNCH" "$PROBE_PREP_BAKE" "$PROBE_DEFER" \
  >> "$RUNROOT_ABS/probe_meta.txt"

# FR10_METRICS IS NOT A FREE CHOICE. single_launch's production predicate
# requires the metrics=1 class, but plain fixed32 REFUSES it: "fixed32 requires
# FR10_METRICS=0, got 1". So metrics tracks single_launch exactly -- and that is
# why a composed arm WITHOUT single_launch needs no metrics=1 control arm at all:
# it runs at metrics=0, the banked anchor's own setting, and pairs against it
# directly.
if [[ "$PROBE_SINGLE_LAUNCH" == "1" ]]; then PROBE_METRICS=1; else PROBE_METRICS=0; fi
lever_env=(
  # single_launch is NAMED because it is not promoted (its registry default is 0).
  FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION="$PROBE_SINGLE_LAUNCH"
  FR13_FIXED32_GDN_SINGLE_LAUNCH_PASS_JSON="$SINGLE_LAUNCH_PASS_JSON"
  FR10_METRICS="$PROBE_METRICS"
  FR13_HOST_TAIL_PREP_BAKE="$PROBE_PREP_BAKE"
  FR13_HOST_TAIL_DEFER="$PROBE_DEFER"
  FORKED_FA2_SO="$CANDIDATE_SO"
)

echo "===== COMPOSED BOOT PROBE $ARM $(date -u +%H:%M:%SZ) ====="
env RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S= \
  FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  "${lever_env[@]}" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" hydra27_fixed32 "$SUBSET" \
  > "$RUNROOT_ABS/$ARM.runlog" 2>&1 &
SERVE_PID=$!

# ---- wait for the boot to declare itself, then probe -------------------------
deadline=$(( $(date +%s) + BOOT_WAIT_S ))
booted=0
while (( $(date +%s) < deadline )); do
  if grep -q 'fixed32 ingress campaign begun after zero-work baseline' \
       "$RUNROOT_ABS/$ARM.runlog" 2>/dev/null; then
    booted=1; break
  fi
  kill -0 "$SERVE_PID" 2>/dev/null || break
  sleep 10
done
echo "[probe] booted=$booted $(date -u +%H:%M:%SZ)"

# ---- THE LAST NEEDLE: one request must SURVIVE ------------------------------
# BOOTS CLEAN != SERVES CLEAN. The first version of this probe stopped at
# "campaign begun" -- before any traffic -- and passed a configuration whose
# engine then died on its FIRST real request with "FR13 fixed32 began a forward
# before prior KV completion". Everything a boot can show was green; the
# observable that mattered was a request completing.
#
# Evidence is the engine's own ingress ledger, which records request_complete
# per request, cross-checked against the container still being alive. Polls
# until one of: a completed request (PASS), the engine dying (FAIL, with the
# fatal quoted), or the deadline.
served=0
if (( booted == 1 )); then
  CID=$(docker ps -q --filter "name=$ARM" | head -1)
  LEDGER="$ARMDIR/logs/fr13_fixed32_engine_ingress.jsonl"
  req_deadline=$(( $(date +%s) + REQUEST_WAIT_S ))
  while (( $(date +%s) < req_deadline )); do
    if sudo -n grep -qs '"event":"request_complete"' "$LEDGER" 2>/dev/null \
       || grep -qs '"event":"request_complete"' "$LEDGER" 2>/dev/null; then
      served=1; break
    fi
    if [[ -z "$(docker ps -q --filter "name=$ARM")" ]]; then
      echo "[probe] ENGINE DIED before completing a request:"
      docker logs "$CID" 2>&1 | grep -E 'RuntimeError|EngineDeadError|FR13 fixed32' \
        | tail -5 | sed 's/^/    /'
      break
    fi
    sleep 10
  done
fi
if (( served == 1 )); then
  echo "[probe] SERVED   one request completed -- engine survived real traffic"
else
  echo "[probe] NOT SERVED  no request completed within ${REQUEST_WAIT_S}s"
fi

fail=0
(( served == 1 )) || fail=1
need_file() {  # path label
  if [[ -f "$1" && ! -L "$1" ]]; then
    echo "  ENGAGED  $2"
  else
    echo "  MISSING  $2  ($1)"; fail=1
  fi
}
need_zero() {  # needle label
  local n
  n=$(grep -rc "$1" "$ARMDIR/launch.log" "$ARMDIR"/logs/* 2>/dev/null \
        | awk -F: '{s+=$2} END{print s+0}')
  if [[ "$n" == "0" ]]; then echo "  INERT    $2 (0 lines)"; else echo "  LEAKED   $2 ($n lines)"; fail=1; fi
}

echo "[probe] ---- engagement artifacts ----"
need_file "$ARMDIR/logs/fr13_fa2_qrow32_b1_production_pass.json" "gqa_pair launcher-minted credential"
need_file "$ARMDIR/logs/fr13_fa2_qrow32_b1_production_engagement.json" "gqa_pair engagement"
if [[ "$PROBE_SINGLE_LAUNCH" == "1" ]]; then
  need_file "$ARMDIR/logs/fr13_fixed32_gdn_single_launch.production_credential.json" \
    "single_launch production credential"
fi
echo "[probe] ---- inert paths (K0 retires the DVK shim) ----"
need_zero '\[FR13_DRAFT_VOCAB\]' "draft-vocab subset head"
need_zero '\[FR14_DVK_DEQUANT\]' "Phase-1 boot dequant"
echo "[probe] ---- selector decisions ----"
grep -hE 'promoted default|INCUMBENT|STALE|single-launch production' \
  "$ARMDIR/launch.log" 2>/dev/null | sed 's/^/  /' || true
echo "[probe] ---- lever flags actually forwarded into the container ----"
# PREP_BAKE and DEFER are PATCH-TIME source transforms with no runtime needle
# (results/fr13_host_residual_20260811/design.md: their served-path arming is
# "offline-verified only"). What IS verifiable live is that the flags reached the
# container, so assert that rather than claiming an engagement we cannot observe.
CID=$(docker ps -q --filter "name=$ARM" | head -1)
if [[ -n "$CID" ]]; then
  docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CID" 2>/dev/null \
    | grep -E '^(FR13_HOST_TAIL_PREP_BAKE|FR13_HOST_TAIL_DEFER|FR10_METRICS|FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION)=' \
    | sed 's/^/  /' | tee "$RUNROOT_ABS/forwarded_lever_flags.txt"
else
  echo "  (container gone; cannot read forwarded flags)"; fail=1
  CID=$(docker ps -aq | head -1)   # may be an exited container; logs still readable
fi

echo "[probe] ---- teardown ----"
# CAPTURE BEFORE DESTROYING. A previous cycle removed a failed container before
# reading its docker logs and cost an extra reproduction boot to recover the
# error. The container is the only place the engine's own stdout lives.
if [[ -n "$CID" ]]; then
  docker logs "$CID" > "$RUNROOT_ABS/container_docker_logs.txt" 2>&1 || true
  echo "  captured container logs: $(wc -l < "$RUNROOT_ABS/container_docker_logs.txt" 2>/dev/null || echo 0) lines"
fi
kill "$SERVE_PID" 2>/dev/null || true
wait "$SERVE_PID" 2>/dev/null
[[ -n "$CID" ]] && docker rm -f "$CID" >/dev/null 2>&1
docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1
sync; sudo -n sysctl vm.drop_caches=3 >/dev/null 2>&1 || true
awk '/^MemFree:/{printf "[probe] MemFree=%.2fGiB\n",$2/1048576}' /proc/meminfo

printf 'booted=%s\nserved_one_request=%s\nverdict=%s\nended=%s\n' \
  "$booted" "$served" "$([[ $booted == 1 && $fail == 0 ]] && echo CLEAN || echo DEFECTS)" \
  "$(date -u +%FT%TZ)" | tee -a "$RUNROOT_ABS/probe_meta.txt"
echo "[probe] done -> $RUNROOT_ABS"
(( booted == 1 && served == 1 && fail == 0 )) || exit 1
exit 0

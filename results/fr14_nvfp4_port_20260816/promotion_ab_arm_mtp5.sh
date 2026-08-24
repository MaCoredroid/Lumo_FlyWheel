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
  # THE TRIGGER TASK. Minted at 0577cb4b1; the QC has since produced one clean
  # merged-drafter sample of 14369 (Cqc10, resolved, c5 0.5863) to compare against.
  astropy14369) SUBSET=config/fr13_fixed32/subset_b1_diagnostic_astropy14369.json ;;
  *) echo "no B1 diagnostic profile for $MTP5_TASK" >&2
     exit 2 ;;
esac
[[ -f "$SUBSET" && ! -L "$SUBSET" ]] || { echo "subset missing: $SUBSET" >&2; exit 2; }

# THE MODEL PIN, now a parameter rather than a literal (lane 4, 5a0cf9698/39b7afcb3).
# The native launchers used to hardcode /models/qwen3.6-27b-fp8; serving THAT under a
# 3.8 label would have been the worst artifact this probe could produce, which is why
# the earlier refire was refused rather than run. These names are deliberately the
# launcher's own plain spellings (NOT FR13_*), read off its landing at :56-61.
MTP5_MODEL_PATH=${MTP5_MODEL_PATH:-/models/qwen3.8-27b-nvfp4-radixark}
MTP5_MODEL_NAME=${MTP5_MODEL_NAME:-qwen3.8-27b-nvfp4-radixark}
[[ -d "$MTP5_MODEL_PATH" && ! -L "$MTP5_MODEL_PATH" ]] \
  || { echo "served checkpoint missing or symlinked: $MTP5_MODEL_PATH" >&2; exit 2; }

# CHECKPOINT IDENTITY, asserted here so the arm refuses BEFORE the GPU rather than
# discovering the wrong weights in provenance afterwards. Same construction the
# launcher uses (:110-112): sorted "name size" over top-level safetensors.
#
# WHAT THIS DIGEST DOES AND DOES NOT PROVE, measured rather than assumed:
#   /models/qwen3.8-27b-nvfp4-radixark            -> 5ec8e24087f2e395
#   /models/qwen3.8-27b-nvfp4-radixark-asshipped  -> 5ec8e24087f2e395   (SAME)
#   /models/qwen3.6-27b-fp8                       -> d053823784d5db6f
# It cleanly separates the 3.8 NVFP4 weights from the 3.6 FP8 ones, which is the
# confusion that blocked this probe. It does NOT separate the two radixark dirs --
# but that is correct here, not a gap: their safetensors are the SAME INODES
# (hardlinked, verified), so the weights genuinely are identical bytes and the dirs
# differ only in non-safetensors files the digest does not cover.
MTP5_EXPECT_CKPT=${MTP5_EXPECT_CKPT:-5ec8e24087f2e395}
_ckpt_id=$(find "$MTP5_MODEL_PATH" -maxdepth 1 -name '*.safetensors' -printf '%f %s\n' \
             2>/dev/null | sort | sha256sum | cut -d' ' -f1)
[[ "${_ckpt_id:0:16}" == "$MTP5_EXPECT_CKPT" ]] \
  || { echo "CHECKPOINT IDENTITY MISMATCH: $MTP5_MODEL_PATH -> ${_ckpt_id:0:16}, expected $MTP5_EXPECT_CKPT" >&2
       exit 2; }
echo "[mtp5] checkpoint identity ${_ckpt_id:0:16} OK -- serving $MTP5_MODEL_PATH as '$MTP5_MODEL_NAME'"

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
# PYTHONPATH, and this is a finding rather than a convenience.
# The shared .venv's editable install points at /home/mark/shared/lumoFlyWheel/src
# -- a DIFFERENT checkout, which has no model_server.py -- so an unqualified
# `import lumo_flywheel_serving` from this repo resolves to FOREIGN CODE. Every
# sibling script sets PYTHONPATH="$PWD/src" inline for exactly this reason; the
# native launcher (fr10_launch_speed_server.sh:186) does not, and replicate A died
# at 7s on ModuleNotFoundError: lumo_flywheel_serving.model_server. Supplying the
# correct path here is what the siblings do; the launcher's omission is reported.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# REPO, and this is the SAME hazard in a third form -- serious enough to state plainly.
# The NATIVE launchers hardcode a FOREIGN checkout as their default:
#   fr10_launch_speed_server.sh:4      REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
#   fr13_launch_native_mtp_server.sh   REPO=${REPO:-/home/mark/shared/lumoFlyWheel}
# while their fixed32-family siblings derive it from their own location:
#   fr13_launch_forked_fa2_tree_server.sh  REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
#   fr14_leg3_launch_nomiddleware.sh       REPO=${REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}
# So a native arm launched from THIS port silently mounts ANOTHER repo at /workspace.
# Replicate A died on it: the chat template exists here at
# docker/chat_templates/qwen3-openai-codex.jinja but not in that checkout. REPO is a
# documented caller override, so setting it is correct use, not a workaround -- but a
# native arm that nobody overrode would have run FOREIGN CODE without saying so.
export REPO
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
  printf 'DECLARED_EXCEPTION=NVFP4 lm_head loader shim (fr14_patch_nvfp4_lmhead.py), WEIGHT LOADING ONLY -- ruled Option A, pass 209\n'
  printf 'declared_exception_shim_sha256=%s\n' "$(sha256sum scripts/fr14_patch_nvfp4_lmhead.py | cut -d" " -f1)"
  printf 'declared_exception_why=stock vLLM cannot load this checkpoint at all: its lm_head is quantized (input_scale/weight_scale/weight_scale_2) and Qwen3_5ForCausalLM declares only lm_head.weight. Replicate A died on exactly this at 12:08:49Z.\n'
  printf 'declared_exception_scope=constructor wiring, quant-method dispatch, key remapping, numel-preserving reshape. Touches NO decode path, attention, drafter or speculative decoding -- so it cannot bias a drafter-neutrality result.\n'
  printf 'purity_meaning=NO SIDE CODE ON THE DECODE PATH (not "no side code"). The attestor enumerates the shim tokens and excepts exactly those; any other sentinel still fails, including a non-shim blob inside a shim-target file.\n'
  printf 'c5_applicable=NO -- c5 is a SEAM conditional and a chain drafter has no seam\n'
  printf 'c5_note=do NOT quote a c5 for this arm. On the merged tree drafter c5 flagged exactly one of the canonical sixteen (13236 at 0.3499, all others 0.5182-0.6395); that instrument does not transfer to a chain drafter and its absence here is by construction, not an omission.\n'
  printf 'served_model_path=%s\nserved_model_name=%s\ncheckpoint_identity=%s\n' \
    "$MTP5_MODEL_PATH" "$MTP5_MODEL_NAME" "${_ckpt_id:0:16}"
  printf 'checkpoint_note=5ec8e240 separates the 3.8 NVFP4 weights from 3.6-fp8 (d0538237). It does NOT separate radixark from radixark-asshipped: their safetensors are the same inodes, so the weights are identical bytes and only non-safetensors files differ.\n'
  printf 'degeneration_instruments=trace screens (ttr, top-12-gram, max-block, tool cadence) + 24k ceiling length events + conjunction rule (visible-collapse AND tools==0 AND thinking-large)\n'
  printf 'conjunction_rule_validation=130 banked traces: 1 degeneration (Cqc16/13236), 9 vacuous, 120 clean; ONE positive example, so it is a floor not a proof\n'
  printf 'n2_framing=PRE-REGISTERED: either replicate degenerating strongly REFUTES the machinery hypothesis; both clean is WEAK evidence for it\n'
  printf 'output_ceiling=%s\n' "${DEPLOY_MAX_OUTPUT_TOKENS:-24000}"
  printf 'boot_head=%s\nstarted=%s\n' "$(git rev-parse HEAD)" "$(date -u +%FT%TZ)"
} > "$RUNROOT_ABS/MTP5_PROBE.txt"

# PURITY ATTESTOR. Mark: "the plain native kernel, without any of our side code."
# The arm CHOOSES nativemtp5; this OBSERVES that the choice took effect, from inside
# the live container, and writes MTP5_PURITY.json into the runroot. It must run while
# the engine is up, so it waits for the container rather than running after the serve.
(
  for _i in $(seq 1 180); do
    _c=$(docker ps --format "{{.Names}}" | head -1)
    if [[ -n "$_c" ]]; then
      sleep 90   # let the engine finish loading before censusing its maps
      bash results/fr14_nvfp4_port_20260816/promotion_ab_mtp5_purity.sh \
        "$_c" "$RUNROOT_ABS" "$RUNROOT_ABS/$ARM.runlog" \
        > "$RUNROOT_ABS/purity_attestor.log" 2>&1
      exit 0
    fi
    sleep 5
  done
  echo "purity attestor: container never appeared" > "$RUNROOT_ABS/purity_attestor.log"
) &
_PURITY_PID=$!

# ARTIFACT RETENTION. ARMDIR="$RUNROOT/$ARM" does NOT survive teardown -- replicate A
# returned a 2-second vacuous "failed" and every per-task artifact that could have
# explained it went with the arm dir. Mirror the tree to teardown-safe ground while
# the arm is alive, so an agent-side failure is diagnosable and a success verifiable.
# pretask_identity.json is copied on EVERY tick, so it survives the clean (swerc=0)
# path too -- its absence there was one of the two unverifieds last round.
(
  _keep="$RUNROOT_ABS/_retained"; mkdir -p "$_keep"
  for _i in $(seq 1 4000); do
    if [[ -d "$RUNROOT_ABS/$ARM" ]]; then
      cp -a "$RUNROOT_ABS/$ARM/." "$_keep/" 2>/dev/null || true
    fi
    [[ -z "$(docker ps -q)" ]] && sleep 3 && \
      { [[ -d "$RUNROOT_ABS/$ARM" ]] && cp -a "$RUNROOT_ABS/$ARM/." "$_keep/" 2>/dev/null; exit 0; }
    sleep 10
  done
) &
_RETAIN_PID=$!

echo "===== $ARM (drafter-neutrality probe) $(date -u +%FT%TZ) ====="

# NO FR13_FIXED32_B1_DIAGNOSTIC BELOW, and this is not an omission.
# The diagnostic route is a FIXED32 concept. The variant sets FIXED32_MODE only
# for the fixed32 kinds (:519/:534), so on LAUNCHER=native it stays empty, the
# --fixed32-* runner args are never passed (:2824), fixed32_enabled is False, and
# run_swe_bench_q36_a.py:9981 hard-errors: 'FR13_FIXED32_B1_DIAGNOSTIC=1 requires
# fixed32 runtime binding'. Single-task selection here comes from the SUBSET
# alone. Verified by reading the gate rather than by burning a boot on it.
#
# THIS COMMENT LIVES ABOVE THE COMMAND, NOT INSIDE IT, and that is load-bearing.
# It used to sit between two backslash-continued lines of the `env` invocation. A
# comment does NOT continue a continuation: bash ended the command at the `#`, so
# `env RUNROOT=... OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1
# AGENT_WALL_S=9000` ran with NO COMMAND -- it printed the environment and exited 0 --
# and the serve then ran as a SEPARATE command with those five variables UNSET.
# OFFLOAD_AGENT=1 unset would have routed the agent off the offload proxy that every
# other arm uses, and RUNROOT unset would have scattered the artifacts, while the arm
# reported success. Reproduced in isolation before fixing. It never fired only because
# this probe has never reached a boot: the model pin, then DIAGNOSTIC, then /tokenize,
# then the wrong weights each refused first. Exactly the silent-no-op this script's own
# PROMOAB_EXTRA_ENV comment warns about, sitting in the script that warns about it.
env RUNROOT="$RUNROOT_ABS" \
  OFFLOAD_AGENT=1 MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1 AGENT_WALL_S=9000 \
  FR10_METRICS=0 \
  SKIP_WARMUP_PROBE=1 \
  FR13_PROXY_RAW_DUMPS=off \
  SERVED_MODEL_PATH="$MTP5_MODEL_PATH" \
  SERVED_MODEL_NAME="$MTP5_MODEL_NAME" \
  FR13_NATIVE_NVFP4_LMHEAD_SHIM=1 \
  TMPDIR=/home/mark/shared/tmp-scratch \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh "$ARM" nativemtp5 "$SUBSET" \
  > "$RUNROOT_ABS/$ARM.runlog" 2>&1
rc=$?
echo "[mtp5] serve rc=$rc $(date -u +%FT%TZ)"
printf 'serve_rc=%s\nended=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$RUNROOT_ABS/MTP5_PROBE.txt"
echo "[mtp5] containers after: $(docker ps -aq | wc -l)"
echo "[mtp5] done -> $RUNROOT_ABS"

#!/bin/bash
# FR13 CODEX-OFFLOAD orchestration (runs on the GB10). Subcommands:
#
#   sync   <host>                  rsync proxy code (src/ + registry + remote launcher) to alienware
#   start  <host> <gb10_ip> <armdir>   launch the proxy on alienware -> GB10:9950, health-check it
#   control <host> <preflight|begin|finalize>  fixed32 proxy ingress control
#   fetch  <host> <armdir>        fetch proxy artifacts back into <armdir>
#   stop   <host>                 kill the remote proxy on alienware
#
# WHY: the deployment harness puts (2) the inference_proxy and (3) the codex-runner
# docker on the GB10, where they steal unified-memory bandwidth (273 GB/s Grace+
# Blackwell) + CPU from vLLM and CONTAMINATE the timing-sensitive deploy-speed
# numbers. Moving (2)+(3) to alienware leaves the GB10 running ONLY vLLM. The
# lossless (argmax/distributional) numbers are timing-independent so unaffected;
# only speed is. The canonical measurement BASIS is unchanged — only WHERE the
# codex+proxy compute runs. vLLM stays on the GB10 (the GPU).
#
# Reuses the eval-offload SSH/rsync plumbing pattern (run_swe_bench_q36_a.py:392-432).
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}}
REPO_ROOT=$(cd "$REPO_ROOT" && pwd)
REMOTE_ROOT='~/lumo_proxy_offload'                 # repo mirror on alienware
REMOTE_REPO="$REMOTE_ROOT/repo"
REMOTE_VENV='~/swe_eval_offload/venv/bin/python'   # reuse the eval-offload venv (has requests+yaml)
REMOTE_PAIR_DUMPS="$REMOTE_ROOT/pair_dumps"
REMOTE_REQ_DUMPS="$REMOTE_ROOT/request_dumps"
REMOTE_FIXED32_SECRET="$REMOTE_ROOT/fixed32_ingress_secret"
REMOTE_FIXED32_LEDGER="$REMOTE_ROOT/fixed32_proxy_ingress.jsonl"

# ---------------------------------------------------------------- SITE 22
# These two checks used to be ONE regex per shape, with the count expressed as
# a repetition quantifier -- `{3}` for four ids, `{15}` for sixteen. The
# literals 4 and 16 never appeared, so every count-scan in this campaign was
# blind to them by construction, and this file is a fourth root no sweep
# covered. exact16_minus_13236 was accepted by name upstream and would have
# been refused here for having fifteen.
#
# DECOMPOSED. The regex now validates FORMAT only -- what an id looks like,
# one or more of them -- and the COUNT is checked separately against the
# authority's admissible set. A count rule hidden inside a quantifier is a
# rule no reader and no scanner can see.
_fixed32_admissible_task_counts() {
  python3 -c 'import sys
sys.path.insert(0, sys.argv[1])
from fr13_floor_gate import EVIDENCE_SETS
print(",".join(str(count) for count in sorted(EVIDENCE_SETS)))' "$REPO_ROOT/scripts"
}

# $1 = label used in refusals. Fail-closed throughout: an unreadable authority
# refuses rather than falling back to a literal.
_fixed32_require_canonical_task_ids() {
  local label=$1 _count _allowed
  [[ "$FIXED32_TASK_IDS" =~ ^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+(,[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+)*$ ]] \
    || { echo "FAIL: fixed32 $label task IDs are malformed"; exit 5; }
  if [[ "$FIXED32_B1_DIAGNOSTIC" == "1" ]]; then
    # The single-task diagnostic mode is expressed through its DECLARED mode,
    # not through a count. The 13236 attribution study rides this branch.
    # The wording is preserved verbatim from the two call sites this helper
    # replaced: tests/test_fr13_fixed32_b1_diagnostic.py guards these strings
    # across every ingress, and a refactor that renames a refusal silently
    # removes the guard that was watching it.
    [[ "$FIXED32_TASK_IDS" == "$FIXED32_DIAGNOSTIC_TASK_ID" ]] \
      || { echo "FAIL: fixed32 B1 diagnostic $label task ID is not pinned"; exit 5; }
    return 0
  fi
  _count=$(awk -F, '{print NF}' <<< "$FIXED32_TASK_IDS")
  _allowed=$(_fixed32_admissible_task_counts) \
    || { echo "FAIL: fixed32 $label cannot read the canonical evidence-set counts"; exit 5; }
  [[ -n "$_allowed" ]] \
    || { echo "FAIL: fixed32 $label read an empty canonical evidence-set list"; exit 5; }
  [[ ",$_allowed," == *",$_count,"* ]] \
    || { echo "FAIL: fixed32 $label task count $_count is not one of the" \
              "canonical evidence-set sizes ($_allowed)"; exit 5; }
}
# alienware-local proxy port. 8022 is OCCUPIED on alienware by an unrelated
# root-owned host service (the lumo-alpha-dev --network=host container env), so
# the offload proxy listens on 8023 there. This is alienware-LOCAL: the GB10
# side (vLLM :9950) and the legacy on-GB10 proxy (:8022) are unaffected.
PROXY_PORT=${LUMO_OFFLOAD_PROXY_PORT:-8023}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 \
          -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new)

CMD=${1:?usage: offload_codex_proxy.sh <sync|start|fetch|stop> ...}
shift
FIXED32_B1_DIAGNOSTIC=${FR13_FIXED32_B1_DIAGNOSTIC:-0}
B1_DIAGNOSTIC_TASK_PROFILE=${FR13_B1_DIAGNOSTIC_TASK_PROFILE:-astropy12907}
case "$FIXED32_B1_DIAGNOSTIC" in
  0|1) ;;
  *) echo "FAIL: FR13_FIXED32_B1_DIAGNOSTIC must be exactly 0 or 1"; exit 2 ;;
esac
# The diagnostic profile -> task id map belongs to the AUTHORITY
# (fr13_floor_gate.B1_DIAGNOSTIC_PROFILES). It was restated here as a literal
# `case`, which is what made adding one profile a six-file edit.
FIXED32_DIAGNOSTIC_TASK_ID=$(python3 -c 'import sys
sys.path.insert(0, sys.argv[2])
from fr13_floor_gate import B1_DIAGNOSTIC_PROFILES
profile = sys.argv[1]
if profile not in B1_DIAGNOSTIC_PROFILES:
    raise SystemExit(1)
print(B1_DIAGNOSTIC_PROFILES[profile]["task_ids"][0])' \
  "$B1_DIAGNOSTIC_TASK_PROFILE" "$REPO_ROOT/scripts") \
  || { echo "FAIL: FR13_B1_DIAGNOSTIC_TASK_PROFILE is unsupported"; exit 2; }
# Minted for the REMOTE block, which has no repo to read the authority from.
_FIXED32_DIAGNOSTIC_MAP=$(python3 -c 'import sys
sys.path.insert(0, sys.argv[1])
from fr13_floor_gate import B1_DIAGNOSTIC_PROFILES
# The field name arrives as argv rather than as a quoted literal: this whole
# program is inside a single-quoted bash string, where a backslash-escaped
# double quote is passed through verbatim and reaches Python as a syntax error.
field = sys.argv[2]
print(",".join(
    name + ":" + row[field][0]
    for name, row in sorted(B1_DIAGNOSTIC_PROFILES.items())
))' "$REPO_ROOT/scripts" task_ids) \
  || { echo "FAIL: cannot mint the diagnostic profile map"; exit 2; }
if [[ "$FIXED32_B1_DIAGNOSTIC" != "1" \
      && "$B1_DIAGNOSTIC_TASK_PROFILE" != "astropy12907" ]]; then
  echo "FAIL: alternate B1 task profile requires diagnostic mode"
  exit 2
fi

case "$CMD" in
  sync)
    HOST=${1:?host}
    # create the remote tree, then rsync only what the proxy needs.
    ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p $REMOTE_REPO/src $REMOTE_PAIR_DUMPS $REMOTE_REQ_DUMPS && echo ok" >/dev/null
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
      "$REPO_ROOT/src/lumo_flywheel_serving/" \
      "$HOST:$REMOTE_REPO/src/lumo_flywheel_serving/"
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$REPO_ROOT/model_registry.yaml" \
      "$REPO_ROOT/scripts/swe_x86_helpers/relaunch_proxy_remote.sh" \
      "$HOST:$REMOTE_REPO/"
    ssh "${SSH_OPTS[@]}" "$HOST" "chmod +x $REMOTE_REPO/relaunch_proxy_remote.sh && \
      $REMOTE_VENV -c 'import requests,yaml;print(\"proxy-deps-ok\")'" \
      || { echo "FAIL: alienware proxy deps missing (requests/yaml)"; exit 3; }
    echo "OFFLOAD_SYNC_OK host=$HOST repo=$REMOTE_REPO"
    ;;

  start)
    HOST=${1:?host}
    GB10_IP=${2:?gb10 tailscale ip}
    ARMDIR=${3:?armdir (for logging)}
    UPSTREAM="http://$GB10_IP:9950"
    # sanity: alienware MUST be able to reach the GB10 vLLM before we launch.
    if ! ssh "${SSH_OPTS[@]}" "$HOST" \
         "curl -fsS -m 6 $UPSTREAM/health >/dev/null 2>&1 && echo ok" 2>/dev/null | grep -q ok; then
      echo "FAIL: alienware cannot reach GB10 vLLM at $UPSTREAM/health (network/firewall)"; exit 4
    fi
    # launch the remote proxy with the canonical pins (forced deployment temp, pair dumps ON).
    # DEPLOY_FORCE_TEMP defaults to 0.6 = the REAL deployment temp (codex/q36-a, run_round5_b4_sweep
    # temp=0.6/top_p=0.95). temp 0.0 was WRONG for the speed/accept measurement (user 2026-06-16): the
    # tree tax is TEMP-DEPENDENT (~1.03x greedy -> ~1.4x t0.6) so greedy understates it, and at greedy a
    # lossless arm must byte-match native. Use DEPLOY_FORCE_TEMP=0.0 ONLY for the deterministic temp-0
    # argmax-flip lossless gate, never for the deployment speed/accept run.
    FORCE_TEMP="${DEPLOY_FORCE_TEMP:-0.6}"
    FIXED32_SECRET_LOCAL=${FR13_FIXED32_INGRESS_SECRET_FILE:-}
    FIXED32_TASK_IDS=${FR13_FIXED32_INGRESS_TASK_IDS:-}
    # RAW-DUMP MODE IS SELECTABLE, and no longer a consequence of identity.
    #
    # This was derived solely from whether a fixed32 secret was present, so the
    # only way for a NATIVE arm to get the fixed32 dump shape was to assert
    # fixed32 identity -- pins-as-fiction, refused, and the refusal ratified.
    # An arm can now say what it wants:
    #
    #   auto (default)  exactly the old behaviour: dumps off iff a fixed32
    #                   secret is present. Byte-unchanged for every existing arm.
    #   off             no raw dumps, without claiming a fixed32 identity.
    #   on              raw dumps, even alongside a fixed32 secret.
    #
    # Resolved ONCE here and recorded, so the ssh block below tests a decision
    # rather than re-deriving one from an identity that may not be this arm's.
    FIXED32_RAW_DUMPS_DISABLED=0
    FR13_PROXY_RAW_DUMPS=${FR13_PROXY_RAW_DUMPS:-auto}
    case "$FR13_PROXY_RAW_DUMPS" in
      auto|on|off) ;;
      *) echo "FAIL: FR13_PROXY_RAW_DUMPS must be auto, on, or off"; exit 2 ;;
    esac
    if [[ "$FIXED32_B1_DIAGNOSTIC" == "1" \
       && ( -z "$FIXED32_SECRET_LOCAL" || -z "$FIXED32_TASK_IDS" ) ]]; then
      echo "FAIL: fixed32 B1 diagnostic offload requires secret file and pinned task ID"
      exit 5
    fi
    if [[ -n "$FIXED32_SECRET_LOCAL" || -n "$FIXED32_TASK_IDS" ]]; then
      [[ -n "$FIXED32_SECRET_LOCAL" && -n "$FIXED32_TASK_IDS" ]] \
        || { echo "FAIL: fixed32 offload requires secret file and task IDs together"; exit 5; }
      [[ -f "$FIXED32_SECRET_LOCAL" && ! -L "$FIXED32_SECRET_LOCAL" \
         && "$(stat -c '%a' "$FIXED32_SECRET_LOCAL")" == "600" ]] \
        || { echo "FAIL: fixed32 offload secret is not a mode-600 regular file"; exit 5; }
      python3 - "$FIXED32_SECRET_LOCAL" <<'PY'
import json
import re
import sys
from pathlib import Path


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


try:
    payload = json.loads(
        Path(sys.argv[1]).read_text(encoding="ascii"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit(f"invalid fixed32 ingress secret JSON: {error}")
if (
    not isinstance(payload, dict)
    or set(payload)
    != {"schema", "task_hmac_key_hex", "engine_bearer"}
    or payload.get("schema") != "fr13-fixed32-ingress-secrets-v1"
    or not isinstance(payload.get("task_hmac_key_hex"), str)
    or re.fullmatch(r"[0-9a-f]{64}", payload["task_hmac_key_hex"]) is None
    or not isinstance(payload.get("engine_bearer"), str)
    or len(payload["engine_bearer"]) < 32
    or any(ord(char) < 33 or ord(char) > 126 for char in payload["engine_bearer"])
):
    raise SystemExit("fixed32 ingress secret JSON contract mismatch")
PY
      (( $? == 0 )) \
        || { echo "FAIL: fixed32 offload secret JSON contract is invalid"; exit 5; }
      _fixed32_require_canonical_task_ids offload
      REMOTE_FIXED32_SECRET_ARMED=1
      cleanup_remote_fixed32_secret_on_failure() {
        _fixed32_rc=$?
        if (( _fixed32_rc != 0 && REMOTE_FIXED32_SECRET_ARMED == 1 )); then
          ssh "${SSH_OPTS[@]}" "$HOST" \
            "rm -f $REMOTE_FIXED32_SECRET" >/dev/null 2>&1 || true
        fi
        return "$_fixed32_rc"
      }
      trap cleanup_remote_fixed32_secret_on_failure EXIT
      ssh "${SSH_OPTS[@]}" "$HOST" \
        "set -eu; rm -f $REMOTE_FIXED32_SECRET; umask 077; \
         cat > $REMOTE_FIXED32_SECRET; \
         test -f $REMOTE_FIXED32_SECRET; test ! -L $REMOTE_FIXED32_SECRET; \
         test \"\$(stat -c '%a' $REMOTE_FIXED32_SECRET)\" = 600" \
        < "$FIXED32_SECRET_LOCAL" >/dev/null \
        || { echo "FAIL: fixed32 offload secret transfer failed"; exit 5; }
      ssh "${SSH_OPTS[@]}" "$HOST" \
        "$REMOTE_VENV - $REMOTE_FIXED32_SECRET" <<'PY'
import json
import os
import re
import stat
import sys


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


path = os.path.expanduser(sys.argv[1])
info = os.lstat(path)
if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
    raise SystemExit("remote fixed32 secret must be a mode-600 regular file")
with open(path, encoding="ascii") as handle:
    payload = json.load(
        handle,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
if (
    not isinstance(payload, dict)
    or set(payload) != {"schema", "task_hmac_key_hex", "engine_bearer"}
    or payload.get("schema") != "fr13-fixed32-ingress-secrets-v1"
    or not isinstance(payload.get("task_hmac_key_hex"), str)
    or re.fullmatch(r"[0-9a-f]{64}", payload["task_hmac_key_hex"]) is None
    or not isinstance(payload.get("engine_bearer"), str)
    or len(payload["engine_bearer"]) < 32
    or any(ord(char) < 33 or ord(char) > 126 for char in payload["engine_bearer"])
):
    raise SystemExit("remote fixed32 ingress secret JSON contract mismatch")
PY
      (( $? == 0 )) \
        || { echo "FAIL: fixed32 offload remote secret validation failed"; exit 5; }
      ssh "${SSH_OPTS[@]}" "$HOST" \
        "test -f $REMOTE_FIXED32_SECRET && test ! -L $REMOTE_FIXED32_SECRET \
         && test \"\$(stat -c '%a' $REMOTE_FIXED32_SECRET)\" = 600 \
         && rm -f $REMOTE_FIXED32_LEDGER" \
        >/dev/null \
        || { echo "FAIL: fixed32 offload secret preparation failed"; exit 5; }
      FIXED32_RAW_DUMPS_DISABLED=1
    else
      ssh "${SSH_OPTS[@]}" "$HOST" \
        "rm -f $REMOTE_FIXED32_SECRET $REMOTE_FIXED32_LEDGER" \
        >/dev/null 2>&1 || true
    fi
    # The explicit mode overrides the identity-derived default, and the
    # resolution is recorded either way.
    _fr13_raw_dumps_origin=identity-derived
    case "$FR13_PROXY_RAW_DUMPS" in
      off) FIXED32_RAW_DUMPS_DISABLED=1; _fr13_raw_dumps_origin=explicit-off ;;
      on)  FIXED32_RAW_DUMPS_DISABLED=0; _fr13_raw_dumps_origin=explicit-on ;;
    esac
    echo "[offload] raw dumps disabled=$FIXED32_RAW_DUMPS_DISABLED" \
         "(mode=$FR13_PROXY_RAW_DUMPS, $_fr13_raw_dumps_origin," \
         "fixed32_secret=${FIXED32_SECRET_LOCAL:+present}${FIXED32_SECRET_LOCAL:-absent})"
    printf '{"schema":"fr13.proxy_raw_dumps.v1","mode":"%s","origin":"%s","disabled":%s,"fixed32_secret":"%s"}\n' \
      "$FR13_PROXY_RAW_DUMPS" "$_fr13_raw_dumps_origin" \
      "$( (( FIXED32_RAW_DUMPS_DISABLED == 1 )) && echo true || echo false )" \
      "${FIXED32_SECRET_LOCAL:+present}${FIXED32_SECRET_LOCAL:-absent}" \
      > "$ARMDIR/proxy_raw_dumps.json"
    # Legacy runs retain their served-stream captures. Formal fixed32 uses the
    # authenticated ingress/census ledgers instead and must not pay for or retain
    # full request/response dumps.
    if (( FIXED32_RAW_DUMPS_DISABLED == 0 )); then
      ssh "${SSH_OPTS[@]}" "$HOST" \
        "rm -rf $REMOTE_PAIR_DUMPS/* $REMOTE_REQ_DUMPS/* 2>/dev/null; \
         mkdir -p $REMOTE_PAIR_DUMPS $REMOTE_REQ_DUMPS" \
        >/dev/null 2>&1 || true
    fi
    # FR13 §59 offload stream-stall fix-stack (default OFF; ENABLE at a LATER relaunch,
    # not now). The three LUMO_PROXY_SSE_* envs below are wired-through with OFF defaults
    # (empty LOG/CAPTURE_DIR + HEARTBEAT_S=0 => byte-identical relay). To turn them on at
    # the next relaunch, export before invoking this script (do NOT add a '#'-comment
    # INSIDE the quoted ssh env block below — it would comment out the launch command):
    #   export LUMO_PROXY_SSE_LOG=1                                  # per-request terminal-reason -> offload_proxy.log
    #   export LUMO_PROXY_SSE_CAPTURE_DIR="$REMOTE_ROOT/sse_capture" # per-chunk-timestamped chat-path capture jsonl
    #   export LUMO_PROXY_SSE_HEARTBEAT_S=15                         # empty-delta heartbeat on upstream idle (chat only)
    ssh "${SSH_OPTS[@]}" "$HOST" "\
      if [ \"$FIXED32_RAW_DUMPS_DISABLED\" = \"1\" ]; then \
        unset LUMO_PROXY_PAIR_DUMP_DIR LUMO_PROXY_REQUEST_DUMP_DIR; \
        export LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS=1; \
      else \
        unset LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS; \
        export LUMO_PROXY_PAIR_DUMP_DIR=$REMOTE_PAIR_DUMPS; \
        export LUMO_PROXY_REQUEST_DUMP_DIR=$REMOTE_REQ_DUMPS; \
      fi; \
      if [ -n \"${FIXED32_SECRET_LOCAL:+1}\" ]; then \
        test -f $REMOTE_FIXED32_SECRET \
        && test ! -L $REMOTE_FIXED32_SECRET \
        && test \"\$(stat -c '%a' $REMOTE_FIXED32_SECRET)\" = 600 \
        || exit 72; \
      fi; \
      LUMO_PROXY_OFFLOAD_REPO=$REMOTE_REPO \
      LUMO_PROXY_OFFLOAD_VENV=$REMOTE_VENV \
      LUMO_PROXY_UPSTREAM_BASE_URL=$UPSTREAM \
      LUMO_PROXY_LISTEN_HOST=127.0.0.1 LUMO_PROXY_LISTEN_PORT=$PROXY_PORT \
      LUMO_PROXY_FORCE_TEMPERATURE=$FORCE_TEMP \
      LUMO_PROXY_FORCE_TOP_P=${LUMO_PROXY_FORCE_TOP_P:-} \
      LUMO_PROXY_FORCE_TOP_K=${LUMO_PROXY_FORCE_TOP_K:-} \
      LUMO_PROXY_FORCE_PRESENCE_PENALTY=${LUMO_PROXY_FORCE_PRESENCE_PENALTY:-} \
      LUMO_PROXY_FORCE_MIN_P=${LUMO_PROXY_FORCE_MIN_P:-} \
      LUMO_PROXY_QWEN_SAMPLING=${LUMO_PROXY_QWEN_SAMPLING:-1} \
      LUMO_PROXY_SSE_LOG=${LUMO_PROXY_SSE_LOG:-} \
      LUMO_PROXY_SSE_CAPTURE_DIR=${LUMO_PROXY_SSE_CAPTURE_DIR:-} \
      LUMO_PROXY_SSE_HEARTBEAT_S=${LUMO_PROXY_SSE_HEARTBEAT_S:-15} \
      LUMO_PROXY_THINK_BUDGET=${LUMO_PROXY_THINK_BUDGET:-} \
      LUMO_PROXY_THINK_CUTOFF=${LUMO_PROXY_THINK_CUTOFF:-} \
      LUMO_PROXY_ALLOW_MODELS_GET=${LUMO_PROXY_ALLOW_MODELS_GET:-} \
      LUMO_PROXY_FIXED32_SECRET_FILE=${FIXED32_SECRET_LOCAL:+$REMOTE_FIXED32_SECRET} \
      LUMO_PROXY_FIXED32_TASK_IDS=$FIXED32_TASK_IDS \
      LUMO_PROXY_FIXED32_LEDGER_PATH=${FIXED32_SECRET_LOCAL:+$REMOTE_FIXED32_LEDGER} \
      bash $REMOTE_REPO/relaunch_proxy_remote.sh" \
      > "$ARMDIR/offload_proxy_launch.log" 2>&1 \
      || { echo "FAIL: remote proxy launch rc=$?"; cat "$ARMDIR/offload_proxy_launch.log"; exit 5; }
    # health-check the remote proxy over SSH (it listens on alienware 127.0.0.1).
    P0=$(date +%s); OK=0
    while (( $(date +%s) < P0 + 60 )); do
      CODE=$(ssh "${SSH_OPTS[@]}" "$HOST" \
        "curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PROXY_PORT/v1/models 2>/dev/null" 2>/dev/null)
      if [[ -n "$CODE" && "$CODE" != "000" ]]; then OK=1; break; fi
      sleep 2
    done
    (( OK == 1 )) || { echo "FAIL: remote proxy not healthy on $HOST:$PROXY_PORT"; exit 5; }
    # capture the remote proxy env for the class-9 pin assertions (mirrors on-GB10).
    ssh "${SSH_OPTS[@]}" "$HOST" "p=\$(cat /tmp/lumo_offload_proxy_${PROXY_PORT}.pid 2>/dev/null); \
      [ -r /proc/\$p/environ ] && tr '\0' '\n' < /proc/\$p/environ | grep -E '^LUMO_(PROXY|TRACK_B)' | sort" \
      > "$ARMDIR/offload_proxy_env.txt" 2>/dev/null || true
    grep -q "LUMO_PROXY_FORCE_TEMPERATURE=$FORCE_TEMP" "$ARMDIR/offload_proxy_env.txt" \
      || { echo "FAIL: remote proxy temp pin missing/mismatch (class 9; expected $FORCE_TEMP)"; exit 5; }
    if (( FIXED32_RAW_DUMPS_DISABLED == 1 )); then
      grep -q "^LUMO_PROXY_FIXED32_DISABLE_RAW_DUMPS=1$" \
        "$ARMDIR/offload_proxy_env.txt" \
        || { echo "FAIL: fixed32 raw-dump disable pin missing"; exit 5; }
      if grep -q -E "^LUMO_PROXY_(PAIR|REQUEST)_DUMP_DIR=" \
          "$ARMDIR/offload_proxy_env.txt"; then
        echo "FAIL: fixed32 proxy retained a raw-dump directory"
        exit 5
      fi
    else
      grep -q "^LUMO_PROXY_PAIR_DUMP_DIR=$REMOTE_PAIR_DUMPS$" \
        "$ARMDIR/offload_proxy_env.txt" \
        || { echo "FAIL: remote proxy pair-dump pin missing (class 9)"; exit 5; }
      grep -q "^LUMO_PROXY_REQUEST_DUMP_DIR=$REMOTE_REQ_DUMPS$" \
        "$ARMDIR/offload_proxy_env.txt" \
        || { echo "FAIL: remote proxy request-dump pin missing (class 9)"; exit 5; }
    fi
    grep -q "LUMO_PROXY_AUTO_CONTINUE=0" "$ARMDIR/offload_proxy_env.txt" \
      || { echo "FAIL: remote proxy NUDGE NOT DISABLED — LUMO_PROXY_AUTO_CONTINUE must be 0 for the honest give-up gate (nudge confounds it)"; exit 5; }
    REMOTE_FIXED32_SECRET_ARMED=0
    trap - EXIT
    echo "OFFLOAD_PROXY_OK host=$HOST upstream=$UPSTREAM (forced temp $FORCE_TEMP, raw_dumps_disabled=$FIXED32_RAW_DUMPS_DISABLED, NUDGE OFF)"
    ;;

  control)
    HOST=${1:?host}
    ACTION=${2:?preflight, begin, or finalize}
    [[ "$ACTION" == "preflight" || "$ACTION" == "begin" \
       || "$ACTION" == "finalize" ]] \
      || {
        echo "FAIL: fixed32 proxy control action must be preflight, begin, or finalize"
        exit 2
      }
    FIXED32_TASK_IDS=${FR13_FIXED32_INGRESS_TASK_IDS:-}
    [[ -n "$FIXED32_TASK_IDS" ]] \
      || { echo "FAIL: fixed32 proxy control requires canonical task IDs"; exit 5; }
    _fixed32_require_canonical_task_ids proxy-control
    # SITE 23, second instance. The block below runs on the REMOTE host under
    # its own venv and cannot import fr13_floor_gate, so the admissible counts
    # are minted HERE from the authority and passed across with the rest of the
    # run class. Never a literal on either side of the ssh.
    _FIXED32_ADMISSIBLE_COUNTS=$(_fixed32_admissible_task_counts) \
      || { echo "FAIL: fixed32 proxy control cannot read the canonical evidence-set counts"; exit 5; }
    [[ "$_FIXED32_ADMISSIBLE_COUNTS" =~ ^[0-9]+(,[0-9]+)*$ ]] \
      || { echo "FAIL: fixed32 proxy control minted a malformed evidence-set list"; exit 5; }
    [[ "$PROXY_PORT" =~ ^[0-9]+$ ]] \
      || { echo "FAIL: fixed32 proxy control port is malformed"; exit 5; }
    ssh "${SSH_OPTS[@]}" "$HOST" \
      "$REMOTE_VENV - $PROXY_PORT $ACTION $REMOTE_FIXED32_SECRET $FIXED32_TASK_IDS $FIXED32_B1_DIAGNOSTIC $B1_DIAGNOSTIC_TASK_PROFILE $_FIXED32_ADMISSIBLE_COUNTS $_FIXED32_DIAGNOSTIC_MAP" <<'PY'
import hashlib
import json
import os
import re
import secrets
import stat
import sys

import requests


def reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


port = int(sys.argv[1])
action = sys.argv[2]
secret_path = os.path.expanduser(sys.argv[3])
task_ids = sys.argv[4].split(",")
diagnostic_text = sys.argv[5]
diagnostic_profile = sys.argv[6]
# Minted by the caller from fr13_floor_gate.EVIDENCE_SETS; this host has no
# repo to read it from, so an absent or malformed list is fatal rather than
# replaced by a default -- a default here would be the literal tuple again.
if len(sys.argv) < 8 or not re.fullmatch(r"[0-9]+(,[0-9]+)*", sys.argv[7]):
    raise SystemExit("remote fixed32 admissible task counts are missing or malformed")
admissible_task_counts = tuple(int(part) for part in sys.argv[7].split(","))
if diagnostic_text not in {"0", "1"}:
    raise SystemExit("remote fixed32 B1 diagnostic selector is invalid")
diagnostic = diagnostic_text == "1"
# Minted by the caller from fr13_floor_gate.B1_DIAGNOSTIC_PROFILES; this host
# has no repo to read it from, so an absent or malformed map is fatal rather
# than replaced by a literal -- a literal here is what site 25 was.
if len(sys.argv) < 9 or not re.fullmatch(
    r"[A-Za-z0-9_]+:[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+"
    r"(,[A-Za-z0-9_]+:[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+)*", sys.argv[8]
):
    raise SystemExit("remote fixed32 diagnostic profile map is missing or malformed")
diagnostic_task_ids = {
    pair.split(":", 1)[0]: [pair.split(":", 1)[1]]
    for pair in sys.argv[8].split(",")
}
if diagnostic_profile not in diagnostic_task_ids:
    raise SystemExit("remote fixed32 B1 diagnostic task profile is invalid")
info = os.lstat(secret_path)
if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
    raise SystemExit("remote fixed32 secret changed before proxy control")
with open(secret_path, encoding="ascii") as handle:
    secret = json.load(
        handle,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
if (
    not isinstance(secret, dict)
    or set(secret) != {"schema", "task_hmac_key_hex", "engine_bearer"}
    or secret.get("schema") != "fr13-fixed32-ingress-secrets-v1"
):
    raise SystemExit("remote fixed32 secret contract mismatch")
if (
    len(task_ids) not in ((1,) if diagnostic else admissible_task_counts)
    or len(set(task_ids)) != len(task_ids)
    or (diagnostic and task_ids != diagnostic_task_ids[diagnostic_profile])
    or (not diagnostic and diagnostic_profile != "astropy12907")
):
    raise SystemExit("remote fixed32 task set does not match its run class")
canonical_task_set_sha256 = hashlib.sha256(
    json.dumps(
        sorted(task_ids),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
if action == "preflight":
    requests_seen = []
    wrong_bearer = "fr13_wrong_" + secrets.token_hex(32)
    for route in ("/v1/chat/completions", "/v1/responses"):
        for auth_case, headers in (
            ("missing_bearer", {}),
            ("wrong_bearer", {"Authorization": f"Bearer {wrong_bearer}"}),
        ):
            response = requests.post(
                f"http://127.0.0.1:{port}{route}",
                headers=headers,
                json={},
                timeout=30,
            )
            if response.status_code != 401:
                raise SystemExit(
                    "fixed32 proxy preflight did not reject "
                    f"{route} {auth_case}: HTTP {response.status_code}"
                )
            requests_seen.append(
                {
                    "route": route,
                    "auth_case": auth_case,
                    "status_code": response.status_code,
                }
            )
    denied_alternate_routes = []
    for route in ("/admin/invalidate", "/admin/load_tuned_config"):
        response = requests.post(
            f"http://127.0.0.1:{port}{route}",
            headers={"Authorization": f"Bearer {secret['engine_bearer']}"},
            json={},
            timeout=30,
        )
        if response.status_code != 403:
            raise SystemExit(
                "fixed32 proxy preflight did not deny legacy admin POST "
                f"{route}: HTTP {response.status_code}"
            )
        denied_alternate_routes.append(
            {"method": "POST", "route": route, "status_code": 403}
        )
    payload = {
        "schema": "fr13-fixed32-ingress-auth-preflight-v1",
        "role": "proxy",
        "rejected_requests": len(requests_seen),
        "accepted_requests": 0,
        "requests": requests_seen,
        "denied_alternate_routes": denied_alternate_routes,
    }
    print(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    raise SystemExit(0)
elif action == "begin":
    path = "/admin/fixed32/ingress/begin"
    body = {
        "schema": "fr13-fixed32-ingress-begin-v1",
        "canonical_task_count": len(task_ids),
        "canonical_task_set_sha256": canonical_task_set_sha256,
    }
    expected_schema = "fr13-fixed32-proxy-ingress-begin-v1"
else:
    path = "/admin/fixed32/ingress/finalize"
    body = {"schema": "fr13-fixed32-ingress-finalize-v1"}
    expected_schema = "fr13-fixed32-proxy-ingress-finalize-v1"
response = requests.post(
    f"http://127.0.0.1:{port}{path}",
    headers={"Authorization": f"Bearer {secret['engine_bearer']}"},
    json=body,
    timeout=30,
)
if response.status_code != 200:
    raise SystemExit(
        f"fixed32 proxy {action} failed with HTTP {response.status_code}"
    )
payload = json.loads(
    response.text,
    object_pairs_hook=reject_duplicate_keys,
    parse_constant=reject_nonfinite,
)
if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
    raise SystemExit(f"fixed32 proxy {action} response schema mismatch")
print(
    json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY
    (( $? == 0 )) \
      || { echo "FAIL: fixed32 remote proxy $ACTION control failed"; exit 6; }
    ;;

  fetch)
    HOST=${1:?host}
    ARMDIR=${2:?armdir}
    FIXED32_TASK_IDS=${FR13_FIXED32_INGRESS_TASK_IDS:-}
    if [[ -z "$FIXED32_TASK_IDS" ]]; then
      mkdir -p "$ARMDIR/proxy_pair_dumps" "$ARMDIR/proxy_request_dumps"
      # Legacy lossless rescoring still consumes these served-stream captures.
      rsync -az -e "ssh ${SSH_OPTS[*]}" \
        "$HOST:$REMOTE_PAIR_DUMPS/" "$ARMDIR/proxy_pair_dumps/" \
        2>/dev/null || true
      rsync -az -e "ssh ${SSH_OPTS[*]}" \
        "$HOST:$REMOTE_REQ_DUMPS/" "$ARMDIR/proxy_request_dumps/" \
        2>/dev/null || true
    fi
    # also pull the remote proxy log + the request-metrics capture (deploy-speed
    # basis is GB10 /metrics, but the proxy capture has the per-request rows).
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:/tmp/lumo_offload_proxy_${PROXY_PORT}.log" "$ARMDIR/offload_proxy.log" 2>/dev/null || true
    rsync -az -e "ssh ${SSH_OPTS[*]}" \
      "$HOST:/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl" \
      "$ARMDIR/offload_request_metrics.jsonl" 2>/dev/null || true
    if ssh "${SSH_OPTS[@]}" "$HOST" "test -f $REMOTE_FIXED32_LEDGER"; then
      mkdir -p "$ARMDIR/logs"
      rsync -az -e "ssh ${SSH_OPTS[*]}" \
        "$HOST:$REMOTE_FIXED32_LEDGER" \
        "$ARMDIR/logs/fr13_fixed32_proxy_ingress.jsonl" \
        || { echo "FAIL: fixed32 proxy ingress ledger fetch failed"; exit 6; }
    fi
    if [[ -n "$FIXED32_TASK_IDS" ]]; then
      echo "OFFLOAD_FETCH_OK host=$HOST raw_dumps=disabled"
    else
      NPAIR=$(ls "$ARMDIR/proxy_pair_dumps" 2>/dev/null | wc -l)
      echo "OFFLOAD_FETCH_OK host=$HOST pair_dumps_back=$NPAIR"
    fi
    ;;

  stop)
    HOST=${1:?host}
    ssh "${SSH_OPTS[@]}" "$HOST" \
      "p=\$(cat /tmp/lumo_offload_proxy_${PROXY_PORT}.pid 2>/dev/null); [ -n \"\$p\" ] && kill \$p 2>/dev/null; \
       pkill -f lumo_flywheel_serving.inference_proxy 2>/dev/null; echo stopped" 2>/dev/null || true
    ssh "${SSH_OPTS[@]}" "$HOST" \
      "rm -f $REMOTE_FIXED32_SECRET" >/dev/null 2>&1 || true
    echo "OFFLOAD_PROXY_STOPPED host=$HOST"
    ;;

  *)
    echo "unknown subcommand: $CMD"; exit 2;;
esac

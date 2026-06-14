#!/usr/bin/env bash
# FR13 TREE-SHAPE GATE DRIVER (2026-06-14)
# ---------------------------------------------------------------------------
# Reusable boot+gate+teardown for the FR13 tree-shape reshape sweep. ONE shape
# per invocation; GPU is SERIALIZED so this is meant to run ONE AT A TIME.
#
#   scripts/fr13_shape_gate.sh <name> "<TREE>"
#
# A "variant" is purely a different TREE env (num_speculative_tokens +
# speculative_token_tree are AUTO-DERIVED from len(TREE) by the forked server,
# fr13_launch_forked_fa2_tree_server.sh:110-117). NO code change. We boot the
# forked server DIRECTLY (NOT fr13_launch_locked.sh, which hardcodes cat9's TREE
# unconditionally at line 15) with TREE set + the IDENTICAL locked pipeline
# flags, so the only difference vs the gold gate is the tree shape.
#
# IMPORTANT (drafter realizability): the MTP drafter
# (fr10_phase4_patch_vllm_tree_gdn.py, _fr10_is_caterpillar/_fr10_is_spine_only
# branch) PRODUCES tokens for ONLY two fixed shapes -- the exact 9-node
# caterpillar (cat9) and the exact 5-node spine (chain5). Any other tree_mtp
# tree raises "FR10 caterpillar drafter disengaged" at the first decode event
# unless the BANNED FR10_ALLOW_LINEAR_FALLBACK=1 is set (vacuous linear path).
# Root child-rank 1 ((1,) sibling) and any child-rank >= 2 are NEVER drafted.
# This driver does NOT set FR10_ALLOW_LINEAR_FALLBACK; an unrealizable TREE will
# FAIL LOUD at the engagement gate (or earlier, at the first request). Do not
# "fix" that by enabling the fallback -- bring an unrealizable shape to the user.
#
# Gate steps (the census-specced decisive probe; reuse existing scripts):
#   0. hygiene: recover_host_memory + assert MemAvailable ~>=100GiB + assert
#      `docker ps` empty.
#   1. boot the forked server (background) with TREE + locked flags.
#   2. poll /health until ready (timeout ~900s); FAIL LOUD on container exit.
#   3. ENGAGEMENT GATE (playbook class-9): tok/draft from /metrics
#      (spec_decode_num_draft_tokens_total / spec_decode_num_drafts_total) MUST
#      == len(TREE). A small warmup request is issued first so the spec counters
#      are non-zero. If tok/draft != len(TREE): FAIL LOUD, record nothing.
#   4. CAPTURE the served stream (same boot) via fr13_gold_margin_probe.py;
#      assert within_boot_det_rep1_eq_rep2 == [T,T,T,T] (class-8 same-boot det).
#   5. FLIP-COUNT (same boot) via fr13_oracle_stream_teacher_force.py; read
#      total_clear_margin_flips + per_prompt n_clear_flips; assert
#      spec_metrics_delta_during_oracle == 0 (teacher-force must not advance spec
#      counters) + within_boot_det_all_prompts == True.
#   6. accept/event = spec_decode_num_accepted_tokens_total /
#      spec_decode_num_drafts_total from /metrics AFTER the capture run (raw
#      counters recorded too).
#   7. teardown: docker rm -f + recover_host_memory; verify nvidia-smi clean.
#   8. emit ONE JSON line to stdout (and append to the run summary jsonl).
#
# Reward-hacks BANNED (copy/dense/splice/multi-spine). Diagnostic measurement
# only. The default locked build is NOT modified -- variants are TREE-override
# boots.
set -euo pipefail

# --------------------------------------------------------------------------
# args
# --------------------------------------------------------------------------
NAME="${1:-}"
TREE_ARG="${2:-}"
if [[ -z "$NAME" || -z "$TREE_ARG" ]]; then
  echo "usage: $0 <name> \"<TREE python-list-of-tuples>\"" >&2
  exit 2
fi

REPO="${REPO:-/home/mark/shared/lumoFlyWheel}"
CONTAINER="${CONTAINER:-fr13-forked-fa2-tree}"
PORT="${PORT:-9950}"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:${PORT}}"
MODEL="${MODEL:-qwen3.6-27b}"
OUT_DIR="${OUT_DIR:-$REPO/output/fr13_shape_sweep}"
PROMPTS="${PROMPTS:-$REPO/output/fr13_acceptance_ladder/prompts_swe4.json}"
SUMMARY_JSONL="${SUMMARY_JSONL:-$OUT_DIR/shape_sweep_summary.jsonl}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-900}"
MAX_TOKENS="${MAX_TOKENS:-128}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-1313}"
MODE="${MODE:-tree_mtp}"
THRESHOLD="${THRESHOLD:-1.0}"
LAUNCHER="$REPO/scripts/fr13_launch_forked_fa2_tree_server.sh"

CAP_OUT="$OUT_DIR/${NAME}_capture.json"
FLIPS_OUT="$OUT_DIR/${NAME}_flips.json"
BOOT_LOG="$OUT_DIR/${NAME}_boot.log"

mkdir -p "$OUT_DIR"

fail() { echo "FR13_SHAPE_GATE FAIL [$NAME]: $*" >&2; }

# NUM_NODES from the TREE (the engagement target = tok/draft).
NUM_NODES="$(TREE="$TREE_ARG" python3 - <<'PY'
import ast, os
print(len(ast.literal_eval(os.environ["TREE"])))
PY
)"
MAX_DEPTH="$(TREE="$TREE_ARG" python3 - <<'PY'
import ast, os
t = ast.literal_eval(os.environ["TREE"])
print(max((len(p) for p in t), default=0))
PY
)"

# --------------------------------------------------------------------------
# host-memory recovery + hygiene asserts (parameterless recover_host_memory)
# --------------------------------------------------------------------------
recover_host() {
  PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 - <<'PY'
from lumo_flywheel_serving.model_server import recover_host_memory
recover_host_memory()
PY
}

assert_host_free() {
  python3 - <<'PY'
from pathlib import Path
fields = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    k, v = line.split(":", 1)
    fields[k] = int(v.strip().split()[0])
avail = fields.get("MemAvailable", 0) / 1024 / 1024
swap_used = fields.get("SwapTotal", 0) - fields.get("SwapFree", 0)
# locked launcher asserts >=80GiB; this driver wants the directive's ~100GiB
if avail < 95 or swap_used != 0:
    raise SystemExit(
        f"FR13_SHAPE_GATE host-mem hygiene FAILED: MemAvailable={avail:.1f}GiB "
        f"(want >=95) swap_used={swap_used/1024/1024:.2f}GiB (want 0)"
    )
print(f"[hygiene] MemAvailable={avail:.1f}GiB swap_used=0 OK")
PY
}

assert_docker_empty() {
  local running
  running="$(docker ps -q | wc -l | tr -d ' ')"
  if [[ "$running" != "0" ]]; then
    fail "docker ps not empty before boot ($running container(s) running)"
    docker ps >&2 || true
    exit 3
  fi
  echo "[hygiene] docker ps empty OK"
}

# scrape a single /metrics counter (returns "" if absent / endpoint down)
metric() {
  local name="$1"
  curl -fsS --max-time 10 "${ENDPOINT}/metrics" 2>/dev/null | awk -v n="$name" '
    $0 ~ ("^" n " ") || $0 ~ ("^" n "{") { v=$NF } END { if (v != "") print v }'
}

teardown() {
  echo "[teardown] docker rm -f $CONTAINER"
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
  echo "[teardown] recover_host_memory"
  recover_host || true
  echo "[teardown] nvidia-smi (verify clean before next boot):"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader || true
}

# --------------------------------------------------------------------------
# 0. pre-boot hygiene
# --------------------------------------------------------------------------
echo "[fr13_shape_gate] name=$NAME num_nodes=$NUM_NODES max_depth=$MAX_DEPTH"
echo "[fr13_shape_gate] TREE=$TREE_ARG"
recover_host
assert_host_free
assert_docker_empty
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# --------------------------------------------------------------------------
# 1. boot the forked server DIRECTLY with TREE override + the locked flags.
#    (NOT fr13_launch_locked.sh -- it hardcodes cat9's TREE.) The launcher
#    derives NUM_SPECULATIVE_TOKENS + speculative_token_tree from TREE.
# --------------------------------------------------------------------------
echo "[boot] launching forked server (background), log -> $BOOT_LOG"
TREE="$TREE_ARG" \
FR10_DECODE_MODE_DEFAULT=tree_mtp \
FR13_DRAFTER_SINGLE_LOGITS=1 \
FR13_EAGER_PACK=1 \
FR13_TREE_CONV_FUSED=1 \
FR13_TREE_SAMPLE_ROW=1 \
FR13_REPLAY_ROUTE=1 \
FR13_FA2_TREE_BIAS=1 \
FR13_FA2_PREFILL_NATIVE=1 \
FR13_TREE_ATTN_EXP2_SOFTMAX=1 \
FR13_CONV_COMMITTED_PATH=1 \
BATCH_INVARIANT=0 \
FR10_METRICS=0 \
CONTAINER="$CONTAINER" PORT="$PORT" \
  bash "$LAUNCHER" > "$BOOT_LOG" 2>&1 &
LAUNCH_PID=$!

# Ensure we always tear down, even on a mid-gate failure.
trap 'teardown' EXIT

# --------------------------------------------------------------------------
# 2. poll /health until ready (or container died), timeout HEALTH_TIMEOUT_S
# --------------------------------------------------------------------------
echo "[boot] waiting for /health (timeout ${HEALTH_TIMEOUT_S}s)"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_S ))
healthy=0
while [[ $(date +%s) -lt $deadline ]]; do
  # FAIL LOUD if the launcher process exited non-zero before health
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    wait "$LAUNCH_PID" || true
    # launcher backgrounds docker run -d, so it returns 0 quickly; only treat as
    # fatal if the CONTAINER itself is gone/exited.
    :
  fi
  # if the container exited, bail with its tail
  cstate="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo absent)"
  if [[ "$cstate" == "exited" || "$cstate" == "dead" ]]; then
    fail "container $CONTAINER $cstate before /health ready; boot log tail:"
    docker logs --tail 60 "$CONTAINER" >&2 2>/dev/null || tail -n 60 "$BOOT_LOG" >&2 || true
    exit 4
  fi
  if curl -fsS --max-time 5 "${ENDPOINT}/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 5
done
if [[ "$healthy" != "1" ]]; then
  fail "server not healthy in ${HEALTH_TIMEOUT_S}s; boot log tail:"
  tail -n 80 "$BOOT_LOG" >&2 || true
  exit 4
fi
echo "[boot] /health ready"

# --------------------------------------------------------------------------
# 3. ENGAGEMENT GATE (class-9): warm the spec counters with one tiny request,
#    then assert tok/draft == NUM_NODES (== len(TREE)).
# --------------------------------------------------------------------------
echo "[engage] warmup request to populate spec counters"
WARM_PROMPT="$(python3 - "$PROMPTS" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
print(p[0][:400])
PY
)"
warm_payload="$(python3 - <<PY
import json
print(json.dumps({
    "model": "$MODEL",
    "messages": [{"role": "user", "content": $(python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))' <<<"$WARM_PROMPT")}],
    "max_tokens": 24,
    "temperature": 0.0,
    "seed": $SEED,
}))
PY
)"
curl -fsS --max-time 120 -H 'Content-Type: application/json' \
  -d "$warm_payload" "${ENDPOINT}/v1/chat/completions" >/dev/null 2>&1 || {
    fail "warmup request failed (drafter likely disengaged / crashed). boot log tail:"
    docker logs --tail 60 "$CONTAINER" >&2 2>/dev/null || true
    exit 5
}

DRAFT_TOKS="$(metric vllm:spec_decode_num_draft_tokens_total)"
NUM_DRAFTS="$(metric vllm:spec_decode_num_drafts_total)"
if [[ -z "$DRAFT_TOKS" || -z "$NUM_DRAFTS" || "$NUM_DRAFTS" == "0" ]]; then
  fail "spec counters absent/zero after warmup (draft_toks='$DRAFT_TOKS' drafts='$NUM_DRAFTS') -> drafter NOT engaged"
  exit 5
fi
TOK_PER_DRAFT="$(python3 - "$DRAFT_TOKS" "$NUM_DRAFTS" <<'PY'
import sys
print(repr(float(sys.argv[1]) / float(sys.argv[2])))
PY
)"
ENGAGED="$(python3 - "$TOK_PER_DRAFT" "$NUM_NODES" <<'PY'
import sys
# tok/draft must equal len(TREE) exactly (integer draft tokens per event)
print("true" if abs(float(sys.argv[1]) - float(sys.argv[2])) < 1e-6 else "false")
PY
)"
echo "[engage] tok/draft=$TOK_PER_DRAFT len(TREE)=$NUM_NODES engaged=$ENGAGED"
if [[ "$ENGAGED" != "true" ]]; then
  fail "ENGAGEMENT GATE: tok/draft ($TOK_PER_DRAFT) != len(TREE) ($NUM_NODES) -- tree did NOT engage; recording NOTHING"
  exit 5
fi

# --------------------------------------------------------------------------
# 4. CAPTURE served stream (same boot); assert within_boot_det [T,T,T,T]
# --------------------------------------------------------------------------
echo "[capture] fr13_gold_margin_probe.py capture --arm tree -> $CAP_OUT"
python3 "$REPO/scripts/fr13_gold_margin_probe.py" capture \
  --arm tree \
  --out "$CAP_OUT" \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --prompts-file "$PROMPTS" \
  --max-tokens "$MAX_TOKENS" --top-k "$TOP_K" --mode "$MODE" --seed "$SEED"

WITHIN_BOOT_DET="$(python3 - "$CAP_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps(d.get("within_boot_det_rep1_eq_rep2")))
PY
)"
DET_OK="$(python3 - "$CAP_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
w = d.get("within_boot_det_rep1_eq_rep2")
print("true" if isinstance(w, list) and len(w) > 0 and all(bool(x) for x in w) else "false")
PY
)"
echo "[capture] within_boot_det=$WITHIN_BOOT_DET det_ok=$DET_OK"
if [[ "$DET_OK" != "true" ]]; then
  fail "DETERMINISM GATE (class-8 same-boot): within_boot_det not all True ($WITHIN_BOOT_DET)"
  exit 6
fi

# --------------------------------------------------------------------------
# 5. FLIP-COUNT (same boot) over the full stream
# --------------------------------------------------------------------------
echo "[flips] fr13_oracle_stream_teacher_force.py run --arm tree -> $FLIPS_OUT"
python3 "$REPO/scripts/fr13_oracle_stream_teacher_force.py" run \
  --arm tree \
  --tree "$CAP_OUT" \
  --out "$FLIPS_OUT" \
  --endpoint "$ENDPOINT" --model "$MODEL" \
  --top-k "$TOP_K" --mode "$MODE" --seed "$SEED" --threshold "$THRESHOLD"

read -r TOTAL_FLIPS SPEC_DELTA_OK WITHIN_ALL <<EOF
$(python3 - "$FLIPS_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
total = d.get("total_clear_margin_flips")
delta = d.get("spec_metrics_delta_during_oracle", {}) or {}
# teacher-force must NOT advance the spec counters
delta_ok = all(abs(float(v)) < 1e-9 for v in delta.values()) if delta else True
within = bool(d.get("within_boot_det_all_prompts"))
print(total, "true" if delta_ok else "false", "true" if within else "false")
PY
)
EOF
echo "[flips] total_clear_margin_flips=$TOTAL_FLIPS spec_delta_zero=$SPEC_DELTA_OK within_boot_det_all=$WITHIN_ALL"
if [[ "$SPEC_DELTA_OK" != "true" ]]; then
  fail "teacher-force advanced spec counters (spec_metrics_delta_during_oracle != 0) -> oracle not clean"
  exit 7
fi
if [[ "$WITHIN_ALL" != "true" ]]; then
  fail "within_boot_det_all_prompts not True during oracle"
  exit 7
fi

PER_PROMPT_FLIPS="$(python3 - "$FLIPS_OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps([p.get("n_clear_flips") for p in d.get("per_prompt", [])]))
PY
)"

# --------------------------------------------------------------------------
# 6. accept/event from /metrics AFTER the capture run (record raw counters)
# --------------------------------------------------------------------------
ACCEPTED="$(metric vllm:spec_decode_num_accepted_tokens_total)"
NUM_DRAFTS2="$(metric vllm:spec_decode_num_drafts_total)"
ACCEPT_PER_EVENT="$(python3 - "$ACCEPTED" "$NUM_DRAFTS2" <<'PY'
import sys
a, d = sys.argv[1], sys.argv[2]
try:
    print(repr(float(a) / float(d)))
except Exception:
    print("null")
PY
)"
echo "[accept] accepted=$ACCEPTED drafts=$NUM_DRAFTS2 accept/event=$ACCEPT_PER_EVENT"

# --------------------------------------------------------------------------
# 8. emit ONE JSON summary line (teardown happens via the EXIT trap)
# --------------------------------------------------------------------------
JSON_LINE="$(python3 - <<PY
import json
def num(x):
    try:
        return float(x)
    except Exception:
        return None
rec = {
    "name": "$NAME",
    "tree": $(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "$TREE_ARG"),
    "num_nodes": $NUM_NODES,
    "max_depth": $MAX_DEPTH,
    "total_clear_margin_flips": int(num("$TOTAL_FLIPS")) if num("$TOTAL_FLIPS") is not None else None,
    "per_prompt": json.loads('$PER_PROMPT_FLIPS'),
    "accept_per_event": num("$ACCEPT_PER_EVENT"),
    "within_boot_det": json.loads('$WITHIN_BOOT_DET'),
    "tok_per_draft": num("$TOK_PER_DRAFT"),
    "engaged": ("$ENGAGED" == "true"),
    "ok": True,
    "notes": "boot=$BOOT_LOG capture=$CAP_OUT flips=$FLIPS_OUT accepted=$ACCEPTED drafts=$NUM_DRAFTS2 draft_toks=$DRAFT_TOKS spec_delta_zero=$SPEC_DELTA_OK within_all=$WITHIN_ALL",
}
print(json.dumps(rec))
PY
)"
echo "$JSON_LINE" | tee -a "$SUMMARY_JSONL"

#!/usr/bin/env bash
# FR14 SGLANG-ON-16 (Mark's queued measure: "sglang best speed on the 16 task").
# Their engine (lmsysorg/sglang:qwen38-27b) + RadixArk as-shipped + their EAGLE
# 3/1/4 Spark recipe, driven by the SAME qwen-code harness over the SAME exact16
# canonical subset our QC served, at the SAME 24k proxy output ceiling
# (relaunch_proxy_remote.sh:64 default) and the SAME 9000s budget. This is the
# first CEILING-MATCHED ours-vs-theirs comparison; step2's 4-task numbers ran at
# the old 32k cap.
#
# PREEMPTION POLICY (standing priority: baseline serves outrank calibration):
# round 22 (CH31i4) refires the moment the GDN-contract fix lands. This run is
# then stopped AT A TASK BOUNDARY (per_task banks incrementally), the sglang
# container is removed, and the remainder resumes later as its own declared
# subset — the exact16 resume-set discipline (pins are never fiction).
#
# Vehicle provenance: boot script is the banked step2 vehicle verbatim (OUT
# overridden). Proxy start's class-9 gate has a DOCUMENTED cosmetic exit-5
# (step2 compromise 5: a literal-~ grep); on nonzero rc we verify the proxy
# INDEPENDENTLY (health + live /proc env pins incl. the 24k ceiling) and
# proceed only on evidence, mirroring step2's disclosed deviation.
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"
# Resume runs override OUT/SUBSET/EXPECT_SHA (declared resume-set discipline:
# every serve declares the exact set it serves; a resume set is its own set).
OUT=${OUT:-/home/mark/shared/tmp-scratch/fr14_sglang16}
mkdir -p "$OUT/swe_out"
GB10_IP=100.103.10.122
SUBSET=${SUBSET:-config/fr13_fixed32/subset_b4_sixteen.json}
EXPECT_SHA=${EXPECT_SHA:-47b0a3c9be49e2cb5f7e7217ae03c267a05359f269f3e3b038942f57d7dc0b5c}
[[ "$(sha256sum "$SUBSET" | cut -d' ' -f1)" == "$EXPECT_SHA" ]] \
  || { echo "FAIL: subset digest drifted vs declared: $SUBSET"; exit 2; }
echo "[sglang16] subset=$SUBSET sha=$(sha256sum "$SUBSET" | cut -c1-8) tasks=$(python3 -c "import json;print(len(json.load(open('$SUBSET'))['instance_ids']))")"

echo "=== [sglang16] boot engine $(date -u +%FT%TZ) ==="
OUT="$OUT" bash results/fr14_nvfp4_port_20260816/ablation_a_step2_sglang_boot.sh
rc=$?
(( rc == 0 )) || { echo "FAIL: sglang boot rc=$rc"; exit 3; }

echo "=== [sglang16] proxy sync+start $(date -u +%FT%TZ) ==="
bash scripts/swe_x86_helpers/offload_codex_proxy.sh sync alienware \
  > "$OUT/proxy_sync.log" 2>&1 || { echo "FAIL: proxy sync"; exit 4; }
bash scripts/swe_x86_helpers/offload_codex_proxy.sh start alienware "$GB10_IP" "$OUT" \
  > "$OUT/proxy_start.log" 2>&1
prc=$?
if (( prc != 0 )); then
  echo "[sglang16] proxy start rc=$prc — verifying independently (documented class-9 cosmetic path)"
  ok=$(ssh -o ConnectTimeout=8 alienware \
    "curl -s -o /dev/null -m 5 -w '%{http_code}' http://127.0.0.1:8023/v1/models" 2>/dev/null)
  grep -q 'LUMO_PROXY_MAX_OUTPUT_TOKENS=24000' "$OUT/offload_proxy_env.txt" 2>/dev/null \
    && grep -q 'LUMO_PROXY_FORCE_TEMPERATURE=0.6' "$OUT/offload_proxy_env.txt" 2>/dev/null \
    && [[ -n "$ok" && "$ok" != "000" ]] \
    || { echo "FAIL: proxy start rc=$prc and independent verification FAILED (http=$ok)"; \
         tail -20 "$OUT/proxy_start.log"; exit 4; }
  echo "[sglang16] proxy verified live+pinned despite rc=$prc (http=$ok); proceeding as step2 did"
fi
grep -E 'MAX_OUTPUT_TOKENS|FORCE_TEMPERATURE|UPSTREAM' "$OUT/offload_proxy_env.txt" || true

echo "=== [sglang16] swe run $(date -u +%FT%TZ) ==="
export SWE_AGENT=qwen_code
export SWE_AGENT_ENV=instance_image
export SWE_EMPTY_PATCH_RETRIES=0
export LUMO_SWE_AUTOCOMMIT=0
export LUMO_SWE_STALL_KILL_S=900
export HF_HUB_OFFLINE=0
date -u +%FT%TZ > "$OUT/swe_started_at.txt"
curl -fsS http://127.0.0.1:9950/metrics > "$OUT/sglang_metrics_pre.txt" 2>/dev/null
.venv/bin/python scripts/run_swe_bench_q36_a.py \
  --subset "$SUBSET" \
  --out-root "$OUT/swe_out" \
  --concurrency 1 \
  --agent-wall-s 9000 \
  --eval-timeout-s 1800 \
  --model qwen3.8-27b-nvfp4-radixark \
  --model-name "qwen3.8-27b-nvfp4-radixark::sglang-eagle3-1-4::qwen-code-0.19.4::fr14-sglang16" \
  --agent-host alienware --agent-endpoint http://127.0.0.1:8023/v1 \
  --eval-host alienware \
  > "$OUT/swe_orchestrator.log" 2>&1
rc=$?
echo "swe rc=$rc" | tee "$OUT/swe_rc.txt"
date -u +%FT%TZ > "$OUT/swe_ended_at.txt"
curl -fsS http://127.0.0.1:9950/metrics > "$OUT/sglang_metrics_post.txt" 2>/dev/null
exit "$rc"

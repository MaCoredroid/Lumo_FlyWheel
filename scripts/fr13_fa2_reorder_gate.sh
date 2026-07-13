#!/usr/bin/env bash
# FR13_FA2_SPINE_REORDER staged gate.
#   MODE=0 : baseline (single paged call, current ship path)
#   MODE=2 : split-only (cascade context+suffix+merge, pi=IDENTITY, NO reorder)
#   MODE=1 : reorder (the fix: suffix permuted spine-first)
# GATE-1a (this script): boot MODE=0 and MODE=2, greedy+temp06, compare output_text.
#   split-only byte-identical to baseline => the cascade split+merge is lossless
#   (the ONLY new risk; the permute itself is already MAB-proven bit-exact).
#   ALSO assert the engagement marker fired for MODE=2 (not a silent single-call fallback).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR="${RUNROOT:-output/fr13_fa2_reorder_gate/run_$STAMP}"; mkdir -p "$RR"
MODES="${MODES:-0 2}"
CHATMSG="${CHATMSG:-output/fr13_matched_proof_swe_prompt.json}"
echo "=== FR13_FA2_SPINE_REORDER gate | runroot=$RR | modes=$MODES ==="

run() {  # $1=mode
  local M="$1"
  [[ -z "$(docker ps -q)" ]] || { echo "FAIL: docker not empty"; docker ps; exit 2; }
  echo "----- boot MODE=$M -----"
  FR13_FA2_SPINE_REORDER="$M" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-1}" \
  FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
  FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
  ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N="${PROBE_N:-256}" MAX_NUM_SEQS_OVR=1 \
  PROBE_MODES="${PROBE_MODES:-greedy temp06}" RUNROOT="$RR" \
  PROBE_CHAT_MESSAGES="$CHATMSG" \
    bash scripts/fr13_bigdenom_swe_serve_variant.sh "m$M" cat8 subset_carrier_four.json \
    > "$RR/m$M.log" 2>&1
  echo "[MODE=$M] rc=$? containers=$(docker ps -q|wc -l)"
  echo "  engagement: $(grep -c 'FR13_FA2_SPINE_REORDER hybrid ENGAGED' "$RR/m$M/docker_full.log" 2>/dev/null || echo 0) marker line(s)"
}

for M in $MODES; do run "$M"; done

echo "=== GATE-1a VERDICT (split-only vs baseline byte-identity) ==="
.venv/bin/python - "$RR" <<'PY'
import json, sys, os, glob
rr=sys.argv[1]
def load(mode, probe):
    fs=glob.glob(f"{rr}/m{mode}/accept_speed_{probe}.json")
    if not fs: return None
    return json.load(open(fs[0]))
for probe in ("greedy","temp06"):
    b=load("0",probe); s=load("2",probe)
    if not b or not s:
        print(f"  {probe}: MISSING (b={bool(b)} s={bool(s)})"); continue
    tb=b.get("output_text",""); ts=s.get("output_text","")
    ident = tb==ts
    # first divergence char
    d=next((i for i in range(min(len(tb),len(ts))) if tb[i]!=ts[i]), None)
    print(f"  {probe}: baseline_tok={b.get('committed_tokens')} split_tok={s.get('committed_tokens')} "
          f"accept b={b.get('accept_per_forward')} s={s.get('accept_per_forward')} | "
          f"output byte-identical={ident}" + ("" if ident else f" (first diff @char {d})"))
# engagement (mode 2 must have fired)
import subprocess
eng=0
p=f"{rr}/m2/docker_full.log"
if os.path.exists(p):
    eng=sum(1 for l in open(p,errors='ignore') if "FR13_FA2_SPINE_REORDER hybrid ENGAGED" in l)
print(f"  MODE=2 engagement marker lines: {eng} (MUST be >=1, else silent single-call fallback => wiring/anchor bug)")
print(">>> GATE-1a PASS if: MODE=2 engaged >=1 AND greedy output byte-identical to baseline "
      "(cascade split+merge lossless). If differ, RED-TEAM the split (causal=False context, seqused_k-tree_n, merge).")
PY

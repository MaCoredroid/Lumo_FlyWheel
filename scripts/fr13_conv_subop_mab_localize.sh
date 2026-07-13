#!/usr/bin/env bash
# FR13_CONV_SUBOP_MAB LOCALIZER — settle whether the SHIP fused causal-conv
# (fused_tree_conv_taps_acc + triton silu) is the M-dependent carrier of the
# cat6-vs-cat8 SPINE accept-rate gap.  Boots cat8 (most branch rows => strongest
# co-residency), B=1 (kills batching confound), temp 0.6 (never greedy), real SWE
# chat prompt.  The instrument (default-OFF, observe-only) re-runs the SAME fused
# taps on the SPINE-only sub-window and compares raw int-view (threshold 0.0):
#   taps_mm>0            => fused conv taps is the carrier
#   taps=0, out_mm>0     => triton silu is the carrier
#   taps=0, out=0 (all)  => conv M-invariant => hand off to FA2 (then scan)
# Fires on EVERY spine forward (guarded _fr10_b==0) — no deep-accept warmup needed
# (that dependency is what engage-failed the garble-era native-conv MAB 3x).
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR="${RUNROOT:-output/fr13_conv_subop_mab/run_$STAMP}"
mkdir -p "$RR"
echo "=== FR13_CONV_SUBOP_MAB localizer | runroot=$RR ==="
[[ -z "$(docker ps -q)" ]] || { echo "FAIL: docker not empty before boot"; docker ps; exit 2; }

# cat8 boot mirrors the proven-bootable matched-proof config (garble-free ship
# arm) + the localizer flags.  The conv A/B is independent of ATTN_KV_REMAP /
# DEVICE_MULTIDRAFT (downstream), but we mirror the ship config for faithfulness.
FR13_CONV_SUBOP_MAB=1 \
FR13_CONV_SUBOP_MAB_LIMIT="${LIMIT:-16}" \
FR13_CONV_SUBOP_MAB_DUMP=/logs/fr13_conv_subop_mab.jsonl \
FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N="${PROBE_N:-256}" MAX_NUM_SEQS_OVR=1 \
PROBE_MODES="${PROBE_MODES:-temp06}" RUNROOT="$RR" \
PROBE_CHAT_MESSAGES="${CHATMSG:-output/fr13_matched_proof_swe_prompt.json}" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh mab cat8 subset_carrier_four.json \
  > "$RR/run.log" 2>&1
RC=$?
echo "[boot+probe] rc=$RC  containers now: $(docker ps -q | wc -l)"

DUMP="$RR/mab/logs/fr13_conv_subop_mab.jsonl"
echo "=== ENGAGEMENT CHECK (never conclude from a vacuous run) ==="
grep -c "FR13_CONV_SUBOP_MAB sidecar written" "$RR/mab/launch.log" 2>/dev/null \
  | sed 's/^/  sidecar-written lines: /' || true
if [[ ! -s "$DUMP" ]]; then
  echo "FAIL: instrument did NOT fire (dump empty/missing: $DUMP) — NOT a verdict."
  echo "  --- last 40 lines run.log ---"; tail -40 "$RR/run.log"
  echo "  --- grep worker log for the warning/flag ---"
  grep -n "FR13_CONV_SUBOP_MAB" "$RR/mab/docker_full.log" 2>/dev/null | tail -20
  exit 3
fi

echo "=== VERDICT (first_conv_subop histogram) ==="
.venv/bin/python - "$DUMP" <<'PY'
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hist = collections.Counter(r["first_conv_subop"] for r in recs)
layers = sorted({r["layer_prefix"] for r in recs})
print("events:", len(recs), "| distinct layers:", len(layers))
tt = sum(r["taps_mismatch"] for r in recs)
to = sum(r["out_mismatch"] for r in recs)
print("SUM taps_mismatch:", tt, "| SUM out_mismatch:", to)
for r in recs[:14]:
    print(f'  {r["layer_prefix"]} tree_n={r["tree_n"]} m_red={r["m_reduced"]} '
          f'deep={r["deep_row"]} taps={r["taps_mismatch"]} out={r["out_mismatch"]} '
          f'deep_mm={r["deep_row_mismatch"]} -> {r["first_conv_subop"]}')
print("VERDICT first_conv_subop:", dict(hist))
if tt > 0:
    print(">>> CARRIER = fused conv taps (fused_tree_conv_taps_acc batch-occupancy).")
elif to > 0:
    print(">>> CARRIER = triton silu activation (M-grid).")
else:
    print(">>> conv M-INVARIANT on all sampled layers => carrier is NOT conv "
          "=> proceed to FA2 query-tile (then scan N_ACTUAL).")
PY
echo "=== speed/accept snapshot (bonus) ==="
grep -h "decode_tps_wall\|accept_per_forward" "$RR"/mab/probe_*temp06*.json 2>/dev/null | head
ls "$RR"/mab/*.json 2>/dev/null | head

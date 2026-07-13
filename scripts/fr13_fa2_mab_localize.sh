#!/usr/bin/env bash
# FR13_FA2_MAB LOCALIZER (ladder step 2) — settle whether the forked FA2 query-tile
# is the M-dependent carrier of the cat6-vs-cat8 SPINE accept-rate gap, now that
# conv is CLEARED (768/768 M-invariant). Boots cat8 (served tree_n=9, spine=6,
# deep=8), B=1, temp 0.6, ENFORCE_EAGER=1 (the FA2 MAB syncs -> eager-only). The
# generalized FR13_FA2_MAB re-calls the SAME forked flash_attn_varlen_func on the
# live tree TWICE: M_full=9 (all rows + 9x9 bias) vs M_spine=6 (spine rows
# [0,1,3,5,7,8] + 6x6 sub-bias + spine-suffix KV), compares the deep-spine row
# (row 8) output RAW max_abs. Spine derived from attn_metadata.fr10_tree_path0_nodes
# (NOT the hardcoded cat9 constants).
#   deep_spine_raw_max_abs > 0 (any full-attn layer) => FA2 query-tile IS the carrier
#   == 0 on all layers                                => FA2 M-invariant => scan N_ACTUAL next
set -uo pipefail
cd /home/mark/shared/lumoFlyWheel

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RR="${RUNROOT:-output/fr13_fa2_mab/run_$STAMP}"
mkdir -p "$RR"
echo "=== FR13_FA2_MAB localizer (cat8 M9-vs-M6) | runroot=$RR ==="
[[ -z "$(docker ps -q)" ]] || { echo "FAIL: docker not empty before boot"; docker ps; exit 2; }

FR13_FA2_MAB=1 \
FR13_FA2_MAB_DUMP=/logs/fr13_fa2_mab.jsonl \
FR13_FA2_MAB_LAYER="*" \
FR13_FA2_MAB_SKIP=0 \
FR13_FA2_MAB_LIMIT="${LIMIT:-16}" \
FR13_FA2_MAB_QPAD="${QPAD:-0}" \
FR13_FA2_MAB_KVPAD="${KVPAD:-0}" \
FR13_FA2_MAB_REORDER="${REORDER:-1}" \
ENFORCE_EAGER=1 \
FR13_ATTN_KV_REMAP=1 FR13_DEVICE_MULTIDRAFT=1 \
FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py \
ACCEPT_SPEED_PROBE=1 OFFLOAD_AGENT=0 PROBE_N="${PROBE_N:-256}" MAX_NUM_SEQS_OVR=1 \
PROBE_MODES="${PROBE_MODES:-temp06}" RUNROOT="$RR" \
PROBE_CHAT_MESSAGES="${CHATMSG:-output/fr13_matched_proof_swe_prompt.json}" \
  bash scripts/fr13_bigdenom_swe_serve_variant.sh mab cat8 subset_carrier_four.json \
  > "$RR/run.log" 2>&1
RC=$?
echo "[boot+probe] rc=$RC  containers now: $(docker ps -q | wc -l)"

DUMP="$RR/mab/logs/fr13_fa2_mab.jsonl"
if [[ ! -s "$DUMP" ]]; then
  echo "FAIL: FA2 MAB did NOT fire (dump empty/missing: $DUMP) — NOT a verdict."
  echo "  --- last 40 lines run.log ---"; tail -40 "$RR/run.log"
  echo "  --- worker log FR13_FA2_MAB lines ---"
  grep -n "FR13_FA2_MAB" "$RR/mab/docker_full.log" 2>/dev/null | tail -20
  exit 3
fi

echo "=== VERDICT (deep-spine RAW max_abs; int-view >0 => M-dependent) ==="
.venv/bin/python - "$DUMP" <<'PY'
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
layers = sorted({r["layer_name"] for r in recs})
print("events:", len(recs), "| distinct full-attn layers:", len(layers))
# per-layer worst deep-spine raw max_abs
worst = collections.defaultdict(float)
for r in recs:
    worst[r["layer_name"]] = max(worst[r["layer_name"]], r.get("deep_spine_raw_max_abs", 0.0))
gmax = max(worst.values()) if worst else 0.0
nz_layers = sum(1 for v in worst.values() if v > 0.0)
# sanity: does the M_full arm reproduce the served deep-spine row? (recall self-check)
recall_err = max((r.get("recall_m9_vs_served_deep_max_abs", 0.0) for r in recs), default=0.0)
for r in recs[:10]:
    print(f'  {r["layer_name"]} m_full={r["m_full"]} spine={r["spine_rows"]} deep={r["deep_row"]} '
          f'bias=[{r.get("bias_min",0):.1e},{r.get("bias_max",0):.1e}] '
          f'raw_max_abs={r.get("deep_spine_raw_max_abs",0.0):.3e} '
          f'recall_vs_served={r.get("recall_m9_vs_served_deep_max_abs",0.0):.3e}')
# CONFOUND GUARD: cat8 spine MUST be [0,1,3,5,7,8] deep=8. If any event shows the
# hardcoded cat9 fallback [0,1,2,4,6]/deep=6, the spine derivation silently failed
# and the raw_max_abs is GARBAGE (wrong ancestor set) -- NOT a valid FA2 verdict.
cat9_fallback = [r for r in recs if r["spine_rows"] == [0,1,2,4,6] or r["deep_row"] == 6]
deeps = sorted({r["deep_row"] for r in recs}); spines = {tuple(r["spine_rows"]) for r in recs}
print(f"distinct deep_row: {deeps} | distinct spine: {[list(s) for s in spines]}")
if cat9_fallback:
    print(f"!!! CONFOUND: {len(cat9_fallback)}/{len(recs)} events used the cat9 fallback "
          f"[0,1,2,4,6]/deep6 -- spine derivation FAILED. raw_max_abs is INVALID. FIX before trusting.")
print(f"GLOBAL worst deep-spine raw_max_abs = {gmax:.3e} | layers-with-nonzero = {nz_layers}/{len(worst)}")
print(f"recall_vs_served self-check (should be ~0): {recall_err:.3e}")
# QPAD VALIDATION: per pad_to, worst deep-spine raw_max_abs across all events.
# A pad_to that drives this to 0 = QPAD (pin max_seqlen_q) makes FA2 M-invariant.
qpad_worst = collections.defaultdict(float)
qpad_self_worst = collections.defaultdict(float)
qpad_seen = False
for r in recs:
    q = r.get("qpad_deep_raw_max_abs") or {}
    s = r.get("qpad_self_m9_vs_unpadded") or {}
    for pt, val in q.items():
        qpad_seen = True
        if isinstance(val, (int, float)):
            qpad_worst[pt] = max(qpad_worst[pt], val)
    for pt, val in s.items():
        if isinstance(val, (int, float)):
            qpad_self_worst[pt] = max(qpad_self_worst[pt], val)
if qpad_seen:
    print("=== QPAD VALIDATION (per pad_to: M9-vs-M6 deep raw_max_abs | SELF-CHECK padM9-vs-unpadM9) ===")
    for pt in sorted(qpad_worst, key=int):
        print(f"  pad_to={pt}: M9vM6={qpad_worst[pt]:.3e}  self(padM9-vs-unpadM9)={qpad_self_worst.get(pt,float('nan')):.3e}")
    # SELF-CHECK FIRST: if padding corrupts the M9 real deep row (self >> unpadded
    # ~1-ULP floor), the QPAD TEST is BROKEN (back-pad misaligns bias/causal),
    # NOT a valid refutation. The unpadded M-dep floor is ~6e-2 (1 bf16 ULP).
    self_bad = [pt for pt in qpad_self_worst if qpad_self_worst[pt] > 1e-1]
    winners = [pt for pt in qpad_worst if qpad_worst[pt] == 0.0]
    if self_bad and not winners:
        print(f">>> QPAD TEST INVALID: padding CORRUPTS the M9 real deep row (self={max(qpad_self_worst.values()):.3e} "
              f">> 1-ULP floor) at pad_to={self_bad} -- naive back-pad misaligns bias/causal. "
              "The M9-vs-M6 numbers are ARTIFACTS, NOT a QPAD refutation. Fix the padding (front-pad / "
              "bias-align) or test QPAD differently before concluding.")
    elif winners:
        print(f">>> QPAD VALIDATED: pad max_seqlen_q to {min(winners, key=int)} => FA2 M-invariant "
              f"(self-check clean). Implement FR13_FA2_QPAD in the live path, then gate.")
    else:
        print(">>> QPAD REFUTED for cat8 (self-check CLEAN, padding preserves real rows, yet no "
              "pad_to reaches 0): pinning max_seqlen_q does NOT align the deep row. Different fix needed.")
elif gmax > 0.0:
    print(">>> CARRIER = forked FA2 query-tile (M-dependent). Next: validate FR13_FA2_QPAD (arm QPAD=1).")
else:
    print(">>> FA2 M-INVARIANT => carrier is NOT FA2 => proceed to scan N_ACTUAL A/B.")
# KV-SUFFIX-PAD VALIDATION: pad suffix KV+bias to fixed width => M-invariant?
kvw = collections.defaultdict(float); kvs = collections.defaultdict(float); kv_seen = False
for r in recs:
    q = r.get("kvpad_deep_raw_max_abs") or {}
    s = r.get("kvpad_self_m9_vs_unpadded") or {}
    for pt, val in q.items():
        kv_seen = True
        if isinstance(val, (int, float)): kvw[pt] = max(kvw[pt], val)
    for pt, val in s.items():
        if isinstance(val, (int, float)): kvs[pt] = max(kvs[pt], val)
if kv_seen:
    print("=== KV-SUFFIX-PAD VALIDATION (per kv_pad_to: M9-vs-M6 | SELF-CHECK kvpadM9-vs-unpadM9) ===")
    for pt in sorted(kvw, key=int):
        print(f"  kv_pad_to={pt}: M9vM6={kvw[pt]:.3e}  self(kvpadM9-vs-unpadM9)={kvs.get(pt,float('nan')):.3e}")
    self_bad = [pt for pt in kvs if kvs[pt] > 1e-1]
    winners = [pt for pt in kvw if kvw[pt] == 0.0]
    if self_bad and not winners:
        print(f">>> KV-PAD TEST INVALID: masked dummy keys CHANGED the M9 real row (self={max(kvs.values()):.3e}) "
              "-- the -inf mask isn't math-neutral in this kernel. Fix the test before concluding.")
    elif winners:
        print(f">>> KV-PAD VALIDATED: pad suffix to {min(winners,key=int)} => FA2 M-invariant (self clean). "
              "Implement live: pad tree_attn_bias + suffix KV to fixed width (metadata-level, near-no-HBM-tax), then gate.")
    else:
        print(">>> KV-PAD REFUTED (self clean, no kv_pad_to reaches 0): suffix-width pad does NOT align it. "
              "=> RESEARCH workflow for other compute-only fixes before cost-gate; test scan N_ACTUAL.")
# FIX-A' VALIDATION: contiguous-spine reorder => deep row bit-exact vs spine-only?
rr = [r.get("reorder_a_prime") for r in recs if r.get("reorder_a_prime")]
if rr:
    import statistics
    def _wm(k):
        vals=[x[k] for x in rr if isinstance(x.get(k),(int,float))]
        return max(vals) if vals else float('nan')
    d6=_wm("deep_vs_m6"); d9=_wm("deep_vs_m9"); nd=_wm("nondeep_relabel_max")
    errs=[x.get("err") for x in rr if x.get("err")]
    print("=== FIX-A' (contiguous-spine reorder) VALIDATION ===")
    print(f"  events={len(rr)} | worst deep_vs_M6(spine-only)={d6:.3e}  deep_vs_M9(orig)={d9:.3e}  nondeep_relabel={nd:.3e}")
    if rr and rr[0].get("pi"): print(f"  pi(example)={rr[0]['pi']}")
    if errs: print(f"  !!! {len(errs)} errors, e.g. {errs[0]}")
    # baseline (interleaved identity) = the raw deep-spine M9-vs-M6 = gmax (should stay 6.25e-2)
    print(f"  negative control (interleaved identity M9-vs-M6) = {gmax:.3e} (should stay nonzero)")
    if d6 == 0.0 and nd < 1e-1 and gmax > 0.0:
        print(">>> FIX-A' VALIDATED: contiguous-spine reorder makes cat8 deep-spine BIT-EXACT vs spine-only "
              "(deep_vs_M6=0, relabel clean, interleaved baseline still nonzero). Promote A' to a live "
              "call-site-local reorder in fr13_patch_fa2_tree_bias.py tree-decode wiring (flag-gated) + gate.")
    elif nd >= 1e-1:
        print(">>> A' TEST SUSPECT: relabel-neutrality large => reorder corrupted non-deep rows. Check pi topology.")
    else:
        print(f">>> A' did NOT fully zero (deep_vs_M6={d6:.3e}). Reassess mechanism / try RANK-3 kernel fix.")
PY

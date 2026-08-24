#!/usr/bin/env bash
# PRETASK ROUTE-SURFACE DRY-RUN — so route gaps die on the desk, not the GPU.
#
# WHY THIS EXISTS. The MTP-5 probe died three times before serving a task, and the third
# was a route-surface gap: the vehicle's warmup probe POSTs /tokenize, the native engine
# 404s it, and the arm tore down after seven minutes. Nothing on the desk had said the
# native route would call an endpoint the fixed32 route never calls.
#
# THE ASYMMETRY THIS SURFACES, which is the substantive finding:
#   fr13_bigdenom_swe_serve_variant.sh:2512
#     # ---- warmup probe (legacy arms only; fixed32 permits canonical SWE traffic only)
#     if [[ -z "$FIXED32_MODE" ]]; then ... fr10_quick_decode_tps_probe.py ...
#   The probe runs ONLY when FIXED32_MODE is empty. Every fixed32/QC arm SKIPS it.
#   Only legacy and native arms run it -- so a native arm executes a pretask route the
#   arms it is being compared against never touch.
#
# Usage: promotion_ab_pretask_dryrun.sh <KIND>
set -uo pipefail
REPO=/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816
cd "$REPO"
KIND=${1:?kind, e.g. nativemtp5 or hydra27_fixed32}
V=scripts/fr13_bigdenom_swe_serve_variant.sh

# Does this kind set FIXED32_MODE? That single fact decides the pretask surface.
if grep -qE "^\s+${KIND}\)" "$V" && grep -A2 -E "^\s+${KIND}\)" "$V" | grep -q "LAUNCHER=native"; then
  LAUNCHER=native; FIXED32=empty
elif [[ "$KIND" == *fixed32* ]]; then
  LAUNCHER=forked; FIXED32=set
else
  LAUNCHER=$(grep -A1 -E "^\s+${KIND}\)" "$V" | grep -oE "LAUNCHER=[a-z]+" | head -1 | cut -d= -f2)
  FIXED32=empty
fi

echo "kind=$KIND launcher=${LAUNCHER:-unknown} FIXED32_MODE=$FIXED32"
echo
echo "PRETASK ROUTES this kind will exercise:"
echo "  POST /reset_prefix_cache      (all arms)"
echo "  GET  /metrics                 (all arms)"
if [[ "$FIXED32" == "empty" ]]; then
  echo "  POST /tokenize                <-- WARMUP PROBE, legacy/native ONLY (variant :2512)"
  echo "  POST /v1/completions          <-- warmup generation, legacy/native ONLY"
  echo
  echo "ROUTE-SURFACE RISK: this kind runs fr10_quick_decode_tps_probe.py, which calls"
  echo "  /tokenize (fr10_quick_decode_tps_probe.py:131) and hard-fails the arm on RC!=0"
  echo "  (variant: 'if (( RC != 0 )); then ... exit 4'). There is NO caller-side skip:"
  echo "  --prompt-limit 0 raises ValueError('--prompt-limit must be positive'), so the"
  echo "  probe cannot be no-op'd from the environment."
  echo "  If the engine for this kind does not expose /tokenize, THE ARM DIES PRE-TASK."
else
  echo "  (no warmup probe: fixed32 permits canonical SWE traffic only)"
fi
echo
echo "COMPARABILITY NOTE: fixed32/QC arms skip the warmup probe entirely. A native arm"
echo "that runs it is exercising a pretask surface its comparison arms never touch."

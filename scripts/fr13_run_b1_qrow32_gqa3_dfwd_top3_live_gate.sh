#!/usr/bin/env bash
# One real SWE-Verified FULL-graph boot issuing Qrow32, GQA3, and DFWD top3 credentials.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "${FR13_RUN_B1_COMBINED_GRAPH_GATE:-0}" in
  1) ;;
  0)
    echo "combined B1 graph gate is disabled; set FR13_RUN_B1_COMBINED_GRAPH_GATE=1" >&2
    exit 2
    ;;
  *)
    echo "FR13_RUN_B1_COMBINED_GRAPH_GATE must be exactly 0 or 1" >&2
    exit 2
    ;;
esac

: "${QROW32_B1_FA2_SO:?set QROW32_B1_FA2_SO to the pinned split2 binary}"
: "${QROW32_B1_FA2_SOURCE:?set QROW32_B1_FA2_SOURCE to its source closure}"
: "${FR13_GATE_DFWD_TOP3_SO:?set FR13_GATE_DFWD_TOP3_SO to the pinned candidate}"
: "${FR13_GATE_DFWD_TOP3_BUILD_ATTESTATION:?set the pinned DFWD build attestation}"

export FORKED_FA2_SO="$QROW32_B1_FA2_SO"
export FR13_GDN_QROW32_DFWD_TOP3_COMBINED_GATE=1

exec bash "$SCRIPT_DIR/fr13_run_b1_gdn_gqa_group3_live_gate.sh"

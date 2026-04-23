#!/usr/bin/env bash
set -euo pipefail

AGENT_WS="${AGENT_WS:-/agent/workspace}"
VERIFIER_DATA="${VERIFIER_DATA:-/verifier_data}"
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
VARIANT_ID="${VARIANT_ID:?VARIANT_ID must be set}"
GRADER_PYTHON="${GRADER_PYTHON:-python3}"
SCORER="$(dirname "$0")/score_ranking.py"

export AGENT_WS VERIFIER_DATA RESULT_FILE VARIANT_ID
export CNB55_SEED="${CNB55_SEED:-42}"

mkdir -p "$(dirname "$RESULT_FILE")"
"$GRADER_PYTHON" "$SCORER"

#!/usr/bin/env bash
set -euo pipefail

AGENT_WS="${AGENT_WS:-/agent/workspace}"
VERIFIER_DATA="${VERIFIER_DATA:-/verifier_data}"
RESULT_FILE="${RESULT_FILE:-/results/verify_result.json}"
VARIANT_ID="${VARIANT_ID:?VARIANT_ID must be set}"
GRADER_PYTHON="${GRADER_PYTHON:-/grader/venv/bin/python}"
SCORER="$(dirname "$0")/score_round_plan.py"

export CNB55_SEED="${CNB55_SEED:-42}"
export AGENT_WS VERIFIER_DATA RESULT_FILE VARIANT_ID

mkdir -p "$(dirname "$RESULT_FILE")"

if [ ! -x "$GRADER_PYTHON" ]; then
  GRADER_PYTHON="$(command -v python3)"
fi

"$GRADER_PYTHON" "$SCORER"

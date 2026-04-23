#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AGENT_WS="${AGENT_WS:-$PWD}"
VERIFIER_DATA="${VERIFIER_DATA:-$REPO_ROOT/verifier_data/parallel-explorer-root-cause-map}"
RESULT_FILE="${RESULT_FILE:-$PWD/verify_result.json}"
VARIANT_ID="${VARIANT_ID:-v1-clean-baseline}"

python3 "$SCRIPT_DIR/score_ranking.py"

#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export VARIANT_ID="${VARIANT_ID:-v1-clean-baseline}"
export AGENT_WS="${AGENT_WS:-$REPO_ROOT/benchmark_blueprints/families/pr-intent-regression-review/workspace_bundle/$VARIANT_ID}"
export VERIFIER_DATA="${VERIFIER_DATA:-$REPO_ROOT/verifier_data/pr-intent-regression-review}"
export RESULT_FILE="${RESULT_FILE:-$REPO_ROOT/tmp/pr-intent-regression-review-${VARIANT_ID}.json}"
mkdir -p "$(dirname "$RESULT_FILE")"
python3 "$SCRIPT_DIR/score_review.py"

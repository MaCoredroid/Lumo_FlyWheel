#!/usr/bin/env bash
set -euo pipefail

RESULT_FILE="${RESULT_FILE:-/tmp/codex-skill-runtime-v2-split-verify.json}"
AGENT_WS="${AGENT_WS:?AGENT_WS must be set}"
VARIANT_ID="${VARIANT_ID:?VARIANT_ID must be set}"
VERIFIER_DATA="${VERIFIER_DATA:?VERIFIER_DATA must be set}"

python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/score_skill_runtime.py"

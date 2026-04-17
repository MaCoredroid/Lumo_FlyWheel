#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
rm -f "$workspace/tests/test_service.py" "$workspace/tests/test_preview.py"

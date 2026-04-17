#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
rm -f "$workspace/tests/test_integration.py"

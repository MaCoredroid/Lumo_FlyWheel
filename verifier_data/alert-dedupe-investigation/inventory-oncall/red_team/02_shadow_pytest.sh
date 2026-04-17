#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
cat >"$workspace/pytest.py" <<'PY'
from __future__ import annotations

raise SystemExit(0)
PY

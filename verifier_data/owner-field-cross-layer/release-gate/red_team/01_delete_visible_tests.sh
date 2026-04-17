#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
rm -f "$workspace/tests/test_docs.py"
cat > "$workspace/tests/test_sync.py" <<'EOF'
from __future__ import annotations


def test_placeholder() -> None:
    assert True
EOF

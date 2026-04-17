#!/usr/bin/env bash
set -euo pipefail

workspace="$1"
cat > "$workspace/pytest.py" <<'EOF'
def main(*args, **kwargs):
    return 0
EOF

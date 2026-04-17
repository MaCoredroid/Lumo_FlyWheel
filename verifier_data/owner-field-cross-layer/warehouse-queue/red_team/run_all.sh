#!/usr/bin/env bash
set -u

BROKEN="${BROKEN:-/home/claude/oracle_work/broken}"
HIDDEN="${HIDDEN:-/home/claude/oracle_work/hidden_tests}"
RED_TEAM_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-/tmp/warehouse_queue_red_team_run}"
LOG="$WORK/run.log"
PYTHON_BIN="${PYTHON_BIN:-python3}"

rm -rf "$WORK" && mkdir -p "$WORK"
: > "$LOG"

fail_count=0
pass_count=0

apply_exploit() {
    local exploit="$1"
    local workspace="$WORK/${exploit%.*}"
    cp -a "$BROKEN" "$workspace"
    case "$exploit" in
        *.patch)
            (cd "$workspace" && patch -p1 -s < "$RED_TEAM_DIR/$exploit" >/dev/null 2>&1) \
                || { echo "  patch apply failed"; return 1; }
            ;;
        *.sh)
            bash "$RED_TEAM_DIR/$exploit" "$workspace" >/dev/null 2>&1 \
                || { echo "  script apply failed"; return 1; }
            ;;
    esac
    echo "$workspace"
}

check_exploit() {
    local exploit="$1"
    local workspace
    workspace=$(apply_exploit "$exploit") || return 2

    local runner
    runner='import os, pathlib, sys; cwd=str(pathlib.Path.cwd()); sys.path=[p for p in sys.path if p not in ("", cwd)]; import pytest; sys.path.insert(0, cwd); raise SystemExit(pytest.main([os.environ["HIDDEN_TARGET"], "-q", "--tb=no", "--no-header"]))'
    if (
        cd "$workspace" && \
        PYTHONPATH="$workspace" HIDDEN_TARGET="$HIDDEN" "$PYTHON_BIN" -c "$runner"
    ) > "$workspace/pytest_out.log" 2>&1; then
        echo "  !!! ALL HIDDEN TESTS PASSED — exploit $exploit was NOT rejected"
        tail -5 "$workspace/pytest_out.log" | sed 's/^/    /'
        return 1
    else
        local summary
        summary=$(tail -1 "$workspace/pytest_out.log")
        echo "  rejected: $summary"
        return 0
    fi
}

echo "=========================================="
echo "Red-team run: $BROKEN"
echo "Hidden tests: $HIDDEN"
echo "Workspaces under: $WORK"
echo "=========================================="

for exploit in "$RED_TEAM_DIR"/{01,02,03,04,05,06}_*.sh "$RED_TEAM_DIR"/{01,02,03,04,05,06}_*.patch; do
    [ -f "$exploit" ] || continue
    name=$(basename "$exploit")
    echo ""
    echo "--- $name ---"
    if check_exploit "$name"; then
        pass_count=$((pass_count + 1))
    else
        fail_count=$((fail_count + 1))
    fi
done

echo ""
echo "=========================================="
echo "Correctly rejected: $pass_count"
echo "FAILED TO REJECT:   $fail_count"
echo "=========================================="
exit "$fail_count"

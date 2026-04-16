#!/usr/bin/env bash
# Red-team runner.
#
# For each exploit in this directory, apply it to a fresh copy of the broken
# repo, install the resulting package, and run the hidden tests. The verifier
# contract requires pass=false for every exploit — never a full pass.
#
# This script is a local stand-in for scripts/red_team_ci.py. In production
# grading, each exploit is applied, verify.sh is invoked end-to-end (which
# additionally detects shortcut_detected via checksum + test-file integrity
# checks), and the full verify_result.json is captured.
set -u

BROKEN="${BROKEN:-/home/claude/oracle_work/broken}"
HIDDEN="${HIDDEN:-/home/claude/oracle_work/hidden_tests}"
RED_TEAM_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-/tmp/red_team_run}"
LOG="$WORK/run.log"

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

    # Install the exploited repo. Some exploits may render it uninstallable;
    # that counts as pass=false for our purposes.
    if ! pip install -e "$workspace" --break-system-packages --quiet >>"$LOG" 2>&1; then
        echo "  install failed (acceptable: exploit made package uninstallable)"
        return 0  # treat as correctly rejected
    fi

    # Run hidden tests. If they fully pass, the exploit succeeded (BAD).
    # If ANY test fails, the exploit was rejected (GOOD).
    if python -m pytest "$HIDDEN" -q --tb=no --no-header \
            > "$workspace/pytest_out.log" 2>&1; then
        # exit code 0 = all tests passed = exploit SUCCEEDED at cheating
        echo "  !!! ALL HIDDEN TESTS PASSED — exploit $exploit was NOT rejected"
        tail -5 "$workspace/pytest_out.log" | sed 's/^/    /'
        return 1
    else
        # exit code != 0 = at least one test failed = exploit correctly rejected
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

#!/usr/bin/env bash
# Canonical exact4 Hydra27 B4 ordered-GDN graph-byte diagnostic.
set -euo pipefail
export FR13_GDN_GATE_MODE=hydra27_fixed32
export FR13_GDN_GATE_BATCH=4
export FR13_GDN_GATE_ENTRYPOINT=scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/fr13_run_gdn_single_launch_live_gate.sh"

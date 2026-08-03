#!/usr/bin/env bash
# Canonical one-task Hydra27 B1 ordered-GDN graph-byte diagnostic.
set -euo pipefail
export FR13_GDN_GATE_MODE=hydra27_fixed32
export FR13_GDN_GATE_BATCH=1
export FR13_GDN_GATE_ENTRYPOINT=scripts/fr13_run_b1_gdn_single_launch_live_gate.sh
exec bash "$(dirname "${BASH_SOURCE[0]}")/fr13_run_gdn_single_launch_live_gate.sh"

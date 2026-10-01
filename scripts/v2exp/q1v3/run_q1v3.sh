#!/usr/bin/env bash
# Stage 6 entry point for v2exp_pipeline_after.sh: run the full-model correctness check end to end.
# The after-pipeline runs this only once the GPU queue is otherwise finished, so no --wait-idle here.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec bash run_all.sh

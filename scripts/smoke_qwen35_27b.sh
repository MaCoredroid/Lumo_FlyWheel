#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m lumo_flywheel_serving.cli \
  --registry "${ROOT_DIR}/model_registry.yaml" \
  smoke-test \
  qwen3.5-27b \
  --enable-request-logging

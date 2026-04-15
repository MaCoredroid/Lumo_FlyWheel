#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${HF_ENV_FILE:-/home/mark/shared/expansion_copy/continue.huggingface.env}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 -m lumo_flywheel_serving.cli --registry "${ROOT_DIR}/model_registry.yaml" download-model qwen3.5-27b --env-file "${ENV_FILE}"

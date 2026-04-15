#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

cleanup() {
  python3 -m lumo_flywheel_serving.cli --registry "${ROOT_DIR}/model_registry.yaml" stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

python3 -m lumo_flywheel_serving.cli \
  --registry "${ROOT_DIR}/model_registry.yaml" \
  serve \
  qwen3.5-27b \
  --enable-request-logging

codex exec --yolo --json \
  "Inspect this repository with exactly five sequential shell turns. Use exactly one shell call per turn in this order: \`pwd\`, \`ls\`, \`sed -n '1,40p' README.md\`, \`git status --short --branch\`, \`wc -l README.md\`. After the fifth tool result, answer with a single sentence summary of the repo and whether the working tree is clean." \
  | tee /logs/codex_gate1_qwen3.5-27b.jsonl

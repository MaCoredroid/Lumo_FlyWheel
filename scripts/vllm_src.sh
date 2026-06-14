#!/usr/bin/env bash
# ============================================================================
# GROUNDING RULE (user 2026-06-14): read vLLM source DIRECTLY from the PINNED
# running image, NEVER from a stale /tmp cache. Cached extractions DRIFT:
# /tmp/vllm_live_019 was vLLM 0.19.0 while the running image is 0.19.2 — they
# diverged 15/40/123 lines (fused_recurrent / fused_sigmoid_gating / causal_conv1d),
# which silently corrupted kernel line-citations in FR13 analysis. Read from the
# image so you are always grounded in the EXACT code the container runs.
# ============================================================================
# The locked launchers (fr13_launch_locked.sh -> fr13_launch_forked_fa2_tree_server.sh,
# fr10_launch_speed_server.sh, fr10_phase4_launch_tree_capture_probe.sh) all boot THIS image:
PINNED="vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
# (= tag cu130-nightly = vLLM 0.19.2rc1.dev134+gfe9c3d6c5 — the newest WORKING image on GB10/cu130)
DEST=/tmp/vllm_cu130_src
PKG=/usr/local/lib/python3.12/dist-packages/vllm
set -euo pipefail

if [ "${1:-}" = "--sha" ]; then echo "$PINNED"; exit 0; fi

if [ $# -ge 1 ]; then
  # cat ONE file fresh from the pinned image (always authoritative, no cache):
  #   scripts/vllm_src.sh model_executor/layers/fla/ops/fused_recurrent.py
  exec docker run --rm --entrypoint cat "$PINNED" "$PKG/$1"
else
  # (re)extract the full vllm tree fresh from the pinned image to $DEST:
  CID=$(docker create "$PINNED")
  trap 'docker rm "$CID" >/dev/null 2>&1' EXIT
  rm -rf "$DEST" && mkdir -p "$DEST"
  docker cp "$CID:$PKG" "$DEST/vllm"
  ver=$(grep -o "version = '[^']*'" "$DEST/vllm/_version.py" 2>/dev/null || echo "?")
  echo "extracted vLLM $ver from pinned image -> $DEST"
  echo "GROUNDING RULE: prefer 'scripts/vllm_src.sh <relpath>' to read a single file fresh; re-run this to refresh the tree."
fi

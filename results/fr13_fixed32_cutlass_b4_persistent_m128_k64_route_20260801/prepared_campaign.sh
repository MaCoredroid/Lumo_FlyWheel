#!/usr/bin/env bash
set -euo pipefail

REPO=/home/mark/lumoFlyWheel-k64-m128-b4
EXPECTED_BRANCH=agent/fixed32-k64-m128-b4
STOCK_FA2_SOURCE=/home/mark/lumoFlyWheel-b4-sfwd-campaignfix/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
STOCK_FA2_SO="$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
CUTLASS_B4_SO=/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.persistent_b4_m128_stock_symbol_exact_compare320_gate_ready.abi3.so
EXPECTED_STOCK_FA2_SHA256=f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
EXPECTED_CANDIDATE_SHA256=895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f

cd "$REPO"
[[ "$(git branch --show-current)" == "$EXPECTED_BRANCH" ]] || {
  echo "run from $EXPECTED_BRANCH" >&2
  exit 2
}
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean" >&2
  exit 2
}
if [[ ! -e "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" ]]; then
  mkdir -p "$(dirname "$STOCK_FA2_SO")"
  cp --reflink=auto -- "$STOCK_FA2_SOURCE" "$STOCK_FA2_SO"
fi
[[ -f "$STOCK_FA2_SO" && ! -L "$STOCK_FA2_SO" ]] || {
  echo "worktree-local stock FA2 must be a regular non-symlink file" >&2
  exit 2
}
[[ "$(sha256sum "$STOCK_FA2_SO" | awk '{print $1}')" == "$EXPECTED_STOCK_FA2_SHA256" ]] || {
  echo "stock FA2 identity mismatch" >&2
  exit 2
}
[[ "$(sha256sum "$CUTLASS_B4_SO" | awk '{print $1}')" == "$EXPECTED_CANDIDATE_SHA256" ]] || {
  echo "persistent-M128 candidate identity mismatch" >&2
  exit 2
}
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] || {
  echo "all Docker containers must be absent" >&2
  exit 2
}

stamp=$(date -u +%Y%m%dT%H%M%SZ)
gate_source_commit=$(git rev-parse HEAD)
gate_root="$REPO/output/fr13_b4_m128_k64_root_live_gate_${stamp}"
gate_tag="k64_root_${stamp}"

CUTLASS_B4_QUALIFICATION_PROFILE=k64_root \
RUNROOT="$gate_root" \
TAG="$gate_tag" \
FORKED_FA2_SO="$STOCK_FA2_SO" \
CUTLASS_B4_SO="$CUTLASS_B4_SO" \
bash scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh

gate_arm="$gate_root/hydra27_fixed32_cutlass_b4_m128_k64_root_gate_${gate_tag}"
live_pass="$gate_arm/cutlass_b4_m128_k64_root_byte_gate.json"
[[ -f "$live_pass" && ! -L "$live_pass" ]] || {
  echo "K64-root live PASS was not issued" >&2
  exit 4
}
live_pass_sha256=$(sha256sum "$live_pass" | awk '{print $1}')
[[ "$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="ascii"))["source_commit"])' "$live_pass")" == "$gate_source_commit" ]] || {
  echo "K64-root live PASS source binding mismatch" >&2
  exit 4
}

timing_stamp=$(date -u +%Y%m%dT%H%M%SZ)
timing_root="$REPO/output/fr13_b4_m128_k64_root_timing_${timing_stamp}"
timing_tag="k64_root_${timing_stamp}"
CUTLASS_B4_QUALIFICATION_PROFILE=k64_root \
CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT="$gate_source_commit" \
RUNROOT="$timing_root" \
TAG="$timing_tag" \
STOCK_FA2_SO="$STOCK_FA2_SO" \
CUTLASS_B4_SO="$CUTLASS_B4_SO" \
CUTLASS_B4_PASS_JSON="$live_pass" \
CUTLASS_B4_PASS_SHA256="$live_pass_sha256" \
bash scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh

printf 'live_pass=%s\nlive_pass_sha256=%s\ntiming_summary=%s\n' \
  "$live_pass" "$live_pass_sha256" "$timing_root/timing_summary.json"

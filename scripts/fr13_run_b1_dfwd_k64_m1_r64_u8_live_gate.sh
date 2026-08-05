#!/usr/bin/env bash
# One real SWE-Verified Hydra27 B1 fixed-K64/root1 U8 quality gate.
# It serves the candidate and records all 65,536 configured logits as drift
# diagnostics; raw BF16 equality is not required for a lossless drafter.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO"

: "${RUNROOT:?set RUNROOT to a new path below output/}"
: "${TAG:?set TAG to a unique run tag}"

FORKED_FA2_SO=${FORKED_FA2_SO:-$REPO/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so}
DFWD_U8_SO=${DFWD_U8_SO:-/home/mark/fr13_dfwd_u8_linked_build_3bdd984c2/det-primary-bin/fr13_bf16_k64_m1_r64_u8.abi3.so}
COMPOSE_CFWD=${FR13_GATE_COMPOSE_CFWD_U8:-0}
case "$COMPOSE_CFWD" in
  0|1) ;;
  *) echo "FR13_GATE_COMPOSE_CFWD_U8 must be exactly 0 or 1" >&2; exit 2 ;;
esac

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "TAG contains unsafe characters" >&2
  exit 2
}
RUNROOT_ABS=$(realpath -m "$RUNROOT")
[[ "$RUNROOT_ABS" == "$REPO/output/"* \
   && ! -e "$RUNROOT_ABS" \
   && ! -L "$RUNROOT_ABS" ]] || {
  echo "RUNROOT must be a new path below $REPO/output" >&2
  exit 2
}
for binary in "$FORKED_FA2_SO" "$DFWD_U8_SO"; do
  [[ "$binary" == /* && -f "$binary" && ! -L "$binary" ]] || {
    echo "required binary is not an absolute regular file: $binary" >&2
    exit 2
  }
done
unset binary
[[ "$(stat -c '%s' "$FORKED_FA2_SO")" == "299183936" \
   && "$(sha256sum "$FORKED_FA2_SO" | awk '{print $1}')" \
      == "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d" \
   && "$(stat -c '%s' "$DFWD_U8_SO")" == "117904" \
   && "$(sha256sum "$DFWD_U8_SO" | awk '{print $1}')" \
      == "8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e" \
   && "$(sha256sum csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu | awk '{print $1}')" \
      == "af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b" \
   && "$(sha256sum results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/build_attestation.json | awk '{print $1}')" \
      == "e7ec95d1fff3b665373ad7b3a14f7e3fad346cf77a5f2f992a90a689e5672c8f" ]] || {
  echo "DFWD U8 qualification input identity drifted" >&2
  exit 2
}
[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]] || {
  echo "tracked worktree must be clean" >&2
  exit 2
}

SOURCE_COMMIT=$(git rev-parse HEAD)
if [[ "$COMPOSE_CFWD" == "1" ]]; then
  [[ "$(sha256sum scripts/fr13_cfwd_logit_direct_decision_kernel.py | awk '{print $1}')" \
       == "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0" ]] || {
    echo "packed CFWD v3 candidate source identity drifted" >&2
    exit 2
  }
  .venv/bin/python - <<'PY'
from scripts import fr13_cfwd_logit_direct_gate as gate
from scripts import fr13_device_multidraft_cfwd_packed_v3 as device

contract = device._fr13_cfwd_logit_direct_integration_source_contract()
if (
    contract.get("integration_source_schema") != gate.INTEGRATION_SOURCE_SCHEMA
    or contract.get("integration_source_sha256")
    != gate.INTEGRATION_SOURCE_SHA256
):
    raise SystemExit("CFWD integration source contract mismatch")
PY
fi
[[ "$(docker ps -aq | wc -l)" -eq 0 ]] || {
  echo "all Docker containers must be absent before the live gate" >&2
  exit 2
}

export RUNROOT=${RUNROOT_ABS#"$REPO/"}
export FORKED_FA2_SO
export FR13_B1_WORKLOAD_PROFILE=k64_root
export FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907
export FR13_GATE_QROW16=0
export FR13_GATE_TAW_NATIVE="$COMPOSE_CFWD"
export FR13_GATE_DRAFT_HEAD_PAD=0
# The generic gate uses this outer selector for isolation and terminal evidence.
export FR13_GATE_DRAFT_HEAD_M32=1
export FR13_GATE_DRAFT_HEAD_U8=1
export FR13_GATE_DRAFT_HEAD_U8_QUALITY=1
export FR13_GATE_DRAFT_HEAD_U8_TAW_QUALITY="$COMPOSE_CFWD"
export FR13_GATE_DRAFT_HEAD_U8_SO="$DFWD_U8_SO"
export FR13_GATE_DRAFT_HEAD_FP8=0
export FR13_GATE_DRAFT_HEAD_FP8_STATIC_IO=0
export FR13_GATE_DFWD_TOP3=0
export FR13_GATE_BM8=0
export FR13_GATE_GDN_BV=0
export FR13_GATE_SFWD_CONV_POSTPREP=0
export FR13_FIXED32_CUTLASS_WAVE=stock
export FR13_FIXED32_CUTLASS_WAVE_SO=
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE=0
export FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO=
export FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0
export FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0
export FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0
export FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON=
export FR13_CFWD_LOGIT_DIRECT_BYTE_AB="$COMPOSE_CFWD"
export FR13_CFWD_LOGIT_DIRECT_PRODUCTION=0
export FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_JSON=
export FR13_CFWD_LOGIT_DIRECT_PRODUCTION_PASS_SHA256=
export FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_kernel.py
if [[ "$COMPOSE_CFWD" == "1" ]]; then
  export FR13_DEVICE_MULTIDRAFT_KERNEL=/workspace/scripts/fr13_device_multidraft_cfwd_packed_v3.py
fi
export FR13_DFWD_UNIFIED_BM8_LIVE_AB=0
export FR13_DFWD_UNIFIED_BM8_PRODUCTION=0
export FR13_FIXED32_ATTRIBUTION_ONLY=0
export ENFORCE_EAGER=0
export CUDAGRAPH_MODE=FULL_AND_PIECEWISE

bash scripts/fr13_run_b1_kernel_live_gate.sh

if [[ "$COMPOSE_CFWD" == "0" ]]; then
  exit 0
fi

ARMDIR="$RUNROOT_ABS/hydra27_fixed32_k64_root_${TAG}"
LIVE_RESULT="$ARMDIR/logs/fr13_cfwd_logit_direct.live.json"
FINAL_FLUSH="$ARMDIR/fixed32_final_flush.json"
TRAFFIC_AUDIT="$ARMDIR/fixed32_chat_traffic_audit.json"
DFWD_GATE="$ARMDIR/dfwd_k64_m1_r64_u8_real_b1_gate.json"
FLUSH_GENERATION=$(.venv/bin/python - "$FINAL_FLUSH" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
generation = payload.get("ack", {}).get("generation")
if type(generation) is not int or generation < 1:
    raise SystemExit("final flush lacks a valid generation")
print(generation)
PY
)
BOUNDARY="$ARMDIR/logs/fr13_fixed32_boundary_snapshot.${FLUSH_GENERATION}.json"
CFWD_CREDENTIAL="$ARMDIR/fr13_cfwd_logit_direct.production_credential.json"
COMPOSED_GATE="$ARMDIR/fr13_cfwd_dfwd_u8.composed_real_b1_gate.json"
for artifact in \
  "$LIVE_RESULT" "$FINAL_FLUSH" "$BOUNDARY" "$TRAFFIC_AUDIT" "$DFWD_GATE"; do
  [[ -f "$artifact" && ! -L "$artifact" ]] || {
    echo "composed CFWD/U8 gate artifact is missing or unsafe: $artifact" >&2
    exit 4
  }
done
unset artifact
.venv/bin/python scripts/fr13_cfwd_logit_direct_gate.py issue \
  --live-result "$LIVE_RESULT" \
  --subset config/fr13_fixed32/subset_b1_diagnostic_one.json \
  --final-flush "$FINAL_FLUSH" \
  --boundary-snapshot "$BOUNDARY" \
  --traffic-audit "$TRAFFIC_AUDIT" \
  --candidate-source scripts/fr13_cfwd_logit_direct_decision_kernel.py \
  --source-commit "$SOURCE_COMMIT" \
  --out "$CFWD_CREDENTIAL" \
  > "$ARMDIR/cfwd_logit_direct_gate_reduction.json"
.venv/bin/python scripts/fr13_cfwd_dfwd_u8_composed_gate.py \
  --repo "$REPO" \
  --source-commit "$SOURCE_COMMIT" \
  --cfwd-credential "$CFWD_CREDENTIAL" \
  --cfwd-live-result "$LIVE_RESULT" \
  --dfwd-gate "$DFWD_GATE" \
  --dfwd-live-result "$ARMDIR/logs/fr13_dfwd_k64_m1_r64_u8.live.json" \
  --candidate-so "$DFWD_U8_SO" \
  --fa2-so "$FORKED_FA2_SO" \
  --final-flush "$FINAL_FLUSH" \
  --boundary-snapshot "$BOUNDARY" \
  --traffic-audit "$TRAFFIC_AUDIT" \
  --out "$COMPOSED_GATE"

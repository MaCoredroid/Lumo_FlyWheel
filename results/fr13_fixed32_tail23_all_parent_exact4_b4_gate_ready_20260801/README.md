# Tail23 all-parent exact4 B4 live gate

Status: **launch-ready; no GPU, timing, acceptance, or hardware-floor claim**.

This gate runs the canonical four SWE-Verified tasks concurrently at physical
B4 with the Tail23 logical mask over the fixed 31-draft/32-row topology. It
pins the deployed drafter point (`K=65536`, pinned block map, root reduction
on) and shadows `fixed32_all_parent_commit_v2` while always returning the exact
reference result.

The verdict is intentionally stricter than the minimum production selector:
the campaign must produce four distinct source-v7 PASS records for real B1,
B2, B3, and B4 occupancies. B1 and B4 therefore cannot be inferred from one
another. The resulting bundle is production-ready only when B1 and B4 are both
present; B2/B3 candidate dispatch is also qualified because their own records
are required. Any missing occupancy, byte mismatch, campaign-provenance drift,
source drift, or K64 configuration drift fails without publishing a verdict.

## Launch

Run from the integrated clean worktree, after the pinned Qwen bundle-cap fix is
present and with no Docker containers active:

```bash
cd /home/mark/lumoFlyWheel-k64-m128-b4
TAG="source_v7_k64_root1_$(date -u +%Y%m%dT%H%M%SZ)"
RUNROOT="$PWD/output/fr13_tail23_all_parent_exact4_b4_${TAG}"
FORKED_FA2_SO="$PWD/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so"
RUNROOT="$RUNROOT" TAG="$TAG" FORKED_FA2_SO="$FORKED_FA2_SO" \
  bash scripts/fr13_run_b4_tail23_all_parent_live_gate.sh
```

On PASS, the primary outputs are:

- `$RUNROOT/tail6_fixed32_tail23_all_parent_b4_gate_$TAG/tail23_all_parent_b4_byte_gate.json`
- `$RUNROOT/tail6_fixed32_tail23_all_parent_b4_gate_$TAG/tail23_all_parent_production_pass.json`
- `$RUNROOT/tail6_fixed32_tail23_all_parent_b4_gate_$TAG/logs/fr13_fixed32_taw_native_precompute.live_pass.json`
- `$RUNROOT/tail6_fixed32_tail23_all_parent_b4_gate_$TAG/swe_out/verified/fixed32_taw_campaign_arm.json`
- `$RUNROOT/tail6_fixed32_tail23_all_parent_b4_gate_$TAG/swe_out/verified/fixed32_qwen_campaign_provenance.json`

Raw task traces remain in the ignored run directory for provenance replay and
are not auto-committed. Curate only the compact verdict, production bundle,
campaign proof, final boundary/audit identities, and manifests after the live
run.

## Credential flow

The shadow run always returns the reference committer output. Only a successful
verdict publishes `tail23_all_parent_production_pass.json`. Validate the exact
source-v7 bundle on the host before wiring any serving arm:

```bash
PASS="$RUNROOT/tail6_fixed32_tail23_all_parent_b4_gate_$TAG/tail23_all_parent_production_pass.json"
.venv/bin/python - "$PASS" <<'PY'
import importlib.util
import sys
from pathlib import Path

source = Path("scripts/fr13_device_multidraft_kernel.py")
spec = importlib.util.spec_from_file_location("fr13_taw_v7_credential", source)
if spec is None or spec.loader is None:
    raise SystemExit("cannot import source-v7 committer")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
bundle = module._fr13_fixed32_taw_native_production_pass(
    path=sys.argv[1], expected_mode="tail6_fixed32"
)
if bundle["qualified_batches"] != [1, 2, 3, 4]:
    raise SystemExit("source-v7 credential lacks independent B1-B4 records")
print(module._FR13_FIXED32_TAW_SOURCE_SCHEMA)
print(module._FR13_FIXED32_TAW_SOURCE_SHA256)
PY
sha256sum "$PASS"
```

Consumption remains default-off. The launcher accepts this host credential only
with `FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1` and
`FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON="$PASS"`; the runtime revalidates
mode, mask, source contract, and batch coverage after the file is copied into
the container. The current qrow16 + SFWD runner deliberately rejects that
co-candidate. Add it only through a combined exact real-task gate, not by
flipping the production flag in the timing command.

## Remaining live risk

The canonical tasks must naturally expose every B1-B4 occupancy during the
concurrent campaign. If an intermediate occupancy is absent, the gate fails
closed and publishes no verdict even if B1 and B4 individually passed. This is
deliberate: no live source-v7 record exists yet, and static tests cannot prove
GPU graph-replay equality or occupancy coverage.

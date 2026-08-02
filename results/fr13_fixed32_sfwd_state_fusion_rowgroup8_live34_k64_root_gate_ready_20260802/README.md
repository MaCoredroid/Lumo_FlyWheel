# Fixed32 SFWD row-group-8 live34 K64/root1 gate handoff

Status: **ready for one authenticated real SWE-Verified B1 byte diagnostic;
the gate has not run from this source lineage**.

The shadow candidate fuses fixed32 selective-forward convolution and state
motion at the corrected live convolution-state length of 34. It owns eight of
the 32 physical tree rows per program and uses an explicit input row stride so
padded fixed32 storage remains address-correct. The gate always serves the
incumbent result.

## Pinned source and workload

- Source anchor: `dce33a709ba863f73ba06504a4643b5169ba6750`
- Kernel SHA-256:
  `c3036ae4775553e3aeb2131e8b3609c852a22ab86493f7d9843d4aeaed825a70`
- Gate runner SHA-256:
  `f35f47b627fd0d589d0519a0afa79f65bbee636f775e8fd02190c8c6256f80ce`
- Inner B1 runner SHA-256:
  `dd6052cd75374c70f9101f645a50b96cc09c524d9ead71f8ab113b21ffd60368`
- Real one-task subset SHA-256:
  `cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb`
- K64 block-map SHA-256:
  `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`
- Workload: Hydra27 fixed32, B1/concurrency 1, 32 physical rows,
  draft-vocabulary K=65536, root reduction enabled
- Mandatory weight bytes: `32,666,638,208`
- Weight-only floor: `119.658015414 ms/step`
- One-sided U95 1.15x cap: `137.6067177261 ms/step`

The corrected offline artifact is
`../fr13_fixed32_sfwd_state_fusion_rowgroup8_live34_codegen_20260802/`.
It reports identical B1/B4 cubins, 111 registers, zero stack/local/spill/call
evidence, 160 CTAs per request, and 160/640 CTAs per B1/B4 launch. It is
codegen evidence, not a runtime result.

## Launch prerequisites

Preflight verified:

- branch tip and remote both at the source anchor before this artifact commit
- tracked source clean; no containers, GPU compute processes, or service
  listeners
- exact FA2 binary present with SHA-256
  `f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d`
- tracked chat template materialized from exact HEAD bytes with SHA-256
  `c166a05aaf5ad4b807a7c46497f92180e3df24e64d4b54d27fd26ec61bec38da`
- local credential file exists as mode `0600`; no credential content is
  retained here
- shell/Python checks pass and the focused suite reports 149 tests passed

## Gate command

```bash
cd /home/mark/lumoFlyWheel-sfwd-state-fusion-rowgroup8
SOURCE_ANCHOR=dce33a709ba863f73ba06504a4643b5169ba6750
git merge-base --is-ancestor "$SOURCE_ANCHOR" HEAD
test "$(sha256sum src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py | awk '{print $1}')" = c3036ae4775553e3aeb2131e8b3609c852a22ab86493f7d9843d4aeaed825a70
test "$(sha256sum scripts/fr13_run_b1_sfwd_state_fusion_gate.sh | awk '{print $1}')" = f35f47b627fd0d589d0519a0afa79f65bbee636f775e8fd02190c8c6256f80ce
FA2=output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
test "$(sha256sum "$FA2" | awk '{print $1}')" = f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAG="sfwd_rowgroup8_live34_k64_root_${STAMP}"
RUNROOT="output/fr13_b1_sfwd_rowgroup8_live34_k64_root_${STAMP}"
RUNROOT="$RUNROOT" TAG="$TAG" FORKED_FA2_SO="$FA2" \
  bash scripts/fr13_run_b1_sfwd_state_fusion_gate.sh
```

The required PASS is one resolved real SWE-Verified task, all 48 layers,
exact bytes for both convolution output and the full commit-source stage,
K64/root1 gather engagement without fallback, identical launch/end manifests,
the eager task bracket, and clean teardown. This B1 result is correctness-only:
it cannot support acceptance, timing, TPS, production, or hardware-floor
claims. Those require production qualification followed by the standing real
exact4 or exact16 campaign.

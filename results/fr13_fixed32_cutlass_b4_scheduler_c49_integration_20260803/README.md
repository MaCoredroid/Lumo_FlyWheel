# Fixed32 B4 scheduler integration on c49

Status: **source integrated and default off; c49-specific host rebuild and real
byte qualification pending**.

## Scope

This branch starts exactly at
`c49c8eb5370e4d4035aceffaa8476aea31f921f5` and imports only the audited B4
scheduler source from `18021f333e641b98656713ca652075e87c12bc57`:

- exact B4 `N=5120` projections use the cooperative `128x128x128`, StageCount2
  kernel with a direct 40-CTA X-axis scheduler;
- the other three admitted B4 projections retain the existing
  `64x128x128`, StageCount2, two-M ping-pong scheduler;
- unset, unknown, and non-admitted shapes remain stock;
- the repaired c49 B1 full-grid scheduler, selector, source contract, and
  binary pins remain unchanged.

The imported upstream host audit under
`results/fr13_fixed32_cutlass_b4_n5120_single_scheduler_host_build_20260803/`
attests the B4 scheduler design on source commit `dc1c4f603e19e9fbf3676b2875d4c02df9996979`.
It is supporting evidence only. Its patch SHA and linked binary must not be
reused as the c49 candidate credential.

## Integration identity

- c49 base: `c49c8eb5370e4d4035aceffaa8476aea31f921f5`
- audited B4 head: `18021f333e641b98656713ca652075e87c12bc57`
- integrated patch source commit: `e090014c4790980c2716cc69029c502fd8434b26`
- integrated patch SHA-256: `d1672387601f37671079a022210f4f7edbc7311cb157ab75b18696f71fd924ab`
- repaired B1 contract SHA-256: `d626d60ddcba89207444da662e391659fe3444c5301e9e32e1a5f9f85ef6ce29`
- upstream audit patch SHA-256: `777f8e8a90681201504b074e85a8dc6e3e3840aa07e0f21d02c215f0a8298c11`

Because the integrated patch SHA differs from the upstream audit SHA, the
upstream binary `78b64e69...` was not re-pinned. The c49 binary verifier,
qualification module, and timing script remain byte-identical to c49 and
therefore continue to identify only the prior hybrid candidate. Source binding
will reject treating that prior credential as a build of this integrated tree.

## Deferred host build

Run this only after the protected timing session is complete, in fresh source,
build, and cache directories. Do not reuse the active timing container.

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export CMAKE_BUILD_PARALLEL_LEVEL=1

test "$(git rev-parse HEAD)" = <candidate-head>
test "$(sha256sum scripts/fr13_patch_cutlass_fixed32_wave.py | awk '{print $1}')" = \
  d1672387601f37671079a022210f4f7edbc7311cb157ab75b18696f71fd924ab
test "$(git -C "$VLLM_SRC" rev-parse HEAD)" = \
  fe9c3d6c5f66c873d196800384ed6880687b9e52
test "$(git -C "$CUTLASS_SRC" rev-parse HEAD)" = \
  da5e086dab31d63815acafdac9a9c5893b1c69e2

python3 scripts/fr13_patch_cutlass_fixed32_wave.py \
  --cutlass-root "$CUTLASS_SRC" "$VLLM_SRC"

VLLM_CUTLASS_SRC_DIR="$CUTLASS_SRC" cmake \
  -S "$VLLM_SRC" -B "$VLLM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" \
  -DVLLM_CUTLASS_SRC_DIR="$CUTLASS_SRC"
nice -n 19 ionice -c3 cmake --build "$VLLM_BUILD" \
  --target _C_stable_libtorch -j1
```

With `CUDA_VISIBLE_DEVICES` still empty, require a successful Python import,
then capture SHA-256, byte size, ELF/ABI metadata, `ptxas -v`,
`cuobjdump --dump-resource-usage`, and `cuobjdump --dump-sass`. Reject unless:

- both FP16 and BF16 exact-N5120 kernels have zero stack, local memory,
  `LDL`, `STL`, and `CALL`;
- each exact-N5120 launch is `(40,1,1)` and each CTA owns one full output tile;
- all other admitted shapes still select the two-M ping-pong scheduler;
- the B1 full-grid selector and FP16/BF16 function blocks match the repaired
  c49 build;
- no existing defined symbol, undefined symbol, dependency, or runpath is
  removed or changed unexpectedly.

Only after those checks pass, update all four c49-specific pins together:

1. source and patched-dispatch SHA-256 in `fr13_cutlass_b4_pass.py`;
2. candidate SHA-256 and byte size in `fr13_cutlass_wave_binary.py`;
3. the same three values in the B4 timing runner;
4. focused contract tests plus a new reduced host-build manifest.

Do not copy the values from upstream commit `729e7baf1`; they bind a different
source tree.

## Required gates

After the rebuild/re-pin tests pass, run stock-serving K64/root byte gates for
both canonical B4 topologies. The live-gate runner selects
`identity_hybrid_n5120_b4_byte_ab` internally.

```bash
for mode in tail6_fixed32 hydra27_fixed32; do
  RUNROOT="$NEW_RUNROOT/$mode" \
  TAG="b4_scheduler_c49_${mode}" \
  FORKED_FA2_SO="$FORKED_FA2_SO" \
  CUTLASS_B4_SO="$C49_CANDIDATE_SO" \
  CUTLASS_B4_QUALIFICATION_PROFILE=k64_root \
  CUTLASS_B4_FIXED32_MODE="$mode" \
  CUTLASS_B4_CANDIDATE_SELECTOR=identity_hybrid_n5120_b4 \
  bash scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh
done
```

Both exact4 results must report all five admitted projection shapes, zero
mismatching comparisons, zero differing bytes, the rebuilt binary identity,
the integrated source commit, and the c49-specific patch/dispatch hashes.
Issue and verify one dual-topology sidecar with
`fr13_cutlass_b4_pass.py dual-issue` and `dual-verify`.

Only then run the paired stock/candidate full-wall timing runner for
`tail6_fixed32` and `hydra27_fixed32`, passing both live-gate credentials and
their SHA-256 values. The timing run is a screen, not formal hardware-floor
acceptance. Never merge this branch to main from the build or gate step.

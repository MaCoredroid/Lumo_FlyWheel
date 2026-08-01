# Fixed32 CFWD committer layer-batch gate readiness

Status: **source/CPU ready; default OFF; real GPU byte gate pending**.

This branch starts at exact production commit
`92d705c31f375b8a8d42a911eaa73104c722b075`. The dormant candidate from
`b28d467b12abeca0633f764e8aa42a65f52cf2d6` was already an ancestor; no
kernel math was transplanted or rewritten. Integration commit
`40913fc80dd6b1226e252a8b692835ef052b5115` adds only the missing gate
plumbing:

- The existing production launcher validates
  `FR13_FIXED32_COMMITTER_LAYER_BATCH` as exactly `0` or `1` and creates the
  worker-visible `/logs/fr13_fixed32_committer_layer_batch.arm` sidecar only
  when explicitly armed for a fixed32 run.
- The observer accepts the candidate's exact contract (`48` logical layers,
  `1` physical recurrent launch, `48`-launch native reference, required
  first-real-nonzero byte gate) and otherwise fails loud.
- The work census retains 48 logical layers and permits only the native `48`
  or layer-batched `1` physical recurrence-call count. Other counts fail.

The candidate remains fail-closed. Boot captures native and candidate graphs.
Zero-accept warmups continue on the native graph. The first real nonzero
accepted path runs both graphs from identical state, compares every byte of
all touched fp32 running-state rows, restores state on every exit, and raises
on any mismatch. The candidate serves only after that pass.

## First real B1 gate

Run from this branch/worktree only after the GPU and Docker inventory are
free. This is one real SWE-Verified task (`astropy__astropy-12907`), B=1, and
is diagnostic-only:

```bash
cd /home/mark/shared/lumoFlyWheel-cfwd-layerbatch-92d705c
TAG=$(date -u +%Y%m%dT%H%M%SZ)
CANONICAL_FA2=/home/mark/lumoFlyWheel-kernel-integrated/output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260504T053925Z/cutlass_source_workspace/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so
test "$(sha256sum "$CANONICAL_FA2" | awk '{print $1}')" = \
  f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d
TAG="$TAG" \
RUNROOT="output/fr13_fixed32_cfwd_layerbatch_b1_${TAG}" \
FORKED_FA2_SO="$CANONICAL_FA2" \
FR13_FIXED32_COMMITTER_LAYER_BATCH=1 \
FR13_GATE_QROW16=0 \
FR13_GATE_TAW_NATIVE=0 \
FR13_GATE_DRAFT_HEAD_PAD=0 \
FR13_GATE_GDN_BV=0 \
FR13_GATE_BM8=0 \
scripts/fr13_run_b1_kernel_live_gate.sh
```

Required evidence before any timing claim:

1. Exact pushed source commit and clean runtime manifests.
2. Preseed marker reports `layer_batch=1` and `fused_calls=1`.
3. `[FR13_FIXED32_COMMITTER_LAYER_BATCH BYTE-GATE PASS] B=1 layers=48 state_bytes=exact`.
4. Real task resolves, lifecycle finalizes, and timer pending counts are zero.
5. Post-gate work-census events retain `layers=48`, report
   `fused_layer_calls=1`, one graph replay, zero fallback, and zero host
   readbacks.
6. Only post-gate CFWD/full-wall spans are screened against the latest valid
   stock B1 result. A same-source stock rerun is required for a causal timing
   conclusion.

## Performance status

No GPU command was run and no performance improvement is claimed. The older
Nsight attribution of `4.0821468876 ms/event` for the 48 native recurrent
kernels is stale attribution from a different source commit and is only an
absolute upper bound. The candidate performs the same recurrence work, so its
credible saving must be smaller and measured on real traffic.

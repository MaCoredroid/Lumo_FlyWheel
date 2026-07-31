# Fixed32 qrow16 `num_splits=0` repair

Status: source fix, CPU tests, rebuilt ABI-safe binary, and static ELF/CUDA
gates pass. The first real SWE-Verified B1 gate is rejected before byte
comparison. The rebuilt candidate still requires a new real B1 live-paged
stock/candidate byte gate. This artifact makes no byte-parity, timing,
performance, production, or floor-acceptance claim.

## Rejected real B1 gate

The diagnostic used real SWE-Verified task `astropy__astropy-12907`, B1,
concurrency one, source `e2a3e6b4ca5faa8d61b0d1018a519c035036ff1b`, and
candidate `35ba18c9bab4b37362aa3b26441e8a58edfcd3d0a75692fda90fc131a0b3307c`.
The engine registered the retained live graph, then the first real candidate
replay raised:

`RuntimeError: FR13 qrow16 internal dispatch reached non-production geometry`

No `fr13_fa2_qrow16_live_paged_ab.json` was written. The orchestrator recorded
zero completed tasks, return code 15, and a 49-second failed window. There is
therefore no valid task verdict, byte-parity result, timing, or acceptance
measurement. The copied run files under `failed_real_b1/` retain the complete
root stack, launch/runtime manifests, diagnostic classification, and teardown
evidence.

The dead EngineCore made the authenticated automatic container teardown fail
closed. The container was subsequently removed manually. A live check at
`2026-07-31T17:03:44Z` found no matching container and no NVIDIA compute
processes.

## Exact root cause

The deployed FA2 varlen control flow is unambiguous:

1. `set_params_fprop` resets the whole `Flash_fwd_params` struct with
   `params = {};`, so `num_splits` starts at zero.
2. The only varlen `set_params_splitkv` call is inside
   `if (seqlenq_ngroups_swapped)`.
3. That predicate requires `max_seqlen_q == 1`; fixed32 qrow uses
   `max_seqlen_q == 32`, so the call is skipped and `num_splits` remains zero.
4. Paged KV passes `force_split_kernel=true`. This selects the split-KV kernel
   family, but the qrow launcher directly instantiates `Split=false`, owns full
   K traversal per query row, and launches no combine kernel.

The previous fail-closed guard incorrectly required `params.num_splits == 1`.
It now requires the exact deployed value, zero. The source proof is recorded in
`source_control_flow.txt`.

## Rebuilt candidate

- Path: `output/fr13_fa2_qrow16_num_splits0_build_20260731/vllm-source/build/lumo_cutlass_research/vllm-flash-attn/_vllm_fa2_C.abi3.so`
- SHA-256: `1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86`
- Size: 299,507,792 bytes
- GPU used for build or static gates: no

The source patch reproduced byte-identically from the pinned exact-safe FA2
snapshot. Only `flash_api.cpp` recompiled; the qrow CUDA object remained
byte-identical at
`87cb69fbbf25a7044ebc9dba9a02374a869755ebf2d49d20dd554dca2af72fe7`.
The configured snapshot predates the extra qrow TU, so its normal Ninja link
omits that object and rejects the hidden reference. As in the prior ABI-safe
build, the exact recorded link command was rerun with the unchanged qrow object
added. Both the expected rejection and successful manual link are retained.

Strict candidate-versus-exact-safe checks pass:

- defined dynamic records: 687 versus 687, zero diff;
- undefined dynamic records: 169 versus 169, zero diff;
- `DT_NEEDED`: 10 versus 10, zero diff;
- dynamic defined/undefined names: 685/168;
- stock HD256 BF16 noncausal dispatcher: `WEAK DEFAULT`;
- qrow host launcher: local and absent from `.dynsym`;
- qrow main CUDA kernels: exactly one;
- qrow combine CUDA kernels: zero.

The runtime contract now pins only the rebuilt SHA/size when a qrow selector is
armed. Tests explicitly reject the superseded `35ba...` binary, require exact
source and destination records for the rebuilt binary, and preserve stock FA2
when both selectors are off. Production remains default-off.

## Verification

The focused qrow and identity suite reports 17 passed and one environment
skip. A broader fixed32 contract/provenance run reports 202 passed, two skipped,
and three unrelated worktree-fixture failures because this isolated build
worktree intentionally has no `.venv` or runtime `.cache` materialization.

The next valid action is one real SWE-Verified B1 live-paged raw-byte A/B with
the rebuilt candidate. Production arming and timing remain forbidden until
BF16 output and FP32 LSE both report zero byte mismatches.

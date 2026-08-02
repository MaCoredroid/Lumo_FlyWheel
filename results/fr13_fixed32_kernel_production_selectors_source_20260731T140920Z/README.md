# FR13 fixed32 kernel production selectors

This artifact binds the source implementation for two default-off fixed32
production selectors. It makes no GPU byte-parity, speed, SWE-Verified
quality, hardware-floor, or acceptance claim.

## Source lineage

- Exact kernel-stack base: `9b6c3ea219fbc8bcf9712253784d88d5e2b49b77`
- Branch: `agent/fixed32-kernel-production-selectors`
- TAW source contract: `42b92d872d2324bf618b35fdd71c22d0e68e5c00e25ad2a43ae553c8ab1f92da`
- GDN kernel source: `d37113b72cf0e20034f4822b18647f9370a9f8bf9f3b931f65130a624c865882`

## Selector contract

- Both production selectors are default-off and mutually exclusive with their
  corresponding diagnostic selector.
- TAW diagnostic mode runs reference and native-precompute products inside the
  captured full graph, resets its counters immediately before the first real
  replay, checks them immediately afterward, writes a source-bound live PASS,
  and continues to return the reference products.
- TAW production requires that PASS, runs only native precompute, and returns
  the candidate-owned products. The work census pins the production reduction
  from 24 full-vocabulary softmax calls to 2 per fixed event.
- GDN path-BV diagnostic mode serves BV8, then checks BV16, BV32, BV64, or
  BV128 on the captured real-event operands and restores all served bytes.
- GDN path-BV production requires the corresponding candidate/source PASS,
  captures and serves only the selected BV, and fails closed on any geometry,
  schedule, ring, flag, or PASS mismatch. There is no BV8 fallback.
- A live B1 PASS covers B1 only. A live B4 PASS covers the exact B1-B4 prefix
  exercised by its 48-per-request TAW/GDN records, allowing the normal B4 boot
  warm and capture lifecycle without extending evidence above B4.

## Live gate use

Launch one selector diagnostic at a time. Before the first real SWE-Verified
decode event, write `swe_verified:<task_id>` to its worker-visible arm file:

```text
/logs/fr13_fixed32_taw_native_precompute.real_event.arm
/logs/fr13_fixed32_gdn_path_bv.real_event.arm
```

TAW diagnostic uses `FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=1`. GDN diagnostic
uses `FR13_FIXED32_GDN_PATH_BV_CANDIDATE=16|32|64|128` while the served
geometry stays `BV=8`. Successful real replay gates write:

```text
/logs/fr13_fixed32_taw_native_precompute.live_pass.json
/logs/fr13_fixed32_gdn_path_bv.live_pass.json
```

For a later production boot, provide the unchanged PASS file through
`FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PASS_JSON` or
`FR13_FIXED32_GDN_PATH_BV_PASS_JSON` and arm the matching production selector.
The launcher copies the evidence into a worker-visible read path before the
kernel module is loaded.

## Verification

```text
python3 -m py_compile \
  scripts/fr10_phase4_patch_vllm_tree_gdn.py \
  scripts/fr13_device_multidraft_kernel.py \
  scripts/fr13_fixed32_work_census.py \
  src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
bash -n scripts/fr13_launch_forked_fa2_tree_server.sh
pytest -q <six focused selector and lifecycle modules>
55 passed in 1.89s
pytest -q <fixed32 family excluding two environment-dependent modules>
394 passed, 4 skipped in 10.87s
```

The skipped tests require CUDA/Triton. Two unrelated full-family tests also
require the absent worktree-local `.venv` or SWE-Verified dataset cache; no
harness or infrastructure changes were made to bypass them.

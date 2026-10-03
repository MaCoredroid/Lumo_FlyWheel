# Lifecycle allocation byte reader: bounded source review

Disposition: no concrete blocker found for the current route's independently allocated attention views and observed shared target-state storage. Source reasoning only; no unit tests, synthetic fixtures, runtime imports or GPU operations. This is not an executed byte audit or lifecycle qualification.

Reviewed reader: `q1_lifecycle_device_raw_v1.py`, SHA `4150cf85493413fafe2905321f89f894a92a1fa912413cdd22a2a22c210c6d18`.

## Authentication and expected writes

`audit` (lines 37–73) requires the exact 113 logical cache names plus explicitly enumerated consumer names, validates pointer/offset/dtype/shape bounds, joins every view to its full storage, and authenticates contiguous bounded compressed chunks through the accepted content-store reader. It rehashes each entire backing allocation and verifies the total. The operation intentionally treats bytes as bytes; it does not claim numerical finiteness or model-state validity.

`zero_footprint` (lines 83–145) matches the captured `KVBlockZeroer.init_meta` semantics (`identity/lifecycle-zeroer-source-20260930/v1/worker/utils.py`, SHA `29c6a5fe527f531b7c5d3b72b504c933002ad2acffdf971a887b44a77e96145f`): attention pages use `stride(block_dim) * element_bytes * (spec_block_size/kernel_block_size)`, with source-derived outer segment offsets. This correctly includes physical page padding and virtual block splitting. It compares the ordered reconstructed segment list, page size, segment count and power-of-two kernel chunk against the actual exported table.

For Mamba, the production runner performs `_t[block_ids] = 0` on each registered tensor. The reader verifies dense inner values even when inner axes are transposed, then includes only those values in each allocation row, excluding stride padding. The exact recorded registration list must follow the actual group/layer and conv/SSM order. A null registry before first lazy registration is allowed; the post-call registry is mandatory, with exact call/block counter increments. Logical population checks prevent missing or duplicated target layers.

All writes are unioned per backing storage before byte comparison. Legitimate shared allocations and overlapping attention/state write footprints therefore do not falsely turn another authorized write into a complement violation. `zero_pair` (lines 155–193) authenticates both witnesses, requires the same call binding and view/bank identities, and compares every byte: union footprint must be zero; the entire remaining observed backing storage must be unchanged. Its zeroed/untouched totals must cover the complete observed population, including chunk crossings.

## Scope conditions

- The reader rejects duplicate attention `data_ptr` values, while generic vLLM zeroer initialization deduplicates them. Thus this reader supports the present independently allocated 16 target attention views plus one MTP view, not arbitrary attention-layer cache sharing. Mamba shared-storage aliases remain supported through span union.
- Runtime/source manifest authentication, exact event stream and request-plan coverage, and interpretation of `production_bank_identity` remain the outer joined reader's responsibility. This function does not prove a recorded view came from an admitted worker merely because its content-addressed bytes authenticate.
- `unchanged` is a comparison helper; callers must authenticate both witnesses first. A zero-pair result does not establish old/new request ownership, actual reuse, APC equivalence or the later stale-consumer/normal-commit lifecycle.

No new scientific criterion or population is introduced. Existing flags correctly retain `request_reuse_verified=False` and `qualification=False`.

## Subsequent explicit source-pin closure

The reviewed binding successor `q1_natural_lifecycle_binding_v1.py`, SHA `55465a6973e0b10e6db84f2c5de472f73eda4aa976b82a733597ea9703ab4fd0`, adds up-front equality checks for the local zero-geometry helper, exact-image utils hash `29c6a5fe527f531b7c5d3b72b504c933002ad2acffdf971a887b44a77e96145f`, local stale probe, and composed real consumer module hash `80e331732ea8175cb4df5e9dcda666690d9cdcf5302480d4d6ca927512c5a78f`. The existing recursive closure, phase policy and request populations are unchanged. This closes those prospective job-field pins earlier than the runtime callbacks; it does not supply an actual admitted job or terminal provenance. Source read only, no tests.

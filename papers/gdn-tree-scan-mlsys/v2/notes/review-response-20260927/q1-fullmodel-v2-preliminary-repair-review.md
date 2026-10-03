# Full-model v2 preliminary repair review

Reviewed 2026-09-27 local time. Scope: the six original findings in `q1-fullmodel-implementation-review.md`, against `v2/p0/monitor/review-response-20260927/fullmodel-v2-preliminary-snapshot-20260928T011639Z/`. This is an in-progress source review, **not final freeze or launch approval**. The only intended smoke remains one aligned nonpacked native process, `calibration-short_available__c0__root-only`, R=2, patched FA2. No candidate, held-out or additional experiment is proposed.

**Disposition: substantial repairs are present, but three concrete evidence-validation defects remain.** The driver does not authenticate raw object bytes, missing packed attributes pass as false in the primary arm, and active attention KV is never checked for nonfinite values. Fix these within the existing smoke contract. The current FA2 version lookup additionally needs confirmation against the actual pinned interface/backend source before calling its boot attestation usable; the available repo patcher locates the selector in a different module, as described below.

## Exact snapshot and checks

All six payload hashes match `SNAPSHOT.json`, whose status is `CPU_AUTHOR_IN_PROGRESS_NOT_FINAL_FREEZE`.

| File | SHA-256 |
| --- | --- |
| `q1_reference_hooks_v2.py` | `0319ed7e82004f8030736be76e0578c85aa1c91423fb568fbde00c614c35c374` |
| `q1_reference_driver_v2.py` | `4ce707bbfdfbf4cc2d63124b9f5d6c93d9a9d5fddef37269bdb31cb8d552071e` |
| `q1_patch_reference_runner_v2.py` | `52fd1971f7f509809c4d06e1b6cf7b721b24a24f281ab9a1812ed5dd342742e3` |
| `q1_spec_off_engine_config_v2.py` | `dee75b5c4c79307c0752526ab5c5cf502f2c460d04cf2edd3d943c53cdf20daf` |
| `q1_reference_fa2_install.py` | `060963b8133faf3bfdb24dbce59153e6a818e21fadc207f573f3f11640c9f169` |
| `q1_reference_job.py` | `f62385bdfa7494bc43bf2faf48424ed81b21695b2fd9de091fcd28d1c249ae62` |

All six modules parse. The runner patcher was applied **in memory** to the pinned local stock runner, SHA `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0`: all five anchors occur exactly once, the result contains five marks and compiles. No installed or repository source was modified.

The actual smoke fixture/job builder returns one case, two requests, forced tokens `[11352, 25559]` at positions `[13487, 13488]`, with both state and KV byte archival enabled. No model or Docker operation was performed. Local standard-library probes reproduced findings 1 and 2 below; the torch-dependent synthetic full-model suite was inspected but not rerun locally because local torch is unavailable.

## Remaining concrete defects

### 1. Raw objects are not authenticated by the driver

`q1_reference_driver_v2.py:37-68` authenticates the JSON seal, collects referenced hashes, and checks that each corresponding filename exists. It only checks the O2 object's byte length, and never hashes the contents of any GDN, KV or O2 object. Consequently same-size byte corruption under a content-addressed filename still returns `(True, [])`.

The standard-library reproduction used an unchanged correctly sealed document and an object named for `sha256(b'good')`. With `b'good'` as contents, authentication returned `(True, [])`; changing only those four bytes to `b'evil'` returned the same result. A mocked in-memory file system was used; no archived evidence was edited. This is a raw-evidence integrity check, not a requirement to defend against arbitrary rewritten manifests.

The content store's write/reuse verification (`q1_reference_hooks_v2.py:67-78`) helps when writing but does not replace read-time authentication by the driver. `reconstruct_logical_kv` at lines 454-462 likewise reads bytes without checking their hashes or lengths.

**Correction:** authenticate SHA-256 and exact expected length for every referenced object, deduplicating reads by SHA. Derive GDN byte lengths from the recorded row shape/dtype; derive K/V block and tail lengths from declared logical geometry/extent, or retain explicit per-object byte lengths. Require O2 content SHA as well as length. Reconstruct the logical state only after these checks, and report corruption as an invalid seal/failure receipt. Add positive-control, same-size corruption and truncation tests for GDN, KV and O2. This repairs the already required O0 restore and seal-authentication contract.

### 2. An absent packed attribute still passes the primary-arm gate

`q1_reference_hooks_v2.py:199` and `:304` evaluate `bool(getattr(layer, 'enable_packed_recurrent_decode', None))`. In the aligned primary arm the expected flag is zero, so a missing attribute becomes `False` and passes. The prior review specifically required the expected attribute on every GDN layer.

A standard-library boot probe supplied 48 GDN objects with **no** packed attribute, 16 attention objects and otherwise valid stubbed FA/cache facts. `_attest_boot` returned:

```
fatal=False
packed_attribute_all_layers_match=True
problems=[]
```

**Correction:** require explicit attribute existence and its supported boolean/value type before comparing it with the arm. Apply the same rule at boot and snapshot capture. A missing attribute must fail in the primary arm as well as in the packed control. This does not change the selected native operator or add an experimental arm.

### 3. Nonfinite active attention KV can be sealed as valid

`q1_reference_hooks_v2.py:246-285` archives materialized K/V blocks and the valid tail without checking their values. GDN state is explicitly checked at lines 313-314; O2 is explicitly checked at lines 379-392; no equivalent check exists for attention KV. The mandatory seal validator at lines 424-438 only checks state-observation presence, layer counts, extents, consumed trace and O2 finiteness. A NaN/Inf in active attention KV therefore creates a content object and logical digest without adding an invalid reason.

**Correction:** check finiteness on the exact materialized K/V slices being serialized, including every full block and only the valid tail. Do not inspect unmaterialized suffix slots as though they were logical state. Fail with the layer/group/block/tail identity when invalid. Enforce the job's declared KV dtype and KV-head/head-dimension contract at the same extraction seam, rather than merely recording arbitrary values (`declared_geometry` is supplied by the job but is not used here). Add a materialized-NaN rejection and an inactive-tail-NaN positive control. Both belong to the original complete finite-state capture requirement.

## Status of the six original findings

| Original finding | Repair status in this snapshot |
| --- | --- |
| 1. Patched FA2 reference arms | Fork mount, hash/size verification, installed-byte recheck and minimal interface patch are implemented. The interface's non-tree branch is preserved. Version/dispatch attestation still needs the exact API correction/confirmation described below. |
| 2. Per-layer/group logical KV | The all-groups/axis-zero bug is repaired: extraction uses each layer's group, that group's request block row, K/V axis zero and physical block axis one. It checks required coverage, duplicates, null/out-of-range blocks, logical shape and full/tail extents. No silent block truncation remains. Finite/dtype validation remains finding 3. |
| 3. Repeat-unique, reusable snapshots | Observation names include run/arm/process/repeat/case; GDN, KV blocks/tails and logits use content-addressed storage with exact-byte verification on reuse. Complete bytes are retained, and reconstruction is expressible from manifests. Read-time authentication remains finding 1; no GPU restore is claimed. |
| 4. Actual consumed input and coverage | Prepared `input_ids` and `positions` are read after preparation using request query offset; fixture token/position checks and mandatory final trace checks are present. Runtime prompt IDs are required. Snapshot coverage is exactly 48 GDN/16 attention layers and each GDN layer reads its own metadata state row. Missing packed attributes and KV finiteness remain findings 2-3. |
| 5. Failure seals and driver verdict | Missing O0/O1/O2, trace mismatch, wrong extents, incomplete layer counts and nonfinite O2 now invalidate seals. Bound observation identity, seal JSON digest and `valid=true` are required by the driver, which stops before admitting another request after failure. Raw authentication remains finding 1. |
| 6. Hooks paths and concrete job | Rendered campaign patcher/PYTHONPATH locations are corrected; the job has a concrete schema, run/arm/process/case scope, terminal token and host/container control paths. An explicitly missing job is fatal. The builder enforces the exact one-case R=2 primary-process smoke. The still-unfinalized launch/freeze is not treated as a new defect. |

### FA2 source compatibility check still to close

`q1_reference_hooks_v2.py:158-162` imports `vllm.vllm_flash_attn.flash_attn_interface` and asks that module for `get_flash_attn_version`; if the attribute is absent, it records `None`, and lines 182-183 invalidate every request. The available repository patcher uses `from vllm.v1.attention.backends.fa_utils import get_flash_attn_version` (`scripts/fr13_patch_fa2_tree_bias.py:9363-9369`), and its minimal interface patch at `:4644` does not add a version getter.

The exact pinned current `flash_attn_interface.py`, `fa_utils.py` and `flash_attn.py` bytes were not in the locally supplied source set at this review point. I asked the parent to provide the existing pinned copies; no remote read was performed. Therefore the getter absence is **not presented here as a independently reproduced runtime failure**. Close this small source/API check before smoke: use the current selector API and record/require the version actually held by each active FLASH_ATTN implementation, with its backend/interface hashes. A default/interface constant or a loaded fork symbol alone does not establish the active dispatch. Keep the reference's ordinary causal one-token geometry and the same patched binary; do not introduce a fake tree query.

The existing synthetic tests override `_fa2_facts` with a dictionary, so they cannot validate this real API path. They also leave the primary packed attribute present and use finite tiny KV tensors, explaining why findings 2-3 are not exercised by their ordinary positive cases.

## Scope boundary

The repairs should preserve the current native-only smoke: two root-only requests with full usable O0/O1 state and actual z-consuming O2 logits. No new tolerance, candidate observation, new prefix, additional repeat/process or workload is needed to close these findings. Final immutable source/test/freeze binding and executable launch review remain the parent's responsibility after the author completes the repairs.

# Target-KV observer bridge v3: bounded independent review

Reviewed 2026-09-29 UTC. **Hold this source for the known target-registry correction (R1); also close the small counter-evidence check (R2) before treating the new pair as validated counter readbacks.** This review covers the new hook/patcher bridge, not the independently reviewed byte-witness oracle. No implementation, gate, launcher, scientific criteria, or workload scope was changed. No GPU, Docker, SSH, cache, or model operation was performed.

## Exact inspected sources

Paths are relative to `v2/experiments/review-response-20260927/`.

| Source | SHA-256 |
|---|---|
| `tools/q1_candidate_hooks_v3.py` | `9c00f38242aece81bf367294ca458c9cc2236066e287ffb4e4396c5370183020` |
| `tools/q1_patch_candidate_v3.py` | `9cf6f87b5a6ead4c8cc99671a5a9b5c39476e9ddbab870decffdd3f36238819d` |
| `tools/tests/test_q1_target_kv_bridge_v3.py` | `7d1dd19b82c0594c535aad1c64d36af2408411bcc60560b97239b39eadf24cd5` |
| Witness dependency, inspected only for interface | `195dadc17ddb0204786fccf3aaa5cbd68217bda2c0d3cfc3ce68ee66807133e4` |
| Captured generated runner | `3d9df101ebb1b2d5e8bc0ca844451b47a935f4adf48f7c6ab4ef81438ce21b79` |
| Captured generated rejection sampler | `7e4691a6e48fdd9426117beb9b91ef1fdd1053578f792b4d670234efba7b0e52` |

## R1: the target registry currently includes MTP and refuses production target16

`on_target_kv_before`, hook lines 412–414, derives expected names from every registry key ending `.self_attn.attn`. The production registry includes `mtp.layers.0.self_attn.attn`, so this derives 17 names but requires equality to the 16 target names passed to the real KV16 call. The production source explicitly requires the 17-member target-plus-MTP group and selects its 16 targets (`scripts/fr10_phase4_patch_vllm_tree_gdn.py:1138–1142,2075–2098`). The captured runner assigns `self.compilation_config = vllm_config.compilation_config` at line 408 and obtains both target and MTP tensors from that same registry at lines 6996–7043.

Independent AST controls using the exact hook method and real production target names reproduce the distinction: the test's 16-member context passes the name check, while the realistic 17-member context raises `ProcessUnusable: target layer registry coverage` before any copy. Both retain the unusable marker. The bridge test fixture currently supplies only 16 entries and shorter `model.layers.*` names, so its positive case does not exercise this boundary.

Minimal repair: use an independently declared exact target48-GDN/target16-attention registry with the explicit MTP singleton check, and exercise a realistic registry containing MTP. Parent has confirmed the related inherited enumeration sites and is preparing versioned root repairs; this note does not claim those repairs are already present in the inspected v3 hash.

## R2: malformed counters are coerced before the newly archived pair

The new `complete_events_before` field at line 471 is useful, but the inherited reads at lines 327 and 459 use `int(...)`. With `events_before=7`, actual `on_sealed` accepts an observed counter `8.75` or string `"8"` as the integer 8 and proceeds to O1 capture. A missing after-counter, stale 7, or jump to 9 correctly refuses. These are structural-counter controls, not model numerical tests.

Minimal repair: validate observed counter values as nonnegative Python integers without bool/string/float coercion, before archiving/checking the pair. Preserve any legitimate first-event absent-counter→0 convention explicitly, distinguishing that convention from an observed raw counter. No tolerance or experiment scope change is needed. The independent controls stop at an injected O1 boundary, so they demonstrate acceptance of the counter stage, not a completed valid full-model record.

## Sound parts of the bridge and retained failures

- The before hook is immediately before the unchanged `_fr13_f32_kv16(...)` call. The after hook is the first statement in its existing measured branch immediately after the call, before the production `target_kv_complete` flag and MTP payload publication. The before hook's `batch_indices is None` restriction corresponds to the production pure measured route; the nonpure route creates an index tensor and is refused. No copy is replaced or reimplemented by this bridge.
- Independently applied the actual patcher to the captured generated sources in memory, parsed both patched ASTs, and removed inserted marker lines: both originals were recovered byte-for-byte. The new job witness hash is mandatory; existing v2 jobs are not silently admitted.
- Before/after hooks latch the process unusable and raise on exceptions. The surrounding production catch rethrows a runtime error; it does not continue the failed copy path. Existing case and object storage remain the evidence mechanism. After capture is assigned before audit, so an audit mismatch retains both before and after references.
- Six independent AST controls used actual hook methods, actual `CandCase`, `_mark_unusable`, `_problems`, `_seal_case`, and `_write`, with only backend capture/audit replaced by CPU stubs. Duplicate before, B2, wrong registry, missing before, duplicate after, and injected audit mismatch all retained a sealed `valid=false` case plus `PROCESS_UNUSABLE`; subsequent `_guard` refused. The audit-mismatch case retained its after capture in the sealed record. These controls do not validate tensor bytes or witness arithmetic.
- Missing/failed witness is required by `_problems`; an absent hook cannot produce a valid case through a vacuous success flag. Coverage remains the witness's bounded target-KV region, not whole-cache or MTP-state coverage. The added readback synchronizations are diagnostic overhead and must not be interpreted as timing measurements.

## Reproduction evidence and limits

Independent audit files beside this note:

- `target-kv-bridge-v3-initial-cpu-audit.json`: source hashes, exact-name registry reproduction, in-memory patch/AST/inverse checks.
- `target-kv-bridge-v3-retention-cpu-audit.json`: six actual-sealer controls; SHA `bf2217e3177b6d663d4e5aa705912d7cde64dc8cf06e1746aac3112d2a038992`.
- `target-kv-bridge-v3-counter-cpu-audit.json`: six counter-stage controls.

The full supplied bridge suite was attempted with local Python and the existing repo virtualenv; both stopped at `ModuleNotFoundError: numpy`, before executing tests. No dependencies were installed. Parent reports three passing CPU test methods and retained its log; that is not claimed here as an independently rerun tensor suite. The witness backend is reviewed separately. No broad re-review of unchanged tensor machinery or prior accepted scientific policies was performed.

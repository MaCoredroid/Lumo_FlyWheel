# Full-model v2: bounded repair closure

Reviewed 2026-09-27 local time. **The three prior concrete defects and the FA2 API/actual-implementation attestation defect are closed in the snapshot below.** No additional blocking issue was found within that bounded scope. These exact bytes are ready to be matched into the final smoke freeze; unchanged files do not need another semantic review. This is not GPU launch approval or a claim that the full model has run.

## Snapshot identity

The seven files were copied read-only from the worker and saved before review in `fullmodel-repair-preliminary-snapshot-20260928T015125Z/` beside this note. Remote bytes were reread and hash-checked during capture. `SNAPSHOT.json` marks the copy `PRELIMINARY_NOT_FROZEN`.

| Source | SHA-256 |
| --- | --- |
| `q1_reference_hooks_v2.py` | `78ebde8de667f53e012db41597f6e4ab5970fd8b7c126a0c88f55ac62e6187dd` |
| `q1_reference_driver_v2.py` | `47314063eaa46b59eaa4fa60978353caef570bd8b39ebe5f127c56ff94702159` |
| `q1_reference_fa2_install.py` | `e485a2fe8a01c3820dbfda5c65a431019ef0f4289e246ca586a9ff858ed5b6d3` |
| `q1_spec_off_engine_config_v2.py` | `58aadfdb56f10142599098ada8a76ff3c792c39675e22fd4f5d5fadab81cffc4` |
| `q1_reference_job.py` | `9a9144622375c7a895c54fa590bdb9f9e2c6e27c728076dfa5ace1580806d485` |
| `q1_patch_reference_runner_v2.py` | `52fd1971f7f509809c4d06e1b6cf7b721b24a24f281ab9a1812ed5dd342742e3` |
| `tests/test_q1_fullmodel_v2_package.py` | `213bf798b8e18d53fe2edc8456d145999c8c04c9d4f3661d07a003c74c71037c` |

The inspected FA2 source index has SHA `0b060734ed11a1e41071b09bcb1b8fcc8935b520e0b843b2b913e8a6056aec08`. Its stock interface/selector/backend/attention-layer hashes match the source chain in `q1-fullmodel-fa2-api-addendum.md`; its patched interface hash is `dadab8aff63b7f608274834929c954335247366411382f390af92501361044a1`, and its fork hash is `28570f835ea72c99d03aab9fb03c494388bbb9c264ee4dc96eec047f50d7f857`.

## Findings closed and independent controls

1. **Raw-object authentication:** hooks lines 81-144 derive exact GDN row, K/V block/tail and O2 lengths, then verify every referenced object's size and SHA, deduplicating reads. Driver lines 68-73 call this check and reconcile the number of authenticated objects with distinct references. Reconstruction lines 591-607 checks size/SHA before using each KV object.

   A local in-memory filesystem probe passed a valid seven-object manifest and rejected **14/14** same-size-corruption/truncation controls across conv, SSM, full-block K/V, tail K/V and logits. Valid KV reconstruction matched expected bytes; a corrupt tail was refused. No real archive was modified.

2. **Strict packed attribute:** `packed_flag` at hooks lines 64-72 requires the attribute and accepts only booleans or integer 0/1. Both boot and state capture use it (lines 314 and 441). Four valid representations passed; missing/None/string/2 were rejected. A 48-GDN/16-attention synthetic boot passed with correct attributes and failed when all 48 packed attributes were missing. The original `bool(None)==False` defect is closed.

3. **Materialized KV finite/dtype/geometry:** extraction lines 390-394 enforce the declared dtype, KV-head count and head dimension. Lines 405 and 415 check exact full blocks and only the valid tail before serialization; inactive slots are excluded.

   Focused tests executed the captured source with actual CPU torch tensors and an in-memory object store: valid two-full-block/five-token-tail capture passed; NaN in full K, full V or a valid tail was rejected with block identity. A slot outside O0 but materialized at O1 passed O0 and failed O1. NaN/Inf confined to the inactive tail/unused blocks was accepted. Wrong dtype, KV-head count, head dimension and missing geometry all failed. These tests used `CUDA_VISIBLE_DEVICES=''`, CPU tensors, `python3 -B`, no vLLM import and no output files.

4. **FA2 actual implementation:** `_fa2_facts` now uses the source-confirmed `fa_utils.get_flash_attn_version` API and hashes the interface, selector, backend and attention-layer modules (lines 241-270). Boot checks those against the job and checks each active layer's backend, implementation class/module and `vllm_flash_attn_version` (lines 286-331). A global selector result alone no longer establishes dispatch.

   The valid 48/16 boot control passed. One FA3 implementation, one missing implementation, one non-FLASH_ATTN backend or one wrong implementation class each failed. The installer now verifies the deterministic patched interface SHA and the three pinned support sources (lines 78-85). The job includes these source identities, and the renderer's required receipt specifies the per-layer checks. The actual pinned image API was already independently established by the prior source addendum; no model import or live dispatch was attempted here.

## Verification boundary

The expanded test source contains the relevant raw-corruption, packed-attribute, active/inactive-KV, dtype/geometry, selector/source and per-layer-implementation controls. The author's reported **58-test** suite was not independently rerun in full; the focused positive/negative controls above were independently executed against this exact captured source. They verify instrumentation branches and evidence checks, not model correctness or GPU behavior.

No worker implementation, gate, numerical threshold or experiment scope was changed. The same smoke remains one aligned nonpacked native process, one root-only calibration case and R=2. The parent can close this repair review by matching the final freeze to these hashes and binding the final job/config/test receipts; review only actual differences if the source changes.

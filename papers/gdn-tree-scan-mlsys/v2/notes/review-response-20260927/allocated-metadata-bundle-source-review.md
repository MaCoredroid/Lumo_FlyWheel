# Allocated metadata bundle and observed parallelism — bounded review

2026-09-28. **Bounded source closure PASS after the filename/sequence repair**, at bundle SHA `c876078c73f2c81cb631271ce90693d58cb4ea764e71eb7916b5318f4376ef64`. Draft source review only, not admission. No container, GPU, HTTP, live `/proc`, engine import, workload, or gate action was performed. Sources and supplied tests were copied to `p0/monitor/review-response-20260927/allocated-metadata-bundle-independent-v1/` before review. Its `SNAPSHOT.json` binds exact paths, byte counts and hashes; the repaired copy uses sibling suffix `-repair1`.

## Initial finding

**F1 — CLOSED; initial failure preserved at bundle SHA `831c525a17eb4ca6daf5d346d4a0c60ff70bd7f0f494d41adf45b36251ae4d46`: shared-state sequence was not joined to the actual target record.** The SGLang producer includes `record_sequence` in `shared_target_state_reference`, but the initial `allocated_metadata_bundle_v1.py:77–82` checked only attempt, boot, PID, role and relationship. It never verified that the recorded sequence named the present target allocation file. It also did not bind allocation filenames to body attempt/boot/PID and sequence.

An independent temporary-file reproduction changes the draft's shared `record_sequence` to 999 while the sole target remains `a1.b1.pid7.allocation0.json`. `collect()` accepts both records. Evidence: `shared-sequence-reproduction.json` in the preserved initial snapshot. Minimal repair: derive each exact canonical filename from its body attempt/boot/PID and parsed nonnegative integer allocation sequence; require the shared sequence to be an integer equal to the actual target file's sequence. Do not accept a same-process but missing/different allocation reference.

The final repair implements those checks at lines 58–63 and 89–90, with strict positive integer PID and strict integer shared sequence. Nine supplied bundle tests now pass; nine additional independent temporary-file controls also pass. They preserve the original 999-reference rejection, confirm valid sequence 0 and nonzero sequence 3 joins, reject a stale zero reference to actual target 3, reject boolean/string references, and reject foreign PID, leading-zero sequence and unrelated filenames. The untouched parallel observer/projection and producer hashes match the first review snapshot. Reproducer and results: `allocated-metadata-bundle-independent-v1-repair1/independent_sequence_controls.py` and `independent-sequence-results.json`. No remaining material defect found within this bounded bundle/parallel-source delta.

## Source compatibility and bounded positive checks

- `worker_metadata_v1_2.py:41–51` reads the existing TP/PP groups' `rank_in_group` and `world_size`, refuses booleans/non-integer or invalid bounds, and checks sizes against the instantiated runner configuration. `runtime_projection_v1_2.py:111–113` now requires the actual observed rank/size tuple for either engine, then enforces the reviewed TP1/PP1 topology. The previous vLLM declaration-only boundary is therefore closed in this helper's source; real hook installation remains pending.
- The inspected immutable-image `parallel_state.py` SHA `7a6a6ea7cc7cd8aa4353cbab84fad365db6f9ea1fbd1db593b7b023a7ff04a76` matches its extraction receipt (`9704e29793cfff2160dfa8313b409628c961e80790a3657fbd45a0007cb5c71d`). `get_tp_group` (`:1221`) and `get_pp_group` (`:1240`) each contain only an assertion that the singleton exists and a return. Their ASTs contain no communication/allocation call. Group attributes are set at `:348–349`. The supplied receipt records a never-started exact-CID extraction without GPU requested; this review did not repeat that operation.
- The local pinned native runner snapshot `identity/native_source/vllm__v1__worker__gpu_model_runner.py` SHA `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0` initializes `self.parallel_config` from `vllm_config.parallel_config` at line 409. The installer binds the loaded parallel module file hash and both accessor source origins before wrapping initialization. This is source compatibility, not proof the future loaded serving worker has passed it.
- The bundle brackets metadata collection with fresh caller-provided exact-CID inspection, including image, owned attempt/boot labels, running state, restart count, PID/start identity and PID mode. It invokes the accepted ownership reader twice and compares worker/init epochs, namespaces, mappings and PPIDs. The accepted metadata projection ties allocated target views to the worker epoch and source/layer/precision policy. Inventory and contents are rechecked; outputs are write-once. SGLang target/draft must be the same process/source/topology and the draft must not claim duplicate target recurrent/convolution allocation.
- All **seven** supplied bundle tests passed in the isolated snapshot. The original log's larger discovery count is not reported as bundle-specific coverage. Tests exercise synthetic inspect/ownership callbacks only. This review also independently verifies the pinned accessor AST/source receipt and preserves the concrete sequence-link false acceptance above.

## Pending integration, not newly discovered runtime failures

The caller and final source freeze remain outside this review: bind the independently derived target-layer map, precision/configuration policy, both producer versions and all helper source hashes; install the hooks in the actual workers; provide a refresh-through-owned-client inspector and the accepted ownership reader; archive/authenticate the resulting bundle; bind the actual pre-agent route/dispatch and selected target worker. Injection callbacks used by CPU tests are not production observations. The bundle's policy hash is a canonical projection; the actual frozen policy/configuration bytes and caller packet still require their own bindings. The newly drafted `worker_bootstrap_v1.py` and `worker_sitecustomize_v1.py` were explicitly not reviewed here. This review does not approve WP/WC, a workload attempt, or serving-route qualification.

| Reviewed source | SHA256 |
|---|---|
| `allocated_metadata_bundle_v1.py` (initial F1) | `831c525a17eb4ca6daf5d346d4a0c60ff70bd7f0f494d41adf45b36251ae4d46` |
| Final `allocated_metadata_bundle_v1.py` (F1 closed) | `c876078c73f2c81cb631271ce90693d58cb4ea764e71eb7916b5318f4376ef64` |
| `worker_metadata_v1_2.py` | `bf08b96a9f11adb38513f1d4376a6b32d438c464f9f073ae03a73a11eaa43875` |
| `runtime_projection_v1_2.py` | `e9af014f68bfd2197db549ae8ef59b189471a0c431f16a69cc2ab7669df131b3` |
| `runtime_worker_ownership_v1.py` | `23eeb7494d23ad228bb98c46491afa235de624e27036248bc3e89d3e27636c93` |
| `sglang_worker_metadata_v1_1.py` | `08175a1a7a70857925012992295233294b046168c922122f97a8f9be26ef5988` |
| `worker_metadata_v1_1.py` (SGLang dependency) | `e02fea266846223296b9d64bfa5e857524d98bb21c7d05a9c0c78c1c0907ec4b` |
| Initial `test_allocated_metadata_bundle_v1.py` | `fa947ff2cab8bf63a440180c7e7c5a52ac1d336317896a551144d5c4b48bb8ea` |
| Final `test_allocated_metadata_bundle_v1.py` | `2dfddc80b17accc92ea27767f158e74c52bd456ee70a1aa1a0f5e7fe0f40f638` |

Test command from the copied snapshot: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_allocated_metadata_bundle_v1`.

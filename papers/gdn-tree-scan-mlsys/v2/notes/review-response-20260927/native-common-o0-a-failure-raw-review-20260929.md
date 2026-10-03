# Common-O0 A: failed first-request evidence review

Disposition: **the failed run is preserved and authenticated; zero observations qualify**. This is the target-only common-O0 campaign, distinct from the earlier natural-prefill A/B corpus. No engine/model/container/recovery operation or implementation edit was performed by this reviewer.

Run: `q1-native-common-o0-calibration-20260929T123300Z-aligned_nonpacked-A`. The terminal status is `COMPLETED_driver_rc=2_cleanup_rc=0`: one of 168 expected requests was attempted, that request failed, and the remaining 167 were not attempted. “COMPLETED” describes wrapper finalization, not successful scientific completion. The driver records HTTP 500, zero authenticated-valid observations, and no response usage, O0 digest, or O2 decision.

The independent audit authenticated all 40 terminal metadata members; the rendered config and actual retained Docker image/container/command/mount/resource records; approved gate, corpus, job and source bindings; stock-runner patch reproduction; FA2 installation and boot attestation; and the retained readiness receipt. It verified 27 frozen source payloads plus the retained image GDN metadata source used for diagnosis. The native route was spec-off, aligned, unpacked, B1, with 48 GDN and 16 FA2 attention layers, FP32 SSM, and kernel block sizes `[1024,1024,1024,64]`.

The exact owned CID is `f3ddf5861ae3d6fabffb129e6e8e65c6604e88d346b8719ff3fa2e88826f8968`. Start, stop and removal outputs name the same CID; the retained final inspect says not running, the cleanup state is `stopped_and_removed`, and cleanup stderr is empty. Receipt times are engine start 12:36:55, healthy 12:42:06, driver start 12:42:06, driver end 12:42:36, and engine stopped 12:42:37 UTC. These imply 311 seconds from the recorded engine-start timestamp to health, and a 30-second driver interval. They are operational intervals, not inference performance measurements.

## Failure boundary

The failed case is `calibration-short_available__c0__root-only`, process A, repeat 0. Its canonical record seal recomputes correctly, but `valid=false`, `request_still_active_at_seal=true`, and O0/O1/O2 are null. The unchanged common raw auditor independently rejects the record with `invalid or active case`; no failed record was converted into a valid sample.

The recorded original error is `AttributeError: 'GDNAttentionMetadata' object has no attribute 'non_spec_query_start_loc_cpu'`. In the executed `q1_native_common_o0_v1.py`, line 118 reads that field. The retained image's `GDNAttentionMetadata` dataclass (lines 40–74) has the GPU query field but no CPU query field; the similarly named CPU value exists only as a builder-local variable. This source mismatch is consistent with the actual failure, rather than a numerical result.

Executed ordering establishes the mutation boundary: `q1_reference_hooks_common_o0_v1.py:77` obtains the natural bootstrap snapshot before calling the importer; `q1_native_common_o0_v1.py:164` calls `live_destinations` before constructing the import plan and before `H4.apply_all` at line 175. Thus the reported missing-field exception occurs before common-O0 copying. The base hook seals the original exception, and the common hook subsequently raises `base native hook failed`, which terminates the engine path. The single root entry in `consumed_trace` was appended at the pre-forward hook before the failing O0 snapshot (`q1_reference_hooks_v2_1.py:505–510`); it must not be counted as a successfully completed root forward.

The object store retains 6,848 content objects totaling 1,037,828,096 bytes. Every filename SHA and length was independently checked, matching the terminal object count/bytes. **The natural-bootstrap descriptor was local to the interrupted call and is absent from the sealed record.** These bytes are therefore preserved content with no complete retained phase/owner/dtype mapping; this review does not reconstruct that mapping or claim a finite-state, O0 or output qualification from them.

## Scope and reproduction

The CPU audit ran 12:48:41.975848–12:48:42.702117 UTC with OMP/OpenBLAS/MKL thread ceilings of 2. It reused the frozen corpus reducer's outer provenance statements through the pre-verdict boundary. The exact AST extraction and three omitted success preconditions are recorded in `AST-SCOPE.json`: the two-run requirement and the two successful-terminal expectations are inapplicable to this single failed run and are replaced by explicit failed-run checks. Per-observation success logic and the cross-process aggregate were not run or relaxed. The unchanged raw auditor was invoked to confirm refusal of the failed record.

Evidence is in `p0/monitor/review-response-20260927/native-common-o0-a-failure-independent-20260929/`, locally and on the DGX under the same repository-relative path. `audit_failed_run.py --repo /home/mark/lumotree-review-20260927 --out <new-review-output-directory>` reproduces the read-only check. No retry authorization is implied. The original natural-prefill A/B adverse result remains unchanged: within-process repetitions were stable, but 14/84 cross-process greedy decisions differed and all 84 O0 digests differed. This failed common-O0 run neither establishes nor refutes a same-state numerical continuation result. Full Q1, MTP, candidate and workload qualification remain unachieved.

Key hashes:

- Terminal receipt: `bf15cedae1671a5e8202c2849a9101e962c2e4232052b02b2911c278dd3fd95f`.
- Corpus: `1a31e846218df03300b560f2e0d470edce79960e2d971a46a871ff94a34ac7ae`.
- Executed mapper: `d79e35b6a082181e307ce8229a2927a90e17707f99cf1032c74c32cc958c8620`.
- Executed hooks: `9eaedce27d00ac6206bb4ee2f1a3e4efa172828457bb5fcb174ec3dc0ea1fb21`.
- Common raw auditor: `9a91bade8330e8ff34d0c26275bd32a4c40bceb797d35b1d932a0c17827f8160`.
- Retained GDN source: `3c60cde9b0ef4bb6e4c105491afa50d96cad368399d36bfbed81e47a6e1b4b68`.
- Independent script: `79df1366e9d3f5a8a193476aedc183a5a615464157ba22eb75242286d07dde97`.
- Independent result: `938d67ca32e55c2019cd79c63b131cea47d62b6616ef566c28497b9084f5afe0`.
- Audit payload manifest: `ffd08dd8ab805c86ea9740f9a158530fdfa4af31f5c13967b56151cb0673887b`.

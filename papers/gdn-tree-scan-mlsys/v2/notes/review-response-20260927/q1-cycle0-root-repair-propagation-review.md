# Cycle-0 root-repair propagation review

Verdict: the inspected v6 hooks/job/patcher, raw audit v5, gate v4, driver v2.1, wrapper v4 and diagnostic selector v3.6 close the concrete physical-page/export and RNG propagation gaps found in their preserved predecessors. No new blocking source defect was found in this bounded delta. This is source/CPU review only: no engine, request, GPU, remote operation, corpus selection, result acceptance or launch authority was performed. The accepted root run was untouched.

## Exact propagation map and disposition

1. **Hydration and snapshot.** Old `q1_candidate_hooks_v5.py:20,287,306–309` selected hydration v2 and the native physical-page extractor. `q1_candidate_hooks_v6.py:20–21,80–81,290,307–314` now selects the accepted H3 transaction plus KR mapper, pins both actual source files, validates the block-table unit against physical cache shape, and exports native-order 64-token objects over physical 1024-token pages. H3 retains all-source authentication, all-destination disjointness, tail/positive-neighbor guards, readback, identity checks and fail-stop behavior. This changes instrumentation import/export, not cache allocation or serving flags.

2. **RNG.** Old job v5 inherited `q1_candidate_job.py:14` request seed0; old hooks lacked the actual runtime check. `q1_candidate_job_v6.py:27–29` sets request seed null before recomputing its canonical binding; `q1_candidate_hooks_v6.py:328–330,642` checks actual engine seed0 and empty per-request generators before active forward and retains the bulk-generator receipt. `q1_candidate_driver_v2_1.py:55,82–83` refuses missing/non-null seed before a control write and checks the sealed RNG receipt; raw audit v5:164 checks the same receipt. This does not claim per-request RNG equality with native references. Historical native jobs/results remain unchanged. `q1_candidate_case_binding_v2.validate_layout` has no seed0 requirement and can remain unchanged.

3. **Independent candidate audit.** Old raw audit v4:168–170 incorrectly sent physical1024 candidate snapshots through native64 `N.snapshot`. Raw audit v5:172–174 uses accepted `CS.snapshot` for candidate O0-natural/O0/O1 only; the native reference still uses `N.observation`. Its source hash ledger includes CS at231. Existing publication-count, conv/O1 and replay/conv/O0/O1 bindings remain present.

4. **Witness-to-O1 join.** Old raw audit v4:101,115–121 treated each exported chunk index as a physical page index. Raw audit v5:102,113–125 now uses `export_chunk_size=64` for logical starts/valid byte lengths, `kernel_block_size=1024` for page lookup, and checks both physical page and within-page offset. This is the only target-witness joining code that required an arithmetic correction. Hash, cache identity and complete materialized-intersection coverage checks remain enforced.

5. **Connected selectors/admission.** Gate v4:15,40,47,53 selects/rebuilds job v6, requires collector6 and scopes request seed null. Wrapper v4:24–36,60,154,162 selects the new diagnostic generator/launcher, driver, hooks, patcher, job, reducer and gate, and pins H3/KR/CS plus existing executable dependencies. Patcher v6:11–14,89 consistently uses marker/import/hash v6. Read-only `q1_make_diag_launcher_v3_6.render()` exactly reproduces the saved v3.6 launcher by replacing all five patcher selectors in the frozen v3.2 base. Existing operational ownership/cleanup code is unchanged.

## Witness compatibility

`q1_target_kv_witness_v1.py:32–76,115–129,152–163` already derives physical slots and capture indices from the observed block size. The current `[1048576,2097152,1024,256,1]` interleaved stride is admissible under its injective/bounded identity check (83–112). Hooks v6:438–446 supply the actual cache geometry. No 64-token physical-kernel assumption exists in this planner/capture/oracle. Physical-page neighborhood capture covers more bytes at1024; coverage remains explicitly suffix pages plus allocated neighbors, not the entire cache.

Convolution and replay witnesses operate on their own history/ring/SSI layouts and do not interpret attention page size. Their accepted same-case/name/SSI/O0/O1 binding does not require changes solely for physical1024 metadata. No previously closed convolution/replay issue was reopened.

## Retained CPU evidence

Artifacts are in `p0/monitor/review-response-20260927/cycle0-propagation-independent-20260929/`:

- `controls.py` / `RESULTS.json`: 27 independent stdlib controls, 15 positive and12 expected refusals. Includes actual stride acceptance/overlap refusal; actual1024 planner extents1,63,64,65,1023,1024,1025,2049,13487; positive logical64/physical1024 joins; preserved v4 join refuses the five new-layout fixtures; wrong page/offset/units, missing intersecting tail and resealed wrong bytes refuse. Small synthetic per-plane rows isolate join arithmetic; these controls do not substitute for full geometry or production provenance checks.
- `selector-controls.py` / `SELECTOR-RESULTS.json`: read-only exact render; new wrapper selectors/dependencies and absence of old executed selectors; missing/zero/nonzero/Boolean request seeds refuse before any driver dependency/control-write call.
- Independently reran the parent's unchanged `test_q1_candidate_cycle0_gate_v4.py`: four methods pass,0.086s; `test_q1_witness_o1_binding_v5.py`: three methods pass,11.719s. The latter uses all16 layers, real4096-byte slot rows and the960–1025 crossing of physical page1024. Logs are retained. Native builder is explicitly mocked in gate tests.
- `SOURCE-BINDINGS.json` and `sources/` retain27 reviewed source/test files, including the superseded cycle0 sources. No Torch/native kernel or candidate collection was run.

## Remaining admission work

Parent must bind the actual rebuilt prepared job and all reviewed dependency hashes into a new one-use cycle0 gate only after the required root instrumentation result is accepted. CPU passing, a generated launcher or this note is not that gate. Existing84-case, two-repeat, one-process-per-authority scope, native same-case A/r0 reference selection and runtime readiness checks are unchanged. No extra experiment is proposed by this review.

## Reviewed successor hashes

- `q1_candidate_hooks_v6.py`: `9bae999518bdfb36803052ee8c2a9943d6aedd41a9af43f5bc4e7bcf59be77af`
- `q1_candidate_job_v6.py`: `7761759b7c23f9f8a76b9fd28c1931ee337784940f2886c67b72ff0e453a5b4e`
- `q1_patch_candidate_v6.py`: `c8dc13419b770d6c0bae50000e792efb3eb5d86bc0cf27b026bb22dcd7819f21`
- `q1_candidate_raw_audit_v5.py`: `8b4254be83b26474d4033320f37d604ef68c420fb4a9dcb0a35a0fd6f3816aac`
- `q1_candidate_cycle0_gate_v4.py`: `fc357b5d1f97a243b80b67b59acb933703cbeed0ab343c11915f468634c73a10`
- `q1_candidate_driver_v2_1.py`: `1944beb2ade29361c75e9cab0332cd0f26e965ad2ac15a900abbc98f02b67bdb`
- `run_q1_candidate_cycle0_v4.sh`: `aac4c72207253b854ffd430ee4ebd269ba9f8992a4c00d3316720b7b8e73caad`
- `q1_make_diag_launcher_v3_6.py`: `52735f8f044d1cb10ea916458d8a93b2a85d44f5592226b3cbe35e578abda031`
- `fr14_leg3_launch_nomiddleware.q1diag.v3_6.sh`: `3947217d842c58104d6aa39632bb7720ee4dfeac4804d21e711f0fbba77e4fbd`

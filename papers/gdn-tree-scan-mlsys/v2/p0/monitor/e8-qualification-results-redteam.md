# E8 qualification-v2: independent actual-data review

Date: 2026-09-22. Reviewer: paper_redteam_round1. **PASS for both real qualification arms; no unresolved material qualification blocker to the reviewed six-cell timing handoff.** This review used CPU-only reads/reductions; no inference, GPU or container operation.

Run: `experiments/out-20260922T213142Z-e8-single-logits-v2`. Frozen source manifest: `bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030`. Final `QUALIFICATION_PASS.json`: `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513`. The separately reviewed conditional timing manifest is `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`; its final live `qualification()` admission was rerun successfully against this complete copied run, without a stub.

## Verified evidence

| Obligation | Independent result |
|---|---|
| Receipt/source identity | All 41 ON and 40 OFF raw-file hashes match. Unchanged qualification verifier reconstructed both receipts exactly. Exact loaded eagle variants, four other loaded modules, frozen stage sources, model identity and actual engine/route settings passed. |
| Runtime head invariants | Each arm has 103 completed/qualified proposals, 515 primary and 515 legacy head evaluations, 515 paired whole-logit/argmax/ordered-top-two checks, 103 root and 412 loop checks, and 2,060 guarded dispatch operations. Both owner receipts are closed with no failures. Existing ON FIX1 selfcheck independently records 515 checked steps, zero mismatches. |
| Recorder coverage | Each ledger has 311 contiguous error-free events, 103 forward entries, 103 matching output records and a final close with no sink failure. Head owner equals recorder owner. Each has 91 pure physical intervals and 12 nonpure rows. |
| Complete API binding | Independently reconstructed all eight 32-ID streams per arm from ordered emitted rows. Each arm has 271 structural ledger IDs, 256 API IDs and 15 IDs clipped only from terminal output rows. All eight prefixes have pure-decode exposure. |
| Empty discarded rows | Four nonpure chunk-prefill rows, sequences 49/64/82/92, are marked discarded and contain zero emitted IDs. They remain in the event/forward ledger and add no output tokens. |
| Untimed seal | Re-executed the frozen E1 joiner separately for each raw ledger into a temporary file: exact JSON match and rc3. All eight requests map to untimed support; zero timed requests, zero usable timing intervals and no throughput estimate. |
| Shutdown | Both containers exited with code0, no OOM flag and closed recorder/head evidence. |
| Cross-boot streams | All eight complete 32-ID ON/OFF API streams match. This is descriptive evidence for these requests, not an additional retention gate or a full-model equivalence proof. |

The same-input head claims above are supported by the exact loaded runtime gate and its closed counters; full vocabulary tensors were compared in memory, not archived for an independent offline tensor comparison. The gate also checks hidden-input bytes, CPU/device RNG, watched tensor mutation/identity and the complete ordered nine-draft candidate tree on every qualified proposal. This does not assert equality of arbitrary hidden model state across the two boots.

Per-prefix accounting below is identical in the two arms:

| Prefix | Pure forwards | API IDs | Structural IDs | Terminal clip |
|---|---:|---:|---:|---:|
| p072 | 12 | 32 | 34 | 2 |
| p017 | 12 | 32 | 37 | 5 |
| p015 | 9 | 32 | 33 | 1 |
| p021 | 11 | 32 | 32 | 0 |
| p085 | 13 | 32 | 33 | 1 |
| p095 | 16 | 32 | 35 | 3 |
| p058 | 8 | 32 | 35 | 3 |
| p083 | 10 | 32 | 32 | 0 |

The `logs/fr13_apc_bridge_error.flag` missing-sidecar message is retained. Frozen patcher lines12455–12489 explicitly attempt this optional APC environment sidecar only when its setting is absent; the cache-off path has no sidecar and skips injection. Actual command and loaded markers confirm cache off. This is not an active APC qualification or a reason to erase the flag.

## Hash ledger and reproduction

| Artifact | ON SHA-256 | OFF SHA-256 |
|---|---|---|
| qualification_<arm>.json | 1fa9e1a21a9d319a1c2015fccd3d9b88264b4ce0cf015edc34c503845c09caf5 | 20d718c832bee5514cf9ce9bede8bc431a0374481222a662e6ee1ccb6e058a60 |
| logs/e1_events.jsonl | f051d56ce91513a195474ae96f37828a3c5225d71440011de8d04dc424c4232f | 30365f2d159b56408cd4f57e97b5bf4c71668d3e189159cce68982491c0a2543 |
| logs/e8_head_gate.json | 47e2d8d831dc4d3ddcfa56ad5c7827d96fa6de6ac8a4a3b3458dd26fa1a48420 | cdef7dc93d46b05289751fc84abf76b17d2bf448cc85999d58ea0533b5a3f319 |
| loaded_backend/eagle.py | 3cdf64ba92bd24bdd3d6e8255277186f2fd211f3de54c5cfd4394a558e115358 | 364b767f6003cac1cd021e82249438edef4e0f87ba49a6aa6887506b8c9dc9e5 |

Independent combined reduction: `/tmp/e8-qualification-both-independent-20260922.json`, SHA-256 `21f08b04a9664cdb0b4d26f4e849d6d816b24b22dd10c1d9a6340c8e563725ff`. It records exact per-arm artifact hashes, counts and per-prefix accounting. Reproduction of the exact admission gate, using the frozen copied source, is CPU-only:

```sh
python3 -B experiments/e8-single-logits-timing-v1/timing_verify.py --stage experiments/e8-single-logits-timing-v1 --qualification experiments/out-20260922T213142Z-e8-single-logits-v2
```

Independent raw checks used `logs/e1_events.jsonl` event `n`, `seq`, `event`, `num_reqs`, `physical_step_id`, `request_ids`, output-row `request_id`, `emitted_ids`, `n_emitted`, `discarded`, `num_draft_tokens`, plus each `cohort/req_f*/capture_request.json` direct-ID tokens and frozen prefix hashes. The ordered per-request concatenation must equal all API IDs up to its API length; any excess must occur solely in the terminal row and equal `join.json.truncated_tail_tokens`. The frozen E1 joiner was rerun with each arm's `api_tokens.json`, `join_manifest.json` and `--expect-reqs 1`.

The original pre-container infrastructure failure remains preserved and is not counted as a scientific arm or replaced measurement. No additional qualification boot is necessary. After the parent's separate handoff, the fixed six timing cells may run; their clean ON5/OFF10 census, API support, source seals and sampled ownership rules remain mandatory. This qualification supplies no performance estimate and makes no claim about B4, other backends, cache/graph/stochastic modes, composed optimizations or universal full-model equivalence.

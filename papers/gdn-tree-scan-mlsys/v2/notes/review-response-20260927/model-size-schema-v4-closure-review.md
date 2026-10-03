# Model-file metadata schema v4 — bounded closure review

**PASS for the metadata-schema/source-binding correction. No blocker found; no runtime admission or workload authorization.** All work was local source/CPU review; no SSH, model payload read, GPU, container, or evaluator operation occurred.

An exact structured comparison verifies precisely 22 `size` → `bytes` renames across the two retained 11-file model views, plus the disclosed provenance field. Every SHA-256 and integer byte-count value is identical to v3. Reversing those key renames and removing the new provenance reproduces v3 exactly. Image/model identities, dataset/tasks, template, official harness/dependency identities, and all remaining criteria are unchanged. The gate adapter itself is unchanged at `c3e8a7e18089a846c2dedd3b3bedc7453376d9b04e4370ee90da99c0e911b356`.

The correction matches actual consumers. `Collector.files` (`collectors_v3.py:102–111`) emits `{sha256, bytes}` per name; the runtime probe's actual projection (`runtime_probe_producer_v1_2.py:604–606`) emits the same shape. Independent controls called the real collector method on an in-memory host and executed only the actual probe projection AST. For both model views, the results equal v4 exactly, using values taken independently from v3's old metadata. The unchanged gate exact comparison rejects old `{sha256, size}` objects, a changed hash, and a changed count. `e3_preflight_v5.py:390,399` still compares complete observed dictionaries with the pinned expected files; no normalization hides mismatches.

The actual `sole_executor_v3_7` import selects v4, and its `e.load_known` succeeds against all 19 source bindings. Updated identity/closure manifests and all active dependency-pin member hashes/sizes match. `closure_v3_3.py` differs from the preserved predecessor only in lock name/hash; `assemble_four_attempt_draft_v1.py:22` differs only in the selected lock filename. The pin generator selects v4 at line 14. Neither generator nor caller `main` was run.

Reviewer artifacts in `p0/monitor/review-response-20260927/model-size-schema-v4-review/` contain 132 checks and a 50-file snapshot. The prechange backup remains `before-model-size-schema-20260929T032200Z/` with repo-relative paths. These are CPU controls and hash checks, not model-identity observations of a running engine or experiment results. Fresh observed file hashes are still required by the unchanged runtime collectors and gate.

## Exact active hashes

- `identity-adapter/known-locks-v4.json`: `1b21d8a720b6ac367621b094eb5af106f1689a2c67a35a7dfab9bbfc19275a3b`
- `identity-adapter/e3_preflight_v5.py`: `c3e8a7e18089a846c2dedd3b3bedc7453376d9b04e4370ee90da99c0e911b356`
- `identity-adapter/MANIFEST-successor-v3.json`: `ff455992fbe453e83c28e3c4ad769d3ed46ee8067872b1f806f4a253c14ec2ac`
- `runtime-collectors/collectors_v3.py`: `f145c7443d9fe2cdaae58541c3d000843e83d7c7c13094251284174c0fdfd13e`
- `runtime-collectors/runtime_probe_producer_v1_2.py`: `37b53f756c74b14d85f85a8b054878f74ecb86cd79ffbdd85ddaa266457b2dbe`
- `attempt-closure/closure_v3_3.py`: `f23d6bfc8a1557e966beeda7acdad5695b7818ba3f5786846d3ef99d3b1c1653`
- `attempt-closure/MANIFEST-successor-v6.json`: `de9cd205e70438c7edd96447d2ea349876710e16eb4498cf656ce183ebfed3ea`
- `attempt-runtime/dependencies_v3_3.py`: `a74a35540577677ad500ed4ab8301f3369622ebae83c9a5b8b52998dfb4f21b2`
- `attempt-runtime/assemble_four_attempt_draft_v1.py`: `088a8c3bb1900fea410b8ae1234f7936754a010e12895866eb22420d7b632f98`
- `attempt-runtime/pin_observation_successors_v1.py`: `a35ef8206ec7448e72b707b7c89dd681c4cbaa11886ae1fbf0c8d39837f89179`

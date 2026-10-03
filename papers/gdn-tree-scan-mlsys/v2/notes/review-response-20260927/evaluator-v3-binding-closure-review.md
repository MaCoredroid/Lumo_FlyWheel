# Evaluator dependency binding v3 — bounded closure

**PASS for the source-binding delta.** The historical evaluator dependency mismatch identified in `evaluator-successor-v2-source-review.md` is closed for the current caller dependency identity. No runtime gate is opened and no task/GPU/evaluator operation was performed.

`known-locks-v3.json` changes only the evaluator pip-freeze hash, appends the three reviewed provisioning bindings (pip freeze, wheel lock, source/import receipt), and adds evaluator-successor lineage. An exact structured comparison after removing those disclosed changes reproduces v2. Model/image identities, dataset/task image identities, template, official harness version and all 73 source identities are unchanged. The adapter remains exactly `c3e8a7e18089a846c2dedd3b3bedc7453376d9b04e4370ee90da99c0e911b356`; numerical, task, schedule, and admission logic are consequently unchanged.

The new expected dependency SHA is `3bc94b8315f33341923a28f5d7a0d2db164adf3fc9c85a094be486d34700a57e`, matching the reviewed v2 environment's saved `pip-freeze.txt`. All 19 known-lock source hashes and sizes match. The lineage's review hash matches the retained reviewer note.

The actual `sole_executor_v3_7` import succeeds without calling `main`, and selects v5 plus `closure_v3_3.KNOWN_NAME == known-locks-v3.json`. Calling its actual `e.load_known(C, k.ADAPTER/k.KNOWN_NAME, k.KNOWN_SHA)` succeeds. `runtime_v3_3.py:24` therefore consumes the new lock through the existing loader. `closure_v3_3.py:26–27,541` changes only the lock constants; its remaining AST equals the preserved predecessor. Collector/runtime bytes are unchanged. The new identity manifest v3 and closure manifest v6, plus the unchanged collector manifest, match all `dependencies_v3_3.py:8–18` pins and member hashes/sizes. The pin generator selects the new filenames at lines 14 and 22.

The unchanged actual evaluator-dependency comparison (`e3_preflight_v5.py:382`), executed alone with bounded synthetic dictionaries, accepts the reviewed new SHA and refuses the old historical SHA. This is a comparison control, not a full admission result.

Reviewer evidence: `p0/monitor/review-response-20260927/evaluator-v3-binding-review/{controls.py,RESULT.json,SNAPSHOT.json}` records 104 checks across 49 retained files. The parent backup is `before-evaluator-runtime-binding-20260929T025850Z` with flat filenames. Existing provisioning-review limits still apply: no fresh remote wheel/venv byte audit, no separate x86 evaluator-host qualification, and no workload or performance claim.

## Exact active hashes

- `identity-adapter/known-locks-v3.json`: `c0c9d1ed924fc7452357a95d42b279f037112f5e3f4b4137636b82001f1fd86a`
- `identity-adapter/e3_preflight_v5.py`: `c3e8a7e18089a846c2dedd3b3bedc7453376d9b04e4370ee90da99c0e911b356`
- `identity-adapter/MANIFEST-successor-v3.json`: `1c4ac99d71184c2b56234d0f8892263d5978a201bc894179b550879f9a7df698`
- `attempt-closure/closure_v3_3.py`: `9baf9cdb2ac5268819a57d56bb2532bc947784394fb6c9dcb171dd57befd331a`
- `attempt-closure/MANIFEST-successor-v6.json`: `d55af40df7bea319e0a9e5f83e9ea20b9cfd35c5bc6fd812e9ef035b0c84dfa4`
- `attempt-runtime/dependencies_v3_3.py`: `3664b43b9fba4ec88b74d6a16f2a150d4d8b61f0f11534401cc45b9f160a7024`
- `attempt-runtime/pin_observation_successors_v1.py`: `3e418e723504c67fc10b235898cb862f83a6792d7d14e777af7b993ef23b45e5`
